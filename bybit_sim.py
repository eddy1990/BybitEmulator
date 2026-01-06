# bybit_sim.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

def _now_ms() -> int:
    return int(time.time() * 1000)

def _to_ms(ts: int) -> int:
    ts = int(ts)
    return ts * 1000 if ts < 10_000_000_000 else ts



def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms)/1000, tz=timezone.utc).isoformat()

def _f(x: Any, d: float = 0.0) -> float:
    try:
        if x is None:
            return d
        if isinstance(x, str) and x.strip() == "":
            return d
        return float(x)
    except Exception:
        return d

def _i(x: Any, d: int = 0) -> int:
    try:
        if x is None:
            return d
        return int(float(x))
    except Exception:
        return d

def _s(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer()):
            return str(int(x))
    except Exception:
        pass
    return str(x)

def load_ticks_from_json(path: str) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        if isinstance(raw.get("result"), dict) and isinstance(raw["result"].get("list"), list):
            raw = raw["result"]["list"]
        elif isinstance(raw.get("list"), list):
            raw = raw["list"]
        else:
            raw = []

    out: List[Dict[str, Any]] = []

    def push(ts, o, h, l, c, v=None, t=None):
        ts_ms = _to_ms(_i(ts, 0))
        if ts_ms <= 0:
            return

        o_f, h_f, l_f, c_f = _f(o), _f(h), _f(l), _f(c)

        vol = _f(v, 0.0)
        tov = _f(t, 0.0)

        # Force non-zero if both are 0/0 (that would kill many bots/cleaners)
        if vol == 0.0 and tov == 0.0:
            vol = 1.0
            tov = max(1.0, abs(c_f) * vol)
        elif tov == 0.0 and vol != 0.0:
            tov = max(1.0, abs(c_f) * max(vol, 1e-9))

        out.append(
            {
                "ts": int(ts_ms),
                "open": float(o_f),
                "high": float(h_f),
                "low": float(l_f),
                "close": float(c_f),
                "volume": float(vol),
                "turnover": float(tov),
            }
        )

    if isinstance(raw, list):
        for r in raw:
            if isinstance(r, (list, tuple)) and len(r) >= 5:
                push(
                    r[0], r[1], r[2], r[3], r[4],
                    r[5] if len(r) > 5 else None,
                    r[6] if len(r) > 6 else None,
                )
            elif isinstance(r, dict):
                push(
                    r.get("startTime", r.get("ts", r.get("time", r.get("timestamp")))),
                    r.get("openPrice", r.get("open")),
                    r.get("highPrice", r.get("high")),
                    r.get("lowPrice", r.get("low")),
                    r.get("closePrice", r.get("close")),
                    r.get("volume", r.get("vol", r.get("v"))),
                    r.get("turnover", r.get("quoteVolume", r.get("q"))),
                )

    out.sort(key=lambda x: x["ts"])
    return out


@dataclass
class Order:
    order_id: str
    order_link_id: str
    symbol: str
    side: str          # "Buy"/"Sell"
    order_type: str    # "Market"/"Limit"
    qty: float
    price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    status: str = "New"  # New/Filled/Cancelled
    created_time: int = 0
    updated_time: int = 0

@dataclass
class Position:
    symbol: str
    side: str          # "Buy"/"Sell"
    size: float
    avg_price: float
    leverage: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    created_time: int = 0
    updated_time: int = 0
    cum_realised_pnl: float = 0.0

    def upnl(self, mark: float) -> float:
        return (mark - self.avg_price) * self.size if self.side == "Buy" else (self.avg_price - mark) * self.size

    def position_value(self, mark: float) -> float:
        return abs(self.size) * mark

    def margin_used(self, mark: float) -> float:
        lev = max(self.leverage, 1.0)
        return self.position_value(mark) / lev



@dataclass
class Trade:
    trade_id: int
    side: str              # "long"/"short"
    entry_ts: int
    entry: float
    qty: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    exit_ts: Optional[int] = None
    exit: Optional[float] = None
    pnl: float = 0.0
    reason: str = ""       # "TP"/"SL"/"EXIT"/"FLIP"/"REDUCE"/...

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = d.get("side")  # compatibility with legacy outputs
        d["entry_dt"] = _iso(self.entry_ts)
        d["exit_dt"] = _iso(self.exit_ts) if self.exit_ts is not None else None
        return d

class BybitV5SimHTTP:
    def __init__(
        self,
        candles_15: List[Dict[str, Any]],
        candles_60: Optional[List[Dict[str, Any]]] = None,
        symbol: str = "BTCUSDT",
        base_interval: str = "15",
        initial_balance: float = 10_000.0,
        default_leverage: float = 10.0,
        verbose: bool = True,
    ):
        self.symbol = symbol.upper()
        self.base_interval = str(base_interval).strip()
        self.base_ms = int(self.base_interval) * 60_000

        self.c15 = candles_15
        self.c60 = candles_60 or []
        if not self.c15:
            raise ValueError("No 15m candles")

        self.initial_balance = float(initial_balance)
        self.wallet_balance = float(initial_balance)

        self.default_leverage = float(default_leverage)
        self.leverage_by_symbol: Dict[str, float] = {self.symbol: float(default_leverage)}

        # forming candle cursor (primed on first base kline call)
        self.cursor_forming: Optional[int] = None

        self.position: Optional[Position] = None
        self.orders: Dict[str, Order] = {}

        # --- Live tracking (trades + equity) ---
        self.trade_seq: int = 0
        self.open_trade: Optional[Trade] = None
        self.closed_trades: List[Dict[str, Any]] = []

        # each point: {ts,equity,wallet,upnl,used,avail}
        self.equity_series: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []  # legacy simple plot

        self.state_version: int = 0

        self.finished = False
        self.verbose = verbose

    def now(self) -> int:
        return _now_ms()

    def _mark_price(self) -> float:
        if self.cursor_forming is None:
            return float(self.c15[0]["close"])
        closed_i = self.cursor_forming - 1
        if closed_i < 0:
            closed_i = 0
        return float(self.c15[closed_i]["close"])

    def _equity_snapshot(self) -> Tuple[float, float, float, float]:
        mark = self._mark_price()
        upnl = self.position.upnl(mark) if self.position else 0.0
        used = self.position.margin_used(mark) if self.position else 0.0
        equity = self.wallet_balance + upnl
        avail = max(self.wallet_balance - used, 0.0)
        return equity, upnl, used, avail


    def _market_time_ms(self) -> int:
        """Deterministische 'Marktzeit' (Candle-Ende der letzten geschlossenen Base-Candle)."""
        if self.cursor_forming is None:
            # before priming: use first candle end
            return int(self.c15[0]["ts"]) + self.base_ms
        base_closed_i = max(self.cursor_forming - 1, 0)
        return int(self.c15[base_closed_i]["ts"]) + self.base_ms

    def _bump_version(self) -> None:
        self.state_version += 1

    def _record_equity(self, ts_ms: Optional[int] = None) -> None:
        ts_ms = self._market_time_ms() if ts_ms is None else int(ts_ms)
        eq, upnl, used, avail = self._equity_snapshot()
        pt = {"ts": ts_ms, "wallet": float(self.wallet_balance), "equity": float(eq), "upnl": float(upnl),
              "used": float(used), "avail": float(avail)}
        # avoid duplicates
        if self.equity_series and self.equity_series[-1]["ts"] == pt["ts"] and abs(self.equity_series[-1]["equity"] - pt["equity"]) < 1e-9:
            return
        self.equity_series.append(pt)
        self.equity_curve.append(float(eq))

    def _start_trade(self, side: str, qty: float, price: float, tp: Optional[float], sl: Optional[float], ts_ms: Optional[int] = None) -> None:
        ts_ms = self._market_time_ms() if ts_ms is None else int(ts_ms)
        self.trade_seq += 1
        tside = "long" if side == "Buy" else "short"
        self.open_trade = Trade(
            trade_id=self.trade_seq,
            side=tside,
            entry_ts=ts_ms,
            entry=float(price),
            qty=float(qty),
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
        )

    def _update_open_trade(self, qty: Optional[float] = None, avg_price: Optional[float] = None,
                           tp: Optional[float] = None, sl: Optional[float] = None) -> None:
        if not self.open_trade:
            return
        if qty is not None:
            self.open_trade.qty = float(qty)
        if avg_price is not None:
            self.open_trade.entry = float(avg_price)
        if tp is not None:
            self.open_trade.tp = float(tp)
        if sl is not None:
            self.open_trade.sl = float(sl)

    def _close_trade(self, exit_price: float, reason: str, ts_ms: Optional[int] = None, qty: Optional[float] = None) -> None:
        """Schließt die aktuelle Trade-Instanz (voll oder partial)."""
        if not self.open_trade:
            return
        ts_ms = self._market_time_ms() if ts_ms is None else int(ts_ms)
        close_qty = float(self.open_trade.qty if qty is None else qty)
        # PnL auf Basis entry (avg) und close_qty
        if self.open_trade.side == "long":
            pnl = (float(exit_price) - float(self.open_trade.entry)) * close_qty
        else:
            pnl = (float(self.open_trade.entry) - float(exit_price)) * close_qty

        closed = Trade(
            trade_id=self.open_trade.trade_id,
            side=self.open_trade.side,
            entry_ts=self.open_trade.entry_ts,
            entry=self.open_trade.entry,
            qty=close_qty,
            sl=self.open_trade.sl,
            tp=self.open_trade.tp,
            exit_ts=ts_ms,
            exit=float(exit_price),
            pnl=float(pnl),
            reason=str(reason),
        )
        self.closed_trades.append(closed.to_dict())

        # reduce remaining qty on open trade if partial
        rem = float(self.open_trade.qty) - close_qty
        if rem > 1e-12:
            self.open_trade.qty = rem
        else:
            self.open_trade = None

    def export_state(self, bars: int = 600, interval: str = "15") -> Dict[str, Any]:
        """Snapshot für Live-UI (candles + trades + equity)."""
        bars = max(50, min(int(bars or 600), 5000))
        interval = str(interval)

        candles: List[Dict[str, Any]] = []
        if interval == self.base_interval:
            if self.cursor_forming is not None:
                end_i = int(self.cursor_forming)
                start_i = max(0, end_i - bars + 1)
                for c in self.c15[start_i:end_i + 1]:
                    candles.append({
                        "ts": int(c["ts"]),
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c.get("volume", 0.0)),
                    })

        eq, upnl, used, avail = self._equity_snapshot()
        pos = None
        if self.position:
            pos = {
                "side": "long" if self.position.side == "Buy" else "short",
                "size": float(self.position.size),
                "avg_price": float(self.position.avg_price),
                "tp": self.position.take_profit,
                "sl": self.position.stop_loss,
                "leverage": float(self.position.leverage),
                "upnl": float(self.position.upnl(self._mark_price())),
            }

        orders = []
        for o in self.orders.values():
            if o.status == "New":
                orders.append({
                    "order_id": o.order_id,
                    "link_id": o.order_link_id,
                    "side": "long" if o.side == "Buy" else "short",
                    "type": o.order_type.lower(),
                    "qty": float(o.qty),
                    "price": float(o.price) if o.price is not None else None,
                    "tp": float(o.take_profit) if o.take_profit is not None else None,
                    "sl": float(o.stop_loss) if o.stop_loss is not None else None,
                })

        open_trade = self.open_trade.to_dict() if self.open_trade else None

        # trim series
        eq_series = self.equity_series[-bars:]
        trades = self.closed_trades[-5000:]

        return {
            "version": int(self.state_version),
            "symbol": self.symbol,
            "base_interval": self.base_interval,
            "cursor_forming": self.cursor_forming,
            "finished": bool(self.finished),
            "wallet_balance": float(self.wallet_balance),
            "equity": float(eq),
            "unrealised_pnl": float(upnl),
            "margin_used": float(used),
            "available_balance": float(avail),
            "candles": candles,
            "position": pos,
            "open_orders": orders,
            "open_trade": open_trade,
            "trades": trades,
            "equity_curve": eq_series,
        }


    def _log(self, reason: str) -> None:
        if not self.verbose:
            return
        mark = self._mark_price()
        eq, upnl, used, avail = self._equity_snapshot()
        if self.position:
            p = self.position
            print(
                f"[EMULATOR] {reason} | wallet={self.wallet_balance:.2f} equity={eq:.2f} upnl={upnl:.2f} usedM={used:.2f} avail={avail:.2f} "
                f"| POS {p.side} size={p.size:.6f} avg={p.avg_price:.2f} lev={p.leverage:.1f} SL={p.stop_loss} TP={p.take_profit} mark={mark:.2f}"
            )
        else:
            print(
                f"[EMULATOR] {reason} | wallet={self.wallet_balance:.2f} equity={eq:.2f} upnl={upnl:.2f} usedM={used:.2f} avail={avail:.2f} | POS none mark={mark:.2f}"
            )

    def reset(self, initial_balance: Optional[float] = None) -> None:
        self.wallet_balance = float(self.initial_balance if initial_balance is None else initial_balance)
        self.cursor_forming = None
        self.position = None
        self.orders.clear()

        # --- Live tracking ---
        self.trade_seq = 0
        self.open_trade = None
        self.closed_trades.clear()
        self.equity_series.clear()
        self.equity_curve.clear()
        self.state_version = 0

        self.finished = False
        self._log("RESET")

    def _to_bybit_kline_list(self, candles: List[Dict[str, Any]]) -> List[List[str]]:
        out: List[List[str]] = []
        for c in reversed(candles):
            vol = float(c.get("volume", 0.0))
            tov = float(c.get("turnover", 0.0))
            if vol == 0.0 and tov == 0.0:
                vol = 1.0
                tov = max(1.0, abs(float(c["close"])) * vol)
            elif tov == 0.0 and vol != 0.0:
                tov = max(1.0, abs(float(c["close"])) * max(vol, 1e-9))
            out.append([_s(int(c["ts"])), _s(c["open"]), _s(c["high"]), _s(c["low"]), _s(c["close"]), _s(vol), _s(tov)])
        return out

    def _gen_id(self) -> str:
        return str(uuid.uuid4())
    def _apply_fill(self, side: str, qty: float, price: float, tp: Optional[float], sl: Optional[float]) -> None:
            side = "Buy" if str(side).lower() == "buy" else "Sell"
            qty = float(qty)
            price = float(price)
            lev = float(self.leverage_by_symbol.get(self.symbol, self.default_leverage))
            t_real = _now_ms()
            t_mkt = self._market_time_ms()

            # --- OPEN ---
            if not self.position:
                self.position = Position(self.symbol, side, qty, price, lev, tp, sl, t_real, t_real, 0.0)
                self._start_trade(side=side, qty=qty, price=price, tp=tp, sl=sl, ts_ms=t_mkt)
                self._record_equity(ts_ms=t_mkt)
                self._bump_version()
                self._log("FILL(open)")
                return

            p = self.position

            # --- ADD (same direction) ---
            if p.side == side:
                new_size = p.size + qty
                p.avg_price = (p.avg_price * p.size + price * qty) / new_size
                p.size = new_size
                if tp is not None:
                    p.take_profit = tp
                if sl is not None:
                    p.stop_loss = sl
                p.leverage = lev
                p.updated_time = t_real

                # update open trade snapshot to reflect avg + size
                self._update_open_trade(qty=p.size, avg_price=p.avg_price,
                                       tp=p.take_profit if p.take_profit is not None else None,
                                       sl=p.stop_loss if p.stop_loss is not None else None)
                self._record_equity(ts_ms=t_mkt)
                self._bump_version()
                self._log("FILL(add)")
                return

            # --- REDUCE / FLIP (opposite direction) ---
            if qty < p.size:
                realised = (price - p.avg_price) * qty if p.side == "Buy" else (p.avg_price - price) * qty
                self.wallet_balance += realised
                p.cum_realised_pnl += realised
                p.size -= qty
                p.leverage = lev
                p.updated_time = t_real

                # partial close trade
                self._close_trade(exit_price=price, reason="REDUCE", ts_ms=t_mkt, qty=qty)
                self._update_open_trade(qty=p.size, avg_price=p.avg_price,
                                       tp=p.take_profit if p.take_profit is not None else None,
                                       sl=p.stop_loss if p.stop_loss is not None else None)

                self._record_equity(ts_ms=t_mkt)
                self._bump_version()
                self._log("FILL(reduce)")
                return

            # full close (and possibly flip)
            realised = (price - p.avg_price) * p.size if p.side == "Buy" else (p.avg_price - price) * p.size
            self.wallet_balance += realised
            remaining = qty - p.size

            # close current
            self._close_trade(exit_price=price, reason=("FLIP" if remaining > 0 else "EXIT"), ts_ms=t_mkt, qty=p.size)

            self.position = None
            self._log("FILL(close)")

            # flip into new position
            if remaining > 0:
                self.position = Position(self.symbol, side, remaining, price, lev, tp, sl, t_real, t_real, 0.0)
                self._start_trade(side=side, qty=remaining, price=price, tp=tp, sl=sl, ts_ms=t_mkt)
                self._log("FILL(flip)")

            self._record_equity(ts_ms=t_mkt)
            self._bump_version()

    def _process_limit_fills(self, candle: Dict[str, Any]) -> None:
        hi = float(candle["high"])
        lo = float(candle["low"])
        for o in list(self.orders.values()):
            if o.status != "New" or o.order_type != "Limit" or o.price is None:
                continue
            px = float(o.price)
            touched = (lo <= px) if o.side == "Buy" else (hi >= px)
            if touched:
                o.status = "Filled"
                o.updated_time = _now_ms()
                self._apply_fill(o.side, o.qty, px, o.take_profit, o.stop_loss)

    def _process_tp_sl(self, candle: Dict[str, Any]) -> None:
        if not self.position:
            return
        p = self.position
        hi = float(candle["high"])
        lo = float(candle["low"])
        trig: Optional[float] = None
        reason: Optional[str] = None

        # TP-Priorität (deterministisch, wie im Backtest-Plot)
        if p.side == "Buy":
            if p.take_profit is not None and hi >= p.take_profit:
                trig, reason = float(p.take_profit), "TP"
            elif p.stop_loss is not None and lo <= p.stop_loss:
                trig, reason = float(p.stop_loss), "SL"
        else:
            if p.take_profit is not None and lo <= p.take_profit:
                trig, reason = float(p.take_profit), "TP"
            elif p.stop_loss is not None and hi >= p.stop_loss:
                trig, reason = float(p.stop_loss), "SL"

        if trig is None or reason is None:
            return

        # use candle-end as event time
        ts_ms = int(candle["ts"]) + self.base_ms

        realised = (trig - p.avg_price) * p.size if p.side == "Buy" else (p.avg_price - trig) * p.size
        self.wallet_balance += realised
        p.cum_realised_pnl += realised

        self.position = None
        self._close_trade(exit_price=trig, reason=reason, ts_ms=ts_ms)

        self._record_equity(ts_ms=ts_ms)
        self._bump_version()
        self._log(f"POS(close {reason})")


    def get_kline(self, category: str="linear", symbol: str="BTCUSDT", interval: str="15",
                  start: Optional[int]=None, end: Optional[int]=None, limit: int=200) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        interval = str(interval)
        limit = max(1, min(int(limit or 200), 1000))

        if symbol != self.symbol:
            return {"retCode": 0, "retMsg": "OK", "result": {"symbol": symbol, "category": category or "linear", "list": []}, "retExtInfo": {}, "time": _now_ms()}

        if self.finished:
            return {"retCode": 30001, "retMsg": "Emulator finished", "result": {"symbol": symbol, "category": category or "linear", "list": []}, "retExtInfo": {}, "time": _now_ms()}

        if interval == self.base_interval:
            # PRIME => bot gets exactly `limit` immediately
            if self.cursor_forming is None:
                self.cursor_forming = min(limit - 1, len(self.c15) - 1)
            else:
                self.cursor_forming += 1

            if self.cursor_forming >= len(self.c15):
                self.finished = True
                return {"retCode": 30001, "retMsg": "Emulator finished", "result": {"symbol": symbol, "category": category or "linear", "list": []}, "retExtInfo": {}, "time": _now_ms()}

            closed_i = self.cursor_forming - 1
            if closed_i >= 0:
                closed_candle = self.c15[closed_i]
                self._process_limit_fills(closed_candle)
                self._process_tp_sl(closed_candle)
                # MTM snapshot for live plotting (also when nothing filled)
                self._record_equity(ts_ms=int(closed_candle["ts"]) + self.base_ms)
                self._bump_version()

            end_i = self.cursor_forming
            start_i = max(0, end_i - limit + 1)
            window = self.c15[start_i:end_i+1]
            return {"retCode": 0, "retMsg": "OK", "result": {"symbol": symbol, "category": category or "linear", "list": self._to_bybit_kline_list(window)}, "retExtInfo": {}, "time": _now_ms()}

        # 1h feed: candles closed relative to last CLOSED 15m candle end
        if self.cursor_forming is None:
            return {"retCode": 0, "retMsg": "OK", "result": {"symbol": symbol, "category": category or "linear", "list": []}, "retExtInfo": {}, "time": _now_ms()}

        base_closed_i = max(self.cursor_forming - 1, 0)
        base_end = int(self.c15[base_closed_i]["ts"]) + self.base_ms

        if interval == "60" and self.c60:
            dataset = self.c60
            itv_ms = 60 * 60_000
        else:
            dataset = self.c15
            itv_ms = int(interval) * 60_000 if interval.isdigit() else self.base_ms

        closed = [c for c in dataset if int(c["ts"]) + itv_ms <= base_end]
        window = closed[-limit:]
        return {"retCode": 0, "retMsg": "OK", "result": {"symbol": symbol, "category": category or "linear", "list": self._to_bybit_kline_list(window)}, "retExtInfo": {}, "time": _now_ms()}

    def get_wallet_balance(self, accountType: str="UNIFIED", **_: Any) -> Dict[str, Any]:
        eq, upnl, used, avail = self._equity_snapshot()
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": [{"accountType": accountType, "totalEquity": _s(eq), "totalWalletBalance": _s(self.wallet_balance), "totalAvailableBalance": _s(avail), "totalPerpUPL": _s(upnl)}]},
            "retExtInfo": {},
            "time": _now_ms(),
        }

    def get_position_list(self, category: str="linear", symbol: str="BTCUSDT", **_: Any) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        if symbol != self.symbol:
            return {"retCode": 0, "retMsg": "OK", "result": {"list": []}, "retExtInfo": {}, "time": _now_ms()}

        lst = []
        if self.position:
            mark = self._mark_price()
            p = self.position
            lst.append({"symbol": self.symbol, "side": p.side, "size": _s(p.size), "avgPrice": _s(p.avg_price),
                        "takeProfit": _s(p.take_profit or ""), "stopLoss": _s(p.stop_loss or ""),
                        "markPrice": _s(mark), "unrealisedPnl": _s(p.upnl(mark))})
        return {"retCode": 0, "retMsg": "OK", "result": {"list": lst}, "retExtInfo": {}, "time": _now_ms()}

    def set_trading_stop(self, **payload: Any) -> Dict[str, Any]:
        symbol = str(payload.get("symbol", self.symbol)).upper()
        if symbol != self.symbol:
            return {"retCode": 0, "retMsg": "OK", "result": {}, "retExtInfo": {}, "time": _now_ms()}

        tp = payload.get("takeProfit", None)
        sl = payload.get("stopLoss", None)
        tp_v = _f(tp, 0.0) if tp not in (None, "", "0") else None
        sl_v = _f(sl, 0.0) if sl not in (None, "", "0") else None

        if self.position:
            if tp_v is not None:
                self.position.take_profit = tp_v
            if sl_v is not None:
                self.position.stop_loss = sl_v
            self.position.updated_time = _now_ms()
            self._log("POS(update trading-stop)")
            self._update_open_trade(tp=self.position.take_profit if self.position.take_profit is not None else None,
                                   sl=self.position.stop_loss if self.position.stop_loss is not None else None)
            self._record_equity()
            self._bump_version()

        return {"retCode": 0, "retMsg": "OK", "result": {}, "retExtInfo": {}, "time": _now_ms()}

    def set_leverage(self, **payload: Any) -> Dict[str, Any]:
        symbol = str(payload.get("symbol", self.symbol)).upper()
        buy = _f(payload.get("buyLeverage", payload.get("leverage", self.default_leverage)), self.default_leverage)
        lev = max(1.0, min(float(buy), 100.0))
        self.leverage_by_symbol[symbol] = lev
        self._log(f"LEV(set {lev})")
        return {"retCode": 0, "retMsg": "OK", "result": {}, "retExtInfo": {}, "time": _now_ms()}

    def create_order(self, **payload: Any) -> Dict[str, Any]:
        side = "Buy" if str(payload.get("side", "Buy")).lower() == "buy" else "Sell"
        order_type = "Market" if str(payload.get("orderType", "Market")).lower() == "market" else "Limit"
        qty = _f(payload.get("qty", 0.0), 0.0)
        if qty <= 0:
            return {"retCode": 10002, "retMsg": "Invalid qty", "result": {}, "retExtInfo": {}, "time": _now_ms()}

        price = payload.get("price", None)
        price_v = _f(price, 0.0) if price not in (None, "", "0") else None

        tp = payload.get("takeProfit", None)
        sl = payload.get("stopLoss", None)
        tp_v = _f(tp, 0.0) if tp not in (None, "", "0") else None
        sl_v = _f(sl, 0.0) if sl not in (None, "", "0") else None

        link = str(payload.get("orderLinkId", "") or "")
        oid = self._gen_id()
        t = _now_ms()

        o = Order(oid, link, self.symbol, side, order_type, qty, None if order_type == "Market" else price_v, tp_v, sl_v, "New", t, t)
        self.orders[oid] = o
        self._bump_version()

        if order_type == "Market":
            fill_price = self._mark_price()
            o.status = "Filled"
            o.updated_time = _now_ms()
            self._apply_fill(o.side, o.qty, fill_price, o.take_profit, o.stop_loss)

        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": oid, "orderLinkId": link}, "retExtInfo": {}, "time": _now_ms()}

    def order_realtime(self, **params: Any) -> Dict[str, Any]:
        lst = []
        for o in self.orders.values():
            if o.status != "New":
                continue
            lst.append({"orderId": o.order_id, "orderLinkId": o.order_link_id, "symbol": self.symbol,
                        "side": o.side.lower(),  # IMPORTANT for your bot pending logic
                        "orderType": o.order_type, "price": _s(o.price or ""), "qty": _s(o.qty),
                        "takeProfit": _s(o.take_profit or ""), "stopLoss": _s(o.stop_loss or "")})
        return {"retCode": 0, "retMsg": "OK", "result": {"list": lst, "nextPageCursor": ""}, "retExtInfo": {}, "time": _now_ms()}

    def cancel_order(self, **payload: Any) -> Dict[str, Any]:
        oid = str(payload.get("orderId", "") or "")
        link = str(payload.get("orderLinkId", "") or "")
        if oid in self.orders and self.orders[oid].status == "New":
            self.orders[oid].status = "Cancelled"
            self.orders[oid].updated_time = _now_ms()
            self._log("ORDER(cancel)")
            self._bump_version()
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": oid, "orderLinkId": link}, "retExtInfo": {}, "time": _now_ms()}

    def amend_order(self, **payload: Any) -> Dict[str, Any]:
        oid = str(payload.get("orderId", "") or "")
        if oid in self.orders and self.orders[oid].status == "New":
            o = self.orders[oid]
            if payload.get("qty") is not None:
                o.qty = _f(payload["qty"], o.qty)
            if payload.get("price") is not None:
                o.price = _f(payload["price"], o.price or 0.0)
            o.updated_time = _now_ms()
            self._log("ORDER(amend)")
            self._bump_version()
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": oid, "orderLinkId": str(payload.get("orderLinkId","") or "")}, "retExtInfo": {}, "time": _now_ms()}

    def finalize(self) -> Dict[str, Any]:
        eq, upnl, used, avail = self._equity_snapshot()
        self._log("FINALIZE")
        return {"retCode": 0, "retMsg": "OK", "result": {"walletBalance": _s(self.wallet_balance), "equity": _s(eq), "unrealisedPnl": _s(upnl)}, "retExtInfo": {}, "time": _now_ms()}
