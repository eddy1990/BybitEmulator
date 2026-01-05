# bybit_sim.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _now_ms() -> int:
    return int(time.time() * 1000)

def _to_ms(ts: int) -> int:
    ts = int(ts)
    return ts * 1000 if ts < 10_000_000_000 else ts

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
        t = _now_ms()

        if not self.position:
            self.position = Position(self.symbol, side, qty, price, lev, tp, sl, t, t, 0.0)
            self._log("FILL(open)")
            return

        p = self.position
        if p.side == side:
            new_size = p.size + qty
            p.avg_price = (p.avg_price * p.size + price * qty) / new_size
            p.size = new_size
            if tp is not None:
                p.take_profit = tp
            if sl is not None:
                p.stop_loss = sl
            p.leverage = lev
            p.updated_time = t
            self._log("FILL(add)")
            return

        # opposite direction => reduce/flip
        if qty < p.size:
            realised = (price - p.avg_price) * qty if p.side == "Buy" else (p.avg_price - price) * qty
            self.wallet_balance += realised
            p.cum_realised_pnl += realised
            p.size -= qty
            p.leverage = lev
            p.updated_time = t
            self._log("FILL(reduce)")
            return

        realised = (price - p.avg_price) * p.size if p.side == "Buy" else (p.avg_price - price) * p.size
        self.wallet_balance += realised
        remaining = qty - p.size
        self.position = None
        self._log("FILL(close)")
        if remaining > 0:
            self.position = Position(self.symbol, side, remaining, price, lev, tp, sl, t, t, 0.0)
            self._log("FILL(flip)")

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
        trig = None
        reason = None

        # SL first if both
        if p.side == "Buy":
            if p.stop_loss is not None and lo <= p.stop_loss:
                trig, reason = float(p.stop_loss), "SL"
            elif p.take_profit is not None and hi >= p.take_profit:
                trig, reason = float(p.take_profit), "TP"
        else:
            if p.stop_loss is not None and hi >= p.stop_loss:
                trig, reason = float(p.stop_loss), "SL"
            elif p.take_profit is not None and lo <= p.take_profit:
                trig, reason = float(p.take_profit), "TP"

        if trig is None:
            return

        realised = (trig - p.avg_price) * p.size if p.side == "Buy" else (p.avg_price - trig) * p.size
        self.wallet_balance += realised
        p.cum_realised_pnl += realised
        self.position = None
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
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": oid, "orderLinkId": str(payload.get("orderLinkId","") or "")}, "retExtInfo": {}, "time": _now_ms()}

    def finalize(self) -> Dict[str, Any]:
        eq, upnl, used, avail = self._equity_snapshot()
        self._log("FINALIZE")
        return {"retCode": 0, "retMsg": "OK", "result": {"walletBalance": _s(self.wallet_balance), "equity": _s(eq), "unrealisedPnl": _s(upnl)}, "retExtInfo": {}, "time": _now_ms()}
