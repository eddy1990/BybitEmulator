# bybit_sim.py (PATCH)

import json
from dataclasses import dataclass
from typing import List, Optional

from emulator.stats import compute_stats
from emulator.plot import plot_equity

def load_ticks_from_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        raw = json.load(f)

    candles = []
    for r in raw:
        candles.append({
            "ts": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
        })
    return candles

@dataclass
class Position:
    side: str            # "Buy" | "Sell"
    entry: float
    qty: float
    tp: Optional[float]
    sl: Optional[float]

class BybitV5SimHTTP:
    def __init__(self, candles, balance: float = 10_000):
        self.candles = candles
        self.total = len(candles)

        self.i = 0
        self.window_start = 1  # rolling window start
        self.current = candles[0] if candles else None

        self.balance = balance
        self.equity_curve = [balance]

        self.position: Optional[Position] = None
        self.trades = []
        self.finished = False
        self._finalized = False

    def unrealized_pnl(self, price: Optional[float] = None) -> float:
        if not self.position or not self.current:
            return 0.0
        px = float(price if price is not None else self.current["close"])
        p = self.position
        if p.side == "Buy":
            return (px - p.entry) * p.qty
        return (p.entry - px) * p.qty

    def equity(self) -> float:
        return self.balance + self.unrealized_pnl()

    def _check_tp_sl(self):
        """
        Checks on the current candle OHLC whether TP/SL is hit.
        NOTE: If both TP and SL are inside the same candle, we choose a conservative rule:
              SL first (worst-case for backtests).
        """
        if not self.position or not self.current:
            return

        c = self.current
        p = self.position

        hit_tp = False
        hit_sl = False

        if p.side == "Buy":
            hit_tp = (p.tp is not None) and (c["high"] >= p.tp)
            hit_sl = (p.sl is not None) and (c["low"] <= p.sl)

            if hit_sl:
                self._close(p.sl, "SL")
            elif hit_tp:
                self._close(p.tp, "TP")

        else:  # Sell
            hit_tp = (p.tp is not None) and (c["low"] <= p.tp)
            hit_sl = (p.sl is not None) and (c["high"] >= p.sl)

            if hit_sl:
                self._close(p.sl, "SL")
            elif hit_tp:
                self._close(p.tp, "TP")

    def _close(self, price: float, reason: str):
        p = self.position
        if not p:
            return

        pnl = (price - p.entry) * p.qty if p.side == "Buy" else (p.entry - price) * p.qty
        self.balance += pnl

        self.trades.append({
            "side": p.side,
            "entry": p.entry,
            "exit": float(price),
            "reason": reason,
            "pnl": float(pnl),
        })

        self.position = None

    def finalize_backtest(self):
        if self._finalized:
            return

        print("\n========== EMULATOR BACKTEST RESULTS ==========")

        stats = compute_stats(
            self.trades,
            self.equity_curve,
            self.equity_curve[0],
        )

        for k, v in stats.items():
            print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")

        print("==============================================\n")
        plot_equity(self.equity_curve)

        self._finalized = True

    async def place_order(self, side, qty, price=None, takeProfit=None, stopLoss=None, **_):
        if self.position:
            return {"retCode": 1, "retMsg": "Position already open"}

        if not self.current:
            return {"retCode": 2, "retMsg": "No market data loaded / current candle is None"}

        entry = float(price) if price is not None else float(self.current["close"])

        self.position = Position(
            side="Buy" if str(side).lower() == "buy" else "Sell",
            entry=entry,
            qty=float(qty),
            tp=float(takeProfit) if takeProfit is not None else None,
            sl=float(stopLoss) if stopLoss is not None else None,
        )

        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": "SIMULATED"}}

    async def get_kline(self, limit=200, start=None, end=None):
        limit = int(limit)

        end_idx = self.window_start + limit
        if end_idx >= self.total:
            # Dataset finished: close open position at last close (optional, but practical)
            self.finished = True
            if self.position and self.current:
                self._close(float(self.current["close"]), "EOD")
            self.equity_curve.append(self.equity())
            self.finalize_backtest()
            return {"retCode": 30001, "retMsg": "Emulator dataset finished", "result": None}

        window = self.candles[self.window_start:end_idx]
        self.window_start += 1

        # letzte Candle = "forming"
        self.current = window[-1]

        # >>> WICHTIG: TP/SL auf jeder neuen Candle prüfen <<<
        self._check_tp_sl()

        # Equity pro Step tracken (Wallet/Plot/Stats)
        self.equity_curve.append(self.equity())
        # ✅ Bybit-like Output: newest -> oldest
        window_for_api = list(reversed(window))
        bybit_list = [
            [
                str(c["ts"]),
                str(c["open"]),
                str(c["high"]),
                str(c["low"]),
                str(c["close"]),
                str(c["volume"]),
                "0"
            ]
            for c in window
        ]

        return {"retCode": 0, "retMsg": "OK", "result": {"list": bybit_list}}
