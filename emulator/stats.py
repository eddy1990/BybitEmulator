from __future__ import annotations
from typing import List, Dict, Any


def compute_stats(trades: List[Dict[str, Any]], equity_curve: List[float], start_balance: float) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "winrate": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "final_balance": float(equity_curve[-1]) if equity_curve else float(start_balance),
            "net_pnl": float((equity_curve[-1] - start_balance)) if equity_curve else 0.0,
        }

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    max_win = max((t.get("pnl", 0) for t in trades), default=0.0)
    max_loss = min((t.get("pnl", 0) for t in trades), default=0.0)

    final_balance = float(equity_curve[-1]) if equity_curve else float(start_balance)
    return {
        "trades": len(trades),
        "winrate": len(wins) / len(trades),
        "max_win": float(max_win),
        "max_loss": float(max_loss),
        "final_balance": final_balance,
        "net_pnl": final_balance - float(start_balance),
    }
