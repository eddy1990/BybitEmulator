# emulator/stats.py

def compute_stats(trades, equity_curve, start_balance):
    if not trades:
        return {}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    max_win = max(t["pnl"] for t in trades)
    max_loss = min(t["pnl"] for t in trades)

    return {
        "trades": len(trades),
        "winrate": len(wins) / len(trades),
        "max_win": max_win,
        "max_loss": max_loss,
        "final_balance": equity_curve[-1],
        "net_pnl": equity_curve[-1] - start_balance,
    }
