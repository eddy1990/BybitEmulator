from __future__ import annotations

from typing import List
import os
import matplotlib.pyplot as plt


def plot_equity(equity_curve: List[float], out_path: str = "emulator_out/equity.png") -> None:
    if not equity_curve:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure()
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
