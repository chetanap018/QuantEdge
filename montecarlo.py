"""
montecarlo.py
-------------
Monte Carlo simulation over trade sequences.

Instead of reporting one point estimate (single Sharpe / single drawdown),
resample the realized trade sequence and report confidence intervals.

Methods
-------
* "bootstrap"     -- resample the realized trade Net P&Ls with replacement,
                     re-assemble equity paths, recompute Sharpe & drawdown per
                     path, return percentiles.
* "block"         -- block bootstrap that preserves short-range autocorrelation
                     by resampling contiguous blocks of trades.

Output
------
    {
      "n_sims", "method", "sharpe_p25/p50/p75/p95", "sharpe_ci95",
      "max_drawdown_p25/p50/p75/p95", "max_drawdown_ci95",
      "final_capital_p5/p50/p95", "mean_final_capital"
    }
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def _metrics_from_path(initial: float, final: np.ndarray) -> Dict[str, float]:
    """Sharpe (per-trade returns) and max drawdown for a single equity path."""
    equity = np.concatenate([[initial], final])
    if equity.max() <= 0:
        return {"sharpe": 0.0, "max_drawdown_pct": 0.0}
    dd = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
    max_dd = abs(float(np.nanmin(dd))) * 100.0

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    std = rets.std()
    sharpe = float(rets.mean() / std) if std > 0 else 0.0
    return {"sharpe": sharpe, "max_drawdown_pct": max_dd}


def monte_carlo_confidence(
    trades,
    initial_capital: float,
    n_sims: int = 1000,
    method: str = "bootstrap",
    block_len: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence intervals for Sharpe ratio and max drawdown from the
    realized trade sequence.

    Parameters
    ----------
    trades : Sequence[Trade]
        Realized trades (each must expose .net_pnl).
    initial_capital : float
        Starting capital.
    n_sims : int
        Number of simulated paths.
    method : str
        "bootstrap" or "block".
    block_len : int
        Block length used by the "block" method.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict  -- see module docstring.
    """
    pnls = np.array([float(t.net_pnl) for t in trades]) if trades else np.array([])
    if len(pnls) == 0:
        return {
            "n_sims": int(n_sims), "method": method, "n_trades": 0,
            "sharpe_ci95": None, "max_drawdown_ci95": None,
            "mean_final_capital": float(initial_capital),
            "note": "No trades to simulate.",
        }

    rng = np.random.default_rng(seed)
    n = len(pnls)
    sim_sharpe = np.empty(n_sims)
    sim_dd = np.empty(n_sims)
    sim_final = np.empty(n_sims)

    for s in range(n_sims):
        if method == "block":
            # resample contiguous blocks of trades
            n_blocks = int(np.ceil(n / block_len))
            starts = rng.integers(0, n, size=n_blocks)
            sample = np.concatenate([pnls[starts[b]: starts[b] + block_len] for b in range(n_blocks)])[:n]
        else:
            sample = rng.choice(pnls, size=n, replace=True)

        path = initial_capital + np.cumsum(sample)
        m = _metrics_from_path(initial_capital, path)
        sim_sharpe[s] = m["sharpe"]
        sim_dd[s] = m["max_drawdown_pct"]
        sim_final[s] = path[-1]

    def _pctile(a, p):
        return round(float(np.percentile(a, p)), 3) if len(a) else None

    def _ci(a):
        return {"lower": _pctile(a, 2.5), "upper": _pctile(a, 97.5)}

    return {
        "n_sims": int(n_sims),
        "method": method,
        "n_trades": int(n),
        "sharpe_p25": _pctile(sim_sharpe, 25),
        "sharpe_p50": _pctile(sim_sharpe, 50),
        "sharpe_p75": _pctile(sim_sharpe, 75),
        "sharpe_ci95": _ci(sim_sharpe),
        "max_drawdown_p25": _pctile(sim_dd, 25),
        "max_drawdown_p50": _pctile(sim_dd, 50),
        "max_drawdown_p75": _pctile(sim_dd, 75),
        "max_drawdown_ci95": _ci(sim_dd),
        "final_capital_p5": round(float(np.percentile(sim_final, 5)), 2),
        "final_capital_p50": round(float(np.percentile(sim_final, 50)), 2),
        "final_capital_p95": round(float(np.percentile(sim_final, 95)), 2),
        "mean_final_capital": round(float(sim_final.mean()), 2),
    }