"""
walkforward.py
--------------
Walk-forward / out-of-sample testing.

Provides automatic splitting of historical data into train / validation / test
windows and a walk-forward evaluator that runs the backtest on rolling
train-then-test windows so the reported results are out-of-sample only.

Contrast with the single static date range used today:
    --walkforward --windows 6 --train_pct 0.7
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Time-ordered splits
# ----------------------------------------------------------------------
def split_train_validation_test(
    data: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data (sorted by datetime) into contiguous train / validation / test
    windows.  The split respects time order -- later bars are never used to
    judge earlier bars.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    data = data.sort_values("datetime").reset_index(drop=True)
    n = len(data)
    assert train_frac + val_frac + test_frac > 0, "fractions must sum to > 0"
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = data.iloc[:n_train].reset_index(drop=True)
    val = data.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = data.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


def walk_forward_windows(
    data: pd.DataFrame,
    n_windows: int = 6,
    train_pct: float = 0.7,
    min_train: int = 50,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate (train, test) walk-forward windows.

    Each window uses an EXPANDING train set (everything up to the current cut)
    and a fixed-size test segment immediately after it, so every test report is
    strictly out-of-sample relative to its train window.
    """
    data = data.sort_values("datetime").reset_index(drop=True)
    n = len(data)
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if train_pct <= 0 or train_pct >= 1:
        raise ValueError("train_pct must be in (0, 1)")

    test_len = max(1, int(n * (1 - train_pct) / n_windows))
    for w in range(n_windows):
        train_end = min(n, int(n * train_pct) + w * test_len)
        test_start = train_end
        test_end = min(n, test_start + test_len)
        if test_end > test_start and train_end >= min_train:
            yield (
                data.iloc[:train_end].reset_index(drop=True),
                data.iloc[test_start:test_end].reset_index(drop=True),
            )


# ----------------------------------------------------------------------
# 2. Walk-forward evaluator
# ----------------------------------------------------------------------
def run_walk_forward(
    strategy_factory: Callable[[], object],
    data: pd.DataFrame,
    n_windows: int = 6,
    train_pct: float = 0.7,
    engine_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run a strategy through walk-forward windows and aggregate the
    OUT-OF-SAMPLE (test-window) metrics.

    Parameters
    ----------
    strategy_factory : Callable[[], object]
        Zero-arg callable producing a fresh Strategy instance per window so
        no state leaks between windows.
    data : pd.DataFrame
        Historical OHLCV data with a 'datetime' column.
    n_windows : int
        Number of train/test folds.
    train_pct : float
        Fraction of total data used as the (expanding) train set.
    engine_kwargs : Optional[Dict]
        Extra kwargs for the BacktestEngine (capital, segment, execution, risk...).

    Returns
    -------
    dict
        per_window: list of test-window metric dicts
        oos_aggregate: mean/median/std of key metrics across test windows
    """
    from backtest_engine import BacktestEngine

    engine_kwargs = engine_kwargs or {}
    key_metrics = ["total_trades", "net_profit", "total_return_pct", "sharpe_ratio",
                   "max_drawdown_pct", "win_rate"]
    per_window = []

    for w, (train, test) in enumerate(walk_forward_windows(data, n_windows, train_pct)):
        if test.empty:
            continue
        strategy = strategy_factory()
        engine = BacktestEngine(strategy=strategy, data=test, **engine_kwargs)
        res = engine.run()
        row = {"window": w,
               "train_start": str(train["datetime"].iloc[0]),
               "train_end": str(train["datetime"].iloc[-1]),
               "test_start": str(test["datetime"].iloc[0]),
               "test_end": str(test["datetime"].iloc[-1])}
        for k in key_metrics:
            row[k] = res.get(k)
        per_window.append(row)
        if verbose:
            print(f"  [wf w{w}] test {row['test_start']}..{row['test_end']} "
                  f"trades={row['total_trades']} net={row['net_profit']} "
                  f"sharpe={row['sharpe_ratio']} dd={row['max_drawdown_pct']}%")

    aggregate = {}
    if per_window:
        df = pd.DataFrame(per_window)
        aggregate = {
            "n_windows": len(per_window),
            "mean_net_profit": round(float(df["net_profit"].mean()), 2),
            "median_trades": int(df["total_trades"].median()),
            "mean_sharpe": round(float(df["sharpe_ratio"].mean()), 3),
            "std_sharpe": round(float(df["sharpe_ratio"].std()), 3),
            "mean_max_drawdown_pct": round(float(df["max_drawdown_pct"].mean()), 2),
            "median_win_rate": round(float(df["win_rate"].median()), 2),
        }

    return {"per_window": per_window, "oos_aggregate": aggregate}