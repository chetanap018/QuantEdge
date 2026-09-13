"""
composite.py
------------
Strategy composition: combine signals from multiple sub-strategies.

Two combiners:
  * WeightedVoteStrategy -- weighted vote: composite = sign(sum(w_i * s_i));
    entries only when |weighted sum| >= threshold (default: majority).
  * UnanimousStrategy -- emits a signal only when ALL sub-strategies agree
    (useful as a strict confirmation filter).

Both take already-instantiated Strategy objects (vectorized or
event-driven -- each sub-strategy's generate_signals() is called on the
same data and aligned on datetime), so mixing modes is free.
"""

import pandas as pd
import numpy as np

from strategies.base import Strategy


def _align_signals(strategies, data: pd.DataFrame) -> pd.DataFrame:
    """Run each sub-strategy and align their 'signal' series on datetime."""
    base = data.copy().sort_values("datetime").reset_index(drop=True)
    cols = {}
    for i, strat in enumerate(strategies):
        out = strat.generate_signals(data.copy())
        out = out.sort_values("datetime").reset_index(drop=True)
        # Align by position when lengths match; fallback to datetime merge.
        if len(out) == len(base) and \
                (out["datetime"].values == base["datetime"].values).all():
            cols[f"__s{i}"] = out["signal"].fillna(0).astype(int).values
        else:
            m = pd.merge(base[["datetime"]], out[["datetime", "signal"]],
                         on="datetime", how="left")
            cols[f"__s{i}"] = m["signal"].fillna(0).astype(int).values
    return base, cols


class WeightedVoteStrategy(Strategy):
    """Weighted-vote ensemble of sub-strategies.

    composite_score = sum(w_i * signal_i); signal = sign(score) when
    |score| >= threshold else 0.  With equal weights and threshold =
    n/2 this is majority vote.
    """

    def __init__(self, strategies, weights=None, threshold: float = None,
                 name: str = "Weighted Vote", **kwargs):
        if not strategies:
            raise ValueError("strategies must be non-empty")
        super().__init__(name=name, **kwargs)
        self.sub_strategies = list(strategies)
        n = len(self.sub_strategies)
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError("len(weights) must match len(strategies)")
        self.weights = list(weights)
        self.threshold = threshold if threshold is not None else sum(
            abs(w) for w in self.weights) / 2.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df, cols = _align_signals(self.sub_strategies, data)
        score = np.zeros(len(df))
        for i, w in enumerate(self.weights):
            score = score + w * cols[f"__s{i}"]
        df["vote_score"] = score
        df["signal"] = 0
        df.loc[score >= self.threshold, "signal"] = 1
        df.loc[score <= -self.threshold, "signal"] = -1
        return df


class UnanimousStrategy(Strategy):
    """Emits a signal only when ALL sub-strategies agree (same direction)."""

    def __init__(self, strategies, name: str = "Unanimous", **kwargs):
        if not strategies:
            raise ValueError("strategies must be non-empty")
        super().__init__(name=name, **kwargs)
        self.sub_strategies = list(strategies)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df, cols = _align_signals(self.sub_strategies, data)
        arr = np.column_stack([cols[f"__s{i}"]
                               for i in range(len(self.sub_strategies))])
        df["signal"] = 0
        df.loc[(arr == 1).all(axis=1), "signal"] = 1
        df.loc[(arr == -1).all(axis=1), "signal"] = -1
        df["n_agree_long"] = (arr == 1).sum(axis=1)
        df["n_agree_short"] = (arr == -1).sum(axis=1)
        return df
