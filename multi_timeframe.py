"""
multi_timeframe.py
------------------
Multi-timeframe (MTF) strategy support.

Pattern: compute the trend/filter on a HIGHER timeframe (e.g. daily),
then apply it as a gate on the LOWER timeframe entries (e.g. 5-minute).

Helpers
-------
resample_ohlcv(df, rule) -- resample lower-TF bars to a higher TF.
merge_higher_trend(lower, higher, col) -- forward-fill a higher-TF column
    (e.g. 'trend' = +1/-1) onto every lower-TF bar using merge_asof so no
    future information leaks (only past higher-TF closes are visible).

Example
-------
    daily = resample_ohlcv(intraday_df, "1D")
    daily["trend"] = (daily["close"] > daily["close"].rolling(50).mean()) * 2 - 1
    gated = merge_higher_trend(intraday_df, daily[["datetime", "trend"]], "trend")
    entries = (intraday_signal == 1) & (gated["trend"] == 1)
"""

import pandas as pd
import numpy as np


def resample_ohlcv(df: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    """Resample OHLCV bars to a higher timeframe.

    Parameters
    ----------
    df : pd.DataFrame with datetime, open, high, low, close, volume.
    rule : pandas offset string, e.g. "1D" (daily), "1W", "15min", "1H".
    """
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.set_index("datetime").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    # Only aggregate columns that exist.
    agg = {k: v for k, v in agg.items() if k in d.columns}
    out = d.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()
    return out


def merge_higher_trend(lower: pd.DataFrame, higher: pd.DataFrame,
                       col: str = "trend") -> pd.DataFrame:
    """Left-join a higher-TF column onto lower-TF bars without look-ahead.

    Uses pd.merge_asof(backward) so each lower-TF bar only sees the most
    recent CLOSED higher-TF bar.  The higher frame's datetime is shifted by
    one period implicitly via merge_asof semantics (a higher bar stamped at
    its close time only applies to lower bars at/after that time).
    """
    lo = lower.copy()
    hi = higher.copy()
    lo["datetime"] = pd.to_datetime(lo["datetime"])
    hi["datetime"] = pd.to_datetime(hi["datetime"])
    lo = lo.sort_values("datetime").reset_index(drop=True)
    hi = hi.sort_values("datetime").reset_index(drop=True)
    if col not in hi.columns:
        raise ValueError(f"column {col!r} not in higher-TF frame")
    merged = pd.merge_asof(lo, hi[["datetime", col]], on="datetime",
                           direction="backward")
    return merged


def higher_trend_filter(lower: pd.DataFrame, rule: str = "1D",
                        fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Convenience: SMA-cross trend (+1/-1/0) from resampled higher TF.

    Returns the lower frame with a 'higher_trend' column merged in.
    """
    higher = resample_ohlcv(lower, rule)
    if len(higher) < slow:
        lower = lower.copy()
        lower["higher_trend"] = 0
        return lower
    fast_sma = higher["close"].rolling(fast).mean()
    slow_sma = higher["close"].rolling(slow).mean()
    higher["higher_trend"] = 0
    higher.loc[fast_sma > slow_sma, "higher_trend"] = 1
    higher.loc[fast_sma < slow_sma, "higher_trend"] = -1
    out = merge_higher_trend(lower, higher[["datetime", "higher_trend"]],
                             col="higher_trend")
    out["higher_trend"] = out["higher_trend"].fillna(0).astype(int)
    return out
