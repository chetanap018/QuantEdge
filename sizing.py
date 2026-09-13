"""
sizing.py
---------
Volatility-aware position sizing helpers (Fix 4, part 2).

  * atr()                 -- Wilder ATR from OHLC bars
  * atr_position_size()   -- qty from ATR stop distance + risk budget
  * volatility_target_size() -- scale a base qty by sigma_target/sigma_realised
  * correlation_scale()   -- shrink qty when avg pairwise corr is high
"""

import numpy as np
import pandas as pd


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period,
                  adjust=False).mean()


def atr_position_size(capital: float, risk_pct: float, entry: float,
                      atr_value: float, atr_multiple: float = 2.0,
                      lot_size: int = 1) -> int:
    """qty = (capital*risk_pct) / (atr_multiple*ATR), rounded to lot."""
    stop_dist = max(float(atr_multiple) * float(atr_value), 1e-9)
    risk_money = float(capital) * float(risk_pct)
    qty = int(risk_money // stop_dist)
    if lot_size and lot_size > 1:
        qty = (qty // lot_size) * lot_size
    return max(qty, 0)


def volatility_target_size(base_qty: int, realised_vol: float,
                           target_vol: float,
                           cap_mult: float = 2.0) -> int:
    """Scale qty by target/realised vol, capped at cap_mult x base."""
    if realised_vol is None or realised_vol <= 0 or target_vol <= 0:
        return int(base_qty)
    scale = float(target_vol) / float(realised_vol)
    scale = min(max(scale, 0.0), float(cap_mult))
    return max(int(round(base_qty * scale)), 0)


def correlation_scale(avg_corr: float,
                      threshold: float = 0.5) -> float:
    """Shrink factor in (0, 1] when avg pairwise corr exceeds threshold."""
    avg_corr = float(avg_corr)
    if avg_corr <= threshold:
        return 1.0
    return max(threshold / max(avg_corr, 1e-9), 0.25)

