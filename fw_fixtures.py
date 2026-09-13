"""Shared fixtures for strategy-framework tests."""
import numpy as np
import pandas as pd


def make_data(n=200, seed=7, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    px = 100 + np.cumsum(rng.normal(0.3, 1.5, n))
    return pd.DataFrame({
        "datetime": dates,
        "open": px + rng.normal(0, 0.3, n),
        "high": px + np.abs(rng.normal(0.5, 0.5, n)),
        "low": px - np.abs(rng.normal(0.5, 0.5, n)),
        "close": px,
        "volume": rng.integers(1000, 10000, n),
    })


def make_intraday(days=10, bars_per_day=75, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01 09:15")
    px = 100.0
    for d in range(days):
        for b in range(bars_per_day):
            px += rng.normal(0.02, 0.3)
            ts = base + pd.Timedelta(days=d, minutes=5 * b)
            rows.append((ts, px + rng.normal(0, 0.05),
                         px + abs(rng.normal(0.1, 0.1)),
                         px - abs(rng.normal(0.1, 0.1)),
                         px, int(rng.integers(100, 2000))))
    return pd.DataFrame(rows, columns=["datetime", "open", "high",
        "low", "close", "volume"])
