"""Data-quality validation: gaps, stale prices, outlier bars."""
import numpy as np
import pandas as pd

def expected_step(timeframe):
    return {"ONE_MINUTE": pd.Timedelta(minutes=1), "FIVE_MINUTE": pd.Timedelta(minutes=5),
            "FIFTEEN_MINUTE": pd.Timedelta(minutes=15), "THIRTY_MINUTE": pd.Timedelta(minutes=30),
            "ONE_HOUR": pd.Timedelta(hours=1), "ONE_DAY": pd.Timedelta(days=1)}.get(timeframe)

def validate(df, timeframe="ONE_DAY", price_jump_pct=0.2, stale_n=5):
    issues = {"gaps": [], "stale_runs": [], "outliers": [], "ohlc_violations": 0, "ok": True}
    if df is None or df.empty:
        issues["ok"] = False
        issues["error"] = "empty frame"
        return issues
    d = df.sort_values("datetime").reset_index(drop=True)
    step = expected_step(timeframe)
    if step is not None and timeframe == "ONE_DAY":
        cal = pd.bdate_range(d["datetime"].min(), d["datetime"].max())
        have = set(d["datetime"].dt.normalize())
        issues["gaps"] = [str(x.date()) for x in cal if x not in have]
    elif step is not None:
        diffs = d["datetime"].diff().dropna()
        bad = diffs[diffs > step * 1.5]
        issues["gaps"] = [str(d["datetime"].iloc[i]) for i in bad.index.tolist()[:20]]
    run, start = 1, 0
    for i in range(1, len(d)):
        if d["close"].iloc[i] == d["close"].iloc[i - 1]:
            run += 1
        else:
            if run >= stale_n:
                issues["stale_runs"].append((str(d["datetime"].iloc[start]), run))
            run, start = 1, i
    if run >= stale_n:
        issues["stale_runs"].append((str(d["datetime"].iloc[start]), run))
    ret = d["close"].pct_change().abs()
    issues["outliers"] = [int(i) for i in ret[ret > price_jump_pct].index.tolist()]
    bad_ohlc = d[(d["high"] < d[["open", "close"]].max(axis=1) - 1e-9) |
                 (d["low"] > d[["open", "close"]].min(axis=1) + 1e-9) |
                 (d["high"] < d["low"]) | (d["close"] <= 0)]
    issues["ohlc_violations"] = int(len(bad_ohlc))
    issues["ok"] = not (issues["gaps"] or issues["stale_runs"] or issues["outliers"] or issues["ohlc_violations"])
    issues["rows"], issues["start"], issues["end"] = len(d), str(d["datetime"].iloc[0]), str(d["datetime"].iloc[-1])
    return issues
