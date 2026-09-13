"""Tests for Fix 2: data layer (adapters, adjust, quality, point-in-time)."""
import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_adapters as DA
import data_adjust as ADJ
import data_quality as DQ
import data_pit as PIT
from data_fetcher import DataFetcher

passed, failed = 0, 0
def check(name, fn):
    global passed, failed
    try:
        fn()
        print("  PASS: " + name)
        passed += 1
    except AssertionError as e:
        print("  FAIL: " + name + " -> " + str(e))
        failed += 1

def bars(n=60, start="2024-01-01", px=100.0):
    dates = pd.bdate_range(start, periods=n)
    c = px * np.exp(np.cumsum(np.full(n, 0.001)))
    df = pd.DataFrame({"datetime": dates, "close": c})
    df["open"] = df["close"].shift(1).fillna(px)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = 100000
    return df

def t_canonical():
    a = DA.SyntheticAdapter().fetch("X", "NSE", "ONE_DAY", "2024-01-01", "2024-02-01")
    assert list(a.columns) == ["datetime", "open", "high", "low", "close", "volume"], a.columns.tolist()
    assert len(a) > 0

def t_fallback_order():
    class Empty(DA.BaseAdapter):
        name = "empty1"
        def fetch(self, *a, **k):
            return pd.DataFrame(columns=DA.CANONICAL)
    class Full(DA.BaseAdapter):
        name = "full1"
        def fetch(self, *a, **k):
            return bars(10)
    DA.REGISTRY["empty1"] = Empty()
    DA.REGISTRY["full1"] = Full()
    out = DA.fetch_with_fallback("X", sources=["empty1", "full1"])
    assert len(out) == 10, len(out)
    assert out.attrs.get("data_source") == "full1"

def t_split():
    df = bars(20, px=1000.0)
    out = ADJ.apply_splits(df, [("2024-01-15", 10)])
    before = out[out["datetime"] < "2024-01-15"]["close"].iloc[0]
    assert before < 150, before

def t_bonus():
    df = bars(20, px=500.0)
    out = ADJ.apply_bonus(df, [("2024-01-15", 1, 1)])
    assert out[out["datetime"] < "2024-01-15"]["close"].iloc[0] < 300

def t_quality_gaps():
    df = bars(10)
    df = df.drop(df.index[5]).reset_index(drop=True)
    rep = DQ.validate(df, timeframe="ONE_DAY")
    assert not rep["ok"] and len(rep["gaps"]) > 0, rep

def t_quality_stale():
    df = bars(20)
    df.loc[10:16, "close"] = 100.0
    df.loc[10:16, "open"] = 100.0
    df.loc[10:16, "high"] = 100.5
    df.loc[10:16, "low"] = 99.5
    rep = DQ.validate(df, timeframe="ONE_DAY", stale_n=5)
    assert len(rep["stale_runs"]) > 0, rep

def t_quality_outlier():
    df = bars(20)
    df.loc[10, "close"] = df.loc[9, "close"] * 1.5
    df.loc[10, "high"] = df.loc[10, "close"]
    rep = DQ.validate(df, timeframe="ONE_DAY", price_jump_pct=0.2)
    assert 10 in rep["outliers"], rep["outliers"]

def t_pit_clip():
    df = bars(30)
    u = PIT.UniverseHistory(members={"RELIANCE": [("2024-01-10", None)]})
    out = PIT.clip_to_listing(df, "RELIANCE", u)
    assert str(out["datetime"].iloc[0].date()) >= "2024-01-10", str(out["datetime"].iloc[0])

def t_pit_violation():
    df = bars(5)
    try:
        PIT.assert_point_in_time(df, when="2024-01-03")
        raise SystemExit("should have raised")
    except ValueError:
        pass

def t_fetcher_synthetic():
    f = DataFetcher.__new__(DataFetcher)
    f.connected, f.smart_api = False, None
    df = f._fetch_from_source("synthetic", "RELIANCE", "NSE", "ONE_DAY", "2024-01-01", "2024-02-01", None)
    assert len(df) > 0

def t_fetcher_csv(tmp_ok=True):
    os.makedirs("data", exist_ok=True)
    b = bars(15)
    b.to_csv("data/UNITTEST_ONE_DAY.csv", index=False)
    f = DataFetcher.__new__(DataFetcher)
    f.connected, f.smart_api = False, None
    import config
    old = getattr(config, "DATA_CSV_DIR", "data")
    config.DATA_CSV_DIR = "data"
    df = f._fetch_from_source("csv", "UNITTEST", "NSE", "ONE_DAY", "2024-01-01", "2024-12-31", None)
    config.DATA_CSV_DIR = old
    assert len(df) == 15, len(df)

if __name__ == "__main__":
    for n, fn in [("adapters canonical", t_canonical), ("fallback order", t_fallback_order),
                  ("split adjust", t_split), ("bonus adjust", t_bonus),
                  ("quality gaps", t_quality_gaps), ("quality stale", t_quality_stale),
                  ("quality outlier", t_quality_outlier), ("PIT clip", t_pit_clip),
                  ("PIT violation", t_pit_violation), ("fetcher synthetic", t_fetcher_synthetic),
                  ("fetcher csv", t_fetcher_csv)]:
        check(n, fn)
    print("DOMAIN 2 DATA LAYER: %d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
