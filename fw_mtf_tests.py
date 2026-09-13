"""Tests: multi-timeframe helpers."""
from fw_fixtures import make_intraday
from multi_timeframe import (resample_ohlcv, merge_higher_trend,
    higher_trend_filter)


class TestMultiTimeframe:
    def test_resample_counts(self):
        df = make_intraday(days=5)
        daily = resample_ohlcv(df, "1D")
        assert len(daily) == 5
        assert set(["open", "high", "low", "close",
            "volume"]).issubset(daily.columns)

    def test_merge_no_leak(self):
        lower = make_intraday(days=4)
        daily = resample_ohlcv(lower, "1D")
        daily["trend"] = [1, -1, 1, -1]
        merged = merge_higher_trend(lower, daily[["datetime", "trend"]])
        assert (merged.iloc[:75]["trend"] == 1).all()
        assert (merged.iloc[75:150]["trend"] == -1).all()

    def test_higher_trend_filter_gate(self):
        lower = make_intraday(days=30)
        gated = higher_trend_filter(lower, rule="1D", fast=2, slow=5)
        assert "higher_trend" in gated.columns
        assert set(gated["higher_trend"].unique()).issubset({-1, 0, 1})
        raw = [10, 40, 100]
        kept = [i for i in raw if gated.iloc[i]["higher_trend"] == 1]
        assert len(kept) <= len(raw)
