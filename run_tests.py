"""
run_tests.py
------------
Simple test suite to verify the backtesting framework works.

Run: python3 run_tests.py
"""

import sys
import traceback

import pandas as pd
import numpy as np

PASSED = []
FAILED = []


def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS: {name}")
        except Exception as e:
            FAILED.append((name, str(e)))
            print(f"  FAIL: {name}")
            print(f"        {e}")
        return fn
    return decorator


# ============================================================
# TEST 1: Core module imports
# ============================================================
@test("Core modules import (config, broker_charges, data_fetcher, backtest_engine, main, backtest_config)")
def test_core_imports():
    import config
    import broker_charges
    import data_fetcher
    import backtest_engine
    import main
    import backtest_config
    assert hasattr(config, "CHARGES_CONFIG")
    assert hasattr(backtest_config, "SYMBOL")


# ============================================================
# TEST 2: Strategy imports & registry
# ============================================================
@test("All strategies import and are registered in main.py")
def test_strategy_registry():
    from strategies import (
        SMACrossoverStrategy,
        RSIMeanReversionStrategy,
        MACDCrossoverStrategy,
        OpeningCandleStrategy,
        SuzlonStrategy,
        HeroOrbStrategy,
    )
    import main
    expected = {"sma_crossover", "rsi_mean_reversion", "macd_crossover", "opening_candle", "suzlon", "hero_orb"}
    actual = set(main.STRATEGY_REGISTRY.keys())
    assert expected == actual, f"Registry mismatch: {actual}"


# ============================================================
# TEST 3: Broker charges — all segments
# ============================================================
@test("Broker charges: intraday long P&L sign correct (+2000)")
def test_charges_long():
    from broker_charges import calculate_charges
    r = calculate_charges("buy", 500, 520, 100, "intraday_equity")
    assert r["gross_pnl"] == 2000, f"Expected 2000, got {r['gross_pnl']}"
    assert r["net_pnl"] < r["gross_pnl"], "Charges not deducted"
    assert r["total_charges"] > 0


@test("Broker charges: intraday short P&L sign correct (+2000)")
def test_charges_short():
    from broker_charges import calculate_charges
    r = calculate_charges("sell", 500, 520, 100, "intraday_equity")
    assert r["gross_pnl"] == 2000, f"Short P&L inverted! Got {r['gross_pnl']}"


@test("Broker charges: delivery_equity has zero brokerage")
def test_charges_delivery():
    from broker_charges import calculate_charges
    r = calculate_charges("buy", 500, 520, 100, "delivery_equity")
    assert r["brokerage"] == 0, f"Delivery should be free, got {r['brokerage']}"
    assert r["stt"] > 0, "Delivery STT should apply"


@test("Broker charges: options flat Rs 20/order (Rs 40 round-trip)")
def test_charges_options():
    from broker_charges import calculate_charges
    r = calculate_charges("buy", 100, 110, 50, "options")
    assert r["brokerage"] == 40, f"Expected 40, got {r['brokerage']}"


@test("Broker charges: futures P&L correct")
def test_charges_futures():
    from broker_charges import calculate_charges
    r = calculate_charges("buy", 50000, 50100, 1, "futures")
    assert r["gross_pnl"] == 100, f"Expected 100, got {r['gross_pnl']}"


@test("Broker charges: invalid segment raises ValueError")
def test_charges_invalid_segment():
    from broker_charges import calculate_charges
    try:
        calculate_charges("buy", 100, 110, 1, "invalid_segment")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ============================================================
# TEST 4: Backtest engine with synthetic data (all strategies)
# ============================================================
def make_daily_data(n=120, seed=42):
    """Synthetic daily OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.random.randn(n).cumsum()
    open_ = close + np.random.randn(n) * 0.5
    high = np.maximum(open_, close) + np.abs(np.random.randn(n))
    low = np.minimum(open_, close) - np.abs(np.random.randn(n))
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n, freq="D"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.randint(1000, 5000, n),
    })


def make_intraday_data(days=5, bars_per_day=75, seed=42):
    """Synthetic 5-min intraday data (09:15 to 15:30)."""
    np.random.seed(seed)
    rows = []
    price = 2000.0
    for d in range(days):
        day = pd.Timestamp("2024-01-15") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(minutes=9 * 60 + 15 + b * 5)
            o = price
            c = price + np.random.randn() * 4
            h = max(o, c) + abs(np.random.randn()) * 2
            l = min(o, c) - abs(np.random.randn()) * 2
            rows.append({"datetime": ts, "open": o, "high": h, "low": l, "close": c,
                         "volume": np.random.randint(1000, 5000)})
            price = c
    return pd.DataFrame(rows)


@test("Backtest engine: sma_crossover runs on daily data")
def test_engine_sma():
    from strategies import SMACrossoverStrategy
    from backtest_engine import BacktestEngine
    data = make_daily_data()
    strat = SMACrossoverStrategy(fast_period=5, slow_period=20)
    engine = BacktestEngine(strategy=strat, data=data, initial_capital=100000)
    r = engine.run()
    assert "total_trades" in r and "net_profit" in r and "win_rate" in r
    assert r["final_capital"] > 0


@test("Backtest engine: rsi_mean_reversion runs on daily data")
def test_engine_rsi():
    from strategies import RSIMeanReversionStrategy
    from backtest_engine import BacktestEngine
    data = make_daily_data(seed=7)
    strat = RSIMeanReversionStrategy()
    engine = BacktestEngine(strategy=strat, data=data, initial_capital=100000)
    r = engine.run()
    assert "total_trades" in r and "net_profit" in r


@test("Backtest engine: macd_crossover runs on daily data")
def test_engine_macd():
    from strategies import MACDCrossoverStrategy
    from backtest_engine import BacktestEngine
    data = make_daily_data(seed=13)
    strat = MACDCrossoverStrategy()
    engine = BacktestEngine(strategy=strat, data=data, initial_capital=100000)
    r = engine.run()
    assert "total_trades" in r and "net_profit" in r


@test("Backtest engine: opening_candle runs on intraday data")
def test_engine_opening_candle():
    from strategies import OpeningCandleStrategy
    from backtest_engine import BacktestEngine
    data = make_intraday_data(days=5)
    strat = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    engine = BacktestEngine(strategy=strat, data=data, initial_capital=100000)
    r = engine.run()
    assert "total_trades" in r and "net_profit" in r
    assert r["total_trades"] >= 1, "Should have at least 1 trade in 5 days"
# ============================================================
# TEST 5: Opening candle strategy logic
# ============================================================
@test("Opening candle: bullish open -> buy signal on 2nd candle")
def test_oc_bullish():
    from strategies import OpeningCandleStrategy
    dates = pd.date_range("2024-01-15 09:15", periods=5, freq="5min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100, 101, 101.5, 102, 102.5],
        "high": [101, 102, 102, 103, 103],
        "low": [99.5, 100.5, 101, 101.5, 102],
        "close": [101, 101.5, 101.8, 102.5, 102.8],
        "volume": [1000] * 5,
    })
    s = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    r = s.generate_signals(df)
    assert r.iloc[0]["signal"] == 0, "Opening candle should have no signal"
    assert r.iloc[1]["signal"] == 1, "2nd candle should be BUY after bullish open"


@test("Opening candle: bearish open -> sell signal on 2nd candle")
def test_oc_bearish():
    from strategies import OpeningCandleStrategy
    dates = pd.date_range("2024-01-15 09:15", periods=5, freq="5min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100, 99, 98.5, 98, 97.5],
        "high": [100.5, 99.5, 99, 98.5, 98],
        "low": [99, 98, 97.5, 97, 96.5],
        "close": [99, 98.5, 98.2, 97.5, 97.2],
        "volume": [1000] * 5,
    })
@test("Opening candle: target hit -> exit signal")
def test_oc_target():
    from strategies import OpeningCandleStrategy
    dates = pd.date_range("2024-01-15 09:15", periods=4, freq="5min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100, 101, 101, 101],
        "high": [101, 102, 102, 135],   # target = 101 + 30 = 131 -> hit at 135
        "low": [99.5, 100.5, 100.5, 100.5],
        "close": [101, 101.5, 101.5, 134],
        "volume": [1000] * 4,
    })
    s = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    r = s.generate_signals(df)
    assert r.iloc[3]["signal"] == -1, "Should exit when target hit"


@test("Opening candle: stop loss hit -> exit signal")
def test_oc_stoploss():
    from strategies import OpeningCandleStrategy
    dates = pd.date_range("2024-01-15 09:15", periods=4, freq="5min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100, 101, 101, 101],
        "high": [101, 102, 102, 102],
        "low": [99.5, 100.5, 100.5, 65],  # SL = 101 - 30 = 71 -> hit at 65
        "close": [101, 101.5, 101.5, 66],
        "volume": [1000] * 4,
    })
    s = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    r = s.generate_signals(df)
    assert r.iloc[3]["signal"] == -1, "Should exit when stop loss hit"


# ============================================================
# TEST 6: Config integrity
# ============================================================
@test("Config: all 4 segments defined with required keys")
def test_config_segments():
    import config
    required = {"brokerage_rate", "brokerage_cap", "stt_buy", "stt_sell",
                "exchange_txn", "gst_rate", "sebi_rate", "stamp_duty_buy"}
    for seg in ["intraday_equity", "delivery_equity", "futures", "options"]:
        assert seg in config.CHARGES_CONFIG, f"Missing segment {seg}"
        missing = required - set(config.CHARGES_CONFIG[seg].keys())
        assert not missing, f"{seg} missing keys: {missing}"


@test("backtest_config: all documented strategies are registered")
def test_backtest_config_consistency():
    import backtest_config
    import main
    for strat in backtest_config.STRATEGIES:
        normalized = main.normalize_strategy_name(strat)
        assert normalized in main.STRATEGY_REGISTRY, \
            f"'{strat}' in backtest_config not registered (normalized: '{normalized}')"


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print(f"  RESULTS: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 60)
if FAILED:
    print("  Failed tests:")
    for name, err in FAILED:
        print(f"    - {name}: {err}")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED - framework is working correctly!")
    sys.exit(0)

    s = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    r = s.generate_signals(df)
    assert r.iloc[1]["signal"] == -1, "2nd candle should be SELL after bearish open"



@test("Opening candle: target hit -> exit signal")
def test_oc_target():
    from strategies import OpeningCandleStrategy
    dates = pd.date_range("2024-01-15 09:15", periods=4, freq="5min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100, 101, 101, 101],
        "high": [101, 102, 102, 135],
        "low": [99.5, 100.5, 100.5, 100.5],
        "close": [101, 101.5, 101.5, 134],
        "volume": [1000] * 4,
    })
    s = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    r = s.generate_signals(df)
    assert r.iloc[3]["signal"] == -1, "Should exit when target hit"


@test("Opening candle: stop loss hit -> exit signal")
def test_oc_stoploss():
    from strategies import OpeningCandleStrategy
    dates = pd.date_range("2024-01-15 09:15", periods=4, freq="5min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [100, 101, 101, 101],
        "high": [101, 102, 102, 102],
        "low": [99.5, 100.5, 100.5, 65],
        "close": [101, 101.5, 101.5, 66],
        "volume": [1000] * 4,
    })
    s = OpeningCandleStrategy(target_points=30, stop_loss_points=30)
    r = s.generate_signals(df)
    assert r.iloc[3]["signal"] == -1, "Should exit when stop loss hit"


# ============================================================
# TEST 6: Config integrity
# ============================================================
@test("Config: all 4 segments defined with required keys")
def test_config_segments():
    import config
    required = {"brokerage_rate", "brokerage_cap", "stt_buy", "stt_sell",
                "exchange_txn", "gst_rate", "sebi_rate", "stamp_duty_buy"}
    for seg in ["intraday_equity", "delivery_equity", "futures", "options"]:
        assert seg in config.CHARGES_CONFIG, f"Missing segment {seg}"
        missing = required - set(config.CHARGES_CONFIG[seg].keys())
        assert not missing, f"{seg} missing keys: {missing}"


@test("backtest_config: all documented strategies are registered")
def test_backtest_config_consistency():
    import backtest_config
    import main
    for strat in backtest_config.STRATEGIES:
        normalized = main.normalize_strategy_name(strat)
        assert normalized in main.STRATEGY_REGISTRY,             f"'{strat}' in backtest_config not registered"


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print(f"  RESULTS: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 60)
if FAILED:
    print("  Failed tests:")
    for name, err in FAILED:
        print(f"    - {name}: {err}")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED - framework is working correctly!")
    sys.exit(0)
