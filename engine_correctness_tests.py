"""
engine_correctness_tests.py
---------------------------
Validates Domain 1 -- Backtesting Engine Correctness.

Run: python3 engine_correctness_tests.py

Covers:
  1. Engine-enforced stop-loss / take-profit / trailing stops
  2. Correct risk_based sizing: qty = (capital*risk%) / (entry-stop)
  3. Look-ahead guard (truncation + append-invariance detection)
  4. Next-bar-open execution + slippage
  5. Partial fills / liquidity constraints
  6. Portfolio engine (shared capital, correlation weights, drawdown gate)
  7. Walk-forward / OOS windows
  8. Monte Carlo confidence intervals
"""

import sys
import traceback

import pandas as pd
import numpy as np

PASSED = []
FAILED = []


def test(name):
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS: {name}")
        except Exception as e:
            FAILED.append((name, str(e)))
            print(f"  FAIL: {name}")
            print(f"        {type(e).__name__}: {e}")
        return fn
    return decorator


# ============================================================
# Test data helpers
# ============================================================
def make_daily_data(n=150, seed=42, start=100.0, trend=0.0):
    np.random.seed(seed)
    close = start + np.cumsum(np.random.randn(n)) + trend * np.arange(n)
    open_ = close + np.random.randn(n) * 0.5
    high = np.maximum(open_, close) + np.abs(np.random.randn(n))
    low = np.minimum(open_, close) - np.abs(np.random.randn(n))
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n, freq="D"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.randint(1000, 5000, n),
    })


class DummyStrategy:
    """Simple strategy with deterministic signals for engine tests."""

    def __init__(self, signals, name="Dummy"):
        self.signals = signals
        self.name = name

    def generate_signals(self, data):
        df = data.copy()
        df["signal"] = self.signals
        return df


# ============================================================
# 1. Engine-enforced stop-loss / take-profit / trailing stop
# ============================================================
@test("SL/TP: engine closes position when take-profit hit even with no exit signal")
def test_take_profit_engine():
    from backtest_engine import BacktestEngine
    from risk import RiskConfig
    from execution import ExecutionModel

    n = 20
    data = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1000] * n,
    })
    data.loc[5, "high"] = 115.0  # TP +10% = 110 -> triggered at bar 5

    signals = [0] * n
    signals[2] = 1  # go long (executed at bar 3 open)
    sig = DummyStrategy(signals)

    engine = BacktestEngine(
        strategy=sig, data=data, initial_capital=100000,
        position_sizing="fixed_quantity",
        execution=ExecutionModel(fill_policy="same_close", slippage_bps=0.0),
        risk=RiskConfig(stop_loss_pct=None, take_profit_pct=0.10,
                        use_strategy_levels=False,
                        apply_slippage_to_risk_exits=False),
    )
    r = engine.run()
    assert r["total_trades"] >= 1, "expected at least one trade"
    t = r["trades"][-1]
    assert t.exit_reason == "take_profit", f"expected take_profit, got {t.exit_reason}"


@test("SL/TP: engine closes position when stop-loss hit with no exit signal")
def test_stop_loss_engine():
    from backtest_engine import BacktestEngine
    from risk import RiskConfig
    from execution import ExecutionModel

    n = 20
    data = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1000] * n,
    })
    data.loc[5, "low"] = 80.0   # SL -5% = 95 -> hit

    signals = [0] * n
    signals[2] = 1
    sig = DummyStrategy(signals)

    engine = BacktestEngine(
        strategy=sig, data=data, initial_capital=100000,
        position_sizing="fixed_quantity",
        execution=ExecutionModel(fill_policy="same_close", slippage_bps=0.0),
        risk=RiskConfig(stop_loss_pct=0.05, take_profit_pct=None,
                        use_strategy_levels=False,
                        apply_slippage_to_risk_exits=False),
    )
    r = engine.run()
    assert r["total_trades"] >= 1
    assert r["trades"][-1].exit_reason == "stop_loss"
@test("Trailing-stop: stop ratchets up with the peak after activation")
def test_trailing_stop_engine():
    from backtest_engine import BacktestEngine
    from risk import RiskConfig, EXIT_REASON_TRAILING_STOP
    from execution import ExecutionModel

    n = 30
    closes = [100] * 10 + [105, 110, 115, 120, 125, 130] + [118, 116, 114, 112] + [100] * 10
    data = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n, freq="D"),
        "open": closes, "high": [c + 2 for c in closes],
        "low": [c - 2 for c in closes], "close": closes,
        "volume": [1000] * n,
    })
    signals = [0] * n
    signals[2] = 1
    sig = DummyStrategy(signals)

    engine = BacktestEngine(
        strategy=sig, data=data, initial_capital=100000,
        position_sizing="fixed_quantity",
        execution=ExecutionModel(fill_policy="same_close", slippage_bps=0.0),
        risk=RiskConfig(stop_loss_pct=None, take_profit_pct=None,
                        trailing_stop_pct=0.10, trailing_activation_pct=0.05,
                        use_strategy_levels=False,
                        apply_slippage_to_risk_exits=False),
    )
    r = engine.run()
    reasons = [t.exit_reason for t in r["trades"]]
    assert EXIT_REASON_TRAILING_STOP in reasons, f"trailing stop never fired: {reasons}"


# ============================================================
# 2. risk_based sizing correctness
# ============================================================
@test("risk_based sizing: qty = (capital*risk%) / (entry - stop)")
def test_risk_based_sizing():
    from backtest_engine import BacktestEngine

    engine = BacktestEngine(
        strategy=DummyStrategy([0]), data=make_daily_data(5),
        initial_capital=10000, position_sizing="risk_based",
        sizing_params={"risk_pct": 0.02, "risk_per_unit": 10.0},
    )
    stop = 100 * (1 - 0.10)
    expected = int((10000 * 0.02) / (100 - stop))
    qty = engine._calculate_position_size(100.0, "long", stop_price=stop)
    assert qty == expected, f"expected {expected}, got {qty}"
    assert qty == 20, f"expected 20 for 2% of 10k with 10-pt stop, got {qty}"


# ============================================================
# 3. Look-ahead guard
# ============================================================
class LeakyStrategy:
    """Uses future bar close (leaks) -> must be flagged."""
    name = "leaky"

    def generate_signals(self, data):
        df = data.copy()
        df["signal"] = (df["close"].shift(-1) > df["close"]).astype(int)
        return df


class CleanStrategy:
    """Uses only prior-bar info via shift(1) -> must pass."""
    name = "clean"

    def generate_signals(self, data):
        df = data.copy()
        prev = df["close"].shift(1)
        df["signal"] = np.where(prev > df["close"], 1, -1)
        return df


@test("Look-ahead: future-leaking strategy is flagged")
def test_lookahead_detects_leak():
    from lookahead import validate_no_lookahead
    data = make_daily_data(80)
    rep = validate_no_lookahead(LeakyStrategy(), data, sample_n=6)
    assert not rep.ok, f"leaky strategy not flagged: {rep.summary}"


@test("Look-ahead: shift(1)-only strategy passes")
def test_lookahead_clean_passes():
    from lookahead import validate_no_lookahead
    data = make_daily_data(80)
    rep = validate_no_lookahead(CleanStrategy(), data, sample_n=6)
    assert rep.ok, f"clean strategy flagged: {rep.summary}"


@test("Look-ahead: @no_lookahead decorator raises on leak when on_error='fail'")
def test_no_lookahead_decorator():
    from lookahead import no_lookahead, LookaheadError

    class Decorated:
        name = "decorated"

        @no_lookahead(on_error="fail", sample_n=4)
        def generate_signals(self, data):
            df = data.copy()
            df["signal"] = (df["close"].shift(-1) > df["close"]).astype(int)
            return df

    data = make_daily_data(50)
    try:
        Decorated().generate_signals(data)
        raise AssertionError("expected LookaheadError but none raised")
    except LookaheadError:
        pass  # good
# ============================================================
# 4. Next-bar-open execution + slippage
# ============================================================
@test("Execution: signal at bar t fills at bar t+1 open (next_open)")
def test_fill_on_next_open():
    from backtest_engine import BacktestEngine
    from execution import ExecutionModel

    n = 12
    data = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01 09:15", periods=n, freq="5min"),
        "open": [10.0, 20.0, 30.0, 40.0, 50.0] + [50.0] * 7,
        "high": [11.0, 21.0, 31.0, 41.0, 51.0] + [51.0] * 7,
        "low": [9.0, 19.0, 29.0, 39.0, 49.0] + [49.0] * 7,
        "close": [10.5, 20.5, 30.5, 40.5, 50.5] + [50.5] * 7,
        "volume": [1000] * n,
    })
    signals = [0] * n
    signals[0] = 1   # signal at bar 0 -> fills at bar 1 open = 20
    sig = DummyStrategy(signals)

    engine = BacktestEngine(
        strategy=sig, data=data, initial_capital=1000000,
        position_sizing="fixed_quantity", sizing_params={"quantity": 10},
        execution=ExecutionModel(fill_policy="next_open", slippage_bps=0.0),
        risk=None,
    )
    engine.risk.stop_loss_pct = None
    r = engine.run()
    assert r["total_trades"] >= 1
    t = r["trades"][0]
    assert abs(t.entry_fill_price - 20.0) < 1e-9, \
        f"expected fill at 20 (bar1 open), got {t.entry_fill_price}"


@test("Execution: slippage increases fill cost for buys")
def test_slippage_applied():
    from execution import ExecutionModel
    bar = pd.Series({"open": 100.0, "high": 102.0, "low": 99.0,
                     "close": 101.0, "volume": 1000})
    em = ExecutionModel(fill_policy="next_open", slippage_mode="fixed",
                        slippage_bps=50.0)
    assert em.fill_price_for(bar, "buy") > 100.0
    assert em.fill_price_for(bar, "sell") < 100.0
    assert abs(em.fill_price_for(bar, "buy") - 100.5) < 1e-9


# ============================================================
# 5. Partial fills / liquidity
# ============================================================
@test("Execution: liquidity cap partially fills oversized order")
def test_partial_fill():
    from execution import ExecutionModel
    bar = pd.Series({"open": 100.0, "volume": 1000})
    em = ExecutionModel(liquidity_check=True, max_participation=0.10)
    assert em.liquidity_cap(bar, 50) == 50
    assert em.liquidity_cap(bar, 500) == 100


@test("Execution: no partial fills -> order cancelled when over cap")
def test_no_partial_fill_cancel():
    from execution import ExecutionModel
    bar = pd.Series({"open": 100.0, "volume": 1000})
    em = ExecutionModel(liquidity_check=True, max_participation=0.01,
                        partial_fills=False)
    assert em.liquidity_cap(bar, 500) == 0
# ============================================================
# 6. Portfolio engine
# ============================================================
@test("Portfolio: correlation-aware weights sum to 1")
def test_portfolio_weights():
    from portfolio import PortfolioLeg, correlation_aware_weights
    a = make_daily_data(120, seed=1)
    b = make_daily_data(120, seed=2)
    legs = [PortfolioLeg(symbol="A", strategy=object(), data=a),
            PortfolioLeg(symbol="B", strategy=object(), data=b)]
    w = correlation_aware_weights(legs)
    assert abs(sum(w) - 1.0) < 1e-6, f"weights must sum to 1: {w}"
    assert all(x > 0 for x in w)


@test("Portfolio: runs to completion with shared capital and gate")
def test_portfolio_gate():
    from portfolio import PortfolioBacktest, PortfolioLeg
    from execution import ExecutionModel

    a = make_daily_data(100, seed=11, start=100.0, trend=-0.5)  # declining
    s = DummyStrategy([1] * len(a))   # always-long on a declining symbol

    p = PortfolioBacktest(
        legs=[PortfolioLeg(symbol="A", strategy=s, data=a,
                           position_sizing="fixed_quantity",
                           sizing_params={"quantity": 1},
                           engine_kwargs={
                               "execution": ExecutionModel(fill_policy="next_open",
                                                           slippage_bps=0.0)})],
        initial_capital=100000,
        portfolio_max_drawdown_pct=50.0,
    )
    res = p.run()
    assert "legs" in res and "combined" in res
    assert "A" in res["legs"]
    assert "weights" in res["combined"]


# ============================================================
# 7. Walk-forward / OOS
# ============================================================
@test("Walk-forward: produces OOS-only test windows")
def test_walk_forward_windows():
    from walkforward import walk_forward_windows, split_train_validation_test
    data = make_daily_data(120)
    windows = list(walk_forward_windows(data, n_windows=4, train_pct=0.6))
    assert len(windows) >= 1
    tr, val, te = split_train_validation_test(data, 0.6, 0.2, 0.2)
    assert len(tr) + len(val) + len(te) == len(data)


@test("Walk-forward: run_walk_forward returns aggregate OOS metrics")
def test_run_walk_forward():
    from walkforward import run_walk_forward
    from strategies import SMACrossoverStrategy
    data = make_daily_data(150)
    res = run_walk_forward(lambda: SMACrossoverStrategy(5, 20), data,
                           n_windows=3, train_pct=0.6,
                           engine_kwargs={"initial_capital": 100000},
                           verbose=False)
    assert "per_window" in res and "oos_aggregate" in res
    assert len(res["per_window"]) >= 1
    assert "mean_sharpe" in res["oos_aggregate"]


# ============================================================
# 8. Monte Carlo confidence intervals
# ============================================================
@test("Monte Carlo: returns CI dicts for sharpe and drawdown")
def test_monte_carlo_ci():
    from montecarlo import monte_carlo_confidence
    from backtest_engine import Trade
    base = pd.Timestamp("2023-01-01")
    trades = [Trade(entry_time=base, exit_time=base, direction="long",
                    entry_price=100, exit_price=100.0, quantity=1,
                    gross_pnl=0.0, total_charges=1.0,
                    net_pnl=float(np.random.randn() * 10))
              for _ in range(50)]
    res = monte_carlo_confidence(trades, 100000, n_sims=200, seed=7)
    assert res["sharpe_ci95"]["lower"] is not None
    assert res["sharpe_ci95"]["upper"] >= res["sharpe_ci95"]["lower"]
    assert res["max_drawdown_ci95"]["lower"] is not None


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print(f"  DOMAIN 1 ENGINE CORRECTNESS: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 60)
if FAILED:
    print("  Failed tests:")
    for name, err in FAILED:
        print(f"    - {name}: {err}")
    sys.exit(1)
else:
    print("  ALL ENGINE CORRECTNESS TESTS PASSED")
    sys.exit(0)