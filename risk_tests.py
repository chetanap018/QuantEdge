"""
risk_tests.py (part 1)
----------------------
Fix 4 tests: RiskGate limits + sizing/tail unit tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS: {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL: {name}: {e}")
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR: {name}: {type(e).__name__}: {e}")


def make_daily(n=120, seed=11, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    px = 100 + np.cumsum(rng.normal(0.2, 1.2, n))
    return pd.DataFrame({
        "datetime": dates,
        "open": px + rng.normal(0, 0.2, n),
        "high": px + np.abs(rng.normal(0.4, 0.4, n)),
        "low": px - np.abs(rng.normal(0.4, 0.4, n)),
        "close": px,
        "volume": rng.integers(5000, 20000, n),
    })


class AlwaysLong:
    """Vectorized stub: long signal on bar 0 only."""

    name = "AlwaysLong"

    def generate_signals(self, data):
        df = data.copy()
        df["signal"] = 0
        df.loc[df.index[0], "signal"] = 1
        return df

    def reset(self):
        pass


def t_symbol_exposure():
    from risk_limits import RiskLimits, RiskGate
    g = RiskGate(RiskLimits(max_exposure_per_symbol_pct=0.5), 100000)
    ok, _ = g.can_open("AAA", 40000, equity=100000)
    assert ok, "40% exposure should be allowed"
    g.register_open("AAA", 40000)
    ok2, reason = g.can_open("AAA", 20000, equity=100000)
    assert not ok2 and reason == "max_exposure_per_symbol", f"got {ok2} {reason}"


def t_concurrent_positions():
    from risk_limits import RiskLimits, RiskGate
    g = RiskGate(RiskLimits(max_concurrent_positions=1), 100000)
    g.register_open("AAA", 10000)
    ok, reason = g.can_open("BBB", 10000, equity=100000)
    assert not ok and reason == "max_concurrent_positions", f"got {ok} {reason}"
    ok2, _ = g.can_open("AAA", 5000, equity=100000)
    assert ok2, "adding to existing symbol should be allowed"


def t_sector_exposure():
    from risk_limits import RiskLimits, RiskGate
    g = RiskGate(RiskLimits(max_sector_exposure_pct=0.5), 100000)
    g.register_open("AAA", 40000, sector="BANK")
    ok, reason = g.can_open("BBB", 20000, sector="BANK", equity=100000)
    assert not ok and reason == "max_sector_exposure", f"got {ok} {reason}"
    ok2, _ = g.can_open("CCC", 20000, sector="IT", equity=100000)
    assert ok2, "different sector should be allowed"


def t_daily_circuit_breaker():
    from risk_limits import RiskLimits, RiskGate
    g = RiskGate(RiskLimits(max_daily_loss_pct=0.02), 100000)
    d0 = pd.Timestamp("2024-01-01 10:00")
    g.update_day(d0, 100000)
    g.update_day(d0, 97500)  # -2.5% intraday
    ok, reason = g.can_open("AAA", 5000, timestamp=d0, equity=97500)
    assert not ok and reason == "daily_loss_circuit_breaker", f"got {ok} {reason}"
    d1 = pd.Timestamp("2024-01-02 10:00")
    g.update_day(d1, 97500)
    ok2, _ = g.can_open("AAA", 5000, timestamp=d1, equity=97500)
    assert ok2, "breaker should reset on new day"


def t_portfolio_heat():
    from risk_limits import RiskLimits, RiskGate
    g = RiskGate(RiskLimits(max_portfolio_heat_pct=0.5), 100000)
    g.register_open("AAA", 30000)
    ok, reason = g.can_open("BBB", 30000, equity=100000)
    assert not ok and reason == "max_portfolio_heat", f"got {ok} {reason}"


def t_atr_math():
    from sizing import atr_position_size
    qty = atr_position_size(100000, 0.02, 100.0, atr_value=2.0, atr_multiple=2.0)
    assert qty == 500, f"expected 500, got {qty}"  # 2000 / 4


def t_vol_target():
    from sizing import volatility_target_size
    up = volatility_target_size(100, realised_vol=0.01, target_vol=0.02)
    assert up == 200, f"expected 200, got {up}"
    down = volatility_target_size(100, realised_vol=0.04, target_vol=0.02)
    assert down == 50, f"expected 50, got {down}"
    flat = volatility_target_size(100, realised_vol=0.0, target_vol=0.02)
    assert flat == 100, "zero realised vol must leave qty unchanged"


def t_corr_scale():
    from sizing import correlation_scale
    assert correlation_scale(0.2) == 1.0, "low corr must not shrink"
    s = correlation_scale(0.9)
    assert 0.25 <= s < 1.0, f"high corr must shrink, got {s}"


def t_var_cvar_known():
    from tailrisk import var_historical, cvar_historical
    pnls = [-100, -50, -20, 10, 30, 50]
    v = var_historical(pnls, 0.95)
    c = cvar_historical(pnls, 0.95)
    assert v > 0, "VaR magnitude must be positive"
    assert c >= v, f"CVaR {c} must be >= VaR {v}"


def t_var_empty():
    from tailrisk import tail_report
    r = tail_report([], 0.95)
    assert r["var_95"] == 0.0 and r["cvar_95"] == 0.0, r


def t_engine_gate_blocks():
    from backtest_engine import BacktestEngine
    from risk_limits import RiskLimits
    data = make_daily(60)
    eng = BacktestEngine(
        strategy=AlwaysLong(), data=data, initial_capital=100000,
        position_sizing="fixed_capital_pct",
        sizing_params={"capital_pct": 1.0},
        risk_limits=RiskLimits(max_exposure_per_symbol_pct=0.01),
        symbol="AAA",
    )
    res = eng.run()
    assert res["total_trades"] == 0, f"expected 0 trades, got {res['total_trades']}"
    assert len(res.get("risk_gate_halts", [])) >= 1, "halt must be recorded"


def t_engine_tail_keys():
    from backtest_engine import BacktestEngine
    data = make_daily(80)
    eng = BacktestEngine(
        strategy=AlwaysLong(), data=data, initial_capital=100000,
        position_sizing="fixed_quantity", sizing_params={"quantity": 10},
    )
    res = eng.run()
    assert "var_95" in res and "cvar_95" in res, f"missing tail keys: {sorted(res)}"


def t_engine_atr_stop():
    from backtest_engine import BacktestEngine
    data = make_daily(100)
    eng = BacktestEngine(
        strategy=AlwaysLong(), data=data, initial_capital=100000,
        position_sizing="atr_stop",
        sizing_params={"risk_pct": 0.02, "atr_period": 14, "atr_multiple": 2.0},
    )
    res = eng.run()
    assert res["total_trades"] >= 0  # must run without crashing


def t_portfolio_concurrent_cap():
    # Direct gate check: the portfolio attaches ONE shared RiskGate to all
    # legs; the second leg's entry attempt while the first leg's exposure
    # is still registered must be halted for max_concurrent_positions.
    from portfolio import PortfolioBacktest, PortfolioLeg
    from risk_limits import RiskLimits
    pb = PortfolioBacktest(
        [PortfolioLeg(symbol=f"S{i}", strategy=AlwaysLong(),
                      data=make_daily(60, seed=100 + i))
         for i in range(2)],
        initial_capital=200000,
        max_concurrent_positions=1,
    )
    assert pb.risk_gate is not None, "portfolio should attach a shared gate"
    pb.risk_gate.register_open("S0", 10000)
    ok, reason = pb.risk_gate.can_open("S1", 10000, equity=200000)
    assert not ok and reason == "max_concurrent_positions", \
        f"got {ok} {reason}"


TESTS = [
    ("gate: per-symbol exposure", t_symbol_exposure),
    ("gate: concurrent positions", t_concurrent_positions),
    ("gate: sector exposure", t_sector_exposure),
    ("gate: daily loss breaker", t_daily_circuit_breaker),
    ("gate: portfolio heat", t_portfolio_heat),
    ("sizing: ATR math", t_atr_math),
    ("sizing: vol target", t_vol_target),
    ("sizing: corr scale", t_corr_scale),
    ("tail: VaR/CVaR known", t_var_cvar_known),
    ("tail: empty series", t_var_empty),
    ("engine: gate blocks entry", t_engine_gate_blocks),
    ("engine: tail keys in results", t_engine_tail_keys),
    ("engine: atr_stop end-to-end", t_engine_atr_stop),
    ("portfolio: concurrent cap", t_portfolio_concurrent_cap),
]


def main():
    print("=" * 60)
    print("  FIX 4 RISK & PORTFOLIO TESTS")
    print("=" * 60)
    for name, fn in TESTS:
        check(name, fn)
    print("=" * 60)
    print(f"  FIX 4 RISK: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
