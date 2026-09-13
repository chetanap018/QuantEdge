"""Tests: event-driven mode (on_bar)."""
import pandas as pd
from fw_fixtures import make_data
from strategies.base import (
    Strategy, EventDrivenStrategy, is_event_driven)
from strategies.sma_crossover import SMACrossoverStrategy


class TestEventDriven:
    def test_flag(self):
        assert is_event_driven(SMACrossoverStrategy()) is False

        class S(EventDrivenStrategy):
            def on_bar(self, bar, history, state):
                return 0

        assert is_event_driven(S()) is True

    def test_matches_vectorized(self):
        data = make_data(120)

        class CrossUp(EventDrivenStrategy):
            def on_bar(self, bar, history, state):
                if len(history) < 21:
                    return 0
                sma = history["close"].iloc[-20:].mean()
                prev = history["close"].iloc[-21:-1].mean()
                if history["close"].iloc[-2] <= prev and bar["close"] > sma:
                    return 1
                return 0

        class CrossUpVec(Strategy):
            def generate_signals(self, data):
                df = data.copy()
                sma = df["close"].rolling(20).mean()
                df["signal"] = 0
                cross = (df["close"].shift(1) <= sma.shift(1)) & (
                    df["close"] > sma)
                df.loc[cross.fillna(False), "signal"] = 1
                return df

        ev = CrossUp().generate_signals(data)["signal"].tolist()
        vv = CrossUpVec().generate_signals(data)["signal"].tolist()
        assert ev == vv

    def test_state_machine_multi_leg(self):
        class Hold2(EventDrivenStrategy):
            def reset_state(self, state):
                state.clear()
                state["hold"] = 0

            def on_bar(self, bar, history, state):
                if state["hold"] > 0:
                    state["hold"] -= 1
                    if state["hold"] == 0:
                        return -1
                    return 0
                if len(history) > 5:
                    state["hold"] = 2
                    return 1
                return 0

        sigs = Hold2().generate_signals(make_data(15))["signal"].tolist()
        assert sigs[5] == 1 and sigs[7] == -1

    def test_never_sees_future(self):
        data = make_data(30)
        seen = []

        class Spy(EventDrivenStrategy):
            def on_bar(self, bar, history, state):
                seen.append(len(history))
                assert history["datetime"].max() <= bar["datetime"]
                return 0

        Spy().generate_signals(data)
        assert seen == list(range(1, 31))

    def test_engine_runs_event_strategy(self):
        from backtest_engine import BacktestEngine
        from execution import ExecutionModel
        from risk import RiskConfig

        class Always(EventDrivenStrategy):
            def on_bar(self, bar, history, state):
                if state.get("done"):
                    return 0
                if len(history) > 10:
                    state["done"] = True
                    return 1
                return 0

        eng = BacktestEngine(
            strategy=Always(), data=make_data(60), initial_capital=100000,
            position_sizing="fixed_quantity",
            execution=ExecutionModel(fill_policy="same_close",
                slippage_bps=0.0),
            risk=RiskConfig(stop_loss_pct=None, take_profit_pct=None,
                use_strategy_levels=False),
            sizing_params={"quantity": 10})
        assert eng.run()["total_trades"] >= 1
