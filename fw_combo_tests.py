"""Tests: strategy composition."""
from fw_fixtures import make_data
from strategies.base import EventDrivenStrategy
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from composite import WeightedVoteStrategy, UnanimousStrategy


class TestComposite:
    def test_weighted_vote_majority(self):
        data = make_data(120)
        s1 = SMACrossoverStrategy(fast_period=5, slow_period=20)
        s2 = RSIMeanReversionStrategy()
        out = WeightedVoteStrategy([s1, s2]).generate_signals(data)
        assert "signal" in out.columns and "vote_score" in out.columns
        assert set(out["signal"].unique()).issubset({-1, 0, 1})

    def test_unanimous_stricter(self):
        data = make_data(160)
        s1 = SMACrossoverStrategy(fast_period=5, slow_period=20)
        s2 = SMACrossoverStrategy(fast_period=10, slow_period=30)
        uni = UnanimousStrategy([s1, s2]).generate_signals(data)
        v1 = s1.generate_signals(data.copy())
        assert (uni["signal"] != 0).sum() <= (v1["signal"] != 0).sum()
        for _, row in uni[uni["signal"] == 1].head(5).iterrows():
            assert row["n_agree_long"] == 2

    def test_mixed_modes(self):
        data = make_data(80)

        class Ev(EventDrivenStrategy):
            def on_bar(self, bar, history, state):
                return 1 if len(history) > 30 else 0

        combo = WeightedVoteStrategy(
            [SMACrossoverStrategy(fast_period=5, slow_period=20), Ev()],
            weights=[1.0, 1.0], threshold=0.5)
        out = combo.generate_signals(data)
        assert (out["signal"] != 0).sum() > 0
