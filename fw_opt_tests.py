"""Tests: optimizer with overfitting guards."""
from fw_fixtures import make_data
from strategies.sma_crossover import SMACrossoverStrategy
from optimize import grid_search, bayesian_search


class TestOptimizer:
    def test_grid_search_is_oos(self):
        data = make_data(160)
        out = grid_search(SMACrossoverStrategy,
            {"fast_period": [5, 10], "slow_period": [20, 30]},
            data, train_frac=0.7, verbose=False)
        assert len(out["candidates"]) == 4
        best = out["best"]
        for key in ("is_sharpe_ratio", "oos_sharpe_ratio", "overfit_gap",
                "sensitivity", "params"):
            assert key in best, f"missing guard field {key}"
        assert best["sensitivity"] in ("ok", "HIGH_SENSITIVITY")

    def test_best_is_top_oos(self):
        data = make_data(160)
        out = grid_search(SMACrossoverStrategy,
            {"fast_period": [5, 10], "slow_period": [20, 30]},
            data, train_frac=0.7, verbose=False)
        oos = [c["oos_sharpe_ratio"] for c in out["candidates"]]
        assert out["best"]["oos_sharpe_ratio"] == max(oos)

    def test_bayesian_random_fallback(self):
        data = make_data(160)
        out = bayesian_search(SMACrossoverStrategy,
            {"fast_period": (3, 12), "slow_period": (15, 40)},
            data, n_trials=4, train_frac=0.7, verbose=False)
        assert out["method"] in ("optuna", "random")
        assert len(out["candidates"]) == 4
        assert "overfit_gap" in out["candidates"][0]
