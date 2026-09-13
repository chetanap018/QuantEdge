"""
optimize.py
-----------
Parameter optimization for strategies, with built-in overfitting guards.

* grid_search() -- exhaustive cartesian search over a param grid.
* bayesian_search() -- Optuna-based search when optuna is installed
  (falls back to seeded random search otherwise).

Overfitting guards (reported for every candidate):
  * In-sample vs out-of-sample metrics side by side -- the data is split
    time-ordered (train_frac) and every candidate is evaluated on BOTH.
  * overfit_gap = IS Sharpe - OOS Sharpe (large positive gap = overfit).
  * Parameter sensitivity flag -- candidates whose OOS Sharpe collapses
    (>50% worse than IS, or OOS Sharpe < 0 while IS > 0.5) are flagged
    HIGH_SENSITIVITY so you don't ship them.

Metric maximized: OOS Sharpe by default (configurable via `metric`).
"""

import itertools
import logging
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _split_is_oos(data: pd.DataFrame, train_frac: float = 0.7):
    data = data.sort_values("datetime").reset_index(drop=True)
    cut = int(len(data) * train_frac)
    return (data.iloc[:cut].reset_index(drop=True),
            data.iloc[cut:].reset_index(drop=True))


def _evaluate(factory: Callable, data: pd.DataFrame,
              engine_kwargs: Optional[Dict] = None) -> Dict:
    from backtest_engine import BacktestEngine
    engine_kwargs = engine_kwargs or {}
    engine = BacktestEngine(strategy=factory(), data=data, **engine_kwargs)
    res = engine.run()
    return res


def _sensitivity_flag(is_sharpe: float, oos_sharpe: float) -> str:
    if is_sharpe > 0.5 and oos_sharpe < 0:
        return "HIGH_SENSITIVITY"
    if is_sharpe > 0 and (is_sharpe - oos_sharpe) / max(is_sharpe, 1e-9) > 0.5:
        return "HIGH_SENSITIVITY"
    return "ok"


def _score_candidate(strategy_cls, params: Dict, train: pd.DataFrame,
                     test: pd.DataFrame, engine_kwargs: Optional[Dict],
                     metric: str):
    factory = lambda: strategy_cls(**params)  # noqa: E731
    is_res = _evaluate(factory, train, engine_kwargs)
    oos_res = _evaluate(factory, test, engine_kwargs)
    is_m, oos_m = is_res.get(metric, 0.0), oos_res.get(metric, 0.0)
    gap = (is_res.get("sharpe_ratio", 0.0)
           - oos_res.get("sharpe_ratio", 0.0))
    return {
        "params": dict(params),
        f"is_{metric}": round(float(is_m), 4),
        f"oos_{metric}": round(float(oos_m), 4),
        "is_sharpe": round(float(is_res.get("sharpe_ratio", 0.0)), 4),
        "oos_sharpe": round(float(oos_res.get("sharpe_ratio", 0.0)), 4),
        "is_net_profit": round(float(is_res.get("net_profit", 0.0)), 2),
        "oos_net_profit": round(float(oos_res.get("net_profit", 0.0)), 2),
        "is_trades": int(is_res.get("total_trades", 0)),
        "oos_trades": int(oos_res.get("total_trades", 0)),
        "overfit_gap": round(float(gap), 4),
        "sensitivity": _sensitivity_flag(
            float(is_res.get("sharpe_ratio", 0.0)),
            float(oos_res.get("sharpe_ratio", 0.0))),
    }


def grid_search(strategy_cls, param_grid: Dict[str, List],
                data: pd.DataFrame, train_frac: float = 0.7,
                metric: str = "sharpe_ratio",
                engine_kwargs: Optional[Dict] = None,
                verbose: bool = True) -> Dict:
    """Exhaustive grid search. Returns ranked candidates + best (by OOS)."""
    train, test = _split_is_oos(data, train_frac)
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            row = _score_candidate(strategy_cls, params, train, test,
                                   engine_kwargs, metric)
        except Exception as e:  # one bad combo must not kill the sweep
            logger.warning("grid combo %s failed: %s", params, e)
            continue
        results.append(row)
        if verbose:
            print(f"  {params} -> IS {row[f'is_{metric}']} / "
                  f"OOS {row[f'oos_{metric}']} [{row['sensitivity']}]")
    results.sort(key=lambda r: r[f"oos_{metric}"], reverse=True)
    return {"metric": metric, "train_frac": train_frac,
            "candidates": results,
            "best": results[0] if results else None}


def _suggest_params(param_space: Dict, rng=None, trial=None):
    """Draw one candidate from param_space (optuna trial or random)."""
    params = {}
    for name, spec in param_space.items():
        if isinstance(spec, list):  # categorical
            if trial is not None:
                params[name] = trial.suggest_categorical(name, spec)
            else:
                params[name] = rng.choice(spec)
        else:
            lo, hi = spec
            is_int = isinstance(lo, int) and isinstance(hi, int)
            if trial is not None:
                params[name] = (trial.suggest_int(name, lo, hi) if is_int
                                else trial.suggest_float(name, lo, hi))
            else:
                params[name] = (rng.randint(lo, hi) if is_int
                                else rng.uniform(lo, hi))
    return params


def bayesian_search(strategy_cls, param_space: Dict,
                    data: pd.DataFrame, n_trials: int = 30,
                    train_frac: float = 0.7, metric: str = "sharpe_ratio",
                    engine_kwargs: Optional[Dict] = None,
                    seed: int = 42, verbose: bool = True) -> Dict:
    """Optuna search if installed, else seeded random search.

    param_space: {name: (low, high)} for ints/floats, or
                 {name: [choices...]} for categoricals.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _has_optuna = True
    except ImportError:
        _has_optuna = False

    train, test = _split_is_oos(data, train_frac)
    results: List[Dict] = []

    if _has_optuna:
        def _objective(trial):
            params = _suggest_params(param_space, trial=trial)
            row = _score_candidate(strategy_cls, params, train, test,
                                   engine_kwargs, metric)
            results.append(row)
            return row[f"oos_{metric}"]
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(_objective, n_trials=n_trials)
        if verbose:
            print(f"  optuna best: {study.best_params} "
                  f"OOS {study.best_value:.4f}")
    else:
        rng = random.Random(seed)
        for _ in range(n_trials):
            params = _suggest_params(param_space, rng=rng)
            try:
                row = _score_candidate(strategy_cls, params, train, test,
                                       engine_kwargs, metric)
            except Exception as e:
                logger.warning("trial %s failed: %s", params, e)
                continue
            results.append(row)
        if verbose:
            print("  (optuna not installed -- used seeded random search)")

    results.sort(key=lambda r: r[f"oos_{metric}"], reverse=True)
    return {"metric": metric, "train_frac": train_frac,
            "method": "optuna" if _has_optuna else "random",
            "candidates": results,
            "best": results[0] if results else None}