"""
portfolio.py
------------
Portfolio-level backtesting across multiple symbols.

Today the framework backtests one symbol at a time with a single capital pool;
this module adds the ability to run several strategies/symbols CONCURRENTLY on the
same shared capital with:

  * shared capital allocation across legs (weighted by inverse-volatility and a
    correlation penalty so correlated symbols don't all get big allocations),
  * correlation-aware sizing  -- weights are computed from the bar-return
    correlation matrix of the actual data supplied,
  * portfolio-level drawdown limits -- when the combined equity drawdown exceeds
    `portfolio_max_drawdown_pct`, NEW entries are halted across all legs until
    equity recovers halfway back to the prior peak (re-entry throttle),
  * per-leg results plus a combined equity path / metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# Leg definition
# ======================================================================
@dataclass
class PortfolioLeg:
    """One concurrent strategy/data stream inside the portfolio."""

    symbol: str
    strategy: object
    data: pd.DataFrame
    segment: str = "intraday_equity"
    position_sizing: str = "fixed_capital_pct"
    sizing_params: Optional[Dict] = None
    weight: Optional[float] = None          # set automatically by allocate_weights
    engine_kwargs: Optional[Dict] = field(default_factory=dict)


# ======================================================================
# Correlation-aware allocation
# ======================================================================
def _bar_returns(data: pd.DataFrame) -> pd.Series:
    """Per-bar simple returns from closes."""
    return data["close"].pct_change().dropna()


def correlation_aware_weights(
    legs: List[PortfolioLeg],
    window: int = 60,
    min_corr: float = 0.0,
) -> List[float]:
    """
    Compute capital-allocation weights from the data.

    Step 1 -- inverse-volatility weights   w_i = (1 / vol_i) / sum(...)
    Step 2 -- correlation penalty          w_i *= 1 / (1 + sum(|corr(i,j)| for j!=i))
             i.e. symbols that move with many others get *less* capital so
             correlated exposure (the classic portfolio failure) is reduced.
    Step 3 -- renormalize to sum to 1.

    Returns human-legible weights aligned with `legs`.
    """
    vols = []
    returns = {}
    for leg in legs:
        r = _bar_returns(leg.data)
        r = r.iloc[-window:] if window else r
        vol = float(r.std()) if len(r) > 1 else 1.0
        vols.append(vol if vol > 0 else 1.0)
        returns[leg.symbol] = r

    weights = np.array([1.0 / v for v in vols], dtype=float)
    weights = weights / weights.sum()

    # correlation penalty (only when we have 2+ legs)
    if len(legs) > 1:
        common = None
        for r in returns.values():
            common = r.index if common is None else common.intersection(r.index)
        if len(common) >= 5:
            mat = pd.DataFrame({leg.symbol: returns[leg.symbol].reindex(common)
                                .fillna(0.0) for leg in legs}).corr().fillna(0.0)
            for i, leg in enumerate(legs):
                others = [mat.loc[leg.symbol, o.symbol] for o in legs if o.symbol != leg.symbol]
                penalty = 1.0 / (1.0 + sum(abs(o) for o in others if min_corr <= abs(o)))
                weights[i] *= penalty
            weights = weights / weights.sum()

    return [float(w) for w in weights]


# ======================================================================
# ======================================================================
# Portfolio engine
# ======================================================================
class PortfolioBacktest:
    """
    Run multiple legs concurrently with shared capital and a portfolio-level
    drawdown gate.

    Each leg is simulated by its own BacktestEngine.  The engine consults the
    portfolio gate (allow_entry/update_equity) so that a single shared capital
    pool and portfolio-level drawdown limits are enforced across ALL symbols
    at once -- not per-symbol in isolation.
    """

    def __init__(
        self,
        legs: List[PortfolioLeg],
        initial_capital: float = 1_000_000.0,
        portfolio_max_drawdown_pct: Optional[float] = None,  # e.g. 25.0 = halt at -25%
        re_entry_recovery: float = 0.5,
        correlation_window: int = 60,
    ):
        if not legs:
            raise ValueError("PortfolioBacktest requires at least one leg.")
        self.legs = legs
        self.initial_capital = float(initial_capital)
        self.portfolio_max_drawdown_pct = portfolio_max_drawdown_pct
        self.re_entry_recovery = re_entry_recovery
        self.correlation_window = correlation_window

        weights = correlation_aware_weights(legs, window=correlation_window)
        for leg, w in zip(legs, weights):
            leg.weight = w

        # shared portfolio state
        self.portfolio_equity: List[float] = []
        self.portfolio_equity_dates: List = []
        self.peak_equity = float(initial_capital)
        self.gate_closed = False
        self.entry_halts: List[Dict] = []   # log of blocked entries

    # ------------------------------------------------------------------
    # gate API (called by engines)
    # ------------------------------------------------------------------
    def allow_entry(self, symbol: str, timestamp, price: float, qty: int) -> bool:
        """Ask the portfolio whether a new position may be opened."""
        if not self.gate_closed:
            return True
        self.entry_halts.append({
            "symbol": symbol, "time": timestamp, "price": float(price),
            "quantity": int(qty), "reason": "portfolio_drawdown_gate",
        })
        return False

    def update_equity(self, equity: float, timestamp) -> None:
        """Engine feeds its equity snapshot; portfolio updates the gate."""
        self.portfolio_equity.append(equity)
        self.portfolio_equity_dates.append(timestamp)

        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.portfolio_max_drawdown_pct is None:
            return

        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100.0
        if dd_pct >= self.portfolio_max_drawdown_pct:
            self.gate_closed = True
        elif self.gate_closed:
            # reopen when drawdown recovers halfway back toward the peak
            if dd_pct <= self.portfolio_max_drawdown_pct * (1.0 - self.re_entry_recovery):
                self.gate_closed = False

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Run every leg concurrently and aggregate portfolio results."""
        results = {}
        for leg in self.legs:
            cap_alloc = self.initial_capital * leg.weight
            size_params = dict(leg.sizing_params or {})
            size_params.setdefault("risk_pct", 0.02)

            from backtest_engine import BacktestEngine
            engine = BacktestEngine(
                strategy=leg.strategy,
                data=leg.data,
                initial_capital=cap_alloc,
                position_sizing=leg.position_sizing,
                segment=leg.segment,
                sizing_params=size_params,
                portfolio=self,               # enables gate integration
                portfolio_symbol=leg.symbol,
                **leg.engine_kwargs,
            )
            res = engine.run()
            res["allocated_capital"] = cap_alloc
            results[leg.symbol] = res

        final_equity = self.portfolio_equity[-1] if self.portfolio_equity else self.initial_capital
        combined = {
            "symbols": [leg.symbol for leg in self.legs],
            "weights": {leg.symbol: round(leg.weight, 4) for leg in self.legs},
            "initial_capital": round(self.initial_capital, 2),
            "final_capital": round(float(final_equity), 2),
            "net_profit": round(float(final_equity - self.initial_capital), 2),
            "total_return_pct": round((float(final_equity) / self.initial_capital - 1) * 100, 2),
            "entry_halts": len(self.entry_halts),
            "gate_reached": self.portfolio_max_drawdown_pct is not None and bool(self.entry_halts),
            "portfolio_max_drawdown_pct": self.portfolio_max_drawdown_pct,
        }
        if self.portfolio_equity:
            eq = np.array(self.portfolio_equity, dtype=float)
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak * 100.0
            combined["max_drawdown_pct"] = round(float(abs(dd.min())), 2)

        return {
            "legs": results,
            "combined": combined,
            "portfolio_equity": self.portfolio_equity,
            "portfolio_equity_dates": self.portfolio_equity_dates,
            "entry_halts": self.entry_halts,
        }
# Portfolio engine (continued in portfolio.py below)
# ======================================================================