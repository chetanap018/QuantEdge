"""
execution.py
------------
Realistic order execution model for the backtesting engine.

Features
--------
- Fill policy:
    * "next_open"   (default) -- signals computed on bar t fill at bar t+1's OPEN.
      This removes the same-bar-close look-ahead that overstates results.
    * "same_close"  -- legacy fill at the signal bar's close (opt-in only).
- Slippage:
    * "fixed"       -- fixed basis points (slippage_bps) on the fill price.
    * "volatility"  -- slippage = vol_slippage_factor * (high - low) of the fill bar,
      scaled to the fill price (wider bars cost more to cross).
- Liquidity / partial fills:
    * When liquidity_check is on, a fill is capped at
      max_participation * fill_bar_volume.  If the requested quantity exceeds the
      cap: partial_fills=True  -> fill the capped quantity (>=1)
                         False -> cancel the order entirely (qty 0).

All price impacts are adversarial:
    buy  -> fill price = price + impact
    sell -> fill price = price - impact
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class ExecutionModel:
    """Cost / timing model applied to every order fill."""

    fill_policy: str = "next_open"          # "next_open" | "same_close"
    slippage_mode: str = "fixed"            # "fixed" | "volatility"
    slippage_bps: float = 5.0               # fixed slippage in basis points (1 bp = 0.01%)
    vol_slippage_factor: float = 0.5        # vol-based: k * (high - low)
    liquidity_check: bool = False           # enforce volume-based fill caps
    max_participation: float = 0.05         # max 5% of a bar's traded volume
    partial_fills: bool = True              # scale down instead of cancelling
    vol_column: str = "volume"

    def __post_init__(self) -> None:
        if self.fill_policy not in ("next_open", "same_close"):
            raise ValueError(
                f"fill_policy must be 'next_open' or 'same_close', got '{self.fill_policy}'."
            )
        if self.slippage_mode not in ("fixed", "volatility"):
            raise ValueError(
                f"slippage_mode must be 'fixed' or 'volatility', got '{self.slippage_mode}'."
            )
        if not 0.0 <= self.max_participation <= 1.0:
            raise ValueError("max_participation must be within [0, 1].")

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------
    def slippage_impact(self, bar: pd.Series, ref_price: Optional[float] = None) -> float:
        """
        Price impact in INR per share for the given fill bar.

        Parameters
        ----------
        bar : pd.Series
            The bar on which the fill occurs (must contain open/high/low/close).
        ref_price : Optional[float]
            Base price to compare against (defaults to bar['open']).
        """
        price = ref_price if ref_price is not None else float(bar["open"])

        if self.slippage_mode == "volatility":
            rng = float(bar["high"]) - float(bar["low"])
            return max(self.vol_slippage_factor * rng, 0.0)

        return price * self.slippage_bps / 10_000.0

    def fill_price_for(
        self,
        bar: pd.Series,
        side: str,
        ref_price: Optional[float] = None,
    ) -> float:
        """
        Adversarial fill price on `bar`.

        Parameters
        ----------
        bar : pd.Series
            The bar the order is executed on.
        side : str
            "buy" or "sell".
        ref_price : Optional[float]
            Base price (defaults to bar['open']).
        """
        price = ref_price if ref_price is not None else float(bar["open"])
        impact = self.slippage_impact(bar, ref_price=price)
        if side == "buy":
            return price + impact
        return price - impact

    # ------------------------------------------------------------------
    # Liquidity / partial fills
    # ------------------------------------------------------------------
    def liquidity_cap(self, bar: pd.Series, requested_qty: int) -> int:
        """
        Return the maximum quantity that can be filled on `bar`.

        Returns 0 when the order must be cancelled (partial_fills=False and the
        requested quantity exceeds the cap, or when there is no tradeable volume).
        """
        if not self.liquidity_check:
            return requested_qty

        volume = float(bar.get(self.vol_column, 0.0) or 0.0)
        cap = int(volume * self.max_participation)

        if requested_qty <= cap:
            return requested_qty
        if self.partial_fills and cap >= 1:
            return cap
        return 0  # cannot fill -> cancel

    def to_dict(self) -> dict:
        return {
            "fill_policy": self.fill_policy,
            "slippage_mode": self.slippage_mode,
            "slippage_bps": self.slippage_bps,
            "vol_slippage_factor": self.vol_slippage_factor,
            "liquidity_check": self.liquidity_check,
            "max_participation": self.max_participation,
            "partial_fills": self.partial_fills,
        }


# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------
def default_execution() -> ExecutionModel:
    """Realistic defaults (fill at next bar open, 5 bps fixed slippage)."""
    return ExecutionModel(fill_policy="next_open", slippage_mode="fixed", slippage_bps=5.0)


def no_cost_execution() -> ExecutionModel:
    """Zero-slippage, same-close baseline (for A/B comparison only)."""
    return ExecutionModel(fill_policy="same_close", slippage_bps=0.0)