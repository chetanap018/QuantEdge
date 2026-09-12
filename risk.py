"""
risk.py
-------
Engine-enforced risk management: stop-loss, take-profit and trailing stops.

These exits are enforced INSIDE the backtest engine so results do not depend
on the strategy remembering to implement risk logic.  Strategies may also
emit their own per-bar exit levels (e.g. `long_stop` / `long_target` columns)
which the engine will prefer when `use_strategy_levels` is True.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# Exit reason constants (also stored on every Trade for diagnostics)
EXIT_REASON_SIGNAL = "signal"
EXIT_REASON_STOP_LOSS = "stop_loss"
EXIT_REASON_TAKE_PROFIT = "take_profit"
EXIT_REASON_TRAILING_STOP = "trailing_stop"
EXIT_REASON_END_OF_DATA = "eod"


@dataclass
class RiskConfig:
    """Configuration for engine-level risk-managed exits."""

    stop_loss_pct: Optional[float] = 0.05       # 5% adverse move closes the position
    take_profit_pct: Optional[float] = None     # optional fixed-profit target
    trailing_stop_pct: Optional[float] = None   # optional trailing stop (fractional)
    trailing_activation_pct: float = 0.0       # profit needed before trailing engages
    use_strategy_levels: bool = True            # prefer long_stop/long_target cols
    apply_slippage_to_risk_exits: bool = True   # conservative: slippage on stops too

    def stop_price(self, direction: str, entry_price: float, bar: Dict = None) -> Optional[float]:
        """
        Static stop-loss level for a position.

        If the strategy emitted a column `{dir}_stop` with a usable value on
        `bar`, that level is preferred (when use_strategy_levels=True).
        """
        if direction == "" :
            return None
        if bar is not None and self.use_strategy_levels:
            col = f"{direction}_stop"
            val = bar.get(col)
            if val is not None and not _isnan(val):
                return float(val)

        if self.stop_loss_pct in (None, 0):
            return None
        if direction == "long":
            return entry_price * (1.0 - self.stop_loss_pct)
        return entry_price * (1.0 + self.stop_loss_pct)

    def target_price(self, direction: str, entry_price: float, bar: Dict = None) -> Optional[float]:
        """Take-profit level (strategy column preferred when available)."""
        if bar is not None and self.use_strategy_levels:
            col = f"{direction}_target"
            val = bar.get(col)
            if val is not None and not _isnan(val):
                return float(val)

        if self.take_profit_pct in (None, 0):
            return None
        if direction == "long":
            return entry_price * (1.0 + self.take_profit_pct)
        return entry_price * (1.0 - self.take_profit_pct)

    def trailing_stop_price(self, direction: str, extreme_price: float) -> Optional[float]:
        """
        Trailing stop computed from the extreme (best) price seen since entry.

        For a long: extreme = highest high seen; stop = extreme * (1 - pct).
        For a short: extreme = lowest low seen;   stop = extreme * (1 + pct).
        """
        if self.trailing_stop_pct in (None, 0):
            return None
        if direction == "long":
            return extreme_price * (1.0 - self.trailing_stop_pct)
        return extreme_price * (1.0 + self.trailing_stop_pct)

    def trailing_active(self, direction: str, entry_price: float, extreme_price: float) -> bool:
        """Whether the trailing stop has engaged (position in profit by activation pct)."""
        if self.trailing_stop_pct in (None, 0):
            return False
        if direction == "long":
            return extreme_price >= entry_price * (1.0 + self.trailing_activation_pct)
        return extreme_price <= entry_price * (1.0 - self.trailing_activation_pct)

    def to_dict(self) -> dict:
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "trailing_activation_pct": self.trailing_activation_pct,
            "use_strategy_levels": self.use_strategy_levels,
        }


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False