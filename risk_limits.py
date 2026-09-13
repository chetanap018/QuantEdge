"""
risk_limits.py (part 1)
-----------------------
Position- and portfolio-level risk limits enforced BEFORE entries.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class RiskLimits:
    """Risk-limit configuration. None disables a limit."""

    max_exposure_per_symbol_pct: Optional[float] = None
    max_sector_exposure_pct: Optional[float] = None
    max_concurrent_positions: Optional[int] = None
    max_daily_loss_pct: Optional[float] = None
    max_portfolio_heat_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "max_exposure_per_symbol_pct": self.max_exposure_per_symbol_pct,
            "max_sector_exposure_pct": self.max_sector_exposure_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_portfolio_heat_pct": self.max_portfolio_heat_pct,
        }


class RiskGate:
    """Stateful pre-trade gate shared by engine and portfolio legs."""

    def __init__(self, limits: Optional[RiskLimits] = None,
                 initial_capital: float = 100000.0):
        self.limits = limits or RiskLimits()
        self.initial_capital = float(initial_capital)
        self.open_positions: Dict[str, dict] = {}
        self.halts: list = []
        self.current_day = None
        self.start_of_day_equity: float = float(initial_capital)
        self.day_halted: bool = False
        self.last_equity: float = float(initial_capital)

    def update_day(self, timestamp, equity: float) -> None:
        day = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if self.current_day is None:
            self.current_day = day
            self.start_of_day_equity = float(equity)
        elif day != self.current_day:
            self.current_day = day
            self.start_of_day_equity = float(self.last_equity)
            self.day_halted = False
        self.last_equity = float(equity)
        lim = self.limits.max_daily_loss_pct
        if lim not in (None, 0) and not self.day_halted:
            base = abs(self.start_of_day_equity) or 1.0
            day_pnl = (float(equity) - self.start_of_day_equity) / base
            if day_pnl <= -abs(lim):
                self.day_halted = True
                self.halts.append({
                    "time": timestamp, "price": float(equity),
                    "quantity": 0, "reason": "daily_loss_circuit_breaker",
                })

    def can_open(self, symbol: str, notional: float,
                 sector: Optional[str] = None,
                 timestamp=None,
                 equity: Optional[float] = None) -> Tuple[bool, str]:
        lim = self.limits
        base = abs(float(equity)) if equity else self.initial_capital
        if base <= 0:
            base = self.initial_capital
        if self.day_halted:
            return False, "daily_loss_circuit_breaker"
        if lim.max_concurrent_positions is not None:
            others = set(self.open_positions) - {symbol}
            if symbol not in self.open_positions and \
                    len(others) >= lim.max_concurrent_positions:
                return False, "max_concurrent_positions"
        if lim.max_exposure_per_symbol_pct is not None:
            existing = self.open_positions.get(symbol, {}).get("notional", 0.0)
            if (existing + abs(float(notional))) / base > \
                    lim.max_exposure_per_symbol_pct + 1e-12:
                return False, "max_exposure_per_symbol"
        if lim.max_sector_exposure_pct is not None and sector:
            mine = self.open_positions.get(symbol, {})
            mine_same = mine.get("notional", 0.0) if mine.get("sector") == sector else 0.0
            sector_total = sum(
                p["notional"] for s, p in self.open_positions.items()
                if p.get("sector") == sector and s != symbol)
            if (sector_total + mine_same + abs(float(notional))) / base > \
                    lim.max_sector_exposure_pct + 1e-12:
                return False, "max_sector_exposure"
        if lim.max_portfolio_heat_pct is not None:
            total = sum(p["notional"] for p in self.open_positions.values())
            if (total + abs(float(notional))) / base > \
                    lim.max_portfolio_heat_pct + 1e-12:
                return False, "max_portfolio_heat"
        return True, ""

    def register_open(self, symbol: str, notional: float,
                      sector: Optional[str] = None) -> None:
        self.open_positions[symbol] = {
            "notional": abs(float(notional)), "sector": sector}

    def register_close(self, symbol: str) -> None:
        self.open_positions.pop(symbol, None)

    @property
    def open_count(self) -> int:
        return len(self.open_positions)


