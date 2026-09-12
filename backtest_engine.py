"""
backtest_engine.py
------------------
Core backtesting engine that simulates trades bar-by-bar.

Takes a strategy object, historical data, and initial capital;
applies broker charges (Zerodha rates), tracks positions,
equity curve, and computes performance metrics.

Supports:
- Long and short positions
- Position sizing: fixed_quantity, fixed_capital_pct, risk_based
- Realistic charge calculation per trade
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from strategies.base import Strategy
from broker_charges import calculate_charges, estimate_entry_charges
import config

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single completed trade (round trip)."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    total_charges: float
    net_pnl: float
    charges_breakdown: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine that simulates trades based on strategy signals.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_capital: float = None,
        position_sizing: str = None,
        segment: str = "intraday_equity",
    ):
        self.strategy = strategy
        # Ensure data is sorted by datetime ascending
        self.data = data.sort_values("datetime").reset_index(drop=True).copy()
        self.initial_capital = initial_capital or config.DEFAULT_INITIAL_CAPITAL
        self.position_sizing = position_sizing or config.DEFAULT_POSITION_SIZING
        self.segment = segment

        # State
        self.capital = self.initial_capital
        self.position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []

    def _calculate_position_size(self, price: float) -> int:
        """Calculate shares to trade based on sizing method."""
        if self.position_sizing == "fixed_quantity":
            return config.DEFAULT_FIXED_QUANTITY
        elif self.position_sizing == "fixed_capital_pct":
            capital_to_use = self.capital * config.DEFAULT_CAPITAL_PCT
            charges = estimate_entry_charges(price, 1, self.segment)
            available = capital_to_use - charges
            qty = int(available / price)
            return max(qty, 1)
        elif self.position_sizing == "risk_based":
            # NOTE: This is sizing based on a nominal risk unit, NOT an enforced stop-loss.
            # Positions only close on the next opposite signal, not at a price threshold.
            risk_amount = self.capital * config.DEFAULT_RISK_PCT
            risk_per_unit = config.DEFAULT_RISK_PER_UNIT
            qty = int(risk_amount / risk_per_unit)
            return max(qty, 1)
        else:
            logger.warning(f"Unknown sizing '{self.position_sizing}', using qty=1.")
            return 1

    def run(self) -> Dict:
        """Run the backtest simulation."""
        logger.info(f"Running backtest: {self.strategy.name}")
        logger.info(f"  Capital: {self.initial_capital:,.2f}")
        logger.info(f"  Sizing: {self.position_sizing}")
        logger.info(f"  Segment: {self.segment}")
        logger.info(f"  Bars: {len(self.data)}")

        signals_data = self.strategy.generate_signals(self.data)

        for i, row in signals_data.iterrows():
            signal = row.get("signal", 0)
            price = row["close"]
            timestamp = row["datetime"]

            if signal == 1 and self.position <= 0:
                if self.position < 0:
                    self._close_position(timestamp, price)
                self._open_position(timestamp, price, "long")
            elif signal == -1 and self.position >= 0:
                if self.position > 0:
                    self._close_position(timestamp, price)
                self._open_position(timestamp, price, "short")

            unrealized = 0.0
            if self.position > 0:
                unrealized = (price - self.position_entry_price) * self.position
            elif self.position < 0:
                unrealized = (self.position_entry_price - price) * abs(self.position)

            self.equity_curve.append(self.capital + unrealized)

        if self.position != 0:
            last_row = signals_data.iloc[-1]
            self._close_position(last_row["datetime"], last_row["close"])
            self.equity_curve[-1] = self.capital

        results = self._compute_metrics()
        results["trades"] = self.trades
        results["equity_curve"] = self.equity_curve
        results["trade_log"] = self.trade_log
        return results

    def _open_position(self, timestamp, price: float, direction: str):
        """Open a new position."""
        qty = self._calculate_position_size(price)
        self.position = qty if direction == "long" else -qty
        self.position_entry_price = price
        self.position_entry_time = timestamp
        self.trade_log.append({
            "time": timestamp,
            "action": f"OPEN {direction.upper()}",
            "price": price,
            "quantity": qty,
        })
        logger.debug(f"  OPEN {direction.upper()} @ {price:.2f} x {qty}")

    def _close_position(self, timestamp, price: float):
        """Close the current position."""
        if self.position == 0:
            return

        direction = "long" if self.position > 0 else "short"
        qty = abs(self.position)

        buy_price = self.position_entry_price if direction == "long" else price
        sell_price = price if direction == "long" else self.position_entry_price

        charges = calculate_charges(
            order_type=direction,
            buy_price=buy_price,
            sell_price=sell_price,
            quantity=qty,
            segment=self.segment,
        )

        self.capital += charges["net_pnl"]

        trade = Trade(
            entry_time=self.position_entry_time,
            exit_time=timestamp,
            direction=direction,
            entry_price=self.position_entry_price,
            exit_price=price,
            quantity=qty,
            gross_pnl=charges["gross_pnl"],
            total_charges=charges["total_charges"],
            net_pnl=charges["net_pnl"],
            charges_breakdown=charges,
        )
        self.trades.append(trade)

        self.trade_log.append({
            "time": timestamp,
            "action": f"CLOSE {direction.upper()}",
            "price": price,
            "quantity": qty,
            "net_pnl": charges["net_pnl"],
            "charges": charges["total_charges"],
        })

        logger.debug(
            f"  CLOSE {direction.upper()} @ {price:.2f} | "
            f"P&L: {charges['net_pnl']:.2f} | Charges: {charges['total_charges']:.2f}"
        )

        self.position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None

    def _compute_metrics(self) -> Dict:
        """Compute performance metrics from the backtest results."""
        if not self.trades:
            return {
                "strategy": self.strategy.name,
                "total_trades": 0,
                "total_return_pct": 0.0,
                "cagr": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_charges": 0.0,
                "net_profit": 0.0,
            }

        net_pnls = [t.net_pnl for t in self.trades]
        gross_pnls = [t.gross_pnl for t in self.trades]
        total_charges = sum(t.total_charges for t in self.trades)
        net_profit = sum(net_pnls)

        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        win_rate = len(wins) / len(net_pnls) * 100 if net_pnls else 0

        total_gross_profit = sum(p for p in gross_pnls if p > 0)
        total_gross_loss = abs(sum(p for p in gross_pnls if p < 0))
        profit_factor = (
            total_gross_profit / total_gross_loss
            if total_gross_loss > 0
            else float("inf")
        )

        total_return_pct = (net_profit / self.initial_capital) * 100

        n_bars = len(self.equity_curve)
        
        # Derive annualization factor from actual timestamps (supports intraday data)
        if len(self.data) > 1:
            time_deltas = self.data["datetime"].diff().dropna()
            median_delta = time_deltas.median()
            bars_per_day = pd.Timedelta(days=1) / median_delta
        else:
            bars_per_day = 252  # fallback for daily bars
        
        annualization_factor = bars_per_day * 252
        
        if n_bars > 1:
            years = n_bars / annualization_factor
            if years > 0 and self.capital > 0:
                cagr = (
                    (self.capital / self.initial_capital) ** (1 / years) - 1
                ) * 100
            else:
                cagr = 0.0
        else:
            cagr = 0.0

        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown_pct = abs(drawdown.min())

        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(annualization_factor)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "strategy": self.strategy.name,
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_charges": round(total_charges, 2),
            "net_profit": round(net_profit, 2),
            "final_capital": round(self.capital, 2),
        }
