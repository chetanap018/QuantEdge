"""
backtest_engine.py
------------------
Core backtesting engine that simulates trades bar-by-bar.

Correctness fixes implemented in this version
---------------------------------------------
1. ENGINE-ENFORCED RISK EXITS
   Stop-loss, take-profit and trailing stops live in the engine (see risk.py),
   not only in strategy logic.  A position now CLOSES when its risk level is
   touched even if the strategy never emits an opposite signal.  Strategies may
   still supply per-bar exit levels (long_stop / long_target ...) which are
   preferred when use_strategy_levels=True.

2. CORRECT risk_based SIZING
   quantity = (capital * risk_pct) / (entry_price - stop_price)
   Previously sizing used a nominal "risk per unit" figure and ignored the
   actual distance to the stop, so the intended 2% risk-per-trade was not real.

3. NO LOOK-AHEAD EXECUTION
   Signals computed on bar t fill at bar t+1's OPEN (not the signal bar's
   close), matching how orders really work.  A look-ahead validation module
   (lookahead.py) can flag strategies that peek at future bars.

4. SLIPPAGE & PARTIAL FILLS / LIQUIDITY
   Fixed-bps or volatility-based slippage is applied to every fill, and an
   optional liquidity check caps fills at max_participation * bar volume.

5. PORTFOLIO INTEGRATION
   When a portfolio.PortfolioBacktest gate object is passed, entries are
   halted while the portfolio-level drawdown limit is breached.

Supports:
- Long and short positions
- Position sizing: fixed_quantity, fixed_capital_pct, risk_based
- Realistic charge calculation per trade
- Exit-reason tracking (stop_loss / take_profit / trailing_stop / signal / eod)
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from strategies.base import Strategy
from tailrisk import tail_report as _tail_report


def _apply_corr_scale(engine, qty: int) -> int:
    """Shrink qty when the engine's recent avg pairwise correlation is high."""
    try:
        avg_corr = engine.sizing_params.get("avg_corr")
        if avg_corr is None:
            return int(qty)
        from sizing import correlation_scale as _cs
        thr = float(engine.sizing_params.get("corr_threshold", 0.5))
        return max(int(round(int(qty) * _cs(float(avg_corr), thr))), 0)
    except Exception:
        return int(qty)
from broker_charges import calculate_charges, estimate_entry_charges
from execution import ExecutionModel, default_execution
from risk import RiskConfig, EXIT_REASON_SIGNAL, EXIT_REASON_STOP_LOSS, \
    EXIT_REASON_TAKE_PROFIT, EXIT_REASON_TRAILING_STOP, EXIT_REASON_END_OF_DATA
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
    entry_fill_price: float = 0.0
    exit_fill_price: float = 0.0
    slippage_cost: float = 0.0
    exit_reason: str = EXIT_REASON_SIGNAL
    requested_quantity: int = 0
    filled_quantity: int = 0


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
        execution: Optional[ExecutionModel] = None,
        risk: Optional[RiskConfig] = None,
        sizing_params: Optional[Dict] = None,
        portfolio=None,
        portfolio_symbol: str = "",
        sector: str = "",
        risk_limits=None,
        symbol: str = "",
    ):
        import backtest_config  # single source of truth for user-facing knobs

        self.strategy = strategy
        # Ensure data is sorted by datetime ascending
        self.data = data.sort_values("datetime").reset_index(drop=True).copy()
        self.initial_capital = initial_capital or config.DEFAULT_INITIAL_CAPITAL
        self.position_sizing = position_sizing or config.DEFAULT_POSITION_SIZING
        self.segment = segment

        if execution is not None:
            self.execution = execution
        else:
            from execution import ExecutionModel as _EM
            self.execution = _EM(
                fill_policy=getattr(backtest_config, "FILL_POLICY", "next_open"),
                slippage_mode=getattr(backtest_config, "SLIPPAGE_MODE", "fixed"),
                slippage_bps=getattr(backtest_config, "SLIPPAGE_BPS", 5.0),
                vol_slippage_factor=getattr(backtest_config, "VOL_SLIPPAGE_FACTOR", 0.5),
                liquidity_check=getattr(backtest_config, "LIQUIDITY_CHECK", False),
                max_participation=getattr(backtest_config, "MAX_PARTICIPATION", 0.05),
                partial_fills=getattr(backtest_config, "PARTIAL_FILLS", True),
            )

        if risk is not None:
            self.risk = risk
        else:
            from risk import RiskConfig as _RC
            self.risk = _RC(
                stop_loss_pct=getattr(backtest_config, "STOP_LOSS_PCT", 0.05),
                take_profit_pct=getattr(backtest_config, "TAKE_PROFIT_PCT", None),
                trailing_stop_pct=getattr(backtest_config, "TRAILING_STOP_PCT", None),
                trailing_activation_pct=getattr(backtest_config, "TRAILING_ACTIVATION_PCT", 0.0),
                use_strategy_levels=getattr(backtest_config, "USE_STRATEGY_LEVELS",
                                            getattr(config, "DEFAULT_USE_STRATEGY_LEVELS", True)),
            )

        self.sizing_params = dict(sizing_params or {})
        self.portfolio = portfolio
        self.portfolio_symbol = portfolio_symbol or getattr(strategy, "name", "leg")

        # State
        self.capital = self.initial_capital
        self.position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self._extreme_price = 0.0           # best price since entry (for trailing stop)
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

        self.sector = sector or ""
        self.symbol = symbol or portfolio_symbol or ""
        if risk_limits is not None:
            from risk_limits import RiskGate
            self.risk_gate = risk_limits if isinstance(
                risk_limits, RiskGate) else RiskGate(risk_limits,
                                                    self.initial_capital)
        else:
            self.risk_gate = None
            try:
                from risk_limits import RiskGate as _RG, RiskLimits as _RL
                _rl = _RL(
                    max_exposure_per_symbol_pct=getattr(
                        backtest_config, "MAX_EXPOSURE_PER_SYMBOL_PCT", None),
                    max_sector_exposure_pct=getattr(
                        backtest_config, "MAX_SECTOR_EXPOSURE_PCT", None),
                    max_concurrent_positions=getattr(
                        backtest_config, "MAX_CONCURRENT_POSITIONS", None),
                    max_daily_loss_pct=getattr(
                        backtest_config, "MAX_DAILY_LOSS_PCT", None),
                    max_portfolio_heat_pct=getattr(
                        backtest_config, "MAX_PORTFOLIO_HEAT_PCT", None),
                )
                if any(v is not None for v in _rl.to_dict().values()):
                    self.risk_gate = _RG(_rl, self.initial_capital)
            except Exception:
                self.risk_gate = None

        # Event-driven strategies hold bar-by-bar state -- reset it so
        # re-runs (walk-forward windows, optimizer trials) start clean.
        reset_fn = getattr(self.strategy, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass
        self.trade_log: List[Dict] = []
        self.exit_reason_counts = {"signal": 0, "stop_loss": 0,
                                   "take_profit": 0, "trailing_stop": 0, "eod": 0}
        self.total_slippage_cost = 0.0
        self.unfilled_orders = 0

    def _sizing_fallback(self, key, default):
        return self.sizing_params.get(key, getattr(config, default))

    def _calculate_position_size(self, price: float, direction: str = "long",
                                 stop_price: Optional[float] = None,
                                 bar_index: Optional[int] = None) -> int:
        """
        Calculate shares to trade based on sizing method.

        risk_based now uses the REAL distance-to-stop:
            qty = (capital * risk_pct) / (entry_price - stop_price)
        so the capital at risk truly equals risk_pct of capital.
        """
        if self.position_sizing == "fixed_quantity":
            return int(self._sizing_fallback("quantity", "DEFAULT_FIXED_QUANTITY"))
        elif self.position_sizing == "fixed_capital_pct":
            pct = float(self._sizing_fallback("capital_pct", "DEFAULT_CAPITAL_PCT"))
            capital_to_use = self.capital * pct
            charges = estimate_entry_charges(price, 1, self.segment)
            available = capital_to_use - charges
            qty = int(available / price)
            return max(qty, 1)
        elif self.position_sizing == "risk_based":
            risk_pct = float(self._sizing_fallback("risk_pct", "DEFAULT_RISK_PCT"))
            risk_per_unit = float(self._sizing_fallback("risk_per_unit", "DEFAULT_RISK_PER_UNIT"))

            if stop_price is not None and stop_price > 0 and stop_price != price:
                risk_per_share = abs(price - stop_price)
            else:
                # No real stop configured.  risk_based sizing is meaningless
                # without a stop — warn loudly so the user notices instead of
                # silently sizing off a nominal figure.
                risk_per_share = risk_per_unit if risk_per_unit > 0 else price * 0.01
                logger.warning(
                    "risk_based sizing for %s has no real stop-loss level "
                    "(stop_price=None). Using nominal risk_per_share=%.4f. "
                    "Set STOP_LOSS_PCT or atr_multiple so sizing reflects the "
                    "actual distance to the stop.",
                    self.strategy.name, risk_per_share,
                )

            risk_amount = self.capital * risk_pct
            qty = int(risk_amount / risk_per_share)
            _vt = self.sizing_params.get("vol_target")
            if _vt and bar_index is not None:
                try:
                    from sizing import volatility_target_size as _vts
                    _lb = int(self.sizing_params.get("vol_lookback", 20))
                    _hist = self.data["close"].iloc[max(0, bar_index - _lb):bar_index + 1]
                    _rets = _hist.pct_change().dropna()
                    _rv = float(_rets.std()) if len(_rets) > 1 else 0.0
                    qty = _vts(qty, _rv, float(_vt))
                except Exception:
                    pass
            qty = _apply_corr_scale(self, qty)
            return max(int(qty), 1)
        elif self.position_sizing == "atr_stop":
            # ATR-based sizing as a first-class method:
            #   qty = (capital * risk_pct) / (atr_multiple * ATR)
            # The engine also uses the same ATR distance as the stop
            # (cached per entry bar), so rupee risk == capital * risk_pct.
            from sizing import atr as _atr_fn, atr_position_size as _aps
            _per = int(self.sizing_params.get("atr_period", 14))
            _mult = float(self.sizing_params.get("atr_multiple", 2.0))
            _rp = float(self._sizing_fallback("risk_pct", "DEFAULT_RISK_PCT"))
            _idx = bar_index if bar_index is not None else len(self.data) - 1
            try:
                if getattr(self, "_atr_cache", None) is None:
                    self._atr_cache = _atr_fn(self.data, period=_per)
                _av = float(self._atr_cache.iloc[_idx])
            except Exception:
                _av = 0.0
            if _av > 0:
                return max(int(_apply_corr_scale(
                    self, _aps(self.capital, _rp, price, _av,
                               atr_multiple=_mult))), 1)
            return 1
        else:
            logger.warning(f"Unknown sizing '{self.position_sizing}', using qty=1.")
            return 1

    def run(self) -> Dict:
        """Run the backtest simulation with realistic, look-ahead-safe execution.

        Execution model:
            * A signal produced on bar t is executed at bar t+1's OPEN
              (unless fill_policy="same_close").
            * Risk exits (stop-loss / take-profit / trailing) are checked
              intra-bar using high/low and fill at their trigger level.
            * Portfolio gate (optional) can halt new entries.
        """
        logger.info(f"Running backtest: {self.strategy.name}")
        logger.info(f"  Capital: {self.initial_capital:,.2f}")
        logger.info(f"  Sizing: {self.position_sizing}")
        logger.info(f"  Segment: {self.segment}")
        logger.info(f"  Fill policy: {self.execution.fill_policy}")
        logger.info(f"  Bars: {len(self.data)}")

        signals_data = self.strategy.generate_signals(self.data)

        # Build ATR-stop cache when sizing needs it (atr_multiple set or
        # position_sizing == "atr_stop").  The cache maps entry bar -> stop
        # price so entries and risk exits use the SAME ATR distance.
        try:
            _need_atr = (self.position_sizing == "atr_stop"
                         or self.sizing_params.get("atr_multiple") is not None)
            if _need_atr:
                from sizing import atr as _atr_fn
                _per = int(self.sizing_params.get("atr_period", 14))
                _mult = float(self.sizing_params.get("atr_multiple", 2.0))
                _series = _atr_fn(self.data, period=_per)
                self._atr_cache = _series
                self._atr_stop_cache = {}
                for _bi in range(len(self.data)):
                    try:
                        _av = float(_series.iloc[_bi])
                    except Exception:
                        continue
                    if _av > 0:
                        _px = float(self.data["close"].iloc[_bi])
                        self._atr_stop_cache[_bi] = {
                            "long": _px - _mult * _av,
                            "short": _px + _mult * _av,
                        }
            else:
                self._atr_stop_cache = {}
        except Exception:
            self._atr_stop_cache = {}

        pending = None  # (signal, signal_bar_idx); filled at NEXT bar's open
        signal_bar_idx = {i: i for i in range(len(signals_data))}
        sig_positions = list(signals_data.index)
        for pos, (_, row) in enumerate(signals_data.iterrows()):
            timestamp = row["datetime"]
            fill_base = row["open"] if self.execution.fill_policy == "next_open" else row["close"]

            # ---- 1) Execute any pending order (from bar i-1) at this bar's open ----
            # pending = (signal, signal_bar_pos).  bar_index=pos threads the
            # fill bar into sizing so ATR/vol-target use only data <= signal.
            if pending is not None:
                signal, sig_pos = pending
                pending = None
                if signal == 1 and self.position <= 0:
                    if self.position < 0:
                        self._close_position(timestamp, fill_base, EXIT_REASON_SIGNAL,
                                             side="sell", fill_bar=row)
                    self._open_from_signal(timestamp, fill_base, row, "long",
                                           bar_index=sig_pos)
                elif signal == -1 and self.position >= 0:
                    if self.position > 0:
                        self._close_position(timestamp, fill_base, EXIT_REASON_SIGNAL,
                                             side="buy", fill_bar=row)
                    self._open_from_signal(timestamp, fill_base, row, "short",
                                           bar_index=sig_pos)

            # ---- 2) Risk exits for the open position (checked intra-bar) ----
            if self.position != 0:
                reason, level = self._check_risk_exit(row)
                if reason:
                    exit_price = self._risk_exit_price(row, reason, level)
                    self._close_position(timestamp, exit_price, reason)

            # ---- 3) New signal at bar i -> schedule execution for next bar ----
            signal = row.get("signal", 0)
            if signal in (1, -1):
                pending = (signal, pos)

            # ---- 4) Mark-to-market equity at bar close ----
            unrealized = 0.0
            mark = row["close"]
            if self.position > 0:
                unrealized = (mark - self.position_entry_price) * self.position
            elif self.position < 0:
                unrealized = (self.position_entry_price - mark) * abs(self.position)
            equity = self.capital + unrealized
            self.equity_curve.append(equity)
            if self.portfolio is not None:
                self.portfolio.update_equity(equity, timestamp)

        # ---- Forced close at end of data ----
        if self.position != 0:
            last_row = signals_data.iloc[-1]
            eod_side = "sell" if self.position > 0 else "buy"
            self._close_position(last_row["datetime"], last_row["close"],
                                 EXIT_REASON_END_OF_DATA, side=eod_side,
                                 fill_bar=last_row)
            if self.equity_curve:
                self.equity_curve[-1] = self.capital

        results = self._compute_metrics()
        results["trades"] = self.trades
        results["equity_curve"] = self.equity_curve
        results["trade_log"] = self.trade_log
        results["exit_reason_counts"] = dict(self.exit_reason_counts)
        results["total_slippage_cost"] = round(self.total_slippage_cost, 2)
        results["unfilled_orders"] = self.unfilled_orders
        return results

    def _open_position(self, timestamp, price: float, direction: str,
                       qty: int, requested_qty: int = 0):
        """Open a new position."""
        self.position = qty if direction == "long" else -qty
        self.position_entry_price = price
        self.position_entry_time = timestamp
        self._extreme_price = price
        self._open_requested_qty = requested_qty if requested_qty else qty
        self.trade_log.append({
            "time": timestamp,
            "action": f"OPEN {direction.upper()}",
            "price": price,
            "quantity": qty,
            "requested_quantity": self._open_requested_qty,
        })
        logger.debug(f"  OPEN {direction.upper()} @ {price:.2f} x {qty} (req {requested_qty})")

    def _open_from_signal(self, timestamp, base_price: float, bar, direction: str,
                          bar_index: Optional[int] = None):
        """
        Open a position from a signal at the next bar's open.

        Order flow:
            requested_qty = sizing (risk_based uses real distance-to-stop)
            fill_qty      = liquidity cap may partially fill / cancel
            fill_price    = base +/- slippage (fixed bps or volatility)
        """
        stop_price = self.risk.stop_price(direction, base_price, None)
        if stop_price is None and getattr(self, "_atr_stop_cache", None) \
                and bar_index is not None:
            try:
                stop_price = self._atr_stop_cache.get(
                    int(bar_index), {}).get(direction)
            except Exception:
                stop_price = None
        requested = self._calculate_position_size(
            base_price, direction, stop_price, bar_index=bar_index)
        if requested < 1:
            logger.debug("  no sizing -> skip")
            return

        notional = abs(float(base_price)) * int(requested)
        if getattr(self, "risk_gate", None) is not None:
            try:
                self.risk_gate.update_day(
                    timestamp, float(self.capital))
                ok, reason = self.risk_gate.can_open(
                    self.symbol or self.portfolio_symbol or direction,
                    notional, sector=self.sector or None,
                    timestamp=timestamp, equity=float(self.capital))
                if not ok:
                    self.risk_gate.halts.append({
                        "time": timestamp, "price": float(base_price),
                        "quantity": int(requested),
                        "reason": f"risk_gate:{reason}",
                    })
                    logger.debug(f"  ENTRY BLOCKED (risk gate: {reason})")
                    return
            except Exception:
                pass

        if self.portfolio is not None:
            ok = self.portfolio.allow_entry(self.portfolio_symbol, timestamp, base_price, requested)
            if not ok:
                logger.debug(f"  ENTRY BLOCKED (portfolio gate) {direction.upper()} x{requested}")
                return

        fill_bar = self._bar_for_fill(bar)
        qty = self.execution.liquidity_cap(fill_bar, requested)
        if qty <= 0:
            self.unfilled_orders += 1
            logger.debug(f"  ORDER CANCELLED (no liquidity): {direction.upper()} x{requested} @ {base_price:.2f}")
            return

        side = "buy" if direction == "long" else "sell"
        fill_price = self.execution.fill_price_for(fill_bar, side, ref_price=base_price)
        at_risk = abs(fill_price - base_price) * qty
        self.total_slippage_cost += at_risk

        self._open_position(timestamp, fill_price, direction, qty, requested)

    def _bar_for_fill(self, signal_bar):
        """
        The bar used to size liquidity/slippage for a fill.

        With next_open execution, the fill happens at the CURRENT bar's open
        (the bar after the signal), which is exactly `signal_bar`. This bar's
        own volume/high-low describe the liquidity & volatility the order with
        `fill_policy="same_close"` would face on the signal bar itself.
        """
        return signal_bar

    def _check_risk_exit(self, bar):
        """Check engine-enforced stop-loss / take-profit / trailing stop.

        Returns
        -------
        (reason, trigger_level) or (None, None) if nothing is hit.
        """
        if self.position == 0:
            return (None, None)

        direction = "long" if self.position > 0 else "short"
        entry = self.position_entry_price

        if direction == "long":
            self._extreme_price = max(self._extreme_price, float(bar["high"]))
            stop = self.risk.stop_price("long", entry, bar)
            if stop is not None and float(bar["low"]) <= stop:
                return (EXIT_REASON_STOP_LOSS, stop)
            target = self.risk.target_price("long", entry, bar)
            if target is not None and float(bar["high"]) >= target:
                return (EXIT_REASON_TAKE_PROFIT, target)
            if self.risk.trailing_active("long", entry, self._extreme_price):
                trail = self.risk.trailing_stop_price("long", self._extreme_price)
                if trail is not None and float(bar["low"]) <= trail:
                    return (EXIT_REASON_TRAILING_STOP, trail)
        else:
            self._extreme_price = min(self._extreme_price, float(bar["low"]))
            stop = self.risk.stop_price("short", entry, bar)
            if stop is not None and float(bar["high"]) >= stop:
                return (EXIT_REASON_STOP_LOSS, stop)
            target = self.risk.target_price("short", entry, bar)
            if target is not None and float(bar["low"]) <= target:
                return (EXIT_REASON_TAKE_PROFIT, target)
            if self.risk.trailing_active("short", entry, self._extreme_price):
                trail = self.risk.trailing_stop_price("short", self._extreme_price)
                if trail is not None and float(bar["high"]) >= trail:
                    return (EXIT_REASON_TRAILING_STOP, trail)

        return (None, None)

    def _risk_exit_price(self, bar, reason: str, level: float) -> float:
        """
        Fill price for a risk-managed exit.

        Take-profit is treated as a LIMIT order -> fills exactly at the level.
        Stop-loss / trailing-stop are MARKET orders -> slippage is applied
        adversarially (configurable via risk.apply_slippage_to_risk_exits).
        """
        if reason == EXIT_REASON_TAKE_PROFIT or not self.risk.apply_slippage_to_risk_exits:
            return float(level)

        direction = "long" if self.position > 0 else "short"
        side = "sell" if direction == "long" else "buy"
        impact = self.execution.slippage_impact(bar, ref_price=float(level))
        # adversarial: selling into slippage -> lower price; buying -> higher price
        if side == "sell":
            return float(level) - impact
        return float(level) + impact

    def _close_position(self, timestamp, price: float,
                         reason: str = EXIT_REASON_SIGNAL,
                         side: str = None, fill_bar=None):
        """Close the current position.

        When *side* and *fill_bar* are provided (signal-based exits), slippage
        is applied adversarially to the fill price, matching the entry side:
        signal exits also cross the spread, they don't get the clean open.
        Risk exits computed by *_risk_exit_price* already include their own
        slippage and should omit these params.
        """
        if self.position == 0:
            return

        direction = "long" if self.position > 0 else "short"
        qty = abs(self.position)

        # Apply slippage for signal-based exits (same as entry side)
        slippage_cost = 0.0
        if side is not None and fill_bar is not None:
            fill_price = self.execution.fill_price_for(fill_bar, side,
                                                        ref_price=price)
            slippage_cost = abs(fill_price - price) * qty
            self.total_slippage_cost += slippage_cost
            price = fill_price

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
            entry_fill_price=self.position_entry_price,
            exit_fill_price=price,
            slippage_cost=round(slippage_cost, 4),
            exit_reason=reason,
            requested_quantity=getattr(self, "_open_requested_qty", qty),
            filled_quantity=qty,
        )
        self.trades.append(trade)
        self.exit_reason_counts[reason] = self.exit_reason_counts.get(reason, 0) + 1

        self.trade_log.append({
            "time": timestamp,
            "action": f"CLOSE {direction.upper()} ({reason})",
            "price": price,
            "quantity": qty,
            "net_pnl": charges["net_pnl"],
            "charges": charges["total_charges"],
        })

        logger.debug(
            f"  CLOSE {direction.upper()} [{reason}] @ {price:.2f} | "
            f"P&L: {charges['net_pnl']:.2f} | Charges: {charges['total_charges']:.2f}"
        )

        self.position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self._extreme_price = 0.0
    def _compute_metrics(self) -> Dict:
        """Compute performance metrics from the backtest results."""
        if not self.trades:
            tail0 = _tail_report([], 0.95)
            gate0 = list(getattr(self.risk_gate, "halts", []) or []) if getattr(self, "risk_gate", None) is not None else []
            return {
                "strategy": self.strategy.name,
                "total_trades": 0,
                "total_return_pct": 0.0,
                "cagr": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                **tail0,
                "risk_gate_halts": gate0,
                "total_charges": 0.0,
                "net_profit": 0.0,
                "final_capital": round(self.capital, 2),
                "exit_reasons": {
                    k: self.exit_reason_counts.get(k, 0) for k in
                    ("signal", "stop_loss", "take_profit", "trailing_stop", "eod")
                },
                "total_slippage_cost": round(self.total_slippage_cost, 2),
                "unfilled_orders": self.unfilled_orders,
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

        tail = _tail_report([t.net_pnl for t in self.trades], 0.95)
        gate_halts = list(getattr(self.risk_gate, "halts", []) or []) if getattr(self, "risk_gate", None) is not None else []
        return {
            "strategy": self.strategy.name,
            **tail,
            "risk_gate_halts": gate_halts,
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
            "exit_reasons": {
                k: self.exit_reason_counts.get(k, 0) for k in
                ("signal", "stop_loss", "take_profit", "trailing_stop", "eod")
            },
            "total_slippage_cost": round(self.total_slippage_cost, 2),
            "unfilled_orders": self.unfilled_orders,
        }
