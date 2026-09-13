"""
strategies/base.py
------------------
Abstract base class for all trading strategies.

Two authoring modes are supported (Fix 3):

1. VECTORIZED (simple, default) -- subclass Strategy and implement
   generate_signals(data) -> DataFrame with a 'signal' column:
       1  = Buy (enter long)
      -1  = Sell (enter short / exit long)
       0  = Hold (no action)

2. EVENT-DRIVEN (state machines, multi-leg logic, order-book-level
   decisions) -- subclass EventDrivenStrategy and implement
   on_bar(bar, history, state) -> int.  The base class adapts it to the
   vectorized generate_signals() API by replaying bars in order, so the
   backtest engine, web UI, and optimizer all work unchanged.

Use is_event_driven(strategy) to detect the mode.
"""

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import numpy as np


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclass this and implement generate_signals() to create a new strategy.
    The strategy should add a 'signal' column to the input DataFrame.
    """

    # Set True by event-driven subclasses (see EventDrivenStrategy).
    event_driven: bool = False

    def __init__(self, name: str = "BaseStrategy", **kwargs):
        """
        Initialize the strategy.

        Parameters
        ----------
        name : str
            Human-readable name for the strategy.
        **kwargs
            Strategy-specific parameters (e.g., lookback periods).
        """
        self.name = name
        self.params = kwargs

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns: datetime, open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with an additional 'signal' column:
                1  = Buy signal
               -1  = Sell signal
                0  = Hold (no action)
        """
        pass

    def get_params(self) -> dict:
        """Return the strategy's parameters."""
        return self.params

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


def is_event_driven(strategy) -> bool:
    """True when the strategy runs in event-driven (on_bar) mode."""
    return bool(getattr(strategy, "event_driven", False))


class EventDrivenStrategy(Strategy):
    """
    Base class for event-driven (bar-by-bar) strategies.

    Implement on_bar() instead of generate_signals().  The default
    generate_signals() replays bars in chronological order and calls on_bar
    once per bar, collecting the returned signals.  Because the replay only
    ever passes history up to and including the current bar, strategies
    written this way are look-ahead safe by construction.

    on_bar contract
    ---------------
    on_bar(bar, history, state) -> int
        bar     : pd.Series -- the current bar (open/high/low/close/volume).
        history : pd.DataFrame -- all bars up to AND including the current
                  bar (never future bars).
        state   : dict -- persistent mutable dict; use it for position
                  flags, counters, pending multi-leg orders, etc.
        returns : 1 (buy), -1 (sell), or 0 (hold).

    Optional hooks
    --------------
    reset_state(state) -- re-initialize the state dict (called before each
                          generate_signals run and by engine re-runs).
    """

    event_driven: bool = True

    def __init__(self, name: str = "EventDrivenStrategy", **kwargs):
        super().__init__(name=name, **kwargs)
        self._state: Dict = {}
        self.reset_state(self._state)

    def reset_state(self, state: Dict) -> None:
        """Reset persistent state.  Override to seed custom keys."""
        state.clear()

    @abstractmethod
    def on_bar(self, bar: pd.Series, history: pd.DataFrame, state: Dict) -> int:
        """Decide the signal for the current bar.  Must return -1, 0, or 1."""
        pass

    def reset(self) -> None:
        """Public reset (called by tests / live loops between runs)."""
        self.reset_state(self._state)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replay bars in order, calling on_bar() per bar."""
        df = data.copy().sort_values("datetime").reset_index(drop=True)
        self.reset_state(self._state)
        signals = []
        for i in range(len(df)):
            bar = df.iloc[i]
            history = df.iloc[: i + 1]
            try:
                sig = self.on_bar(bar, history, self._state)
            except Exception:
                sig = 0
            signals.append(1 if sig == 1 else (-1 if sig == -1 else 0))
        df["signal"] = signals
        return df

