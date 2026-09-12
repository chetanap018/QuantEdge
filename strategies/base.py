"""
base.py
-------
Abstract base class for all trading strategies.

All strategies must inherit from Strategy and implement generate_signals().
The backtest engine calls generate_signals(data) which must return a DataFrame
with at least a 'signal' column containing:
    1  = Buy (enter long)
   -1  = Sell (enter short / exit long)
    0  = Hold (no action)
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclass this and implement generate_signals() to create a new strategy.
    The strategy should add a 'signal' column to the input DataFrame.
    """

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
