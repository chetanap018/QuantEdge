"""
sma_crossover.py
----------------
Simple Moving Average Crossover Strategy.

Generates buy signals when the fast SMA crosses above the slow SMA,
and sell signals when the fast SMA crosses below the slow SMA.

Parameters:
    fast_period (int): Fast SMA lookback period (default: 20)
    slow_period (int): Slow SMA lookback period (default: 50)
"""

import pandas as pd
import numpy as np
from strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    """
    SMA Crossover Strategy.

    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50, **kwargs):
        super().__init__(
            name="SMA Crossover",
            fast_period=fast_period,
            slow_period=slow_period,
            **kwargs,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SMA crossover signals.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with 'close' column.

        Returns
        -------
        pd.DataFrame
            Data with 'fast_sma', 'slow_sma', and 'signal' columns added.
        """
        df = data.copy()

        # Calculate SMAs
        df["fast_sma"] = df["close"].rolling(window=self.fast_period).mean()
        df["slow_sma"] = df["close"].rolling(window=self.slow_period).mean()

        # Generate raw position: 1 when fast > slow, -1 when fast < slow
        df["position"] = 0
        df.loc[df["fast_sma"] > df["slow_sma"], "position"] = 1
        df.loc[df["fast_sma"] < df["slow_sma"], "position"] = -1

        # Generate signals on crossovers only (change in position)
        df["signal"] = df["position"].diff()

        # Map: 2 (0->1) = buy, -2 (0->-1) = sell, etc.
        df["signal"] = df["signal"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

        # Drop rows with NaN (before slow SMA is ready)
        df.dropna(subset=["fast_sma", "slow_sma"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
