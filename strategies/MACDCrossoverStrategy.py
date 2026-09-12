"""
macd_crossover.py
------------------
MACD (Moving Average Convergence Divergence) Crossover Strategy.

One of the most widely used trend-following momentum strategies.
Generates buy signals when the MACD line crosses above the signal line
(bullish momentum), and sell signals when it crosses below (bearish momentum).

MACD Line   = EMA(fast_period) - EMA(slow_period)
Signal Line = EMA(MACD Line, signal_period)
Histogram   = MACD Line - Signal Line

Parameters:
    fast_period (int): Fast EMA period (default: 12)
    slow_period (int): Slow EMA period (default: 26)
    signal_period (int): Signal line EMA period (default: 9)
"""

import os
import sys

if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
from strategies.base import Strategy


class MACDCrossoverStrategy(Strategy):
    """
    MACD Crossover Strategy.

    Buy when MACD line crosses above the signal line.
    Sell when MACD line crosses below the signal line.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs,
    ):
        super().__init__(
            name="MACD Crossover",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            **kwargs,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _calculate_macd(self, prices: pd.Series):
        """
        Calculate MACD line, signal line, and histogram.

        Parameters
        ----------
        prices : pd.Series
            Price series (typically close prices).

        Returns
        -------
        tuple of pd.Series
            (macd_line, signal_line, histogram)
        """
        fast_ema = prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MACD crossover signals.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with 'close' column.

        Returns
        -------
        pd.DataFrame
            Data with 'macd_line', 'signal_line', 'histogram', and 'signal'
            columns added.
        """
        df = data.copy()

        # Calculate MACD components
        df["macd_line"], df["signal_line"], df["histogram"] = self._calculate_macd(
            df["close"]
        )

        # Generate raw position: 1 when MACD > signal, -1 when MACD < signal
        df["position"] = 0
        df.loc[df["macd_line"] > df["signal_line"], "position"] = 1
        df.loc[df["macd_line"] < df["signal_line"], "position"] = -1

        # Generate signals on crossovers only (change in position)
        df["signal"] = df["position"].diff()
        df["signal"] = df["signal"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

        # The slow EMA needs slow_period bars to stabilize meaningfully;
        # drop the initial warm-up window rather than using raw NaN checks,
        # since EMA (unlike SMA) doesn't produce NaN, just an unstable value.
        warmup = self.slow_period + self.signal_period
        df = df.iloc[warmup:].reset_index(drop=True)

        return df