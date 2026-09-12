"""
rsi_mean_reversion.py
---------------------
RSI Mean Reversion Strategy.

Buys when RSI falls below the oversold threshold (indicating a potential bounce),
and sells when RSI rises above the overbought threshold (indicating a pullback).

Parameters:
    rsi_period (int): RSI lookback period (default: 14)
    oversold (float): RSI level considered oversold (default: 30)
    overbought (float): RSI level considered overbought (default: 70)
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


class RSIMeanReversionStrategy(Strategy):
    """
    RSI Mean Reversion Strategy.

    Buy when RSI crosses above oversold level (exit oversold).
    Sell when RSI crosses below overbought level (exit overbought).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        **kwargs,
    ):
        super().__init__(
            name="RSI Mean Reversion",
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            **kwargs,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Parameters
        ----------
        prices : pd.Series
            Price series (typically close prices).
        period : int
            Lookback period.

        Returns
        -------
        pd.Series
            RSI values (0-100).
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RSI mean-reversion signals.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with 'close' column.

        Returns
        -------
        pd.DataFrame
            Data with 'rsi' and 'signal' columns added.
        """
        df = data.copy()

        # Calculate RSI
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # Generate signals
        df["signal"] = 0

        # Buy when RSI crosses above oversold (exiting oversold zone)
        df.loc[
            (df["rsi"] > self.oversold) & (df["rsi"].shift(1) <= self.oversold),
            "signal",
        ] = 1

        # Sell when RSI crosses below overbought (exiting overbought zone)
        df.loc[
            (df["rsi"] < self.overbought) & (df["rsi"].shift(1) >= self.overbought),
            "signal",
        ] = -1

        # Drop rows with NaN
        df.dropna(subset=["rsi"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
