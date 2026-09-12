"""
opening_candle.py
-----------------
Opening Candle Direction Strategy for F&O (Options).

Logic:
1. Identify the first candle of each trading day (opening candle).
2. If the opening candle is bullish (close > open) → BUY signal (Call Option).
3. If the opening candle is bearish (close < open) → SELL signal (Put Option).
4. Trade is entered at the OPEN of the second candle.
5. Target = 30 points | Stop Loss = 30 points (1:1 Risk-Reward Ratio).
6. Position exits when either target or stop loss is hit on subsequent bars.

Parameters:
    target_points (float): Target profit in points (default: 30)
    stop_loss_points (float): Stop loss in points (default: 30)
"""

import pandas as pd
import numpy as np
from strategies.base import Strategy


class OpeningCandleStrategy(Strategy):
    """
    Opening Candle Direction Strategy.

    Enters in the direction of the first candle of the day,
    with fixed target and stop loss (1:1 RRR by default).
    """

    def __init__(
        self,
        target_points: float = 30.0,
        stop_loss_points: float = 30.0,
        **kwargs,
    ):
        super().__init__(
            name="Opening Candle (30x30)",
            target_points=target_points,
            stop_loss_points=stop_loss_points,
            **kwargs,
        )
        self.target_points = target_points
        self.stop_loss_points = stop_loss_points

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate opening candle direction signals with target/stoploss exits.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with 'datetime', 'open', 'high', 'low', 'close', 'volume'.

        Returns
        -------
        pd.DataFrame
            Data with 'signal' column added:
                1  = Buy (Call Option)
                -1 = Sell (Put Option)
                0  = Hold
        """
        df = data.copy()

        # Ensure datetime is proper type
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date

        # ---- Identify opening candle (first bar of each day) ----
        df["is_opening_candle"] = False
        mask_first = ~df["date"].duplicated(keep="first")
        df.loc[mask_first, "is_opening_candle"] = True

        # ---- Determine opening candle direction ----
        # Bullish (close > open) → 1, Bearish (close < open) → -1, Doji → 0
        df["opening_direction"] = 0
        mask_bullish = (df["is_opening_candle"]) & (df["close"] > df["open"])
        mask_bearish = (df["is_opening_candle"]) & (df["close"] < df["open"])
        df.loc[mask_bullish, "opening_direction"] = 1
        df.loc[mask_bearish, "opening_direction"] = -1

        # ---- Propagate direction to all bars of that day ----
        day_direction = df.groupby("date")["opening_direction"].first()
        df["day_direction"] = df["date"].map(day_direction)

        # ---- Generate signals with target/stoploss logic ----
        signals = []
        in_position = False
        position_direction = 0
        entry_price = 0.0
        current_date = None

        for i, row in df.iterrows():
            signal = 0
            row_date = row["date"]

            # Reset at start of new day
            if row_date != current_date:
                if in_position:
                    signal = -position_direction
                    in_position = False
                    position_direction = 0
                    entry_price = 0.0
                current_date = row_date

            # Opening candle: observe only
            if row["is_opening_candle"]:
                signals.append(0)
                continue

            # Enter on second candle if direction exists
            if not in_position and row["day_direction"] != 0:
                signal = row["day_direction"]
                in_position = True
                position_direction = row["day_direction"]
                entry_price = row["open"]
                signals.append(signal)
                continue

            # Check target / stop loss
            if in_position:
                if position_direction == 1:
                    if row["high"] >= entry_price + self.target_points:
                        signal = -1
                        in_position = False
                        position_direction = 0
                        entry_price = 0.0
                    elif row["low"] <= entry_price - self.stop_loss_points:
                        signal = -1
                        in_position = False
                        position_direction = 0
                        entry_price = 0.0
                elif position_direction == -1:
                    if row["low"] <= entry_price - self.target_points:
                        signal = 1
                        in_position = False
                        position_direction = 0
                        entry_price = 0.0
                    elif row["high"] >= entry_price + self.stop_loss_points:
                        signal = 1
                        in_position = False
                        position_direction = 0
                        entry_price = 0.0

            signals.append(signal)

        df["signal"] = signals

        # Clean up helper columns
        df.drop(
            columns=["date", "is_opening_candle", "opening_direction", "day_direction"],
            inplace=True,
        )

        return df