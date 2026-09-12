"""
suzlon_intraday.py
--------------------
Suzlon Intraday Confluence Strategy.

Concept:
    15-minute trend filter + 5-minute pullback/rejection entry.

    The strategy is intentionally parameterized so the parameters can be
    optimized against historical SUZLON OHLCV data.

Expected input:
    pandas.DataFrame with at least:
        open, high, low, close, volume

    The framework provides timestamps in a `datetime` column. It is promoted
    to a DatetimeIndex internally (so session VWAP resets per day, the session
    time filter works, and higher-timeframe mapping is lookahead-free), and
    restored as a column in the returned DataFrame (the backtest engine
    expects a `datetime` column).

Higher-timeframe mode:
    When `higher_timeframe_data` (15-minute SUZLON candles) is supplied, the
    15-minute EMA/VWAP trend state is mapped onto the primary 5-minute
    timeframe and used as the directional filter. Only COMPLETED 15-minute
    bars are used (bars are shifted to their end time before forward-filling),
    so there is no lookahead from the currently-forming 15-minute candle.

Signal convention:
    +1 = LONG entry
    -1 = SHORT entry
     0 = no new signal
"""

import numpy as np
import pandas as pd

from typing import Optional

from strategies.base import Strategy


class SuzlonStrategy(Strategy):
    """
    SUZLON intraday confluence strategy.



    The intended architecture is:
        15m trend -> 5m pullback -> rejection -> breakout confirmation.


    This implementation can also operate on a single timeframe when
    higher_timeframe_data is not supplied.
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 20,
        rsi_period: int =  14,
        volume_period: int =  20,
        volume_threshold: float =  1.0,
        rsi_long_min: float =  45.0,
        rsi_long_max: float =  70.0,
        rsi_short_min: float =  30.0,
        rsi_short_max: float =  55.0,
        pullback_tolerance: float =  0.0025,
        stop_loss_pct: float =  0.0030,
        target_pct: float =  0.0050,
        use_vwap: bool = True,
        use_rsi: bool = True,
        use_volume: bool = True,
        use_time_filter: bool = True,
        session_start: str = "09:20",
        session_end: str = "15:00",
        **kwargs,
    ):
        super().__init__(
            name="Suzlon Intraday Confluence",
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_period=rsi_period,
            volume_period=volume_period,
            volume_threshold=volume_threshold,
            rsi_long_min=rsi_long_min,
            rsi_long_max=rsi_long_max,
            rsi_short_min=rsi_short_min,
            rsi_short_max=rsi_short_max,
            pullback_tolerance=pullback_tolerance,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            use_vwap=use_vwap,
            use_rsi=use_rsi,
            use_volume=use_volume,
            use_time_filter=use_time_filter,
            session_start=session_start,
            session_end=session_end,
            **kwargs,
        )

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold

        self.rsi_long_min = rsi_long_min
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.rsi_short_max = rsi_short_max

        self.pullback_tolerance = pullback_tolerance
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct

        self.use_vwap = use_vwap
        self.use_rsi = use_rsi
        self.use_volume = use_volume
        self.use_time_filter = use_time_filter

        self.session_start = session_start
        self.session_end = session_end

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Promote the datetime column to a DatetimeIndex, if present."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.set_index("datetime")

        return df

    @staticmethod
    def _restore_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
        """Restore the datetime column so the backtest engine keeps working."""
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "datetime" not in df.columns:
                df.index.name = "datetime"
                df = df.reset_index()
        else:
            df = df.reset_index(drop=True)

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using Wilder-style exponential smoothing."""
        delta = prices.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # If there has been no loss, RSI is conventionally 100.
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)

        return rsi

    @staticmethod
    def _calculate_vwap(data: pd.DataFrame) -> pd.Series:
        """
        Calculate session VWAP.

        VWAP resets at each trading day. If the index is not datetime-like,
        cumulative VWAP is used instead.
        """
        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        volume = data["volume"].fillna(0)

        if isinstance(data.index, pd.DatetimeIndex):
            session = data.index.normalize()
            cumulative_pv = (typical_price * volume).groupby(session).cumsum()
            cumulative_volume = volume.groupby(session).cumsum()
            return cumulative_pv / cumulative_volume.replace(0, np.nan)

        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

        return cumulative_pv / cumulative_volume.replace(0, np.nan)

    def _apply_time_filter(self, df: pd.DataFrame) -> pd.Series:
        """Return True for bars inside the configured trading session."""
        if not self.use_time_filter:
            return pd.Series(True, index=df.index)

        if not isinstance(df.index, pd.DatetimeIndex):
            # Without timestamps, do not silently discard all data.
            return pd.Series(True, index=df.index)

        times = df.index.time
        start = pd.Timestamp(self.session_start).time()
        end = pd.Timestamp(self.session_end).time()

        return pd.Series(
            [(start <= t <= end) for t in times],
            index=df.index,
        )

    def _build_trend_filter(
        self,
        df: pd.DataFrame,
    ) -> tuple:
        """
        Build bullish/bearish trend states.

        A trend is stronger when EMA alignment and VWAP alignment agree.
        """
        bullish = (
            (df["ema_fast"] > df["ema_slow"])
            & (df["close"] > df["ema_slow"])
        )

        bearish = (
            (df["ema_fast"] < df["ema_slow"])
            & (df["close"] < df["ema_slow"])
        )

        if self.use_vwap:
            bullish &= df["close"] > df["vwap"]
            bearish &= df["close"] < df["vwap"]

        return bullish, bearish

    def _pullback_to_reference(
        self,
        df: pd.DataFrame,
        bullish: pd.Series,
        bearish: pd.Series,
    ) -> tuple:
        """
        Detect a pullback toward EMA slow / VWAP.

        The tolerance is deliberately parameterized for optimization.
        """
        reference = df["ema_slow"]

        if self.use_vwap:
            reference = pd.concat(
                [df["ema_slow"], df["vwap"]],
                axis=1,
            ).mean(axis=1)

        distance = (
            (df["close"] - reference).abs()
            / df["close"].replace(0, np.nan)
        )

        near_reference = distance <= self.pullback_tolerance

        long_pullback = bullish.shift(1, fill_value=False) & near_reference
        short_pullback = bearish.shift(1, fill_value=False) & near_reference

        return long_pullback, short_pullback

    def _confirmation_filters(
        self,
        df: pd.DataFrame,
    ) -> tuple:
        """Return independent long and short confirmation filters."""
        long_confirm = pd.Series(True, index=df.index)
        short_confirm = pd.Series(True, index=df.index)

        if self.use_rsi:
            long_confirm &= (
                (df["rsi"] >= self.rsi_long_min)
                & (df["rsi"] <= self.rsi_long_max)
            )

            short_confirm &= (
                (df["rsi"] >= self.rsi_short_min)
                & (df["rsi"] <= self.rsi_short_max)
            )

        if self.use_volume:
            long_confirm &= (
                df["relative_volume"] >= self.volume_threshold
            )
            short_confirm &= (
                df["relative_volume"] >= self.volume_threshold
            )

        return long_confirm, short_confirm

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators used by the strategy."""
        df = data.copy()

        df["ema_fast"] = df["close"].ewm(
            span=self.ema_fast,
            adjust=False,
        ).mean()

        df["ema_slow"] = df["close"].ewm(
            span=self.ema_slow,
            adjust=False,
        ).mean()

        df["rsi"] = self._calculate_rsi(
            df["close"],
            self.rsi_period,
        )

        if "volume" in df.columns:
            df["average_volume"] = df["volume"].rolling(
                self.volume_period
            ).mean()

            df["relative_volume"] = (
                df["volume"]
                / df["average_volume"].replace(0, np.nan)
            )

            df["vwap"] = self._calculate_vwap(df)
        else:
            df["average_volume"] = np.nan
            df["relative_volume"] = np.nan
            df["vwap"] = np.nan

        # Candle structure.
        df["candle_range"] = (df["high"] - df["low"]).replace(0, np.nan)
        df["body"] = (df["close"] - df["open"]).abs()

        df["upper_wick"] = (
            df["high"] - df[["open", "close"]].max(axis=1)
        )

        df["lower_wick"] = (
            df[["open", "close"]].min(axis=1) - df["low"]
        )

        # Rejection characteristics.
        df["bullish_candle"] = df["close"] > df["open"]
        df["bearish_candle"] = df["close"] < df["open"]

        df["bullish_rejection"] = (
            df["bullish_candle"]
            & (df["lower_wick"] >= df["body"])
        )

        df["bearish_rejection"] = (
            df["bearish_candle"]
            & (df["upper_wick"] >= df["body"])
        )

        return df

    def generate_signals(
        self,
        data: pd.DataFrame,
        higher_timeframe_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate SUZLON confluence signals.

        Parameters
        ----------
        data:
            Primary OHLCV data, preferably 5-minute SUZLON candles.

        higher_timeframe_data:
            Optional 15-minute SUZLON OHLCV data.

            When supplied, the 15-minute EMA/VWAP trend is mapped onto
            the primary timeframe and used as the directional filter. Only
            completed 15-minute bars are used (no lookahead).

        Returns
        -------
        pd.DataFrame
            Original data plus indicators, setup columns and `signal`.

        Signal:
            +1 = long
            -1 = short
             0 = no new entry
        """
        df = self._ensure_datetime_index(data)
        df = self._calculate_indicators(df)

        # Primary timeframe trend.
        primary_bullish, primary_bearish = self._build_trend_filter(df)

        # Default to primary timeframe if no 15-minute data is supplied.
        df["higher_tf_bullish"] = primary_bullish
        df["higher_tf_bearish"] = primary_bearish

        if higher_timeframe_data is not None:
            ht = self._ensure_datetime_index(higher_timeframe_data)
            if not (
                isinstance(df.index, pd.DatetimeIndex)
                and isinstance(ht.index, pd.DatetimeIndex)
            ):
                raise ValueError(
                    "A datetime column or DatetimeIndex is required on both "
                    "data and higher_timeframe_data when using the "
                    "higher-timeframe trend filter."
                )

            ht = self._calculate_indicators(ht)
            ht_bullish, ht_bearish = self._build_trend_filter(ht)

            ht_state = pd.DataFrame(
                {
                    "bullish": ht_bullish.astype(int),
                    "bearish": ht_bearish.astype(int),
                },
                index=ht.index,
            )

            # Only use information from ALREADY COMPLETED 15m bars:
            # shift each 15m bar to its end time so forward-fill never
            # leaks the currently-forming 15m bar onto earlier 5m bars.
            ht_series = ht.index.to_series()
            ht_step = ht_series.diff().median()
            if pd.notna(ht_step):
                ht_step = pd.Timedelta(ht_step)
            else:
                ht_step = pd.Timedelta(minutes=15)
            ht_state.index = ht_state.index + ht_step

            ht_state = ht_state.reindex(df.index, method="ffill")

            df["higher_tf_bullish"] = ht_state["bullish"].fillna(0).astype(bool)
            df["higher_tf_bearish"] = ht_state["bearish"].fillna(0).astype(bool)

        # Pullback and rejection setup.
        long_pullback, short_pullback = self._pullback_to_reference(
            df,
            df["higher_tf_bullish"],
            df["higher_tf_bearish"],
        )

        long_confirm, short_confirm = self._confirmation_filters(df)

        session_ok = self._apply_time_filter(df)

        # Breakout confirmation:
        # enter only when the current candle takes out the previous candle.
        long_breakout = df["high"] > df["high"].shift(1)
        short_breakout = df["low"] < df["low"].shift(1)

        long_setup = (
            df["higher_tf_bullish"]
            & long_pullback
            & df["bullish_rejection"]
            & long_confirm
            & long_breakout
            & session_ok
        )

        short_setup = (
            df["higher_tf_bearish"]
            & short_pullback
            & df["bearish_rejection"]
            & short_confirm
            & short_breakout
            & session_ok
        )

        # Entry signals.
        df["long_setup"] = long_setup
        df["short_setup"] = short_setup

        df["position"] = 0
        df.loc[long_setup, "position"] = 1
        df.loc[short_setup, "position"] = -1

        # Each qualifying setup produces a fresh signal.
        df["signal"] = df["position"]

        # Reference exit levels for backtesters that consume them.
        df["long_stop"] = df["close"] * (1 - self.stop_loss_pct)
        df["long_target"] = df["close"] * (1 + self.target_pct)

        df["short_stop"] = df["close"] * (1 + self.stop_loss_pct)
        df["short_target"] = df["close"] * (1 - self.target_pct)

        warmup = max(
            self.ema_slow,
            self.rsi_period,
            self.volume_period,
        )

        df = df.iloc[warmup:].copy()
        df = self._restore_datetime_column(df)

        return df