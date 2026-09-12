"""
hero_orb.py
-----------
HERO ORB Breakout - Opening Range Breakout with cooldown.

Derived from a quantitative pattern study of a randomly picked NIFTY 50
stock (HEROMOTOCO, 15-minute bars, Apr-Jun 2026, real Angel One data):

  * 85% of days break the first-30-minute opening range.
  * Downside ORB breaks followed through to the close 70% of the time
    (avg +30.4 pts); upside breaks faded (avg -7.4 pts) in that window.
  * Avg winner +50.6 pts vs avg loser -31.1 pts holding to the close;
    ATR(14) was ~13 pts. The window was a BEAR regime (stock fell ~3.6%).

Strategy rules (always-in-market rotation; the engine reverses on an
opposite signal, so every exit flips into the new position):

  1. Opening range = high/low of the first `orb_bars` bars of each day.
  2. SHORT: close below the current day's ORB low (the strong edge).
  3. LONG : close above the current day's ORB high, gated by the trend
            EMA so the system stays short-biased in down-trends.
  4. Flip to the opposite side only when price breaks the current day's
            ORB range in the opposite direction (pattern-aligned).
  5. min_hold_bars cooldown after entry prevents whipsaw flips.
  6. Disaster stop: adverse move beyond stop_atr * ATR (rare).
  7. No entries after `max_entry_time`; positions held overnight.

All indicator columns except 'signal' are dropped before returning.
"""

import os
import sys

if __package__ in (None, ''):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import pandas as pd
from strategies.base import Strategy


class HeroOrbStrategy(Strategy):
    """Opening Range Breakout with trend gate and cooldown."""

    def __init__(
        self,
        orb_bars: int = 4,
        trend_ema: int = 30,
        atr_period: int = 14,
        stop_atr: float = 7.0,
        min_hold_bars: int = 8,
        max_entry_time: str = '14:30',
        **kwargs,
    ):
        super().__init__(
            name='HERO ORB Breakout',
            orb_bars=orb_bars,
            trend_ema=trend_ema,
            atr_period=atr_period,
            stop_atr=stop_atr,
            min_hold_bars=min_hold_bars,
            max_entry_time=max_entry_time,
            **kwargs,
        )
        self.orb_bars = int(orb_bars)
        self.trend_ema = int(trend_ema)
        self.atr_period = int(atr_period)
        self.stop_atr = float(stop_atr)
        self.min_hold_bars = int(min_hold_bars)
        self.min_hold_bars = int(min_hold_bars)
        self.max_entry_time = str(max_entry_time)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.strftime('%H:%M')
        df['bar_of_day'] = df.groupby('date').cumcount()

        # ---- opening range per day: high/low of the first orb_bars bars ----
        forming = df['bar_of_day'] < self.orb_bars
        orb_hi_map = df[forming].groupby('date')['high'].max()
        orb_lo_map = df[forming].groupby('date')['low'].min()
        df['orb_hi'] = df['date'].map(orb_hi_map)
        df['orb_lo'] = df['date'].map(orb_lo_map)

        # ---- trend EMA (long side gate) ----
        df['ema'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()

        # ---- ATR on true range ----
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period, min_periods=1).mean()
        df['atr'] = df['atr'].fillna(df['high'] - df['low'])

        signals = []
        in_position = False
        direction = 0
        entry_price = 0.0
        bars_held = 0
        current_date = None

        for _, row in df.iterrows():
            signal = 0
            if row['date'] != current_date:
                current_date = row['date']
                bars_held = 0

            orb_ready = not pd.isna(row['orb_hi']) and not pd.isna(row['orb_lo'])
            range_live = orb_ready and row['bar_of_day'] >= self.orb_bars
            stop_dist = self.stop_atr * row['atr']
            time_ok = row['time'] <= self.max_entry_time
            trend_up = row['close'] > row['ema']

            if in_position:
                bars_held = bars_held + 1
                # ---- disaster stop (rare, caps adverse runs) ----
                if direction == 1 and row['low'] <= entry_price - stop_dist:
                    signal = -1
                    direction = -1
                    entry_price = row['close']
                    bars_held = 0
                    signals.append(signal)
                    continue
                if direction == -1 and row['high'] >= entry_price + stop_dist:
                    signal = 1
                    direction = 1
                    entry_price = row['close']
                    bars_held = 0
                    signals.append(signal)
                    continue
                # ---- pattern-aligned flip after cooldown ----
                if range_live and time_ok and bars_held >= self.min_hold_bars:
                    if direction == 1 and row['close'] < row['orb_lo']:
                        signal = -1
                        direction = -1
                        entry_price = row['close']
                        bars_held = 0
                        signals.append(signal)
                        continue
                    if direction == -1 and row['close'] > row['orb_hi'] and trend_up:
                        signal = 1
                        direction = 1
                        entry_price = row['close']
                        bars_held = 0
                        signals.append(signal)
                        continue
                signals.append(0)
                continue

            # ---- flat: initial entry ----
            if not range_live:
                signals.append(0)
                continue
            if not time_ok:
                signals.append(0)
                continue
            if pd.isna(row['atr']) or row['atr'] <= 0:
                signals.append(0)
                continue

            if row['close'] < row['orb_lo']:
                signal = -1
                direction = -1
                entry_price = row['close']
                bars_held = 0
                in_position = True
            elif row['close'] > row['orb_hi'] and trend_up:
                signal = 1
                direction = 1
                entry_price = row['close']
                bars_held = 0
                in_position = True
            signals.append(signal)

        df['signal'] = signals
        df.drop(columns=[
            'date', 'time', 'bar_of_day', 'orb_hi', 'orb_lo',
            'ema', 'atr',
        ], inplace=True)
        return df