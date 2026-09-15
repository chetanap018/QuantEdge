from strategies.base import Strategy
import pandas as pd
import numpy as np


class SuzlonNewStrategy(Strategy):
    def __init__(self, fast=9, slow=20):
        super().__init__(name="Suzlon_new", fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()

        df["signal"] = 0
        df.loc[ema_fast > ema_slow, "signal"] = 1
        df.loc[ema_fast < ema_slow, "signal"] = -1

        return df
