"""Point-in-time integrity: as-of universe, delisting guard, availability check."""
import pandas as pd

class UniverseHistory:
    """Tracks index constituents / listing status as-of any date.

    members: dict symbol -> list of (start_str, end_str|None).
    Use for survivorship-bias-free backtests: only trade symbols that were
    actually listed/tradable at bar time.
    """
    def __init__(self, members=None, delisted=None):
        self.members = members or {}
        self.delisted = delisted or {}

    def is_tradable(self, symbol, when):
        when = pd.to_datetime(when)
        for s, e in self.members.get(symbol, [(None, None)]):
            s = pd.to_datetime(s) if s else pd.Timestamp.min
            e = pd.to_datetime(e) if e else pd.Timestamp.max
            if s <= when <= e:
                if symbol in self.delisted and when >= pd.to_datetime(self.delisted[symbol]):
                    return False
                return True
        return False

    def tradable_at(self, when, symbols):
        return [s for s in symbols if self.is_tradable(s, when)]

def assert_point_in_time(df, asof_col=None, when=None):
    """Guard: frame must not contain rows newer than the decision time."""
    if df is None or df.empty or when is None:
        return True
    when = pd.to_datetime(when)
    col = asof_col or "datetime"
    future = df[df[col] > when]
    if not future.empty:
        raise ValueError("Point-in-time violation: %d rows newer than %s" % (len(future), when))
    return True

def clip_to_listing(df, symbol, universe=None):
    """Drop bars outside the symbol's known listing window."""
    if universe is None or symbol not in universe.members:
        return df
    windows = universe.members[symbol]
    out = []
    for s, e in windows:
        m = pd.Series(True, index=df.index)
        if s:
            m &= df["datetime"] >= pd.to_datetime(s)
        if e:
            m &= df["datetime"] <= pd.to_datetime(e)
        out.append(df[m])
    if not out:
        return df.iloc[0:0]
    return pd.concat(out).sort_values("datetime").reset_index(drop=True)
