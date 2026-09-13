"""Corporate-action adjustment: splits, bonuses, dividends."""
import pandas as pd

def apply_splits(df, splits):
    """splits: list of (ex_date_str, ratio) where ratio = new/old (e.g. 5 for 5:1)."""
    df = df.copy()
    for ex, ratio in sorted(splits, key=lambda x: x[0]):
        m = df["datetime"] < pd.to_datetime(ex)
        for c in ["open", "high", "low", "close"]:
            df.loc[m, c] = df.loc[m, c] / ratio
        df.loc[m, "volume"] = df.loc[m, "volume"] * ratio
    return df

def apply_bonus(df, bonuses):
    """bonuses: list of (ex_date_str, n_new, n_old); e.g. 1:1 -> (date,1,1)."""
    df = df.copy()
    for ex, n_new, n_old in sorted(bonuses, key=lambda x: x[0]):
        ratio = (n_old + n_new) / n_old
        m = df["datetime"] < pd.to_datetime(ex)
        for c in ["open", "high", "low", "close"]:
            df.loc[m, c] = df.loc[m, c] / ratio
        df.loc[m, "volume"] = df.loc[m, "volume"] * ratio
    return df

def apply_dividends(df, dividends, reinvest=False):
    """dividends: list of (ex_date_str, cash_amount). Price-only series
    cannot be perfectly adjusted; we subtract cash on ex-date unless
    reinvest=True in which case we leave prices (document choice)."""
    if reinvest:
        return df.copy()
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    for ex, cash in dividends:
        i = df.index[df["datetime"] >= pd.to_datetime(ex)]
        if len(i):
            j = i[0]
            for c in ["open", "high", "low", "close"]:
                df.loc[j, c] = max(0.01, df.loc[j, c] - cash)
    return df

def adjust(df, splits=None, bonuses=None, dividends=None, **kw):
    out = df.copy()
    if splits:
        out = apply_splits(out, splits)
    if bonuses:
        out = apply_bonus(out, bonuses)
    if dividends:
        out = apply_dividends(out, dividends, reinvest=kw.get("reinvest_dividends", False))
    out.attrs["adjusted"] = True
    return out
