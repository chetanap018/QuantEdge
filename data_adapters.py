"""Adapters: angelone/yahoo/nse/csv/synthetic. Canonical: datetime,open,high,low,close,volume."""
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)
CANONICAL = ["datetime", "open", "high", "low", "close", "volume"]

def canonicalize(df):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=CANONICAL)
    alt = {"date": "datetime", "timestamp": "datetime", "o": "open", "h": "high",
           "l": "low", "c": "close", "v": "volume", "vol": "volume"}
    df = df.rename(columns={k: v for k, v in alt.items() if k in df.columns})
    for c in CANONICAL:
        if c not in df.columns:
            df[c] = np.nan
    df = df[CANONICAL].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out

class BaseAdapter:
    name = "base"
    def fetch(self, symbol, exchange="NSE", timeframe="ONE_DAY",
              start_date=None, end_date=None, **kw):
        raise NotImplementedError
    def is_available(self):
        return True
class YahooAdapter(BaseAdapter):
    name = "yahoo"
    def fetch(self, symbol, exchange="NSE", timeframe="ONE_DAY",
              start_date=None, end_date=None, **kw):
        suf = ".NS" if exchange.upper() in ("NSE", "NFO") else ".BO"
        ysym = kw.get("yahoo_symbol") or (symbol.upper() + suf)
        iv = {"ONE_MINUTE": "1m", "FIVE_MINUTE": "5m", "FIFTEEN_MINUTE": "15m",
              "THIRTY_MINUTE": "30m", "ONE_HOUR": "60m"}.get(timeframe, "1d")
        try:
            import yfinance as yf
            df = yf.Ticker(ysym).history(start=start_date, end=end_date,
                                         interval=iv, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={"Date": "datetime",
                    "Datetime": "datetime", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"})
                return canonicalize(df)
        except ImportError:
            logger.info("yfinance missing; stooq fallback.")
        except Exception as e:
            logger.warning("yahoo fail %s", e)
        try:
            sym = ysym.lower().replace(".ns", ".in")
            d1 = start_date.replace("-", "") if start_date else ""
            d2 = end_date.replace("-", "") if end_date else ""
            sdf = pd.read_csv("https://stooq.com/q/d/l/?s=" + sym + "&d1=" + d1 + "&d2=" + d2 + "&i=d")
            sdf = sdf.rename(columns={"Date": "datetime", "Open": "open", "High": "high",
                                      "Low": "low", "Close": "close", "Volume": "volume"})
            return canonicalize(sdf)
        except Exception as e:
            logger.warning("stooq fail %s", e)
            return pd.DataFrame(columns=CANONICAL)


class SyntheticAdapter(BaseAdapter):
    name = "synthetic"
    def fetch(self, symbol, exchange="NSE", timeframe="ONE_DAY",
              start_date=None, end_date=None, **kw):
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else end - timedelta(days=365)
        freq = {"ONE_MINUTE": "min", "FIVE_MINUTE": "5min", "FIFTEEN_MINUTE": "15min",
                "THIRTY_MINUTE": "30min", "ONE_HOUR": "h", "ONE_DAY": "D"}.get(timeframe, "D")
        dates = pd.date_range(start=start, end=end, freq=freq)
        if freq != "D":
            dates = dates[(dates.hour >= 9) & (dates.hour <= 15)]
        if len(dates) == 0:
            return pd.DataFrame(columns=CANONICAL)
        rng = np.random.default_rng(kw.get("seed", 42))
        px = 150 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(dates))))
        df = pd.DataFrame({"datetime": dates, "close": px})
        df["open"] = df["close"].shift(1).fillna(px[0])
        nz = rng.uniform(0.001, 0.01, len(dates))
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + nz)
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - nz)
        df["volume"] = rng.integers(10000, 1000000, len(dates))
        return canonicalize(df)

class CSVAdapter(BaseAdapter):
    name = "csv"
    def fetch(self, symbol, exchange="NSE", timeframe="ONE_DAY",
              start_date=None, end_date=None, **kw):
        path = kw.get("csv_path") or os.path.join(kw.get("csv_dir", "data"), symbol + "_" + timeframe + ".csv")
        if not os.path.exists(path):
            return pd.DataFrame(columns=CANONICAL)
        df = canonicalize(pd.read_csv(path))
        if start_date:
            df = df[df["datetime"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["datetime"] <= pd.to_datetime(end_date) + timedelta(days=1)]
        if end_date:
            df = df[df["datetime"] <= pd.to_datetime(end_date) + timedelta(days=1)]
        return df.reset_index(drop=True)

class AngelOneAdapter(BaseAdapter):
    name = "angelone"
    def __init__(self, smart_api=None):
        self.smart_api = smart_api
    def is_available(self):
        if self.smart_api is not None:
            return True
        try:
            from SmartApi.smartConnect import SmartConnect
            return True
        except ImportError:
            return False
    def fetch(self, symbol, exchange="NSE", timeframe="ONE_DAY",
              start_date=None, end_date=None, **kw):
        api = self.smart_api or kw.get("smart_api")
        token = kw.get("token")
        if api is None or token is None:
            look = kw.get("token_lookup")
            if look is not None and api is not None:
                token = look(symbol, exchange)
            if token is None:
                return pd.DataFrame(columns=CANONICAL)
        import time as _t
        days = 25 if timeframe != "ONE_DAY" else 365
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        chunks, cur = [], start
        while cur < end:
            ce = min(cur + timedelta(days=days), end)
            p = {"exchange": exchange, "symboltoken": token, "interval": timeframe,
                 "fromdate": cur.strftime("%Y-%m-%d") + " 09:15",
                 "todate": ce.strftime("%Y-%m-%d") + " 15:30"}
            for a in range(3):
                try:
                    r = api.getCandleData(p)
                    if r.get("status") and r.get("data"):
                        chunks.extend(r["data"])
                        break
                    if "rate limit" not in r.get("message", "").lower():
                        break
                    _t.sleep((a + 1) * 2)
                except Exception:
                    _t.sleep((a + 1) * 2)
                    break
            cur = ce + timedelta(days=1)
        if not chunks:
            return pd.DataFrame(columns=CANONICAL)
        df = pd.DataFrame(chunks, columns=["datetime", "open", "high", "low", "close", "volume"])
        return canonicalize(df)

class NSEAdapter(BaseAdapter):
    name = "nse"
    def fetch(self, symbol, exchange="NSE", timeframe="ONE_DAY",
              start_date=None, end_date=None, **kw):
        try:
            import requests
        except ImportError:
            return pd.DataFrame(columns=CANONICAL)
        try:
            sym = symbol.upper().replace("-EQ", "")
            h = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
            s = requests.Session()
            s.get("https://www.nseindia.com", headers=h, timeout=10)
            f = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            t = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            url = "https://www.nseindia.com/api/historical/cm/equity?symbol=" + sym
            url += "&series=[%22EQ%22]&from=" + f + "&to=" + t
            rows = s.get(url, headers=h, timeout=20).json().get("data", [])
            if not rows:
                return pd.DataFrame(columns=CANONICAL)
            df = pd.DataFrame(rows).rename(columns={"CH_TIMESTAMP": "datetime",
                "CH_OPENING_PRICE": "open", "CH_TRADE_HIGH_PRICE": "high",
                "CH_TRADE_LOW_PRICE": "low", "CH_CLOSING_PRICE": "close",
                "CH_TOT_TRADED_QTY": "volume"})
            return canonicalize(df)
        except Exception as e:
            logger.warning("nse fail %s", e)
            return pd.DataFrame(columns=CANONICAL)

REGISTRY = {}

def get_adapter(name, **kw):
    key = name.lower()
    short = {"angel": "angelone", "smartapi": "angelone", "yfinance": "yahoo"}.get(key, key)
    if short not in REGISTRY:
        if short == "angelone":
            REGISTRY[short] = AngelOneAdapter(smart_api=kw.get("smart_api"))
        elif short == "yahoo":
            REGISTRY[short] = YahooAdapter()
        elif short == "nse":
            REGISTRY[short] = NSEAdapter()
        elif short == "csv":
            REGISTRY[short] = CSVAdapter()
        else:
            REGISTRY[short] = SyntheticAdapter()
    return REGISTRY[short]

def fetch_with_fallback(symbol, exchange="NSE", timeframe="ONE_DAY",
                        start_date=None, end_date=None, sources=None, **kw):
    sources = sources or ["angelone", "yahoo", "nse", "synthetic"]
    for name in sources:
        try:
            ad = get_adapter(name, **kw)
            if not ad.is_available():
                continue
            df = ad.fetch(symbol, exchange, timeframe, start_date, end_date, **kw)
            if df is not None and not df.empty:
                df.attrs["data_source"] = name
                return df
        except Exception as e:
            logger.warning("source %s failed: %s", name, e)
    return pd.DataFrame(columns=CANONICAL)

