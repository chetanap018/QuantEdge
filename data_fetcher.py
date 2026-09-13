"""
data_fetcher.py
---------------
Handles authentication and historical data retrieval via Angel One SmartAPI.
Includes caching to avoid repeated API calls for the same symbol/timeframe.

=== HOW TO SET UP ANGEL ONE API CREDENTIALS ===
1. Install the SmartAPI: pip install smartapi-python
2. Get your API key from https://smartapi.angelone.in/
3. Set credentials in config.py or via environment variables.
4. On first login, you'll receive an OTP on your registered mobile.
   The TOTP secret (ANGEL_TOTP_KEY) can be found in your Angel One profile.

Note: If smartapi-python is not installed, the fetcher falls back to
      generating synthetic data for testing purposes.
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches historical market data from Angel One SmartAPI with caching.
    Falls back to synthetic data if API is unavailable.
    """

    def __init__(self):
        self.connected = False
        self.smart_api = None
        self.auth_token = None
        self.refresh_token = None
        self._connect()

    def _connect(self):
        """Attempt to connect to Angel One SmartAPI."""
        try:
            from SmartApi.smartConnect import SmartConnect

            self.smart_api = SmartConnect(api_key=config.ANGEL_API_KEY)

            data = self.smart_api.generateSession(
                config.ANGEL_CLIENT_ID,
                config.ANGEL_CLIENT_PIN,
                self._get_totp(),
            )

            if data.get("status"):
                self.auth_token = data["data"]["jwtToken"]
                self.refresh_token = data["data"]["refreshToken"]
                self.connected = True
                logger.info("Successfully connected to Angel One SmartAPI.")
            else:
                logger.warning(
                    f"Angel One login failed: {data.get('message', 'Unknown error')}. "
                    "Falling back to synthetic data."
                )
                self.connected = False

        except ImportError:
            logger.warning(
                "smartapi-python not installed. "
                "Install with: pip install smartapi-python. "
                "Falling back to synthetic data."
            )
            self.connected = False
        except Exception as e:
            logger.warning(
                f"Angel One connection error: {e}. Falling back to synthetic data."
            )
            self.connected = False

    def _get_totp(self) -> str:
        """Generate TOTP from the secret key."""
        try:
            import pyotp
            totp = pyotp.TOTP(config.ANGEL_TOTP_KEY)
            return totp.now()
        except ImportError:
            logger.warning("pyotp not installed. Using placeholder TOTP.")
            return "000000"

    def _get_cache_key(self, symbol: str, timeframe: str, start: str, end: str) -> str:
        """Generate a unique cache key for the query."""
        raw = f"{symbol}_{timeframe}_{start}_{end}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if it exists and is not expired."""
        if not config.USE_CACHE:
            return None

        cache_path = os.path.join(config.CACHE_DIR, f"{cache_key}.csv")
        meta_path = os.path.join(config.CACHE_DIR, f"{cache_key}.meta.json")

        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        cached_time = datetime.fromisoformat(meta["timestamp"])
        if datetime.now() - cached_time > timedelta(hours=config.CACHE_EXPIRY_HOURS):
            logger.info("Cache expired, fetching fresh data.")
            return None

        if meta.get("source") == "synthetic":
            logger.warning("Cached data is synthetic, not live. Clear cache to retry live fetch.")

        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        return df

    def _cache_data(self, cache_key: str, df: pd.DataFrame, source: str = "live"):
        """Save data to cache."""
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(config.CACHE_DIR, f"{cache_key}.csv")
        meta_path = os.path.join(config.CACHE_DIR, f"{cache_key}.meta.json")

        df.to_csv(cache_path, index=False)
        with open(meta_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "source": source}, f)

        logger.info(f"Data cached to {cache_path} (source: {source})")

    def fetch_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        timeframe: str = "ONE_DAY",
        start_date: str = None,
        end_date: str = None,
        token: str = None,
        source: str = None,
        fallback: bool = True,
        allow_synthetic: bool = None,
        validate_data: bool = True,
        adjust_actions: dict = None,
        universe=None,
        on_quality: str = "warn",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "RELIANCE", "INFY").
        exchange : str
            Exchange code: "NSE", "BSE", "NFO", "MCX".
        timeframe : str
            Candle interval: "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
            "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY".
        start_date : str
            Start date in "YYYY-MM-DD" format.
        end_date : str
            End date in "YYYY-MM-DD" format.
        token : str
            Angel One instrument token (if None, will look up).
        source : str
            Preferred source: "angelone" | "yahoo" | "nse" | "csv" | "synthetic".
            None = use config.DATA_SOURCES order.
        fallback : bool
            If True, try remaining sources when the preferred one is empty.
        allow_synthetic : bool, optional
            If True, permit simulated GBM data (SyntheticAdapter) when every
            real source fails.  If False (default from config.ALLOW_SYNTHETIC_DATA),
            a hard error is raised instead so you can NEVER accidentally run a
            backtest on fake data without explicitly opting in.
        validate_data : bool
            Run data-quality validation (gaps / stale / outliers / OHLC).
        adjust_actions : dict
            Corporate actions, e.g. {"splits": [("2023-07-01", 5)],
            "bonuses": [...], "dividends": [("2023-08-16", 9.0)]}.
        universe :
            data_pit.UniverseHistory for point-in-time listing clipping.
        on_quality : str
            "warn" (log) | "raise" (error) | "ignore" when issues found.

        Returns
        -------
        pd.DataFrame
            Columns: datetime, open, high, low, close, volume
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if allow_synthetic is None:
            allow_synthetic = bool(getattr(config, "ALLOW_SYNTHETIC_DATA", False))

        # Check cache first
        cache_key = self._get_cache_key(
            f"{exchange}:{symbol}", timeframe, start_date, end_date
        )
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            cached_source = getattr(cached, "attrs", {}).get("data_source")
            if cached_source == "synthetic" and not allow_synthetic:
                raise RuntimeError(
                    f"REFUSING synthetic cached data for {symbol}: cached bars came "
                    "from the synthetic (simulated GBM) source but synthetic data is "
                    "not allowed. Enable it explicitly (config.ALLOW_SYNTHETIC_DATA "
                    "= True or allow_synthetic=True) or clear the cache and retry a "
                    "real source."
                )
            if validate_data or adjust_actions or universe is not None:
                return self._post_process(
                    cached, symbol, timeframe, validate_data=validate_data,
                    adjust_actions=adjust_actions, universe=universe,
                    on_quality=on_quality,
                )
            return cached

        # Multi-source fetch with fallback (Fix 2: no single-broker dependency).
        df = self._fetch_multi_source(
            symbol, exchange, timeframe, start_date, end_date, token,
            source=source, fallback=fallback, allow_synthetic=allow_synthetic,
        )
        source_used = df.attrs.get("data_source", source or "angelone")
        self._cache_data(cache_key, df, source=source_used)
        return self._post_process(
            df, symbol, timeframe, validate_data=validate_data,
            adjust_actions=adjust_actions, universe=universe,
            on_quality=on_quality,
        )

    def _fetch_multi_source(self, symbol, exchange, timeframe, start_date,
                            end_date, token, source=None, fallback=True,
                            allow_synthetic=None):
        """Try `source` first, then fall back through other adapters."""
        if allow_synthetic is None:
            allow_synthetic = bool(getattr(config, "ALLOW_SYNTHETIC_DATA", False))
        order = list(getattr(config, "DATA_SOURCES", None) or ["angelone", "yahoo", "nse", "synthetic"])
        if source:
            order = [source] + [s for s in order if s != source]
        if not fallback:
            order = order[:1]
        last = None
        for name in order:
            if name == "synthetic" and not allow_synthetic:
                raise RuntimeError(
                    f"No data available for {symbol} ({timeframe} {start_date} to "
                    f"{end_date}): all real sources failed and synthetic data is not "
                    f"allowed. Possible fixes: (1) intraday Yahoo data only covers "
                    f"the last 60 days — try a narrower date range or ONE_DAY "
                    f"timeframe; (2) check Angel One credentials in .env; "
                    f"(3) set ALLOW_SYNTHETIC_DATA=true in .env to allow simulated "
                    f"data for testing."
                )
            try:
                df = self._fetch_from_source(name, symbol, exchange, timeframe,
                                             start_date, end_date, token)
                if df is not None and not df.empty:
                    df.attrs["data_source"] = name
                    if name == "synthetic":
                        logger.warning(
                            "USING SYNTHETIC DATA for %s — these are simulated "
                            "GBM prices, NOT real market results.",
                            symbol,
                        )
                    elif name != order[0]:
                        logger.info("Data for %s served by fallback %s.", symbol, name)
                    return df
                last = df
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning("Source %s failed for %s: %s", name, symbol, e)
        if last is not None:
            return last
        from data_adapters import CANONICAL as _C
        return pd.DataFrame(columns=_C)

    def _fetch_from_source(self, name, symbol, exchange, timeframe,
                           start_date, end_date, token):
        if name in ("angel", "angelone", "angel_one", "smartapi"):
            if self.connected and self.smart_api is not None:
                fetched = self._fetch_from_api(symbol, exchange, timeframe, start_date, end_date, token)
                if fetched is not None and not fetched.empty:
                    return fetched
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        from data_adapters import get_adapter
        adapter = get_adapter(name, smart_api=self.smart_api if self.connected else None,
                              token=token, token_lookup=self._get_instrument_token)
        kw = {}
        if name == "csv":
            kw["csv_dir"] = getattr(config, "DATA_CSV_DIR", "data")
        return adapter.fetch(symbol, exchange, timeframe, start_date, end_date, **kw)

    def _post_process(self, df, symbol, timeframe, validate_data=True,
                      adjust_actions=None, universe=None, on_quality="warn"):
        """Corporate-action adjust -> PIT clip -> quality validation."""
        if df is None or df.empty:
            return df
        if adjust_actions:
            try:
                from data_adjust import adjust as _adjust
                df = _adjust(df, splits=adjust_actions.get("splits"),
                             bonuses=adjust_actions.get("bonuses"),
                             dividends=adjust_actions.get("dividends"),
                             reinvest_dividends=adjust_actions.get("reinvest_dividends", False))
            except Exception as e:
                logger.warning("Adjustment failed for %s: %s", symbol, e)
        if universe is not None:
            try:
                from data_pit import clip_to_listing as _clip
                df = _clip(df, symbol, universe)
            except Exception as e:
                logger.warning("PIT clip failed for %s: %s", symbol, e)
        if validate_data:
            try:
                from data_quality import validate as _validate
                report = _validate(df, timeframe=timeframe)
                df.attrs["quality"] = report
                if not report.get("ok"):
                    msg = ("Data-quality issues for %s: gaps=%d stale=%d outliers=%d ohlc=%d"
                           % (symbol, len(report.get("gaps", [])), len(report.get("stale_runs", [])),
                              len(report.get("outliers", [])), report.get("ohlc_violations", 0)))
                    if on_quality == "raise":
                        raise ValueError(msg)
                    elif on_quality == "warn":
                        logger.warning(msg)
            except ValueError:
                raise
            except Exception as e:
                logger.warning("Validation failed for %s: %s", symbol, e)
        return df

    def _fetch_from_api(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        token: str = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Angel One SmartAPI with rate limit handling."""
        if token is None:
            token = self._get_instrument_token(symbol, exchange)
            if token is None:
                logger.error(f"Could not find instrument token for {exchange}:{symbol}")
                return None

        # For intraday timeframes, chunk the date range (API limits ~30 days per request)
        if timeframe != "ONE_DAY":
            return self._fetch_chunked(symbol, exchange, timeframe, start_date, end_date, token)

        historic_param = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": timeframe,
            "fromdate": start_date + " 09:15",
            "todate": end_date + " 15:30",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.smart_api.getCandleData(historic_param)

                if response.get("status"):
                    candles = response["data"]
                    df = pd.DataFrame(
                        candles,
                        columns=["datetime", "open", "high", "low", "close", "volume"],
                    )
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.dropna(inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    return df
                else:
                    error_msg = response.get("message", "")
                    if "rate limit" in error_msg.lower() or "too many" in error_msg.lower():
                        wait_time = (attempt + 1) * 2
                        logger.warning(
                            f"Rate limit hit. Waiting {wait_time}s before retry..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API error: {error_msg}")
                        return None

            except Exception as e:
                if "rate limit" in str(e).lower():
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limit error. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed: {e}")
                    return None

        logger.error("Max retries exceeded.")
        return None

    def _fetch_chunked(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        token: str = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday data in chunks to work around API limits."""
        if token is None:
            token = self._get_instrument_token(symbol, exchange)
            if token is None:
                logger.error(f"Could not find instrument token for {exchange}:{symbol}")
                return None

        chunk_days = 25  # Angel One limit for intraday data
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_chunks = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            logger.info(
                f"Fetching chunk: {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
            )
            
            historic_param = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": timeframe,
                "fromdate": current.strftime("%Y-%m-%d") + " 09:15",
                "todate": chunk_end.strftime("%Y-%m-%d") + " 15:30",
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.smart_api.getCandleData(historic_param)
                    if response.get("status"):
                        candles = response["data"]
                        if candles:
                            all_chunks.extend(candles)
                        break
                    else:
                        error_msg = response.get("message", "")
                        if "rate limit" in error_msg.lower():
                            wait_time = (attempt + 1) * 2
                            logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"API error: {error_msg}")
                            break
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"Rate limit error. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Request failed: {e}")
                        break
            
            current = chunk_end + timedelta(days=1)
        
        if not all_chunks:
            return None
            
        df = pd.DataFrame(
            all_chunks,
            columns=["datetime", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing when API is unavailable.
        Uses geometric Brownian motion for realistic price simulation.
        """
        np.random.seed(hash(symbol) % (2**31))

        freq_map = {
            "ONE_MINUTE": "min",
            "FIVE_MINUTE": "5min",
            "FIFTEEN_MINUTE": "15min",
            "THIRTY_MINUTE": "30min",
            "ONE_HOUR": "h",
            "ONE_DAY": "D",
        }
        freq = freq_map.get(timeframe, "D")

        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        if freq != "D":
            dates = dates[(dates.hour >= 9) & (dates.hour <= 15)]
            dates = dates[~((dates.hour == 9) & (dates.minute < 15))]
            dates = dates[~((dates.hour == 15) & (dates.minute > 30))]

        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        # Geometric Brownian Motion
        initial_price = 100 + np.random.uniform(50, 500)
        mu = 0.0002
        sigma = 0.02

        returns = np.random.normal(mu, sigma, n)
        price_series = initial_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame()
        df["datetime"] = dates
        df["close"] = price_series
        df["open"] = df["close"].shift(1).fillna(initial_price)
        noise = np.random.uniform(0.001, 0.01, n)
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + noise)
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - noise)
        df["volume"] = np.random.randint(10000, 1000000, n)

        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df["open"] = df["open"].round(2)
        df["high"] = df["high"].round(2)
        df["low"] = df["low"].round(2)
        df["close"] = df["close"].round(2)
        df["volume"] = df["volume"].astype(int)

        return df

    def logout(self):
        """Logout from Angel One session."""
        if self.connected and self.smart_api:
            try:
                self.smart_api.terminateSession(config.ANGEL_CLIENT_ID)
                logger.info("Logged out from Angel One.")
            except Exception as e:
                logger.error(f"Logout error: {e}")

    def _get_instrument_token(self, symbol: str, exchange: str, token: str = None) -> Optional[str]:
        """
        Look up the instrument token for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "RELIANCE24APRFUT")
            exchange: Exchange segment (e.g., "NSE", "NFO")
            token: Optional explicit token to use instead of looking up
        """
        # If explicit token provided, use it directly
        if token is not None:
            return token
            
        try:
            result = self.smart_api.searchScrip(exchange, symbol)
            if result.get("status"):
                for item in result["data"]:
                    # Match equity symbols by -EQ suffix
                    if item["tradingsymbol"].endswith("-EQ"):
                        return str(item["symboltoken"])
                # For non-equity (F&O), return first match if no -EQ found
                if result["data"]:
                    logger.warning(
                        f"No -EQ match found for {symbol}, using first result: "
                        f"{result['data'][0]['tradingsymbol']}"
                    )
                    return str(result["data"][0]["symboltoken"])
        except Exception as e:
            logger.error(f"Token lookup failed: {e}")
        return None
