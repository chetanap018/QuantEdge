"""
config.py
---------
Configuration file for the backtesting framework.

=== HOW TO PLUG IN ANGEL ONE API CREDENTIALS ===
1. Sign up at https://smartapi.angelone.in/ and get your API keys.
2. Set your credentials below OR via environment variables (recommended for security):
     export ANGEL_API_KEY="your_api_key"
     export ANGEL_CLIENT_ID="your_client_id"
     export ANGEL_CLIENT_PIN="your_client_pin"
     export ANGEL_TOTP_KEY="your_totp_secret"
3. If environment variables are not set, the values below will be used as fallback.

For Zerodha Kite (alternative data source), set:
     export ZERODHA_API_KEY="your_zerodha_api_key"
     export ZERODHA_ACCESS_TOKEN="your_access_token"
"""

import os

# ============================================================
# Auto-load a local .env file into the environment (recommended).
# This is how Angel One credentials are "exported" for the project
# without editing config.py or typing exports every session.
# python-dotenv is used if installed, otherwise a tiny built-in parser
# is used. Environment variables already set in the shell take priority.
# ============================================================
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

if os.path.exists(_ENV_PATH):
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=_ENV_PATH, override=False)
    except ImportError:
        for _line in open(_ENV_PATH, encoding="utf-8"):
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _val = _line.split("=", 1)
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            os.environ.setdefault(_key, _val)

# ============================================================
# Angel One SmartAPI Credentials
# ============================================================
ANGEL_API_KEY = os.environ.get("ANGEL_API_KEY", "your_api_key_here")
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "your_client_id_here")
ANGEL_CLIENT_PIN = os.environ.get("ANGEL_CLIENT_PIN", "your_client_pin_here")
ANGEL_TOTP_KEY = os.environ.get("ANGEL_TOTP_KEY", "your_totp_secret_here")

# ============================================================
# Default Backtest Parameters
# ============================================================
DEFAULT_INITIAL_CAPITAL = 100000.0  # INR
DEFAULT_POSITION_SIZING = "fixed_quantity"  # Options: "fixed_quantity", "fixed_capital_pct", "risk_based"
DEFAULT_FIXED_QUANTITY = 1
DEFAULT_CAPITAL_PCT = 0.10  # 10% of capital per trade
DEFAULT_RISK_PCT = 0.02  # 2% risk per trade (for risk-based sizing)
DEFAULT_RISK_PER_UNIT = 10.0  # Risk amount per unit for stop-loss calculation

# ============================================================
# Data Fetching
# ============================================================
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
USE_CACHE = True  # Set to False to always fetch fresh data
CACHE_EXPIRY_HOURS = 24  # How long cached data remains valid

# ============================================================
# Broker Charges Configuration (Zerodha rates)
# ============================================================
# Note: Even though we use Angel One for data, we use Zerodha's
# brokerage structure for realistic cost simulation.
CHARGES_CONFIG = {
    "intraday_equity": {
        "brokerage_rate": 0.0003,  # 0.03% per order
        "brokerage_cap": 20.0,     # Max ₹20 per executed order
        "stt_buy": 0.0,            # No STT on buy for intraday
        "stt_sell": 0.00025,       # 0.025% on sell side
        "exchange_txn": 0.0000345, # NSE: 0.00345%
        "gst_rate": 0.18,          # 18% on brokerage + exchange + SEBI
        "sebi_rate": 0.000001,     # ₹10 per crore = 0.0001%
        "stamp_duty_buy": 0.00003, # 0.003% on buy
        "stamp_duty_sell": 0.0,    # No stamp duty on sell for intraday
    },
    "delivery_equity": {
        "brokerage_rate": 0.0,     # Free delivery
        "brokerage_cap": 0.0,
        "stt_buy": 0.001,          # 0.1% on buy
        "stt_sell": 0.001,         # 0.1% on sell
        "exchange_txn": 0.0000345, # NSE: 0.00345%
        "gst_rate": 0.18,
        "sebi_rate": 0.000001,
        "stamp_duty_buy": 0.00015, # 0.015% on buy
        "stamp_duty_sell": 0.0,
    },
    "futures": {
        "brokerage_rate": 0.0003,  # 0.03% per order
        "brokerage_cap": 20.0,     # Max ₹20 per executed order
        "stt_buy": 0.0,
        "stt_sell": 0.0001,        # 0.01% on sell side
        "exchange_txn": 0.0000173, # NSE: 0.00173%
        "gst_rate": 0.18,
        "sebi_rate": 0.000001,
        "stamp_duty_buy": 0.00002, # 0.002% on buy
        "stamp_duty_sell": 0.0,
    },
    "options": {
        "brokerage_rate": 0.0,     # Flat ₹20 per order (handled separately)
        "brokerage_cap": 20.0,     # Flat ₹20 per executed order
        "stt_buy": 0.0,
        "stt_sell": 0.0005,        # 0.05% on sell (on premium)
        "exchange_txn": 0.0005,    # 0.05% on premium
        "gst_rate": 0.18,
        "sebi_rate": 0.000001,
        "stamp_duty_buy": 0.00003, # 0.003% on buy
        "stamp_duty_sell": 0.0,
    },
}

# ============================================================
# Logging
# ============================================================
LOG_LEVEL = "INFO"
VERBOSE = True
