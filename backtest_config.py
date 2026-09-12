"""
backtest_config.py
-----------------
Single file to edit all your backtest variables.

HOW TO USE:
1. Edit the variables below to your liking.
2. Run: python3 main.py
   (When no CLI args are given, this file is used automatically.)

You can still override any variable via CLI:
    python3 main.py --symbol TATASTEEL --timeframe FIVE_MINUTE
"""

# ============================================================
# PROJECT IDENTITY
# ============================================================
PROJECT_NAME = "QuantEdge"

# ============================================================
# SYMBOL & EXCHANGE
# ============================================================
# Trading symbol (as listed on the exchange)
SYMBOL = "SUZLON"  # QuantEdge target - "RELIANCE", "NIFTY23SEP17500CE"

# Exchange: "NSE" (equity), "NFO" (F&O), "MCX" (commodities), "BSE"
EXCHANGE = "NSE"

# ============================================================
# TIMEFRAME
# ============================================================
# Available timeframes (Angel One SmartAPI format):
#   "ONE_MINUTE"       - 1-minute candles
#   "FIVE_MINUTE"      - 5-minute candles
#   "FIFTEEN_MINUTE"   - 15-minute candles
#   "THIRTY_MINUTE"    - 30-minute candles
#   "ONE_HOUR"         - 1-hour candles
#   "ONE_DAY"          - Daily candles (default)
#   "ONE_WEEK"         - Weekly candles
#   "ONE_MONTH"        - Monthly candles
TIMEFRAME = "FIVE_MINUTE"

# ============================================================
# DATE RANGE
# ============================================================
# Format: "YYYY-MM-DD"
# Set to None to use defaults (last ~25 days for intraday, ~1 year for daily)
START_DATE = "2026-07-01"  # e.g., "2024-01-01"
END_DATE = "2026-09-01"    # e.g., "2024-12-31"

# ============================================================
# STRATEGY
# ============================================================
# Available strategies:
#   "sma_crossover"       - Simple Moving Average Crossover
#   "rsi_mean_reversion"  - RSI Mean Reversion
#   "macd_crossover"      - MACD Crossover
#   "opening_candle"      - Opening Candle Direction (30pt target, 30pt SL, 1:1 RRR)
#   "suzlon"              - Suzlon Intraday Confluence (15m trend + 5m pullback/rejection)
#   "hero_orb"            - HERO ORB Breakout (opening range + VWAP filter, ATR target/stop)
# Run multiple: ["sma_crossover", "rsi_mean_reversion"]
STRATEGIES = ["opening_candle"]

# ============================================================
# CAPITAL & POSITION SIZING
# ============================================================
# Initial capital in INR
INITIAL_CAPITAL = 10000.0

# Position sizing method:
#   "fixed_quantity"     - Trade fixed number of shares (see FIXED_QUANTITY below)
#   "fixed_capital_pct"  - Use fixed % of capital per trade (see CAPITAL_PCT below)
#   "risk_based"         - Size based on risk per trade (see RISK_PCT below)
POSITION_SIZING = "fixed_capital_pct"

# Used when POSITION_SIZING = "fixed_quantity"
FIXED_QUANTITY = 10

# Used when POSITION_SIZING = "fixed_capital_pct" (e.g., 0.10 = 10% of capital)
CAPITAL_PCT = 1

# Used when POSITION_SIZING = "risk_based" (e.g., 0.02 = 2% risk per trade)
RISK_PCT = 0.02

# Risk amount per unit for stop-loss calculation (used in risk_based sizing)
RISK_PER_UNIT = 10.0

# ============================================================
# SEGMENT (affects brokerage calculation)
# ============================================================
#   "intraday_equity"  - Intraday equity (0.03% brokerage, capped at ₹20)
#   "delivery_equity"  - Delivery equity (free brokerage)
#   "futures"          - F&O Futures (0.03% brokerage, capped at ₹20)
#   "options"          - F&O Options (flat ₹20 per order)
SEGMENT = "intraday_equity"

# ============================================================
# OUTPUT OPTIONS
# ============================================================
# Set to False to disable plotting
ENABLE_PLOT = True

# Set to False to disable CSV export of trade logs
ENABLE_CSV = True

# ============================================================
# DATA CACHING
# ============================================================
# Set to False to always fetch fresh data from API
USE_CACHE = True

# How long cached data remains valid (hours)
CACHE_EXPIRY_HOURS = 24
