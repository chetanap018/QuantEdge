# Backtesting Framework

A modular Python backtesting framework for trading strategies using Angel One SmartAPI for data and Zerodha's brokerage structure for realistic cost simulation — now with a built-in web dashboard.

## Backtest Studio (Web UI)

No more editing `backtest_config.py` for every run — configure everything in the browser:

```bash
python3 app.py
# open http://127.0.0.1:5000
```

The UI reads its **defaults from `backtest_config.py`** (it stays the single source of truth), but every parameter — symbol, exchange, timeframe, dates, capital, sizing, segment and all strategy parameters — can be changed live in the sidebar. `backtest_config.py` and the CLI (`python main.py`) keep working exactly as before.

Features:

- **Config sidebar** — all backtest variables with per-strategy parameter forms (schema-driven)
- **KPI cards** — Net Profit, Total Return, CAGR, Sharpe, Max Drawdown, Win Rate, Profit Factor, Trades, Charges, Final Capital
- **Overview tab** — multi-strategy equity curves (zoomable), drawdown area chart with max-DD marker, monthly returns heatmap, long/short P&L split, trade-quality chips (best/worst trade, streaks, avg holding time)
- **Price & Signals tab** — candlestick + volume chart with long/short entry and exit markers per strategy
- **Trades tab** — per-trade P&L bars, filterable trade log table, **CSV / JSON export**
- **Charges tab** — brokerage, STT, exchange, GST, SEBI, stamp duty stacked per strategy + donut split
- **Compare tab** — net profit bars and full metrics table across strategies

New files: `app.py` (Flask API), `templates/index.html` (dashboard), `static/echarts.min.js` (charting, local copy). Requires `pip install flask`.

## Project Structure

```
backtesting-framework/
├── config.py              # API keys, default params, charges config
├── broker_charges.py      # Zerodha brokerage & statutory charges calculator
├── data_fetcher.py        # Angel One SmartAPI data with caching
├── backtest_engine.py     # Core simulation engine
├── main.py                # Entry point (CLI)
├── requirements.txt       # Python dependencies
├── strategies/
│   ├── __init__.py        # Strategy package init
│   ├── base.py            # Abstract Strategy base class
│   ├── sma_crossover.py   # SMA Crossover strategy (example)
│   └── rsi_mean_reversion.py  # RSI Mean Reversion strategy (example)
├── cache/                 # Cached market data (auto-created)
└── output/                # Trade logs & plots (auto-created)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials
Edit `config.py` or set environment variables:
```bash
export ANGEL_API_KEY="your_api_key"
export ANGEL_CLIENT_ID="your_client_id"
export ANGEL_CLIENT_PIN="your_pin"
export ANGEL_TOTP_KEY="your_totp_secret"
```

### 3. Run a Backtest
```bash
# Default: SMA Crossover on RELIANCE
python main.py

# Custom symbol and strategy
python main.py --symbol INFY --strategy sma_crossover rsi_mean_reversion

# Full options
python main.py --symbol RELIANCE \
    --exchange NSE \
    --timeframe ONE_DAY \
    --start_date 2023-01-01 \
    --end_date 2024-01-01 \
    --strategy sma_crossover \
    --capital 100000 \
    --sizing fixed_capital_pct \
    --segment intraday_equity
```

## Adding a New Strategy

1. Create a new file in `strategies/` (e.g., `strategies/macd_strategy.py`)
2. Subclass `Strategy` and implement `generate_signals()`
3. Register it in `main.py`'s `STRATEGY_REGISTRY`

```python
from strategies.base import Strategy
import pandas as pd

class MACDStrategy(Strategy):
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(name="MACD", fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Your MACD logic here
        # Must add a 'signal' column: 1 (buy), -1 (sell), 0 (hold)
        df["signal"] = 0
        return df
```

Then in `main.py`:
```python
from strategies.macd_strategy import MACDStrategy
STRATEGY_REGISTRY["macd"] = MACDStrategy
```

## Position Sizing Options

- `fixed_quantity`: Trade a fixed number of shares (set in config.py)
- `fixed_capital_pct`: Use a fixed % of capital per trade
- `risk_based`: Size based on % risk per trade with stop-loss

## Broker Charges (Zerodha Rates)

Calculated by `broker_charges.py`:
- **Intraday Equity**: 0.03% or ₹20/order (whichever is lower)
- **Delivery Equity**: Free brokerage
- **F&O Futures**: 0.03% or ₹20/order
- **F&O Options**: Flat ₹20/order

Plus statutory charges: STT, Exchange Txn, GST (18%), SEBI fees, Stamp Duty.

## Performance Metrics

- Total Return (%)
- CAGR (%)
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)
- Profit Factor
- Total Charges Paid
- Net Profit

## Notes

- If Angel One API is unavailable, synthetic data is generated for testing
- Data is cached in `cache/` to avoid repeated API calls
- Trade logs are exported to `output/` as CSV files
- Equity curve plots are saved as PNG files
