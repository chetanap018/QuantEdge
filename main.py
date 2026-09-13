"""
main.py
-------
Entry point for the backtesting framework.

Usage:
    python main.py
    python main.py --symbol RELIANCE --strategy sma_crossover
"""

import os
import re
import argparse
import logging
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_fetcher import DataFetcher
from backtest_engine import BacktestEngine
from strategies import (
    SMACrossoverStrategy,
    RSIMeanReversionStrategy,
    MACDCrossoverStrategy,
    OpeningCandleStrategy,
    SuzlonStrategy,
    HeroOrbStrategy,
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STRATEGY_REGISTRY = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "macd_crossover": MACDCrossoverStrategy,
    "opening_candle": OpeningCandleStrategy,
    "suzlon": SuzlonStrategy,
    "hero_orb": HeroOrbStrategy,
}


def normalize_strategy_name(name: str) -> str:
    """Accept strategy aliases, filenames, or class names."""
    cleaned = os.path.splitext(os.path.basename(str(name).strip()))[0]
    if not cleaned:
        return ""

    if "_" in cleaned:
        normalized = cleaned.lower().strip("_")
    else:
        tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", cleaned)
        normalized = "_".join(tokens).lower()

    if normalized.endswith("_strategy"):
        normalized = normalized[:-len("_strategy")]
    return normalized


def parse_args():
    """Parse command-line arguments. Falls back to backtest_config.py for defaults."""
    import backtest_config

    parser = argparse.ArgumentParser(
        description="Backtesting Framework for Trading Strategies"
    )
    parser.add_argument("--symbol", type=str, default=backtest_config.SYMBOL)
    parser.add_argument("--exchange", type=str, default=backtest_config.EXCHANGE)
    parser.add_argument("--timeframe", type=str, default=backtest_config.TIMEFRAME)
    parser.add_argument("--start_date", type=str, default=backtest_config.START_DATE)
    parser.add_argument("--end_date", type=str, default=backtest_config.END_DATE)
    parser.add_argument("--strategy", type=str, nargs="+", default=backtest_config.STRATEGIES)
    parser.add_argument("--capital", type=float, default=backtest_config.INITIAL_CAPITAL)
    parser.add_argument("--sizing", type=str, default=backtest_config.POSITION_SIZING,
                        choices=["fixed_quantity", "fixed_capital_pct", "risk_based"])
    parser.add_argument("--segment", type=str, default=backtest_config.SEGMENT,
                        choices=["intraday_equity", "delivery_equity", "futures", "options"])
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_csv", action="store_true", help="Disable CSV export")
    parser.add_argument("--allow_synthetic", action="store_true", default=None,
                        help="Allow synthetic (simulated GBM) data when all real sources fail. "
                             "Default: use config.ALLOW_SYNTHETIC_DATA.")
    return parser.parse_args()


def print_results(results: dict, symbol: str):
    """Print formatted results to console."""
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS: {symbol}")
    print(f"  Strategy: {results['strategy']}")
    print("=" * 60)
    print(f"  Total Trades:       {results['total_trades']}")
    print(f"  Winning Trades:     {results.get('winning_trades', 0)}")
    print(f"  Losing Trades:      {results.get('losing_trades', 0)}")
    print(f"  Win Rate:           {results['win_rate']:.1f}%")
    print(f"  Profit Factor:      {results['profit_factor']:.2f}")
    print("-" * 60)
    print(f"  Total Return:       {results['total_return_pct']:.2f}%")
    print(f"  CAGR:               {results['cagr']:.2f}%")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
    print("-" * 60)
    print(f"  Total Charges:      {results['total_charges']:,.2f}")
    print(f"  Net Profit:         {results['net_profit']:,.2f}")
    print(f"  Final Capital:      {results['final_capital']:,.2f}")
    print("=" * 60)


def plot_equity_curves(all_results: dict, symbol: str, data: pd.DataFrame):
    """Plot equity curves for all strategies."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    ax1.plot(data["datetime"], data["close"], color="gray", alpha=0.5, linewidth=0.8, label="Price")
    ax1.set_ylabel("Price")
    ax1.set_title(f"Backtest Results - {symbol}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for idx, (name, results) in enumerate(all_results.items()):
        equity = results["equity_curve"]
        color = colors[idx % len(colors)]
        ax2.plot(range(len(equity)), equity, color=color, linewidth=1.2, label=name)

    ax2.set_ylabel("Equity")
    ax2.set_xlabel("Bars")
    ax2.set_title("Equity Curves")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(
        config.OUTPUT_DIR,
        f"equity_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {plot_path}")
    plt.close()


def export_trade_log(results: dict, symbol: str):
    """Export the trade log to a CSV file."""
    trade_log = results.get("trade_log", [])
    if not trade_log:
        return

    df = pd.DataFrame(trade_log)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(
        config.OUTPUT_DIR,
        f"trades_{symbol}_{results['strategy'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"  Trade log saved: {csv_path}")


def run_backtest(
    symbol: str,
    strategy_names: list,
    start_date: str = None,
    end_date: str = None,
    timeframe: str = "ONE_DAY",
    exchange: str = "NSE",
    capital: float = None,
    sizing: str = None,
    segment: str = "intraday_equity",
    plot: bool = True,
    export_csv: bool = True,
    allow_synthetic: bool = None,
):
    """Run backtest for one or more strategies on a symbol."""
    print(f"\n  Fetching data for {exchange}:{symbol} ({timeframe})...")
    fetcher = DataFetcher()
    data = fetcher.fetch_historical_data(
        symbol=symbol, exchange=exchange, timeframe=timeframe,
        start_date=start_date, end_date=end_date,
        allow_synthetic=allow_synthetic,
    )
    # Safety: ensure data is sorted by datetime ascending
    if not data.empty and "datetime" in data.columns:
        data = data.sort_values("datetime").reset_index(drop=True)
    print(f"  Loaded {len(data)} bars.")
    fetcher.logout()

    if data.empty:
        logger.error("No data available. Exiting.")
        return

    all_results = {}
    for strat_name in strategy_names:
        strat_name_lower = normalize_strategy_name(strat_name)
        if strat_name_lower not in STRATEGY_REGISTRY:
            print(f"  WARNING: Unknown strategy '{strat_name}', skipping.")
            print(f"  Available: {list(STRATEGY_REGISTRY.keys())}")
            continue

        strategy_class = STRATEGY_REGISTRY[strat_name_lower]
        strategy = strategy_class()

        print(f"\n  Running strategy: {strategy.name}...")
        engine = BacktestEngine(
            strategy=strategy, data=data,
            initial_capital=capital, position_sizing=sizing, segment=segment,
        )
        results = engine.run()
        all_results[strat_name_lower] = results

        print_results(results, symbol)

        if export_csv:
            export_trade_log(results, symbol)

    if plot and all_results:
        plot_equity_curves(all_results, symbol, data)


def main():
    """Main entry point."""
    import backtest_config

    args = parse_args()

    print("\n" + "=" * 60)
    print("  BACKTESTING FRAMEWORK")
    print("  Data: Angel One SmartAPI | Charges: Zerodha")
    print("=" * 60)

    run_backtest(
        symbol=args.symbol, strategy_names=args.strategy,
        start_date=args.start_date, end_date=args.end_date,
        timeframe=args.timeframe, exchange=args.exchange,
        capital=args.capital, sizing=args.sizing, segment=args.segment,
        plot=backtest_config.ENABLE_PLOT and not args.no_plot,
        export_csv=backtest_config.ENABLE_CSV and not args.no_csv,
        allow_synthetic=args.allow_synthetic,
    )


if __name__ == "__main__":
    main()
