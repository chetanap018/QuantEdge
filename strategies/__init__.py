"""
strategies/
------------
Plug-in directory for trading strategies.

To add a new strategy:
1. Create a new .py file in this folder.
2. Subclass Strategy (from strategies.base).
3. Implement generate_signals(data) -> DataFrame.
4. The backtest engine will automatically discover it.

Example:
    from strategies.base import Strategy

    class MyStrategy(Strategy):
        def generate_signals(self, data):
            # Your logic here
            return data_with_signals
"""

from strategies.base import Strategy, EventDrivenStrategy, is_event_driven
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.MACDCrossoverStrategy import MACDCrossoverStrategy
from strategies.opening_candle import OpeningCandleStrategy
from strategies.suzlon_intraday import SuzlonStrategy
from strategies.hero_orb import HeroOrbStrategy

__all__ = [
    "Strategy",
    "EventDrivenStrategy",
    "is_event_driven",
    "SMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "MACDCrossoverStrategy",
    "OpeningCandleStrategy",
    "SuzlonStrategy",
    "HeroOrbStrategy",
]
