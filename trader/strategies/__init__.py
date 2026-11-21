"""Strategy Research Library, quantitative strategy implementations, and backtesting."""

from .base import BaseStrategy, StrategySignal, StrategyMetadata
from .vwap_pullback import VWAPPullbackStrategy
from .orb_breakout import ORBBreakoutStrategy
from .ema_cross import EMACrossStrategy
from .supertrend import SupertrendStrategy
from .backtester import IntradayBacktester, BacktestResult

__all__ = [
    "BaseStrategy",
    "StrategySignal",
    "StrategyMetadata",
    "VWAPPullbackStrategy",
    "ORBBreakoutStrategy",
    "EMACrossStrategy",
    "SupertrendStrategy",
    "IntradayBacktester",
    "BacktestResult",
]
