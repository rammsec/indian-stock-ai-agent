"""NSE & BSE Cross-Exchange Arbitrage, executable spread deduction, and paired order simulation."""

from .spread_calculator import (
    calculate_executable_arbitrage_spread,
    CrossExchangeOpportunity,
)
from .paired_executor import PairedArbitrageExecutor, PairedExecutionResult

__all__ = [
    "calculate_executable_arbitrage_spread",
    "CrossExchangeOpportunity",
    "PairedArbitrageExecutor",
    "PairedExecutionResult",
]
