"""Abstract Base Strategy and metadata contracts for Strategy Research Library."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import pandas as pd

from ..data.schema import Candle5M


@dataclass(frozen=True)
class StrategyMetadata:
    """Catalog metadata for each quantitative research strategy."""

    strategy_id: str
    name: str
    origin_source: str
    source_url: str
    license_type: str
    timeframe: str = "5m"
    market_applicability: str = "NSE & BSE Cash"
    required_indicators: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    validation_status: Literal["APPROVED_PAPER", "IN_VALIDATION", "REJECTED"] = "IN_VALIDATION"


@dataclass
class StrategySignal:
    """Trading signal generated after a candle closes."""

    strategy_id: str
    symbol: str
    timestamp: datetime
    action: Literal["ENTER_LONG", "EXIT", "HOLD", "NO_SIGNAL"]
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    confidence: float = 0.5
    rationale: list[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """Abstract base class for all testable intraday strategies."""

    def __init__(self, metadata: StrategyMetadata) -> None:
        self.metadata = metadata

    @abstractmethod
    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        """Evaluate trading rules at a specific closed candle index without lookahead."""
        pass
