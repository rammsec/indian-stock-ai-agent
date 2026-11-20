"""Typed data models and schemas for 5-minute candles and instruments."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candle5M(_StrictModel):
    """Represents a single 5-minute OHLCV market candle."""

    symbol: str
    exchange: Literal["NSE", "BSE"]
    timestamp: datetime  # Must be timezone-aware (IST)
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    vwap: float | None = None
    trades_count: int | None = None

    @field_validator("timestamp")
    @classmethod
    def require_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Candle timestamp must be timezone-aware (Asia/Kolkata / IST)")
        return v

    @model_validator(mode="after")
    def validate_ohlc_invariants(self) -> Candle5M:
        """Enforce strict OHLC relationship invariants."""
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) cannot be less than Open ({self.open})")
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) cannot be less than Close ({self.close})")
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) cannot be greater than Open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) cannot be greater than Close ({self.close})")
        return self


class InstrumentInfo(_StrictModel):
    """Metadata for dual-listed Indian equities with ISIN resolution."""

    symbol: str               # Canonical symbol (e.g., RELIANCE)
    nse_symbol: str           # NSE ticker (e.g., RELIANCE)
    bse_code: str             # BSE scrip code (e.g., 500325)
    isin: str                 # International Securities Identification Number (e.g., INE002A01018)
    company_name: str
    sector: str
    lot_size: int = 1
    tick_size: float = 0.05
    is_fno_eligible: bool = False


class CandleValidationResult(_StrictModel):
    """Summary of data-quality and integrity checks on a candle series."""

    symbol: str
    exchange: str
    total_input_bars: int
    valid_bars_count: int
    dropped_bars_count: int
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    missing_intervals: list[str] = Field(default_factory=list)
