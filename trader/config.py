"""Configuration and constants for the Personal AI Intraday Trading Platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

# Timezone definition
IST_TZ = ZoneInfo("Asia/Kolkata")

# Market Session Timings (IST)
MARKET_PREOPEN_START = time(9, 0)
MARKET_OPEN_TIME = time(9, 15)
MARKET_CLOSE_TIME = time(15, 30)

# Intraday Execution Timings (IST)
INTRADAY_ENTRY_CUTOFF_TIME = time(14, 45)      # No new positions permitted after 14:45 IST
INTRADAY_MANDATORY_SQUAREOFF_TIME = time(15, 15) # Force-close all intraday positions at 15:15 IST

# Official NSE Trading Holidays (YYYY-MM-DD)
NSE_TRADING_HOLIDAYS: set[str] = {
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-03", "2026-03-20", "2026-03-31", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-05-27", "2026-06-26", "2026-09-14",
    "2026-10-02", "2026-10-20", "2026-11-09", "2026-11-24", "2026-12-25",
}


def is_nse_trading_day(dt: date | None = None) -> bool:
    """Return True if `dt` (in Asia/Kolkata) is a weekday and not an official NSE holiday."""
    target = dt or datetime.now(IST_TZ).date()
    if target.weekday() >= 5:  # Saturday or Sunday
        return False
    return target.strftime("%Y-%m-%d") not in NSE_TRADING_HOLIDAYS


# Default Portfolio Settings
DEFAULT_INITIAL_CAPITAL_INR = 1_000_000.0  # ₹10,00,000 (10 Lakhs)
MAX_RISK_PER_TRADE_PCT = 0.01              # 1.0% maximum portfolio risk per trade
MAX_POSITION_WEIGHT_PCT = 0.15             # 15.0% maximum allocation per single stock
MAX_PORTFOLIO_EXPOSURE_PCT = 0.80          # 80.0% maximum total capital invested
MAX_DAILY_LOSS_PCT = 0.025                 # 2.5% daily drawdown circuit breaker (HALT_TRADING_TODAY)
MAX_CONCURRENT_POSITIONS = 5               # Maximum open positions at any time


@dataclass(frozen=True)
class CostConfig:
    """Configurable statutory Indian transaction friction rates (Intraday MIS)."""

    brokerage_pct: float = 0.0003            # 0.03% or max ₹20 per executed order
    max_brokerage_per_order_inr: float = 20.0
    stt_sell_pct: float = 0.00025            # 0.025% on Sell turnover only
    stamp_duty_buy_pct: float = 0.00003      # 0.003% on Buy turnover only
    nse_transaction_charge_pct: float = 0.0000297  # 0.00297% per leg
    bse_transaction_charge_pct: float = 0.0000375  # 0.00375% per leg
    sebi_turnover_fee_pct: float = 0.000001        # ₹10 per Crore (0.0001%)
    gst_rate_pct: float = 0.18                     # 18% on (Brokerage + Exchange Charge + SEBI Fee)
    base_slippage_bps: float = 5.0                 # 5 bps base slippage per leg for liquid large-caps


# Default cost instance
DEFAULT_COST_CONFIG = CostConfig()
