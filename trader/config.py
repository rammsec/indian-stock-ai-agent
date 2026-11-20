"""Configuration and constants for the Personal AI Intraday Trading Platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
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

# Default Portfolio Settings
DEFAULT_INITIAL_CAPITAL_INR = 1_000_000.0  # ₹10,00,000 (10 Lakhs)
MAX_RISK_PER_TRADE_PCT = 0.01              # 1.0% maximum portfolio risk per trade
MAX_POSITION_WEIGHT_PCT = 0.15             # 15.0% maximum allocation per single stock
MAX_PORTFOLIO_EXPOSURE_PCT = 0.80          # 80.0% maximum total capital invested
MAX_DAILY_LOSS_PCT = 0.025                 # 2.5% daily drawdown circuit breaker (HALT_TRADING_TODAY)
MAX_CONCURRENT_POSITIONS = 5               # Maximum open positions at any time

# Statutory Indian Transaction Friction Rates (Intraday MIS)
INTRADAY_BROKERAGE_PCT = 0.0003            # 0.03% or ₹20 per executed order (whichever is lower)
MAX_BROKERAGE_PER_ORDER_INR = 20.0
INTRADAY_STT_SELL_PCT = 0.00025            # 0.025% on Sell turn-over only
INTRADAY_STAMP_DUTY_BUY_PCT = 0.00003      # 0.003% on Buy turnover only
NSE_TRANSACTION_CHARGE_PCT = 0.0000297     # 0.00297% per leg
BSE_TRANSACTION_CHARGE_PCT = 0.0000375     # 0.00375% per leg
SEBI_TURNOVER_FEE_PCT = 0.000001           # ₹10 per Crore (0.0001%)
GST_RATE_PCT = 0.18                        # 18% on (Brokerage + Exchange Charge + SEBI Fee)
BASE_SLIPPAGE_BPS = 5.0                    # 5 bps base slippage per leg for liquid large-caps
