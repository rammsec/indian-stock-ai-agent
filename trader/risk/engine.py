"""Deterministic Risk Management Engine (Hard-Coded Python Gatekeeper).

Implements non-negotiable financial arithmetic with absolute veto authority over AI agents:
1. Fixed fractional risk position sizing
2. Position size ceilings (max 15% equity per stock)
3. Portfolio exposure limits (max 80% total invested)
4. Daily drawdown circuit breaker (halts trading at 2.5% daily loss)
5. Time-of-day execution filters (no entries after 14:45 IST, square-off at 15:15 IST)
6. Concurrent position ceilings (max 5 simultaneous positions)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal

from ..config import (
    IST_TZ,
    INTRADAY_ENTRY_CUTOFF_TIME,
    INTRADAY_MANDATORY_SQUAREOFF_TIME,
    MAX_RISK_PER_TRADE_PCT,
    MAX_POSITION_WEIGHT_PCT,
    MAX_PORTFOLIO_EXPOSURE_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_CONCURRENT_POSITIONS,
)
from ..portfolio.ledger import IntradayPaperPortfolio


@dataclass(frozen=True)
class RiskEvaluationResult:
    """Outcome of deterministic risk evaluation."""

    approved: bool
    status: Literal["APPROVED", "SIZED_DOWN", "REJECTED"]
    approved_quantity: int
    requested_quantity: int
    rejection_reasons: list[str]
    max_risk_amount_inr: float
    estimated_trade_risk_inr: float


class DeterministicRiskEngine:
    """Hard-coded risk engine with zero LLM override capability."""

    def __init__(
        self,
        max_risk_per_trade_pct: float = MAX_RISK_PER_TRADE_PCT,
        max_position_weight_pct: float = MAX_POSITION_WEIGHT_PCT,
        max_portfolio_exposure_pct: float = MAX_PORTFOLIO_EXPOSURE_PCT,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        max_concurrent_positions: int = MAX_CONCURRENT_POSITIONS,
    ) -> None:
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_weight_pct = max_position_weight_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.is_daily_halted: bool = False

    def check_daily_loss_limit(self, portfolio: IntradayPaperPortfolio) -> bool:
        """Check if cumulative daily loss exceeds circuit breaker threshold."""
        if portfolio.day_start_equity <= 0:
            return False

        daily_pnl = portfolio.total_equity - portfolio.day_start_equity
        daily_loss_pct = abs(daily_pnl) / portfolio.day_start_equity if daily_pnl < 0 else 0.0

        if daily_loss_pct >= self.max_daily_loss_pct:
            self.is_daily_halted = True
            return True
        return False

    def evaluate_order(
        self,
        symbol: str,
        side: str,
        requested_quantity: int,
        entry_price: float,
        stop_loss_price: float | None,
        portfolio: IntradayPaperPortfolio,
        current_time: datetime,
    ) -> RiskEvaluationResult:
        """Evaluate order proposal against all hard-coded risk constraints."""
        reasons: list[str] = []
        side_upper = side.upper()
        current_t = current_time.astimezone(IST_TZ).time()

        # 1. Check if daily circuit breaker has halted trading
        if self.is_daily_halted or self.check_daily_loss_limit(portfolio):
            reasons.append("Daily loss limit breached (2.5%); terminal in HALT_TRADING_TODAY mode")
            return RiskEvaluationResult(
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                rejection_reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 2. Check time of day filters
        if current_t >= INTRADAY_MANDATORY_SQUAREOFF_TIME:
            reasons.append(f"At or past 15:15 IST mandatory square-off time ({current_t.strftime('%H:%M')})")
            return RiskEvaluationResult(
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                rejection_reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        if side_upper == "BUY" and current_t >= INTRADAY_ENTRY_CUTOFF_TIME:
            reasons.append(f"Past 14:45 IST intraday entry cutoff time ({current_t.strftime('%H:%M')})")
            return RiskEvaluationResult(
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                rejection_reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # For SELL (exits), allow execution
        if side_upper == "SELL":
            return RiskEvaluationResult(
                approved=True,
                status="APPROVED",
                approved_quantity=requested_quantity,
                requested_quantity=requested_quantity,
                rejection_reasons=[],
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # --- BUY Order Validations ---

        # 3. Stop-Loss is mandatory for intraday long entries
        if stop_loss_price is None or stop_loss_price >= entry_price:
            reasons.append(f"Invalid stop-loss price ({stop_loss_price}) for entry ({entry_price}); must be below entry")
            return RiskEvaluationResult(
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                rejection_reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 4. Check concurrent open positions ceiling
        if symbol not in portfolio.positions and len(portfolio.positions) >= self.max_concurrent_positions:
            reasons.append(f"Maximum concurrent positions ({self.max_concurrent_positions}) reached")
            return RiskEvaluationResult(
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                rejection_reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 5. Fixed Fractional Risk Sizing
        total_equity = portfolio.total_equity
        max_risk_inr = total_equity * self.max_risk_per_trade_pct
        risk_per_share = entry_price - stop_loss_price
        max_shares_by_risk = int(max_risk_inr / risk_per_share) if risk_per_share > 0 else 0

        # 6. Position Size Ceiling (max 15% equity per stock)
        max_capital_per_pos = total_equity * self.max_position_weight_pct
        existing_pos_val = portfolio.positions[symbol].quantity * entry_price if symbol in portfolio.positions else 0.0
        remaining_pos_capital = max(0.0, max_capital_per_pos - existing_pos_val)
        max_shares_by_cap = int(remaining_pos_capital / entry_price)

        # 7. Total Portfolio Exposure Limit (max 80% total invested)
        max_portfolio_invested = total_equity * self.max_portfolio_exposure_pct
        remaining_portfolio_capital = max(0.0, max_portfolio_invested - portfolio.invested_capital)
        max_shares_by_exposure = int(remaining_portfolio_capital / entry_price)

        # 8. Available Cash Limit
        max_shares_by_cash = int(portfolio.cash / entry_price)

        # Determine approved quantity
        approved_qty = min(
            requested_quantity,
            max_shares_by_risk,
            max_shares_by_cap,
            max_shares_by_exposure,
            max_shares_by_cash,
        )

        if approved_qty <= 0:
            reasons.append(
                f"Position sized down to 0: risk_shares={max_shares_by_risk}, "
                f"pos_cap_shares={max_shares_by_cap}, exposure_shares={max_shares_by_exposure}, "
                f"cash_shares={max_shares_by_cash}"
            )
            return RiskEvaluationResult(
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                rejection_reasons=reasons,
                max_risk_amount_inr=max_risk_inr,
                estimated_trade_risk_inr=0.0,
            )

        status: Literal["APPROVED", "SIZED_DOWN"] = "APPROVED" if approved_qty == requested_quantity else "SIZED_DOWN"
        if approved_qty < requested_quantity:
            reasons.append(
                f"Order sized down from {requested_quantity} to {approved_qty} "
                f"(capped by risk/capital/exposure constraints)"
            )

        trade_risk = round(approved_qty * risk_per_share, 2)

        return RiskEvaluationResult(
            approved=True,
            status=status,
            approved_quantity=approved_qty,
            requested_quantity=requested_quantity,
            rejection_reasons=reasons,
            max_risk_amount_inr=round(max_risk_inr, 2),
            estimated_trade_risk_inr=trade_risk,
        )

    def reset_daily_halt(self) -> None:
        """Reset daily halt status at the opening of a new daily trading session."""
        self.is_daily_halted = False

