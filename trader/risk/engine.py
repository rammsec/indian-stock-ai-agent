"""Deterministic Risk Management Engine (Hard-Coded Python Gatekeeper).

Implements non-negotiable financial arithmetic with absolute veto authority over AI agents:
1. 3-state session machine: ACTIVE -> REDUCING (15:05 IST wind-down) -> HALTED
2. Fixed fractional risk position sizing
3. Position size ceilings (max 15% equity per stock)
4. Portfolio exposure limits (max 80% total invested)
5. Daily drawdown & intraday MTM peak-to-trough circuit breakers
6. Thread-safe CashReservationLedger (anti-double-spend) & sliding-window OrderRateThrottler
7. Fail-closed naive-datetime rejection & SHA-256 hash-chained RiskAuditEvent log
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Literal, Sequence

from ..config import (
    INTRADAY_ENTRY_CUTOFF_TIME,
    INTRADAY_MANDATORY_SQUAREOFF_TIME,
    INTRADAY_REDUCING_WIND_DOWN_TIME,
    IST_TZ,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_LOSS_PCT,
    MAX_INTRADAY_MTM_DRAWDOWN_PCT,
    MAX_ORDERS_PER_MINUTE,
    MAX_PORTFOLIO_EXPOSURE_PCT,
    MAX_POSITION_WEIGHT_PCT,
    MAX_RISK_PER_TRADE_PCT,
)
from ..portfolio.cost_calculator import calculate_intraday_costs
from ..portfolio.ledger import IntradayPaperPortfolio


class RiskState(str, Enum):
    """3-state deterministic intraday risk state machine."""

    ACTIVE = "ACTIVE"
    REDUCING = "REDUCING"
    HALTED = "HALTED"


@dataclass(frozen=True)
class RiskAuditEvent:
    """Immutable SHA-256 hash-chained risk evaluation audit record."""

    seq: int
    timestamp_ist: str
    symbol: str
    side: str
    risk_state: str
    approved: bool
    status: str
    approved_quantity: int
    reasons: tuple[str, ...]
    prev_hash: str
    event_hash: str


def _compute_audit_event_hash(
    *,
    seq: int,
    timestamp_ist: str,
    symbol: str,
    side: str,
    risk_state: str,
    approved: bool,
    status: str,
    approved_quantity: int,
    reasons: tuple[str, ...],
    prev_hash: str,
) -> str:
    payload = json.dumps(
        {
            "seq": seq,
            "timestamp_ist": timestamp_ist,
            "symbol": symbol,
            "side": side,
            "risk_state": risk_state,
            "approved": approved,
            "status": status,
            "approved_quantity": approved_quantity,
            "reasons": list(reasons),
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CashReservationLedger:
    """Thread-safe pending cash reservation ledger preventing concurrent double-spend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reservations: dict[str, tuple[str, float]] = {}

    def reserve(self, order_id: str, symbol: str, amount_inr: float) -> None:
        if amount_inr < 0:
            raise ValueError(f"Reservation amount must be non-negative, got {amount_inr}")
        with self._lock:
            self._reservations[order_id] = (symbol.upper(), float(amount_inr))

    def release(self, order_id: str) -> float:
        with self._lock:
            entry = self._reservations.pop(order_id, None)
            return entry[1] if entry is not None else 0.0

    def clear(self) -> None:
        with self._lock:
            self._reservations.clear()

    @property
    def total_reserved_inr(self) -> float:
        with self._lock:
            return sum(amt for _, amt in self._reservations.values())

    def reserved_for_symbol(self, symbol: str) -> float:
        sym_u = symbol.upper()
        with self._lock:
            return sum(amt for sym, amt in self._reservations.values() if sym == sym_u)


class OrderRateThrottler:
    """Thread-safe sliding-window order rate limiter."""

    def __init__(self, max_orders_per_minute: int = MAX_ORDERS_PER_MINUTE) -> None:
        self.max_orders_per_minute = max_orders_per_minute
        self._lock = threading.Lock()
        self._timestamps: deque[datetime] = deque()

    def allow_and_record(self, ts_ist: datetime) -> bool:
        cutoff = ts_ist - timedelta(seconds=60)
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_orders_per_minute:
                return False
            self._timestamps.append(ts_ist)
            return True

    def clear(self) -> None:
        with self._lock:
            self._timestamps.clear()


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
    risk_state: str = RiskState.ACTIVE.value
    audit_event_hash: str = ""


class DeterministicRiskEngine:
    """Hard-coded risk engine with zero LLM override capability."""

    _GENESIS_HASH = "0" * 64

    def __init__(
        self,
        max_risk_per_trade_pct: float = MAX_RISK_PER_TRADE_PCT,
        max_position_weight_pct: float = MAX_POSITION_WEIGHT_PCT,
        max_portfolio_exposure_pct: float = MAX_PORTFOLIO_EXPOSURE_PCT,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        max_concurrent_positions: int = MAX_CONCURRENT_POSITIONS,
        max_intraday_mtm_drawdown_pct: float = MAX_INTRADAY_MTM_DRAWDOWN_PCT,
        max_orders_per_minute: int = MAX_ORDERS_PER_MINUTE,
    ) -> None:
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_weight_pct = max_position_weight_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_intraday_mtm_drawdown_pct = max_intraday_mtm_drawdown_pct

        self.state: RiskState = RiskState.ACTIVE
        self._session_date: date | None = None
        self._intraday_peak_equity: float = 0.0
        self.cash_ledger = CashReservationLedger()
        self.rate_throttler = OrderRateThrottler(max_orders_per_minute=max_orders_per_minute)
        self._audit_lock = threading.Lock()
        self._audit_log: list[RiskAuditEvent] = []

    @property
    def is_daily_halted(self) -> bool:
        return self.state == RiskState.HALTED

    @is_daily_halted.setter
    def is_daily_halted(self, value: bool) -> None:
        if value:
            self.state = RiskState.HALTED
        elif self.state == RiskState.HALTED:
            self.state = RiskState.ACTIVE

    @property
    def audit_log(self) -> tuple[RiskAuditEvent, ...]:
        with self._audit_lock:
            return tuple(self._audit_log)

    def verify_audit_chain(self, events: Sequence[RiskAuditEvent] | None = None) -> bool:
        """Verify SHA-256 hash chain integrity across all recorded RiskAuditEvents."""
        chain = list(events) if events is not None else list(self.audit_log)
        expected_prev = self._GENESIS_HASH
        for idx, ev in enumerate(chain):
            if ev.seq != idx or ev.prev_hash != expected_prev:
                return False
            recomputed = _compute_audit_event_hash(
                seq=ev.seq,
                timestamp_ist=ev.timestamp_ist,
                symbol=ev.symbol,
                side=ev.side,
                risk_state=ev.risk_state,
                approved=ev.approved,
                status=ev.status,
                approved_quantity=ev.approved_quantity,
                reasons=ev.reasons,
                prev_hash=ev.prev_hash,
            )
            if ev.event_hash != recomputed:
                return False
            expected_prev = ev.event_hash
        return True

    def _record_audit(
        self,
        *,
        timestamp_ist: datetime,
        symbol: str,
        side: str,
        approved: bool,
        status: Literal["APPROVED", "SIZED_DOWN", "REJECTED"],
        approved_quantity: int,
        requested_quantity: int,
        reasons: list[str],
        max_risk_amount_inr: float,
        estimated_trade_risk_inr: float,
    ) -> RiskEvaluationResult:
        ts_str = timestamp_ist.isoformat()
        with self._audit_lock:
            prev_hash = self._audit_log[-1].event_hash if self._audit_log else self._GENESIS_HASH
            seq = len(self._audit_log)
            reasons_tuple = tuple(reasons)
            ev_hash = _compute_audit_event_hash(
                seq=seq,
                timestamp_ist=ts_str,
                symbol=symbol,
                side=side,
                risk_state=self.state.value,
                approved=approved,
                status=status,
                approved_quantity=approved_quantity,
                reasons=reasons_tuple,
                prev_hash=prev_hash,
            )
            self._audit_log.append(
                RiskAuditEvent(
                    seq=seq,
                    timestamp_ist=ts_str,
                    symbol=symbol,
                    side=side,
                    risk_state=self.state.value,
                    approved=approved,
                    status=status,
                    approved_quantity=approved_quantity,
                    reasons=reasons_tuple,
                    prev_hash=prev_hash,
                    event_hash=ev_hash,
                )
            )
        return RiskEvaluationResult(
            approved=approved,
            status=status,
            approved_quantity=approved_quantity,
            requested_quantity=requested_quantity,
            rejection_reasons=reasons,
            max_risk_amount_inr=max_risk_amount_inr,
            estimated_trade_risk_inr=estimated_trade_risk_inr,
            risk_state=self.state.value,
            audit_event_hash=ev_hash,
        )

    def check_daily_loss_limit(self, portfolio: IntradayPaperPortfolio) -> bool:
        """Check if cumulative daily loss or intraday MTM drawdown breaches circuit breaker."""
        total_equity = float(portfolio.total_equity)
        if total_equity > self._intraday_peak_equity:
            self._intraday_peak_equity = total_equity

        if portfolio.day_start_equity > 0:
            daily_pnl = total_equity - portfolio.day_start_equity
            daily_loss_pct = abs(daily_pnl) / portfolio.day_start_equity if daily_pnl < 0 else 0.0
            if daily_loss_pct >= self.max_daily_loss_pct:
                self.state = RiskState.HALTED
                return True

        if self._intraday_peak_equity > 0 and self.max_intraday_mtm_drawdown_pct > 0:
            mtm_dd = (self._intraday_peak_equity - total_equity) / self._intraday_peak_equity
            if mtm_dd >= self.max_intraday_mtm_drawdown_pct:
                self.state = RiskState.HALTED
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
        *,
        force_reduce_only: bool = False,
        action_type: str | None = None,
        include_entry_friction_in_cash_cap: bool = False,
    ) -> RiskEvaluationResult:
        """Evaluate order proposal against all hard-coded risk constraints."""
        reasons: list[str] = []
        side_upper = side.upper()
        action_upper = action_type.upper() if action_type is not None else None

        if action_upper is not None:
            is_exit_order = action_upper in ("EXIT_LONG", "EXIT_SHORT", "COVER_SHORT", "RISK_EXIT")
            is_short_entry = action_upper in ("ENTRY_SHORT", "OPEN_SHORT")
        else:
            # Backward-compatible default when action_type is omitted: BUY=long entry, SELL=long exit
            is_exit_order = side_upper == "SELL"
            is_short_entry = False

        # 0. Fail closed on timezone-naive datetimes
        if current_time.tzinfo is None:
            now_ist = datetime.now(IST_TZ)
            reasons.append("Naive datetime rejected fail-closed; current_time must be timezone-aware")
            return self._record_audit(
                timestamp_ist=now_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        current_ist = current_time.astimezone(IST_TZ)
        if self._session_date is None:
            self._session_date = current_ist.date()
            self._intraday_peak_equity = max(self._intraday_peak_equity, float(portfolio.total_equity))
        current_t = current_ist.time()

        # 1. Check daily loss / intraday MTM drawdown circuit breaker
        if self.state == RiskState.HALTED or self.check_daily_loss_limit(portfolio):
            if is_exit_order and force_reduce_only:
                pos = portfolio.positions.get(symbol)
                held_qty = pos.quantity if pos is not None else 0
                if held_qty <= 0:
                    reasons.append(f"No held position in {symbol} for forced square-off")
                    return self._record_audit(
                        timestamp_ist=current_ist,
                        symbol=symbol,
                        side=side_upper,
                        approved=False,
                        status="REJECTED",
                        approved_quantity=0,
                        requested_quantity=requested_quantity,
                        reasons=reasons,
                        max_risk_amount_inr=0.0,
                        estimated_trade_risk_inr=0.0,
                    )
                approved_exit = min(requested_quantity, held_qty)
                status_exit: Literal["APPROVED", "SIZED_DOWN"] = (
                    "APPROVED" if approved_exit == requested_quantity else "SIZED_DOWN"
                )
                return self._record_audit(
                    timestamp_ist=current_ist,
                    symbol=symbol,
                    side=side_upper,
                    approved=True,
                    status=status_exit,
                    approved_quantity=approved_exit,
                    requested_quantity=requested_quantity,
                    reasons=[],
                    max_risk_amount_inr=0.0,
                    estimated_trade_risk_inr=0.0,
                )

            reasons.append("Daily loss limit breached (or MTM drawdown); terminal in HALTED mode")
            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 2. Transition ACTIVE -> REDUCING at 15:05 IST wind-down
        if self.state == RiskState.ACTIVE and current_t >= INTRADAY_REDUCING_WIND_DOWN_TIME:
            self.state = RiskState.REDUCING

        # 3. For exits (EXIT_LONG / COVER_SHORT / RISK_EXIT / legacy SELL), allow in ACTIVE and REDUCING
        if is_exit_order:
            pos = portfolio.positions.get(symbol)
            held_qty = pos.quantity if pos is not None else 0
            if self.state == RiskState.REDUCING or action_upper is not None:
                if held_qty <= 0:
                    reasons.append(f"No held position in {symbol} to exit in {self.state.value} state")
                    return self._record_audit(
                        timestamp_ist=current_ist,
                        symbol=symbol,
                        side=side_upper,
                        approved=False,
                        status="REJECTED",
                        approved_quantity=0,
                        requested_quantity=requested_quantity,
                        reasons=reasons,
                        max_risk_amount_inr=0.0,
                        estimated_trade_risk_inr=0.0,
                    )
                if pos is not None and action_upper == "EXIT_LONG" and pos.side != "LONG":
                    reasons.append(f"EXIT_LONG rejected: open position in {symbol} is {pos.side}, not LONG")
                    return self._record_audit(
                        timestamp_ist=current_ist,
                        symbol=symbol,
                        side=side_upper,
                        approved=False,
                        status="REJECTED",
                        approved_quantity=0,
                        requested_quantity=requested_quantity,
                        reasons=reasons,
                        max_risk_amount_inr=0.0,
                        estimated_trade_risk_inr=0.0,
                    )
                if pos is not None and action_upper in ("EXIT_SHORT", "COVER_SHORT") and pos.side != "SHORT":
                    reasons.append(f"COVER_SHORT rejected: open position in {symbol} is {pos.side}, not SHORT")
                    return self._record_audit(
                        timestamp_ist=current_ist,
                        symbol=symbol,
                        side=side_upper,
                        approved=False,
                        status="REJECTED",
                        approved_quantity=0,
                        requested_quantity=requested_quantity,
                        reasons=reasons,
                        max_risk_amount_inr=0.0,
                        estimated_trade_risk_inr=0.0,
                    )
                approved_exit = min(requested_quantity, held_qty)
                status_exit = "APPROVED" if approved_exit == requested_quantity else "SIZED_DOWN"
                return self._record_audit(
                    timestamp_ist=current_ist,
                    symbol=symbol,
                    side=side_upper,
                    approved=True,
                    status=status_exit,
                    approved_quantity=approved_exit,
                    requested_quantity=requested_quantity,
                    reasons=[],
                    max_risk_amount_inr=0.0,
                    estimated_trade_risk_inr=0.0,
                )

            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=True,
                status="APPROVED",
                approved_quantity=requested_quantity,
                requested_quantity=requested_quantity,
                reasons=[],
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # --- NEW ENTRY Order Validations (ENTRY_LONG / ENTRY_SHORT) ---

        # 4. Block new entries in REDUCING or past entry/square-off cutoffs
        if self.state == RiskState.REDUCING or current_t >= INTRADAY_MANDATORY_SQUAREOFF_TIME:
            reasons.append(
                f"{action_upper or side_upper} blocked in {self.state.value} / square-off window ({current_t.strftime('%H:%M')} IST)"
            )
            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        if current_t >= INTRADAY_ENTRY_CUTOFF_TIME:
            reasons.append(f"Past 14:45 IST intraday entry cutoff time ({current_t.strftime('%H:%M')})")
            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 5. Sliding-window order rate throttle
        if not self.rate_throttler.allow_and_record(current_ist):
            reasons.append("Order rate throttle exceeded (max orders per minute)")
            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 6. Stop-Loss is mandatory for intraday entries (below entry for LONG, above entry for SHORT)
        if is_short_entry:
            if stop_loss_price is None or stop_loss_price <= entry_price:
                reasons.append(
                    f"Invalid short stop-loss price ({stop_loss_price}) for entry ({entry_price}); must be above entry"
                )
                return self._record_audit(
                    timestamp_ist=current_ist,
                    symbol=symbol,
                    side=side_upper,
                    approved=False,
                    status="REJECTED",
                    approved_quantity=0,
                    requested_quantity=requested_quantity,
                    reasons=reasons,
                    max_risk_amount_inr=0.0,
                    estimated_trade_risk_inr=0.0,
                )
            risk_per_share = stop_loss_price - entry_price
        else:
            if stop_loss_price is None or stop_loss_price >= entry_price:
                reasons.append(
                    f"Invalid stop-loss price ({stop_loss_price}) for entry ({entry_price}); must be below entry"
                )
                return self._record_audit(
                    timestamp_ist=current_ist,
                    symbol=symbol,
                    side=side_upper,
                    approved=False,
                    status="REJECTED",
                    approved_quantity=0,
                    requested_quantity=requested_quantity,
                    reasons=reasons,
                    max_risk_amount_inr=0.0,
                    estimated_trade_risk_inr=0.0,
                )
            risk_per_share = entry_price - stop_loss_price

        # 7. Check concurrent open positions ceiling
        if symbol not in portfolio.positions and len(portfolio.positions) >= self.max_concurrent_positions:
            reasons.append(f"Maximum concurrent positions ({self.max_concurrent_positions}) reached")
            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=0.0,
                estimated_trade_risk_inr=0.0,
            )

        # 8. Fixed Fractional Risk Sizing
        total_equity = portfolio.total_equity
        max_risk_inr = total_equity * self.max_risk_per_trade_pct
        max_shares_by_risk = int(max_risk_inr / risk_per_share) if risk_per_share > 0 else 0

        # 9. Position Size Ceiling (max 15% equity per stock)
        max_capital_per_pos = total_equity * self.max_position_weight_pct
        existing_pos_val = portfolio.positions[symbol].quantity * entry_price if symbol in portfolio.positions else 0.0
        remaining_pos_capital = max(0.0, max_capital_per_pos - existing_pos_val)
        max_shares_by_cap = int(remaining_pos_capital / entry_price) if entry_price > 0 else 0

        # 10. Total Portfolio Exposure Limit (max 80% total invested)
        max_portfolio_invested = total_equity * self.max_portfolio_exposure_pct
        remaining_portfolio_capital = max(0.0, max_portfolio_invested - portfolio.invested_capital)
        max_shares_by_exposure = int(remaining_portfolio_capital / entry_price) if entry_price > 0 else 0

        # 11. Available Cash Limit (net of pending CashReservationLedger holds)
        available_cash = max(0.0, float(portfolio.cash) - self.cash_ledger.total_reserved_inr)
        max_shares_by_cash = int(available_cash / entry_price) if entry_price > 0 else 0
        if include_entry_friction_in_cash_cap:
            while max_shares_by_cash > 0:
                est_cost = calculate_intraday_costs(
                    price=entry_price,
                    quantity=max_shares_by_cash,
                    side=side_upper,
                    cost_config=portfolio.cost_config,
                ).total_friction_inr
                if (entry_price * max_shares_by_cash) + est_cost <= available_cash:
                    break
                max_shares_by_cash -= 1

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
            return self._record_audit(
                timestamp_ist=current_ist,
                symbol=symbol,
                side=side_upper,
                approved=False,
                status="REJECTED",
                approved_quantity=0,
                requested_quantity=requested_quantity,
                reasons=reasons,
                max_risk_amount_inr=max_risk_inr,
                estimated_trade_risk_inr=0.0,
            )

        status: Literal["APPROVED", "SIZED_DOWN"] = (
            "APPROVED" if approved_qty == requested_quantity else "SIZED_DOWN"
        )
        if approved_qty < requested_quantity:
            reasons.append(
                f"Order sized down from {requested_quantity} to {approved_qty} "
                f"(capped by risk/capital/exposure constraints)"
            )

        trade_risk = round(approved_qty * risk_per_share, 2)

        return self._record_audit(
            timestamp_ist=current_ist,
            symbol=symbol,
            side=side_upper,
            approved=True,
            status=status,
            approved_quantity=approved_qty,
            requested_quantity=requested_quantity,
            reasons=reasons,
            max_risk_amount_inr=round(max_risk_inr, 2),
            estimated_trade_risk_inr=trade_risk,
        )

    def reset_daily_session(self, session_date: date | None = None) -> None:
        """Reset risk state, peak equity, cash reservations, and throttler for a new IST date."""
        if session_date is not None and self._session_date is not None and session_date <= self._session_date:
            # Explicit same/earlier date still resets state if called directly, while updating session_date
            pass
        self._session_date = session_date
        self.state = RiskState.ACTIVE
        self._intraday_peak_equity = 0.0
        self.cash_ledger.clear()
        self.rate_throttler.clear()

    def reset_daily_halt(self) -> None:
        """Reset daily halt status at the opening of a new daily trading session."""
        self.reset_daily_session(self._session_date)
