"""CYCLE-010 Phase 1 Acceptance & Unit Tests:
- NSE 15:15-15:30 IST shortened 1h closing bar causality in ParquetCSVMarketFeed
- Protective exits never trapped on mid-session directional permission flips
- Expired human instructions fail closed on new entries
- Action-to-candle alignment under ALLOW_BOTH
- stop_loss / target forwarding through MandatoryExecutionGate.execute_with_gate_protection
- Explicit OPEN_LONG / EXIT_LONG / OPEN_SHORT / COVER_SHORT separation & double-entry capital conservation
- Intrabar OHLC stop-first precedence & ₹0.05 NSE tick-size rounding
- Risk Engine short-entry validation & friction-aware cash sizing
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trader.authority import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    MandatoryExecutionGate,
    PermissionType,
)
from trader.config import IST_TZ
from trader.data.market_feed import ParquetCSVMarketFeed
from trader.portfolio.ledger import (
    IntradayPaperPortfolio,
    OrderSide,
    OrderStatus,
    round_to_nse_tick,
)
from trader.risk.engine import DeterministicRiskEngine
from trader.timing.candle_intelligence import CandleBias
from trader.timing.candle_timing_agent import (
    AlignmentStatus,
    TimingAssessment,
    TimingDecision,
)
from trader.timing.central_directional_agent import DirectionalBias


def _make_timing(
    *,
    symbol: str = "TCS",
    decision: TimingDecision = TimingDecision.ENTER_NOW,
    candle_bias: CandleBias = CandleBias.BULLISH,
    central_direction: DirectionalBias = DirectionalBias.BULLISH,
    alignment: AlignmentStatus = AlignmentStatus.CONFIRMED,
    ts: datetime | None = None,
) -> TimingAssessment:
    now_ist = ts or datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    return TimingAssessment(
        symbol=symbol,
        as_of_ist=now_ist,
        decision=decision,
        alignment_status=alignment,
        central_direction=central_direction,
        candle_bias=candle_bias,
        timing_quality_score=0.82,
        price_action_score=0.75 if candle_bias == CandleBias.BULLISH else -0.75,
        entry_price=2400.0,
        invalidation_price=2364.0 if candle_bias == CandleBias.BULLISH else 2436.0,
        target_price=2472.0 if candle_bias == CandleBias.BULLISH else 2328.0,
        risk_reward_ratio=2.0,
        detected_patterns=("BULLISH_MARUBOZU",),
        timeframe_statuses={"1h": "VALID"},
        contradiction_reasons=(),
        rationale=("Unit test timing assessment",),
    )


def test_1515_shortened_1h_bar_clamped_to_1530_ist(tmp_path: Path) -> None:
    """15:15 IST NSE 1h bar closes at 15:30 IST (not 16:15 IST): invisible at 15:25, visible at 15:30."""
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 11, 20, 14, 15, tzinfo=IST_TZ),
                datetime(2026, 11, 20, 15, 15, tzinfo=IST_TZ),
            ],
            "open": [2380.0, 2390.0],
            "high": [2392.0, 2398.0],
            "low": [2375.0, 2388.0],
            "close": [2390.0, 2395.0],
            "volume": [10000, 8000],
        }
    )
    df.to_parquet(tmp_path / "TCS_1h.parquet", index=False)
    feed = ParquetCSVMarketFeed(data_dir=tmp_path)

    # At 15:25 IST, only the 14:15->15:15 bar is closed; the 15:15->15:30 bar is still forming
    ev_1525 = feed.get_bars("TCS", "1h", datetime(2026, 11, 20, 15, 25, tzinfo=IST_TZ))
    assert ev_1525.bar_count == 1
    assert ev_1525.frame["ts_close_ist"].iloc[-1].to_pydatetime() == datetime(2026, 11, 20, 15, 15, tzinfo=IST_TZ)

    # At 15:30 IST, the shortened 15:15->15:30 bar is closed and visible
    ev_1530 = feed.get_bars("TCS", "1h", datetime(2026, 11, 20, 15, 30, tzinfo=IST_TZ))
    assert ev_1530.bar_count == 2
    assert ev_1530.frame["ts_close_ist"].iloc[-1].to_pydatetime() == datetime(2026, 11, 20, 15, 30, tzinfo=IST_TZ)


def test_mid_session_authority_flip_never_traps_open_long_or_short() -> None:
    """Protective exits (EXIT_LONG / EXIT_SHORT) remain allowed even when human permission flips direction."""
    t0 = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    t1 = datetime(2026, 11, 20, 11, 15, tzinfo=IST_TZ)
    t_end = datetime(2026, 11, 20, 15, 30, tzinfo=IST_TZ)

    portfolio = IntradayPaperPortfolio(initial_capital=100_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    # 1. Open LONG under ALLOW_LONG
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-LONG",
            authority=AuthoritySource.HUMAN,
            scope="TCS",
            permission=PermissionType.ALLOW_LONG,
            valid_from=t0,
            expires_at=t_end,
            reason="Bullish morning bias",
        )
    )
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-LONG",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="TCS",
            permission=PermissionType.ALLOW_LONG,
            valid_from=t0,
            expires_at=t_end,
            reason="Central bullish",
        )
    )
    outcome_open, executed_open = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=t0,
        portfolio=portfolio,
        requested_quantity=5,
        price=2400.0,
        candle_assessment=_make_timing(ts=t0),
        stop_loss_price=2364.0,
        target_price=2472.0,
    )
    assert outcome_open.approved and executed_open
    assert portfolio.positions["TCS"].side == "LONG"
    assert portfolio.positions["TCS"].stop_loss == 2364.0
    assert portfolio.positions["TCS"].target == 2472.0

    # 2. Flip Human permission mid-session to ALLOW_SHORT -> EXIT_LONG must still be APPROVED!
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-FLIP-SHORT",
            authority=AuthoritySource.HUMAN,
            scope="TCS",
            permission=PermissionType.ALLOW_SHORT,
            valid_from=t1,
            expires_at=t_end,
            reason="Bearish reversal flip",
        )
    )
    outcome_exit, executed_exit = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.EXIT_LONG,
        current_time=t1,
        portfolio=portfolio,
        requested_quantity=5,
        price=2390.0,
    )
    assert outcome_exit.approved and executed_exit
    assert "TCS" not in portfolio.positions

    # 3. Now open SHORT under ALLOW_SHORT, flip Human to ALLOW_LONG -> EXIT_SHORT must still be APPROVED!
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-SHORT",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="TCS",
            permission=PermissionType.ALLOW_SHORT,
            valid_from=t1,
            expires_at=t_end,
            reason="Central bearish",
        )
    )
    t2 = t1 + timedelta(minutes=15)
    outcome_short, executed_short = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_SHORT,
        current_time=t2,
        portfolio=portfolio,
        requested_quantity=5,
        price=2390.0,
        candle_assessment=_make_timing(
            candle_bias=CandleBias.BEARISH,
            central_direction=DirectionalBias.BEARISH,
            ts=t2,
        ),
        stop_loss_price=2425.0,
        target_price=2320.0,
    )
    assert outcome_short.approved and executed_short
    assert portfolio.positions["TCS"].side == "SHORT"
    assert portfolio.positions["TCS"].stop_loss == 2425.0
    assert portfolio.positions["TCS"].target == 2320.0

    t3 = t2 + timedelta(minutes=15)
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-FLIP-LONG",
            authority=AuthoritySource.HUMAN,
            scope="TCS",
            permission=PermissionType.ALLOW_LONG,
            valid_from=t3,
            expires_at=t_end,
            reason="Reclaim bullish flip",
        )
    )
    outcome_cover, executed_cover = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.EXIT_SHORT,
        current_time=t3,
        portfolio=portfolio,
        requested_quantity=5,
        price=2360.0,
    )
    assert outcome_cover.approved and executed_cover
    assert "TCS" not in portfolio.positions
    ok, msg = portfolio.verify_ledger_integrity()
    assert ok, msg


def test_expired_human_instruction_fails_closed_and_allow_both_checks_candle_direction() -> None:
    t0 = datetime(2026, 11, 20, 9, 30, tzinfo=IST_TZ)
    t_exp = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    t_after = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    t_end = datetime(2026, 11, 20, 15, 30, tzinfo=IST_TZ)

    portfolio = IntradayPaperPortfolio(initial_capital=100_000.0)
    gate = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())

    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-SHORT-WINDOW",
            authority=AuthoritySource.HUMAN,
            scope="TCS",
            permission=PermissionType.ALLOW_LONG,
            valid_from=t0,
            expires_at=t_exp,
            reason="Expires at 10:00",
        )
    )
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-BOTH",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="TCS",
            permission=PermissionType.ALLOW_BOTH,
            valid_from=t0,
            expires_at=t_end,
            reason="Two-way allowed",
        )
    )

    # Expired human instruction must fail closed
    out_exp = gate.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=t_after,
        portfolio=portfolio,
        candle_assessment=_make_timing(ts=t_after),
        requested_quantity=5,
        entry_price=2400.0,
        stop_loss_price=2364.0,
    )
    assert not out_exp.approved
    assert any("HUMAN_INSTRUCTION_EXPIRED_OR_INVALID" in r for r in out_exp.rejection_reasons)

    # Valid human ALLOW_BOTH + Central ALLOW_BOTH, but ENTRY_LONG with BEARISH candle -> must reject!
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-BOTH",
            authority=AuthoritySource.HUMAN,
            scope="TCS",
            permission=PermissionType.ALLOW_BOTH,
            valid_from=t0,
            expires_at=t_end,
            reason="Both directions allowed",
        )
    )
    out_mismatch = gate.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=t_after,
        portfolio=portfolio,
        candle_assessment=_make_timing(
            candle_bias=CandleBias.BEARISH,
            central_direction=DirectionalBias.BEARISH,
            ts=t_after,
        ),
        requested_quantity=5,
        entry_price=2400.0,
        stop_loss_price=2364.0,
    )
    assert not out_mismatch.approved
    assert any("CANDLE_ACTION_DIRECTION_MISMATCH" in r for r in out_mismatch.rejection_reasons)


def test_strict_long_short_separation_and_double_entry_capital_conservation() -> None:
    """EXIT_LONG never opens a short; COVER_SHORT never opens a long; short P&L conserves capital."""
    t0 = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    t1 = datetime(2026, 11, 20, 11, 15, tzinfo=IST_TZ)
    p = IntradayPaperPortfolio(initial_capital=100_000.0)

    # EXIT_LONG when flat must be REJECTED (never opens a short)
    rej_exit = p.submit_order("TCS", OrderSide.SELL, 5, 2400.0, timestamp=t0, position_effect="EXIT_LONG")
    assert rej_exit.status == OrderStatus.REJECTED
    assert "TCS" not in p.positions

    # COVER_SHORT when flat must be REJECTED (never opens a long)
    rej_cover = p.submit_order("TCS", OrderSide.BUY, 5, 2400.0, timestamp=t0, position_effect="COVER_SHORT")
    assert rej_cover.status == OrderStatus.REJECTED
    assert "TCS" not in p.positions

    # OPEN_SHORT succeeds and preserves capital conservation while open and after cover
    ord_short = p.submit_order(
        "TCS",
        OrderSide.SELL,
        10,
        2400.0,
        stop_loss=2440.0,
        target=2320.0,
        timestamp=t0,
        position_effect="OPEN_SHORT",
    )
    assert ord_short.status == OrderStatus.FILLED
    assert p.positions["TCS"].side == "SHORT"
    ok_open, msg_open = p.verify_ledger_integrity()
    assert ok_open, msg_open

    # Attempting OPEN_LONG while SHORT is open must fail closed
    rej_long = p.submit_order("TCS", OrderSide.BUY, 5, 2380.0, timestamp=t1, position_effect="OPEN_LONG")
    assert rej_long.status == OrderStatus.REJECTED

    # Profitable COVER_SHORT at 2350 (+500 gross profit minus friction)
    ord_cover = p.submit_order("TCS", OrderSide.BUY, 10, 2350.0, timestamp=t1, position_effect="COVER_SHORT")
    assert ord_cover.status == OrderStatus.FILLED
    assert "TCS" not in p.positions
    assert len(p.closed_trades) == 1
    assert p.closed_trades[0]["side"] == "SHORT"
    assert p.closed_trades[0]["gross_pnl_inr"] == 500.0
    assert p.closed_trades[0]["net_pnl_inr"] > 450.0
    ok_closed, msg_closed = p.verify_ledger_integrity()
    assert ok_closed, msg_closed


def test_intrabar_ohlc_stop_first_precedence_and_nse_tick_rounding() -> None:
    assert round_to_nse_tick(2484.53) == 2484.55
    assert round_to_nse_tick(2484.52) == 2484.50

    t0 = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    t1 = datetime(2026, 11, 20, 11, 15, tzinfo=IST_TZ)
    p = IntradayPaperPortfolio(initial_capital=100_000.0)
    p.submit_order(
        "RELIANCE",
        OrderSide.BUY,
        10,
        1360.0,
        stop_loss=1345.03,  # rounds to 1345.05
        target=1390.02,  # rounds to 1390.00
        timestamp=t0,
        position_effect="OPEN_LONG",
        round_ticks=True,
    )
    assert p.positions["RELIANCE"].stop_loss == 1345.05
    assert p.positions["RELIANCE"].target == 1390.00

    # Both low <= stop_loss AND high >= target in the same bar -> conservative stop-first fills at 1345.05
    exits = p.update_candle_bar(
        symbol="RELIANCE",
        open_price=1360.0,
        high_price=1395.0,
        low_price=1340.0,
        close_price=1385.0,
        timestamp=t1,
    )
    assert len(exits) == 1
    assert exits[0].status == OrderStatus.FILLED
    assert exits[0].filled_price == 1345.05
    assert "RELIANCE" not in p.positions
    ok, msg = p.verify_ledger_integrity()
    assert ok, msg


def test_risk_engine_short_entry_stop_validation_and_friction_cash_cap() -> None:
    ts = datetime(2026, 11, 20, 10, 30, tzinfo=IST_TZ)
    p = IntradayPaperPortfolio(initial_capital=10_000.0)
    risk = DeterministicRiskEngine(
        max_risk_per_trade_pct=0.5,
        max_position_weight_pct=1.0,
        max_portfolio_exposure_pct=1.0,
    )

    # ENTRY_SHORT with stop_loss <= entry_price must reject
    bad_short = risk.evaluate_order(
        "TCS", "SELL", 5, 2000.0, 1950.0, p, ts, action_type="ENTRY_SHORT"
    )
    assert not bad_short.approved
    assert any("Invalid short stop-loss" in r for r in bad_short.rejection_reasons)

    # ENTRY_SHORT with stop_loss > entry_price and friction-aware cash cap sizes down from 5 to 4 so ledger never bounces
    good_short = risk.evaluate_order(
        "TCS",
        "SELL",
        5,
        2000.0,
        2050.0,
        p,
        ts,
        action_type="ENTRY_SHORT",
        include_entry_friction_in_cash_cap=True,
    )
    assert good_short.approved
    assert good_short.approved_quantity == 4
    order = p.submit_order(
        "TCS",
        OrderSide.SELL,
        good_short.approved_quantity,
        2000.0,
        stop_loss=2050.0,
        timestamp=ts,
        position_effect="OPEN_SHORT",
    )
    assert order.status == OrderStatus.FILLED
