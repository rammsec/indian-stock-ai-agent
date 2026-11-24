"""CYCLE-010 Phase 3 Acceptance & Unit Tests:
- Continuous bar-by-bar IntradayCandleMonitoringSession (09:15-15:30 IST)
- Real committed historical TCS 2026-10-29 1h session monitoring & post-session outcome review
- Next-bar-open causal execution, intrabar OHLC stops, trailing stop ratcheting, 15:05 REDUCING, 15:15 square-off
- Short lifecycle + mid-session Human Authority flip without trapped positions
- Post-session counterfactual outcome evaluation (MFE/MAE, missed move %, saved loss ₹/%)
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from trader.authority import AuthorityInstruction, AuthoritySource, PermissionType
from trader.config import IST_TZ
from trader.data.market_feed import ParquetCSVMarketFeed
from trader.evaluation.outcome_evaluator import evaluate_session_outcomes
from trader.timing.central_directional_agent import DirectionalBias
from trader.timing.monitoring_session import IntradayCandleMonitoringSession


def test_real_historical_tcs_2026_10_29_full_day_continuous_monitoring() -> None:
    """Run full-day continuous monitoring on real committed TCS_1h.parquet (2026-10-29)."""
    feed = ParquetCSVMarketFeed(data_dir=Path("data/intraday"))
    session = IntradayCandleMonitoringSession(feed=feed, initial_capital=100_000.0)

    res = session.run_session(
        symbol="TCS",
        session_date=date(2026, 10, 29),
        timeframe="1h",
        direction=DirectionalBias.BULLISH,
        human_permission=PermissionType.ALLOW_LONG,
        requested_quantity=10,
    )

    assert res.symbol == "TCS"
    assert res.paper_only is True
    assert res.confidence_calibration_status == "UNCALIBRATED"
    assert res.warmup_bars_count >= 50
    assert res.session_bars_count == 7
    # 100% of bars must be logged (never dropping WAIT/HOLD bars)
    assert len(res.decisions) == res.session_bars_count
    assert res.final_risk_state == "REDUCING"
    assert res.decisions[-1].ts_close_ist.endswith("15:30:00+05:30")
    assert res.decisions[-1].position_qty_after == 0  # Flat at session close
    assert res.ledger_integrity_ok is True
    assert res.risk_audit_chain_ok is True
    assert res.authority_audit_chain_ok is True
    assert res.outcome_report.total_decisions == res.session_bars_count


def test_continuous_session_next_bar_open_fill_hold_trailing_stop_and_1515_squareoff(tmp_path: Path) -> None:
    """Verify next-bar-open fill, HOLD trailing stop ratcheting, 15:15 mandatory square-off & Indian costs."""
    rows = []
    px = 1000.0
    # 4 prior warmup trading days (28 1h bars) + target session 2026-11-20 (7 1h bars)
    trading_days = [
        date(2026, 11, 16),
        date(2026, 11, 17),
        date(2026, 11, 18),
        date(2026, 11, 19),
        date(2026, 11, 20),
    ]
    bar_hours = [
        time(9, 15),
        time(10, 15),
        time(11, 15),
        time(12, 15),
        time(13, 15),
        time(14, 15),
        time(15, 15),
    ]
    for d in trading_days:
        for t_open in bar_hours:
            o = px
            c = px + 2.5
            h = c + 0.3
            l = o - 0.3
            v = 18000.0 if d == date(2026, 11, 20) else 10000.0
            rows.append(
                {
                    "timestamp": datetime.combine(d, t_open, tzinfo=IST_TZ),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )
            px = c

    pd.DataFrame(rows).to_parquet(tmp_path / "TCS_1h.parquet", index=False)
    feed = ParquetCSVMarketFeed(data_dir=tmp_path)
    session = IntradayCandleMonitoringSession(feed=feed, initial_capital=100_000.0)

    res = session.run_session(
        symbol="TCS",
        session_date=date(2026, 11, 20),
        timeframe="1h",
        direction=DirectionalBias.BULLISH,
        human_permission=PermissionType.ALLOW_LONG,
        requested_quantity=10,
    )

    assert len(res.decisions) == 7
    # Bar 0 (09:15->10:15) signals OPEN_LONG, queued for Bar 1 open (NOT filled at bar 0 close!)
    assert res.decisions[0].effective_call == "OPEN_LONG"
    assert res.decisions[0].position_qty_after == 0
    # Bar 1 (10:15->11:15) fills at bar 1 open and emits HOLD
    assert res.decisions[1].position_side_after == "LONG"
    assert res.decisions[1].position_qty_after > 0
    assert res.decisions[1].effective_call == "HOLD"

    # Verify trailing stop ratcheted upward during HOLD bars
    hold_stops = [d.stop_loss_after for d in res.decisions[1:5] if d.stop_loss_after is not None]
    assert len(hold_stops) >= 2
    assert hold_stops[-1] >= hold_stops[0]

    # Mandatory 15:15 IST square-off closes position cleanly with real statutory friction deducted
    assert res.decisions[-1].position_qty_after == 0
    assert len(res.closed_trades) >= 1
    assert res.closed_trades[0]["side"] == "LONG"
    assert res.total_friction_inr > 0.0
    assert res.final_risk_state == "REDUCING"
    assert res.ledger_integrity_ok is True
    assert res.risk_audit_chain_ok is True
    assert res.authority_audit_chain_ok is True


def test_continuous_session_short_lifecycle_and_mid_session_authority_flip(tmp_path: Path) -> None:
    """Verify SHORT entry, mid-session Human flip to ALLOW_LONG, and clean protective cover / square-off."""
    rows = []
    px = 1200.0
    trading_days = [
        date(2026, 11, 16),
        date(2026, 11, 17),
        date(2026, 11, 18),
        date(2026, 11, 19),
        date(2026, 11, 20),
    ]
    bar_hours = [time(9, 15), time(10, 15), time(11, 15), time(12, 15), time(13, 15), time(14, 15), time(15, 15)]
    for d in trading_days:
        for t_open in bar_hours:
            o = px
            c = px - 2.5
            h = o + 0.3
            l = c - 0.3
            v = 18000.0 if d == date(2026, 11, 20) else 10000.0
            rows.append(
                {
                    "timestamp": datetime.combine(d, t_open, tzinfo=IST_TZ),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )
            px = c

    pd.DataFrame(rows).to_parquet(tmp_path / "TCS_1h.parquet", index=False)
    feed = ParquetCSVMarketFeed(data_dir=tmp_path)
    session = IntradayCandleMonitoringSession(feed=feed, initial_capital=100_000.0)

    # Mid-session human flip to ALLOW_LONG at 12:00 IST while SHORT is open
    flip_ts = datetime(2026, 11, 20, 12, 0, tzinfo=IST_TZ)
    flip_instr = AuthorityInstruction(
        instruction_id="H-MID-FLIP",
        authority=AuthoritySource.HUMAN,
        scope="TCS",
        permission=PermissionType.ALLOW_LONG,
        valid_from=flip_ts,
        expires_at=datetime(2026, 11, 20, 16, 0, tzinfo=IST_TZ),
        reason="Mid-session flip to ALLOW_LONG",
    )

    res = session.run_session(
        symbol="TCS",
        session_date=date(2026, 11, 20),
        timeframe="1h",
        direction=DirectionalBias.BEARISH,
        human_permission=PermissionType.ALLOW_SHORT,
        mid_session_human_instructions={flip_ts: flip_instr},
        requested_quantity=10,
    )

    assert res.decisions[0].effective_call == "OPEN_SHORT"
    assert res.decisions[1].position_side_after == "SHORT"
    # Even after mid-session flip to ALLOW_LONG, protective 15:15 square-off closes SHORT cleanly!
    assert res.decisions[-1].position_qty_after == 0
    assert len(res.closed_trades) >= 1
    assert res.closed_trades[0]["side"] == "SHORT"
    assert res.ledger_integrity_ok is True
    assert res.risk_audit_chain_ok is True
    assert res.authority_audit_chain_ok is True


def test_post_session_outcome_evaluator_saved_loss_and_missed_move() -> None:
    """Verify evaluate_session_outcomes distinguishes saved loss on WAIT/REJECT/EXIT vs missed opportunity."""
    bars = pd.DataFrame(
        {
            "open": [1000.0, 998.0, 980.0, 960.0],
            "high": [1002.0, 1000.0, 985.0, 965.0],
            "low": [996.0, 978.0, 958.0, 950.0],
            "close": [999.0, 980.0, 960.0, 955.0],
        }
    )
    decisions = [
        {
            "bar_index": 0,
            "timestamp_ist": "2026-10-29T10:15:00+05:30",
            "call": "WAIT",
            "intended_direction": "LONG",
            "reference_price": 999.0,
            "stop_loss": 985.0,
            "target": 1030.0,
            "quantity": 10,
        },
        {
            "bar_index": 1,
            "timestamp_ist": "2026-10-29T11:15:00+05:30",
            "call": "EXIT_LONG",
            "intended_direction": "LONG",
            "reference_price": 980.0,
            "quantity": 10,
        },
    ]
    report = evaluate_session_outcomes(symbol="TCS", decision_records=decisions, session_bars=bars)
    assert report.total_decisions == 2
    assert report.per_decision_outcomes[0].verdict == "CORRECT_WAIT_SAVED_LOSS"
    assert report.per_decision_outcomes[0].saved_loss_inr > 0.0
    assert report.per_decision_outcomes[1].verdict == "PROTECTIVE_EXIT_SAVED_LOSS"
    assert report.per_decision_outcomes[1].saved_loss_inr > 0.0
    assert report.saved_loss_count == 2
    assert report.total_saved_loss_inr > 0.0
