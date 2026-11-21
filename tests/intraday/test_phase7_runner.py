"""Phase 7 tests: ContinuousPaperTradingRunner, stale-feed detection, recovery, and daily journal."""

from datetime import datetime, timedelta
from pathlib import Path
import pytest

from trader.config import IST_TZ
from trader.data.loader import generate_synthetic_5m_session
from trader.runner import (
    BrokerExecutionDisabledError,
    ContinuousPaperTradingRunner,
    StaleFeedError,
)


def test_broker_execution_strictly_disabled():
    """Verify live broker execution cannot be enabled."""
    with pytest.raises(BrokerExecutionDisabledError, match="strictly DISABLED"):
        ContinuousPaperTradingRunner(symbols=["RELIANCE"], broker_execution_enabled=True)


def test_continuous_runner_session_lifecycle_and_recovery(tmp_path):
    """Test ingestion, stale-feed detection, automatic recovery, 15:15 square-off, and daily journal."""
    runner = ContinuousPaperTradingRunner(
        symbols=["RELIANCE"],
        initial_capital=1_000_000.0,
        max_feed_age_seconds=300.0,
        max_consecutive_errors=3,
    )

    session_date = datetime(2026, 11, 20, tzinfo=IST_TZ).date()
    candles = generate_synthetic_5m_session(symbol="RELIANCE", session_date=session_date, num_bars=75)

    # 1. Outside market hours -> SKIPPED
    early_ts = datetime(2026, 11, 20, 8, 45, tzinfo=IST_TZ)
    res_early = runner.step_cycle(lambda s: candles[:30], current_time=early_ts)
    assert res_early["status"] == "SKIPPED"

    # 2. Normal active market cycle at 11:45 IST (bar 30 is 11:40 IST -> 300s age = fresh)
    active_ts = candles[29].timestamp + timedelta(minutes=2)
    res_ok = runner.step_cycle(lambda s: candles[:30], current_time=active_ts)
    assert res_ok["status"] == "OK"
    assert runner.health.cycles_completed == 1
    assert runner.health.last_council_reassessment == active_ts

    # 3. Stale feed error (10 minutes old > 300s) -> increments consecutive_errors & stale_feed_events
    stale_ts = candles[29].timestamp + timedelta(minutes=12)
    res_stale = runner.step_cycle(lambda s: candles[:30], current_time=stale_ts)
    assert res_stale["status"] == "OK"  # Still below max_consecutive_errors=3
    assert runner.health.stale_feed_events == 1
    assert runner.health.consecutive_errors == 1

    # 4. Feed recovers on next bar -> total_recoveries increments, consecutive_errors resets to 0
    recovered_ts = candles[32].timestamp + timedelta(minutes=1)
    res_rec = runner.step_cycle(lambda s: candles[:33], current_time=recovered_ts)
    assert res_rec["status"] == "OK"
    assert runner.health.total_recoveries == 1
    assert runner.health.consecutive_errors == 0

    # 5. Repeated failures >= max_consecutive_errors -> HALTED
    def failing_fetch(_sym):
        raise ConnectionError("Feed socket dropped")

    for _ in range(3):
        out = runner.step_cycle(failing_fetch, current_time=recovered_ts)
    assert out["status"] == "HALTED"
    assert runner.health.is_running is False

    # 6. Generate daily journal
    journal_path = tmp_path / "daily_journal.md"
    journal_md = runner.generate_daily_journal(output_path=journal_path)
    assert journal_path.exists()
    assert "Daily Paper Trading Journal" in journal_md
    assert "Broker Integration**: DISABLED" in journal_md
    assert "Stale Feed Events**: 1" in journal_md
    assert "Auto-Recoveries**: 1" in journal_md
