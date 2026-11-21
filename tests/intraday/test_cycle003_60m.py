"""CYCLE-003 Regression Tests: 1h (60m) Extended History Loader, Replay & Cost Separation."""

from __future__ import annotations

from trader.config import LAB_DEFAULT_UNIVERSE
from trader.data.loader_1h import get_available_1h_dates, load_1h_session_with_warmup
from trader.data.schema import DataSourceMode
from trader.lab.replay_engine import STRATEGY_REGISTRY, HistoricalLabReplayEngine


def test_1h_extended_history_exceeds_60_sessions() -> None:
    elig, excl = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
    assert len(elig) >= 60, f"Expected >= 60 eligible 1h sessions, got {len(elig)}"
    w_bars, s_bars, prov = load_1h_session_with_warmup("RELIANCE", elig[-1], warmup_days=10)
    assert len(s_bars) >= 6
    assert len(w_bars) > 0
    assert prov.source_name == "yahoo_finance_1h"


def test_60m_replay_and_slippage_sensitivity() -> None:
    elig, _ = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
    d = elig[-1]
    engine = HistoricalLabReplayEngine()

    assert "STRAT-015-60M-TREND-BREAKOUT" in STRATEGY_REGISTRY
    assert "STRAT-016-60M-VWAP-MOMENTUM" in STRATEGY_REGISTRY
    assert "STRAT-017-60M-ORB-FILTERED" in STRATEGY_REGISTRY

    res = engine.run_date_range(
        d,
        d,
        strategy_id="STRAT-015-60M-TREND-BREAKOUT",
        data_mode=DataSourceMode.HISTORICAL_REAL,
        eligible_dates=[d],
        timeframe_minutes=60,
    )
    assert res["sessions_evaluated"] == 1
    assert res["timeframe_minutes"] == 60
    assert "total_brokerage_inr" in res
    assert "total_statutory_inr" in res
    assert "total_slippage_inr" in res
