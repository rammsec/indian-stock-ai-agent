"""Comprehensive regression tests for Historical AI Trading Lab (data quality, AI attribution, multi-day, and DB)."""

import json
from datetime import date, datetime
from pathlib import Path
import httpx
import pytest

from trader.agents.router import GeminiProvider, ModelRouter, TokenBudgetManager
from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, LAB_INITIAL_CAPITAL_INR
from trader.data.instruments import get_instrument_by_isin, get_instrument_by_symbol
from trader.data.loader import generate_synthetic_5m_session
from trader.data.schema import DataSourceMode
from trader.lab.data_service import (
    EXPECTED_SESSION_BARS,
    _classify_stock_quality,
    generate_pre_backtest_quality_report,
    load_session_with_warmup,
)
from trader.lab.database import HistoricalLabDatabase
from trader.lab.replay_engine import HistoricalLabReplayEngine, STRATEGY_REGISTRY


def test_five_stock_universe_and_birlacable_isin():
    """Verify fixed 5-stock universe and BIRLACABLE ISIN/BSE mapping."""
    assert LAB_DEFAULT_UNIVERSE == ("RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI")
    assert LAB_INITIAL_CAPITAL_INR == 50_000.0

    bc = get_instrument_by_symbol("BIRLACABLE")
    assert bc is not None
    assert bc.isin == "INE800A01015"
    assert bc.bse_code == "500060"
    assert get_instrument_by_isin("INE800A01015").symbol == "BIRLACABLE"


def test_strict_data_quality_incomplete_never_ready():
    """INCOMPLETE (<75 bars or missing intervals) must NEVER count as READY; partial dates labelled PARTIAL."""
    d = date(2026, 11, 20)
    full_bars = generate_synthetic_5m_session("BIRLACABLE", session_date=d, num_bars=75)
    partial_bars = full_bars[:73]  # 73/75 bars -> Incomplete

    q_full = _classify_stock_quality("BIRLACABLE", d, full_bars, 0, "Cached Historical", "HISTORICAL_REAL", "Parquet", "2026-11-20T16:00:00+05:30")
    q_inc = _classify_stock_quality("RELIANCE", d, partial_bars, 0, "Cached Historical", "HISTORICAL_REAL", "Parquet", "2026-11-20T16:00:00+05:30")
    q_empty = _classify_stock_quality("TCS", d, [], 0, "Unavailable", "HISTORICAL_REAL", "None", "2026-11-20T16:00:00+05:30")

    assert q_full.status == "Ready"
    assert q_full.candles_count == EXPECTED_SESSION_BARS
    assert q_inc.status == "Incomplete"
    assert q_empty.status == "Unavailable"

    # Official holiday must be classified EXCHANGE_HOLIDAY
    rep_holiday = generate_pre_backtest_quality_report(date(2026, 1, 26))
    assert rep_holiday.date_status == "EXCHANGE_HOLIDAY"
    assert rep_holiday.all_stocks_ready is False
    assert rep_holiday.ready_count == 0


def test_warmup_slicing_zero_future_leakage():
    """Ensure load_session_with_warmup never returns candles after target_date."""
    target = date(2026, 11, 18)
    warmup, session, quality = load_session_with_warmup("RELIANCE", target, warmup_days=5, mode=DataSourceMode.HISTORICAL_REAL)
    assert len(session) > 50
    assert quality.status in ("Ready", "Incomplete")
    assert quality.source_mode == "HISTORICAL_REAL"

    for c in warmup:
        assert c.timestamp.astimezone(IST_TZ).date() < target
    for c in session:
        assert c.timestamp.astimezone(IST_TZ).date() == target


def test_replay_determinism_and_no_fabricated_vix():
    """Replaying identical inputs produces identical ₹50,000 results and explicitly flags unavailable VIX/L1."""
    eng = HistoricalLabReplayEngine()
    target = date(2026, 11, 18)

    r1 = eng.run_historical_day(target, strategy_id="STRAT-001-VWAP-PULLBACK", data_mode=DataSourceMode.HISTORICAL_REAL)
    r2 = eng.run_historical_day(target, strategy_id="STRAT-001-VWAP-PULLBACK", data_mode=DataSourceMode.HISTORICAL_REAL)

    assert r1["run"]["starting_capital"] == 50_000.0
    assert r1["run"]["final_portfolio_value"] == r2["run"]["final_portfolio_value"]
    assert r1["run"]["total_trades"] == r2["run"]["total_trades"]
    assert r1["run"]["execution_config"]["execution_rule"] == "NEXT_BAR_OPEN"
    assert r1["benchmark"]["initial_capital_inr"] == 50_000.0
    assert len(STRATEGY_REGISTRY) >= 9

    # Check that indicator snapshots explicitly mark VIX and L1 microstructure as UNAVAILABLE
    if r1["decisions"]:
        snap = r1["decisions"][0]["indicator_snapshot"]
        assert snap["vix_status"] == "UNAVAILABLE"
        assert snap["microstructure_l1_status"] == "UNAVAILABLE"


def test_ai_attribution_caching_and_fallback_honesty(tmp_path, monkeypatch):
    """Verify genuine AI invocation caches by input_hash, records real cost, and never mislabels fallback as LLM."""
    db = HistoricalLabDatabase(db_path=tmp_path / "ai_test.db")
    eng = HistoricalLabReplayEngine()

    # 1. Without API key, AI_AGENT mode MUST label decisions DETERMINISTIC_FALLBACK with $0.00 cost
    router_no_key = ModelRouter(provider=GeminiProvider(api_key=""))
    res_fb = eng.run_historical_day(
        date(2026, 11, 18),
        strategy_id="AGENT-CENTRAL-COUNCIL",
        symbols=["BIRLACABLE"],
        data_mode=DataSourceMode.HISTORICAL_REAL,
        execution_mode="AI_AGENT",
        db=db,
        router=router_no_key,
    )
    assert res_fb["run"]["ai_api_cost_usd"] == 0.0
    for dec in res_fb["decisions"]:
        assert dec["execution_mode"] == "DETERMINISTIC_FALLBACK"
        assert dec["provider"] == "deterministic_fallback"

    # 2. With mocked Gemini key, records actual LLM provider, tokens, and cost
    call_counter = {"n": 0}

    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "Hold position; wait for VWAP confirmation."}]}}],
                "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30},
            }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            call_counter["n"] += 1
            return DummyResp()

    monkeypatch.setattr(httpx, "Client", DummyClient)
    db2 = HistoricalLabDatabase(db_path=tmp_path / "ai_test_llm.db")
    router_llm = ModelRouter(budget_manager=TokenBudgetManager(daily_budget_usd=1.0), provider=GeminiProvider(api_key="fake-key"))

    res_llm = eng.run_historical_day(
        date(2026, 11, 18),
        strategy_id="AGENT-AUTONOMOUS-TRADER",
        symbols=["BIRLACABLE"],
        data_mode=DataSourceMode.HISTORICAL_REAL,
        execution_mode="AI_AGENT",
        db=db2,
        router=router_llm,
    )
    assert call_counter["n"] > 0
    assert res_llm["run"]["ai_api_cost_usd"] > 0.0
    assert any(d["execution_mode"] == "LLM" and d["provider"] == "google-gemini" for d in res_llm["decisions"])

    # Re-running identical snapshot hits input_hash cache ($0 incremental cost)
    prev_calls = call_counter["n"]
    res_cached = eng.run_historical_day(
        date(2026, 11, 18),
        strategy_id="AGENT-AUTONOMOUS-TRADER",
        symbols=["BIRLACABLE"],
        data_mode=DataSourceMode.HISTORICAL_REAL,
        execution_mode="AI_AGENT",
        db=db2,
        router=router_llm,
    )
    assert call_counter["n"] == prev_calls
    assert res_cached["run"]["ai_api_cost_usd"] == 0.0


def test_multi_day_range_and_walk_forward():
    """Verify multi-day date range and walk-forward Train/Validation/Unseen splits on real data."""
    eng = HistoricalLabReplayEngine()
    res_range = eng.run_date_range(date(2026, 11, 16), date(2026, 11, 20), strategy_id="STRAT-001-VWAP-PULLBACK")
    assert res_range["sessions_evaluated"] == 5
    assert res_range["initial_capital_inr"] == 50_000.0
    assert len(res_range["daily_runs"]) == 5

    wf = eng.run_walk_forward(
        [date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 20)],
        strategy_id="STRAT-001-VWAP-PULLBACK",
    )
    assert wf["train"]["sessions_evaluated"] >= 1
    assert wf["validation"]["sessions_evaluated"] >= 1
    assert wf["unseen_test"]["sessions_evaluated"] >= 1


def test_lab_dashboard_files_and_json_contract():
    """Verify docs/lab/ static files and generated JSON index contract."""
    lab_dir = Path("docs/lab")
    assert (lab_dir / "index.html").exists()
    assert (lab_dir / "styles.css").exists()
    assert (lab_dir / "app.js").exists()
    idx_file = lab_dir / "data" / "lab_index.json"
    assert idx_file.exists()

    idx = json.loads(idx_file.read_text(encoding="utf-8"))
    assert idx["initial_capital_inr"] == 50_000.0
    assert idx["universe"] == ["RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI"]
    assert len(idx["strategies"]) == 9
    assert "coverage_summary" in idx
    assert "data_quality_analytics" in idx
    assert "multi_day_leaderboard" in idx
    assert "slippage_sensitivity" in idx
    assert idx["available_dates"][0]["date_status"] in ("FULLY_READY", "PARTIAL")


def test_phase8_data_quality_analytics_and_slippage_sensitivity():
    """Verify Phase 8 data-quality analytics and 5/10/20 bps slippage sensitivity."""
    from trader.lab.data_service import compute_data_quality_analytics

    dq = compute_data_quality_analytics(min_bars_for_research=70)
    assert dq["total_available_days"] >= 30
    assert dq["research_eligible_sessions"] >= 30
    assert dq["fully_ready_sessions"] == 0  # Honest: 4 stocks miss 1-2 closing bars in yfinance
    assert len(dq["coverage_by_stock"]) == 5

    eng = HistoricalLabReplayEngine()
    sens = eng.run_slippage_sensitivity(
        date(2026, 11, 16), date(2026, 11, 20), strategy_id="STRAT-001-VWAP-PULLBACK", slippage_bps_list=(5.0, 10.0, 20.0)
    )
    assert len(sens) == 3
    assert [s["slippage_bps"] for s in sens] == [5.0, 10.0, 20.0]


def test_phase9_cost_aware_filter_wrapper():
    """Verify CostAwareFilterWrapper filters out signals whose expected move is below the cost hurdle."""
    from trader.features.indicators import compute_all_features
    from trader.strategies.base import BaseStrategy, StrategyMetadata, StrategySignal
    from trader.strategies.cost_aware_wrapper import CostAwareFilterWrapper

    class DummyLowTargetStrat(BaseStrategy):
        def __init__(self):
            super().__init__(
                StrategyMetadata(
                    strategy_id="DUMMY-TEST",
                    name="Dummy",
                    origin_source="Test",
                    source_url="internal://test",
                    license_type="MIT",
                )
            )

        def evaluate_bar(self, symbol, features_df, current_bar_idx):
            r = features_df.iloc[current_bar_idx]
            ts = r["timestamp"].to_pydatetime() if hasattr(r["timestamp"], "to_pydatetime") else r["timestamp"]
            close = float(r["close"])
            # Tiny 0.2% target (below 1.2% cost hurdle)
            return StrategySignal("DUMMY-TEST", symbol, ts, "ENTER_LONG", close, close * 0.99, close * 1.002, 0.8, ["Test"])

    candles = generate_synthetic_5m_session("RELIANCE", base_price=2500.0, num_bars=40)
    df = compute_all_features(candles)
    wrapped = CostAwareFilterWrapper(DummyLowTargetStrat(), min_expected_move_pct=0.012)
    sig = wrapped.evaluate_bar("RELIANCE", df, 25)
    assert sig.action == "NO_SIGNAL"
    assert "Filtered:" in sig.rationale[0]


def test_phase10_daily_5m_archive_merge_dedup(tmp_path, monkeypatch):
    """Verify daily 5m archiver merges new bars without losing existing historical bars or creating duplicates."""
    import pandas as pd
    import scripts.archive_daily_5m as arch
    from trader.data.schema import DataProvenance

    monkeypatch.setattr(arch, "CACHE_DIR", tmp_path)
    day1 = date(2026, 11, 19)
    day2 = date(2026, 11, 20)
    c1 = generate_synthetic_5m_session("RELIANCE", session_date=day1, num_bars=10)
    c2 = generate_synthetic_5m_session("RELIANCE", session_date=day2, num_bars=10)

    # Seed existing parquet with day1
    pd.DataFrame([
        {"timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
        for c in c1
    ]).to_parquet(tmp_path / "RELIANCE_NSE_5m.parquet", index=False)

    # Mock fetch returning overlapping last 2 bars of day1 + all 10 bars of day2
    prov = DataProvenance(
        source_mode=DataSourceMode.HISTORICAL_REAL,
        source_name="MockYF",
        ingested_at=datetime.now(tz=IST_TZ),
        symbol="RELIANCE",
        exchange="NSE",
        bars_count=12,
        start_timestamp=c1[-2].timestamp,
        end_timestamp=c2[-1].timestamp,
        sha256_hash="0" * 64,
    )
    monkeypatch.setattr(arch, "load_historical_5m_candles", lambda **kw: (c1[-2:] + c2, prov))

    rep = arch.merge_and_archive_symbol("RELIANCE", days=5)
    assert rep["net_added"] == 10
    merged = pd.read_parquet(tmp_path / "RELIANCE_NSE_5m.parquet")
    assert len(merged) == 20



