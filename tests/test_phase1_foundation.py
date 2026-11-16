"""Automated test suite for Phase 1 — Data Foundation, Provenance, DuckDB/Parquet & Market Regime."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from backend.collectors.market_regime_collector import classify_market_regime
from backend.collectors.nse_market_collector import (
    _parse_optional_float,
    fetch_nse_indices_daily_archive,
    fetch_nse_sec_bhavdata_full,
)
from backend.config import (
    BASE_DIR,
    ENABLE_BIGQUERY,
    ENABLE_GEMINI_EXTRACTOR,
    is_nse_trading_day,
)
from backend.models import (
    CashAndBalanceSheet,
    CompanyOverview,
    DataRecord,
    EquityDailyBar,
    IndexDailyBar,
    RDAndInnovation,
    StockholderAnalysis,
)
from backend.storage import (
    DuckDBStore,
    ensure_storage_directories,
    evaluate_stock_data_quality,
)
from backend.technical_engine import build_technical_snapshot


def test_storage_directories_docs_and_cost_guards() -> None:
    dirs = ensure_storage_directories()
    for key in ("raw", "processed", "features", "predictions", "backtests"):
        assert key in dirs
        assert dirs[key].exists()

    # Mandatory cost guards must be disabled by default
    assert ENABLE_GEMINI_EXTRACTOR is False
    assert ENABLE_BIGQUERY is False

    # Mandatory Phase 1 documentation deliverables
    assert (BASE_DIR / "COST_CONTROL.md").exists()
    assert (BASE_DIR / "DATA_DICTIONARY.md").exists()
    assert (BASE_DIR / "DATA_SOURCES.md").exists()

    # Holiday & weekend calendar checks
    assert is_nse_trading_day(date(2026, 1, 26)) is False  # Republic Day
    assert is_nse_trading_day(date(2026, 11, 15)) is False  # Sunday
    assert is_nse_trading_day(date(2026, 11, 13)) is True  # Friday


def test_no_silent_zero_or_synthetic_high_low_fabrication() -> None:
    assert _parse_optional_float(None) is None
    assert _parse_optional_float("-") is None
    assert _parse_optional_float("NA") is None
    assert _parse_optional_float("") is None
    assert _parse_optional_float("1,234.50") == 1234.50

    # Close-only bars must NOT fabricate synthetic intraday High/Low > Close
    close_only_rows = [
        {"date": f"2026-10-{day:02d}", "close": 100.0 + day}
        for day in range(1, 25)
    ]
    snap = build_technical_snapshot(close_only_rows)
    assert snap.current_price == 124.0
    assert snap.day_high == 124.0
    assert snap.day_low == 123.0  # Open was previous close (123.0), Close was 124.0


def test_duckdb_and_parquet_roundtrip(tmp_path: Path) -> None:
    store = DuckDBStore(processed_dir=tmp_path, db_path=tmp_path / "test.duckdb")

    bars = [
        EquityDailyBar(
            symbol="RELIANCE",
            observation_date="2026-11-12",
            open=1350.0,
            high=1370.0,
            low=1345.0,
            close=1360.0,
            volume=5000000.0,
            deliverable_qty=None,
            delivery_pct=None,
            source="YahooFinance_Supplementary",
            publication_timestamp="2026-11-12T15:30:00+05:30",
            collection_timestamp="2026-11-15T20:00:00+05:30",
            ohlcv_validation_status="VERIFIED",
            delivery_validation_status="MISSING",
        ),
        EquityDailyBar(
            symbol="RELIANCE",
            observation_date="2026-11-13",
            open=1360.0,
            high=1375.0,
            low=1355.0,
            close=1365.9,
            vwap=1364.2,
            volume=6200000.0,
            deliverable_qty=3800000.0,
            delivery_pct=61.29,
            source="NSE_SecBhavdata_Full",
            publication_timestamp="2026-11-13T18:00:00+05:30",
            collection_timestamp="2026-11-15T20:00:00+05:30",
            ohlcv_validation_status="VERIFIED",
            delivery_validation_status="VERIFIED",
        ),
    ]
    count = store.upsert_equity_bars(bars)
    assert count == 2

    df = store.query_df(
        "SELECT observation_date, close, delivery_pct, delivery_validation_status FROM v_equities_daily ORDER BY observation_date"
    )
    assert len(df) == 2
    # Missing delivery_pct must remain NULL/NaN, never 0.0
    assert df.iloc[0]["delivery_validation_status"] == "MISSING"
    assert df.iloc[0].isna()["delivery_pct"]
    assert df.iloc[1]["delivery_validation_status"] == "VERIFIED"
    assert abs(float(df.iloc[1]["delivery_pct"]) - 61.29) < 1e-4

    idx_bar = IndexDailyBar(
        index_code="NIFTY_50",
        index_name="Nifty 50",
        category="broad",
        observation_date="2026-11-13",
        close=25144.3,
        change_pct=-0.61,
        pe_ratio=21.8,
        source="NSE_IndCloseAll_Archive",
        publication_timestamp="2026-11-13T18:00:00+05:30",
        collection_timestamp="2026-11-15T20:00:00+05:30",
    )
    assert store.upsert_index_bars([idx_bar]) == 1

    rec = DataRecord(
        symbol="RELIANCE",
        metric="delivery_pct",
        value=61.29,
        source="NSE_SecBhavdata_Full",
        observation_date="2026-11-13",
        publication_timestamp="2026-11-13T18:00:00+05:30",
        collection_timestamp="2026-11-15T20:00:00+05:30",
        validation_status="VERIFIED",
    )
    assert store.upsert_provenance_records([rec]) == 1


def test_market_regime_classifier_all_5_states() -> None:
    # 1. Strong Bull: steady uptrend > 50-DMA > 200-DMA, 20D return > 2%, low VIX
    bull_closes = [20000.0 + i * 35.0 for i in range(220)]
    snap_strong = classify_market_regime(
        nifty50_closes=bull_closes,
        india_vix_closes=[13.5, 13.2],
        observation_date="2026-11-13",
        breadth={"advance_count": 1500, "decline_count": 800, "advance_decline_ratio": 1.875},
    )
    assert snap_strong.regime == "Strong Bull"
    assert snap_strong.allow_aggressive_longs is True

    # 2. High Volatility: VIX >= 21.0
    snap_vol = classify_market_regime(
        nifty50_closes=bull_closes,
        india_vix_closes=[19.0, 23.4],
        observation_date="2026-11-13",
    )
    assert snap_vol.regime == "High Volatility"
    assert snap_vol.allow_aggressive_longs is False

    # 3. Bear: Nifty < 200-DMA and < 50-DMA
    bear_closes = [25000.0 - i * 30.0 for i in range(220)]
    snap_bear = classify_market_regime(
        nifty50_closes=bear_closes,
        india_vix_closes=[17.0, 17.5],
        observation_date="2026-11-13",
    )
    assert snap_bear.regime == "Bear"
    assert snap_bear.allow_aggressive_longs is False

    # 4. Weak Bull: Above 200-DMA, modest 20D gain (+0.5%)
    weak_bull_closes = [20000.0 + i * 20.0 for i in range(200)] + [24000.0 + i * 5.0 for i in range(25)]
    snap_weak = classify_market_regime(
        nifty50_closes=weak_bull_closes,
        india_vix_closes=[15.0, 15.2],
        observation_date="2026-11-13",
    )
    assert snap_weak.regime == "Weak Bull"
    assert snap_weak.allow_aggressive_longs is True

    # 5. Sideways: Above 200-DMA, slightly below 50-DMA with -1% 20D return
    sideways_closes = [20000.0 + i * 25.0 for i in range(200)] + [25000.0 - i * 12.0 for i in range(25)]
    snap_side = classify_market_regime(
        nifty50_closes=sideways_closes,
        india_vix_closes=[14.5, 14.6],
        observation_date="2026-11-13",
    )
    assert snap_side.regime == "Sideways"


def test_data_quality_gate_triggers_no_trade() -> None:
    ov = CompanyOverview(symbol="TESTCO", name="Test Incomplete Corp")
    cbs = CashAndBalanceSheet()
    shp = StockholderAnalysis()
    rd = RDAndInnovation()
    tech = build_technical_snapshot([])

    report, records = evaluate_stock_data_quality(
        symbol="TESTCO",
        overview=ov,
        annual_5=[],
        quarterly_5=[],
        cash_bs=cbs,
        shp=shp,
        rd=rd,
        tech=tech,
        price_rows=[],
    )
    assert report.trade_eligible is False
    assert report.forced_action == "NO TRADE"
    assert len(report.critical_failures) >= 3
    assert any(r.validation_status == "MISSING" for r in records)


def test_live_nse_bhavcopy_and_index_archive() -> None:
    core_set = {"RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE"}
    bars, breadth = fetch_nse_sec_bhavdata_full(symbols_filter=core_set)
    assert len(bars) == 5
    for b in bars:
        assert b.close is not None and b.close > 0
        assert b.vwap is not None and b.vwap > 0
        assert b.delivery_pct is not None and 0 < b.delivery_pct <= 100.0
        assert b.delivery_validation_status == "VERIFIED"
    assert breadth["advance_count"] > 0

    idx_bars = fetch_nse_indices_daily_archive()
    assert len(idx_bars) >= 8
    codes = {ib.index_code for ib in idx_bars}
    assert "NIFTY_50" in codes
    assert "INDIA_VIX" in codes
