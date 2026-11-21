"""Unit and regression tests for 5-minute candle validation and instrument mapping."""

from datetime import datetime, timedelta
import pytest

from trader.config import IST_TZ
from trader.data.schema import Candle5M
from trader.data.validator import validate_candle_series, detect_missing_candles
from trader.data.instruments import (
    INSTRUMENT_MASTER,
    get_instrument_by_symbol,
    get_instrument_by_isin,
    get_instrument_by_bse_code,
)
from trader.data.loader import generate_synthetic_5m_session


def test_candle_ohlc_invariants():
    """Assert invalid OHLC relationships raise ValueError."""
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Valid candle passes
    c = Candle5M(
        symbol="RELIANCE",
        exchange="NSE",
        timestamp=ts,
        open=2500.0,
        high=2520.0,
        low=2490.0,
        close=2510.0,
        volume=1000.0,
    )
    assert c.high == 2520.0

    # High < Open fails
    with pytest.raises(ValueError, match="High .* cannot be less than Open"):
        Candle5M(
            symbol="RELIANCE",
            exchange="NSE",
            timestamp=ts,
            open=2500.0,
            high=2490.0,
            low=2480.0,
            close=2485.0,
            volume=1000.0,
        )

    # Low > Close fails
    with pytest.raises(ValueError, match="Low .* cannot be greater than Close"):
        Candle5M(
            symbol="RELIANCE",
            exchange="NSE",
            timestamp=ts,
            open=2520.0,
            high=2520.0,
            low=2515.0,
            close=2510.0,
            volume=1000.0,
        )


def test_session_filtering_and_deduplication():
    """Validate outside-session bar filtering and duplicate timestamp dropping."""
    ts_pre = datetime(2026, 11, 20, 9, 0, tzinfo=IST_TZ)   # Before 09:15
    ts_open = datetime(2026, 11, 20, 9, 15, tzinfo=IST_TZ) # 09:15 (valid)
    ts_next = datetime(2026, 11, 20, 9, 20, tzinfo=IST_TZ) # 09:20 (valid)
    ts_post = datetime(2026, 11, 20, 15, 35, tzinfo=IST_TZ) # After 15:30

    candles = [
        Candle5M(symbol="TCS", exchange="NSE", timestamp=ts_pre, open=3500.0, high=3505.0, low=3495.0, close=3500.0, volume=500.0),
        Candle5M(symbol="TCS", exchange="NSE", timestamp=ts_open, open=3500.0, high=3510.0, low=3495.0, close=3505.0, volume=1200.0),
        Candle5M(symbol="TCS", exchange="NSE", timestamp=ts_open, open=3500.0, high=3510.0, low=3495.0, close=3505.0, volume=1200.0), # duplicate
        Candle5M(symbol="TCS", exchange="NSE", timestamp=ts_next, open=3505.0, high=3515.0, low=3500.0, close=3510.0, volume=800.0),
        Candle5M(symbol="TCS", exchange="NSE", timestamp=ts_post, open=3510.0, high=3510.0, low=3505.0, close=3505.0, volume=300.0),
    ]

    valid, report = validate_candle_series(candles)
    assert len(valid) == 2
    assert report.valid_bars_count == 2
    assert report.dropped_bars_count == 3
    assert valid[0].timestamp == ts_open
    assert valid[1].timestamp == ts_next


def test_missing_candles_detection():
    """Detect unexpected gap in candle series."""
    ts1 = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    ts2 = datetime(2026, 11, 20, 10, 20, tzinfo=IST_TZ)  # 20 min gap (missing 10:05, 10:10, 10:15)

    c1 = Candle5M(symbol="INFY", exchange="NSE", timestamp=ts1, open=1800.0, high=1805.0, low=1795.0, close=1802.0, volume=500.0)
    c2 = Candle5M(symbol="INFY", exchange="NSE", timestamp=ts2, open=1802.0, high=1810.0, low=1800.0, close=1808.0, volume=600.0)

    gaps = detect_missing_candles([c1, c2], expected_interval_minutes=5)
    assert len(gaps) == 1
    assert "20m gap" in gaps[0]


def test_synthetic_session_generation():
    """Verify that synthetic session produces exactly 75 valid 5-minute bars."""
    candles = generate_synthetic_5m_session(symbol="RELIANCE", exchange="NSE")
    assert len(candles) == 75
    valid, report = validate_candle_series(candles)
    assert report.is_valid is True
    assert len(valid) == 75
    assert valid[0].timestamp.time().strftime("%H:%M") == "09:15"
    assert valid[-1].timestamp.time().strftime("%H:%M") == "15:25"


def test_instrument_master_mappings():
    """Assert correct cross-referencing between NSE symbols, BSE codes, and ISINs."""
    reliance = get_instrument_by_symbol("RELIANCE")
    assert reliance is not None
    assert reliance.isin == "INE002A01018"
    assert reliance.bse_code == "500325"

    tcs = get_instrument_by_isin("INE467B01029")
    assert tcs is not None
    assert tcs.symbol == "TCS"

    bse_inst = get_instrument_by_bse_code("500180")
    assert bse_inst is not None
    assert bse_inst.symbol == "HDFCBANK"


def test_holiday_and_weekend_candle_rejection():
    """Verify that candles dated on exchange holidays or weekends are strictly rejected."""
    # 2026-01-26 is Republic Day (official NSE trading holiday)
    holiday_ts = datetime(2026, 1, 26, 10, 0, tzinfo=IST_TZ)
    # 2026-11-21 is Saturday (weekend)
    weekend_ts = datetime(2026, 11, 21, 10, 0, tzinfo=IST_TZ)

    candles = [
        Candle5M(symbol="SBIN", exchange="NSE", timestamp=holiday_ts, open=800.0, high=805.0, low=795.0, close=800.0, volume=1000.0),
        Candle5M(symbol="SBIN", exchange="NSE", timestamp=weekend_ts, open=800.0, high=805.0, low=795.0, close=800.0, volume=1000.0),
    ]

    valid, report = validate_candle_series(candles)
    assert len(valid) == 0
    assert report.is_valid is False
    assert len(report.holiday_rejections) == 2
    assert "2026-01-26 is an official exchange holiday" in report.holiday_rejections[0]
    assert "2026-11-21 is an official exchange holiday or weekend" in report.holiday_rejections[1]


def test_data_unavailable_error_on_real_mode_failure():
    """Verify that HISTORICAL_REAL mode NEVER silently substitutes synthetic data upon failure."""
    from trader.data.loader import load_historical_5m_candles, DataUnavailableError, DataSourceMode

    # Query an invalid symbol with cache disabled to trigger fetch error
    with pytest.raises(DataUnavailableError, match="Silent fallback to synthetic data is blocked under strict HISTORICAL_REAL mode"):
        load_historical_5m_candles(
            symbol="NONEXISTENT_TICKER_XYZ99",
            exchange="NSE",
            days=1,
            mode=DataSourceMode.HISTORICAL_REAL,
            use_cache=False,
        )


def test_data_provenance_sha256_verification():
    """Verify dataset provenance tracking and deterministic SHA-256 integrity."""
    from trader.data.loader import load_historical_5m_candles, DataSourceMode

    candles, prov = load_historical_5m_candles(
        symbol="TCS",
        exchange="NSE",
        mode=DataSourceMode.SYNTHETIC_TEST,
    )

    assert prov.source_mode == DataSourceMode.SYNTHETIC_TEST
    assert prov.symbol == "TCS"
    assert prov.bars_count == 75
    assert len(prov.sha256_hash) == 64
    assert prov.start_timestamp == candles[0].timestamp
    assert prov.end_timestamp == candles[-1].timestamp

