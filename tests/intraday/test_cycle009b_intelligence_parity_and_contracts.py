"""CYCLE-009B — Shared Intelligence Layer Numerical Parity & Contract Tests.

Covers:
1. Numerical parity (atol=1e-12, rtol=1e-12) across all 5 committed data/intraday/*_1h.parquet files.
2. FeatureRegistry immutability & duplicate registration guard.
3. Dual-timestamp OHLCVEvidenceFrame lookahead/integrity enforcement & Yahoo provenance cap.
4. NSE 15:15-15:30 1h resampler fix & backward MTF aligner (afternoon truncation invariance).
5. EvidenceBundle MappingProxyType immutability, UNCALIBRATED enforcement & deterministic hash.
6. Zero private `_`-helper cross-imports in trader/timing/, trader/data/loader_1h.py, trader/lab/data_service.py.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from trader.config import IST_TZ
from trader.data.resampler import resample_5m_candles
from trader.data.schema import Candle5M
from trader.features import indicators
from trader.intelligence import (
    DEFAULT_FEATURE_REGISTRY,
    DuplicateFeatureRegistrationError,
    EvidenceBundle,
    FeatureRegistry,
    FeatureSnapshot,
    FeatureSpec,
    FeatureValue,
    LookaheadViolationError,
    OHLCVEvidenceFrame,
    ProvenanceTier,
    align_htf_to_ltf_ist,
    build_evidence_bundle,
    build_ohlcv_evidence_frame,
    cap_provenance_tier,
    compute_atr_series,
    compute_cpr_pivots_dict,
    compute_ema_series,
    compute_latest_session_vwap,
    compute_macd_series,
    compute_rsi_series,
    compute_rvol_series,
    compute_session_vwap_series,
)
from trader.timing import candle_intelligence

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO_ROOT / "data" / "intraday"
PARQUET_FILES = sorted(PARQUET_DIR.glob("*_1h.parquet"))
SYMBOLS_1H = [p.stem.replace("_1h", "") for p in PARQUET_FILES]


def _load_1h_parquet(symbol: str) -> pd.DataFrame:
    path = PARQUET_DIR / f"{symbol}_1h.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST_TZ)
    return df.sort_values("timestamp").reset_index(drop=True)


def _make_5m_session(
    year: int = 2026,
    month: int = 1,
    day: int = 15,
    symbol: str = "RELIANCE",
    drop_times: set[tuple[int, int]] | None = None,
) -> list[Candle5M]:
    """Generate a full 75-bar NSE 5m session (09:15 to 15:25 IST inclusive)."""
    start = datetime(year, month, day, 9, 15, tzinfo=IST_TZ)
    bars: list[Candle5M] = []
    for i in range(75):
        ts = start + timedelta(minutes=5 * i)
        if drop_times and (ts.hour, ts.minute) in drop_times:
            continue
        base = 2500.0 + i * 0.5
        bars.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=ts,
                open=base,
                high=base + 2.0,
                low=base - 1.5,
                close=base + 0.75,
                volume=1000.0 + i * 10.0,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# 1. Numerical Parity across all 5 committed data/intraday/*_1h.parquet files
# ---------------------------------------------------------------------------


def test_five_committed_1h_parquets_present() -> None:
    assert len(PARQUET_FILES) == 5, f"Expected 5 *_1h.parquet files, found {PARQUET_FILES}"


@pytest.mark.parametrize("symbol", SYMBOLS_1H)
def test_atr_wilder_and_sma_parity_1h_parquet(symbol: str) -> None:
    df = _load_1h_parquet(symbol)

    # Wilder ATR(14) parity vs trader.features.indicators.compute_atr
    canonical_wilder = compute_atr_series(
        df["high"], df["low"], df["close"], period=14, method="wilder"
    )
    legacy_wilder = indicators.compute_atr(df["high"], df["low"], df["close"], period=14)
    np.testing.assert_allclose(
        canonical_wilder.to_numpy(dtype=float),
        legacy_wilder.to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )

    # SMA ATR(14) parity vs candle_intelligence._compute_atr & CandleIntelligenceEngine
    canonical_sma = compute_atr_series(
        df["high"], df["low"], df["close"], period=14, method="sma"
    )
    ci_atr = candle_intelligence._compute_atr(df, period=14)
    assert ci_atr is not None
    np.testing.assert_allclose(
        float(canonical_sma.iloc[-1]),
        float(ci_atr),
        atol=1e-12,
        rtol=1e-12,
    )

    as_of = df["timestamp"].iloc[-1].to_pydatetime() + timedelta(hours=1)
    tf_analysis = candle_intelligence.CandleIntelligenceEngine().analyze_timeframe(
        df, symbol=symbol, timeframe="1h", as_of_ist=as_of
    )
    assert tf_analysis.atr is not None
    np.testing.assert_allclose(
        float(canonical_sma.iloc[-1]),
        float(tf_analysis.atr),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("symbol", SYMBOLS_1H)
@pytest.mark.parametrize("span", [9, 21, 50])
def test_ema_parity_1h_parquet(symbol: str, span: int) -> None:
    df = _load_1h_parquet(symbol)
    canonical = compute_ema_series(df["close"], span=span)
    legacy = indicators.compute_ema(df["close"], span=span)
    np.testing.assert_allclose(
        canonical.to_numpy(dtype=float),
        legacy.to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("symbol", SYMBOLS_1H)
def test_session_vwap_parity_1h_parquet(symbol: str) -> None:
    df = _load_1h_parquet(symbol)

    # Full series parity vs indicators.compute_intraday_vwap
    canonical_series = compute_session_vwap_series(df, zero_vol_policy="typical_price")
    legacy_series = indicators.compute_intraday_vwap(df)
    np.testing.assert_allclose(
        canonical_series.to_numpy(dtype=float),
        legacy_series.to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )

    # Latest session VWAP parity vs candle_intelligence._compute_session_vwap
    c_vwap, c_pos, c_dist = compute_latest_session_vwap(df, zero_vol_policy="unavailable")
    ci_vwap, ci_pos, ci_dist = candle_intelligence._compute_session_vwap(df)
    assert c_pos == ci_pos
    if c_vwap is None:
        assert ci_vwap is None
        assert c_dist is None and ci_dist is None
    else:
        assert ci_vwap is not None and ci_dist is not None and c_dist is not None
        np.testing.assert_allclose(c_vwap, ci_vwap, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(c_dist, ci_dist, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("symbol", SYMBOLS_1H)
def test_rsi_legacy_50_parity_1h_parquet(symbol: str) -> None:
    df = _load_1h_parquet(symbol)
    canonical = compute_rsi_series(
        df["close"], period=14, warmup_fill=50.0, zero_loss_mode="legacy_50"
    )
    legacy = indicators.compute_rsi(df["close"], period=14)
    np.testing.assert_allclose(
        canonical.to_numpy(dtype=float),
        legacy.to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )


def test_rsi_strict_100_on_monotonic_rally() -> None:
    rally = pd.Series([100.0 + i * 1.5 for i in range(30)], dtype=float)
    strict = compute_rsi_series(rally, period=14, warmup_fill=50.0, zero_loss_mode="strict_100")
    legacy = compute_rsi_series(rally, period=14, warmup_fill=50.0, zero_loss_mode="legacy_50")

    # After 14-bar warmup, strict_100 must return 100.0 on zero-loss rally, while legacy_50 returns 50.0
    assert float(strict.iloc[-1]) == 100.0
    assert float(legacy.iloc[-1]) == 50.0


@pytest.mark.parametrize("symbol", SYMBOLS_1H)
def test_rvol_modes_parity_1h_parquet(symbol: str) -> None:
    df = _load_1h_parquet(symbol)
    vol = df["volume"].astype(float)

    # inclusive
    rvol_inc = compute_rvol_series(vol, window=20, mode="inclusive")
    expected_inc = vol / vol.rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    np.testing.assert_allclose(
        rvol_inc.to_numpy(dtype=float),
        expected_inc.to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )

    # prior_floored
    rvol_pf = compute_rvol_series(vol, window=20, mode="prior_floored")
    expected_pf = vol / vol.shift(1).rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    np.testing.assert_allclose(
        rvol_pf.to_numpy(dtype=float),
        expected_pf.to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )

    # prior_nonzero: verify exact semantics against reference loop
    rvol_pnz = compute_rvol_series(vol, window=14, mode="prior_nonzero")
    vals = vol.to_numpy(dtype=float)
    for idx in range(min(50, len(vals))):
        curr = vals[idx]
        if curr == 0.0:
            assert float(rvol_pnz.iloc[idx]) == 0.0
        else:
            prior = vals[max(0, idx - 14) : idx]
            nz = prior[prior > 0.0]
            if len(nz) == 0:
                assert np.isnan(rvol_pnz.iloc[idx])
            else:
                np.testing.assert_allclose(
                    float(rvol_pnz.iloc[idx]),
                    curr / float(np.mean(nz)),
                    atol=1e-12,
                    rtol=1e-12,
                )


def test_rvol_prior_nonzero_synthetic_zero_handling() -> None:
    s = pd.Series([100.0, 0.0, 200.0, 0.0, 300.0])
    out = compute_rvol_series(s, window=4, mode="prior_nonzero")
    assert np.isnan(out.iloc[0])
    assert float(out.iloc[1]) == 0.0
    np.testing.assert_allclose(float(out.iloc[2]), 200.0 / 100.0, atol=1e-12, rtol=1e-12)
    assert float(out.iloc[3]) == 0.0
    np.testing.assert_allclose(float(out.iloc[4]), 300.0 / 150.0, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("symbol", SYMBOLS_1H)
def test_macd_and_cpr_pivots_parity_1h_parquet(symbol: str) -> None:
    df = _load_1h_parquet(symbol)

    c_macd, c_sig, c_hist = compute_macd_series(df["close"])
    l_macd, l_sig, l_hist = indicators.compute_macd(df["close"])
    np.testing.assert_allclose(c_macd.to_numpy(dtype=float), l_macd.to_numpy(dtype=float), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(c_sig.to_numpy(dtype=float), l_sig.to_numpy(dtype=float), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(c_hist.to_numpy(dtype=float), l_hist.to_numpy(dtype=float), atol=1e-12, rtol=1e-12)

    prev_high = float(df["high"].iloc[-2])
    prev_low = float(df["low"].iloc[-2])
    prev_close = float(df["close"].iloc[-2])
    c_cpr = compute_cpr_pivots_dict(prev_high, prev_low, prev_close)
    l_cpr = indicators.compute_cpr_pivots(prev_high, prev_low, prev_close)
    assert c_cpr == l_cpr


# ---------------------------------------------------------------------------
# 2. Registry Immutability & Duplicate Guard
# ---------------------------------------------------------------------------


def test_registry_immutability_and_duplicate_guard() -> None:
    assert isinstance(DEFAULT_FEATURE_REGISTRY.specs, MappingProxyType)

    with pytest.raises(TypeError):
        DEFAULT_FEATURE_REGISTRY.specs["rogue_feature"] = DEFAULT_FEATURE_REGISTRY["ema_9"]  # type: ignore[index]

    with pytest.raises(FrozenInstanceError):
        DEFAULT_FEATURE_REGISTRY["ema_9"].formula_variant = "tampered"  # type: ignore[misc]

    # Attempting to register an existing feature_id raises DuplicateFeatureRegistrationError
    with pytest.raises(DuplicateFeatureRegistrationError):
        DEFAULT_FEATURE_REGISTRY.register(DEFAULT_FEATURE_REGISTRY["ema_9"])

    # Fresh registry also enforces duplicate guard
    spec = FeatureSpec(
        feature_id="custom_feat",
        formula_variant="v1",
        min_bars_required=5,
        output_dtype="float64",
        is_recursive=False,
        description="Test spec",
    )
    reg = FeatureRegistry([spec])
    assert "custom_feat" in reg
    with pytest.raises(DuplicateFeatureRegistrationError):
        reg.register(spec)


# ---------------------------------------------------------------------------
# 3. Dual-Timestamp OHLCVEvidenceFrame & Yahoo Provenance Cap
# ---------------------------------------------------------------------------


def _valid_ohlcv_df() -> pd.DataFrame:
    t0 = datetime(2026, 1, 15, 9, 15, tzinfo=IST_TZ)
    return pd.DataFrame(
        {
            "ts_open_ist": [t0, t0 + timedelta(hours=1)],
            "open": [100.0, 102.0],
            "high": [103.0, 105.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "volume": [1000.0, 1500.0],
        }
    )


def test_ohlcv_evidence_frame_causal_timestamps_and_yahoo_cap() -> None:
    df = _valid_ohlcv_df()
    # Last bar opens 10:15, closes 11:15 IST
    as_of = datetime(2026, 1, 15, 11, 15, tzinfo=IST_TZ)

    # Yahoo provenance cap: VERIFIED -> PARTIALLY_VERIFIED
    assert cap_provenance_tier("VERIFIED", "yahoo_finance_1h") == ProvenanceTier.PARTIALLY_VERIFIED
    assert cap_provenance_tier(ProvenanceTier.VERIFIED, "yahoo_finance_5m") == ProvenanceTier.PARTIALLY_VERIFIED

    frame = build_ohlcv_evidence_frame(
        df,
        symbol="RELIANCE",
        timeframe="1h",
        as_of_ist=as_of,
        provenance_status=ProvenanceTier.VERIFIED,
        source_name="yahoo_finance_1h",
    )
    assert isinstance(frame, OHLCVEvidenceFrame)
    assert frame.provenance_status == ProvenanceTier.PARTIALLY_VERIFIED
    assert frame.bar_count == 2

    # ts_init_ist > as_of_ist -> LookaheadViolationError
    with pytest.raises(LookaheadViolationError):
        build_ohlcv_evidence_frame(
            df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            ts_init_ist=as_of + timedelta(minutes=1),
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="yahoo_finance_1h",
        )

    # ts_close_ist > as_of_ist -> LookaheadViolationError
    with pytest.raises(LookaheadViolationError):
        build_ohlcv_evidence_frame(
            df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of - timedelta(minutes=1),
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="yahoo_finance_1h",
        )

    # ts_event_ist (ts_close_ist) > ts_init_ist -> LookaheadViolationError
    with pytest.raises(LookaheadViolationError):
        build_ohlcv_evidence_frame(
            df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of + timedelta(hours=1),
            ts_init_ist=datetime(2026, 1, 15, 10, 0, tzinfo=IST_TZ),
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="yahoo_finance_1h",
        )


def test_ohlcv_evidence_frame_validation_and_unavailable() -> None:
    as_of = datetime(2026, 1, 15, 12, 0, tzinfo=IST_TZ)

    # Naive as_of_ist rejected
    with pytest.raises(ValueError, match="Timezone-naive"):
        build_ohlcv_evidence_frame(
            _valid_ohlcv_df(),
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=datetime(2026, 1, 15, 12, 0),
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    # Naive column timestamps rejected
    naive_df = _valid_ohlcv_df()
    naive_df["ts_open_ist"] = naive_df["ts_open_ist"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="Timezone-naive"):
        build_ohlcv_evidence_frame(
            naive_df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    # Non-monotonic timestamps rejected
    non_mono = _valid_ohlcv_df().iloc[[1, 0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="Non-monotonic"):
        build_ohlcv_evidence_frame(
            non_mono,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    # Duplicate timestamps rejected
    dup_df = _valid_ohlcv_df().iloc[[0, 0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build_ohlcv_evidence_frame(
            dup_df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    # NaN / Inf rejected
    nan_df = _valid_ohlcv_df()
    nan_df.loc[0, "close"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        build_ohlcv_evidence_frame(
            nan_df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    inf_df = _valid_ohlcv_df()
    inf_df.loc[1, "high"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        build_ohlcv_evidence_frame(
            inf_df,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    # OHLC violation rejected (high < max(open, close))
    bad_ohlc = _valid_ohlcv_df()
    bad_ohlc.loc[0, "high"] = 99.5
    with pytest.raises(ValueError, match="OHLC violation"):
        build_ohlcv_evidence_frame(
            bad_ohlc,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="nse_official",
        )

    # Explicit UNAVAILABLE empty frame succeeds
    unavail = build_ohlcv_evidence_frame(
        None,
        symbol="RELIANCE",
        timeframe="5m",
        as_of_ist=as_of,
        provenance_status=ProvenanceTier.UNAVAILABLE,
        source_name="cache_5m",
        missing_reason="NO_5M_PARQUET",
    )
    assert unavail.provenance_status == ProvenanceTier.UNAVAILABLE
    assert unavail.bar_count == 0
    assert unavail.missing_reason == "NO_5M_PARQUET"

    # Empty frame with non-UNAVAILABLE status rejected
    with pytest.raises(ValueError, match="Empty OHLCV frame rejected"):
        build_ohlcv_evidence_frame(
            pd.DataFrame(),
            symbol="RELIANCE",
            timeframe="5m",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name="cache_5m",
        )


# ---------------------------------------------------------------------------
# 4. NSE 15:15–15:30 1h Resampler Fix & Backward MTF Aligner
# ---------------------------------------------------------------------------


def test_resampler_preserves_nse_1515_closing_1h_bar_and_drops_incomplete() -> None:
    full_bars = _make_5m_session()
    assert len(full_bars) == 75

    resampled = resample_5m_candles(full_bars, target_minutes=60, include_incomplete=False)
    assert len(resampled) == 7
    for bar in resampled[:6]:
        assert bar.source_bar_count == 12
        assert bar.expected_bar_count == 12
        assert bar.is_complete is True

    closing_bar = resampled[-1]
    assert closing_bar.timestamp.hour == 15 and closing_bar.timestamp.minute == 15
    assert closing_bar.source_bar_count == 3
    assert closing_bar.expected_bar_count == 3
    assert closing_bar.is_complete is True

    # Drop 1 bar from 10:15 bucket (11/12) and 1 bar from 15:15 bucket (2/3)
    partial_bars = _make_5m_session(drop_times={(10, 20), (15, 20)})
    resampled_strict = resample_5m_candles(partial_bars, target_minutes=60, include_incomplete=False)
    assert len(resampled_strict) == 5
    hours_strict = [(b.timestamp.hour, b.timestamp.minute) for b in resampled_strict]
    assert (10, 15) not in hours_strict
    assert (15, 15) not in hours_strict

    resampled_with_inc = resample_5m_candles(partial_bars, target_minutes=60, include_incomplete=True)
    assert len(resampled_with_inc) == 7
    inc_map = {(b.timestamp.hour, b.timestamp.minute): b for b in resampled_with_inc}
    assert inc_map[(10, 15)].is_complete is False
    assert inc_map[(10, 15)].source_bar_count == 11
    assert inc_map[(15, 15)].is_complete is False
    assert inc_map[(15, 15)].source_bar_count == 2


def test_align_htf_to_ltf_ist_backward_truncation_invariance() -> None:
    bars_5m = _make_5m_session()
    ltf_df = pd.DataFrame(
        {
            "ts_close_ist": [b.timestamp + timedelta(minutes=5) for b in bars_5m],
            "close": [b.close for b in bars_5m],
        }
    )

    bars_1h = resample_5m_candles(bars_5m, target_minutes=60, include_incomplete=False)
    htf_df = pd.DataFrame(
        {
            "ts_close_ist": [
                (b.timestamp + timedelta(minutes=15))
                if (b.timestamp.hour == 15 and b.timestamp.minute == 15)
                else (b.timestamp + timedelta(hours=1))
                for b in bars_1h
            ],
            "htf_signal": [float(i + 1) for i in range(len(bars_1h))],
        }
    )

    aligned_full = align_htf_to_ltf_ist(ltf_df, htf_df, htf_cols=["htf_signal"])

    # Before 10:15 IST (when first 1h bar closes), htf_signal MUST be NaN (no lookahead!)
    before_first_close = aligned_full[
        aligned_full["ts_close_ist"] < datetime(2026, 1, 15, 10, 15, tzinfo=IST_TZ)
    ]
    assert before_first_close["htf_signal"].isna().all()

    # Truncate both LTF and HTF at 13:30 IST (afternoon truncation invariance)
    cutoff = datetime(2026, 1, 15, 13, 30, tzinfo=IST_TZ)
    ltf_trunc = ltf_df[ltf_df["ts_close_ist"] <= cutoff].copy()
    htf_trunc = htf_df[htf_df["ts_close_ist"] <= cutoff].copy()
    aligned_trunc = align_htf_to_ltf_ist(ltf_trunc, htf_trunc, htf_cols=["htf_signal"])

    full_prefix = aligned_full[aligned_full["ts_close_ist"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_series_equal(
        full_prefix["htf_signal"],
        aligned_trunc["htf_signal"],
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 5. EvidenceBundle Immutability, UNCALIBRATED Enforcement & Deterministic Hash
# ---------------------------------------------------------------------------


def test_evidence_bundle_immutability_calibration_and_hash() -> None:
    as_of = datetime(2026, 1, 15, 11, 15, tzinfo=IST_TZ)
    frame = build_ohlcv_evidence_frame(
        _valid_ohlcv_df(),
        symbol="RELIANCE",
        timeframe="1h",
        as_of_ist=as_of,
        provenance_status=ProvenanceTier.VERIFIED,
        source_name="yahoo_finance_1h",
    )
    snap = FeatureSnapshot(
        symbol="RELIANCE",
        timeframe="1h",
        as_of_ist=as_of,
        features={
            "ema_9": FeatureValue("ema_9", "ewm_span_9_adjust_false", 102.5),
        },
    )

    bundle_a = build_evidence_bundle(
        symbol="RELIANCE",
        as_of_ist=as_of,
        market_frames={"1h": frame},
        feature_snapshots={"1h": snap},
        regime_evidence={"regime": "TREND"},
        directional_inputs={"bias": "BULLISH"},
        execution_context={"slippage_bps": 5},
    )
    bundle_b = build_evidence_bundle(
        symbol="RELIANCE",
        as_of_ist=as_of,
        market_frames={"1h": frame},
        feature_snapshots={"1h": snap},
        regime_evidence={"regime": "TREND"},
        directional_inputs={"bias": "BULLISH"},
        execution_context={"slippage_bps": 5},
    )

    assert isinstance(bundle_a, EvidenceBundle)
    assert bundle_a.confidence_calibration_status == "UNCALIBRATED"
    assert len(bundle_a.evidence_hash) == 64
    assert bundle_a.evidence_hash == bundle_b.evidence_hash

    # Changing an input changes the hash
    bundle_c = build_evidence_bundle(
        symbol="RELIANCE",
        as_of_ist=as_of,
        market_frames={"1h": frame},
        feature_snapshots={"1h": snap},
        regime_evidence={"regime": "RANGE"},
        directional_inputs={"bias": "BULLISH"},
        execution_context={"slippage_bps": 5},
    )
    assert bundle_c.evidence_hash != bundle_a.evidence_hash

    # MappingProxyType immutability
    for attr in (
        "market_frames",
        "feature_snapshots",
        "regime_evidence",
        "directional_inputs",
        "execution_context",
    ):
        proxy = getattr(bundle_a, attr)
        assert isinstance(proxy, MappingProxyType)
        with pytest.raises(TypeError):
            proxy["tamper"] = 1  # type: ignore[index]

    assert isinstance(snap.features, MappingProxyType)
    with pytest.raises(TypeError):
        snap.features["tamper"] = FeatureValue("x", "y", 1.0)  # type: ignore[index]

    # Reject CALIBRATED status
    with pytest.raises(ValueError, match="UNCALIBRATED"):
        build_evidence_bundle(
            symbol="RELIANCE",
            as_of_ist=as_of,
            confidence_calibration_status="CALIBRATED",
        )


# ---------------------------------------------------------------------------
# 6. Zero Private `_`-Helper Cross-Imports
# ---------------------------------------------------------------------------


def test_zero_private_helper_cross_imports() -> None:
    target_files = sorted((REPO_ROOT / "trader" / "timing").glob("*.py")) + [
        REPO_ROOT / "trader" / "data" / "loader_1h.py",
        REPO_ROOT / "trader" / "lab" / "data_service.py",
    ]
    assert len(target_files) >= 3

    violations: list[str] = []
    for path in target_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports private {alias.name} from {node.module}"
                        )

    assert not violations, "Private _-helper cross-imports found:\n" + "\n".join(violations)
