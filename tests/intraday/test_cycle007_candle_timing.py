"""CYCLE-007 — Comprehensive Isolated Tests for Numerical Candlestick Intelligence,
Central Directional Agent, Independent Candle Timing Agent, Multi-Timeframe Honesty,
Lookahead Prevention, and End-to-End Paper-Trading Pipeline Integration.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.run_cycle_007_timing_dry_run import run_cycle_007_dry_run
from trader.authority.instructions import PermissionType
from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.paper.cycle006b import DEFAULT_CACHE_DIR, sha256_file
from trader.timing.candle_intelligence import (
    CandleBias,
    CandleIntelligenceEngine,
    CandlePattern,
    TrendStructure,
    compute_candle_anatomy,
    detect_candle_patterns,
    validate_ohlcv_frame,
)
from trader.timing.candle_timing_agent import (
    AlignmentStatus,
    CandleTimingAgent,
    CandleTimingConfig,
    TimingAssessment,
    TimingDecision,
)
from trader.timing.central_directional_agent import (
    CentralDirectionalAgent,
    CentralDirectionalConfig,
    DirectionalBias,
    DirectionalHypothesis,
)
from trader.timing.cycle007_pipeline import (
    CYCLE007_IMPLEMENTATION_MODULES,
    DEFAULT_HOLDOUT_REGISTRY_PATH,
    Cycle007Config,
    Cycle007Pipeline,
)


# ---------------------------------------------------------------------------
# Deterministic Test Fixture Helpers (strictly in-memory or tmp_path)
# ---------------------------------------------------------------------------


def _ist(y: int, m: int, d: int, hh: int = 9, mm: int = 15) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=IST_TZ)


def _make_ohlcv_df(
    rows: list[tuple[datetime, float, float, float, float, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def _make_bullish_uptrend_df(
    as_of_ist: datetime,
    *,
    n_bars: int = 24,
    step_minutes: int = 60,
    base_price: float = 1000.0,
    with_bullish_engulfing_end: bool = True,
) -> pd.DataFrame:
    """Create a clean, deterministic bullish uptrend ending at <= as_of_ist."""
    start_ts = as_of_ist - timedelta(minutes=step_minutes * (n_bars - 1))
    rows: list[tuple[datetime, float, float, float, float, float]] = []
    px = base_price
    for i in range(n_bars - 2):
        ts = start_ts + timedelta(minutes=step_minutes * i)
        o = px
        c = px + 2.0
        h = c + 0.6
        l = o - 0.5
        vol = 100_000.0 + (i % 3) * 5_000.0
        rows.append((ts, o, h, l, c, vol))
        px = c

    ts_prev = start_ts + timedelta(minutes=step_minutes * (n_bars - 2))
    ts_curr = start_ts + timedelta(minutes=step_minutes * (n_bars - 1))
    if with_bullish_engulfing_end:
        # Small pullback bearish candle followed by strong bullish engulfing breakout
        prev_o = px + 1.0
        prev_c = px - 1.5
        prev_h = prev_o + 0.5
        prev_l = prev_c - 0.5
        rows.append((ts_prev, prev_o, prev_h, prev_l, prev_c, 110_000.0))

        curr_o = prev_c - 0.3
        curr_c = prev_o + 4.5
        curr_h = curr_c + 0.3
        curr_l = curr_o - 0.2
        rows.append((ts_curr, curr_o, curr_h, curr_l, curr_c, 180_000.0))
    else:
        for ts in (ts_prev, ts_curr):
            o = px
            c = px + 2.5
            rows.append((ts, o, c + 0.3, o - 0.2, c, 150_000.0))
            px = c

    return _make_ohlcv_df(rows)


def _make_bearish_downtrend_df(
    as_of_ist: datetime,
    *,
    n_bars: int = 24,
    step_minutes: int = 60,
    base_price: float = 1200.0,
) -> pd.DataFrame:
    """Create a clean, deterministic bearish downtrend ending in BEARISH_ENGULFING."""
    start_ts = as_of_ist - timedelta(minutes=step_minutes * (n_bars - 1))
    rows: list[tuple[datetime, float, float, float, float, float]] = []
    px = base_price
    for i in range(n_bars - 2):
        ts = start_ts + timedelta(minutes=step_minutes * i)
        o = px
        c = px - 2.0
        h = o + 0.5
        l = c - 0.6
        vol = 100_000.0 + (i % 3) * 5_000.0
        rows.append((ts, o, h, l, c, vol))
        px = c

    ts_prev = start_ts + timedelta(minutes=step_minutes * (n_bars - 2))
    ts_curr = start_ts + timedelta(minutes=step_minutes * (n_bars - 1))
    # Small bullish bounce followed by decisive bearish engulfing breakdown
    prev_o = px - 1.0
    prev_c = px + 1.5
    rows.append((ts_prev, prev_o, prev_c + 0.5, prev_o - 0.5, prev_c, 105_000.0))

    curr_o = prev_c + 0.3
    curr_c = prev_o - 4.5
    rows.append((ts_curr, curr_o, curr_o + 0.2, curr_c - 0.3, curr_c, 190_000.0))
    return _make_ohlcv_df(rows)


def _make_indecisive_doji_df(
    as_of_ist: datetime,
    *,
    n_bars: int = 20,
    step_minutes: int = 60,
    base_price: float = 1000.0,
) -> pd.DataFrame:
    """Create a sideways range ending in a classic DOJI + INSIDE_BAR."""
    start_ts = as_of_ist - timedelta(minutes=step_minutes * (n_bars - 1))
    rows: list[tuple[datetime, float, float, float, float, float]] = []
    for i in range(n_bars - 2):
        ts = start_ts + timedelta(minutes=step_minutes * i)
        offset = 1.0 if i % 2 == 0 else -1.0
        o = base_price
        c = base_price + offset
        rows.append((ts, o, base_price + 2.5, base_price - 2.5, c, 100_000.0))

    ts_prev = start_ts + timedelta(minutes=step_minutes * (n_bars - 2))
    ts_curr = start_ts + timedelta(minutes=step_minutes * (n_bars - 1))
    rows.append((ts_prev, base_price - 1.0, base_price + 4.0, base_price - 4.0, base_price + 1.0, 100_000.0))
    # Tiny body inside prior bar's range -> DOJI and INSIDE_BAR
    rows.append((ts_curr, base_price, base_price + 2.0, base_price - 2.0, base_price + 0.1, 90_000.0))
    return _make_ohlcv_df(rows)


def _make_stage1_payloads(
    *,
    symbols: tuple[str, ...] = ("RELIANCE", "TCS", "INFY"),
    obs_date: str = "2026-11-18",
    generated_at: str = "2026-11-18T21:00:00+05:30",
    regime: str = "Bull",
    prob_map: dict[str, float] | None = None,
    ret_map: dict[str, float] | None = None,
    sent_map: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    probs = prob_map or {s: 0.65 for s in symbols}
    rets = ret_map or {s: 1.2 for s in symbols}
    sents = sent_map or {s: 0.25 for s in symbols}

    predictions = {
        "generated_at": generated_at,
        "predictions": [
            {
                "symbol": sym,
                "observation_date": obs_date,
                "prob_next_day_up": probs[sym],
                "expected_next_day_return_pct": rets[sym],
            }
            for sym in symbols
        ],
    }
    intelligence = {
        "model_version": "5.1.0",
        "generated_at": generated_at,
        "stocks": {
            sym: {
                "observation_date": obs_date,
                "news_sentiment_score": sents[sym],
            }
            for sym in symbols
        },
    }
    regime_payload = {
        "regime": regime,
        "observation_date": obs_date,
        "generated_at": generated_at,
    }
    return predictions, intelligence, regime_payload


def _make_hypothesis(
    *,
    symbol: str = "RELIANCE",
    direction: DirectionalBias = DirectionalBias.BULLISH,
    as_of_ist: datetime | None = None,
) -> DirectionalHypothesis:
    ts = as_of_ist or _ist(2026, 11, 19, 9, 15)
    return DirectionalHypothesis(
        symbol=symbol,
        direction=direction,
        confidence=0.75 if direction != DirectionalBias.NEUTRAL else 0.25,
        prob_next_day_up=0.68 if direction == DirectionalBias.BULLISH else (0.32 if direction == DirectionalBias.BEARISH else 0.50),
        expected_next_day_return_pct=1.4 if direction == DirectionalBias.BULLISH else (-1.4 if direction == DirectionalBias.BEARISH else 0.05),
        news_sentiment_score=0.2 if direction == DirectionalBias.BULLISH else (-0.2 if direction == DirectionalBias.BEARISH else 0.0),
        market_regime="Bull" if direction == DirectionalBias.BULLISH else "Bear",
        observation_date=date(2026, 11, 18),
        as_of_ist=ts,
        rationale=("unit-test directional hypothesis",),
    )


# ===========================================================================
# 1. Numerical Candlestick Anatomy & Patterns
# ===========================================================================


class TestNumericalCandleAnatomyAndPatterns:
    """Dimension 1: compute_candle_anatomy, detect_candle_patterns, structure & metrics."""

    def test_compute_candle_anatomy_bullish_and_bearish_and_zero_range(self) -> None:
        ts = _ist(2026, 11, 18, 14, 15)

        bull = compute_candle_anatomy(ts, open_px=100.0, high_px=110.0, low_px=98.0, close_px=108.0, volume=50_000.0)
        assert bull.is_bullish is True
        assert bull.is_bearish is False
        assert bull.candle_range == pytest.approx(12.0)
        assert bull.body_size == pytest.approx(8.0)
        assert bull.body_ratio == pytest.approx(8.0 / 12.0)
        assert bull.upper_wick_ratio == pytest.approx(2.0 / 12.0)
        assert bull.lower_wick_ratio == pytest.approx(2.0 / 12.0)
        assert bull.close_location_value == pytest.approx(10.0 / 12.0)
        assert bull.range_pct == pytest.approx(12.0)

        bear = compute_candle_anatomy(ts, open_px=108.0, high_px=110.0, low_px=98.0, close_px=100.0, volume=60_000.0)
        assert bear.is_bullish is False
        assert bear.is_bearish is True
        assert bear.body_ratio == pytest.approx(8.0 / 12.0)
        assert bear.close_location_value == pytest.approx(2.0 / 12.0)

        flat = compute_candle_anatomy(ts, open_px=100.0, high_px=100.0, low_px=100.0, close_px=100.0, volume=0.0)
        assert flat.candle_range == 0.0
        assert flat.body_ratio == 0.0
        assert flat.close_location_value == 0.5
        assert flat.is_bullish is False and flat.is_bearish is False

        with pytest.raises(ValueError, match="Timezone-naive"):
            compute_candle_anatomy(datetime(2026, 11, 18, 14, 15), 100.0, 105.0, 99.0, 104.0, 1000.0)

    def test_detect_all_canonical_candlestick_patterns(self) -> None:
        t1 = _ist(2026, 11, 18, 13, 15)
        t2 = _ist(2026, 11, 18, 14, 15)

        # BULLISH_ENGULFING
        prev_bear = compute_candle_anatomy(t1, 105.0, 106.0, 101.0, 102.0, 10_000.0)
        curr_bull = compute_candle_anatomy(t2, 101.5, 108.0, 101.0, 107.0, 20_000.0)
        assert CandlePattern.BULLISH_ENGULFING in detect_candle_patterns(curr_bull, prev_bear)

        # BEARISH_ENGULFING
        prev_bull = compute_candle_anatomy(t1, 102.0, 106.0, 101.0, 105.0, 10_000.0)
        curr_bear = compute_candle_anatomy(t2, 105.5, 106.0, 100.0, 100.5, 20_000.0)
        assert CandlePattern.BEARISH_ENGULFING in detect_candle_patterns(curr_bear, prev_bull)

        # HAMMER & BULLISH_PIN_BAR (long lower wick >= 60%, small body, close near high)
        hammer = compute_candle_anatomy(t2, 108.0, 110.0, 100.0, 109.5, 15_000.0)
        hammer_pats = detect_candle_patterns(hammer)
        assert CandlePattern.HAMMER in hammer_pats
        assert CandlePattern.BULLISH_PIN_BAR in hammer_pats

        # SHOOTING_STAR & BEARISH_PIN_BAR (long upper wick >= 60%, small body, close near low)
        shooting = compute_candle_anatomy(t2, 102.0, 110.0, 100.0, 100.5, 15_000.0)
        shooting_pats = detect_candle_patterns(shooting)
        assert CandlePattern.SHOOTING_STAR in shooting_pats
        assert CandlePattern.BEARISH_PIN_BAR in shooting_pats

        # DOJI
        doji = compute_candle_anatomy(t2, 105.0, 110.0, 100.0, 105.5, 12_000.0)
        assert CandlePattern.DOJI in detect_candle_patterns(doji)

        # BULLISH_MARUBOZU
        bull_maru = compute_candle_anatomy(t2, 100.0, 110.5, 99.8, 110.0, 25_000.0)
        assert CandlePattern.BULLISH_MARUBOZU in detect_candle_patterns(bull_maru)

        # BEARISH_MARUBOZU
        bear_maru = compute_candle_anatomy(t2, 110.0, 110.2, 99.5, 100.0, 25_000.0)
        assert CandlePattern.BEARISH_MARUBOZU in detect_candle_patterns(bear_maru)

        # INSIDE_BAR
        mother = compute_candle_anatomy(t1, 100.0, 112.0, 96.0, 108.0, 20_000.0)
        inside = compute_candle_anatomy(t2, 103.0, 109.0, 100.0, 105.0, 12_000.0)
        assert CandlePattern.INSIDE_BAR in detect_candle_patterns(inside, mother)

    def test_swing_structure_ema_vwap_atr_relvol_and_bounded_score(self) -> None:
        engine = CandleIntelligenceEngine(min_bars=5)
        cutoff = _ist(2026, 11, 18, 15, 15)

        up_df = _make_bullish_uptrend_df(cutoff, n_bars=24)
        up_res = engine.analyze_timeframe(up_df, symbol="RELIANCE", timeframe="1h", as_of_ist=cutoff)
        assert up_res.swing_structure == TrendStructure.UPTREND
        assert up_res.ema_alignment == "BULLISH"
        assert up_res.vwap_position == "ABOVE"
        assert up_res.atr is not None and up_res.atr > 0
        assert up_res.atr_pct is not None and up_res.atr_pct > 0
        assert up_res.relative_volume_ratio is not None and up_res.relative_volume_ratio > 1.0
        assert -1.0 <= up_res.price_action_score <= 1.0
        assert up_res.directional_lean == CandleBias.BULLISH

        down_df = _make_bearish_downtrend_df(cutoff, n_bars=24)
        down_res = engine.analyze_timeframe(down_df, symbol="RELIANCE", timeframe="1h", as_of_ist=cutoff)
        assert down_res.swing_structure == TrendStructure.DOWNTREND
        assert down_res.ema_alignment == "BEARISH"
        assert down_res.vwap_position == "BELOW"
        assert down_res.atr is not None and down_res.atr > 0
        assert -1.0 <= down_res.price_action_score <= 1.0
        assert down_res.directional_lean == CandleBias.BEARISH

        range_df = _make_indecisive_doji_df(cutoff, n_bars=20)
        range_res = engine.analyze_timeframe(range_df, symbol="RELIANCE", timeframe="1h", as_of_ist=cutoff)
        assert range_res.swing_structure == TrendStructure.RANGE
        assert -1.0 <= range_res.price_action_score <= 1.0
        assert CandlePattern.DOJI in range_res.detected_patterns
        assert CandlePattern.INSIDE_BAR in range_res.detected_patterns

    def test_engine_rejects_non_positive_swing_and_breakout_lookbacks(self) -> None:
        with pytest.raises(ValueError, match="swing_lookback"):
            CandleIntelligenceEngine(swing_lookback=0)
        with pytest.raises(ValueError, match="swing_lookback"):
            CandleIntelligenceEngine(swing_lookback=-3)
        with pytest.raises(ValueError, match="breakout_lookback"):
            CandleIntelligenceEngine(breakout_lookback=0)
        with pytest.raises(ValueError, match="breakout_lookback"):
            CandleIntelligenceEngine(breakout_lookback=-1)


# ===========================================================================
# 2. Strict Data Validation & Lookahead Prevention
# ===========================================================================


class TestDataValidationAndLookaheadPrevention:
    """Dimension 2: Fail-closed validation of OHLCV invariants, timestamps & zero lookahead."""

    def test_rejects_empty_insufficient_and_missing_columns(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        with pytest.raises(ValueError, match="Empty OHLCV"):
            validate_ohlcv_frame(pd.DataFrame(), as_of_ist=cutoff, symbol="RELIANCE")

        valid = _make_bullish_uptrend_df(cutoff, n_bars=10)
        with pytest.raises(ValueError, match="Insufficient bars"):
            validate_ohlcv_frame(valid.iloc[:3], as_of_ist=cutoff, symbol="RELIANCE", min_bars=5)

        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            validate_ohlcv_frame(valid.drop(columns=["volume"]), as_of_ist=cutoff, symbol="RELIANCE")

    def test_rejects_nan_inf_non_positive_prices_negative_volume_and_ohlc_violations(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        base = _make_bullish_uptrend_df(cutoff, n_bars=10)

        nan_df = base.copy()
        nan_df.loc[2, "close"] = np.nan
        with pytest.raises(ValueError, match="NaN OHLCV"):
            validate_ohlcv_frame(nan_df, as_of_ist=cutoff, symbol="RELIANCE")

        inf_df = base.copy()
        inf_df.loc[3, "high"] = math.inf
        with pytest.raises(ValueError, match="Non-finite"):
            validate_ohlcv_frame(inf_df, as_of_ist=cutoff, symbol="RELIANCE")

        zero_px = base.copy()
        zero_px.loc[1, "low"] = 0.0
        with pytest.raises(ValueError, match="Non-positive OHLC"):
            validate_ohlcv_frame(zero_px, as_of_ist=cutoff, symbol="RELIANCE")

        neg_vol = base.copy()
        neg_vol.loc[4, "volume"] = -10.0
        with pytest.raises(ValueError, match="Negative volume"):
            validate_ohlcv_frame(neg_vol, as_of_ist=cutoff, symbol="RELIANCE")

        bad_high = base.copy()
        bad_high.loc[5, "high"] = min(bad_high.loc[5, "open"], bad_high.loc[5, "close"]) - 1.0
        with pytest.raises(ValueError, match="OHLC invariant violation"):
            validate_ohlcv_frame(bad_high, as_of_ist=cutoff, symbol="RELIANCE")

        bad_low = base.copy()
        bad_low.loc[5, "low"] = max(bad_low.loc[5, "open"], bad_low.loc[5, "close"]) + 1.0
        with pytest.raises(ValueError, match="OHLC invariant violation"):
            validate_ohlcv_frame(bad_low, as_of_ist=cutoff, symbol="RELIANCE")

    def test_rejects_naive_duplicate_non_monotonic_and_future_timestamps(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        base = _make_bullish_uptrend_df(cutoff, n_bars=10)

        # Timezone-naive as_of_ist
        with pytest.raises(ValueError, match="Timezone-naive"):
            validate_ohlcv_frame(base, as_of_ist=datetime(2026, 11, 19, 9, 15), symbol="RELIANCE")

        # Timezone-naive bar timestamps (datetime64)
        naive_df = base.copy()
        naive_df["timestamp"] = naive_df["timestamp"].dt.tz_localize(None)
        with pytest.raises(ValueError, match="Timezone-naive"):
            validate_ohlcv_frame(naive_df, as_of_ist=cutoff, symbol="RELIANCE")

        # Mixed/naive object dtype timestamps
        obj_df = base.copy()
        obj_ts = list(obj_df["timestamp"])
        obj_ts[0] = datetime(2026, 11, 18, 9, 15)  # naive
        obj_df["timestamp"] = pd.Series(obj_ts, dtype=object)
        with pytest.raises(ValueError, match="Timezone-naive"):
            validate_ohlcv_frame(obj_df, as_of_ist=cutoff, symbol="RELIANCE")

        # Duplicate timestamps
        dup_df = base.copy()
        dup_df.loc[4, "timestamp"] = dup_df.loc[3, "timestamp"]
        with pytest.raises(ValueError, match="Duplicate timestamps"):
            validate_ohlcv_frame(dup_df, as_of_ist=cutoff, symbol="RELIANCE")

        # Non-monotonic timestamps
        swapped = base.copy()
        t_a, t_b = swapped.loc[2, "timestamp"], swapped.loc[3, "timestamp"]
        swapped.loc[2, "timestamp"] = t_b
        swapped.loc[3, "timestamp"] = t_a
        with pytest.raises(ValueError, match="Non-monotonic timestamps"):
            validate_ohlcv_frame(swapped, as_of_ist=cutoff, symbol="RELIANCE")

        # Future bar injection (timestamp > as_of_ist)
        future_df = base.copy()
        future_df.loc[len(future_df) - 1, "timestamp"] = cutoff + timedelta(minutes=1)
        with pytest.raises(ValueError, match="Lookahead violation"):
            validate_ohlcv_frame(future_df, as_of_ist=cutoff, symbol="RELIANCE")


# ===========================================================================
# 3. Stage 1 — Central Directional Agent
# ===========================================================================


class TestCentralDirectionalAgent:
    """Dimension 3: Classification accuracy, regime gating, freshness & fail-closed guards."""

    def test_emits_bullish_bearish_neutral_and_applies_bear_regime_gating(self) -> None:
        agent = CentralDirectionalAgent()
        as_of = _ist(2026, 11, 19, 9, 15)

        preds, intel, regime = _make_stage1_payloads(
            symbols=("RELIANCE", "TCS", "INFY"),
            regime="Bull",
            prob_map={"RELIANCE": 0.68, "TCS": 0.32, "INFY": 0.51},
            ret_map={"RELIANCE": 1.35, "TCS": -1.10, "INFY": 0.10},
            sent_map={"RELIANCE": 0.30, "TCS": -0.25, "INFY": 0.05},
        )
        res = agent.evaluate_from_payloads(
            symbols=("RELIANCE", "TCS", "INFY"),
            as_of_ist=as_of,
            predictions_payload=preds,
            intelligence_payload=intel,
            regime_payload=regime,
        )
        assert res.hypotheses["RELIANCE"].direction == DirectionalBias.BULLISH
        assert res.hypotheses["TCS"].direction == DirectionalBias.BEARISH
        assert res.hypotheses["INFY"].direction == DirectionalBias.NEUTRAL
        for h in res.hypotheses.values():
            assert 0.0 <= h.confidence <= 1.0
            assert len(h.rationale) >= 1

        # Bear regime gates strong bullish signal to NEUTRAL when block_bullish_in_bear_regime=True
        preds_b, intel_b, regime_b = _make_stage1_payloads(
            symbols=("RELIANCE",),
            regime="Bear",
            prob_map={"RELIANCE": 0.72},
            ret_map={"RELIANCE": 1.80},
            sent_map={"RELIANCE": 0.40},
        )
        res_bear = agent.evaluate_from_payloads(
            symbols=("RELIANCE",),
            as_of_ist=as_of,
            predictions_payload=preds_b,
            intelligence_payload=intel_b,
            regime_payload=regime_b,
        )
        assert res_bear.hypotheses["RELIANCE"].direction == DirectionalBias.NEUTRAL
        assert any("bearish regime" in r.lower() for r in res_bear.hypotheses["RELIANCE"].rationale)

    def test_fails_closed_on_naive_future_stale_spread_nan_and_missing_symbol(self) -> None:
        agent = CentralDirectionalAgent(
            CentralDirectionalConfig(max_staleness_hours=48.0, max_input_spread_hours=12.0)
        )
        as_of = _ist(2026, 11, 19, 9, 15)

        # 1. Timezone-naive as_of_ist
        preds, intel, regime = _make_stage1_payloads(symbols=("RELIANCE",))
        with pytest.raises(ValueError, match="Timezone-naive"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=datetime(2026, 11, 19, 9, 15),
                predictions_payload=preds,
                intelligence_payload=intel,
                regime_payload=regime,
            )

        # 2. Timezone-naive generated_at
        preds_naive, intel, regime = _make_stage1_payloads(
            symbols=("RELIANCE",), generated_at="2026-11-18T21:00:00"
        )
        with pytest.raises(ValueError, match="naive|timezone"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=as_of,
                predictions_payload=preds_naive,
                intelligence_payload=intel,
                regime_payload=regime,
            )

        # 3. Future-dated generated_at
        preds_fut, intel, regime = _make_stage1_payloads(symbols=("RELIANCE",))
        preds_fut["generated_at"] = "2026-11-19T09:16:00+05:30"
        with pytest.raises(ValueError, match="Future-dated input rejected"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=as_of,
                predictions_payload=preds_fut,
                intelligence_payload=intel,
                regime_payload=regime,
            )

        # 4. Future-dated observation_date
        preds_fobs, intel, regime = _make_stage1_payloads(symbols=("RELIANCE",))
        preds_fobs["predictions"][0]["observation_date"] = "2026-11-19"
        with pytest.raises(ValueError, match="Future-dated prediction observation_date"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=as_of,
                predictions_payload=preds_fobs,
                intelligence_payload=intel,
                regime_payload=regime,
            )

        # 5. Stale observation (> max_staleness_hours)
        preds_stale, intel_stale, regime_stale = _make_stage1_payloads(
            symbols=("RELIANCE",),
            obs_date="2026-11-10",
            generated_at="2026-11-10T21:00:00+05:30",
        )
        with pytest.raises(ValueError, match="Stale input rejected"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=as_of,
                predictions_payload=preds_stale,
                intelligence_payload=intel_stale,
                regime_payload=regime_stale,
            )

        # 6. Excessive cross-source spread (> max_input_spread_hours=12h)
        preds_sp, intel_sp, regime_sp = _make_stage1_payloads(
            symbols=("RELIANCE",),
            generated_at="2026-11-18T21:00:00+05:30",
        )
        intel_sp["generated_at"] = "2026-11-18T05:00:00+05:30"  # 16h spread > 12h
        with pytest.raises(ValueError, match="Conflicting input timestamps rejected"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=as_of,
                predictions_payload=preds_sp,
                intelligence_payload=intel_sp,
                regime_payload=regime_sp,
            )

        # 7. NaN / Inf values
        preds_nan, intel_nan, regime_nan = _make_stage1_payloads(symbols=("RELIANCE",))
        preds_nan["predictions"][0]["prob_next_day_up"] = float("nan")
        with pytest.raises(ValueError, match="finite|NaN"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE",),
                as_of_ist=as_of,
                predictions_payload=preds_nan,
                intelligence_payload=intel_nan,
                regime_payload=regime_nan,
            )

        # 8. Missing symbol
        preds_ms, intel_ms, regime_ms = _make_stage1_payloads(symbols=("RELIANCE",))
        with pytest.raises(ValueError, match="Mismatched instruments"):
            agent.evaluate_from_payloads(
                symbols=("RELIANCE", "TCS"),
                as_of_ist=as_of,
                predictions_payload=preds_ms,
                intelligence_payload=intel_ms,
                regime_payload=regime_ms,
            )

    def test_directional_hypothesis_confidence_calibration_status_and_validation(self) -> None:
        as_of = _ist(2026, 11, 19, 9, 15)
        agent = CentralDirectionalAgent()
        preds, intel, regime = _make_stage1_payloads(symbols=("RELIANCE", "TCS"))
        res = agent.evaluate_from_payloads(
            symbols=("RELIANCE", "TCS"),
            as_of_ist=as_of,
            predictions_payload=preds,
            intelligence_payload=intel,
            regime_payload=regime,
        )
        for hyp in (*res.hypotheses.values(), _make_hypothesis(as_of_ist=as_of)):
            assert hyp.confidence_calibration_status == "UNCALIBRATED"
            assert isinstance(hyp.confidence_calibration_note, str)
            assert hyp.confidence_calibration_note.strip() != ""
            note_lower = hyp.confidence_calibration_note.lower()
            assert "heuristic" in note_lower
            assert "not" in note_lower and "calibrated probability" in note_lower

            payload = hyp.to_dict()
            assert payload["confidence_calibration_status"] == "UNCALIBRATED"
            assert payload["confidence_calibration_note"] == hyp.confidence_calibration_note

        for bad_status in ("CALIBRATED", "", None):
            with pytest.raises(ValueError, match="confidence_calibration_status"):
                DirectionalHypothesis(
                    symbol="RELIANCE",
                    direction=DirectionalBias.BULLISH,
                    confidence=0.75,
                    prob_next_day_up=0.68,
                    expected_next_day_return_pct=1.4,
                    news_sentiment_score=0.2,
                    market_regime="Bull",
                    observation_date=date(2026, 11, 18),
                    as_of_ist=as_of,
                    rationale=("invalid calibration status test",),
                    confidence_calibration_status=bad_status,  # type: ignore[arg-type]
                )

        with pytest.raises(ValueError, match="confidence_calibration_note"):
            DirectionalHypothesis(
                symbol="RELIANCE",
                direction=DirectionalBias.BULLISH,
                confidence=0.75,
                prob_next_day_up=0.68,
                expected_next_day_return_pct=1.4,
                news_sentiment_score=0.2,
                market_regime="Bull",
                observation_date=date(2026, 11, 18),
                as_of_ist=as_of,
                rationale=("empty calibration note test",),
                confidence_calibration_note="   ",
            )


# ===========================================================================
# 4. Stage 2 — Independent Candle Timing Agent
# ===========================================================================


class TestIndependentCandleTimingAgent:
    """Dimension 4: Proves genuine independence from Central Directional Agent."""

    def test_enter_now_confirmed_when_bullish_central_and_bullish_candles_align(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent()
        hyp = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BULLISH, as_of_ist=cutoff)
        bars = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24)

        assessment = agent.evaluate(hypothesis=hyp, bars_primary=bars, as_of_ist=cutoff)
        assert assessment.decision == TimingDecision.ENTER_NOW
        assert assessment.alignment_status == AlignmentStatus.CONFIRMED
        assert assessment.candle_bias == CandleBias.BULLISH
        assert assessment.timing_quality_score >= agent.config.entry_quality_threshold
        assert assessment.entry_price is not None
        assert assessment.invalidation_price is not None
        assert assessment.target_price is not None
        assert assessment.invalidation_price < assessment.entry_price < assessment.target_price
        assert assessment.risk_reward_ratio is not None
        assert assessment.risk_reward_ratio >= agent.config.min_risk_reward_ratio

    def test_contradict_central_when_bullish_central_faces_strong_bearish_candles(self) -> None:
        """Hard independence test: MUST NOT blindly agree with Central BULLISH."""
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent()
        hyp = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BULLISH, as_of_ist=cutoff)
        bear_bars = _make_bearish_downtrend_df(cutoff - timedelta(hours=1), n_bars=24)

        assessment = agent.evaluate(hypothesis=hyp, bars_primary=bear_bars, as_of_ist=cutoff)
        assert assessment.decision == TimingDecision.CONTRADICT_CENTRAL
        assert assessment.alignment_status == AlignmentStatus.CONTRADICTED
        assert assessment.candle_bias == CandleBias.BEARISH
        assert assessment.timing_quality_score == 0.0
        assert len(assessment.contradiction_reasons) >= 1

        # Symmetric test: Central BEARISH contradicted by strong bullish candles
        hyp_bear = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BEARISH, as_of_ist=cutoff)
        bull_bars = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24)
        assessment_sym = agent.evaluate(hypothesis=hyp_bear, bars_primary=bull_bars, as_of_ist=cutoff)
        assert assessment_sym.decision == TimingDecision.CONTRADICT_CENTRAL
        assert assessment_sym.alignment_status == AlignmentStatus.CONTRADICTED
        assert assessment_sym.candle_bias == CandleBias.BULLISH

    def test_wait_when_central_bullish_but_candles_indecisive(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent()
        hyp = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BULLISH, as_of_ist=cutoff)
        doji_bars = _make_indecisive_doji_df(cutoff - timedelta(hours=1), n_bars=20)

        assessment = agent.evaluate(hypothesis=hyp, bars_primary=doji_bars, as_of_ist=cutoff)
        assert assessment.decision == TimingDecision.WAIT
        assert assessment.alignment_status == AlignmentStatus.UNCONFIRMED
        assert CandlePattern.DOJI in assessment.detected_patterns

    def test_no_trade_when_central_neutral_regardless_of_candle_bias(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent()
        hyp_neutral = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.NEUTRAL, as_of_ist=cutoff)

        bull_bars = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24)
        assmt_bull = agent.evaluate(hypothesis=hyp_neutral, bars_primary=bull_bars, as_of_ist=cutoff)
        assert assmt_bull.decision == TimingDecision.NO_TRADE
        assert assmt_bull.alignment_status == AlignmentStatus.NEUTRAL_CENTRAL
        assert assmt_bull.candle_bias == CandleBias.BULLISH

        bear_bars = _make_bearish_downtrend_df(cutoff - timedelta(hours=1), n_bars=24)
        assmt_bear = agent.evaluate(hypothesis=hyp_neutral, bars_primary=bear_bars, as_of_ist=cutoff)
        assert assmt_bear.decision == TimingDecision.NO_TRADE
        assert assmt_bear.alignment_status == AlignmentStatus.NEUTRAL_CENTRAL
        assert assmt_bear.candle_bias == CandleBias.BEARISH

    def test_reject_on_corrupt_future_missing_data_or_excessive_atr_volatility(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent(config=CandleTimingConfig(max_atr_pct=3.0))
        hyp = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BULLISH, as_of_ist=cutoff)

        # Missing bars
        r_missing = agent.evaluate(hypothesis=hyp, bars_primary=None, as_of_ist=cutoff)
        assert r_missing.decision == TimingDecision.REJECT
        assert r_missing.alignment_status == AlignmentStatus.REJECTED

        # Corrupt/future bar
        future_bars = _make_bullish_uptrend_df(cutoff + timedelta(hours=2), n_bars=12)
        r_future = agent.evaluate(hypothesis=hyp, bars_primary=future_bars, as_of_ist=cutoff)
        assert r_future.decision == TimingDecision.REJECT
        assert r_future.alignment_status == AlignmentStatus.REJECTED
        assert any("Lookahead violation" in msg for msg in r_future.rationale)

        # Excessive ATR volatility (> 3.0%)
        start_ts = cutoff - timedelta(hours=15)
        volatile_rows = []
        for i in range(15):
            ts = start_ts + timedelta(hours=i)
            o = 100.0
            h = 112.0  # 20% bar range -> ATR% >> 3.0%
            l = 92.0
            c = 105.0
            volatile_rows.append((ts, o, h, l, c, 100_000.0))
        r_vol = agent.evaluate(hypothesis=hyp, bars_primary=_make_ohlcv_df(volatile_rows), as_of_ist=cutoff)
        assert r_vol.decision == TimingDecision.REJECT
        assert r_vol.alignment_status == AlignmentStatus.REJECTED
        assert any("Volatility guard" in msg for msg in r_vol.rationale)

    def test_config_rejects_negative_atr_buffer_and_min_relative_volume(self) -> None:
        with pytest.raises(ValueError, match="atr_stop_buffer_multiplier"):
            CandleTimingConfig(atr_stop_buffer_multiplier=-0.1)
        with pytest.raises(ValueError, match="min_relative_volume_for_entry"):
            CandleTimingConfig(min_relative_volume_for_entry=-0.01)

    def test_reject_on_naive_or_future_hypothesis_as_of_ist(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent()
        bars = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24)

        # Timezone-naive hypothesis.as_of_ist must fail closed even when eval as_of_ist is aware
        hyp_naive = _make_hypothesis(
            symbol="RELIANCE",
            direction=DirectionalBias.BULLISH,
            as_of_ist=datetime(2026, 11, 19, 9, 15),
        )
        res_naive = agent.evaluate(hypothesis=hyp_naive, bars_primary=bars, as_of_ist=cutoff)
        assert res_naive.decision == TimingDecision.REJECT
        assert res_naive.alignment_status == AlignmentStatus.REJECTED
        assert any("hypothesis.as_of_ist" in msg for msg in res_naive.rationale)

        # Future hypothesis.as_of_ist (> eval_ist) must fail closed as a lookahead violation
        hyp_future = _make_hypothesis(
            symbol="RELIANCE",
            direction=DirectionalBias.BULLISH,
            as_of_ist=cutoff + timedelta(minutes=1),
        )
        res_future = agent.evaluate(hypothesis=hyp_future, bars_primary=bars, as_of_ist=cutoff)
        assert res_future.decision == TimingDecision.REJECT
        assert res_future.alignment_status == AlignmentStatus.REJECTED
        assert any("Lookahead violation" in msg for msg in res_future.rationale)

    def test_timing_assessment_central_confidence_calibration_status_and_validation(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        agent = CandleTimingAgent()
        hyp_bull = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BULLISH, as_of_ist=cutoff)
        hyp_neut = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.NEUTRAL, as_of_ist=cutoff)

        assessments = [
            agent.evaluate(
                hypothesis=hyp_bull,
                bars_primary=_make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24),
                as_of_ist=cutoff,
            ),
            agent.evaluate(
                hypothesis=hyp_bull,
                bars_primary=_make_bearish_downtrend_df(cutoff - timedelta(hours=1), n_bars=24),
                as_of_ist=cutoff,
            ),
            agent.evaluate(
                hypothesis=hyp_bull,
                bars_primary=_make_indecisive_doji_df(cutoff - timedelta(hours=1), n_bars=20),
                as_of_ist=cutoff,
            ),
            agent.evaluate(
                hypothesis=hyp_neut,
                bars_primary=_make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24),
                as_of_ist=cutoff,
            ),
            agent.evaluate(hypothesis=hyp_bull, bars_primary=None, as_of_ist=cutoff),
        ]
        for assmt in assessments:
            assert assmt.central_confidence_calibration_status == "UNCALIBRATED"
            assert assmt.to_dict()["central_confidence_calibration_status"] == "UNCALIBRATED"

        for bad_status in ("CALIBRATED", ""):
            with pytest.raises(ValueError, match="central_confidence_calibration_status"):
                TimingAssessment(
                    symbol="RELIANCE",
                    as_of_ist=cutoff,
                    decision=TimingDecision.WAIT,
                    alignment_status=AlignmentStatus.UNCONFIRMED,
                    central_direction=DirectionalBias.BULLISH,
                    candle_bias=CandleBias.NEUTRAL,
                    timing_quality_score=0.5,
                    price_action_score=0.1,
                    entry_price=None,
                    invalidation_price=None,
                    target_price=None,
                    risk_reward_ratio=None,
                    detected_patterns=(),
                    timeframe_statuses={"1h": "AVAILABLE", "15m": "UNAVAILABLE", "5m": "UNAVAILABLE"},
                    contradiction_reasons=(),
                    rationale=("invalid calibration status test",),
                    central_confidence_calibration_status=bad_status,
                )


# ===========================================================================
# 5. Multi-Timeframe Handling & Provenance Honesty
# ===========================================================================


class TestMultiTimeframeAndProvenanceHonesty:
    """Dimension 5: Honest UNAVAILABLE reporting, clean 15m/5m confirmation, fail-closed on corrupt MTF."""

    def test_absent_lower_timeframes_reported_unavailable_with_zero_fabricated_bars(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        engine = CandleIntelligenceEngine(min_bars=5)
        bars_1h = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24, step_minutes=60)

        report = engine.analyze(
            symbol="RELIANCE",
            as_of_ist=cutoff,
            bars_primary=bars_1h,
            primary_timeframe="1h",
            bars_15m=None,
            bars_5m=None,
        )
        assert report.primary.timeframe_status == "AVAILABLE"
        assert report.confirming_timeframes["15m"].timeframe_status == "UNAVAILABLE"
        assert report.confirming_timeframes["15m"].bar_count == 0
        assert report.confirming_timeframes["5m"].timeframe_status == "UNAVAILABLE"
        assert report.confirming_timeframes["5m"].bar_count == 0
        assert report.composite_price_action_score == pytest.approx(report.primary.price_action_score)

    def test_clean_15m_and_5m_confirmation_frames_evaluated_and_blended(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        engine = CandleIntelligenceEngine(min_bars=5)
        bars_1h = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24, step_minutes=60)
        bars_15m = _make_bullish_uptrend_df(cutoff - timedelta(minutes=15), n_bars=20, step_minutes=15)
        bars_5m = _make_bullish_uptrend_df(cutoff - timedelta(minutes=5), n_bars=20, step_minutes=5)

        report = engine.analyze(
            symbol="RELIANCE",
            as_of_ist=cutoff,
            bars_primary=bars_1h,
            primary_timeframe="1h",
            bars_15m=bars_15m,
            bars_5m=bars_5m,
        )
        assert report.confirming_timeframes["15m"].timeframe_status == "AVAILABLE"
        assert report.confirming_timeframes["15m"].bar_count == 20
        assert report.confirming_timeframes["5m"].timeframe_status == "AVAILABLE"
        assert report.confirming_timeframes["5m"].bar_count == 20

        expected_avg = (
            report.confirming_timeframes["15m"].price_action_score
            + report.confirming_timeframes["5m"].price_action_score
        ) / 2.0
        expected_comp = max(-1.0, min(1.0, 0.70 * report.primary.price_action_score + 0.30 * expected_avg))
        assert report.composite_price_action_score == pytest.approx(expected_comp)

    def test_corrupt_or_future_lower_timeframe_fails_closed(self) -> None:
        cutoff = _ist(2026, 11, 19, 9, 15)
        bars_1h = _make_bullish_uptrend_df(cutoff - timedelta(hours=1), n_bars=24, step_minutes=60)
        future_5m = _make_bullish_uptrend_df(cutoff + timedelta(minutes=10), n_bars=12, step_minutes=5)
        corrupt_15m = _make_bullish_uptrend_df(cutoff - timedelta(minutes=15), n_bars=12, step_minutes=15)
        corrupt_15m.loc[2, "close"] = np.nan

        engine = CandleIntelligenceEngine(min_bars=5)
        with pytest.raises(ValueError, match="Lookahead violation"):
            engine.analyze(
                symbol="RELIANCE",
                as_of_ist=cutoff,
                bars_primary=bars_1h,
                bars_5m=future_5m,
            )
        with pytest.raises(ValueError, match="NaN OHLCV"):
            engine.analyze(
                symbol="RELIANCE",
                as_of_ist=cutoff,
                bars_primary=bars_1h,
                bars_15m=corrupt_15m,
            )

        agent = CandleTimingAgent()
        hyp = _make_hypothesis(symbol="RELIANCE", direction=DirectionalBias.BULLISH, as_of_ist=cutoff)
        assmt = agent.evaluate(
            hypothesis=hyp,
            bars_primary=bars_1h,
            bars_5m=future_5m,
            as_of_ist=cutoff,
        )
        assert assmt.decision == TimingDecision.REJECT
        assert assmt.alignment_status == AlignmentStatus.REJECTED


# ===========================================================================
# 6. End-to-End Paper-Trading Pipeline Integration
# ===========================================================================


class TestCycle007EndToEndPipelineIntegration:
    """Dimension 6: Real-data 2026-11-19 dry run, synthetic bullish execution fixture,
    deterministic replay, and holdout registry immutability.
    """

    def test_genuine_committed_2026_11_19_real_data_dry_run_and_holdout_immutability(
        self, tmp_path: Path
    ) -> None:
        holdout_sha_before = sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)

        json_out = tmp_path / "cycle_007_report.json"
        md_out = tmp_path / "cycle_007_report.md"
        report = run_cycle_007_dry_run(
            session_date=date(2026, 11, 19),
            snapshot_dir=tmp_path / "snapshot",
            json_out=json_out,
            md_out=md_out,
        )

        assert json_out.exists()
        assert md_out.exists()
        assert report["cycle_id"] == "CYCLE-007"
        assert report["run_status"] == "COMPLETED"
        assert report["order_count"] == 0
        assert report["ledger_reconciliation"]["sha256_chain_valid"] is True
        assert report["ledger_reconciliation"]["cash_plus_invested_matches_equity"] is True
        assert report["synthetic_candles_used"] == 0
        assert report["external_api_calls"] == 0
        assert report["broker_connections"] == 0
        assert report["paid_api_cost"] == 0.0
        assert report["visual_ai_enabled"] is False
        assert report["live_trading_enabled"] is False
        assert report["qualification_allowed"] is False
        assert report["unseen_evaluation_status"] == "NOT_EVALUATED"
        assert report["holdout_registry_integrity"]["unchanged"] is True

        # Honest provenance classification
        prov = report["market_data_provenance"]
        assert prov["provenance_status"] == "PARTIALLY_VERIFIED"
        assert prov["provenance_verified"] is False
        tf_prov = prov["timeframe_provenance"]
        assert tf_prov["1h"]["classification"] == "PARTIALLY_VERIFIED"
        assert tf_prov["15m"]["classification"] == "UNAVAILABLE"
        assert tf_prov["5m"]["classification"] == "UNAVAILABLE"

        # Holdout registry SHA-256 unchanged and status counts intact (48 EXPOSED, 0 AVAILABLE, 0 SEALED)
        holdout_sha_after = sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)
        assert holdout_sha_after == holdout_sha_before
        registry = json.loads(DEFAULT_HOLDOUT_REGISTRY_PATH.read_text(encoding="utf-8"))
        statuses = [d["state"] for d in registry.get("records", {}).values()]
        assert statuses.count("EXPOSED") == 48
        assert statuses.count("AVAILABLE") == 0
        assert statuses.count("SEALED") == 0

    def test_synthetic_bullish_execution_fixture_and_deterministic_replay(
        self, tmp_path: Path
    ) -> None:
        """UNIT-TEST FIXTURE ONLY (in tmp_path): verifies full
        Central -> CandleTiming (ENTER_NOW) -> DeterministicRiskEngine -> IntradayPaperPortfolio
        execution, transaction cost accounting, equity reconciliation, SHA-256 ledger integrity,
        and deterministic replay across two identical runs.
        """
        # Build a clean 1h parquet in tmp_path by cloning committed RELIANCE_1h.parquet
        # (preserving >= 60 valid sessions for validate_1h_provenance) and shaping only
        # the last 24 pre-cutoff bars + 2026-11-19 session bars for a deterministic bullish setup.
        src_pq = DEFAULT_CACHE_DIR / "RELIANCE_1h.parquet"
        df = pd.read_parquet(src_pq).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST_TZ)

        cutoff = _ist(2026, 11, 19, 9, 15)
        pre_mask = df["timestamp"] < pd.Timestamp(cutoff)
        day_mask = df["timestamp"].dt.date == date(2026, 11, 19)

        pre_indices = df.index[pre_mask].tolist()
        tail_24 = pre_indices[-24:]
        px = 1480.0
        for idx_pos, row_idx in enumerate(tail_24[:-2]):
            o = px
            c = px + 1.5
            df.loc[row_idx, "open"] = o
            df.loc[row_idx, "high"] = c + 0.5
            df.loc[row_idx, "low"] = o - 0.4
            df.loc[row_idx, "close"] = c
            df.loc[row_idx, "volume"] = 120_000.0 + (idx_pos % 3) * 5_000.0
            px = c

        # Penultimate pre-cutoff bar: small bearish candle
        i_prev, i_curr = tail_24[-2], tail_24[-1]
        prev_o = px + 0.8
        prev_c = px - 1.2
        df.loc[i_prev, ["open", "high", "low", "close", "volume"]] = [
            prev_o, prev_o + 0.4, prev_c - 0.4, prev_c, 115_000.0
        ]
        # Final pre-cutoff bar: strong bullish engulfing breakout
        curr_o = prev_c - 0.2
        curr_c = prev_o + 4.0
        df.loc[i_curr, ["open", "high", "low", "close", "volume"]] = [
            curr_o, curr_c + 0.3, curr_o - 0.2, curr_c, 220_000.0
        ]

        # Session bars on 2026-11-19 open cleanly near curr_c and drift upward
        day_indices = df.index[day_mask].tolist()
        assert len(day_indices) >= 6
        s_px = curr_c + 0.5
        for step, row_idx in enumerate(day_indices):
            o = s_px
            c = s_px + 2.0
            df.loc[row_idx, "open"] = o
            df.loc[row_idx, "high"] = c + 0.8
            df.loc[row_idx, "low"] = o - 0.5
            df.loc[row_idx, "close"] = c
            df.loc[row_idx, "volume"] = 150_000.0 + step * 5_000.0
            s_px = c

        fixture_cache = tmp_path / "fixture_cache"
        fixture_cache.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fixture_cache / "RELIANCE_1h.parquet", index=False)

        preds, intel, regime = _make_stage1_payloads(
            symbols=("RELIANCE",),
            obs_date="2026-11-18",
            generated_at="2026-11-18T21:00:00+05:30",
            regime="Bull",
            prob_map={"RELIANCE": 0.72},
            ret_map={"RELIANCE": 1.65},
            sent_map={"RELIANCE": 0.35},
        )
        pred_path = tmp_path / "pred.json"
        intel_path = tmp_path / "intel.json"
        regime_path = tmp_path / "regime.json"
        pred_path.write_text(json.dumps(preds), encoding="utf-8")
        intel_path.write_text(json.dumps(intel), encoding="utf-8")
        regime_path.write_text(json.dumps(regime), encoding="utf-8")

        cfg = Cycle007Config(
            session_date=date(2026, 11, 19),
            symbols=("RELIANCE",),
            cache_dir=fixture_cache,
            predictions_path=pred_path,
            intelligence_path=intel_path,
            regime_path=regime_path,
        )

        report_1 = Cycle007Pipeline(cfg, human_permission=PermissionType.ALLOW_LONG).run()
        report_2 = Cycle007Pipeline(cfg, human_permission=PermissionType.ALLOW_LONG).run()

        assert report_1["run_status"] == "COMPLETED"
        assert report_1["stage1_central_hypotheses"]["RELIANCE"]["direction"] == "BULLISH"
        assert report_1["stage2_timing_assessments"]["RELIANCE"]["decision"] == "ENTER_NOW"
        assert report_1["stage2_timing_assessments"]["RELIANCE"]["alignment_status"] == "CONFIRMED"
        assert report_1["risk_decisions"][0]["approved"] is True
        assert report_1["order_count"] >= 2  # Entry BUY + EOD square-off or target SELL
        assert report_1["filled_order_count"] >= 2
        assert report_1["open_positions_count"] == 0
        assert report_1["closed_trades_count"] >= 1
        assert report_1["cost_accounting"]["total_friction_inr"] > 0.0
        assert report_1["ledger_reconciliation"]["sha256_chain_valid"] is True
        assert report_1["ledger_reconciliation"]["cash_plus_invested_matches_equity"] is True
        assert report_1["external_api_calls"] == 0
        assert report_1["broker_connections"] == 0
        assert report_1["paid_api_cost"] == 0.0
        assert report_1["holdout_registry_integrity"]["unchanged"] is True

        # Deterministic replay: identical decisions, orders, risk decisions, and portfolio results
        assert report_1["per_symbol_decisions"] == report_2["per_symbol_decisions"]
        assert report_1["risk_decisions"] == report_2["risk_decisions"]
        assert report_1["portfolio_results"] == report_2["portfolio_results"]
        assert report_1["cost_accounting"] == report_2["cost_accounting"]
        assert len(report_1["orders"]) == len(report_2["orders"])
        for o1, o2 in zip(report_1["orders"], report_2["orders"]):
            assert o1["symbol"] == o2["symbol"]
            assert o1["side"] == o2["side"]
            assert o1["quantity"] == o2["quantity"]
            assert o1["filled_price"] == o2["filled_price"]
            assert o1["status"] == o2["status"]

    def test_open_gap_rr_degraded_and_stop_distance_exceeded_reject_cleanly(
        self, tmp_path: Path
    ) -> None:
        """UNIT-TEST FIXTURE ONLY (in tmp_path): verifies OPEN_GAP_RR_DEGRADED and
        OPEN_GAP_STOP_DISTANCE_EXCEEDED fail closed with zero orders and intact ledger.
        """
        src_pq = DEFAULT_CACHE_DIR / "RELIANCE_1h.parquet"
        base_df = pd.read_parquet(src_pq).copy()
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"]).dt.tz_convert(IST_TZ)

        cutoff = _ist(2026, 11, 19, 9, 15)
        pre_indices = base_df.index[base_df["timestamp"] < pd.Timestamp(cutoff)].tolist()
        day_indices = base_df.index[base_df["timestamp"].dt.date == date(2026, 11, 19)].tolist()
        assert len(day_indices) >= 6

        tail_24 = pre_indices[-24:]
        px = 1480.0
        for idx_pos, row_idx in enumerate(tail_24[:-2]):
            o = px
            c = px + 1.5
            base_df.loc[row_idx, ["open", "high", "low", "close", "volume"]] = [
                o, c + 0.5, o - 0.4, c, 120_000.0 + (idx_pos % 3) * 5_000.0
            ]
            px = c

        i_prev, i_curr = tail_24[-2], tail_24[-1]
        prev_o, prev_c = px + 0.8, px - 1.2
        base_df.loc[i_prev, ["open", "high", "low", "close", "volume"]] = [
            prev_o, prev_o + 0.4, prev_c - 0.4, prev_c, 115_000.0
        ]
        curr_o, curr_c = prev_c - 0.2, prev_o + 4.0
        base_df.loc[i_curr, ["open", "high", "low", "close", "volume"]] = [
            curr_o, curr_c + 0.3, curr_o - 0.2, curr_c, 220_000.0
        ]

        preds, intel, regime = _make_stage1_payloads(
            symbols=("RELIANCE",),
            obs_date="2026-11-18",
            generated_at="2026-11-18T21:00:00+05:30",
            regime="Bull",
            prob_map={"RELIANCE": 0.72},
            ret_map={"RELIANCE": 1.65},
            sent_map={"RELIANCE": 0.35},
        )
        pred_path = tmp_path / "pred.json"
        intel_path = tmp_path / "intel.json"
        regime_path = tmp_path / "regime.json"
        pred_path.write_text(json.dumps(preds), encoding="utf-8")
        intel_path.write_text(json.dumps(intel), encoding="utf-8")
        regime_path.write_text(json.dumps(regime), encoding="utf-8")

        # Case 1: Moderate open gap up degrades realized R:R below min_risk_reward_ratio (1.5)
        # while staying within max_stop_distance_pct (0.06) and below target.
        df_rr = base_df.copy()
        s_px = curr_c + 12.0
        for step, row_idx in enumerate(day_indices):
            df_rr.loc[row_idx, ["open", "high", "low", "close", "volume"]] = [
                s_px, s_px + 2.0, s_px - 0.5, s_px + 1.5, 150_000.0 + step * 5_000.0
            ]
            s_px += 1.5

        cache_rr = tmp_path / "cache_rr"
        cache_rr.mkdir(parents=True, exist_ok=True)
        df_rr.to_parquet(cache_rr / "RELIANCE_1h.parquet", index=False)

        report_rr = Cycle007Pipeline(
            Cycle007Config(
                session_date=date(2026, 11, 19),
                symbols=("RELIANCE",),
                cache_dir=cache_rr,
                predictions_path=pred_path,
                intelligence_path=intel_path,
                regime_path=regime_path,
            )
        ).run()

        assert report_rr["run_status"] == "COMPLETED"
        assert report_rr["stage2_timing_assessments"]["RELIANCE"]["decision"] == "ENTER_NOW"
        assert report_rr["risk_decisions"][0]["approved"] is False
        assert report_rr["risk_decisions"][0]["status"] == "OPEN_GAP_RR_DEGRADED"
        assert report_rr["order_count"] == 0
        assert report_rr["ledger_reconciliation"]["sha256_chain_valid"] is True

        # Case 2: Large open gap up exceeds max_stop_distance_pct (0.06) while remaining < target
        # (using target_rr_multiplier=4.0 so target stays above the gapped open).
        df_stop = base_df.copy()
        s_px = curr_c * 1.055
        for step, row_idx in enumerate(day_indices):
            df_stop.loc[row_idx, ["open", "high", "low", "close", "volume"]] = [
                s_px, s_px + 2.0, s_px - 0.5, s_px + 1.5, 150_000.0 + step * 5_000.0
            ]
            s_px += 1.5

        cache_stop = tmp_path / "cache_stop"
        cache_stop.mkdir(parents=True, exist_ok=True)
        df_stop.to_parquet(cache_stop / "RELIANCE_1h.parquet", index=False)

        report_stop = Cycle007Pipeline(
            Cycle007Config(
                session_date=date(2026, 11, 19),
                symbols=("RELIANCE",),
                cache_dir=cache_stop,
                predictions_path=pred_path,
                intelligence_path=intel_path,
                regime_path=regime_path,
                timing_config=CandleTimingConfig(target_rr_multiplier=4.0),
            )
        ).run()

        assert report_stop["run_status"] == "COMPLETED"
        assert report_stop["stage2_timing_assessments"]["RELIANCE"]["decision"] == "ENTER_NOW"
        assert report_stop["risk_decisions"][0]["approved"] is False
        assert report_stop["risk_decisions"][0]["status"] == "OPEN_GAP_STOP_DISTANCE_EXCEEDED"
        assert report_stop["order_count"] == 0
        assert report_stop["ledger_reconciliation"]["sha256_chain_valid"] is True

    def test_pipeline_json_and_markdown_reports_preserve_calibration_and_execution_provenance(
        self, tmp_path: Path
    ) -> None:
        """Verify CYCLE-007 JSON & Markdown reports expose UNCALIBRATED status (top-level and
        per-symbol) and complete execution_provenance covering all 6 CYCLE-007 modules.
        """
        json_out = tmp_path / "cycle_007_provenance_report.json"
        md_out = tmp_path / "cycle_007_provenance_report.md"
        report = run_cycle_007_dry_run(
            session_date=date(2026, 11, 19),
            snapshot_dir=tmp_path / "snapshot",
            json_out=json_out,
            md_out=md_out,
        )

        loaded_json = json.loads(json_out.read_text(encoding="utf-8"))
        md_text = md_out.read_text(encoding="utf-8")

        expected_modules = {
            "trader/timing/__init__.py",
            "trader/timing/candle_intelligence.py",
            "trader/timing/central_directional_agent.py",
            "trader/timing/candle_timing_agent.py",
            "trader/timing/cycle007_pipeline.py",
            "scripts/run_cycle_007_timing_dry_run.py",
        }
        assert set(CYCLE007_IMPLEMENTATION_MODULES) == expected_modules
        assert len(expected_modules) == 6

        for payload in (report, loaded_json):
            assert payload["confidence_calibration_status"] == "UNCALIBRATED"
            note = payload["confidence_calibration_note"]
            assert isinstance(note, str) and note.strip() != ""
            assert "heuristic" in note.lower()
            assert "not" in note.lower() and "calibrated probability" in note.lower()

            per_sym = payload["per_symbol_decisions"]
            assert len(per_sym) >= 1
            for entry in per_sym:
                assert entry["confidence_calibration_status"] == "UNCALIBRATED"
                assert isinstance(entry["confidence_calibration_note"], str)
                assert entry["confidence_calibration_note"].strip() != ""
                assert entry["stage1_central"]["confidence_calibration_status"] == "UNCALIBRATED"
                assert entry["stage2_timing"]["central_confidence_calibration_status"] == "UNCALIBRATED"

            for hyp_dict in payload["stage1_central_hypotheses"].values():
                assert hyp_dict["confidence_calibration_status"] == "UNCALIBRATED"
            for timing_dict in payload["stage2_timing_assessments"].values():
                assert timing_dict["central_confidence_calibration_status"] == "UNCALIBRATED"

            exec_prov = payload["execution_provenance"]
            assert isinstance(exec_prov["implementation_commit_sha"], str)
            assert len(exec_prov["implementation_commit_sha"]) >= 7
            assert exec_prov["implementation_commit_sha"] == payload["code_commit_sha"]
            assert isinstance(exec_prov["working_tree_clean_at_execution"], bool)
            assert isinstance(exec_prov["dirty_paths_at_execution"], list)
            assert isinstance(exec_prov["report_commit_distinction"], str)
            assert exec_prov["report_commit_distinction"].strip() != ""

            file_hashes = exec_prov["implementation_file_hashes"]
            assert isinstance(file_hashes, dict)
            assert set(file_hashes.keys()) == expected_modules
            for rel_path, digest in file_hashes.items():
                assert isinstance(digest, str) and len(digest) == 64, f"Bad SHA-256 for {rel_path}"

        # Markdown report assertions
        assert "UNCALIBRATED" in md_text
        assert "confidence_calibration_status" in md_text
        assert "## 5. Execution Provenance" in md_text
        assert "implementation_commit_sha" in md_text
        assert "working_tree_clean_at_execution" in md_text
        assert "dirty_paths_at_execution" in md_text
        assert "report_commit_distinction" in md_text
        assert loaded_json["execution_provenance"]["implementation_commit_sha"] in md_text


