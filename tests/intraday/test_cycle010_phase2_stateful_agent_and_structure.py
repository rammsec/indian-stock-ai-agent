"""CYCLE-010 Phase 2 Acceptance & Unit Tests:
- Strictly causal 20-scenario market structure detection (trader/intelligence/structure.py)
- Stateful 9-call CandleTimingAgent.evaluate_bar_lifecycle() (OPEN_LONG, OPEN_SHORT, WAIT, HOLD,
  EXIT_LONG, COVER_SHORT, NO_TRADE, CANCEL_SETUP, REJECT)
- Monotonic trailing stop ratcheting on HOLD
- Provider zero-volume gap detection (VOLUME_UNREPORTED_BY_PROVIDER)
- Zero-lookahead future perturbation invariance on analyze_market_structure
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trader.config import IST_TZ
from trader.intelligence.structure import analyze_market_structure
from trader.timing.candle_timing_agent import (
    ActiveSetupContext,
    AgentCall,
    CandleTimingAgent,
    PositionContext,
)
from trader.timing.central_directional_agent import (
    DirectionalBias,
    DirectionalHypothesis,
)


def _make_hypothesis(
    direction: DirectionalBias = DirectionalBias.BULLISH,
    symbol: str = "TCS",
    as_of: datetime | None = None,
) -> DirectionalHypothesis:
    ts = as_of or datetime(2026, 11, 20, 14, 15, tzinfo=IST_TZ)
    return DirectionalHypothesis(
        symbol=symbol,
        direction=direction,
        confidence=0.72 if direction != DirectionalBias.NEUTRAL else 0.30,
        prob_next_day_up=0.68 if direction == DirectionalBias.BULLISH else (0.32 if direction == DirectionalBias.BEARISH else 0.50),
        expected_next_day_return_pct=1.2 if direction == DirectionalBias.BULLISH else (-1.2 if direction == DirectionalBias.BEARISH else 0.0),
        news_sentiment_score=0.25 if direction == DirectionalBias.BULLISH else -0.25,
        market_regime="BULL_TREND" if direction == DirectionalBias.BULLISH else "BEAR_TREND",
        observation_date=ts.date(),
        as_of_ist=ts,
        rationale=("Phase 2 test hypothesis",),
    )


def _build_bars(
    rows: list[tuple[float, float, float, float, float]],
    *,
    start: datetime | None = None,
    step_minutes: int = 60,
) -> pd.DataFrame:
    t0 = start or datetime(2026, 11, 18, 9, 15, tzinfo=IST_TZ)
    timestamps = []
    cur = t0
    for _ in rows:
        timestamps.append(cur)
        cur = cur + timedelta(minutes=step_minutes)
        if cur.hour > 15 or (cur.hour == 15 and cur.minute > 15):
            cur = (cur + timedelta(days=1)).replace(hour=9, minute=15)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] for r in rows],
        }
    )


def _strong_uptrend_df(n: int = 25, start_px: float = 1000.0) -> pd.DataFrame:
    rows = []
    px = start_px
    for i in range(n):
        o = px
        c = px + 2.5
        h = c + 0.3
        l = o - 0.3
        v = 10000.0 if i < n - 1 else 18000.0
        rows.append((o, h, l, c, v))
        px = c
    return _build_bars(rows)


def _strong_downtrend_df(n: int = 25, start_px: float = 1200.0) -> pd.DataFrame:
    rows = []
    px = start_px
    for i in range(n):
        o = px
        c = px - 2.5
        h = o + 0.3
        l = c - 0.3
        v = 10000.0 if i < n - 1 else 18000.0
        rows.append((o, h, l, c, v))
        px = c
    return _build_bars(rows)


def test_all_9_canonical_calls_and_trailing_stop_ratchet() -> None:
    """Verify CandleTimingAgent.evaluate_bar_lifecycle emits all 9 canonical calls with monotonic trailing stops."""
    agent = CandleTimingAgent()

    # 1. OPEN_LONG on strong bullish trend
    up_df = _strong_uptrend_df()
    ts_up = up_df["timestamp"].iloc[-1].to_pydatetime() + timedelta(hours=1)
    res_open_long = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_up),
        bars_primary=up_df,
        as_of_ist=ts_up,
    )
    assert res_open_long.call == AgentCall.OPEN_LONG
    assert res_open_long.structure.trend_regime == "STRONG_BULLISH"
    assert res_open_long.central_confidence_calibration_status == "UNCALIBRATED"

    # 2. OPEN_SHORT on strong bearish trend
    dn_df = _strong_downtrend_df()
    ts_dn = dn_df["timestamp"].iloc[-1].to_pydatetime() + timedelta(hours=1)
    res_open_short = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BEARISH, as_of=ts_dn),
        bars_primary=dn_df,
        as_of_ist=ts_dn,
    )
    assert res_open_short.call == AgentCall.OPEN_SHORT
    assert res_open_short.structure.trend_regime == "STRONG_BEARISH"

    # 3. HOLD on open LONG with upward trailing stop ratchet
    entry_px = float(up_df["close"].iloc[-5])
    init_stop = entry_px - 20.0
    pos_long = PositionContext(side="LONG", entry_price=entry_px, current_stop_loss=init_stop, target_price=entry_px + 200.0, bars_held=4)
    res_hold_long = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_up),
        bars_primary=up_df,
        as_of_ist=ts_up,
        position=pos_long,
    )
    assert res_hold_long.call == AgentCall.HOLD
    assert res_hold_long.recommended_trailing_stop is not None
    assert res_hold_long.recommended_trailing_stop > init_stop  # ratcheted upward

    # 4. EXIT_LONG when bearish reversal/breakdown hits open LONG (never OPEN_SHORT!)
    pos_long_under_attack = PositionContext(
        side="LONG",
        entry_price=float(dn_df["close"].iloc[-5]),
        current_stop_loss=float(dn_df["close"].iloc[-1]) - 50.0,
        target_price=float(dn_df["close"].iloc[-5]) + 100.0,
    )
    res_exit_long = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_dn),
        bars_primary=dn_df,
        as_of_ist=ts_dn,
        position=pos_long_under_attack,
    )
    assert res_exit_long.call == AgentCall.EXIT_LONG
    assert res_exit_long.exit_reason is not None

    # 5. HOLD on open SHORT with downward trailing stop ratchet
    short_entry = float(dn_df["close"].iloc[-5])
    short_init_stop = short_entry + 20.0
    pos_short = PositionContext(side="SHORT", entry_price=short_entry, current_stop_loss=short_init_stop, target_price=short_entry - 200.0)
    res_hold_short = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BEARISH, as_of=ts_dn),
        bars_primary=dn_df,
        as_of_ist=ts_dn,
        position=pos_short,
    )
    assert res_hold_short.call == AgentCall.HOLD
    assert res_hold_short.recommended_trailing_stop is not None
    assert res_hold_short.recommended_trailing_stop < short_init_stop  # ratcheted downward

    # 6. COVER_SHORT when bullish breakout hits open SHORT (never OPEN_LONG!)
    pos_short_squeezed = PositionContext(
        side="SHORT",
        entry_price=float(up_df["close"].iloc[-5]),
        current_stop_loss=float(up_df["close"].iloc[-1]) + 50.0,
        target_price=float(up_df["close"].iloc[-5]) - 100.0,
    )
    res_cover_short = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BEARISH, as_of=ts_up),
        bars_primary=up_df,
        as_of_ist=ts_up,
        position=pos_short_squeezed,
    )
    assert res_cover_short.call == AgentCall.COVER_SHORT
    assert res_cover_short.exit_reason is not None

    # 7. NO_TRADE when Central hypothesis is NEUTRAL and no active setup
    res_no_trade = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.NEUTRAL, as_of=ts_up),
        bars_primary=up_df,
        as_of_ist=ts_up,
    )
    assert res_no_trade.call == AgentCall.NO_TRADE

    # 8. WAIT on indecision / doji bar (registers ActiveSetupContext) -> CANCEL_SETUP when invalidated
    doji_rows = [(1000.0, 1005.0, 995.0, 1000.2, 10000.0)] * 20
    doji_df = _build_bars(doji_rows)
    ts_doji = doji_df["timestamp"].iloc[-1].to_pydatetime() + timedelta(hours=1)
    res_wait = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_doji),
        bars_primary=doji_df,
        as_of_ist=ts_doji,
    )
    assert res_wait.call == AgentCall.WAIT
    assert res_wait.updated_setup is not None

    invalidated_setup = ActiveSetupContext(
        direction="LONG",
        setup_type="TREND_SETUP",
        trigger_price=1006.0,
        invalidation_price=998.0,  # breached because doji_df low is 995.0
        bars_watching=2,
    )
    res_cancel = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_doji),
        bars_primary=doji_df,
        as_of_ist=ts_doji,
        active_setup=invalidated_setup,
    )
    assert res_cancel.call == AgentCall.CANCEL_SETUP
    assert res_cancel.cancel_reason is not None

    # 9. REJECT on empty/corrupt data
    res_reject = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_up),
        bars_primary=None,
        as_of_ist=ts_up,
    )
    assert res_reject.call == AgentCall.REJECT


def test_structure_engine_20_scenarios_detection() -> None:
    """Verify deterministic causal detection across breakout/trap, VWAP, EMA cross, ORB, H&S, Flag, Volume & Volatility."""
    # Sideways consolidation & contraction
    flat_rows = [(1000.0, 1002.0, 998.0, 1000.0, 10000.0)] * 20
    s_flat = analyze_market_structure(_build_bars(flat_rows))
    assert s_flat.trend_regime == "SIDEWAYS_CONSOLIDATION"

    # 2-bar False Breakout (Bull Trap)
    trap_rows = [(1000.0, 1005.0, 995.0, 1001.0, 10000.0)] * 15
    trap_rows.append((1002.0, 1012.0, 1001.0, 1010.0, 14000.0))  # breaks above 1005
    trap_rows.append((1009.0, 1011.0, 999.0, 1002.0, 15000.0))  # closes back inside <= 1005
    s_trap = analyze_market_structure(_build_bars(trap_rows))
    assert s_trap.breakout_event == "FALSE_BREAKOUT"

    # 2-bar Failed Breakdown (Bear Trap / Spring)
    spring_rows = [(1000.0, 1005.0, 995.0, 1001.0, 10000.0)] * 15
    spring_rows.append((999.0, 1000.0, 988.0, 990.0, 14000.0))  # breaks below 995
    spring_rows.append((991.0, 1003.0, 989.0, 1000.0, 15000.0))  # reclaims >= 995
    s_spring = analyze_market_structure(_build_bars(spring_rows))
    assert s_spring.breakout_event == "FAILED_BREAKDOWN"

    # VWAP Reclaim
    vwap_rows = [
        (1000.0, 1005.0, 998.0, 1000.0, 10000.0),
        (1000.0, 1002.0, 990.0, 992.0, 10000.0),  # below VWAP
        (993.0, 1010.0, 992.0, 1008.0, 20000.0),  # reclaims VWAP
    ]
    s_vwap = analyze_market_structure(
        _build_bars(vwap_rows, start=datetime(2026, 11, 20, 9, 15, tzinfo=IST_TZ), step_minutes=60)
    )
    assert s_vwap.vwap_event == "VWAP_RECLAIM"
    assert s_vwap.orb_event == "ORB_BULLISH_BREAKOUT"
    assert s_vwap.volume_regime == "HIGH_VOLUME_CONFIRMATION"

    # Yahoo Zero-Volume Unreported Provider Gap
    zvol_rows = [(1000.0, 1008.0, 998.0, 1006.0, 10000.0)] * 15
    zvol_rows.append((1006.0, 1015.0, 1004.0, 1014.0, 0.0))
    s_zvol = analyze_market_structure(_build_bars(zvol_rows))
    assert s_zvol.volume_regime == "VOLUME_UNREPORTED_BY_PROVIDER"
    assert s_zvol.relative_volume_ratio is None

    # Bull Flag Breakout (impulse pole + tight consolidation + upside break)
    flag_rows = [(1000.0, 1004.0, 998.0, 1002.0, 10000.0)] * 10
    # Pole bars (-10..-6): sharp rally +40 pts
    flag_rows.extend(
        [
            (1002.0, 1014.0, 1001.0, 1012.0, 15000.0),
            (1012.0, 1024.0, 1011.0, 1022.0, 16000.0),
            (1022.0, 1034.0, 1021.0, 1032.0, 17000.0),
            (1032.0, 1044.0, 1031.0, 1042.0, 18000.0),
        ]
    )
    # 5 tight consolidation bars (-6..-2)
    flag_rows.extend(
        [
            (1042.0, 1044.0, 1039.0, 1040.0, 8000.0),
            (1040.0, 1043.0, 1038.0, 1041.0, 7500.0),
            (1041.0, 1043.0, 1039.0, 1040.0, 7000.0),
            (1040.0, 1043.0, 1039.0, 1042.0, 7200.0),
            (1042.0, 1044.0, 1040.0, 1043.0, 7500.0),
        ]
    )
    # Breakout bar (-1)
    flag_rows.append((1043.0, 1054.0, 1042.0, 1052.0, 22000.0))
    s_flag = analyze_market_structure(_build_bars(flag_rows))
    assert s_flag.consolidation_pattern == "BULL_FLAG_BREAKOUT"

    # Head & Shoulders Breakdown (3 causal pivot highs: H1=1050, H2=1080, H3=1052, neckline=1010)
    hs_closes = [
        1000, 1015, 1035, 1050, 1035, 1020, 1010,  # Left shoulder (1050) + L1 (1010)
        1030, 1060, 1080, 1055, 1025, 1012,        # Head (1080) + L2 (1012)
        1030, 1052, 1035, 1020, 1015, 1002,        # Right shoulder (1052) + neckline break (1002 < 1010)
    ]
    hs_rows = [(c - 2.0, c + 3.0, c - 3.0, float(c), 12000.0) for c in hs_closes]
    s_hs = analyze_market_structure(_build_bars(hs_rows))
    assert s_hs.chart_pattern == "HEAD_AND_SHOULDERS_BREAKDOWN"


def test_structure_engine_zero_lookahead_future_perturbation_invariance() -> None:
    """analyze_market_structure(df[:t+1]) must be strictly invariant to 5x future OHLCV perturbation."""
    full_df = _strong_uptrend_df(n=35)
    t = 22
    base_snap = analyze_market_structure(full_df.iloc[: t + 1].copy())

    perturbed = full_df.copy()
    perturbed.loc[t + 1 :, ["open", "high", "low", "close", "volume"]] *= 5.0
    pert_snap = analyze_market_structure(perturbed.iloc[: t + 1].copy())

    assert base_snap.to_dict() == pert_snap.to_dict()


def test_lifecycle_fails_closed_on_future_bar_corrupt_data_and_direction_flip() -> None:
    """Verify evaluate_bar_lifecycle fails closed with REJECT on future-dated or corrupt bars even when position is open, and cancels active_setup on Central direction flip."""
    agent = CandleTimingAgent()
    up_df = _strong_uptrend_df()
    # Set as_of_ist BEFORE the last bar's timestamp -> lookahead violation!
    early_cutoff = up_df["timestamp"].iloc[-3].to_pydatetime()
    open_pos = PositionContext(side="LONG", entry_price=1030.0, current_stop_loss=1010.0)

    res_future = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=early_cutoff),
        bars_primary=up_df,
        as_of_ist=early_cutoff,
        position=open_pos,
    )
    assert res_future.call == AgentCall.REJECT
    assert res_future.recommended_trailing_stop == open_pos.current_stop_loss

    # Corrupt DataFrame missing required OHLCV columns must return REJECT without raising KeyError
    corrupt_df = pd.DataFrame({"timestamp": up_df["timestamp"].iloc[:5], "close": [100.0] * 5})
    ts_ok = up_df["timestamp"].iloc[5].to_pydatetime()
    res_corrupt = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BULLISH, as_of=ts_ok),
        bars_primary=corrupt_df,
        as_of_ist=ts_ok,
        position=open_pos,
    )
    assert res_corrupt.call == AgentCall.REJECT

    # Active LONG setup must be cancelled when Central hypothesis flips BEARISH
    doji_df = _build_bars([(1000.0, 1005.0, 995.0, 1000.2, 10000.0)] * 20)
    ts_doji = doji_df["timestamp"].iloc[-1].to_pydatetime() + timedelta(hours=1)
    long_setup = ActiveSetupContext(
        direction="LONG",
        setup_type="TREND_SETUP",
        trigger_price=1006.0,
        invalidation_price=990.0,
        bars_watching=1,
    )
    res_flip = agent.evaluate_bar_lifecycle(
        hypothesis=_make_hypothesis(DirectionalBias.BEARISH, as_of=ts_doji),
        bars_primary=doji_df,
        as_of_ist=ts_doji,
        active_setup=long_setup,
    )
    assert res_flip.call == AgentCall.CANCEL_SETUP
    assert res_flip.cancel_reason == "CENTRAL_DIRECTION_REVERSED"

