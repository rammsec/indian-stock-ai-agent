"""CYCLE-023: Multi-Timeframe (1m vs 5m vs 15m) Causal Resampler & Walk-Forward Optimizer.

Causally resamples genuine 1-minute NSE candles (`data/intraday_1m/*.parquet`) into exact
NSE-session-aligned `5m` and `15m` bars (`09:15`, `09:20`, ... / `09:15`, `09:30`, ...) with
ZERO lookahead and ZERO cross-day bar merging, then benchmarks `1m` vs `5m` vs `15m` across
all 5 NSE stocks & all 4 strategy profiles on a strict chronological Train -> unseen OOS
Test split (`max(train_dates) < min(test_dates)`).

HARD INVARIANTS:
- Strictly paper-only (`LIVE_TRADING_ENABLED = False`).
- Zero synthetic bar fabrication or cross-date leakage.
- Exact Indian NSE statutory friction (`calculate_intraday_costs`) on every trade.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Sequence

from trader.config import IST_TZ
from trader.data.loader_1m import (
    DEFAULT_1M_DIR,
    DEFAULT_SYMBOLS_1M,
    OneMinuteSessionBundle,
    load_1m_sessions,
)
from trader.data.schema import Candle5M
from trader.evaluation.walkforward_optimizer import (
    ALL_STRATEGY_PROFILES,
    FastSessionConfig,
    aggregate_session_metrics,
    precompute_causal_1m_arrays,
    simulate_fast_1m_session,
    split_bundles_chronologically,
)
from trader.intelligence.chart_overlays import StrategyProfileLiteral

LIVE_TRADING_ENABLED: Final[bool] = False
SUPPORTED_RESAMPLE_MINUTES: Final[tuple[int, ...]] = (1, 5, 15)


def resample_bars_nse_causal(bars: Sequence[Candle5M], timeframe_min: int) -> list[Candle5M]:
    """Causally resample 1m NSE bars into exact 09:15-IST-aligned `timeframe_min` bars.

    Never merges bars across different trading dates and never fabricates missing buckets.
    """
    if timeframe_min not in SUPPORTED_RESAMPLE_MINUTES:
        raise ValueError(
            f"Unsupported timeframe_min={timeframe_min}; expected one of {SUPPORTED_RESAMPLE_MINUTES}"
        )
    if timeframe_min == 1 or not bars:
        return list(bars)

    ordered = sorted(bars, key=lambda b: b.timestamp.astimezone(IST_TZ))
    buckets: OrderedDict[tuple[date, int], list[Candle5M]] = OrderedDict()

    for b in ordered:
        ts = b.timestamp.astimezone(IST_TZ)
        elapsed_min = (ts.hour - 9) * 60 + (ts.minute - 15)
        if elapsed_min < 0 or elapsed_min >= 375:
            continue
        b_idx = elapsed_min // timeframe_min
        key = (ts.date(), b_idx)
        buckets.setdefault(key, []).append(b)

    out: list[Candle5M] = []
    for (d, b_idx), grp in buckets.items():
        b_open_ts = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST_TZ) + timedelta(
            minutes=b_idx * timeframe_min
        )
        out.append(
            Candle5M(
                symbol=grp[0].symbol,
                exchange=grp[0].exchange,
                timestamp=b_open_ts,
                open=float(grp[0].open),
                high=max(float(x.high) for x in grp),
                low=min(float(x.low) for x in grp),
                close=float(grp[-1].close),
                volume=sum(float(x.volume) for x in grp),
            )
        )
    return out


def resample_bundle_causal(
    bundle: OneMinuteSessionBundle,
    timeframe_min: int,
) -> OneMinuteSessionBundle:
    """Resample a `OneMinuteSessionBundle` (both warmup & session bars) causally."""
    if timeframe_min == 1:
        return bundle
    return OneMinuteSessionBundle(
        symbol=bundle.symbol,
        trading_date=bundle.trading_date,
        warmup_bars=resample_bars_nse_causal(bundle.warmup_bars, timeframe_min),
        session_bars=resample_bars_nse_causal(bundle.session_bars, timeframe_min),
        source_providers=bundle.source_providers,
    )


def default_config_for_timeframe(
    timeframe_min: int,
    strategy_profile: StrategyProfileLiteral,
    initial_capital_inr: float = 50_000.0,
) -> FastSessionConfig:
    """Return timeframe-calibrated friction-shielded config for 1m / 5m / 15m."""
    if timeframe_min == 1:
        return FastSessionConfig(
            strategy_profile=strategy_profile,
            initial_capital_inr=initial_capital_inr,
            cooldown_bars=5,
            max_entries_per_day=4,
            min_reward_to_friction=3.0,
            rr_target=2.5,
            auto_direction=True,
        )
    if timeframe_min == 5:
        return FastSessionConfig(
            strategy_profile=strategy_profile,
            initial_capital_inr=initial_capital_inr,
            cooldown_bars=2,
            max_entries_per_day=2,
            min_reward_to_friction=5.0,
            rr_target=2.5,
            auto_direction=True,
        )
    # 15m: high-conviction, ultra-low-friction intraday swing setups
    return FastSessionConfig(
        strategy_profile=strategy_profile,
        initial_capital_inr=initial_capital_inr,
        cooldown_bars=1,
        max_entries_per_day=2,
        min_reward_to_friction=5.0,
        rr_target=2.5,
        auto_direction=True,
    )


def run_mtf_walkforward_comparison(
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS_1M,
    base_dir: str | Path = DEFAULT_1M_DIR,
    timeframes_min: Sequence[int] = (1, 5, 15),
    profiles: Sequence[StrategyProfileLiteral] = ALL_STRATEGY_PROFILES,
    initial_capital_inr: float = 50_000.0,
    train_ratio: float = 0.70,
    max_days: int | None = None,
    bundles_override: dict[str, list[OneMinuteSessionBundle]] | None = None,
) -> dict[str, Any]:
    """Compare 1m vs 5m vs 15m strictly chronologically on Train -> unseen OOS Test."""
    if bundles_override is not None:
        raw_bundles = bundles_override
    else:
        raw_bundles = {
            sym: load_1m_sessions(
                sym,
                base_dir=base_dir,
                warmup_bars_count=375,  # 1 full prior NSE day so 5m (75 bars) & 15m (25 bars) are fully warm
                max_days=max_days,
            )
            for sym in symbols
        }

    train_dates, test_dates, train_bundles_1m, test_bundles_1m = split_bundles_chronologically(
        raw_bundles, train_ratio=train_ratio
    )
    assert max(train_dates) < min(test_dates), "Zero-lookahead chronological split violated!"

    tf_results: dict[str, Any] = {}
    best_overall_key: tuple[str, str] | None = None
    best_overall_train_net = -float("inf")

    for tf_min in timeframes_min:
        tf_label = f"{tf_min}m"
        tr_bundles = [resample_bundle_causal(b, tf_min) for b in train_bundles_1m]
        te_bundles = [resample_bundle_causal(b, tf_min) for b in test_bundles_1m]
        tr_precomp = [precompute_causal_1m_arrays(b) for b in tr_bundles]
        te_precomp = [precompute_causal_1m_arrays(b) for b in te_bundles]

        profile_matrix: dict[str, Any] = {}
        best_tf_prof: str | None = None
        best_tf_train_net = -float("inf")

        for prof in profiles:
            cfg = default_config_for_timeframe(tf_min, prof, initial_capital_inr=initial_capital_inr)
            tr_res = [
                simulate_fast_1m_session(b, cfg, precomputed=pc)
                for b, pc in zip(tr_bundles, tr_precomp)
            ]
            te_res = [
                simulate_fast_1m_session(b, cfg, precomputed=pc)
                for b, pc in zip(te_bundles, te_precomp)
            ]
            tr_m = aggregate_session_metrics(tr_res)
            te_m = aggregate_session_metrics(te_res)
            profile_matrix[prof] = {
                "train": tr_m,
                "oos_test": te_m,
            }
            if tr_m["total_net_pnl_inr"] > best_tf_train_net:
                best_tf_train_net = tr_m["total_net_pnl_inr"]
                best_tf_prof = prof
            if tr_m["total_net_pnl_inr"] > best_overall_train_net:
                best_overall_train_net = tr_m["total_net_pnl_inr"]
                best_overall_key = (tf_label, prof)

        tf_results[tf_label] = {
            "timeframe_min": tf_min,
            "train_selected_profile": best_tf_prof,
            "train_selected_oos_metrics": (
                profile_matrix[best_tf_prof]["oos_test"] if best_tf_prof else None
            ),
            "profiles": profile_matrix,
        }

    return {
        "schema_version": "cycle023_mtf_walkforward_v1",
        "live_trading_enabled": False,
        "zero_lookahead_verified": max(train_dates) < min(test_dates),
        "initial_capital_inr": float(initial_capital_inr),
        "symbols": sorted(raw_bundles.keys()),
        "train_date_range": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "oos_test_date_range": [test_dates[0].isoformat(), test_dates[-1].isoformat()],
        "train_days": len(train_dates),
        "oos_test_days": len(test_dates),
        "train_sessions": len(train_bundles_1m),
        "oos_test_sessions": len(test_bundles_1m),
        "train_selected_overall": (
            {
                "timeframe": best_overall_key[0],
                "profile": best_overall_key[1],
                "oos_test": tf_results[best_overall_key[0]]["profiles"][best_overall_key[1]]["oos_test"],
            }
            if best_overall_key is not None
            else None
        ),
        "timeframes": tf_results,
    }
