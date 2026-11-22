"""CYCLE-009C — Lookahead, Causal Cutoff, MTF Alignment & Recursive Convergence CI Gate.

Tier 1 (`@pytest.mark.fast_ci`, target <15s):
1. Five intentionally biased fixture strategies/indicators — EVERY ONE must fail
   `result.passed is False` with its specific `BiasViolation.code`:
   - Future Shift Leak (`shift(-1)`)
   - Global Normalization / Cross-Sectional Panel Rank Leak
   - Unshifted HTF Merge Leak (incomplete 1h bar peek)
   - Future `ts_init` / Delayed-Collection News & Bar Leak
   - Non-Converged EMA / Insufficient Warmup (`min_periods=1` / slow convergence)
2. Clean zero-leak controls — EVERY ONE must pass `result.passed is True`:
   - Causal `trader.intelligence.registry` indicators (EMA 9/21/50, ATR Wilder 14, RSI Wilder 14, MACD 12/26/9)
   - `trader.intelligence.mtf_aligner.align_htf_to_ltf_ist` (afternoon truncation, 12:30 trap, 15:15-15:30 closing bar)
   - `trader.intelligence.data_contract.build_ohlcv_evidence_frame` (rejects unclosed & delayed `ts_init`, causal cutoff passes)
   - Strict `pub <= as_of & col <= as_of` event filter matching `trader.paper.cycle006c_snapshot`

Tier 2 (`@pytest.mark.research_slow`):
- Real committed `data/intraday/*_1h.parquet` sweep (`LAB_DEFAULT_UNIVERSE`) for
  recursive convergence (`W in [50, 100, 200, 500]`) and rolling bar-index truncation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.resampler import resample_5m_candles
from trader.data.schema import Candle5M
from trader.evaluation.bias_auditor import (
    audit_bar_index_lookahead,
    audit_cross_sectional_lookahead,
    audit_future_event_exclusion,
    audit_mtf_backward_alignment,
    audit_recursive_convergence,
    audit_ts_init_cutoff,
    audit_warmup_sensitivity,
)
from trader.intelligence import (
    DEFAULT_FEATURE_REGISTRY,
    LookaheadViolationError,
    ProvenanceTier,
    align_htf_to_ltf_ist,
    build_ohlcv_evidence_frame,
    compute_atr_series,
    compute_ema_series,
    compute_macd_series,
    compute_rsi_series,
)
from trader.paper.cycle006c_snapshot import _to_ist_datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO_ROOT / "data" / "intraday"


# ---------------------------------------------------------------------------
# Deterministic Synthetic Fixtures (Zero Network / Zero Disk I/O in fast_ci)
# ---------------------------------------------------------------------------


def _make_1h_synthetic_df(n_bars: int = 600, base_price: float = 2500.0, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic, monotonic IST 1h OHLCV DataFrame with realistic small returns."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.0025, size=n_bars)
    closes = base_price * np.cumprod(1.0 + returns)
    opens = np.concatenate([[base_price], closes[:-1]])
    spread = np.abs(rng.normal(loc=0.0015, scale=0.0005, size=n_bars)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    volumes = rng.integers(50_000, 250_000, size=n_bars).astype(float)

    start_open = datetime(2026, 1, 1, 9, 15, tzinfo=IST_TZ)
    ts_open = [start_open + timedelta(hours=i) for i in range(n_bars)]
    ts_close = [t + timedelta(hours=1) for t in ts_open]

    return pd.DataFrame(
        {
            "ts_open_ist": pd.DatetimeIndex(ts_open),
            "ts_close_ist": pd.DatetimeIndex(ts_close),
            "ts_init_ist": pd.DatetimeIndex(ts_close),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def _make_5m_session_df(
    year: int = 2026,
    month: int = 1,
    day: int = 15,
    symbol: str = "RELIANCE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a full 75-bar 5m NSE session (09:15..15:25 open, 09:20..15:30 close) and resampled 1h bars."""
    start = datetime(year, month, day, 9, 15, tzinfo=IST_TZ)
    candles: list[Candle5M] = []
    ltf_rows: list[dict] = []
    for i in range(75):
        ts_open = start + timedelta(minutes=5 * i)
        ts_close = ts_open + timedelta(minutes=5)
        px = 2500.0 + i * 0.8 + (1.5 if i % 3 == 0 else -0.4)
        o, h, l, c, v = px, px + 2.0, px - 1.5, px + 0.6, 1000.0 + i * 25.0
        candles.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=ts_open,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
        )
        ltf_rows.append(
            {
                "ts_open_ist": ts_open,
                "ts_close_ist": ts_close,
                "ts_init_ist": ts_close,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )

    ltf_df = pd.DataFrame(ltf_rows)
    htf_candles = resample_5m_candles(candles, target_minutes=60)
    htf_rows = []
    for hc in htf_candles:
        # NSE closing 1h bucket starts at 15:15 and closes at 15:30 (15m), all others are 60m
        dur = timedelta(minutes=15) if (hc.timestamp.hour == 15 and hc.timestamp.minute == 15) else timedelta(hours=1)
        htf_rows.append(
            {
                "ts_open_ist": hc.timestamp,
                "ts_close_ist": hc.timestamp + dur,
                "ts_init_ist": hc.timestamp + dur,
                "open": hc.open,
                "high": hc.high,
                "low": hc.low,
                "close": hc.close,
                "volume": hc.volume,
                "htf_close": hc.close,
                "htf_ema_9": hc.close * 0.998,
            }
        )
    htf_df = pd.DataFrame(htf_rows)
    return ltf_df, htf_df


def _make_events_df(as_of_ist: datetime) -> pd.DataFrame:
    """Create clean pre-cutoff news/event fixture with both publication and collection <= as_of_ist."""
    return pd.DataFrame(
        [
            {
                "event_id": "ev-1",
                "ticker": "RELIANCE",
                "published_at_ist": as_of_ist - timedelta(hours=3),
                "collected_at_ist": as_of_ist - timedelta(hours=2, minutes=55),
                "sentiment_score": 0.45,
            },
            {
                "event_id": "ev-2",
                "ticker": "TCS",
                "published_at_ist": as_of_ist - timedelta(hours=1),
                "collected_at_ist": as_of_ist - timedelta(minutes=50),
                "sentiment_score": -0.20,
            },
            {
                "event_id": "ev-3",
                "ticker": "ITC",
                "published_at_ist": as_of_ist - timedelta(minutes=15),
                "collected_at_ist": as_of_ist - timedelta(minutes=5),
                "sentiment_score": 0.15,
            },
        ]
    )


def _strict_event_filter(events_df: pd.DataFrame, as_of_ist: datetime) -> pd.DataFrame:
    """Causal event filter matching trader.paper.cycle006c_snapshot (pub <= as_of & col <= as_of)."""
    keep_idx: list[int] = []
    for idx, row in events_df.iterrows():
        raw_pub = row.get("published_at_ist")
        raw_col = row.get("collected_at_ist")
        if raw_pub is None or pd.isna(raw_pub) or raw_col is None or pd.isna(raw_col):
            continue
        try:
            pub_dt = _to_ist_datetime(raw_pub, "published_at_ist")
            col_dt = _to_ist_datetime(raw_col, "collected_at_ist")
        except (ValueError, TypeError):
            continue
        if pub_dt <= as_of_ist and col_dt <= as_of_ist:
            keep_idx.append(idx)
    return events_df.loc[keep_idx].copy()


# ===========================================================================
# TIER 1 — @pytest.mark.fast_ci (<15s)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Five Intentionally Biased Fixture Strategies/Indicators (MUST FAIL)
# ---------------------------------------------------------------------------


@pytest.mark.fast_ci
def test_biased_1_future_shift_leak_caught() -> None:
    """Future shift(-1) / iloc[t+1] leak must fail audit_bar_index_lookahead."""
    df = _make_1h_synthetic_df(n_bars=120)
    cutoffs = [
        df["ts_close_ist"].iloc[40].to_pydatetime(),
        df["ts_close_ist"].iloc[80].to_pydatetime(),
    ]

    def leaky_future_shift(frame: pd.DataFrame) -> pd.Series:
        return frame["close"].shift(-1)

    res = audit_bar_index_lookahead(df, leaky_future_shift, cutoffs)
    assert res.passed is False
    codes = {v.code for v in res.violations}
    assert "TRUNCATION_LOOKAHEAD" in codes
    assert "FUTURE_PERTURBATION_LEAKAGE" in codes


@pytest.mark.fast_ci
def test_biased_2_global_normalization_and_cross_sectional_rank_leak_caught() -> None:
    """Global full-series z-score and cross-sectional panel rank leaks must fail."""
    df_rel = _make_1h_synthetic_df(n_bars=120, base_price=2500.0, seed=11)
    df_tcs = _make_1h_synthetic_df(n_bars=120, base_price=3800.0, seed=22)
    cutoffs = [
        df_rel["ts_close_ist"].iloc[50].to_pydatetime(),
        df_rel["ts_close_ist"].iloc[90].to_pydatetime(),
    ]

    # 2a. Single-series global z-score (full-series mean/std)
    def leaky_global_zscore(frame: pd.DataFrame) -> pd.Series:
        c = frame["close"].astype(float)
        return (c - c.mean()) / (c.std() + 1e-8)

    res_single = audit_bar_index_lookahead(df_rel, leaky_global_zscore, cutoffs)
    assert res_single.passed is False
    single_codes = {v.code for v in res_single.violations}
    assert "TRUNCATION_LOOKAHEAD" in single_codes
    assert "FUTURE_PERTURBATION_LEAKAGE" in single_codes

    # 2b. Cross-sectional global panel rank + full-panel normalization
    panel = {"RELIANCE": df_rel, "TCS": df_tcs}

    def leaky_panel_rank(panel_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        # Global mean across all peers and all timestamps + cross-sectional rank at each timestamp
        all_closes = pd.concat([d["close"].astype(float) for d in panel_map.values()], ignore_index=True)
        global_mean = float(all_closes.mean())
        aligned = pd.concat(
            {sym: d.set_index("ts_close_ist")["close"].astype(float) for sym, d in panel_map.items()},
            axis=1,
        )
        cs_rank = aligned.rank(axis=1, pct=True)
        out: dict[str, pd.Series] = {}
        for sym, d in panel_map.items():
            ts_idx = pd.DatetimeIndex(d["ts_close_ist"])
            ranked = cs_rank[sym].reindex(ts_idx).to_numpy(dtype=float)
            out[sym] = pd.Series((d["close"].to_numpy(dtype=float) - global_mean) * ranked, index=d.index)
        return out

    res_cs = audit_cross_sectional_lookahead(panel, "RELIANCE", leaky_panel_rank, cutoffs)
    assert res_cs.passed is False
    cs_codes = {v.code for v in res_cs.violations}
    assert "PEER_FUTURE_PERTURBATION_LEAKAGE" in cs_codes
    assert "PEER_CURRENT_DROP_FUTURE_LEAKAGE" in cs_codes


@pytest.mark.fast_ci
def test_biased_3_unshifted_htf_merge_leak_caught() -> None:
    """Unshifted HTF merge (peeking into incomplete 1h bar) must fail lookahead & MTF audits."""
    ltf_df, htf_df = _make_5m_session_df()
    # Cutoffs mid-hour: 12:15 (inside 11:15-12:15 -> closed, wait 12:30 is inside 12:15-13:15 incomplete bar)
    cutoff_1230 = datetime(2026, 1, 15, 12, 30, tzinfo=IST_TZ)
    cutoff_1330 = datetime(2026, 1, 15, 13, 30, tzinfo=IST_TZ)

    # 3a. Leaky feature_fn that computes 1h bucket close using ts_open_ist floor and merges onto 5m without waiting for bar close
    def leaky_unshifted_htf_merge(frame: pd.DataFrame) -> pd.Series:
        work = frame.copy()
        # Bucket each 5m bar by its 1h open bucket (12:15..13:10 -> 12:15) and attach the bucket's final close
        # to EVERY 5m bar in that bucket, leaking future 5m bars (e.g. 13:10 close onto 12:30 bar)!
        shifted = work["ts_open_ist"] - timedelta(minutes=15)
        work["bucket"] = shifted.dt.floor("1h") + timedelta(minutes=15)
        htf_agg = work.groupby("bucket")["close"].transform("last")
        return htf_agg

    res_bar = audit_bar_index_lookahead(ltf_df, leaky_unshifted_htf_merge, [cutoff_1230, cutoff_1330])
    assert res_bar.passed is False
    bar_codes = {v.code for v in res_bar.violations}
    assert "TRUNCATION_LOOKAHEAD" in bar_codes
    assert "FUTURE_PERTURBATION_LEAKAGE" in bar_codes


@pytest.mark.fast_ci
def test_biased_4_future_ts_init_and_delayed_collection_leak_caught() -> None:
    """Delayed ts_init (ts_event <= as_of < ts_init) and delayed collection news leaks must fail."""
    df = _make_1h_synthetic_df(n_bars=80)
    as_of = df["ts_close_ist"].iloc[-1].to_pydatetime()

    # 4a. Leaky bar feature_fn that ignores ts_init_ist > as_of_ist
    def leaky_ignore_ts_init(frame: pd.DataFrame) -> float:
        return float(frame["close"].iloc[-1])

    res_ts_init = audit_ts_init_cutoff(df, leaky_ignore_ts_init, as_of)
    assert res_ts_init.passed is False
    ts_codes = {v.code for v in res_ts_init.violations}
    assert "DELAYED_TS_INIT_VISIBLE" in ts_codes

    # 4b. Leaky event filter that only checks published_at_ist <= as_of_ist (ignoring collected_at_ist & naive ts)
    events_df = _make_events_df(as_of)

    def leaky_event_filter(ev_df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
        pub = pd.to_datetime(ev_df["published_at_ist"], errors="coerce", utc=True)
        return ev_df.loc[pub <= cutoff].copy()

    res_ev = audit_future_event_exclusion(events_df, leaky_event_filter, as_of)
    assert res_ev.passed is False
    ev_codes = {v.code for v in res_ev.violations}
    assert "ADVERSARIAL_EVENTS_NOT_EXCLUDED" in ev_codes


@pytest.mark.fast_ci
def test_biased_5_non_converged_ema_and_insufficient_warmup_caught() -> None:
    """Silent float below min_warmup_bars and non-converged recursive EMA must fail."""
    df = _make_1h_synthetic_df(n_bars=600)
    series = df["close"]

    # 5a. Silent un-converged EMA(50) emitting finite float at 15 bars (< min_warmup_bars=50)
    def silent_ema_50(s: pd.Series) -> pd.Series:
        return compute_ema_series(s, span=50, min_periods=1)

    res_warmup = audit_warmup_sensitivity(
        series,
        silent_ema_50,
        min_warmup_bars=50,
        below_probe_bars=15,
    )
    assert res_warmup.passed is False
    warmup_codes = {v.code for v in res_warmup.violations}
    assert "WARMUP_SILENT_FLOAT_BELOW_MIN" in warmup_codes

    # 5b. Non-converged ultra-slow EMA (span=400) where W=200 has NOT converged to W=500
    def non_converged_ema_400(s: pd.Series) -> pd.Series:
        return compute_ema_series(s, span=400, min_periods=1)

    res_conv = audit_recursive_convergence(
        series,
        non_converged_ema_400,
        span_or_period=400,
        windows=(50, 100, 200, 500),
        ref_window=500,
        rtol=1e-4,
    )
    assert res_conv.passed is False
    conv_codes = {v.code for v in res_conv.violations}
    assert "CONVERGENCE_DRIFT_EXCEEDED" in conv_codes


# ---------------------------------------------------------------------------
# 2. Clean Zero-Leak Controls (EVERY ONE MUST PASS result.passed is True)
# ---------------------------------------------------------------------------


@pytest.mark.fast_ci
@pytest.mark.parametrize("span", [9, 21, 50])
def test_clean_registry_ema_passes_lookahead_convergence_and_warmup(span: int) -> None:
    """Canonical EMA(9/21/50) passes bar-index lookahead, recursive convergence, and warmup sensitivity."""
    df = _make_1h_synthetic_df(n_bars=600, seed=100 + span)
    cutoffs = [
        df["ts_close_ist"].iloc[150].to_pydatetime(),
        df["ts_close_ist"].iloc[350].to_pydatetime(),
        df["ts_close_ist"].iloc[500].to_pydatetime(),
    ]
    min_bars = DEFAULT_FEATURE_REGISTRY[f"ema_{span}"].min_bars_required

    res_lookahead = audit_bar_index_lookahead(
        df,
        lambda frame: compute_ema_series(frame["close"], span=span, min_periods=min_bars),
        cutoffs,
    )
    assert res_lookahead.passed is True, res_lookahead.violations

    res_conv = audit_recursive_convergence(
        df["close"],
        lambda s: compute_ema_series(s, span=span, min_periods=min_bars),
        span_or_period=span,
    )
    assert res_conv.passed is True, res_conv.violations

    res_warmup = audit_warmup_sensitivity(
        df["close"],
        lambda s: compute_ema_series(s, span=span, min_periods=min_bars),
        min_warmup_bars=min_bars,
        above_windows=(min_bars + 150, min_bars + 250),
    )
    assert res_warmup.passed is True, res_warmup.violations


@pytest.mark.fast_ci
def test_clean_registry_atr_rsi_macd_pass_lookahead_convergence_and_warmup() -> None:
    """Canonical ATR(14 Wilder), RSI(14 Wilder), and MACD(12,26,9) pass all causal & convergence gates."""
    df = _make_1h_synthetic_df(n_bars=600, seed=77)
    cutoffs = [
        df["ts_close_ist"].iloc[200].to_pydatetime(),
        df["ts_close_ist"].iloc[450].to_pydatetime(),
    ]

    # ATR Wilder 14
    atr_min = DEFAULT_FEATURE_REGISTRY["atr_wilder_14"].min_bars_required
    res_atr_la = audit_bar_index_lookahead(
        df,
        lambda f: compute_atr_series(f["high"], f["low"], f["close"], period=14, method="wilder", min_periods=atr_min),
        cutoffs,
    )
    assert res_atr_la.passed is True, res_atr_la.violations

    # RSI Wilder 14
    rsi_min = DEFAULT_FEATURE_REGISTRY["rsi_wilder_14"].min_bars_required
    res_rsi_la = audit_bar_index_lookahead(
        df,
        lambda f: compute_rsi_series(f["close"], period=14, warmup_fill=None, zero_loss_mode="strict_100"),
        cutoffs,
    )
    assert res_rsi_la.passed is True, res_rsi_la.violations

    res_rsi_conv = audit_recursive_convergence(
        df["close"],
        lambda s: compute_rsi_series(s, period=14, warmup_fill=None, zero_loss_mode="strict_100"),
        span_or_period=14,
        rtol=5e-3,
    )
    assert res_rsi_conv.passed is True, res_rsi_conv.violations

    res_rsi_warm = audit_warmup_sensitivity(
        df["close"],
        lambda s: compute_rsi_series(s, period=14, warmup_fill=None, zero_loss_mode="strict_100"),
        min_warmup_bars=rsi_min,
        above_windows=(rsi_min + 100, rsi_min + 200),
        max_drift=0.05,
    )
    assert res_rsi_warm.passed is True, res_rsi_warm.violations

    # MACD (12, 26, 9)
    res_macd_la = audit_bar_index_lookahead(
        df,
        lambda f: compute_macd_series(f["close"], fast=12, slow=26, signal=9)[0],
        cutoffs,
    )
    assert res_macd_la.passed is True, res_macd_la.violations

    res_macd_conv = audit_recursive_convergence(
        df["close"],
        lambda s: compute_macd_series(s, fast=12, slow=26, signal=9)[0],
        span_or_period=26,
        rtol=5e-2,
    )
    assert res_macd_conv.passed is True, res_macd_conv.violations


@pytest.mark.fast_ci
def test_clean_mtf_aligner_passes_afternoon_truncation_1230_and_closing_bar() -> None:
    """align_htf_to_ltf_ist passes 12:15/13:30 truncation, 12:30 in-progress trap, and 15:15-15:30 closing bar."""
    ltf_df, htf_df = _make_5m_session_df()
    cutoffs = [
        datetime(2026, 1, 15, 12, 15, tzinfo=IST_TZ),
        datetime(2026, 1, 15, 12, 30, tzinfo=IST_TZ),  # in-progress 1h bar trap
        datetime(2026, 1, 15, 13, 30, tzinfo=IST_TZ),
    ]

    res = audit_mtf_backward_alignment(
        ltf_df,
        htf_df,
        htf_cols=["htf_close", "htf_ema_9"],
        cutoffs_ist=cutoffs,
    )
    assert res.passed is True, res.violations
    assert res.metrics["closing_bar_sub_bars"] == 3


@pytest.mark.fast_ci
def test_clean_data_contract_rejects_unclosed_and_delayed_ts_init_and_passes_cutoff() -> None:
    """build_ohlcv_evidence_frame rejects unclosed & delayed bars; causal filter passes audit_ts_init_cutoff."""
    df = _make_1h_synthetic_df(n_bars=80)
    as_of = df["ts_close_ist"].iloc[-1].to_pydatetime()

    # Direct contract check: unclosed bar (ts_open < as_of < ts_close) must raise LookaheadViolationError
    unclosed = pd.DataFrame(
        {
            "ts_open_ist": [as_of - timedelta(minutes=20)],
            "ts_close_ist": [as_of + timedelta(minutes=40)],
            "ts_init_ist": [as_of + timedelta(minutes=40)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        }
    )
    with pytest.raises(LookaheadViolationError):
        build_ohlcv_evidence_frame(
            unclosed,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="test_contract",
        )

    # Direct contract check: delayed ts_init > as_of must raise LookaheadViolationError
    delayed = pd.DataFrame(
        {
            "ts_open_ist": [as_of - timedelta(hours=1)],
            "ts_close_ist": [as_of],
            "ts_init_ist": [as_of + timedelta(minutes=5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        }
    )
    with pytest.raises(LookaheadViolationError):
        build_ohlcv_evidence_frame(
            delayed,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="test_contract",
        )

    # Causal feature_fn that filters ts_close_ist <= as_of and ts_init_ist <= as_of before build_ohlcv_evidence_frame
    def causal_evidence_ema(frame: pd.DataFrame) -> float:
        visible = frame.loc[
            (frame["ts_close_ist"] <= as_of) & (frame["ts_init_ist"] <= as_of)
        ].copy()
        ev = build_ohlcv_evidence_frame(
            visible,
            symbol="RELIANCE",
            timeframe="1h",
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="test_contract",
        )
        return float(compute_ema_series(ev.frame["close"], span=9, min_periods=9).iloc[-1])

    res_ts_init = audit_ts_init_cutoff(df, causal_evidence_ema, as_of)
    assert res_ts_init.passed is True, res_ts_init.violations


@pytest.mark.fast_ci
def test_clean_cycle006c_event_filter_passes_future_event_exclusion() -> None:
    """Strict pub <= as_of & col <= as_of event filter passes audit_future_event_exclusion with 0 violations."""
    as_of = datetime(2026, 1, 15, 11, 30, tzinfo=IST_TZ)
    events_df = _make_events_df(as_of)

    res = audit_future_event_exclusion(events_df, _strict_event_filter, as_of)
    assert res.passed is True, res.violations
    assert res.metrics["excluded_count"] == 4


# ===========================================================================
# TIER 2 — @pytest.mark.research_slow (Committed data/intraday/*_1h.parquet)
# ===========================================================================


def _load_committed_1h(symbol: str) -> pd.DataFrame:
    path = PARQUET_DIR / f"{symbol}_1h.parquet"
    df = pd.read_parquet(path)
    ts_open = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST_TZ)
    # NSE 15:15 bar closes at 15:30 (15m); all other 1h bars close +60m
    ts_close = ts_open.apply(
        lambda t: t + timedelta(minutes=15) if (t.hour == 15 and t.minute == 15) else t + timedelta(hours=1)
    )
    out = df.copy()
    out["ts_open_ist"] = ts_open
    out["ts_close_ist"] = ts_close
    out["ts_init_ist"] = ts_close
    return out.sort_values("ts_close_ist").reset_index(drop=True)


@pytest.mark.research_slow
@pytest.mark.parametrize("symbol", LAB_DEFAULT_UNIVERSE)
def test_real_1h_parquet_recursive_convergence_and_rolling_truncation(symbol: str) -> None:
    """Real committed *_1h.parquet sweep: recursive convergence (W in [50,100,200,500]) & rolling truncation."""
    df = _load_committed_1h(symbol)
    assert len(df) >= 600, f"{symbol}_1h.parquet expected >= 600 bars, got {len(df)}"

    # 1. Recursive convergence on real price series for EMA(9/21/50) and Wilder RSI(14)
    for span in (9, 21, 50):
        min_bars = DEFAULT_FEATURE_REGISTRY[f"ema_{span}"].min_bars_required
        res_conv = audit_recursive_convergence(
            df["close"],
            lambda s, sp=span, mb=min_bars: compute_ema_series(s, span=sp, min_periods=mb),
            span_or_period=span,
            windows=(50, 100, 200, 500),
            ref_window=500,
        )
        assert res_conv.passed is True, f"{symbol} EMA({span}) failed convergence: {res_conv.violations}"

    # 2. Rolling bar-index truncation sweep across 5 cutoffs
    n = len(df)
    cutoffs = [
        df["ts_close_ist"].iloc[int(n * frac)].to_pydatetime()
        for frac in (0.25, 0.45, 0.65, 0.80, 0.90)
    ]

    def causal_bundle(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ema_9": compute_ema_series(frame["close"], span=9, min_periods=9),
                "ema_21": compute_ema_series(frame["close"], span=21, min_periods=21),
                "ema_50": compute_ema_series(frame["close"], span=50, min_periods=50),
                "atr_14": compute_atr_series(
                    frame["high"], frame["low"], frame["close"], period=14, method="wilder", min_periods=14
                ),
            },
            index=frame.index,
        )

    res_la = audit_bar_index_lookahead(df, causal_bundle, cutoffs)
    assert res_la.passed is True, f"{symbol} rolling truncation failed: {res_la.violations}"
