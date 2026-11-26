"""CYCLE-019..022 Comprehensive Test Suite.

Verifies:
- CYCLE-019: AUTO live market direction Python <-> JS parity, overlay toggle bar DOM,
  and browser engine AUTO mode.
- CYCLE-020: Fast causal 1m walk-forward optimizer zero-lookahead invariance and
  strictly chronological Train / OOS Test split across profiles & capital tiers.
- CYCLE-021: Live market-hours read-only paper-trading runner safety invariants and
  end-to-end offline JSONL tick replay + Parquet archival + live_paper_state.json.
- CYCLE-022: Leak-free walk-forward ML setup scorer training, calibration metrics,
  JSON SHA-256 round-trip & tamper detection, and OOS ML-filtered evaluation.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from trader.config import IST_TZ
from trader.data.loader_1m import OneMinuteSessionBundle
from trader.data.schema import Candle5M
from trader.data.tick_recorder import OfflineJsonlTickAdapter
from trader.evaluation.walkforward_optimizer import (
    FastSessionConfig,
    precompute_causal_1m_arrays,
    run_walkforward_optimization,
    simulate_fast_1m_session,
)
from trader.intelligence.chart_overlays import (
    ChartOverlaySnapshot,
    infer_auto_market_direction,
)
from trader.intelligence.ml_setup_scorer import (
    CausalSetupScorerModel,
    compute_brier_score,
    compute_log_loss,
    compute_roc_auc,
    train_walkforward_setup_scorer,
)
from trader.timing.live_paper_runner import (
    LIVE_TRADING_ENABLED,
    READ_ONLY_MODE,
    LivePaperConfig,
    LivePaperTradingRunner,
    assert_paper_only_live_runner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synthetic_day_bundle(
    symbol: str,
    trading_date: date,
    *,
    base_price: float = 1000.0,
    trend_per_bar: float = 0.25,
    n_bars: int = 120,
    seed: int = 42,
) -> OneMinuteSessionBundle:
    """Create a deterministic in-memory 1m bundle for fast unit testing (CI-safe)."""
    rng = np.random.default_rng(seed)
    prev_day = trading_date - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)

    warmup: list[Candle5M] = []
    px = base_price - 5.0
    t_warm = datetime(prev_day.year, prev_day.month, prev_day.day, 14, 30, tzinfo=IST_TZ)
    for k in range(45):
        o = px
        c = max(10.0, o + float(rng.normal(0.0, 0.3)))
        h = max(o, c) + 0.4
        l = min(o, c) - 0.4
        warmup.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t_warm + timedelta(minutes=k),
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=10_000.0,
            )
        )
        px = c

    session: list[Candle5M] = []
    t_open = datetime(trading_date.year, trading_date.month, trading_date.day, 9, 15, tzinfo=IST_TZ)
    for k in range(n_bars):
        o = px
        drift = trend_per_bar if k >= 15 else 0.05
        c = max(10.0, o + drift + float(rng.normal(0.0, 0.25)))
        h = max(o, c) + 0.45
        l = min(o, c) - 0.45
        session.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t_open + timedelta(minutes=k),
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=12_000.0 + float(k * 50),
            )
        )
        px = c

    return OneMinuteSessionBundle(
        symbol=symbol,
        trading_date=trading_date,
        warmup_bars=warmup,
        session_bars=session,
        source_providers=("unit_test_bundle",),
    )


# ---------------------------------------------------------------------------
# CYCLE-019 Tests
# ---------------------------------------------------------------------------


def test_cycle019_auto_direction_python_js_parity() -> None:
    """Python infer_auto_market_direction and JS inferAutoMarketDirection must agree on all regimes."""
    cases = [
        # (mtf, st_dir, close, vwap, orb_ready, orb_state, expected)
        ("BULLISH", "BULLISH", 1010.0, 1000.0, True, "ORB_BREAKOUT_UP", "BULLISH"),
        ("BEARISH", "BEARISH", 990.0, 1000.0, True, "ORB_BREAKOUT_DOWN", "BEARISH"),
        ("BULLISH", "BEARISH", 1000.0, 1000.0, False, "ORB_FORMING", "NEUTRAL"),
        ("NEUTRAL", "BULLISH", 1005.0, 1000.0, True, "INSIDE_ORB", "BULLISH"),
    ]

    for mtf, st_dir, close, vwap, orb_ready, orb_state, expected in cases:
        snap = ChartOverlaySnapshot(
            adx_14=25.0,
            plus_di_14=28.0,
            minus_di_14=16.0,
            market_regime="TRENDING",
            supertrend_value=vwap,
            supertrend_direction=st_dir,
            pdh=None,
            pdl=None,
            orb_high=None,
            orb_low=None,
            orb_ready=orb_ready,
            orb_state=orb_state,
            vwap=vwap,
            vwap_upper_1sd=None,
            vwap_lower_1sd=None,
            vwap_upper_2sd=None,
            vwap_lower_2sd=None,
            vwap_stretch="NORMAL",
            near_key_level=False,
            nearest_key_level_name=None,
            liquidity_sweep="NONE",
            recent_fvg=None,
        )
        py_dir = infer_auto_market_direction(mtf_5m_trend=mtf, overlays=snap, current_close=close)
        assert py_dir == expected

    # Verify JS inferAutoMarketDirection matches all 4 cases identically
    node_script = (
        "const eng = require('./docs/candle-agent/engine.js');\n"
        "const cases = " + json.dumps(cases) + ";\n"
        "const out = cases.map(([mtf, st, close, vwap, orbReady, orbState]) => "
        "eng.inferAutoMarketDirection(mtf, {supertrendDirection: st, orbReady, orbState}, close, vwap));\n"
        "console.log(JSON.stringify(out));\n"
    )
    res = subprocess.run(
        ["node", "-e", node_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    js_results = json.loads(res.stdout.strip())
    assert js_results == [c[6] for c in cases]


def test_cycle019_dashboard_overlay_toggles_and_auto_option() -> None:
    """docs/candle-agent/index.html must expose AUTO default and all 6 chart overlay toggles."""
    html = (REPO_ROOT / "docs/candle-agent/index.html").read_text(encoding="utf-8")
    assert 'value="AUTO" selected' in html
    assert 'id="overlay-toggles"' in html
    for cb_id in ("ov-supertrend", "ov-vwap-bands", "ov-orb", "ov-pdhl", "ov-fvg", "ov-chandelier"):
        assert f'id="{cb_id}"' in html


# ---------------------------------------------------------------------------
# CYCLE-020 Tests
# ---------------------------------------------------------------------------


def test_cycle020_precompute_zero_lookahead_invariance() -> None:
    """Mutating future bars (> t) must NEVER alter precomputed causal indicators at bar t."""
    b1 = _make_synthetic_day_bundle("TCS", date(2026, 10, 12), trend_per_bar=0.30, n_bars=80, seed=7)
    # Create identical bundle up to bar 40, then violently spike bars 41..79
    mutated_session = list(b1.session_bars[:41])
    for bar in b1.session_bars[41:]:
        mutated_session.append(
            Candle5M(
                symbol=bar.symbol,
                exchange=bar.exchange,
                timestamp=bar.timestamp,
                open=bar.open * 1.5,
                high=bar.high * 1.6,
                low=bar.low * 1.4,
                close=bar.close * 1.55,
                volume=bar.volume * 10.0,
            )
        )
    b2 = OneMinuteSessionBundle(
        symbol=b1.symbol,
        trading_date=b1.trading_date,
        warmup_bars=b1.warmup_bars,
        session_bars=mutated_session,
        source_providers=b1.source_providers,
    )

    a1 = precompute_causal_1m_arrays(b1)
    a2 = precompute_causal_1m_arrays(b2)
    for key in ("ema9", "ema21", "rsi14", "atr14", "adx14", "st_dir", "vwap", "orb_high", "mtf_5m_dir"):
        np.testing.assert_allclose(a1[key][:41], a2[key][:41], equal_nan=True)


def test_cycle020_walkforward_chronological_split_and_tiers() -> None:
    """Walk-forward optimizer must enforce max(train_dates) < min(test_dates) across capital tiers."""
    dates = [date(2026, 10, 12) + timedelta(days=d) for d in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]
    bundles = {
        "TCS": [_make_synthetic_day_bundle("TCS", d, trend_per_bar=0.35, seed=10 + i) for i, d in enumerate(dates)],
        "ITC": [_make_synthetic_day_bundle("ITC", d, base_price=450.0, trend_per_bar=-0.20, seed=50 + i) for i, d in enumerate(dates)],
    }
    report = run_walkforward_optimization(
        bundles_override=bundles,
        capital_tiers=(50_000.0, 200_000.0),
        train_ratio=0.70,
    )
    assert report["live_trading_enabled"] is False
    assert report["zero_lookahead_verified"] is True
    assert report["train_date_range"][1] < report["oos_test_date_range"][0]
    assert report["train_days"] == 7
    assert report["oos_test_days"] == 3
    assert "INR_50000" in report["capital_tiers"]
    assert "INR_200000" in report["capital_tiers"]
    assert report["capital_tiers"]["INR_50000"]["train_selected_profile"] is not None


# ---------------------------------------------------------------------------
# CYCLE-021 Tests
# ---------------------------------------------------------------------------


def test_cycle021_read_only_safety_invariants() -> None:
    """LivePaperTradingRunner must enforce LIVE_TRADING_ENABLED=False and READ_ONLY_MODE=True."""
    assert LIVE_TRADING_ENABLED is False
    assert READ_ONLY_MODE is True
    assert_paper_only_live_runner()
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        assert_paper_only_live_runner(live_trading_enabled=True)
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        assert_paper_only_live_runner(read_only=False)

    src = (REPO_ROOT / "trader/timing/live_paper_runner.py").read_text(encoding="utf-8")
    for forbidden in ("place_order", "modify_order", "cancel_order", "kite.order"):
        assert forbidden not in src


def test_cycle021_live_paper_runner_offline_jsonl_end_to_end(tmp_path: Path) -> None:
    """End-to-end offline JSONL tick stream -> causal 1m bars -> Parquet archive + live_paper_state.json."""
    jsonl_path = tmp_path / "ticks.jsonl"
    base_ts = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST_TZ)
    lines = []
    px = 3500.0
    cum_vol = 0.0
    # 35 minutes of ticks (2 ticks/min) so ORB forms and bars evaluate causally
    for m in range(35):
        for sec in (10, 45):
            ts = base_ts + timedelta(minutes=m, seconds=sec)
            px += 0.40 if m >= 15 else 0.05
            cum_vol += 500.0
            lines.append(
                json.dumps(
                    {
                        "symbol": "TCS",
                        "exchange": "NSE",
                        "timestamp_ist": ts.isoformat(),
                        "ltp": round(px, 2),
                        "cumulative_volume": cum_vol,
                    }
                )
            )
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    adapter = OfflineJsonlTickAdapter(jsonl_path=jsonl_path)
    out_dir = tmp_path / "live_out"
    runner = LivePaperTradingRunner(
        adapter=adapter,
        output_dir=out_dir,
        config=LivePaperConfig(strategy_profile="ORB_BREAKOUT", initial_capital_inr=50_000.0),
    )
    summary = runner.run(["TCS"])

    assert summary.live_trading_enabled is False
    assert summary.read_only_mode is True
    assert summary.ticks_accepted == 70
    assert summary.bars_1m_evaluated == 35
    assert len(summary.archived_provenance_files) == 1
    assert Path(summary.archived_provenance_files[0]).exists()
    assert summary.state_json_path is not None
    state_data = json.loads(Path(summary.state_json_path).read_text(encoding="utf-8"))
    assert state_data["live_trading_enabled"] is False
    assert state_data["bars_1m_evaluated"] == 35


# ---------------------------------------------------------------------------
# CYCLE-022 Tests
# ---------------------------------------------------------------------------


def test_cycle022_calibration_metrics_and_ml_scorer_roundtrip(tmp_path: Path) -> None:
    """Verify Brier/LogLoss/ROC-AUC math, leak-free walk-forward ML training, and SHA-256 JSON integrity."""
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_brier_score(y, p) < 0.05
    assert compute_log_loss(y, p) < 0.25
    assert compute_roc_auc(y, p) == pytest.approx(1.0)

    dates = [date(2026, 10, 12) + timedelta(days=d) for d in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]
    bundles = {
        "TCS": [
            _make_synthetic_day_bundle("TCS", d, trend_per_bar=0.45 if i % 2 == 0 else -0.35, seed=100 + i)
            for i, d in enumerate(dates)
        ],
    }
    model, report = train_walkforward_setup_scorer(
        bundles_override=bundles,
        strategy_profile="BALANCED_ALL",
        initial_capital_inr=50_000.0,
        train_ratio=0.70,
    )
    assert report["live_trading_enabled"] is False
    assert report["zero_lookahead_verified"] is True
    assert model.train_date_range[1] < model.oos_test_date_range[0]
    assert model.calibration_status in ("EMPIRICAL_WALKFORWARD_CALIBRATED", "UNCALIBRATED")

    sample_feat = {"adx_14": 28.0, "side_sign": 1.0, "reward_to_friction": 5.0}
    sample_prob = model.predict_proba(sample_feat)
    assert 0.0 <= sample_prob <= 1.0

    # JSON save/load roundtrip + SHA-256 tamper detection
    model_path = tmp_path / "setup_scorer_1m.json"
    model.save_json(model_path)
    loaded = CausalSetupScorerModel.load_json(model_path)
    assert loaded.weights_sha256 == model.weights_sha256
    assert loaded.predict_proba(sample_feat) == pytest.approx(sample_prob)

    # Tampering with weights must fail SHA-256 verification
    tampered = json.loads(model_path.read_text(encoding="utf-8"))
    tampered["bias"] = tampered["bias"] + 9.99
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CausalSetupScorerModel.load_json(tampered_path)
