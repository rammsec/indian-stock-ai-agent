"""CYCLE-024: Multi-Stock Causal Intraday Opportunity Scanner & Watchlist Ranker.

Scans multiple NSE equities simultaneously at any causal `as_of_ist` timestamp (using ONLY
bars closed at or before `as_of_ist`), computes live `AUTO` directional bias, `ADX(14)`
regime, `Supertrend(10,3)`, `ORB 15m`, `VWAP ±2σ` stretch, `Liquidity Sweeps`, `FVG`, and
optional `CYCLE-022` ML win probability, and ranks symbols into a Top Setup Watchlist.

HARD INVARIANTS:
- Strictly paper-only & read-only (`LIVE_TRADING_ENABLED = False`).
- Zero lookahead: strictly filters `timestamp <= as_of_ist`.
- Deterministic scoring (`0.0` to `100.0`) with transparent component breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Final, Literal, Mapping

import pandas as pd

from trader.config import IST_TZ
from trader.data.loader_1m import OneMinuteSessionBundle
from trader.intelligence.chart_overlays import (
    compute_chart_overlays,
    infer_auto_market_direction,
)
from trader.timing.monitoring_session import evaluate_causal_5m_mtf_trend

LIVE_TRADING_ENABLED: Final[bool] = False

ActionTagLiteral = Literal["TOP_LONG", "TOP_SHORT", "WATCH_LONG", "WATCH_SHORT", "CHOP_SKIP"]


@dataclass(frozen=True)
class StockOpportunityRow:
    """Single symbol's causal intraday opportunity snapshot."""

    rank: int
    symbol: str
    as_of_ist: str
    last_close: float
    auto_direction: str
    mtf_5m_trend: str
    supertrend_direction: str
    adx_14: float | None
    market_regime: str
    orb_state: str
    vwap_stretch: str
    liquidity_sweep: str
    fvg_direction: str | None
    opportunity_score: float
    action_tag: ActionTagLiteral
    ml_win_prob: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "as_of_ist": self.as_of_ist,
            "last_close": round(self.last_close, 2),
            "auto_direction": self.auto_direction,
            "mtf_5m_trend": self.mtf_5m_trend,
            "supertrend_direction": self.supertrend_direction,
            "adx_14": round(self.adx_14, 2) if self.adx_14 is not None else None,
            "market_regime": self.market_regime,
            "orb_state": self.orb_state,
            "vwap_stretch": self.vwap_stretch,
            "liquidity_sweep": self.liquidity_sweep,
            "fvg_direction": self.fvg_direction,
            "opportunity_score": round(self.opportunity_score, 2),
            "action_tag": self.action_tag,
            "ml_win_prob": round(self.ml_win_prob, 4) if self.ml_win_prob is not None else None,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MultiStockScanResult:
    """Ranked multi-stock intraday opportunity watchlist at a causal timestamp."""

    as_of_ist: str
    symbols_scanned: int
    top_pick: StockOpportunityRow | None
    ranked_opportunities: tuple[StockOpportunityRow, ...]
    live_trading_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cycle024_multi_stock_scanner_v1",
            "live_trading_enabled": False,
            "as_of_ist": self.as_of_ist,
            "symbols_scanned": self.symbols_scanned,
            "top_pick": self.top_pick.to_dict() if self.top_pick else None,
            "ranked_opportunities": [r.to_dict() for r in self.ranked_opportunities],
        }


def _bundle_to_causal_df(
    bundle: OneMinuteSessionBundle,
    as_of_ist: datetime | None,
) -> pd.DataFrame:
    rows = []
    cutoff = as_of_ist.astimezone(IST_TZ) if as_of_ist is not None else None
    for b in list(bundle.warmup_bars) + list(bundle.session_bars):
        ts = b.timestamp.astimezone(IST_TZ)
        if cutoff is not None and ts > cutoff:
            continue
        rows.append(
            {
                "timestamp": ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            }
        )
    return pd.DataFrame(rows)


def scan_multi_stock_opportunities(
    bundles_by_symbol: Mapping[str, OneMinuteSessionBundle],
    *,
    as_of_ist: datetime | None = None,
    ml_scorer: Callable[[dict[str, float]], float] | None = None,
) -> MultiStockScanResult:
    """Causally evaluate and rank intraday trading setups across multiple NSE stocks."""
    if not bundles_by_symbol:
        ts_str = (as_of_ist.astimezone(IST_TZ) if as_of_ist else datetime.now(IST_TZ)).isoformat()
        return MultiStockScanResult(
            as_of_ist=ts_str,
            symbols_scanned=0,
            top_pick=None,
            ranked_opportunities=(),
            live_trading_enabled=False,
        )

    raw_rows: list[StockOpportunityRow] = []
    latest_ts_seen: datetime | None = None

    for sym, bundle in sorted(bundles_by_symbol.items()):
        df = _bundle_to_causal_df(bundle, as_of_ist)
        if df.empty:
            continue
        last_row = df.iloc[-1]
        bar_ts: datetime = last_row["timestamp"]
        if latest_ts_seen is None or bar_ts > latest_ts_seen:
            latest_ts_seen = bar_ts
        close_px = float(last_row["close"])

        overlays = compute_chart_overlays(df, session_date=bar_ts.date())
        mtf_5m = evaluate_causal_5m_mtf_trend(df)
        auto_dir = infer_auto_market_direction(
            mtf_5m_trend=mtf_5m,
            overlays=overlays,
            current_close=close_px,
        )

        # Deterministic transparent scoring (0..100)
        score = 20.0
        reasons: list[str] = []

        if auto_dir in ("BULLISH", "BEARISH"):
            score += 25.0
            reasons.append(f"AUTO_{auto_dir}")
        else:
            reasons.append("AUTO_NEUTRAL")

        adx = overlays.adx_14
        if adx is not None:
            if adx >= 25.0:
                score += 20.0
                reasons.append(f"ADX_STRONG({adx:.1f})")
            elif adx >= 20.0:
                score += 12.0
                reasons.append(f"ADX_TRENDING({adx:.1f})")
            else:
                score -= 15.0
                reasons.append(f"ADX_CHOP({adx:.1f})")

        if auto_dir == "BULLISH" and overlays.orb_state == "ORB_BREAKOUT_UP":
            score += 15.0
            reasons.append("ORB_UP")
        elif auto_dir == "BEARISH" and overlays.orb_state == "ORB_BREAKOUT_DOWN":
            score += 15.0
            reasons.append("ORB_DOWN")

        if auto_dir == overlays.supertrend_direction and auto_dir != "NEUTRAL":
            score += 10.0
            reasons.append("SUPERTREND_ALIGNED")

        if auto_dir == mtf_5m and auto_dir != "NEUTRAL":
            score += 10.0
            reasons.append("MTF_5M_ALIGNED")

        if (auto_dir == "BULLISH" and overlays.liquidity_sweep == "BULLISH_SWEEP") or (
            auto_dir == "BEARISH" and overlays.liquidity_sweep == "BEARISH_SWEEP"
        ):
            score += 8.0
            reasons.append(overlays.liquidity_sweep)

        fvg_dir = overlays.recent_fvg.direction if overlays.recent_fvg is not None else None
        if fvg_dir == auto_dir and auto_dir != "NEUTRAL":
            score += 6.0
            reasons.append(f"FVG_{fvg_dir}")

        # Anti-chase VWAP ±2σ stretch penalty
        if (auto_dir == "BULLISH" and overlays.vwap_stretch == "OVEREXTENDED_ABOVE_2SD") or (
            auto_dir == "BEARISH" and overlays.vwap_stretch == "OVEREXTENDED_BELOW_2SD"
        ):
            score -= 22.0
            reasons.append("VWAP_2SD_OVEREXTENDED")

        ml_prob: float | None = None
        if ml_scorer is not None and auto_dir in ("BULLISH", "BEARISH"):
            side_sign = 1.0 if auto_dir == "BULLISH" else -1.0
            feat = {
                "side_sign": side_sign,
                "adx_14": float(adx or 0.0),
                "di_diff_aligned": side_sign * ((overlays.plus_di_14 or 0.0) - (overlays.minus_di_14 or 0.0)),
                "mtf_5m_aligned": 1.0 if mtf_5m == auto_dir else 0.0,
                "orb_breakout_aligned": 1.0
                if (
                    (auto_dir == "BULLISH" and overlays.orb_state == "ORB_BREAKOUT_UP")
                    or (auto_dir == "BEARISH" and overlays.orb_state == "ORB_BREAKOUT_DOWN")
                )
                else 0.0,
                "sweep_aligned": 1.0
                if (
                    (auto_dir == "BULLISH" and overlays.liquidity_sweep == "BULLISH_SWEEP")
                    or (auto_dir == "BEARISH" and overlays.liquidity_sweep == "BEARISH_SWEEP")
                )
                else 0.0,
                "fvg_aligned": 1.0 if fvg_dir == auto_dir else 0.0,
                "near_pdhl": 1.0 if overlays.near_key_level else 0.0,
            }
            ml_prob = float(ml_scorer(feat))
            score += (ml_prob - 0.25) * 20.0
            reasons.append(f"ML_P={ml_prob:.2f}")

        score = max(0.0, min(100.0, score))

        if auto_dir == "NEUTRAL" or (adx is not None and adx < 20.0):
            tag: ActionTagLiteral = "CHOP_SKIP"
        elif auto_dir == "BULLISH":
            tag = "TOP_LONG" if score >= 70.0 else "WATCH_LONG"
        else:
            tag = "TOP_SHORT" if score >= 70.0 else "WATCH_SHORT"

        raw_rows.append(
            StockOpportunityRow(
                rank=0,
                symbol=sym.strip().upper(),
                as_of_ist=bar_ts.isoformat(),
                last_close=close_px,
                auto_direction=auto_dir,
                mtf_5m_trend=mtf_5m,
                supertrend_direction=overlays.supertrend_direction,
                adx_14=adx,
                market_regime=overlays.market_regime,
                orb_state=overlays.orb_state,
                vwap_stretch=overlays.vwap_stretch,
                liquidity_sweep=overlays.liquidity_sweep,
                fvg_direction=fvg_dir,
                opportunity_score=score,
                action_tag=tag,
                ml_win_prob=ml_prob,
                reasons=tuple(reasons),
            )
        )

    raw_rows.sort(key=lambda r: (-r.opportunity_score, r.symbol))
    ranked = tuple(
        StockOpportunityRow(**{**r.__dict__, "rank": idx + 1})
        for idx, r in enumerate(raw_rows)
    )
    resolved_as_of = (
        as_of_ist.astimezone(IST_TZ).isoformat()
        if as_of_ist is not None
        else (latest_ts_seen.isoformat() if latest_ts_seen is not None else datetime.now(IST_TZ).isoformat())
    )
    return MultiStockScanResult(
        as_of_ist=resolved_as_of,
        symbols_scanned=len(ranked),
        top_pick=ranked[0] if ranked else None,
        ranked_opportunities=ranked,
        live_trading_enabled=False,
    )
