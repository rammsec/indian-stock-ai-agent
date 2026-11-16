"""Multi-Horizon Signal Combiner (1D, 5D, 10D) (Phase 5 — Section 6).

Clearly separates:
- Quantitative model signal
- News & event intelligence signal
- Cross-stock & macro relationship signal
before combining them into calibrated 1D, 5D, and 10D outlooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import CrossStockImpactItem, MultiHorizonSignalSummary

_HORIZON_PARAMS: dict[str, dict[str, Any]] = {
    "1D": {
        "news_w": 0.25,
        "cross_w": 0.12,
        "cost_pct": 0.18,
        "buy_prob": 0.54,
        "prob_key": "prob_next_day_up",
        "ret_key": "expected_next_day_return_pct",
    },
    "5D": {
        "news_w": 0.22,
        "cross_w": 0.15,
        "cost_pct": 0.22,
        "buy_prob": 0.54,
        "prob_key": "prob_5d_up",
        "ret_key": "expected_5d_return_pct",
    },
    "10D": {
        "news_w": 0.18,
        "cross_w": 0.18,
        "cost_pct": 0.24,
        "buy_prob": 0.55,
        "prob_key": "prob_10d_up",
        "ret_key": "expected_10d_return_pct",
    },
}


def _label_from_score(score: float, pos_thresh: float = 0.08, neg_thresh: float = -0.08) -> str:
    if score >= pos_thresh:
        return "BULLISH"
    if score <= neg_thresh:
        return "BEARISH"
    return "NEUTRAL"


def _optional_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except Exception:
        return None


def build_multi_horizon_outlooks(
    symbol: str,
    quant_pred: dict[str, Any] | None,
    news_features: dict[str, float],
    cross_impacts: list[CrossStockImpactItem],
    regime_label: str = "Sideways",
    trade_eligible: bool = True,
    data_freshness: str = "Current",
) -> dict[str, MultiHorizonSignalSummary]:
    """Generate separated & combined 1D, 5D, and 10D outlooks with strict horizon isolation."""
    q_conf = float(quant_pred.get("confidence_score", 50.0)) / 100.0 if quant_pred else 0.40
    q_cost = float(quant_pred.get("roundtrip_delivery_cost_pct", 0.22)) if quant_pred else 0.22
    pred_date = str(quant_pred.get("observation_date") or "") or None if quant_pred else None

    # News score in [-1, 1] (always independent of quant availability)
    raw_news_score = (
        0.45 * float(news_features.get("news_company_sentiment_5d", 0.0))
        + 0.20 * float(news_features.get("news_promoter_sentiment_5d", 0.0))
        + 0.20 * float(news_features.get("news_industry_sentiment_5d", 0.0))
        + 0.15 * float(news_features.get("news_novelty_weighted_sentiment", 0.0))
        - 0.10 * float(news_features.get("news_regulatory_risk_score", 0.0))
        + 0.10 * float(news_features.get("news_regulatory_positive_score", 0.0))
    )
    news_score = float(np.clip(raw_news_score, -1.0, 1.0))

    # Cross-stock score in [-1, 1] (always independent of quant availability)
    cross_raw = float(news_features.get("news_cross_stock_net_impact", 0.0))
    if cross_raw == 0.0 and cross_impacts:
        s_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        m_map = {"low": 0.5, "medium": 1.0, "high": 1.5}
        cross_raw = sum(
            s_map.get(c.expected_impact_direction, 0.0) * m_map.get(c.impact_magnitude, 1.0) * c.confidence
            for c in cross_impacts
        )
    cross_score = float(np.clip(cross_raw / 2.0, -1.0, 1.0))

    news_label = _label_from_score(news_score, 0.08, -0.08)
    cross_label = _label_from_score(cross_score, 0.08, -0.08)

    out: dict[str, MultiHorizonSignalSummary] = {}
    for horizon in ("1D", "5D", "10D"):
        hp = _HORIZON_PARAMS[horizon]
        news_w = float(hp["news_w"])
        cross_w = float(hp["cross_w"])
        quant_w = 1.0 - news_w - cross_w

        q_prob_h = _optional_float(quant_pred.get(hp["prob_key"])) if quant_pred else None
        q_ret_h = _optional_float(quant_pred.get(hp["ret_key"])) if quant_pred else None

        # Strict Phase 5.2 rule: never substitute 1D prob or scaled 5D return if horizon model is missing
        if q_prob_h is None or q_ret_h is None:
            out[horizon] = MultiHorizonSignalSummary(
                horizon=horizon,  # type: ignore[arg-type]
                prob_positive=None,
                quant_prob_positive=None,
                expected_gross_return_pct=None,
                expected_net_return_pct=None,
                quant_signal="UNAVAILABLE",
                news_signal=news_label,
                cross_stock_signal=cross_label,
                combined_signal="UNAVAILABLE",
                confidence=0.0,
                uncertainty=100.0,
                decision="UNAVAILABLE",
                model_prediction_date=pred_date,
                data_freshness=data_freshness,
                is_model_generated=False,
                status_reason=(
                    f"Genuine {horizon} quantitative model prediction unavailable; "
                    "refusing to substitute 1D probability or scaled 5D return."
                ),
            )
            continue

        quant_score = float(np.clip((q_prob_h - 0.50) * 6.0, -1.0, 1.0))
        quant_label = _label_from_score(quant_score, 0.05, -0.05)

        adj_ret = (news_score * 0.45 * news_w + cross_score * 0.35 * cross_w) * (1.0 if horizon == "1D" else 2.2)
        exp_gross = round(float(q_ret_h + adj_ret), 3)
        cost = q_cost if horizon != "1D" else min(q_cost, float(hp["cost_pct"]))
        exp_net = round(float(exp_gross - cost), 3)

        comb_prob = float(np.clip(q_prob_h + 0.06 * news_w * news_score + 0.05 * cross_w * cross_score, 0.05, 0.95))
        comb_score = quant_w * quant_score + news_w * news_score + cross_w * cross_score
        comb_label = _label_from_score(comb_score, 0.06, -0.06)

        signs = [np.sign(quant_score), np.sign(news_score), np.sign(cross_score)]
        nonzero_signs = [s for s in signs if s != 0]
        agreement = 1.0 if len(set(nonzero_signs)) <= 1 else 0.65
        confidence = round(float(np.clip((0.55 * q_conf + 0.45 * agreement) * 100.0, 15.0, 92.0)), 1)
        uncertainty = round(float(np.clip(100.0 - confidence + (12.0 if regime_label == "High Volatility" else 0.0), 8.0, 88.0)), 1)

        if not trade_eligible or regime_label == "High Volatility":
            decision = "NO TRADE"
        elif comb_prob >= float(hp["buy_prob"]) and exp_net > 0.05 and comb_label == "BULLISH":
            decision = "BUY"
        elif comb_prob >= 0.50 and exp_net >= -0.05:
            decision = "HOLD"
        elif comb_prob < 0.46 or exp_net < -0.25:
            decision = "NO TRADE"
        else:
            decision = "WATCH"

        out[horizon] = MultiHorizonSignalSummary(
            horizon=horizon,  # type: ignore[arg-type]
            prob_positive=round(comb_prob, 4),
            quant_prob_positive=round(q_prob_h, 4),
            expected_gross_return_pct=exp_gross,
            expected_net_return_pct=exp_net,
            quant_signal=quant_label,
            news_signal=news_label,
            cross_stock_signal=cross_label,
            combined_signal=comb_label,
            confidence=confidence,
            uncertainty=uncertainty,
            decision=decision,  # type: ignore[arg-type]
            model_prediction_date=pred_date,
            data_freshness=data_freshness,
            is_model_generated=True,
            status_reason=f"Genuine {horizon} quantitative model combined with independent news and cross-stock signals.",
        )

    return out
