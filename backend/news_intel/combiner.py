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

_HORIZON_PARAMS: dict[str, dict[str, float]] = {
    "1D": {"news_w": 0.25, "cross_w": 0.12, "ret_scale": 1.0, "cost_pct": 0.18, "buy_prob": 0.54},
    "5D": {"news_w": 0.22, "cross_w": 0.15, "ret_scale": 1.0, "cost_pct": 0.22, "buy_prob": 0.54},
    "10D": {"news_w": 0.18, "cross_w": 0.18, "ret_scale": 1.55, "cost_pct": 0.24, "buy_prob": 0.55},
}


def _label_from_score(score: float, pos_thresh: float = 0.08, neg_thresh: float = -0.08) -> str:
    if score >= pos_thresh:
        return "BULLISH"
    if score <= neg_thresh:
        return "BEARISH"
    return "NEUTRAL"


def build_multi_horizon_outlooks(
    symbol: str,
    quant_pred: dict[str, Any] | None,
    news_features: dict[str, float],
    cross_impacts: list[CrossStockImpactItem],
    regime_label: str = "Sideways",
    trade_eligible: bool = True,
) -> dict[str, MultiHorizonSignalSummary]:
    """Generate separated & combined 1D, 5D, and 10D outlooks for a pilot stock."""
    q_prob_1d = float(quant_pred.get("prob_next_day_up", 0.50)) if quant_pred else 0.50
    q_ret_1d = float(quant_pred.get("expected_next_day_return_pct", 0.0)) if quant_pred else 0.0
    q_ret_5d = float(quant_pred.get("expected_5d_return_pct", 0.0)) if quant_pred else 0.0
    q_conf = float(quant_pred.get("confidence_score", 50.0)) / 100.0 if quant_pred else 0.40
    q_cost = float(quant_pred.get("roundtrip_delivery_cost_pct", 0.22)) if quant_pred else 0.22

    # News score in [-1, 1]
    raw_news_score = (
        0.45 * float(news_features.get("news_company_sentiment_5d", 0.0))
        + 0.20 * float(news_features.get("news_promoter_sentiment_5d", 0.0))
        + 0.20 * float(news_features.get("news_industry_sentiment_5d", 0.0))
        + 0.15 * float(news_features.get("news_novelty_weighted_sentiment", 0.0))
        - 0.10 * float(news_features.get("news_regulatory_risk_score", 0.0))
        + 0.10 * float(news_features.get("news_regulatory_positive_score", 0.0))
    )
    news_score = float(np.clip(raw_news_score, -1.0, 1.0))

    # Cross-stock score in [-1, 1]
    cross_raw = float(news_features.get("news_cross_stock_net_impact", 0.0))
    if cross_raw == 0.0 and cross_impacts:
        s_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        m_map = {"low": 0.5, "medium": 1.0, "high": 1.5}
        cross_raw = sum(
            s_map.get(c.expected_impact_direction, 0.0) * m_map.get(c.impact_magnitude, 1.0) * c.confidence
            for c in cross_impacts
        )
    cross_score = float(np.clip(cross_raw / 2.0, -1.0, 1.0))

    quant_score = float(np.clip((q_prob_1d - 0.50) * 6.0, -1.0, 1.0))
    quant_label = _label_from_score(quant_score, 0.05, -0.05)
    news_label = _label_from_score(news_score, 0.08, -0.08)
    cross_label = _label_from_score(cross_score, 0.08, -0.08)

    out: dict[str, MultiHorizonSignalSummary] = {}
    for horizon in ("1D", "5D", "10D"):
        hp = _HORIZON_PARAMS[horizon]
        news_w = hp["news_w"]
        cross_w = hp["cross_w"]
        quant_w = 1.0 - news_w - cross_w

        # Base expected return from quant
        base_ret = q_ret_1d if horizon == "1D" else q_ret_5d * hp["ret_scale"]
        # Bounded news & cross-stock adjustment
        adj_ret = (news_score * 0.45 * news_w + cross_score * 0.35 * cross_w) * (1.0 if horizon == "1D" else 2.2)
        exp_gross = round(float(base_ret + adj_ret), 3)
        cost = q_cost if horizon != "1D" else min(q_cost, hp["cost_pct"])
        exp_net = round(float(exp_gross - cost), 3)

        # Combined probability
        comb_prob = float(np.clip(q_prob_1d + 0.06 * news_w * news_score + 0.05 * cross_w * cross_score, 0.05, 0.95))
        comb_score = quant_w * quant_score + news_w * news_score + cross_w * cross_score
        comb_label = _label_from_score(comb_score, 0.06, -0.06)

        # Agreement-adjusted confidence & uncertainty
        signs = [np.sign(quant_score), np.sign(news_score), np.sign(cross_score)]
        nonzero_signs = [s for s in signs if s != 0]
        agreement = 1.0 if len(set(nonzero_signs)) <= 1 else 0.65
        confidence = round(float(np.clip((0.55 * q_conf + 0.45 * agreement) * 100.0, 15.0, 92.0)), 1)
        uncertainty = round(float(np.clip(100.0 - confidence + (12.0 if regime_label == "High Volatility" else 0.0), 8.0, 88.0)), 1)

        if not trade_eligible or regime_label == "High Volatility":
            decision = "NO TRADE"
        elif comb_prob >= hp["buy_prob"] and exp_net > 0.05 and comb_label == "BULLISH":
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
            expected_gross_return_pct=exp_gross,
            expected_net_return_pct=exp_net,
            quant_signal=quant_label,
            news_signal=news_label,
            cross_stock_signal=cross_label,
            combined_signal=comb_label,
            confidence=confidence,
            uncertainty=uncertainty,
            decision=decision,  # type: ignore[arg-type]
        )

    return out
