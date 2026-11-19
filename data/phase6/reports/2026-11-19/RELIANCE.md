# BharatAlpha Phase 6 Multi-Agent Research Report: RELIANCE
**As Of:** 2026-11-19 | **Generated:** 2026-11-19T23:11:21.501477+00:00 | **Status:** COMPLETE
**Overall Conclusion:** `REJECTED_BY_RISK` | **Confidence:** `0.75`

## 1. Investment Thesis
Balanced accumulation thesis

## 2. Debate Summary
### Bull Case
Operating leverage

### Bear Case
Multiple contraction

## 3. Specialist Analyst Assessments
### Fundamental
Healthy
### Technical
Constructive
### News Intelligence
Neutral
### Promoter & Governance
Strong
### Competitor & Industry
Dominant

## 4. Quantitative Model Predictions (Read-Only)
```json
{
  "prediction_id": "RELIANCE_2026-11-18_ml_ensemble_3.2.0_20261118T210426Z",
  "symbol": "RELIANCE",
  "observation_date": "2026-11-18",
  "generated_at": "2026-11-18T21:04:26.591210+00:00",
  "model_name": "ml_ensemble",
  "model_version": "3.2.0",
  "market_regime": "Weak Bull",
  "prob_next_day_up": 0.5049,
  "prob_5d_up": 0.4525,
  "prob_10d_up": 0.4394,
  "prob_lgbm_up": 0.4688,
  "prob_xgb_up": 0.4677000045776367,
  "prob_linear_up": 0.4759,
  "expected_next_day_return_pct": 0.0226,
  "expected_5d_return_pct": -0.364,
  "expected_10d_return_pct": -0.5525,
  "downside_risk_mae_pct": -0.9933,
  "expected_upside_mfe_pct": 0.9925,
  "reward_to_risk_ratio": 0.999,
  "model_agreement_score": 0.9544,
  "confidence_score": 67.71,
  "roundtrip_delivery_cost_pct": 0.3538,
  "roundtrip_intraday_cost_pct": 0.2026,
  "net_expected_1d_return_pct": -0.1799,
  "net_expected_5d_return_pct": -0.7178,
  "recommended_action": "HOLD / WATCH",
  "abstention_reason": null,
  "top_bullish_drivers": [
    {
      "feature": "vd_rel_vol_100d",
      "family": "2. Volume & Delivery",
      "value": 1.2978,
      "contribution": 0.00943,
      "direction": "BULLISH"
    },
    {
      "feature": "pm_ret_20d_pct",
      "family": "1. Price & Technical Momentum",
      "value": 13.3943,
      "contribution": 0.00905,
      "direction": "BULLISH"
    },
    {
      "feature": "fq_asset_turnover",
      "family": "6. Financial Quality",
      "value": 0.4848,
      "contribution": 0.00828,
      "direction": "BULLISH"
    },
    {
      "feature": "ms_india_vix_zscore_20d",
      "family": "3. Market & Sector Strength",
      "value": -0.6452,
      "contribution": 0.00666,
      "direction": "BULLISH"
    },
    {
      "feature": "fq_roe_pct",
      "family": "6. Financial Quality",
      "value": 10.5919,
      "contribution": 0.00589,
      "direction": "BULLISH"
    }
  ],
  "top_bearish_drivers": [
    {
      "feature": "ms_nifty_ret_1d_pct",
      "family": "3. Market & Sector Strength",
      "value": -0.6281,
      "contribution": -0.0374,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_corr_nifty_20d",
      "family": "3. Market & Sector Strength",
      "value": 0.4281,
      "contribution": -0.03662,
      "direction": "BEARISH"
    },
    {
      "feature": "pm_atr_ratio_7_21",
      "family": "1. Price & Technical Momentum",
      "value": 0.9887,
      "contribution": -0.00735,
      "direction": "BEARISH"
    },
    {
      "feature": "pm_rsi_14_chg_3d",
      "family": "1. Price & Technical Momentum",
      "value": 2.3099,
      "contribution": -0.00615,
      "direction": "BEARISH"
    },
    {
      "feature": "inst_named_holders_count",
      "family": "4. Institutional & Promoter Activity",
      "value": 14.0,
      "contribution": -0.00425,
      "direction": "BEARISH"
    }
  ],
  "target_session_date": "2026-11-19",
  "last_close": 1363.0,
  "atr_14_pct": 2.089,
  "momentum_prob_up": 0.518,
  "momentum_5d_return_pct": 3.1948,
  "momentum_prob_5d_up": 0.5362,
  "momentum_prob_10d_up": 0.5542,
  "momentum_1d_return_pct": 0.1829,
  "momentum_10d_return_pct": 4.8947,
  "feature_snapshot": {
    "ne_is_earnings_window": 1.0,
    "ms_corr_nifty_20d": 0.42813,
    "fq_asset_turnover": 0.48485,
    "inst_dii_chg_4q": 1.37,
    "fq_debt_to_equity": 0.44574,
    "eg_qtr_np_qoq_pct": 27.39,
    "inst_named_holders_count": 14.0,
    "pm_ret_180d_pct": -4.55182,
    "ms_india_vix_zscore_20d": -0.64524,
    "fq_roce_pct": 13.70054
  },
  "outcome_evaluated_at": null,
  "realized_next_day_return_pct": null,
  "realized_5d_return_pct": null
}
```

## 5. Risk Assessment
**Verdict:** `REJECT`
**Rationale:** Deterministic verdict REJECT. Reasons: 7 UNKNOWN risk dimensions (insufficient risk data): downside_risk, volatility_risk, liquidity_risk, concentration_risk, event_risk, drawdown_risk, risk_reward. Manageable across dimensions
**Rejection Reasons:** 7 UNKNOWN risk dimensions (insufficient risk data): downside_risk, volatility_risk, liquidity_risk, concentration_risk, event_risk, drawdown_risk, risk_reward

## 6. Agent Disagreements
- competitor (BULLISH) vs quantitative (BEARISH) analysts disagree
- fundamental (BULLISH) vs quantitative (BEARISH) analysts disagree
- promoter (BULLISH) vs quantitative (BEARISH) analysts disagree
- quantitative (BEARISH) vs technical (BULLISH) analysts disagree