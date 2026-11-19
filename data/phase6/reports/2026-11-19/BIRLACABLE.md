# BharatAlpha Phase 6 Multi-Agent Research Report: BIRLACABLE
**As Of:** 2026-11-19 | **Generated:** 2026-11-19T23:11:22.514758+00:00 | **Status:** COMPLETE
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
  "prediction_id": "BIRLACABLE_2026-11-18_ml_ensemble_3.2.0_20261118T210426Z",
  "symbol": "BIRLACABLE",
  "observation_date": "2026-11-18",
  "generated_at": "2026-11-18T21:04:26.591210+00:00",
  "model_name": "ml_ensemble",
  "model_version": "3.2.0",
  "market_regime": "Weak Bull",
  "prob_next_day_up": 0.4681,
  "prob_5d_up": 0.4987,
  "prob_10d_up": 0.5219,
  "prob_lgbm_up": 0.4621,
  "prob_xgb_up": 0.4650000035762787,
  "prob_linear_up": 0.4985,
  "expected_next_day_return_pct": -0.4444,
  "expected_5d_return_pct": -0.0572,
  "expected_10d_return_pct": -0.0044,
  "downside_risk_mae_pct": -1.4977,
  "expected_upside_mfe_pct": 1.672,
  "reward_to_risk_ratio": 1.116,
  "model_agreement_score": 0.9511,
  "confidence_score": 71.55,
  "roundtrip_delivery_cost_pct": 0.5155,
  "roundtrip_intraday_cost_pct": 0.3643,
  "net_expected_1d_return_pct": -0.8087,
  "net_expected_5d_return_pct": -0.5727,
  "recommended_action": "HOLD / WATCH",
  "abstention_reason": null,
  "top_bullish_drivers": [
    {
      "feature": "pm_ret_180d_pct",
      "family": "1. Price & Technical Momentum",
      "value": 119.5299,
      "contribution": 0.01701,
      "direction": "BULLISH"
    },
    {
      "feature": "pm_atr_14_pct",
      "family": "1. Price & Technical Momentum",
      "value": 6.2147,
      "contribution": 0.00676,
      "direction": "BULLISH"
    },
    {
      "feature": "vd_rel_vol_100d",
      "family": "2. Volume & Delivery",
      "value": 1.6125,
      "contribution": 0.0067,
      "direction": "BULLISH"
    },
    {
      "feature": "ms_india_vix_zscore_20d",
      "family": "3. Market & Sector Strength",
      "value": -0.6452,
      "contribution": 0.00589,
      "direction": "BULLISH"
    },
    {
      "feature": "fq_roe_pct",
      "family": "6. Financial Quality",
      "value": 6.0498,
      "contribution": 0.00443,
      "direction": "BULLISH"
    }
  ],
  "top_bearish_drivers": [
    {
      "feature": "ms_nifty_ret_1d_pct",
      "family": "3. Market & Sector Strength",
      "value": -0.6281,
      "contribution": -0.03827,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_corr_nifty_20d",
      "family": "3. Market & Sector Strength",
      "value": 0.2199,
      "contribution": -0.02926,
      "direction": "BEARISH"
    },
    {
      "feature": "inst_retail_dilution_4q",
      "family": "4. Institutional & Promoter Activity",
      "value": -0.92,
      "contribution": -0.01382,
      "direction": "BEARISH"
    },
    {
      "feature": "fq_asset_turnover",
      "family": "6. Financial Quality",
      "value": 1.6616,
      "contribution": -0.01347,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_corr_nifty_60d",
      "family": "3. Market & Sector Strength",
      "value": 0.1371,
      "contribution": -0.00928,
      "direction": "BEARISH"
    }
  ],
  "target_session_date": "2026-11-19",
  "last_close": 302.6,
  "atr_14_pct": 6.215,
  "momentum_prob_up": 0.4132,
  "momentum_5d_return_pct": -2.7123,
  "momentum_prob_5d_up": 0.404,
  "momentum_prob_10d_up": 0.4082,
  "momentum_1d_return_pct": -0.8067,
  "momentum_10d_return_pct": -3.5907,
  "feature_snapshot": {
    "ne_is_earnings_window": 1.0,
    "ms_corr_nifty_20d": 0.21994,
    "fq_asset_turnover": 1.66164,
    "inst_dii_chg_4q": 0.05,
    "fq_debt_to_equity": 0.47331,
    "eg_qtr_np_qoq_pct": 22.58,
    "inst_named_holders_count": 9.0,
    "pm_ret_180d_pct": 119.52989,
    "ms_india_vix_zscore_20d": -0.64524,
    "fq_roce_pct": 11.35266
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