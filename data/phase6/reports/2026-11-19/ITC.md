# BharatAlpha Phase 6 Multi-Agent Research Report: ITC
**As Of:** 2026-11-19 | **Generated:** 2026-11-19T23:11:22.400362+00:00 | **Status:** COMPLETE
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
  "prediction_id": "ITC_2026-11-18_ml_ensemble_3.2.0_20261118T210426Z",
  "symbol": "ITC",
  "observation_date": "2026-11-18",
  "generated_at": "2026-11-18T21:04:26.591210+00:00",
  "model_name": "ml_ensemble",
  "model_version": "3.2.0",
  "market_regime": "Weak Bull",
  "prob_next_day_up": 0.4996,
  "prob_5d_up": 0.4502,
  "prob_10d_up": 0.4242,
  "prob_lgbm_up": 0.4583,
  "prob_xgb_up": 0.46160000562667847,
  "prob_linear_up": 0.4732,
  "expected_next_day_return_pct": 0.0183,
  "expected_5d_return_pct": -0.331,
  "expected_10d_return_pct": -0.5453,
  "downside_risk_mae_pct": -0.9898,
  "expected_upside_mfe_pct": 1.0442,
  "reward_to_risk_ratio": 1.055,
  "model_agreement_score": 0.7217,
  "confidence_score": 61.94,
  "roundtrip_delivery_cost_pct": 0.3574,
  "roundtrip_intraday_cost_pct": 0.2062,
  "net_expected_1d_return_pct": -0.1879,
  "net_expected_5d_return_pct": -0.6884,
  "recommended_action": "HOLD / WATCH",
  "abstention_reason": null,
  "top_bullish_drivers": [
    {
      "feature": "pm_atr_ratio_7_21",
      "family": "1. Price & Technical Momentum",
      "value": 1.105,
      "contribution": 0.02163,
      "direction": "BULLISH"
    },
    {
      "feature": "vd_rel_vol_100d",
      "family": "2. Volume & Delivery",
      "value": 1.0073,
      "contribution": 0.00892,
      "direction": "BULLISH"
    },
    {
      "feature": "pm_ret_20d_pct",
      "family": "1. Price & Technical Momentum",
      "value": 13.727,
      "contribution": 0.00886,
      "direction": "BULLISH"
    },
    {
      "feature": "ms_india_vix_zscore_20d",
      "family": "3. Market & Sector Strength",
      "value": -0.6452,
      "contribution": 0.00644,
      "direction": "BULLISH"
    },
    {
      "feature": "pm_ret_accel_20d_60d",
      "family": "1. Price & Technical Momentum",
      "value": 0.5333,
      "contribution": 0.00336,
      "direction": "BULLISH"
    }
  ],
  "top_bearish_drivers": [
    {
      "feature": "ms_corr_nifty_20d",
      "family": "3. Market & Sector Strength",
      "value": 0.3588,
      "contribution": -0.04376,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_nifty_ret_1d_pct",
      "family": "3. Market & Sector Strength",
      "value": -0.6281,
      "contribution": -0.03382,
      "direction": "BEARISH"
    },
    {
      "feature": "fq_asset_turnover",
      "family": "6. Financial Quality",
      "value": 0.8423,
      "contribution": -0.01206,
      "direction": "BEARISH"
    },
    {
      "feature": "inst_fii_chg_4q",
      "family": "4. Institutional & Promoter Activity",
      "value": -4.17,
      "contribution": -0.00725,
      "direction": "BEARISH"
    },
    {
      "feature": "eg_qtr_np_yoy_pct",
      "family": "7. Earnings & Management Guidance",
      "value": -18.4693,
      "contribution": -0.00603,
      "direction": "BEARISH"
    }
  ],
  "target_session_date": "2026-11-19",
  "last_close": 296.6,
  "atr_14_pct": 2.296,
  "momentum_prob_up": 0.5204,
  "momentum_5d_return_pct": 3.5084,
  "momentum_prob_5d_up": 0.5449,
  "momentum_prob_10d_up": 0.5612,
  "momentum_1d_return_pct": 0.2581,
  "momentum_10d_return_pct": 5.2115,
  "feature_snapshot": {
    "ne_is_earnings_window": 1.0,
    "ms_corr_nifty_20d": 0.35883,
    "fq_asset_turnover": 0.84227,
    "inst_dii_chg_4q": 2.4,
    "fq_debt_to_equity": 0.03309,
    "eg_qtr_np_qoq_pct": -6.21,
    "inst_named_holders_count": 14.0,
    "pm_ret_180d_pct": -8.85065,
    "ms_india_vix_zscore_20d": -0.64524,
    "fq_roce_pct": 36.45369
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