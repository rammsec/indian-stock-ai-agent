# BharatAlpha Phase 6 Multi-Agent Research Report: TCS
**As Of:** 2026-11-19 | **Generated:** 2026-11-19T23:11:22.163902+00:00 | **Status:** COMPLETE
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
  "prediction_id": "TCS_2026-11-18_ml_ensemble_3.2.0_20261118T210426Z",
  "symbol": "TCS",
  "observation_date": "2026-11-18",
  "generated_at": "2026-11-18T21:04:26.591210+00:00",
  "model_name": "ml_ensemble",
  "model_version": "3.2.0",
  "market_regime": "Weak Bull",
  "prob_next_day_up": 0.4618,
  "prob_5d_up": 0.4779,
  "prob_10d_up": 0.4694,
  "prob_lgbm_up": 0.4344,
  "prob_xgb_up": 0.42820000648498535,
  "prob_linear_up": 0.4417,
  "expected_next_day_return_pct": -0.1679,
  "expected_5d_return_pct": -0.2212,
  "expected_10d_return_pct": -0.4595,
  "downside_risk_mae_pct": -1.0918,
  "expected_upside_mfe_pct": 1.0797,
  "reward_to_risk_ratio": 0.989,
  "model_agreement_score": 0.9618,
  "confidence_score": 72.69,
  "roundtrip_delivery_cost_pct": 0.3628,
  "roundtrip_intraday_cost_pct": 0.2116,
  "net_expected_1d_return_pct": -0.3795,
  "net_expected_5d_return_pct": -0.584,
  "recommended_action": "NO TRADE",
  "abstention_reason": "Expected move (1D -0.17%, 5D -0.34%) does not clear Indian round-trip transaction costs (0.21% intraday / 0.36% delivery).",
  "top_bullish_drivers": [
    {
      "feature": "ms_india_vix_zscore_20d",
      "family": "3. Market & Sector Strength",
      "value": -0.6452,
      "contribution": 0.00688,
      "direction": "BULLISH"
    },
    {
      "feature": "inst_fii_chg_4q",
      "family": "4. Institutional & Promoter Activity",
      "value": -1.32,
      "contribution": 0.00304,
      "direction": "BULLISH"
    },
    {
      "feature": "pm_ret_180d_pct",
      "family": "1. Price & Technical Momentum",
      "value": -11.0899,
      "contribution": 0.00286,
      "direction": "BULLISH"
    },
    {
      "feature": "pm_macd_hist_chg_1d",
      "family": "1. Price & Technical Momentum",
      "value": -0.0863,
      "contribution": 0.00273,
      "direction": "BULLISH"
    },
    {
      "feature": "eg_qtr_np_yoy_pct",
      "family": "7. Earnings & Management Guidance",
      "value": 14.8627,
      "contribution": 0.0025,
      "direction": "BULLISH"
    }
  ],
  "top_bearish_drivers": [
    {
      "feature": "vd_rel_vol_100d",
      "family": "2. Volume & Delivery",
      "value": 0.3338,
      "contribution": -0.05233,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_corr_nifty_20d",
      "family": "3. Market & Sector Strength",
      "value": 0.3584,
      "contribution": -0.03937,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_nifty_ret_1d_pct",
      "family": "3. Market & Sector Strength",
      "value": -0.6281,
      "contribution": -0.03625,
      "direction": "BEARISH"
    },
    {
      "feature": "fq_asset_turnover",
      "family": "6. Financial Quality",
      "value": 1.4739,
      "contribution": -0.01256,
      "direction": "BEARISH"
    },
    {
      "feature": "ms_corr_nifty_60d",
      "family": "3. Market & Sector Strength",
      "value": 0.3256,
      "contribution": -0.00835,
      "direction": "BEARISH"
    }
  ],
  "target_session_date": "2026-11-19",
  "last_close": 2379.5,
  "atr_14_pct": 2.674,
  "momentum_prob_up": 0.4603,
  "momentum_5d_return_pct": -1.1772,
  "momentum_prob_5d_up": 0.447,
  "momentum_prob_10d_up": 0.4456,
  "momentum_1d_return_pct": -0.2321,
  "momentum_10d_return_pct": -1.5323,
  "feature_snapshot": {
    "ne_is_earnings_window": 1.0,
    "ms_corr_nifty_20d": 0.35837,
    "fq_asset_turnover": 1.47389,
    "inst_dii_chg_4q": 0.93,
    "fq_debt_to_equity": 0.10521,
    "eg_qtr_np_qoq_pct": 3.83,
    "inst_named_holders_count": 12.0,
    "pm_ret_180d_pct": -11.08994,
    "ms_india_vix_zscore_20d": -0.64524,
    "fq_roce_pct": 61.0835
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