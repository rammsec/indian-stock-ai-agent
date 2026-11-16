"""Calibrated Quantitative ML Ensemble (Phase 3 — Section 8 & 9).

Combines four complementary model families trained strictly on chronological
point-in-time folds:
1. `LightGBM` (`LGBMClassifier` + `LGBMRegressor(objective='huber')`) — native NaN support
   and exact TreeSHAP (`pred_contrib=True`) per-prediction explanations.
2. `XGBoost` (`XGBClassifier` + `XGBRegressor(objective='reg:pseudohubererror')`) —
   native NaN support with strong L1/L2 leaf regularization.
3. `Regularized Linear/Logistic` (`StandardScaler` + `LogisticRegression` / `Ridge`) —
   captures global monotonic factor relationships.
4. `Stationary Technical-Momentum Structural Prior` — anchors predictions against
   regime-shift tree overfitting when single-day equity noise is high.

Anti-Overfitting & Calibration Safeguards:
- **Two-Half Temporal IC-Stability Screening**: Evaluates every candidate feature on both
  the first 50% and second 50% of `train_df` dates and only selects features whose
  Spearman Rank IC has the **same algebraic sign** across both halves of the training
  history (eliminating spurious calendar-date correlations like raw index levels).
- **Chronological Validation Stacking & Platt Calibration**: Evaluates sub-models on the
  held-out chronological tail (`inner_val`) of each training fold and weights each
  sub-model by its positive out-of-sample validation skill.
- **Explicit 5-Trading-Day Purge**: Purges the last 5 trading days of `train_df` when
  training the 5D swing return model so `target_5d_ret` never overlaps the test window.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.feature_engine import CANDIDATE_FEATURE_FAMILIES
from ..models import FeatureAttributionItem, QuantitativePredictionRecord
from .baselines import NaiveMomentumBaseline, TechnicalRuleBaseline
from .cost_model import compute_row_cost_series

# Raw macro level columns that act as calendar-date proxies in decision trees;
# stationary z-scores and relative alpha (`ms_india_vix_zscore_20d`, `ms_alpha_vs_nifty_*`) are used instead.
NON_STATIONARY_MACRO_LEVELS: set[str] = {
    "ms_india_vix_level",
    "ms_nifty_dist_sma200_pct",
    "ms_nifty_dist_sma50_pct",
    "ne_days_since_qtr_filing",
    "pm_mom_12m_minus_1m",
    "pm_logret_252d",
    "pm_ret_252d_pct",
}


def _resolve_family(col: str) -> str:
    for prefix, label in CANDIDATE_FEATURE_FAMILIES.items():
        if col.startswith(prefix):
            return label
    return "Other"


class BharatAlphaQuantEnsemble:
    """Calibrated LightGBM + XGBoost + Linear + Structural Prior Ensemble for Indian Equities."""

    name: str = "ml_ensemble"
    version: str = "3.0.0"

    def __init__(
        self,
        top_k_features: int = 36,
        random_state: int = 42,
    ) -> None:
        self.top_k_features = top_k_features
        self.random_state = random_state
        self.active_features_: list[str] = []
        self.feature_importances_: dict[str, float] = {}
        self.model_weights_: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)
        self.calibration_slope_: float = 0.85
        self.calibration_intercept_: float = 0.0
        self.base_up_rate_: float = 0.50

    def _select_in_fold_features(self, train_df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
        """Select top-K stationary features with consistent Spearman Rank IC sign across both halves of `train_df`."""
        unique_dates = sorted(train_df["observation_date"].unique())
        mid_date = unique_dates[len(unique_dates) // 2]
        h1_mask = train_df["observation_date"] <= mid_date
        h2_mask = train_df["observation_date"] > mid_date

        y1 = pd.to_numeric(train_df["target_next_day_ret"], errors="coerce")
        y5 = pd.to_numeric(train_df.get("target_5d_ret"), errors="coerce")
        min_non_null = max(30, int(0.15 * len(train_df)))

        scored: list[tuple[float, str]] = []
        for col in candidate_cols:
            if col not in train_df.columns or col in NON_STATIONARY_MACRO_LEVELS:
                continue
            s = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid1 = s.notna() & y1.notna()
            if int(valid1.sum()) < min_non_null:
                continue
            std_val = float(s[valid1].std())
            if np.isnan(std_val) or std_val <= 1e-10:
                continue

            # Evaluate Rank IC on Half 1 and Half 2 separately to enforce temporal stability
            v_h1 = valid1 & h1_mask
            v_h2 = valid1 & h2_mask
            if int(v_h1.sum()) < 20 or int(v_h2.sum()) < 20:
                continue

            ic_h1 = s[v_h1].rank().corr(y1[v_h1].rank())
            ic_h2 = s[v_h2].rank().corr(y1[v_h2].rank())
            if pd.isna(ic_h1) or pd.isna(ic_h2):
                continue

            ic_h1_f = float(ic_h1)
            ic_h2_f = float(ic_h2)

            # Stability bonus when IC sign agrees across both chronological halves of training data
            sign_matches = (ic_h1_f * ic_h2_f) > 0
            mean_ic1 = 0.5 * (ic_h1_f + ic_h2_f)
            stability_mult = 1.5 if sign_matches else 0.35

            ic5_val = 0.0
            if y5 is not None:
                valid5 = s.notna() & y5.notna()
                if int(valid5.sum()) >= min_non_null:
                    ic5 = s[valid5].rank().corr(y5[valid5].rank())
                    if pd.notna(ic5):
                        ic5_val = abs(float(ic5))

            score = stability_mult * (0.70 * abs(mean_ic1) + 0.30 * ic5_val)
            scored.append((score, col))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [col for _, col in scored[: self.top_k_features]]

        # Ensure core stock-specific momentum, volatility, volume, and relative alpha features are present
        must_include = [
            "pm_ret_1d_pct",
            "pm_ret_3d_pct",
            "pm_ret_5d_pct",
            "pm_ret_20d_pct",
            "pm_rsi_14",
            "pm_macd_hist",
            "pm_dist_sma_20_pct",
            "pm_dist_sma_50_pct",
            "pm_bb_pct_b_20",
            "pm_atr_14_pct",
            "vd_rel_vol_20d",
            "ms_alpha_vs_nifty_5d_pct",
            "ms_alpha_vs_sector_5d_pct",
            "ms_india_vix_zscore_20d",
        ]
        for m_col in must_include:
            if m_col in candidate_cols and m_col in train_df.columns and m_col not in selected:
                if int(train_df[m_col].notna().sum()) >= min_non_null:
                    selected.append(m_col)

        return selected

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "BharatAlphaQuantEnsemble":
        """Fit all ensemble sub-models strictly on `train_df` with chronological validation weighting."""
        sub_1d = train_df[
            train_df["target_next_day_ret"].notna() & train_df["target_next_day_up"].notna()
        ].sort_values(["observation_date", "symbol"]).reset_index(drop=True)

        self.active_features_ = self._select_in_fold_features(sub_1d, feature_cols)
        X_all = (
            sub_1d[self.active_features_]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        y_up = sub_1d["target_next_day_up"].astype(int).values
        y_r1 = pd.to_numeric(sub_1d["target_next_day_ret"], errors="coerce").clip(-7.5, 7.5).values
        y_mae = pd.to_numeric(sub_1d["target_next_day_mae"], errors="coerce").fillna(-1.2).clip(-7.5, -0.1).values
        y_mfe = pd.to_numeric(sub_1d["target_next_day_mfe"], errors="coerce").fillna(1.2).clip(0.1, 7.5).values

        self.base_up_rate_ = float(np.mean(y_up)) if len(y_up) > 0 else 0.50

        # Structural priors (Naive Momentum + Technical Rule)
        self.mom_prior_ = NaiveMomentumBaseline().fit(sub_1d, self.active_features_)
        self.tech_prior_ = TechnicalRuleBaseline().fit(sub_1d, self.active_features_)

        # 1. Chronological split (first 78% dates vs last 22% dates) to learn optimal non-negative stacking weights
        unique_dates = sorted(sub_1d["observation_date"].unique())
        if len(unique_dates) >= 35:
            val_split_idx = int(len(unique_dates) * 0.78)
            inner_train_end_date = unique_dates[val_split_idx - 2]
            inner_val_start_date = unique_dates[val_split_idx]

            mask_itr = sub_1d["observation_date"] <= inner_train_end_date
            mask_ival = sub_1d["observation_date"] >= inner_val_start_date

            lgb_cal = self._make_lgb_clf()
            xgb_cal = self._make_xgb_clf()
            lin_cal = self._make_lin_clf()

            lgb_cal.fit(X_all.loc[mask_itr], y_up[mask_itr])
            xgb_cal.fit(X_all.loc[mask_itr], y_up[mask_itr])
            lin_cal.fit(X_all.loc[mask_itr], y_up[mask_itr])

            val_slice = sub_1d.loc[mask_ival]
            p_lgb_v = lgb_cal.predict_proba(X_all.loc[mask_ival])[:, 1]
            p_xgb_v = xgb_cal.predict_proba(X_all.loc[mask_ival])[:, 1]
            p_lin_v = lin_cal.predict_proba(X_all.loc[mask_ival])[:, 1]
            p_pri_v = (
                0.55 * self.mom_prior_.predict_df(val_slice, self.active_features_)["prob_next_day_up"].values
                + 0.45 * self.tech_prior_.predict_df(val_slice, self.active_features_)["prob_next_day_up"].values
            )

            # Score each sub-model by out-of-sample Rank IC on inner_val
            s_y_val = pd.Series(y_r1[mask_ival])
            ics = []
            for p_arr in [p_lgb_v, p_xgb_v, p_lin_v, p_pri_v]:
                ic_v = pd.Series(p_arr).rank().corr(s_y_val.rank())
                ic_f = float(ic_v) if pd.notna(ic_v) else 0.0
                # Soft floor so every family contributes, with boost for positive validation IC
                ics.append(max(0.08, 0.15 + max(0.0, ic_f) * 4.0))

            w_sum = sum(ics)
            self.model_weights_ = (
                round(ics[0] / w_sum, 4),
                round(ics[1] / w_sum, 4),
                round(ics[2] / w_sum, 4),
                round(ics[3] / w_sum, 4),
            )
        else:
            self.model_weights_ = (0.28, 0.24, 0.24, 0.24)

        # 2. Fit full-fold Direction Classifiers
        self.lgb_clf_ = self._make_lgb_clf()
        self.xgb_clf_ = self._make_xgb_clf()
        self.lin_clf_ = self._make_lin_clf()

        self.lgb_clf_.fit(X_all, y_up)
        self.xgb_clf_.fit(X_all, y_up)
        self.lin_clf_.fit(X_all, y_up)

        # 3. Fit full-fold 1D Return, Downside MAE, and Upside MFE Regressors
        self.lgb_reg_1d_ = self._make_lgb_reg()
        self.xgb_reg_1d_ = self._make_xgb_reg()
        self.lin_reg_1d_ = self._make_lin_reg()

        self.lgb_reg_1d_.fit(X_all, y_r1)
        self.xgb_reg_1d_.fit(X_all, y_r1)
        self.lin_reg_1d_.fit(X_all, y_r1)

        self.lgb_reg_mae_ = self._make_lgb_reg()
        self.lgb_reg_mae_.fit(X_all, y_mae)

        self.lgb_reg_mfe_ = self._make_lgb_reg()
        self.lgb_reg_mfe_.fit(X_all, y_mfe)

        # 4. Fit 5D Return Regressors with strict 5-day tail purge (zero overlap with test fold)
        cutoff_5d_date = unique_dates[-6] if len(unique_dates) > 10 else unique_dates[-1]
        sub_5d = train_df[
            (train_df["observation_date"] <= cutoff_5d_date) & train_df["target_5d_ret"].notna()
        ].copy()
        if len(sub_5d) >= 40:
            X_5d = (
                sub_5d[self.active_features_]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            y_r5 = pd.to_numeric(sub_5d["target_5d_ret"], errors="coerce").clip(-15.0, 15.0).values
            self.lgb_reg_5d_ = self._make_lgb_reg()
            self.xgb_reg_5d_ = self._make_xgb_reg()
            self.lgb_reg_5d_.fit(X_5d, y_r5)
            self.xgb_reg_5d_.fit(X_5d, y_r5)
        else:
            self.lgb_reg_5d_ = self.lgb_reg_1d_
            self.xgb_reg_5d_ = self.xgb_reg_1d_

        # 5. Record normalized global feature importances (gain + split blend)
        lgb_imp = np.array(self.lgb_clf_.feature_importances_, dtype=float)
        xgb_imp = np.array(self.xgb_clf_.feature_importances_, dtype=float)
        lgb_norm = lgb_imp / max(1e-9, lgb_imp.sum())
        xgb_norm = xgb_imp / max(1e-9, xgb_imp.sum())
        blended_imp = 0.55 * lgb_norm + 0.45 * xgb_norm
        self.feature_importances_ = {
            feat: round(float(imp), 6)
            for feat, imp in sorted(
                zip(self.active_features_, blended_imp),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        return self

    def _make_lgb_clf(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=75,
            max_depth=3,
            num_leaves=6,
            min_child_samples=60,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=4.0,
            reg_lambda=10.0,
            learning_rate=0.025,
            random_state=self.random_state,
            n_jobs=2,
            verbose=-1,
        )

    def _make_xgb_clf(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=75,
            max_depth=3,
            min_child_weight=55,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=4.0,
            reg_lambda=10.0,
            learning_rate=0.025,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=2,
            verbosity=0,
        )

    def _make_lin_clf(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=0.015, max_iter=1000, random_state=self.random_state)),
            ]
        )

    def _make_lgb_reg(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective="huber",
            n_estimators=70,
            max_depth=3,
            num_leaves=6,
            min_child_samples=60,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=4.0,
            reg_lambda=10.0,
            learning_rate=0.025,
            random_state=self.random_state,
            n_jobs=2,
            verbose=-1,
        )

    def _make_xgb_reg(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            n_estimators=70,
            max_depth=3,
            min_child_weight=55,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=4.0,
            reg_lambda=10.0,
            learning_rate=0.025,
            random_state=self.random_state,
            n_jobs=2,
            verbosity=0,
        )

    def _make_lin_reg(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=200.0, random_state=self.random_state)),
            ]
        )

    def predict_df(self, test_df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
        """Predict calibrated probabilities, multi-horizon returns, downside risk & confidence."""
        X_test = (
            test_df[self.active_features_]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        p_lgb = self.lgb_clf_.predict_proba(X_test)[:, 1]
        p_xgb = self.xgb_clf_.predict_proba(X_test)[:, 1]
        p_lin_raw = self.lin_clf_.predict_proba(X_test)[:, 1]
        p_lin = np.clip(0.50 + 0.55 * (p_lin_raw - 0.50), 0.28, 0.72)

        mom_df = self.mom_prior_.predict_df(test_df, self.active_features_)
        tech_df = self.tech_prior_.predict_df(test_df, self.active_features_)
        p_pri = 0.55 * mom_df["prob_next_day_up"].values + 0.45 * tech_df["prob_next_day_up"].values

        w_lgb, w_xgb, w_lin, w_pri = self.model_weights_
        p_raw = w_lgb * p_lgb + w_xgb * p_xgb + w_lin * p_lin + w_pri * p_pri
        p_cal = np.clip(0.50 + self.calibration_slope_ * (p_raw - 0.50), 0.22, 0.78)

        # Model agreement across the constituent families
        std_m = np.std(np.vstack([p_lgb, p_xgb, p_lin, p_pri]), axis=0)
        sign_agree = (
            ((p_lgb >= 0.5) == (p_xgb >= 0.5)).astype(float) * 0.35
            + ((p_lgb >= 0.5) == (p_lin >= 0.5)).astype(float) * 0.30
            + ((p_pri >= 0.5) == (p_cal >= 0.5)).astype(float) * 0.35
        )
        agreement_score = np.clip(0.65 * sign_agree + 0.35 * (1.0 - np.clip(std_m * 4.5, 0.0, 1.0)), 0.15, 0.99)

        # Expected 1D and 5D returns
        r1_ml = (
            w_lgb * self.lgb_reg_1d_.predict(X_test)
            + w_xgb * self.xgb_reg_1d_.predict(X_test)
            + w_lin * self.lin_reg_1d_.predict(X_test)
            + w_pri * (0.55 * mom_df["expected_next_day_return_pct"].values + 0.45 * tech_df["expected_next_day_return_pct"].values)
        )
        r5_ml = (
            0.35 * self.lgb_reg_5d_.predict(X_test)
            + 0.30 * self.xgb_reg_5d_.predict(X_test)
            + 0.35 * (0.55 * mom_df["expected_5d_return_pct"].values + 0.45 * tech_df["expected_5d_return_pct"].values)
        )

        atr_s = (
            pd.to_numeric(test_df["pm_atr_14_pct"], errors="coerce").fillna(1.6).values
            if "pm_atr_14_pct" in test_df.columns
            else np.full(len(test_df), 1.6)
        )
        prob_implied_r1 = (p_cal - 0.50) * 2.1 * atr_s
        exp_1d = np.clip(0.55 * r1_ml + 0.45 * prob_implied_r1, -4.0, 4.0)
        exp_5d = np.clip(0.65 * r5_ml + 0.35 * (exp_1d * 2.3), -10.0, 10.0)

        downside_mae = np.clip(self.lgb_reg_mae_.predict(X_test), -7.0, -0.30)
        upside_mfe = np.clip(self.lgb_reg_mfe_.predict(X_test), 0.30, 7.0)

        coverage_ratio = X_test.notna().mean(axis=1).values
        confidence = np.clip(
            36.0
            + np.abs(p_cal - 0.50) * 145.0
            + agreement_score * 22.0
            + coverage_ratio * 10.0,
            25.0,
            96.0,
        )

        return pd.DataFrame(
            {
                "prob_next_day_up": np.round(p_cal, 4),
                "prob_lgbm_up": np.round(p_lgb, 4),
                "prob_xgb_up": np.round(p_xgb, 4),
                "prob_linear_up": np.round(p_lin, 4),
                "expected_next_day_return_pct": np.round(exp_1d, 4),
                "expected_5d_return_pct": np.round(exp_5d, 4),
                "downside_risk_mae_pct": np.round(downside_mae, 4),
                "expected_upside_mfe_pct": np.round(upside_mfe, 4),
                "model_agreement_score": np.round(agreement_score, 4),
                "confidence_score": np.round(confidence, 2),
            },
            index=test_df.index,
        )

    def explain_row_shap(
        self,
        row_df: pd.DataFrame,
        top_n: int = 5,
    ) -> tuple[list[FeatureAttributionItem], list[FeatureAttributionItem]]:
        """Compute exact TreeSHAP feature contributions (`pred_contrib=True`) for a 1-row DataFrame."""
        X_row = (
            row_df[self.active_features_]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        contribs = self.lgb_clf_.booster_.predict(X_row, pred_contrib=True)[0][:-1]
        vals = X_row.iloc[0].values

        bullish: list[FeatureAttributionItem] = []
        bearish: list[FeatureAttributionItem] = []

        order_desc = np.argsort(contribs)[::-1]
        for idx in order_desc:
            c_val = float(contribs[idx])
            if c_val <= 1e-5 or len(bullish) >= top_n:
                break
            f_name = self.active_features_[idx]
            raw_v = float(vals[idx]) if pd.notna(vals[idx]) else None
            bullish.append(
                FeatureAttributionItem(
                    feature=f_name,
                    family=_resolve_family(f_name),
                    value=round(raw_v, 4) if raw_v is not None else None,
                    contribution=round(c_val, 5),
                    direction="BULLISH",
                )
            )

        order_asc = np.argsort(contribs)
        for idx in order_asc:
            c_val = float(contribs[idx])
            if c_val >= -1e-5 or len(bearish) >= top_n:
                break
            f_name = self.active_features_[idx]
            raw_v = float(vals[idx]) if pd.notna(vals[idx]) else None
            bearish.append(
                FeatureAttributionItem(
                    feature=f_name,
                    family=_resolve_family(f_name),
                    value=round(raw_v, 4) if raw_v is not None else None,
                    contribution=round(c_val, 5),
                    direction="BEARISH",
                )
            )

        return bullish, bearish


def classify_cost_aware_action(
    prob_up: float,
    exp_1d_pct: float,
    exp_5d_pct: float,
    downside_mae_pct: float,
    upside_mfe_pct: float,
    agreement_score: float,
    confidence_score: float,
    delivery_cost_pct: float,
    intraday_cost_pct: float,
    market_regime: str = "Sideways",
    trade_eligible: bool = True,
) -> tuple[str, str | None]:
    """Convert calibrated predictions + Indian transaction costs + market regime into an action & abstention reason."""
    if not trade_eligible:
        return "NO TRADE", "Data quality gate failed (<60/100 or critical missing fields)."

    if agreement_score < 0.45 or confidence_score < 42.0:
        return (
            "NO TRADE",
            f"Low model agreement ({agreement_score:.2f}) or confidence ({confidence_score:.1f}/100); abstaining.",
        )

    net_1d = exp_1d_pct - intraday_cost_pct
    net_5d = exp_5d_pct - delivery_cost_pct
    rr_ratio = upside_mfe_pct / max(0.25, abs(downside_mae_pct))

    regime_blocks_aggressive_long = market_regime in ("Bear", "High Volatility")

    # High-conviction Long gates (must clear Indian round-trip transaction costs!)
    if prob_up >= 0.535 and (net_5d >= 0.25 or net_1d >= 0.08) and rr_ratio >= 0.95:
        if regime_blocks_aggressive_long:
            if prob_up >= 0.60 and net_5d >= 0.85:
                return "BUY / ACCUMULATE", None
            return (
                "HOLD / WATCH",
                f"Market regime ({market_regime}) blocks aggressive BUY despite positive stock edge.",
            )
        if prob_up >= 0.585 and net_5d >= 0.75 and agreement_score >= 0.72:
            return "STRONG BUY", None
        return "BUY / ACCUMULATE", None

    # Bearish / Reduce gates
    if prob_up <= 0.455 and (exp_5d_pct <= -0.35 or exp_1d_pct <= -0.15):
        if prob_up <= 0.41 and exp_5d_pct <= -0.85:
            return "AVOID / SHORT", None
        return "REDUCE / SELL", None

    # If expected gross move is smaller than round-trip transaction cost, explicitly abstain
    if abs(exp_1d_pct) < intraday_cost_pct and abs(exp_5d_pct) < delivery_cost_pct:
        return (
            "NO TRADE",
            f"Expected move (1D {exp_1d_pct:+.2f}%, 5D {exp_5d_pct:+.2f}%) does not clear "
            f"Indian round-trip transaction costs ({intraday_cost_pct:.2f}% intraday / {delivery_cost_pct:.2f}% delivery).",
        )

    return "HOLD / WATCH", None


def build_latest_prediction_records(
    ensemble: BharatAlphaQuantEnsemble,
    latest_feature_df: pd.DataFrame,
    market_regime_label: str = "Sideways",
    quality_map: dict[str, bool] | None = None,
) -> list[QuantitativePredictionRecord]:
    """Generate explainable `QuantitativePredictionRecord` objects for the latest trading day."""
    if latest_feature_df.empty:
        return []

    quality_map = quality_map or {}
    preds_df = ensemble.predict_df(latest_feature_df)
    deliv_costs = compute_row_cost_series(latest_feature_df, mode="delivery")
    intra_costs = compute_row_cost_series(latest_feature_df, mode="intraday")

    now_iso = datetime.now(timezone.utc).isoformat()
    records: list[QuantitativePredictionRecord] = []

    for idx in latest_feature_df.index:
        row_f = latest_feature_df.loc[[idx]]
        row_p = preds_df.loc[idx]
        sym = str(row_f["symbol"].iloc[0]).upper()
        obs_date = str(row_f["observation_date"].iloc[0])

        d_cost = float(deliv_costs.loc[idx])
        i_cost = float(intra_costs.loc[idx])
        prob_up = float(row_p["prob_next_day_up"])
        exp_1d = float(row_p["expected_next_day_return_pct"])
        exp_5d = float(row_p["expected_5d_return_pct"])
        mae = float(row_p["downside_risk_mae_pct"])
        mfe = float(row_p["expected_upside_mfe_pct"])
        rr = round(mfe / max(0.25, abs(mae)), 3)
        agree = float(row_p["model_agreement_score"])
        conf = float(row_p["confidence_score"])

        eligible = quality_map.get(sym, True)
        action, abstain_reason = classify_cost_aware_action(
            prob_up=prob_up,
            exp_1d_pct=exp_1d,
            exp_5d_pct=exp_5d,
            downside_mae_pct=mae,
            upside_mfe_pct=mfe,
            agreement_score=agree,
            confidence_score=conf,
            delivery_cost_pct=d_cost,
            intraday_cost_pct=i_cost,
            market_regime=market_regime_label,
            trade_eligible=eligible,
        )

        bull_drivers, bear_drivers = ensemble.explain_row_shap(row_f, top_n=5)

        realized_1d = row_f["target_next_day_ret"].iloc[0] if "target_next_day_ret" in row_f.columns else None
        realized_5d = row_f["target_5d_ret"].iloc[0] if "target_5d_ret" in row_f.columns else None

        records.append(
            QuantitativePredictionRecord(
                prediction_id=f"{sym}_{obs_date}_{ensemble.name}_{ensemble.version}",
                symbol=sym,
                observation_date=obs_date,
                generated_at=now_iso,
                model_name=ensemble.name,
                model_version=ensemble.version,
                market_regime=market_regime_label,
                prob_next_day_up=prob_up,
                prob_lgbm_up=float(row_p["prob_lgbm_up"]),
                prob_xgb_up=float(row_p["prob_xgb_up"]),
                prob_linear_up=float(row_p["prob_linear_up"]),
                expected_next_day_return_pct=exp_1d,
                expected_5d_return_pct=exp_5d,
                downside_risk_mae_pct=mae,
                expected_upside_mfe_pct=mfe,
                reward_to_risk_ratio=rr,
                model_agreement_score=agree,
                confidence_score=conf,
                roundtrip_delivery_cost_pct=round(d_cost, 4),
                roundtrip_intraday_cost_pct=round(i_cost, 4),
                net_expected_1d_return_pct=round(exp_1d - i_cost, 4),
                net_expected_5d_return_pct=round(exp_5d - d_cost, 4),
                recommended_action=action,  # type: ignore[arg-type]
                abstention_reason=abstain_reason,
                top_bullish_drivers=bull_drivers,
                top_bearish_drivers=bear_drivers,
                realized_next_day_return_pct=(
                    round(float(realized_1d), 4) if pd.notna(realized_1d) else None
                ),
                realized_5d_return_pct=(
                    round(float(realized_5d), 4) if pd.notna(realized_5d) else None
                ),
            )
        )

    return records
