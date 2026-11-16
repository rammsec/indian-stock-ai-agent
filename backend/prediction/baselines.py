"""Baseline Benchmark Models (Phase 3 — Section 8).

Implements three mandatory benchmark models against which the primary ML ensemble
is evaluated in walk-forward out-of-sample testing:
1. `NaiveMomentumBaseline` (`naive_momentum`): Predicts continuation from 5D/10D
   momentum adjusted for 1D short-term overextension.
2. `TechnicalRuleBaseline` (`technical_rule`): Classic multi-indicator technical
   rule system (RSI-14, MACD histogram, Price vs SMA-20/SMA-50, Bollinger %B, and
   20D relative volume).
3. `LinearRidgeLogisticBaseline` (`linear_ridge_logistic`): Regularized L2
   Logistic Regression (`C=0.1`) for `P(next_day_up)` and Ridge Regression
   (`alpha=50.0`) for `1D` return, `5D` return, and `1D MAE` downside risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _col_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


class NaiveMomentumBaseline:
    """Benchmark 1: Naive 5D/10D Momentum & 1D Short-Term Mean-Reversion Baseline."""

    name: str = "naive_momentum"

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "NaiveMomentumBaseline":
        y1 = pd.to_numeric(train_df.get("target_next_day_ret"), errors="coerce").dropna()
        mae = pd.to_numeric(train_df.get("target_next_day_mae"), errors="coerce").dropna()
        self.base_up_rate_ = float((y1 > 0).mean()) if len(y1) > 0 else 0.51
        self.median_mae_ = float(mae.median()) if len(mae) > 0 else -1.10
        return self

    def predict_df(self, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        r1 = _col_series(test_df, "pm_ret_1d_pct", 0.0)
        r5 = _col_series(test_df, "pm_ret_5d_pct", 0.0)
        r20 = _col_series(test_df, "pm_ret_20d_pct", 0.0)
        atr = _col_series(test_df, "pm_atr_14_pct", 1.6)

        # Momentum signal blended with 1D pullback
        raw_score = 0.035 * r5 + 0.015 * r20 - 0.04 * r1
        prob_up = (self.base_up_rate_ + np.tanh(raw_score) * 0.14).clip(0.32, 0.68)
        exp_1d = (0.08 * r5 - 0.06 * r1).clip(-2.5, 2.5)
        exp_5d = (0.22 * r20 + 0.12 * r5).clip(-6.0, 6.0)
        downside_mae = (-0.72 * atr).clip(-6.0, -0.35)
        upside_mfe = (0.75 * atr).clip(0.35, 6.0)

        return pd.DataFrame(
            {
                "prob_next_day_up": prob_up.values,
                "expected_next_day_return_pct": exp_1d.values,
                "expected_5d_return_pct": exp_5d.values,
                "downside_risk_mae_pct": downside_mae.values,
                "expected_upside_mfe_pct": upside_mfe.values,
                "model_agreement_score": np.full(len(test_df), 0.60),
                "confidence_score": (50.0 + (prob_up - 0.5).abs() * 140.0).clip(35.0, 82.0).values,
            },
            index=test_df.index,
        )


class TechnicalRuleBaseline:
    """Benchmark 2: Classic RSI(14) + MACD + SMA(20/50) + Relative Volume Rule System."""

    name: str = "technical_rule"

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "TechnicalRuleBaseline":
        return self

    def predict_df(self, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        rsi14 = _col_series(test_df, "pm_rsi_14", 50.0)
        macd_h = _col_series(test_df, "pm_macd_hist", 0.0)
        dist_sma20 = _col_series(test_df, "pm_dist_sma_20_pct", 0.0)
        dist_sma50 = _col_series(test_df, "pm_dist_sma_50_pct", 0.0)
        rel_vol = _col_series(test_df, "vd_rel_vol_20d", 1.0)
        bb_pct = _col_series(test_df, "pm_bb_pct_b_20", 0.5)
        atr = _col_series(test_df, "pm_atr_14_pct", 1.6)
        regime_sc = _col_series(test_df, "ms_regime_score", 0.0)

        # Rule points in [-1.0, +1.0]
        trend_pts = np.where(dist_sma20 > 0, 0.25, -0.25) + np.where(dist_sma50 > 0, 0.20, -0.20)
        macd_pts = np.where(macd_h > 0, 0.20, -0.20)
        # RSI constructive zone (42-65) or oversold bounce (<32) vs overbought (>74)
        rsi_pts = np.select(
            [rsi14 < 32.0, (rsi14 >= 45.0) & (rsi14 <= 65.0), rsi14 > 74.0],
            [0.25, 0.15, -0.30],
            default=-0.05,
        )
        vol_mult = np.where(rel_vol > 1.15, 1.15, 0.95)
        total_score = (trend_pts + macd_pts + rsi_pts + 0.10 * regime_sc - 0.10 * (bb_pct - 0.5)) * vol_mult

        prob_up = np.clip(0.50 + np.tanh(total_score) * 0.18, 0.30, 0.72)
        exp_1d = np.clip(total_score * 0.45 * atr, -2.8, 2.8)
        exp_5d = np.clip(total_score * 1.15 * atr, -6.5, 6.5)
        downside_mae = (-0.75 * atr).clip(-6.0, -0.35)
        upside_mfe = (0.78 * atr).clip(0.35, 6.0)

        return pd.DataFrame(
            {
                "prob_next_day_up": prob_up,
                "expected_next_day_return_pct": exp_1d,
                "expected_5d_return_pct": exp_5d,
                "downside_risk_mae_pct": downside_mae.values,
                "expected_upside_mfe_pct": upside_mfe.values,
                "model_agreement_score": np.full(len(test_df), 0.65),
                "confidence_score": np.clip(50.0 + np.abs(prob_up - 0.5) * 150.0, 35.0, 86.0),
            },
            index=test_df.index,
        )


class LinearRidgeLogisticBaseline:
    """Benchmark 3: L2-Regularized Logistic Regression + Ridge Regression Baseline."""

    name: str = "linear_ridge_logistic"

    def __init__(self, c_logreg: float = 0.02, ridge_alpha: float = 150.0) -> None:
        self.c_logreg = c_logreg
        self.ridge_alpha = ridge_alpha
        self.active_features_: list[str] = []

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "LinearRidgeLogisticBaseline":
        # Keep features that have at least 15% non-null values in `train_df`
        valid_cols = [
            c for c in feature_cols
            if c in train_df.columns and int(train_df[c].notna().sum()) >= max(25, int(0.15 * len(train_df)))
        ]
        self.active_features_ = valid_cols

        sub_1d = train_df[train_df["target_next_day_ret"].notna() & train_df["target_next_day_up"].notna()].copy()
        X_1d = sub_1d[self.active_features_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y_up = sub_1d["target_next_day_up"].astype(int).values
        y_r1 = pd.to_numeric(sub_1d["target_next_day_ret"], errors="coerce").clip(-8.0, 8.0).values
        y_mae = pd.to_numeric(sub_1d["target_next_day_mae"], errors="coerce").fillna(-1.2).clip(-8.0, 0.0).values
        y_mfe = pd.to_numeric(sub_1d["target_next_day_mfe"], errors="coerce").fillna(1.2).clip(0.0, 8.0).values

        self.clf_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=self.c_logreg, max_iter=1000, random_state=42)),
            ]
        )
        self.clf_.fit(X_1d, y_up)

        self.reg_1d_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.ridge_alpha, random_state=42)),
            ]
        )
        self.reg_1d_.fit(X_1d, y_r1)

        self.reg_mae_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.ridge_alpha, random_state=42)),
            ]
        )
        self.reg_mae_.fit(X_1d, y_mae)

        self.reg_mfe_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.ridge_alpha, random_state=42)),
            ]
        )
        self.reg_mfe_.fit(X_1d, y_mfe)

        # Strictly purge last 5 dates of train_df for 5D target so 5D forward returns never overlap test window
        unique_dates = sorted(train_df["observation_date"].unique())
        cutoff_5d_date = unique_dates[-6] if len(unique_dates) > 10 else unique_dates[-1]
        sub_5d = train_df[
            (train_df["observation_date"] <= cutoff_5d_date) & train_df["target_5d_ret"].notna()
        ].copy()
        if len(sub_5d) >= 30:
            X_5d = sub_5d[self.active_features_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            y_r5 = pd.to_numeric(sub_5d["target_5d_ret"], errors="coerce").clip(-15.0, 15.0).values
            self.reg_5d_ = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=self.ridge_alpha * 1.5, random_state=42)),
                ]
            )
            self.reg_5d_.fit(X_5d, y_r5)
        else:
            self.reg_5d_ = self.reg_1d_

        return self

    def predict_df(self, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        X_test = test_df[self.active_features_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw_prob = self.clf_.predict_proba(X_test)[:, 1]
        # Shrink extreme linear probabilities toward [0.25, 0.75] to prevent uncalibrated linear extrapolation
        prob_up = np.clip(0.50 + 0.55 * (raw_prob - 0.50), 0.25, 0.75)
        exp_1d = np.clip(self.reg_1d_.predict(X_test), -3.5, 3.5)
        exp_5d = np.clip(self.reg_5d_.predict(X_test), -8.5, 8.5)
        downside_mae = np.clip(self.reg_mae_.predict(X_test), -6.5, -0.25)
        upside_mfe = np.clip(self.reg_mfe_.predict(X_test), 0.25, 6.5)

        return pd.DataFrame(
            {
                "prob_next_day_up": prob_up,
                "expected_next_day_return_pct": exp_1d,
                "expected_5d_return_pct": exp_5d,
                "downside_risk_mae_pct": downside_mae,
                "expected_upside_mfe_pct": upside_mfe,
                "model_agreement_score": np.full(len(test_df), 0.70),
                "confidence_score": np.clip(50.0 + np.abs(prob_up - 0.5) * 145.0, 35.0, 88.0),
            },
            index=test_df.index,
        )
