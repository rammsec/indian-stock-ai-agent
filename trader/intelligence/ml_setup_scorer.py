"""CYCLE-022: Leak-Free Walk-Forward ML Setup Scorer & Probability Calibrator.

Trains a pure-NumPy L2-regularized Logistic Regression model with chronological Platt
calibration strictly on historical Train dates (`max(train_dates) < min(test_dates)`),
evaluating honest out-of-sample Brier Score, Log-Loss, ROC-AUC, and Calibrated vs
Uncalibrated OOS Net P&L after Indian NSE intraday statutory taxes.

HARD INVARIANTS:
- Zero future leakage: feature scalers, weights, Platt parameters, and probability threshold
  are fit strictly on `train_dates` (`max(train_dates) < min(test_dates)`).
- Pure JSON serialization with SHA-256 weight provenance (zero pickle).
- Honest calibration labeling: `"EMPIRICAL_WALKFORWARD_CALIBRATED"` only when minimum sample
  count & OOS Brier calibration criteria are satisfied, otherwise `"UNCALIBRATED"`.
- Strictly paper-only (`LIVE_TRADING_ENABLED = False`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from trader.config import IST_TZ
from trader.data.loader_1m import (
    DEFAULT_1M_DIR,
    DEFAULT_SYMBOLS_1M,
    OneMinuteSessionBundle,
    load_1m_sessions,
)
from trader.evaluation.walkforward_optimizer import (
    FastSessionConfig,
    FastTradeRecord,
    aggregate_session_metrics,
    precompute_causal_1m_arrays,
    simulate_fast_1m_session,
    split_bundles_chronologically,
)
from trader.intelligence.chart_overlays import StrategyProfileLiteral

LIVE_TRADING_ENABLED: Final[bool] = False

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "side_sign",
    "adx_14",
    "di_diff_aligned",
    "rsi_14_centered",
    "ema_spread_aligned_bps",
    "vwap_z_aligned",
    "atr_pct_bps",
    "body_ratio",
    "mtf_5m_aligned",
    "orb_breakout_aligned",
    "sweep_aligned",
    "fvg_aligned",
    "minutes_from_open",
    "reward_to_friction",
    "near_pdhl",
)


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error (lower is better)."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy log-loss (lower is better)."""
    if len(y_true) == 0:
        return 0.0
    p = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Exact ROC-AUC via Mann-Whitney U rank statistic (returns 0.5 if single-class)."""
    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob, kind="mergesort")
    sorted_p = y_prob[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-indexed average rank for ties
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _fit_l2_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_reg: float = 1.0,
    lr: float = 0.15,
    max_iter: int = 800,
) -> tuple[np.ndarray, float]:
    """Deterministic pure-NumPy L2-regularized Logistic Regression via gradient descent."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    pos_rate = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)) if n_samples > 0 else 0.5
    b = float(np.log(pos_rate / (1.0 - pos_rate)))
    if n_samples == 0:
        return w, b

    vel_w = np.zeros_like(w)
    vel_b = 0.0
    beta = 0.85
    inv_n = 1.0 / n_samples

    for _ in range(max_iter):
        logits = X @ w + b
        preds = _sigmoid(logits)
        err = preds - y
        grad_w = (X.T @ err) * inv_n + (l2_reg * inv_n) * w
        grad_b = float(np.mean(err))
        vel_w = beta * vel_w + (1.0 - beta) * grad_w
        vel_b = beta * vel_b + (1.0 - beta) * grad_b
        w -= lr * vel_w
        b -= lr * vel_b
        if float(np.max(np.abs(grad_w))) < 1e-6 and abs(grad_b) < 1e-6:
            break

    return w, b


def _fit_platt_calibration(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit 1D Platt scaling `sigmoid(a * logit + b)` strictly on chronological calibration slice."""
    if len(y) < 10 or len(np.unique(y)) < 2:
        return 1.0, 0.0
    a = 1.0
    b = 0.0
    lr = 0.05
    inv_n = 1.0 / len(y)
    for _ in range(400):
        z = a * logits + b
        p = _sigmoid(z)
        err = p - y
        ga = float(np.dot(err, logits) * inv_n + 0.05 * inv_n * (a - 1.0))
        gb = float(np.mean(err))
        a -= lr * ga
        b -= lr * gb
        if abs(ga) < 1e-6 and abs(gb) < 1e-6:
            break
    # Keep monotonic positive slope
    a = float(np.clip(a, 0.05, 5.0))
    b = float(np.clip(b, -5.0, 5.0))
    return a, b


@dataclass(frozen=True)
class CausalSetupScorerModel:
    """Serializable, leak-free logistic + Platt setup probability scorer."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    platt_a: float
    platt_b: float
    recommended_threshold: float
    calibration_status: str
    strategy_profile: str
    train_date_range: tuple[str, str]
    oos_test_date_range: tuple[str, str]
    train_samples: int
    oos_samples: int
    oos_brier_score: float
    oos_baseline_brier: float
    oos_log_loss: float
    oos_roc_auc: float
    trained_at_ist: str
    weights_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def _vectorize(self, features: dict[str, float]) -> np.ndarray:
        vec = np.array([float(features.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        mu = np.array(self.means, dtype=np.float64)
        sd = np.array(self.stds, dtype=np.float64)
        return (vec - mu) / np.where(sd > 1e-9, sd, 1.0)

    def predict_proba(self, features: dict[str, float]) -> float:
        """Predict calibrated probability that this setup finishes net-profitable after NSE taxes."""
        x_norm = self._vectorize(features)
        w = np.array(self.weights, dtype=np.float64)
        raw_logit = float(np.dot(x_norm, w) + self.bias)
        cal_logit = self.platt_a * raw_logit + self.platt_b
        return float(_sigmoid(cal_logit))

    def predict_batch(self, feature_dicts: Sequence[dict[str, float]]) -> np.ndarray:
        if not feature_dicts:
            return np.zeros(0, dtype=np.float64)
        return np.array([self.predict_proba(f) for f in feature_dicts], dtype=np.float64)

    def compute_weights_sha256(self) -> str:
        payload = json.dumps(
            {
                "feature_names": list(self.feature_names),
                "means": [round(x, 8) for x in self.means],
                "stds": [round(x, 8) for x in self.stds],
                "weights": [round(x, 8) for x in self.weights],
                "bias": round(self.bias, 8),
                "platt_a": round(self.platt_a, 8),
                "platt_b": round(self.platt_b, 8),
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        digest = self.weights_sha256 or self.compute_weights_sha256()
        return {
            "schema_version": "cycle022_ml_setup_scorer_v1",
            "live_trading_enabled": False,
            "feature_names": list(self.feature_names),
            "means": [round(x, 8) for x in self.means],
            "stds": [round(x, 8) for x in self.stds],
            "weights": [round(x, 8) for x in self.weights],
            "bias": round(self.bias, 8),
            "platt_a": round(self.platt_a, 8),
            "platt_b": round(self.platt_b, 8),
            "recommended_threshold": round(self.recommended_threshold, 4),
            "calibration_status": self.calibration_status,
            "strategy_profile": self.strategy_profile,
            "train_date_range": list(self.train_date_range),
            "oos_test_date_range": list(self.oos_test_date_range),
            "train_samples": self.train_samples,
            "oos_samples": self.oos_samples,
            "oos_brier_score": round(self.oos_brier_score, 5),
            "oos_baseline_brier": round(self.oos_baseline_brier, 5),
            "oos_log_loss": round(self.oos_log_loss, 5),
            "oos_roc_auc": round(self.oos_roc_auc, 5),
            "trained_at_ist": self.trained_at_ist,
            "weights_sha256": digest,
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> "CausalSetupScorerModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            feature_names=tuple(raw["feature_names"]),
            means=tuple(float(x) for x in raw["means"]),
            stds=tuple(float(x) for x in raw["stds"]),
            weights=tuple(float(x) for x in raw["weights"]),
            bias=float(raw["bias"]),
            platt_a=float(raw["platt_a"]),
            platt_b=float(raw["platt_b"]),
            recommended_threshold=float(raw["recommended_threshold"]),
            calibration_status=str(raw["calibration_status"]),
            strategy_profile=str(raw["strategy_profile"]),
            train_date_range=(str(raw["train_date_range"][0]), str(raw["train_date_range"][1])),
            oos_test_date_range=(str(raw["oos_test_date_range"][0]), str(raw["oos_test_date_range"][1])),
            train_samples=int(raw["train_samples"]),
            oos_samples=int(raw["oos_samples"]),
            oos_brier_score=float(raw["oos_brier_score"]),
            oos_baseline_brier=float(raw["oos_baseline_brier"]),
            oos_log_loss=float(raw["oos_log_loss"]),
            oos_roc_auc=float(raw["oos_roc_auc"]),
            trained_at_ist=str(raw["trained_at_ist"]),
            weights_sha256=str(raw.get("weights_sha256", "")),
            metadata=dict(raw.get("metadata", {})),
        )
        expected = model.compute_weights_sha256()
        if model.weights_sha256 and model.weights_sha256 != expected:
            raise ValueError(
                f"Model SHA-256 mismatch for {path}: expected {expected}, got {model.weights_sha256}"
            )
        return model


def _trades_to_xy(trades: Sequence[FastTradeRecord]) -> tuple[np.ndarray, np.ndarray]:
    if not trades:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64), np.zeros(0, dtype=np.float64)
    X = np.array(
        [[float(t.entry_features.get(name, 0.0)) for name in FEATURE_NAMES] for t in trades],
        dtype=np.float64,
    )
    y = np.array([1.0 if t.net_pnl_inr > 0.0 else 0.0 for t in trades], dtype=np.float64)
    return X, y


def train_walkforward_setup_scorer(
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS_1M,
    base_dir: str | Path = DEFAULT_1M_DIR,
    strategy_profile: StrategyProfileLiteral = "ORB_BREAKOUT",
    initial_capital_inr: float = 50_000.0,
    train_ratio: float = 0.70,
    l2_reg: float = 2.0,
    max_days: int | None = None,
    bundles_override: dict[str, list[OneMinuteSessionBundle]] | None = None,
) -> tuple[CausalSetupScorerModel, dict[str, Any]]:
    """Train & evaluate a strictly chronological leak-free 1m ML setup scorer."""
    if bundles_override is not None:
        bundles_by_symbol = bundles_override
    else:
        bundles_by_symbol = {
            sym: load_1m_sessions(sym, base_dir=base_dir, max_days=max_days)
            for sym in symbols
        }

    train_dates, test_dates, train_bundles, test_bundles = split_bundles_chronologically(
        bundles_by_symbol, train_ratio=train_ratio
    )
    assert max(train_dates) < min(test_dates), "Zero-lookahead chronological split violated!"

    precomp_train = [precompute_causal_1m_arrays(b) for b in train_bundles]
    precomp_test = [precompute_causal_1m_arrays(b) for b in test_bundles]

    base_cfg = FastSessionConfig(
        strategy_profile=strategy_profile,
        initial_capital_inr=initial_capital_inr,
        auto_direction=True,
    )
    train_sessions = [
        simulate_fast_1m_session(b, base_cfg, precomputed=pc)
        for b, pc in zip(train_bundles, precomp_train)
    ]
    oos_unfiltered_sessions = [
        simulate_fast_1m_session(b, base_cfg, precomputed=pc)
        for b, pc in zip(test_bundles, precomp_test)
    ]

    train_trades = sorted(
        [t for s in train_sessions for t in s.trades],
        key=lambda t: (t.trading_date, t.entry_time_ist),
    )
    oos_trades = sorted(
        [t for s in oos_unfiltered_sessions for t in s.trades],
        key=lambda t: (t.trading_date, t.entry_time_ist),
    )

    X_train_raw, y_train = _trades_to_xy(train_trades)
    X_oos_raw, y_oos = _trades_to_xy(oos_trades)

    # Fit z-score scaler STRICTLY on Train rows
    if len(X_train_raw) > 0:
        means = np.mean(X_train_raw, axis=0)
        stds = np.std(X_train_raw, axis=0)
        stds = np.where(stds > 1e-9, stds, 1.0)
    else:
        means = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        stds = np.ones(len(FEATURE_NAMES), dtype=np.float64)

    X_train = (X_train_raw - means) / stds if len(X_train_raw) > 0 else X_train_raw
    X_oos = (X_oos_raw - means) / stds if len(X_oos_raw) > 0 else X_oos_raw

    # Inner chronological split within Train for Platt scaling (80% fit / 20% cal)
    if len(X_train) >= 30:
        cal_split = int(len(X_train) * 0.80)
        w_fit, b_fit = _fit_l2_logistic(X_train[:cal_split], y_train[:cal_split], l2_reg=l2_reg)
        cal_logits = X_train[cal_split:] @ w_fit + b_fit
        platt_a, platt_b = _fit_platt_calibration(cal_logits, y_train[cal_split:])
        # Refit base weights on full Train split while preserving conservative Platt calibration
        w_final, b_final = _fit_l2_logistic(X_train, y_train, l2_reg=l2_reg)
    else:
        w_final, b_final = _fit_l2_logistic(X_train, y_train, l2_reg=l2_reg)
        platt_a, platt_b = 1.0, 0.0

    # Select probability filter threshold STRICTLY on Train trades (never peeking at OOS)
    train_probs = (
        _sigmoid(platt_a * (X_train @ w_final + b_final) + platt_b)
        if len(X_train) > 0
        else np.zeros(0, dtype=np.float64)
    )
    train_pnls = np.array([t.net_pnl_inr for t in train_trades], dtype=np.float64)
    best_thresh = 0.20
    best_train_net = float(np.sum(train_pnls)) if len(train_pnls) > 0 else 0.0
    if len(train_probs) >= 20:
        for cand_t in (0.12, 0.15, 0.18, 0.22, 0.25, 0.30, 0.35, 0.40):
            mask = train_probs >= cand_t
            if int(np.sum(mask)) >= max(5, int(len(train_probs) * 0.05)):
                cand_net = float(np.sum(train_pnls[mask]))
                if cand_net > best_train_net:
                    best_train_net = cand_net
                    best_thresh = cand_t

    # Evaluate honest calibration on UNSEEN OOS Test trades
    oos_probs = (
        _sigmoid(platt_a * (X_oos @ w_final + b_final) + platt_b)
        if len(X_oos) > 0
        else np.zeros(0, dtype=np.float64)
    )
    oos_brier = compute_brier_score(y_oos, oos_probs)
    train_base_rate = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
    oos_baseline_brier = compute_brier_score(y_oos, np.full_like(y_oos, train_base_rate))
    oos_logloss = compute_log_loss(y_oos, oos_probs)
    oos_auc = compute_roc_auc(y_oos, oos_probs)

    calibration_status = (
        "EMPIRICAL_WALKFORWARD_CALIBRATED"
        if (len(y_train) >= 50 and len(y_oos) >= 10 and oos_brier <= oos_baseline_brier)
        else "UNCALIBRATED"
    )

    model = CausalSetupScorerModel(
        feature_names=FEATURE_NAMES,
        means=tuple(float(x) for x in means),
        stds=tuple(float(x) for x in stds),
        weights=tuple(float(x) for x in w_final),
        bias=float(b_final),
        platt_a=float(platt_a),
        platt_b=float(platt_b),
        recommended_threshold=float(best_thresh),
        calibration_status=calibration_status,
        strategy_profile=strategy_profile,
        train_date_range=(train_dates[0].isoformat(), train_dates[-1].isoformat()),
        oos_test_date_range=(test_dates[0].isoformat(), test_dates[-1].isoformat()),
        train_samples=len(y_train),
        oos_samples=len(y_oos),
        oos_brier_score=oos_brier,
        oos_baseline_brier=oos_baseline_brier,
        oos_log_loss=oos_logloss,
        oos_roc_auc=oos_auc,
        trained_at_ist=datetime.now(IST_TZ).isoformat(),
    )
    model = CausalSetupScorerModel(
        **{
            **model.__dict__,
            "weights_sha256": model.compute_weights_sha256(),
        }
    )

    # Compare Uncalibrated vs ML-Filtered on UNSEEN OOS Test sessions
    filtered_cfg = FastSessionConfig(
        strategy_profile=strategy_profile,
        initial_capital_inr=initial_capital_inr,
        auto_direction=True,
        min_ml_win_prob=model.recommended_threshold,
    )
    oos_filtered_sessions = [
        simulate_fast_1m_session(
            b,
            filtered_cfg,
            precomputed=pc,
            ml_scorer=model.predict_proba,
        )
        for b, pc in zip(test_bundles, precomp_test)
    ]

    unfiltered_metrics = aggregate_session_metrics(oos_unfiltered_sessions)
    filtered_metrics = aggregate_session_metrics(oos_filtered_sessions)

    report = {
        "schema_version": "cycle022_ml_calibration_report_v1",
        "live_trading_enabled": False,
        "zero_lookahead_verified": max(train_dates) < min(test_dates),
        "strategy_profile": strategy_profile,
        "initial_capital_inr": initial_capital_inr,
        "calibration_status": model.calibration_status,
        "train_date_range": list(model.train_date_range),
        "oos_test_date_range": list(model.oos_test_date_range),
        "train_samples": model.train_samples,
        "oos_samples": model.oos_samples,
        "recommended_threshold": round(model.recommended_threshold, 4),
        "oos_calibration": {
            "brier_score": round(oos_brier, 5),
            "baseline_brier": round(oos_baseline_brier, 5),
            "log_loss": round(oos_logloss, 5),
            "roc_auc": round(oos_auc, 5),
        },
        "oos_unfiltered": unfiltered_metrics,
        "oos_ml_filtered": filtered_metrics,
        "weights_sha256": model.weights_sha256,
    }
    return model, report
