"""Phase 5 Controlled Walk-Forward Experiments (A–E across 1D, 5D, 10D) & Live Intelligence Runner.

Runs:
1. Live Phase 5 6-category news collection, extraction, cross-stock impact & 1D/5D/10D outlook
   pipeline for the 5 pilot stocks (`RELIANCE`, `TCS`, `TATAELXSI`, `ITC`, `BIRLACABLE`).
2. Controlled Walk-Forward Experiments A–E on the 5 pilot stocks using identical dates,
   v3.2 5-session purge + 1-session embargo (10-session purge for 10D horizon),
   in-fold feature selection, next-open execution, Indian transaction costs, and
   `naive_momentum` benchmark:
   - Exp A: 237 quantitative features only
   - Exp B: Exp A + company news features
   - Exp C: Exp A + company & promoter features
   - Exp D: Exp A + company, promoter, industry & competitor features
   - Exp E: Exp A + all 17 news, regulatory, macro & cross-stock features
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.backtesting.walk_forward import run_walk_forward_backtest
from backend.config import BACKTESTS_DATA_DIR, FEATURES_DATA_DIR, IST_TZ, PROCESSED_DATA_DIR
from backend.news_intel.features import (
    NEWS_CATEGORY_GROUPS,
    NEWS_FEATURE_NAMES,
    reconstruct_pit_historical_news_features,
)
from backend.news_intel.pipeline import run_phase5_intelligence_pipeline
from backend.news_intel.relationships import PILOT_UNIVERSE_5


def _load_237_features() -> list[str]:
    cat_path = FEATURES_DATA_DIR / "feature_catalog.json"
    if cat_path.exists():
        raw = json.loads(cat_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(x["name"]) for x in raw if isinstance(x, dict) and "name" in x]
        if isinstance(raw, dict) and "features" in raw:
            return [str(x["name"]) for x in raw["features"] if isinstance(x, dict) and "name" in x]
    return []


def _horizon_oos_metrics(oos_df: pd.DataFrame, target_ret_col: str) -> dict[str, float]:
    """Compute directional accuracy, ROC AUC, Brier, and Rank IC for 5D or 10D target."""
    if oos_df.empty or target_ret_col not in oos_df.columns:
        return {"roc_auc": 0.5, "directional_accuracy_pct": 50.0, "brier_score": 0.25, "rank_ic": 0.0}

    sub = oos_df.dropna(subset=[target_ret_col, "prob_next_day_up"]).copy()
    if len(sub) < 20:
        return {"roc_auc": 0.5, "directional_accuracy_pct": 50.0, "brier_score": 0.25, "rank_ic": 0.0}

    y_ret = pd.to_numeric(sub[target_ret_col], errors="coerce")
    y_up = (y_ret > 0).astype(int).values
    p_up = pd.to_numeric(sub["prob_next_day_up"], errors="coerce").clip(0.01, 0.99).values
    pred_up = (p_up >= 0.50).astype(int)

    try:
        auc = float(roc_auc_score(y_up, p_up))
    except Exception:
        auc = 0.50
    acc = float((pred_up == y_up).mean() * 100.0)
    brier = float(brier_score_loss(y_up, p_up))
    ic_raw = pd.Series(p_up).rank().corr(y_ret.rank())
    ic = float(ic_raw) if pd.notna(ic_raw) else 0.0
    return {
        "roc_auc": round(auc, 4),
        "directional_accuracy_pct": round(acc, 2),
        "brier_score": round(brier, 4),
        "rank_ic": round(ic, 4),
    }


def run_all_experiments() -> dict[str, object]:
    feat_path = FEATURES_DATA_DIR / "daily_features.parquet"
    eq_path = PROCESSED_DATA_DIR / "equities_daily.parquet"

    df_all = pd.read_parquet(feat_path)
    df_pilot = df_all[df_all["symbol"].isin(PILOT_UNIVERSE_5)].copy()
    df_pilot = df_pilot.sort_values(["observation_date", "symbol"]).reset_index(drop=True)
    df_pilot = reconstruct_pit_historical_news_features(df_pilot)

    ohlc_df = pd.read_parquet(eq_path) if eq_path.exists() else None
    if ohlc_df is not None:
        ohlc_df = ohlc_df[ohlc_df["symbol"].isin(PILOT_UNIVERSE_5)].copy()

    base_237 = [c for c in _load_237_features() if c in df_pilot.columns]
    if not base_237:
        base_237 = [
            c for c in df_pilot.columns
            if not c.startswith("target_")
            and not c.startswith("news_")
            and c not in {"symbol", "observation_date", "sector"}
            and pd.api.types.is_numeric_dtype(df_pilot[c])
        ]

    experiments: dict[str, list[str]] = {
        "A_quant_237_only": base_237,
        "B_quant_plus_company_news": base_237 + NEWS_CATEGORY_GROUPS["company_only"],
        "C_quant_plus_company_promoter": base_237 + NEWS_CATEGORY_GROUPS["company_only"] + NEWS_CATEGORY_GROUPS["promoter"],
        "D_quant_plus_ind_competitor": (
            base_237
            + NEWS_CATEGORY_GROUPS["company_only"]
            + NEWS_CATEGORY_GROUPS["promoter"]
            + NEWS_CATEGORY_GROUPS["industry_competitor"]
        ),
        "E_quant_plus_all_news_cross": base_237 + NEWS_FEATURE_NAMES,
    }

    results: dict[str, dict[str, object]] = {}
    naive_benchmark: dict[str, object] = {}

    for exp_name, feat_cols in experiments.items():
        # 1D & 5D share the 5-session purge + 1-session embargo walk-forward
        summary_1d, oos_all_1d, _ = run_walk_forward_backtest(
            df_pilot,
            feat_cols,
            initial_train_dates=240,
            step_dates=42,
            purge_gap_days=5,
            embargo_sessions=1,
            ohlc_df=ohlc_df,
            top_k_features=40,
        )
        ml_1d = summary_1d.models["ml_ensemble"]
        ml_oos_1d = oos_all_1d[oos_all_1d["model_name"] == "ml_ensemble"]
        m5d = _horizon_oos_metrics(ml_oos_1d, "target_5d_ret")

        # 10D horizon with 10-session purge + 1-session embargo to prevent 10-day label overlap
        df_10d = df_pilot.copy()
        df_10d["target_next_day_ret"] = pd.to_numeric(df_10d["target_10d_ret"], errors="coerce")
        df_10d["target_next_day_up"] = (df_10d["target_next_day_ret"] > 0).astype(float)
        summary_10d, oos_all_10d, _ = run_walk_forward_backtest(
            df_10d,
            feat_cols,
            initial_train_dates=240,
            step_dates=42,
            purge_gap_days=10,
            embargo_sessions=1,
            ohlc_df=ohlc_df,
            top_k_features=40,
        )
        ml_oos_10d = oos_all_10d[oos_all_10d["model_name"] == "ml_ensemble"]
        m10d = _horizon_oos_metrics(ml_oos_10d, "target_10d_ret")

        if not naive_benchmark and "naive_momentum" in summary_1d.models:
            nm = summary_1d.models["naive_momentum"]
            nm_oos = oos_all_1d[oos_all_1d["model_name"] == "naive_momentum"]
            naive_benchmark = {
                "1D": {
                    "roc_auc": nm.roc_auc,
                    "directional_accuracy_pct": nm.directional_accuracy_all_pct,
                    "brier_score": nm.brier_score,
                    "rank_ic": nm.rank_ic_1d,
                    "net_return_pct": nm.net_cumulative_return_pct,
                    "net_sharpe": nm.annualized_sharpe_net,
                    "max_drawdown_pct": nm.max_drawdown_net_pct,
                    "profit_factor": nm.profit_factor_net,
                    "expectancy_bps": nm.expectancy_per_trade_bps,
                },
                "5D": _horizon_oos_metrics(nm_oos, "target_5d_ret"),
                "10D": _horizon_oos_metrics(oos_all_10d[oos_all_10d["model_name"] == "naive_momentum"], "target_10d_ret"),
            }

        results[exp_name] = {
            "feature_count": len(feat_cols),
            "folds": summary_1d.total_walk_forward_folds,
            "1D": {
                "roc_auc": ml_1d.roc_auc,
                "directional_accuracy_pct": ml_1d.directional_accuracy_all_pct,
                "brier_score": ml_1d.brier_score,
                "rank_ic": ml_1d.rank_ic_1d,
                "net_return_pct": ml_1d.net_cumulative_return_pct,
                "net_sharpe": ml_1d.annualized_sharpe_net,
                "max_drawdown_pct": ml_1d.max_drawdown_net_pct,
                "profit_factor": ml_1d.profit_factor_net,
                "expectancy_bps": ml_1d.expectancy_per_trade_bps,
                "active_trades": ml_1d.active_trades_count,
            },
            "5D": m5d,
            "10D": m10d,
        }

    report = {
        "generated_at": datetime.now(IST_TZ).isoformat(),
        "pilot_universe": list(PILOT_UNIVERSE_5),
        "purge_and_embargo": {"1D_5D": "5 sessions purge + 1 session embargo", "10D": "10 sessions purge + 1 session embargo"},
        "execution_mode": "next_open_delivery_capped",
        "historical_pit_note": (
            "Experiments A–E use genuine point-in-time historical event/promoter/sector/macro proxies "
            "(ne_*, inst_*, rel_*, ms_*) reconstructed into the 17 Phase 5 features without fabricating "
            "historical news articles. Live RSS/Gemini extraction accumulates forward-only in structured_events.parquet."
        ),
        "naive_momentum_benchmark": naive_benchmark,
        "experiments": results,
    }

    BACKTESTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BACKTESTS_DATA_DIR / "phase5_experiments.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print("1. Running Phase 5 Live News & Cross-Stock Intelligence Pipeline (5 pilot stocks)...")
    payload = run_phase5_intelligence_pipeline(symbols=PILOT_UNIVERSE_5, use_network=True, persist=True)
    print(f"   Collected {len(payload.consolidated_feed)} deduplicated feed events across {len(payload.stocks)} stocks.")

    print("2. Running Controlled Walk-Forward Experiments A–E (1D, 5D, 10D) on 5 pilot stocks...")
    report = run_all_experiments()
    print(f"   Saved -> {BACKTESTS_DATA_DIR / 'phase5_experiments.json'}")
    for name, res in report["experiments"].items():
        d1 = res["1D"]
        d5 = res["5D"]
        d10 = res["10D"]
        print(
            f"   {name:30s} | 1D AUC={d1['roc_auc']:.4f} Net={d1['net_return_pct']:+.2f}% Sharpe={d1['net_sharpe']:.3f} "
            f"| 5D AUC={d5['roc_auc']:.4f} | 10D AUC={d10['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
