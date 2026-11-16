"""Synthetic integrity tests for purged/embargoed walk-forward validation & PIT alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.backtesting.walk_forward import run_walk_forward_backtest
from backend.features.feature_engine import _build_pit_fundamental_frame


def _make_synthetic_panel(n_dates: int = 120, n_symbols: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_dates).strftime("%Y-%m-%d").tolist()
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rows = []
    for s_idx, sym in enumerate(symbols):
        ret1 = rng.normal(0.05, 1.4, size=n_dates)
        for d_idx, dt in enumerate(dates):
            r5 = float(np.sum(ret1[d_idx + 1 : d_idx + 6])) if d_idx + 5 < n_dates else np.nan
            r1 = float(ret1[d_idx + 1]) if d_idx + 1 < n_dates else np.nan
            rows.append(
                {
                    "symbol": sym,
                    "observation_date": dt,
                    "sector": "IT Services" if s_idx % 2 == 0 else "Banking",
                    "pm_atr_14_pct": 1.6,
                    "pm_ret_1d_pct": float(ret1[d_idx]),
                    "pm_ret_5d_pct": float(np.sum(ret1[max(0, d_idx - 4) : d_idx + 1])),
                    "pm_ret_20d_pct": float(np.sum(ret1[max(0, d_idx - 19) : d_idx + 1])),
                    "pm_rsi_14": float(50.0 + rng.normal(0, 10)),
                    "pm_macd_hist": float(rng.normal(0, 0.5)),
                    "pm_dist_sma50_pct": float(rng.normal(0, 2.0)),
                    "pm_dist_sma200_pct": float(rng.normal(0, 4.0)),
                    "vd_amihud_illiquidity_20d": 0.002,
                    "ms_india_vix_level": 14.5,
                    "ms_nifty_dist_sma50_pct": 1.0,
                    "ms_nifty_dist_sma200_pct": 2.0,
                    "ms_nifty_ret_1d_pct": 0.04,
                    "ms_nifty_ret_20d_pct": 1.2,
                    "f_noise_a": float(rng.normal(0, 1)),
                    "f_noise_b": float(rng.normal(0, 1)),
                    "target_next_day_ret": r1,
                    "target_next_day_up": int(r1 > 0) if pd.notna(r1) else np.nan,
                    "target_5d_ret": r5,
                    "target_next_day_mae": -1.0,
                    "target_next_day_mfe": 1.2,
                }
            )
    return pd.DataFrame(rows)


def test_purge_and_embargo_protect_5d_horizon_and_prevent_overlap() -> None:
    df = _make_synthetic_panel(n_dates=130, n_symbols=4)
    cand_cols = [c for c in df.columns if c.startswith(("pm_", "vd_", "ms_", "f_"))]
    summary, oos_df, _ = run_walk_forward_backtest(
        df,
        cand_cols,
        initial_train_dates=60,
        step_dates=20,
        purge_gap_days=5,
        embargo_sessions=1,
        top_k_features=5,
    )
    assert summary.purge_gap_days == 5
    assert summary.embargo_sessions == 1
    ml_oos = oos_df[oos_df["model_name"] == "ml_ensemble"]
    all_dates = sorted(df["observation_date"].unique().tolist())

    # Fold 1 test must start at index >= 60 + 5 + 1 = 66
    first_test_dt = ml_oos[ml_oos["fold_id"] == 1]["observation_date"].min()
    assert all_dates.index(first_test_dt) >= 66

    # No duplicate (symbol, observation_date) across folds
    assert ml_oos.duplicated(subset=["symbol", "observation_date"]).sum() == 0


def test_in_fold_selection_and_fitting_are_invariant_to_test_future_leak() -> None:
    df = _make_synthetic_panel(n_dates=110, n_symbols=4, seed=99)
    all_dates = sorted(df["observation_date"].unique().tolist())
    test_cutoff = all_dates[66]

    # Add a planted leak feature that is pure noise in train, but exact future target in test
    df_leaked = df.copy()
    is_test = df_leaked["observation_date"] >= test_cutoff
    df_leaked["f_planted_leak"] = np.where(is_test, df_leaked["target_next_day_ret"].fillna(0.0) * 10.0, 0.0)

    df_clean = df.copy()
    df_clean["f_planted_leak"] = 0.0

    cand_cols = [c for c in df_clean.columns if c.startswith(("pm_", "vd_", "ms_", "f_"))]

    _, oos_clean, _ = run_walk_forward_backtest(
        df_clean, cand_cols, initial_train_dates=60, step_dates=40, purge_gap_days=5, embargo_sessions=1, top_k_features=5
    )
    _, oos_leaked, _ = run_walk_forward_backtest(
        df_leaked, cand_cols, initial_train_dates=60, step_dates=40, purge_gap_days=5, embargo_sessions=1, top_k_features=5
    )

    p_clean = oos_clean[oos_clean["model_name"] == "ml_ensemble"]["prob_next_day_up"].values
    p_leaked = oos_leaked[oos_leaked["model_name"] == "ml_ensemble"]["prob_next_day_up"].values
    assert np.allclose(p_clean, p_leaked, atol=1e-6)


def test_pit_fundamental_publication_date_respects_post_market_shift() -> None:
    from backend.storage.quality import _infer_publication_timestamp

    pub_date = pd.Timestamp(_infer_publication_timestamp("Mar 2025")[:10])
    obs_dates = pd.Series([pub_date.strftime("%Y-%m-%d"), (pub_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
    bundle = {
        "overview": {"roe_pct": 45.0, "roce_pct": 50.0},
        "last_5_annual_reports": [
            {"period": "Mar 2025", "revenue_cr": 1000.0, "net_profit_cr": 200.0, "opm_pct": 25.0}
        ],
    }
    pit_df = _build_pit_fundamental_frame(obs_dates, bundle)
    # Post-market 18:00 IST filing on pub_date must NOT be visible on pub_date, only on pub_date + 1
    assert pd.isna(pit_df.loc[0, "fq_opm_annual_pct"])
    assert pit_df.loc[1, "fq_opm_annual_pct"] == 25.0
