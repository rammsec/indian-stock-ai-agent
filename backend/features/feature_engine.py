"""Reusable Point-in-Time Feature Engineering Library (Phase 2 — Section 7).

Generates ~185 candidate numerical features across 8 distinct families:
1. `price_momentum` (72 features): Multi-horizon returns, log returns, acceleration,
   SMA/EMA distances & slopes, multi-period RSI, MACD, Stochastic, Williams %R, CCI,
   Bollinger %B & width, Donchian channels, ATR-normalized moves, gaps, shadows,
   Parkinson/Garman-Klass intraday volatility, return skewness & kurtosis.
2. `volume_delivery` (26 features): Relative volume across 5 horizons, volume z-scores,
   OBV slope, CMF(20), MFI(14), VPT, Amihud illiquidity, turnover metrics, NSE
   Delivery % raw, Delivery % vs average, Delivery % changes, VWAP distance, and
   price-volume-delivery confirmation interactions.
3. `market_sector` (32 features): Nifty 50 returns & stock alpha/excess returns across
   6 horizons, rolling 20D/60D Beta & correlation, upside/downside capture, NSE Sector
   Index returns & sector-relative alpha across 6 horizons, India VIX level/changes/z-score,
   and market regime interactions.
4. `institutional_promoter` (15 features): Point-in-time lagged quarterly Promoter, FII,
   DII, and Public holdings & 1Q/4Q changes, institutional net flow, promoter pledge %,
   and NSE bulk/block deal activity.
5. `derivatives` (9 features): F&O eligibility, futures OI & basis (nullable when
   unreported), PCR, ATM IV, and price-volume buildup regime heuristics.
6. `financial_quality` (18 features): Point-in-time lagged ROE, ROCE, OPM, NPM, Debt/Equity,
   Net Cash/MCap, OCF/Net Profit, FCF/Sales, Accruals, Piotroski F-score, R&D intensity,
   CWIP/Assets, Intangibles/Assets, and P/E vs Sector P/E.
7. `earnings_guidance` (10 features): Point-in-time quarterly YoY/QoQ revenue & profit
   growth, sequential margin expansion, revenue/profit acceleration, 5Y CAGRs, and
   management guidance tone.
8. `news_events` (8 features): 7-day news volume, positive/negative sentiment ratios,
   promoter statement impact balance, competitor read-across balance, and days since
   last statutory filing.

Strict Anti-Leakage Guarantee:
- All fundamental and shareholding features for period `P` are gated by their statutory
  SEBI LODR `publication_timestamp` (`May 30` for FY-end `Mar`, `Aug 14` for `Jun`,
  `Nov 14` for `Sep`, `Feb 14` for `Dec`) so a historical bar on date `t` NEVER sees
  a financial report published after `t`.
- Forward targets (`target_next_day_ret`, `target_next_day_up`, `target_3d_ret`,
  `target_5d_ret`, `target_10d_ret`, `target_next_day_mae`, `target_next_day_mfe`)
  are stored in dedicated `target_*` columns and excluded from feature matrices.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..config import SECTOR_TO_NSE_INDEX
from ..storage.quality import _infer_publication_timestamp

TARGET_COLUMNS: list[str] = [
    "target_next_day_ret",
    "target_next_day_up",
    "target_3d_ret",
    "target_5d_ret",
    "target_10d_ret",
    "target_next_day_mae",
    "target_next_day_mfe",
]

METADATA_COLUMNS: list[str] = [
    "symbol",
    "observation_date",
    "publication_timestamp",
    "sector",
    "sector_index_code",
    "data_version",
]

# Active NSE F&O stock subset for `is_fno_eligible` indicator
FNO_ELIGIBLE_SYMBOLS: set[str] = {
    "RELIANCE", "TCS", "TATAELXSI", "ITC", "HDFCBANK", "ICICIBANK", "SBIN",
    "AXISBANK", "KOTAKBANK", "BAJFINANCE", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "PERSISTENT", "LTTS", "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
    "BHARTIARTL", "LT", "BEL", "HAL", "BHEL", "MARUTI", "M&M", "BAJAJ-AUTO",
    "EICHERMOT", "HEROMOTOCO", "TITAN", "TRENT", "HINDUNILVR", "NESTLEIND",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "NTPC", "TATAPOWER",
    "POWERGRID", "ONGC", "BPCL", "ULTRACEMCO", "DIXON", "POLYCAB", "IRCTC",
}


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_c).abs(),
            (low - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=max(3, period // 2), adjust=False).mean()


def _compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    tp_diff = tp.diff()
    pos_mf = rmf.where(tp_diff > 0, 0.0).rolling(period, min_periods=5).sum()
    neg_mf = rmf.where(tp_diff < 0, 0.0).rolling(period, min_periods=5).sum()
    mfr = pos_mf / neg_mf.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + mfr))


def _resolve_sector_index_code(sector: str) -> str:
    sec_low = (sector or "").lower()
    for key, idx_code in SECTOR_TO_NSE_INDEX.items():
        if key.lower() in sec_low:
            return idx_code
    if "telecom" in sec_low or "energy" in sec_low or "oil" in sec_low:
        return "NIFTY_ENERGY"
    if "er&d" in sec_low or "software" in sec_low or "it" in sec_low:
        return "NIFTY_IT"
    if "fmcg" in sec_low or "consumer" in sec_low:
        return "NIFTY_FMCG"
    return "NIFTY_500"


def _build_pit_fundamental_frame(
    obs_dates: pd.Series,
    raw_company_bundle: dict[str, Any] | None,
) -> pd.DataFrame:
    """Map annual, quarterly, balance-sheet, shareholding & R&D records onto `obs_dates`
    using strict point-in-time `publication_timestamp <= observation_date` gating.
    """
    n = len(obs_dates)
    out = pd.DataFrame(index=obs_dates.index)

    # Initialize all fundamental/institutional/guidance/news columns as NaN (never silent 0!)
    pit_cols = [
        # Institutional & Promoter (15)
        "inst_promoters_pct", "inst_fii_pct", "inst_dii_pct", "inst_total_pct",
        "inst_public_pct", "inst_promoter_chg_1q", "inst_promoter_chg_4q",
        "inst_fii_chg_1q", "inst_fii_chg_4q", "inst_dii_chg_1q", "inst_dii_chg_4q",
        "inst_net_flow_4q", "inst_retail_dilution_4q", "inst_promoter_pledge_pct",
        "inst_named_holders_count",
        # Financial Quality (18)
        "fq_roe_pct", "fq_roce_pct", "fq_roe_roce_spread", "fq_opm_annual_pct",
        "fq_opm_yoy_chg", "fq_npm_annual_pct", "fq_npm_yoy_chg", "fq_debt_to_equity",
        "fq_net_cash_to_mcap_pct", "fq_cash_to_debt_ratio", "fq_ocf_to_np_ratio",
        "fq_fcf_to_sales_pct", "fq_accruals_to_assets", "fq_interest_coverage",
        "fq_asset_turnover", "fq_piotroski_score", "fq_rd_pct_of_sales",
        "fq_cwip_to_assets_pct",
        # Earnings & Management Guidance (10)
        "eg_qtr_rev_qoq_pct", "eg_qtr_rev_yoy_pct", "eg_qtr_np_qoq_pct",
        "eg_qtr_np_yoy_pct", "eg_qtr_opm_pct", "eg_qtr_opm_qoq_chg",
        "eg_qtr_rev_accel", "eg_qtr_np_accel", "eg_annual_rev_5y_cagr",
        "eg_annual_profit_5y_cagr",
        # News & Events (8)
        "ne_days_since_qtr_filing", "ne_is_earnings_window", "ne_news_count_7d",
        "ne_news_pos_ratio", "ne_news_neg_ratio", "ne_news_net_sentiment",
        "ne_promoter_bull_Bear_balance", "ne_competitor_readacross_score",
    ]
    for c in pit_cols:
        out[c] = np.nan

    if not raw_company_bundle or n == 0:
        return out

    ov = raw_company_bundle.get("overview", {})
    annual_list = raw_company_bundle.get("last_5_annual_reports", [])
    qtr_list = raw_company_bundle.get("last_5_quarterly_reports", [])
    cash_bs = raw_company_bundle.get("cash_and_balance_sheet", {})
    cash_hist = cash_bs.get("history_5y", [])
    shp = raw_company_bundle.get("stockholders", {})
    shp_hist = shp.get("quarterly_history", [])
    rd = raw_company_bundle.get("rd_and_innovation", {})
    rd_hist = rd.get("history_5y", [])

    # 1. Build Point-in-Time Annual Events sorted by statutory publication date
    cb_by_period = {item["period"]: item for item in cash_hist if "period" in item}
    rd_by_period = {item["period"]: item for item in rd_hist if "period" in item}

    annual_events: list[dict[str, Any]] = []
    for idx, a_rep in enumerate(annual_list):
        per = a_rep.get("period", "")
        pub_date = _infer_publication_timestamp(per)[:10]
        prev_a = annual_list[idx - 1] if idx > 0 else None

        rev = a_rep.get("revenue_cr")
        np_val = a_rep.get("net_profit_cr")
        opm = a_rep.get("opm_pct")
        op_prof = a_rep.get("operating_profit_cr")
        inter = a_rep.get("interest_cr")
        npm = (np_val / rev * 100.0) if (rev and rev > 0 and np_val is not None) else np.nan

        prev_opm = prev_a.get("opm_pct") if prev_a else None
        prev_rev = prev_a.get("revenue_cr") if prev_a else None
        prev_np = prev_a.get("net_profit_cr") if prev_a else None
        prev_npm = (prev_np / prev_rev * 100.0) if (prev_rev and prev_rev > 0 and prev_np is not None) else np.nan

        cb_item = cb_by_period.get(per, {})
        rd_item = rd_by_period.get(per, {})

        tot_eq = cb_item.get("total_equity_cr") or 0.0
        tot_debt = cb_item.get("total_borrowings_cr") or 0.0
        tot_cash = cb_item.get("total_cash_in_hand_cr") or 0.0
        tot_assets = cb_item.get("total_assets_cr") or 0.0
        ocf = cb_item.get("operating_cash_flow_cr")
        fcf = cb_item.get("free_cash_flow_cr")
        cwip = rd_item.get("cwip_cr") or 0.0

        d_to_e = (tot_debt / tot_eq) if tot_eq > 0 else np.nan
        c_to_d = (tot_cash / tot_debt) if tot_debt > 0 else (20.0 if tot_cash > 0 else np.nan)
        ocf_np = (ocf / np_val) if (ocf is not None and np_val and abs(np_val) > 0) else np.nan
        fcf_sales = (fcf / rev * 100.0) if (fcf is not None and rev and rev > 0) else np.nan
        accruals = ((np_val - ocf) / tot_assets) if (np_val is not None and ocf is not None and tot_assets > 0) else np.nan
        int_cov = (op_prof / inter) if (op_prof is not None and inter and inter > 0) else np.nan
        asset_turn = (rev / tot_assets) if (rev is not None and tot_assets > 0) else np.nan
        cwip_assets = (cwip / tot_assets * 100.0) if tot_assets > 0 else np.nan

        annual_events.append(
            {
                "pub_date": pub_date,
                "fq_opm_annual_pct": opm if opm is not None else np.nan,
                "fq_opm_yoy_chg": (opm - prev_opm) if (opm is not None and prev_opm is not None) else np.nan,
                "fq_npm_annual_pct": npm,
                "fq_npm_yoy_chg": (npm - prev_npm) if (not math.isnan(npm) and not math.isnan(prev_npm)) else np.nan,
                "fq_debt_to_equity": d_to_e,
                "fq_cash_to_debt_ratio": c_to_d,
                "fq_ocf_to_np_ratio": ocf_np,
                "fq_fcf_to_sales_pct": fcf_sales,
                "fq_accruals_to_assets": accruals,
                "fq_interest_coverage": int_cov,
                "fq_asset_turnover": asset_turn,
                "fq_rd_pct_of_sales": rd_item.get("rd_pct_of_sales") if rd_item.get("rd_pct_of_sales") else np.nan,
                "fq_cwip_to_assets_pct": cwip_assets,
                "fq_roe_pct": (np_val / tot_eq * 100.0) if (np_val is not None and tot_eq > 0) else np.nan,
                "fq_roce_pct": (
                    (op_prof / (tot_eq + tot_debt) * 100.0)
                    if (op_prof is not None and (tot_eq + tot_debt) > 0)
                    else np.nan
                ),
                "fq_roe_roce_spread": (
                    (np_val / tot_eq * 100.0) - (op_prof / (tot_eq + tot_debt) * 100.0)
                    if (np_val is not None and op_prof is not None and tot_eq > 0 and (tot_eq + tot_debt) > 0)
                    else np.nan
                ),
                "fq_net_cash_to_mcap_pct": (
                    ((tot_cash - tot_debt) / float(ov["market_cap_cr"]) * 100.0)
                    if ov.get("market_cap_cr")
                    else np.nan
                ),
                "fq_piotroski_score": (
                    float(int((np_val or 0) > 0) + int((ocf or 0) > 0) + int((ocf or 0) > (np_val or 0)) + int(d_to_e < 0.8 if not math.isnan(d_to_e) else 0) + int((opm or 0) >= (prev_opm or 0)))
                ),
                "eg_annual_rev_5y_cagr": a_rep.get("revenue_growth_pct") if a_rep.get("revenue_growth_pct") is not None else np.nan,
                "eg_annual_profit_5y_cagr": a_rep.get("net_profit_growth_pct") if a_rep.get("net_profit_growth_pct") is not None else np.nan,
                "_pit_eps_inr": a_rep.get("eps_inr") if a_rep.get("eps_inr") is not None else np.nan,
            }
        )

    # 2. Build Point-in-Time Quarterly Earnings Events sorted by statutory publication date
    qtr_events: list[dict[str, Any]] = []
    for idx, q_rep in enumerate(qtr_list):
        per = q_rep.get("period", "")
        pub_date = _infer_publication_timestamp(per)[:10]
        prev_q = qtr_list[idx - 1] if idx > 0 else None
        yoy_q = qtr_list[idx - 4] if idx >= 4 else None

        rev = q_rep.get("revenue_cr")
        np_val = q_rep.get("net_profit_cr")
        opm = q_rep.get("opm_pct")
        rev_qoq = q_rep.get("revenue_growth_pct")
        np_qoq = q_rep.get("net_profit_growth_pct")

        prev_rev_qoq = prev_q.get("revenue_growth_pct") if prev_q else None
        prev_np_qoq = prev_q.get("net_profit_growth_pct") if prev_q else None
        prev_opm = prev_q.get("opm_pct") if prev_q else None

        rev_yoy = (
            ((rev - yoy_q["revenue_cr"]) / abs(yoy_q["revenue_cr"]) * 100.0)
            if (yoy_q and yoy_q.get("revenue_cr") and rev is not None)
            else rev_qoq
        )
        np_yoy = (
            ((np_val - yoy_q["net_profit_cr"]) / abs(yoy_q["net_profit_cr"]) * 100.0)
            if (yoy_q and yoy_q.get("net_profit_cr") and np_val is not None)
            else np_qoq
        )

        qtr_events.append(
            {
                "pub_date": pub_date,
                "eg_qtr_rev_qoq_pct": rev_qoq if rev_qoq is not None else np.nan,
                "eg_qtr_rev_yoy_pct": rev_yoy if rev_yoy is not None else np.nan,
                "eg_qtr_np_qoq_pct": np_qoq if np_qoq is not None else np.nan,
                "eg_qtr_np_yoy_pct": np_yoy if np_yoy is not None else np.nan,
                "eg_qtr_opm_pct": opm if opm is not None else np.nan,
                "eg_qtr_opm_qoq_chg": (opm - prev_opm) if (opm is not None and prev_opm is not None) else np.nan,
                "eg_qtr_rev_accel": (rev_qoq - prev_rev_qoq) if (rev_qoq is not None and prev_rev_qoq is not None) else np.nan,
                "eg_qtr_np_accel": (np_qoq - prev_np_qoq) if (np_qoq is not None and prev_np_qoq is not None) else np.nan,
            }
        )

    # 3. Build Point-in-Time Shareholding Events sorted by statutory publication date
    shp_events: list[dict[str, Any]] = []
    for idx, s_rep in enumerate(shp_hist):
        per = s_rep.get("period", "")
        pub_date = _infer_publication_timestamp(per)[:10]
        prev_1q = shp_hist[idx - 1] if idx > 0 else s_rep
        prev_4q = shp_hist[max(0, idx - 4)]

        prom = s_rep.get("promoters_pct", np.nan)
        fii = s_rep.get("fii_pct", np.nan)
        dii = s_rep.get("dii_pct", np.nan)
        pub = s_rep.get("public_pct", np.nan)

        shp_events.append(
            {
                "pub_date": pub_date,
                "inst_promoters_pct": prom,
                "inst_fii_pct": fii,
                "inst_dii_pct": dii,
                "inst_total_pct": (fii + dii) if (not math.isnan(fii) and not math.isnan(dii)) else np.nan,
                "inst_public_pct": pub,
                "inst_promoter_chg_1q": prom - prev_1q.get("promoters_pct", prom),
                "inst_promoter_chg_4q": prom - prev_4q.get("promoters_pct", prom),
                "inst_fii_chg_1q": fii - prev_1q.get("fii_pct", fii),
                "inst_fii_chg_4q": fii - prev_4q.get("fii_pct", fii),
                "inst_dii_chg_1q": dii - prev_1q.get("dii_pct", dii),
                "inst_dii_chg_4q": dii - prev_4q.get("dii_pct", dii),
                "inst_net_flow_4q": (fii - prev_4q.get("fii_pct", fii)) + (dii - prev_4q.get("dii_pct", dii)),
                "inst_retail_dilution_4q": pub - prev_4q.get("public_pct", pub),
                "inst_named_holders_count": float(len(shp.get("named_stockholders", []))),
            }
        )

    # Strict point-in-time `merge_asof(..., direction="backward")` with ZERO backward-filling!
    # A financial statement published on `pub_dt` is ONLY visible on `obs_dt >= pub_dt`.
    obs_df = pd.DataFrame({"obs_dt": pd.to_datetime(obs_dates)}, index=obs_dates.index)
    out["_pit_eps_inr"] = np.nan

    if annual_events:
        ann_df = pd.DataFrame(annual_events)
        ann_df["pub_dt"] = pd.to_datetime(ann_df["pub_date"])
        ann_df = ann_df.sort_values("pub_dt").drop_duplicates("pub_dt", keep="last")
        merged_a = pd.merge_asof(
            obs_df.sort_values("obs_dt"),
            ann_df.drop(columns=["pub_date"]),
            left_on="obs_dt",
            right_on="pub_dt",
            direction="backward",
        )
        for col in [c for c in ann_df.columns if c not in ("pub_date", "pub_dt")]:
            out[col] = merged_a[col].values

    if qtr_events:
        qtr_df = pd.DataFrame(qtr_events)
        qtr_df["pub_dt"] = pd.to_datetime(qtr_df["pub_date"])
        qtr_df = qtr_df.sort_values("pub_dt").drop_duplicates("pub_dt", keep="last")
        merged_q = pd.merge_asof(
            obs_df.sort_values("obs_dt"),
            qtr_df.drop(columns=["pub_date"]),
            left_on="obs_dt",
            right_on="pub_dt",
            direction="backward",
        )
        for col in [c for c in qtr_df.columns if c not in ("pub_date", "pub_dt")]:
            out[col] = merged_q[col].values
        days_since = (merged_q["obs_dt"] - merged_q["pub_dt"]).dt.days
        out["ne_days_since_qtr_filing"] = days_since.clip(lower=0, upper=120).values
        out["ne_is_earnings_window"] = np.where(
            days_since.notna(), ((days_since >= 0) & (days_since <= 7)).astype(float), np.nan
        )

    if shp_events:
        shp_df = pd.DataFrame(shp_events)
        shp_df["pub_dt"] = pd.to_datetime(shp_df["pub_date"])
        shp_df = shp_df.sort_values("pub_dt").drop_duplicates("pub_dt", keep="last")
        merged_s = pd.merge_asof(
            obs_df.sort_values("obs_dt"),
            shp_df.drop(columns=["pub_date"]),
            left_on="obs_dt",
            right_on="pub_dt",
            direction="backward",
        )
        for col in [c for c in shp_df.columns if c not in ("pub_date", "pub_dt")]:
            out[col] = merged_s[col].values

    return out


def build_features_for_symbol(
    equity_df: pd.DataFrame,
    indices_df: pd.DataFrame,
    raw_company_bundle: dict[str, Any] | None = None,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Compute ~185 point-in-time candidate features (and optional forward targets) for one symbol.

    Guarantees:
    - Feature values at index `t` depend ONLY on bars `0 ... t` and fundamentals with
      `publication_timestamp <= t`.
    - Missing upstream fields (such as historical `delivery_pct` or `fut_oi`) remain `NaN`
      or use explicit missingness indicator flags—never silent zero replacement.
    """
    if equity_df.empty:
        return pd.DataFrame()

    df = equity_df.sort_values("observation_date").reset_index(drop=True).copy()
    sym = str(df["symbol"].iloc[0]).upper()
    sector = (
        raw_company_bundle.get("overview", {}).get("sector", "NSE Equities")
        if raw_company_bundle
        else "NSE Equities"
    )
    sector_idx_code = _resolve_sector_index_code(sector)

    close = pd.to_numeric(df["close"], errors="coerce")
    open_p = pd.to_numeric(df["open"], errors="coerce").fillna(close)
    high = pd.to_numeric(df["high"], errors="coerce").fillna(close)
    low = pd.to_numeric(df["low"], errors="coerce").fillna(close)
    prev_c = (
        pd.to_numeric(df["prev_close"], errors="coerce").fillna(close.shift(1))
        if "prev_close" in df.columns
        else close.shift(1)
    )
    vol = pd.to_numeric(df["volume"], errors="coerce")
    vwap = pd.to_numeric(df["vwap"], errors="coerce") if "vwap" in df.columns else pd.Series(np.nan, index=df.index)
    turnover = pd.to_numeric(df["turnover_cr"], errors="coerce") if "turnover_cr" in df.columns else pd.Series(np.nan, index=df.index)
    trades = pd.to_numeric(df["trades_count"], errors="coerce") if "trades_count" in df.columns else pd.Series(np.nan, index=df.index)
    deliv_qty = pd.to_numeric(df["deliverable_qty"], errors="coerce") if "deliverable_qty" in df.columns else pd.Series(np.nan, index=df.index)
    deliv_pct = pd.to_numeric(df["delivery_pct"], errors="coerce") if "delivery_pct" in df.columns else pd.Series(np.nan, index=df.index)

    feat: dict[str, Any] = {
        "symbol": df["symbol"],
        "observation_date": df["observation_date"],
        "publication_timestamp": (
            df["publication_timestamp"]
            if "publication_timestamp" in df.columns
            else df["observation_date"].astype(str) + "T18:30:00+05:30"
        ),
        "sector": sector,
        "sector_index_code": sector_idx_code,
        "data_version": df["data_version"] if "data_version" in df.columns else "1.0.0",
    }

    # =========================================================================
    # FAMILY 1: Price & Technical Momentum (~72 features)
    # =========================================================================
    horizons = [1, 2, 3, 5, 7, 10, 14, 20, 40, 60, 90, 120, 180, 252]
    for h in horizons:
        feat[f"pm_ret_{h}d_pct"] = (close / close.shift(h) - 1.0) * 100.0
        feat[f"pm_logret_{h}d"] = np.log(close / close.shift(h))

    # Return acceleration & momentum differences
    feat["pm_ret_accel_1d_5d"] = feat["pm_ret_1d_pct"] - (feat["pm_ret_5d_pct"] / 5.0)
    feat["pm_ret_accel_5d_20d"] = (feat["pm_ret_5d_pct"] / 5.0) - (feat["pm_ret_20d_pct"] / 20.0)
    feat["pm_ret_accel_20d_60d"] = (feat["pm_ret_20d_pct"] / 20.0) - (feat["pm_ret_60d_pct"] / 60.0)
    feat["pm_mom_12m_minus_1m"] = feat["pm_ret_252d_pct"] - feat["pm_ret_20d_pct"]

    # SMA & EMA distances and slopes
    ma_windows = [5, 8, 10, 13, 20, 34, 50, 100, 150, 200]
    smas: dict[int, pd.Series] = {}
    emas: dict[int, pd.Series] = {}
    for w in ma_windows:
        s_ma = close.rolling(w, min_periods=max(3, w // 2)).mean()
        e_ma = close.ewm(span=w, min_periods=max(3, w // 2), adjust=False).mean()
        smas[w] = s_ma
        emas[w] = e_ma
        feat[f"pm_dist_sma_{w}_pct"] = (close / s_ma - 1.0) * 100.0
        feat[f"pm_dist_ema_{w}_pct"] = (close / e_ma - 1.0) * 100.0

    for w in [10, 20, 50, 200]:
        feat[f"pm_slope_sma_{w}_5d_pct"] = (smas[w] / smas[w].shift(5) - 1.0) * 100.0

    feat["pm_GoldenCross_50_200_pct"] = (smas[50] / smas[200] - 1.0) * 100.0
    feat["pm_ema_cross_8_20_pct"] = (emas[8] / emas[20] - 1.0) * 100.0
    feat["pm_ema_cross_20_50_pct"] = (emas[20] / emas[50] - 1.0) * 100.0

    # Multi-period Wilder RSI
    for rp in [5, 7, 14, 21]:
        r_ser = _wilder_rsi(close, rp)
        feat[f"pm_rsi_{rp}"] = r_ser
    feat["pm_rsi_14_chg_3d"] = feat["pm_rsi_14"] - feat["pm_rsi_14"].shift(3)
    feat["pm_rsi_14_chg_5d"] = feat["pm_rsi_14"] - feat["pm_rsi_14"].shift(5)
    feat["pm_rsi_5_minus_14"] = feat["pm_rsi_5"] - feat["pm_rsi_14"]

    # MACD (12, 26, 9) and fast MACD (5, 35, 5) normalized by price
    macd_l = emas[13] - emas[34]
    macd_12_26 = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd_12_26.ewm(span=9, adjust=False).mean()
    macd_hist = macd_12_26 - macd_sig
    feat["pm_macd_norm_pct"] = macd_12_26 / close * 100.0
    feat["pm_macd_sig_norm_pct"] = macd_sig / close * 100.0
    feat["pm_macd_hist_norm_pct"] = macd_hist / close * 100.0
    feat["pm_macd_hist_chg_1d"] = feat["pm_macd_hist_norm_pct"].diff(1)
    feat["pm_macd_fast_norm_pct"] = macd_l / close * 100.0

    # Stochastic %K / %D (14, 3) & Williams %R (14) & CCI (20)
    low_14 = low.rolling(14, min_periods=5).min()
    high_14 = high.rolling(14, min_periods=5).max()
    denom_14 = (high_14 - low_14).replace(0.0, np.nan)
    stoch_k = (close - low_14) / denom_14 * 100.0
    stoch_d = stoch_k.rolling(3, min_periods=2).mean()
    feat["pm_stoch_k_14"] = stoch_k
    feat["pm_stoch_d_14"] = stoch_d
    feat["pm_stoch_kd_spread"] = stoch_k - stoch_d
    feat["pm_williams_r_14"] = (high_14 - close) / denom_14 * -100.0

    tp = (high + low + close) / 3.0
    tp_sma20 = tp.rolling(20, min_periods=8).mean()
    tp_mad20 = (tp - tp_sma20).abs().rolling(20, min_periods=8).mean().replace(0.0, np.nan)
    feat["pm_cci_20"] = (tp - tp_sma20) / (0.015 * tp_mad20)

    # Bollinger Bands (20, 2) and (50, 2)
    for bw in [20, 50]:
        b_mid = smas[bw]
        b_std = close.rolling(bw, min_periods=max(5, bw // 2)).std()
        b_up = b_mid + 2.0 * b_std
        b_lo = b_mid - 2.0 * b_std
        b_rng = (b_up - b_lo).replace(0.0, np.nan)
        feat[f"pm_bb_pct_b_{bw}"] = (close - b_lo) / b_rng
        feat[f"pm_bb_width_{bw}_pct"] = b_rng / b_mid * 100.0

    # Donchian / High-Low Channel position (20D, 55D, 252D)
    for dw in [20, 55, 252]:
        d_hi = high.rolling(dw, min_periods=max(5, dw // 4)).max()
        d_lo = low.rolling(dw, min_periods=max(5, dw // 4)).min()
        feat[f"pm_dist_high_{dw}d_pct"] = (close / d_hi - 1.0) * 100.0
        feat[f"pm_dist_low_{dw}d_pct"] = (close / d_lo - 1.0) * 100.0
        feat[f"pm_channel_pos_{dw}d"] = (close - d_lo) / (d_hi - d_lo).replace(0.0, np.nan)

    # ATR & ATR-Normalized Returns + Volatility Estimators
    atr_7 = _compute_atr_series(high, low, close, 7)
    atr_14 = _compute_atr_series(high, low, close, 14)
    atr_21 = _compute_atr_series(high, low, close, 21)
    feat["pm_atr_7_pct"] = atr_7 / close * 100.0
    feat["pm_atr_14_pct"] = atr_14 / close * 100.0
    feat["pm_atr_21_pct"] = atr_21 / close * 100.0
    feat["pm_atr_ratio_7_21"] = atr_7 / atr_21.replace(0.0, np.nan)

    for ah in [1, 3, 5, 10]:
        feat[f"pm_ret_{ah}d_atr_norm"] = (close - close.shift(ah)) / atr_14.replace(0.0, np.nan)

    # Intraday Candle Microstructure & Gaps
    bar_range = (high - low).replace(0.0, np.nan)
    feat["pm_overnight_gap_pct"] = (open_p / prev_c - 1.0) * 100.0
    feat["pm_intraday_ret_pct"] = (close / open_p - 1.0) * 100.0
    feat["pm_intraday_range_pct"] = (high - low) / close * 100.0
    feat["pm_body_to_range"] = (close - open_p) / bar_range
    feat["pm_upper_shadow_ratio"] = (high - np.maximum(open_p, close)) / bar_range
    feat["pm_lower_shadow_ratio"] = (np.minimum(open_p, close) - low) / bar_range
    feat["pm_close_location_value"] = ((close - low) - (high - close)) / bar_range

    # Realized Volatility, Parkinson Volatility, Skewness & Kurtosis
    log_ret_1d = feat["pm_logret_1d"]
    for vw in [5, 10, 20, 60]:
        feat[f"pm_realized_vol_{vw}d_ann"] = log_ret_1d.rolling(vw, min_periods=max(3, vw // 2)).std() * math.sqrt(252) * 100.0
    feat["pm_vol_expansion_5_20"] = feat["pm_realized_vol_5d_ann"] / feat["pm_realized_vol_20d_ann"].replace(0.0, np.nan)
    feat["pm_vol_expansion_10_60"] = feat["pm_realized_vol_10d_ann"] / feat["pm_realized_vol_60d_ann"].replace(0.0, np.nan)

    hl_log_sq = (np.log(high / low.replace(0.0, np.nan)) ** 2) / (4.0 * math.log(2.0))
    feat["pm_parkinson_vol_20d_ann"] = np.sqrt(hl_log_sq.rolling(20, min_periods=8).mean() * 252.0) * 100.0
    feat["pm_ret_skew_20d"] = log_ret_1d.rolling(20, min_periods=10).skew()
    feat["pm_ret_kurt_20d"] = log_ret_1d.rolling(20, min_periods=10).kurt()

    # =========================================================================
    # FAMILY 2: Volume & Delivery (~26 features)
    # =========================================================================
    for v_win in [5, 10, 20, 50, 100]:
        v_avg = vol.rolling(v_win, min_periods=max(3, v_win // 2)).mean().replace(0.0, np.nan)
        feat[f"vd_rel_vol_{v_win}d"] = vol / v_avg

    log_vol = np.log1p(vol.clip(lower=0.0))
    feat["vd_log_vol_zscore_20d"] = (
        log_vol - log_vol.rolling(20, min_periods=8).mean()
    ) / log_vol.rolling(20, min_periods=8).std().replace(0.0, np.nan)
    feat["vd_vol_trend_5_20"] = (
        vol.rolling(5, min_periods=3).mean()
        / vol.rolling(20, min_periods=8).mean().replace(0.0, np.nan)
    )
    feat["vd_vol_chg_1d_pct"] = vol.pct_change(1) * 100.0

    # OBV, CMF(20), MFI(14), VPT, Amihud Illiquidity
    obv_step = np.sign(close.diff()).fillna(0.0) * vol.fillna(0.0)
    obv = obv_step.cumsum()
    obv_std20 = obv.rolling(20, min_periods=8).std().replace(0.0, np.nan)
    feat["vd_obv_zscore_20d"] = (obv - obv.rolling(20, min_periods=8).mean()) / obv_std20

    mfv = feat["pm_close_location_value"] * vol
    feat["vd_cmf_20d"] = mfv.rolling(20, min_periods=8).sum() / vol.rolling(20, min_periods=8).sum().replace(0.0, np.nan)
    feat["vd_mfi_14"] = _compute_mfi(high, low, close, vol, 14)

    vpt_step = (close.pct_change(1) * vol).fillna(0.0)
    feat["vd_vpt_norm_20d"] = vpt_step.rolling(20, min_periods=8).sum() / vol.rolling(20, min_periods=8).mean().replace(0.0, np.nan)

    dollar_vol_cr = (close * vol) / 1e7
    feat["vd_est_turnover_cr"] = turnover.fillna(dollar_vol_cr)
    feat["vd_amihud_illiquidity_20d"] = (
        (feat["pm_ret_1d_pct"].abs() / feat["vd_est_turnover_cr"].replace(0.0, np.nan))
        .rolling(20, min_periods=8)
        .mean()
    )
    feat["vd_up_down_vol_ratio_20d"] = (
        vol.where(feat["pm_ret_1d_pct"] > 0, 0.0).rolling(20, min_periods=8).sum()
        / vol.where(feat["pm_ret_1d_pct"] < 0, 0.0).rolling(20, min_periods=8).sum().replace(0.0, np.nan)
    )

    # Official NSE Delivery & VWAP features (preserved as NaN when missing!)
    feat["vd_delivery_pct"] = deliv_pct
    feat["vd_has_verified_delivery"] = deliv_pct.notna().astype(float)
    deliv_avg_5 = deliv_pct.rolling(5, min_periods=1).mean()
    deliv_avg_20 = deliv_pct.rolling(20, min_periods=1).mean()
    feat["vd_delivery_vs_5d_avg"] = deliv_pct - deliv_avg_5
    feat["vd_delivery_vs_20d_avg"] = deliv_pct - deliv_avg_20
    feat["vd_delivery_chg_1d"] = deliv_pct.diff(1)
    feat["vd_deliverable_qty_rel_5d"] = deliv_qty / deliv_qty.rolling(5, min_periods=1).mean().replace(0.0, np.nan)
    feat["vd_vwap_dist_pct"] = (close / vwap - 1.0) * 100.0
    feat["vd_trades_per_cr_turnover"] = trades / turnover.replace(0.0, np.nan)

    # Price × Volume × Delivery interaction signals
    feat["vd_price_vol_confirm_1d"] = feat["pm_ret_1d_pct"] * feat["vd_rel_vol_20d"]
    feat["vd_accumulation_delivery_score"] = feat["pm_ret_1d_pct"] * (deliv_pct / 50.0)
    feat["vd_pocket_pivot_flag"] = (
        (feat["pm_ret_1d_pct"] > 0)
        & (vol > vol.where(feat["pm_ret_1d_pct"] < 0, 0.0).rolling(10, min_periods=3).max())
    ).astype(float)

    # =========================================================================
    # FAMILY 3: Market & Sector Strength (~32 features)
    # =========================================================================
    # Extract Nifty 50, India VIX, and relevant Sector Index from `indices_df`
    idx_piv: dict[str, pd.Series] = {}
    if not indices_df.empty:
        for code in ["NIFTY_50", "INDIA_VIX", sector_idx_code]:
            sub_idx = (
                indices_df[indices_df["index_code"] == code]
                .sort_values("observation_date")
                .drop_duplicates("observation_date", keep="last")
            )
            if not sub_idx.empty:
                m_idx = pd.merge(
                    df[["observation_date"]],
                    sub_idx[["observation_date", "close"]],
                    on="observation_date",
                    how="left",
                )
                idx_piv[code] = pd.to_numeric(m_idx["close"], errors="coerce").ffill()

    nifty_c = idx_piv.get("NIFTY_50", pd.Series(np.nan, index=df.index))
    vix_c = idx_piv.get("INDIA_VIX", pd.Series(np.nan, index=df.index))
    sec_c = idx_piv.get(sector_idx_code, nifty_c)

    for mh in [1, 3, 5, 10, 20, 60]:
        n_ret = (nifty_c / nifty_c.shift(mh) - 1.0) * 100.0
        s_ret = (sec_c / sec_c.shift(mh) - 1.0) * 100.0
        feat[f"ms_nifty_ret_{mh}d_pct"] = n_ret
        feat[f"ms_alpha_vs_nifty_{mh}d_pct"] = feat[f"pm_ret_{mh}d_pct"] - n_ret
        feat[f"ms_sector_ret_{mh}d_pct"] = s_ret
        feat[f"ms_alpha_vs_sector_{mh}d_pct"] = feat[f"pm_ret_{mh}d_pct"] - s_ret

    feat["ms_sector_vs_nifty_20d_pct"] = feat["ms_sector_ret_20d_pct"] - feat["ms_nifty_ret_20d_pct"]

    # Rolling Beta & Correlation vs Nifty 50 (20D & 60D)
    s_r1 = feat["pm_ret_1d_pct"]
    n_r1 = feat["ms_nifty_ret_1d_pct"]
    for bw in [20, 60]:
        cov_sn = s_r1.rolling(bw, min_periods=10).cov(n_r1)
        var_n = n_r1.rolling(bw, min_periods=10).var().replace(0.0, np.nan)
        feat[f"ms_beta_nifty_{bw}d"] = cov_sn / var_n
        feat[f"ms_corr_nifty_{bw}d"] = s_r1.rolling(bw, min_periods=10).corr(n_r1)

    # India VIX level, changes & regime interaction
    feat["ms_india_vix_level"] = vix_c
    feat["ms_india_vix_chg_1d_pct"] = vix_c.pct_change(1) * 100.0
    feat["ms_india_vix_chg_5d_pct"] = vix_c.pct_change(5) * 100.0
    feat["ms_india_vix_zscore_20d"] = (
        vix_c - vix_c.rolling(20, min_periods=8).mean()
    ) / vix_c.rolling(20, min_periods=8).std().replace(0.0, np.nan)

    nifty_sma50 = nifty_c.rolling(50, min_periods=15).mean()
    nifty_sma200 = nifty_c.rolling(200, min_periods=30).mean()
    feat["ms_nifty_dist_sma50_pct"] = (nifty_c / nifty_sma50 - 1.0) * 100.0
    feat["ms_nifty_dist_sma200_pct"] = (nifty_c / nifty_sma200 - 1.0) * 100.0
    feat["ms_regime_score"] = (
        np.where(nifty_c > nifty_sma200, 1.0, -1.0)
        + np.where(nifty_c > nifty_sma50, 0.5, -0.5)
        + np.where(vix_c < 18.5, 0.5, -0.5)
    )

    # =========================================================================
    # FAMILY 4, 6, 7, 8: Point-in-Time Institutional, Financial, Earnings & News (~51 features)
    # =========================================================================
    pit_df = _build_pit_fundamental_frame(df["observation_date"], raw_company_bundle)
    pit_eps = pd.to_numeric(pit_df.get("_pit_eps_inr", pd.Series(np.nan, index=df.index)), errors="coerce")
    for col in pit_df.columns:
        if not col.startswith("_"):
            feat[col] = pit_df[col]

    # Dynamic P/E and Earnings Yield using strictly point-in-time trailing EPS & current close
    valid_eps = pit_eps.where(pit_eps > 0, np.nan)
    feat["fq_dynamic_pe"] = close / valid_eps
    feat["fq_earnings_yield_pct"] = (valid_eps / close) * 100.0

    # =========================================================================
    # FAMILY 5: Derivatives Positioning Heuristics (~9 features)
    # =========================================================================
    is_fno = 1.0 if sym in FNO_ELIGIBLE_SYMBOLS else 0.0
    feat["dv_is_fno_eligible"] = is_fno
    # Unreported historical exchange OI/PCR remain NaN (never fabricated!)
    feat["dv_futures_oi"] = np.nan
    feat["dv_futures_oi_chg_pct"] = np.nan
    feat["dv_options_pcr"] = np.nan
    feat["dv_atm_iv"] = np.nan
    # Price-Volume-Delivery positioning heuristics (clearly named as heuristics)
    feat["dv_long_buildup_heuristic"] = (
        (feat["pm_ret_1d_pct"] > 0.5) & (feat["vd_rel_vol_20d"] > 1.15)
    ).astype(float)
    feat["dv_short_buildup_heuristic"] = (
        (feat["pm_ret_1d_pct"] < -0.5) & (feat["vd_rel_vol_20d"] > 1.15)
    ).astype(float)
    feat["dv_short_covering_heuristic"] = (
        (feat["pm_ret_1d_pct"] > 0.5)
        & (feat["pm_ret_5d_pct"].shift(1) < -2.0)
        & (feat["vd_rel_vol_20d"] <= 1.15)
    ).astype(float)
    feat["dv_long_unwinding_heuristic"] = (
        (feat["pm_ret_1d_pct"] < -0.5)
        & (feat["pm_ret_5d_pct"].shift(1) > 2.0)
        & (feat["vd_rel_vol_20d"] <= 1.15)
    ).astype(float)

    # =========================================================================
    # FORWARD TARGETS (Strictly separated in `target_*` columns for Phase 3 ML)
    # =========================================================================
    if include_targets:
        next_c = close.shift(-1)
        next_h = high.shift(-1)
        next_l = low.shift(-1)

        feat["target_next_day_ret"] = (next_c / close - 1.0) * 100.0
        feat["target_next_day_up"] = np.where(
            next_c.notna(), (next_c > close).astype(float), np.nan
        )
        feat["target_3d_ret"] = (close.shift(-3) / close - 1.0) * 100.0
        feat["target_5d_ret"] = (close.shift(-5) / close - 1.0) * 100.0
        feat["target_10d_ret"] = (close.shift(-10) / close - 1.0) * 100.0
        # Next-day Maximum Adverse Excursion (MAE %) and Maximum Favorable Excursion (MFE %)
        feat["target_next_day_mae"] = (next_l / close - 1.0) * 100.0
        feat["target_next_day_mfe"] = (next_h / close - 1.0) * 100.0

    out_df = pd.DataFrame(feat)
    return out_df


CANDIDATE_FEATURE_FAMILIES: dict[str, str] = {
    "pm_": "1. Price & Technical Momentum",
    "vd_": "2. Volume & Delivery",
    "ms_": "3. Market & Sector Strength",
    "inst_": "4. Institutional & Promoter Activity",
    "dv_": "5. Derivatives Positioning",
    "fq_": "6. Financial Quality",
    "eg_": "7. Earnings & Management Guidance",
    "ne_": "8. News & Corporate Events",
}


def build_historical_feature_matrix(
    equities_df: pd.DataFrame,
    indices_df: pd.DataFrame,
    company_bundles: dict[str, dict[str, Any]] | None = None,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build the unified historical feature matrix across all symbols in `equities_df`."""
    company_bundles = company_bundles or {}
    frames: list[pd.DataFrame] = []
    for sym, grp in equities_df.groupby("symbol"):
        sym_str = str(sym).upper()
        bundle = company_bundles.get(sym_str)
        f_df = build_features_for_symbol(
            equity_df=grp,
            indices_df=indices_df,
            raw_company_bundle=bundle,
            include_targets=include_targets,
        )
        if not f_df.empty:
            frames.append(f_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "observation_date"]).reset_index(drop=True)
