"""NSE Bulk/Block Deals, Corporate Actions & Institutional Intelligence Collector (Phase 4 — Section 5C & 5D).

Collects and normalizes:
1. Official NSE Bulk Deals (`https://nsearchives.nseindia.com/content/equities/bulk.csv`)
2. Official NSE Block Deals (`https://nsearchives.nseindia.com/content/equities/block.csv`)
3. Historical Corporate Actions (Dividends & Stock Splits recorded on `equities_daily.parquet`)
4. Quarterly Promoter, FII, and DII ownership trajectories (`1Q` & `4Q` deltas)
5. Official NSE Bhavcopy Delivery % vs 20D average & F&O positioning heuristics.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import RAW_DATA_DIR
from ..features.feature_engine import FNO_ELIGIBLE_SYMBOLS
from ..models import (
    BulkBlockDealRecord,
    CorporateActionRecord,
    CorporateIntelligenceSummary,
)

NSE_BULK_CSV_URL = "https://nsearchives.nseindia.com/content/equities/bulk.csv"
NSE_BLOCK_CSV_URL = "https://nsearchives.nseindia.com/content/equities/block.csv"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
}


def _normalize_nse_date(raw_date: str) -> str:
    s = str(raw_date or "").strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def fetch_official_nse_bulk_and_block_deals(
    raw_dir: Path | None = None,
    timeout: int = 12,
) -> list[BulkBlockDealRecord]:
    """Download and parse official NSE `bulk.csv` and `block.csv` disclosures."""
    target_dir = (raw_dir or RAW_DATA_DIR) / "nse_bulk_block"
    target_dir.mkdir(parents=True, exist_ok=True)

    records: list[BulkBlockDealRecord] = []
    sources: list[tuple[str, str, Path]] = [
        ("BULK", NSE_BULK_CSV_URL, target_dir / "bulk_latest.csv"),
        ("BLOCK", NSE_BLOCK_CSV_URL, target_dir / "block_latest.csv"),
    ]

    for deal_type, url, cache_path in sources:
        csv_text = ""
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 200 and len(resp.text) > 40:
                csv_text = resp.text
                cache_path.write_text(csv_text, encoding="utf-8")
        except Exception:
            if cache_path.exists():
                csv_text = cache_path.read_text(encoding="utf-8")

        if not csv_text:
            continue

        try:
            df = pd.read_csv(io.StringIO(csv_text))
            df.columns = [str(c).strip() for c in df.columns]
            for _, row in df.iterrows():
                sym = str(row.get("Symbol", "")).strip().upper()
                if not sym or sym == "NAN":
                    continue
                sec_name = str(row.get("Security Name", sym)).strip()
                client_name = str(row.get("Client Name", "Institutional Participant")).strip()
                raw_side = str(row.get("Buy/Sell", "BUY")).strip().upper()
                side = "BUY" if raw_side.startswith("B") else "SELL"

                qty_raw = str(row.get("Quantity Traded", "0")).replace(",", "").strip()
                prc_raw = str(row.get("Trade Price / Wght. Avg. Price", "0")).replace(",", "").strip()
                try:
                    qty = int(float(qty_raw))
                    prc = float(prc_raw)
                except ValueError:
                    continue
                if qty <= 0 or prc <= 0:
                    continue

                val_cr = round((qty * prc) / 1e7, 4)
                dt_iso = _normalize_nse_date(str(row.get("Date", "")))

                records.append(
                    BulkBlockDealRecord(
                        deal_type=deal_type,  # type: ignore[arg-type]
                        deal_date=dt_iso,
                        symbol=sym,
                        security_name=sec_name,
                        client_name=client_name,
                        side=side,  # type: ignore[arg-type]
                        quantity=qty,
                        price_inr=round(prc, 2),
                        value_cr=val_cr,
                        source=f"NSE_{deal_type.capitalize()}_Archive",
                    )
                )
        except Exception:
            continue

    return records


def extract_corporate_actions_from_equities(
    equities_df: pd.DataFrame,
    symbol: str,
) -> list[CorporateActionRecord]:
    """Extract verified historical Dividends and Stock Splits for `symbol` from `equities_daily`."""
    sub = (
        equities_df[equities_df["symbol"] == symbol.upper()]
        .sort_values("observation_date", ascending=False)
        .copy()
    )
    if sub.empty:
        return []

    actions: list[CorporateActionRecord] = []
    if "dividends" in sub.columns:
        div_rows = sub[pd.to_numeric(sub["dividends"], errors="coerce").fillna(0.0) > 0.0]
        for _, r in div_rows.head(8).iterrows():
            amt = round(float(r["dividends"]), 2)
            dt = str(r["observation_date"])
            actions.append(
                CorporateActionRecord(
                    symbol=symbol.upper(),
                    action_type="DIVIDEND",
                    ex_date=dt,
                    amount_or_ratio=amt,
                    description=f"Cash Dividend of ₹{amt:.2f}/share (Ex-Date: {dt})",
                )
            )

    if "stock_splits" in sub.columns:
        split_rows = sub[pd.to_numeric(sub["stock_splits"], errors="coerce").fillna(0.0) > 0.0]
        for _, r in split_rows.head(4).iterrows():
            ratio = round(float(r["stock_splits"]), 2)
            dt = str(r["observation_date"])
            actions.append(
                CorporateActionRecord(
                    symbol=symbol.upper(),
                    action_type="SPLIT",
                    ex_date=dt,
                    amount_or_ratio=ratio,
                    description=f"Stock Split / Bonus adjustment ratio {ratio}:1 (Ex-Date: {dt})",
                )
            )

    actions.sort(key=lambda a: a.ex_date, reverse=True)
    return actions


def build_corporate_intelligence_for_symbol(
    symbol: str,
    equities_df: pd.DataFrame,
    raw_company_bundle: dict[str, Any] | None,
    all_bulk_block_deals: list[BulkBlockDealRecord],
) -> CorporateIntelligenceSummary:
    """Build the Phase 4 `CorporateIntelligenceSummary` for a single stock."""
    sym = symbol.upper()
    now_iso = datetime.now(timezone.utc).isoformat()

    shp = (raw_company_bundle or {}).get("stockholders", {})
    q_hist = shp.get("quarterly_history", [])

    prom_pct = None
    fii_pct = None
    dii_pct = None
    prom_chg_1q = None
    prom_chg_4q = None
    fii_chg_1q = None
    fii_chg_4q = None
    dii_chg_1q = None
    dii_chg_4q = None
    net_inst_4q = None

    if q_hist:
        latest_q = q_hist[-1]
        prev_1q = q_hist[-2] if len(q_hist) >= 2 else latest_q
        prev_4q = q_hist[max(0, len(q_hist) - 5)]

        prom_pct = latest_q.get("promoters_pct")
        fii_pct = latest_q.get("fii_pct")
        dii_pct = latest_q.get("dii_pct")

        if prom_pct is not None and prev_1q.get("promoters_pct") is not None:
            prom_chg_1q = round(float(prom_pct) - float(prev_1q["promoters_pct"]), 2)
        if prom_pct is not None and prev_4q.get("promoters_pct") is not None:
            prom_chg_4q = round(float(prom_pct) - float(prev_4q["promoters_pct"]), 2)

        if fii_pct is not None and prev_1q.get("fii_pct") is not None:
            fii_chg_1q = round(float(fii_pct) - float(prev_1q["fii_pct"]), 2)
        if fii_pct is not None and prev_4q.get("fii_pct") is not None:
            fii_chg_4q = round(float(fii_pct) - float(prev_4q["fii_pct"]), 2)

        if dii_pct is not None and prev_1q.get("dii_pct") is not None:
            dii_chg_1q = round(float(dii_pct) - float(prev_1q["dii_pct"]), 2)
        if dii_pct is not None and prev_4q.get("dii_pct") is not None:
            dii_chg_4q = round(float(dii_pct) - float(prev_4q["dii_pct"]), 2)

        if fii_chg_4q is not None and dii_chg_4q is not None:
            net_inst_4q = round(fii_chg_4q + dii_chg_4q, 2)

    # Delivery % and F&O positioning heuristics from recent `equities_daily` bars
    sub_eq = (
        equities_df[equities_df["symbol"] == sym]
        .sort_values("observation_date")
        .reset_index(drop=True)
    )
    latest_deliv = None
    deliv_vs_avg = None
    fno_elig = sym in FNO_ELIGIBLE_SYMBOLS
    pos_heuristic = "Cash / Neutral"

    if not sub_eq.empty:
        deliv_s = pd.to_numeric(sub_eq.get("delivery_pct"), errors="coerce").dropna()
        if not deliv_s.empty:
            latest_deliv = round(float(deliv_s.iloc[-1]), 2)
            avg_deliv = float(deliv_s.tail(20).mean())
            deliv_vs_avg = round(latest_deliv - avg_deliv, 2)

        close_s = pd.to_numeric(sub_eq["close"], errors="coerce")
        vol_s = pd.to_numeric(sub_eq["volume"], errors="coerce")
        if len(close_s) >= 20:
            ret_1d = float((close_s.iloc[-1] / close_s.iloc[-2] - 1.0) * 100.0)
            ret_5d = float((close_s.iloc[-1] / close_s.iloc[-6] - 1.0) * 100.0)
            rel_vol = float(vol_s.iloc[-1] / max(1.0, vol_s.tail(20).mean()))

            if fno_elig:
                if ret_1d > 0.45 and rel_vol > 1.10:
                    pos_heuristic = "Long Buildup Heuristic (Price Up + Vol Surge)"
                elif ret_1d < -0.45 and rel_vol > 1.10:
                    pos_heuristic = "Short Buildup Heuristic (Price Down + Vol Surge)"
                elif ret_1d > 0.35 and ret_5d < -1.5:
                    pos_heuristic = "Short Covering Bounce Heuristic"
                elif ret_1d < -0.35 and ret_5d > 1.5:
                    pos_heuristic = "Long Unwinding / Profit Booking Heuristic"
                elif latest_deliv and latest_deliv >= 55.0:
                    pos_heuristic = f"High Cash Delivery Accumulation ({latest_deliv:.1f}%)"
                else:
                    pos_heuristic = "Balanced F&O / Cash Activity"
            else:
                if latest_deliv and latest_deliv >= 55.0 and ret_1d >= 0.0:
                    pos_heuristic = f"Cash Delivery Accumulation ({latest_deliv:.1f}%)"
                elif ret_1d < -0.75 and rel_vol > 1.15:
                    pos_heuristic = "Cash Distribution / Supply Pressure"
                else:
                    pos_heuristic = "Standard Cash Segment Trading"

    # Match symbol deals from official NSE Bulk/Block archives
    sym_deals = [d for d in all_bulk_block_deals if d.symbol == sym]
    buy_val = sum(d.value_cr for d in sym_deals if d.side == "BUY")
    sell_val = sum(d.value_cr for d in sym_deals if d.side == "SELL")
    net_deal_cr = round(buy_val - sell_val, 4)

    corp_actions = extract_corporate_actions_from_equities(equities_df, sym)

    # Compute Institutional Conviction Score (0 - 100) and key signals
    score = 50.0
    signals: list[str] = []

    if net_inst_4q is not None:
        if net_inst_4q >= 1.0:
            score += min(18.0, net_inst_4q * 4.5)
            signals.append(f"FII + DII net ownership increased by +{net_inst_4q:.2f}% over last 4 quarters.")
        elif net_inst_4q <= -1.0:
            score -= min(18.0, abs(net_inst_4q) * 4.5)
            signals.append(f"FII + DII net ownership decreased by {net_inst_4q:.2f}% over last 4 quarters.")

    if prom_chg_4q is not None:
        if prom_chg_4q > 0.05:
            score += 10.0
            signals.append(f"Promoters increased stake by +{prom_chg_4q:.2f}% YoY (Current: {prom_pct:.2f}%).")
        elif prom_chg_4q < -0.50:
            score -= 10.0
            signals.append(f"Promoter stake diluted by {prom_chg_4q:.2f}% YoY (Current: {prom_pct:.2f}%).")
        elif prom_pct is not None and prom_pct >= 50.0:
            score += 5.0
            signals.append(f"Stable majority promoter holding at {prom_pct:.2f}%.")

    if latest_deliv is not None:
        if latest_deliv >= 58.0:
            score += 9.0
            signals.append(f"High verified NSE cash delivery at {latest_deliv:.1f}% (vs avg {deliv_vs_avg:+.1f}%).")
        elif latest_deliv <= 35.0:
            score -= 5.0
            signals.append(f"Low NSE cash delivery at {latest_deliv:.1f}% (intraday speculative dominance).")

    if sym_deals:
        if net_deal_cr > 0:
            score += 8.0
            signals.append(f"Net NSE Bulk/Block buying of ₹{net_deal_cr:.2f} Cr across {len(sym_deals)} deal(s).")
        elif net_deal_cr < 0:
            score -= 8.0
            signals.append(f"Net NSE Bulk/Block selling of ₹{abs(net_deal_cr):.2f} Cr across {len(sym_deals)} deal(s).")

    if corp_actions:
        latest_ca = corp_actions[0]
        signals.append(f"Recent Corporate Action: {latest_ca.description}.")

    score = round(float(np.clip(score, 15.0, 95.0)), 1)
    if score >= 68.0:
        verdict = "STRONG ACCUMULATION"
    elif score >= 55.0:
        verdict = "MODERATE ACCUMULATION"
    elif score <= 42.0:
        verdict = "DISTRIBUTION"
    else:
        verdict = "NEUTRAL"

    return CorporateIntelligenceSummary(
        symbol=sym,
        evaluated_at=now_iso,
        promoter_holding_pct=prom_pct,
        promoter_chg_1q=prom_chg_1q,
        promoter_chg_4q=prom_chg_4q,
        promoter_pledge_pct=None,  # Preserved as None unless verified exchange pledge filing is present
        fii_holding_pct=fii_pct,
        fii_chg_1q=fii_chg_1q,
        fii_chg_4q=fii_chg_4q,
        dii_holding_pct=dii_pct,
        dii_chg_1q=dii_chg_1q,
        dii_chg_4q=dii_chg_4q,
        net_institutional_flow_4q=net_inst_4q,
        is_fno_eligible=fno_elig,
        latest_delivery_pct=latest_deliv,
        delivery_vs_20d_avg=deliv_vs_avg,
        fno_positioning_heuristic=pos_heuristic,
        net_bulk_block_value_cr=net_deal_cr,
        bulk_block_deals_count=len(sym_deals),
        institutional_conviction_score=score,
        smart_money_verdict=verdict,  # type: ignore[arg-type]
        bulk_block_deals=sym_deals,
        recent_corporate_actions=corp_actions,
        key_signals=signals,
    )
