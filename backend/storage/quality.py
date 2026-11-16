"""Data Quality & Provenance Validation Engine (Phase 1 — Section 6).

Enforces:
- Explicit provenance tracking (`symbol`, `metric`, `value`, `source`, `observation_date`,
  `publication_timestamp`, `collection_timestamp`, `validation_status`, `data_version`).
- Strict distinction between `VERIFIED`, `ESTIMATED`, `MISSING`, `STALE`, and `CONFLICTING`.
- Never silently replaces missing values with zero.
- Produces a `DataQualityReport` (0–100 score) and forces `NO TRADE` when critical inputs
  are missing, stale, or conflicting.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from ..config import DATA_SCHEMA_VERSION, IST_TZ
from ..models import (
    CashAndBalanceSheet,
    CompanyOverview,
    DataQualityReport,
    DataRecord,
    RDAndInnovation,
    ReportPeriod,
    StockholderAnalysis,
    TechnicalSnapshot,
    ValidationStatus,
)


def _infer_publication_timestamp(period_label: str) -> str:
    """Estimate conservative statutory filing publication date for Indian financial periods.

    Under SEBI LODR Regulation 33:
    - Annual results (e.g. 'Mar 2025') must be filed within 60 days (by May 30).
    - Quarterly results (e.g. 'Dec 2025') must be filed within 45 days (by Feb 14).
    """
    p = (period_label or "").strip()
    parts = p.split()
    if len(parts) == 2 and parts[1].isdigit():
        mon, yr = parts[0][:3].title(), int(parts[1])
        if mon == "Mar":
            return f"{yr}-05-30T18:00:00+05:30"
        if mon == "Jun":
            return f"{yr}-08-14T18:00:00+05:30"
        if mon == "Sep":
            return f"{yr}-11-14T18:00:00+05:30"
        if mon == "Dec":
            return f"{yr + 1}-02-14T18:00:00+05:30"
    return datetime.now(IST_TZ).isoformat(timespec="seconds")


def build_provenance_records(
    symbol: str,
    overview: CompanyOverview,
    annual_5: list[ReportPeriod],
    quarterly_5: list[ReportPeriod],
    cash_bs: CashAndBalanceSheet,
    shp: StockholderAnalysis,
    rd: RDAndInnovation,
    tech: TechnicalSnapshot,
    price_rows: list[dict[str, Any]],
) -> list[DataRecord]:
    """Convert a stock's fundamental and technical snapshot into canonical DataRecord items."""
    sym = symbol.upper().strip()
    now_iso = datetime.now(IST_TZ).isoformat(timespec="seconds")
    latest_bar_date = (
        str(price_rows[-1].get("date", ""))[:10]
        if price_rows
        else datetime.now(IST_TZ).strftime("%Y-%m-%d")
    )

    records: list[DataRecord] = []

    def _add(
        metric: str,
        val: float | str | None,
        source: str,
        obs_date: str,
        pub_ts: str,
        status: ValidationStatus,
    ) -> None:
        records.append(
            DataRecord(
                symbol=sym,
                metric=metric,
                value=val,
                source=source,
                observation_date=obs_date,
                publication_timestamp=pub_ts,
                collection_timestamp=now_iso,
                validation_status=status,
                data_version=DATA_SCHEMA_VERSION,
            )
        )

    # 1. Price & Volume Provenance
    has_ohlc = bool(
        price_rows
        and price_rows[-1].get("high") is not None
        and price_rows[-1].get("low") is not None
        and price_rows[-1].get("high") != price_rows[-1].get("low")
    )
    price_status: ValidationStatus = (
        "VERIFIED" if (tech.current_price and tech.current_price > 0 and len(price_rows) >= 30) else "MISSING"
    )
    if price_status == "VERIFIED" and latest_bar_date:
        try:
            bar_dt = date.fromisoformat(latest_bar_date)
            age_days = (datetime.now(IST_TZ).date() - bar_dt).days
            if age_days > 10:
                price_status = "STALE"
        except Exception:
            pass

    _add(
        "current_price",
        tech.current_price if tech.current_price > 0 else None,
        "NSE/YahooFinance/Screener",
        latest_bar_date,
        f"{latest_bar_date}T15:30:00+05:30",
        price_status,
    )
    _add(
        "intraday_high_low_range",
        round(tech.day_high - tech.day_low, 2) if has_ohlc else None,
        "NSE_Bhavcopy/YahooFinance",
        latest_bar_date,
        f"{latest_bar_date}T15:30:00+05:30",
        "VERIFIED" if has_ohlc else "MISSING",
    )
    _add(
        "volume_latest",
        tech.volume_latest if tech.volume_latest > 0 else None,
        "NSE_Bhavcopy/YahooFinance",
        latest_bar_date,
        f"{latest_bar_date}T15:30:00+05:30",
        "VERIFIED" if tech.volume_latest > 0 else "MISSING",
    )

    latest_deliv = price_rows[-1].get("delivery_pct") if price_rows else None
    _add(
        "delivery_pct",
        latest_deliv,
        "NSE_SecBhavdata_Full",
        latest_bar_date,
        f"{latest_bar_date}T18:00:00+05:30",
        "VERIFIED" if latest_deliv is not None else "MISSING",
    )

    # 2. Annual & Quarterly Reports Provenance
    latest_fy = annual_5[-1] if annual_5 else None
    fy_period = latest_fy.period if latest_fy else "Unknown_FY"
    fy_pub = _infer_publication_timestamp(fy_period)
    _add(
        "annual_revenue_cr",
        latest_fy.revenue_cr if latest_fy else None,
        "Screener_Consolidated_PL",
        fy_period,
        fy_pub,
        "VERIFIED" if (latest_fy and latest_fy.revenue_cr is not None) else "MISSING",
    )
    _add(
        "annual_net_profit_cr",
        latest_fy.net_profit_cr if latest_fy else None,
        "Screener_Consolidated_PL",
        fy_period,
        fy_pub,
        "VERIFIED" if (latest_fy and latest_fy.net_profit_cr is not None) else "MISSING",
    )

    latest_q = quarterly_5[-1] if quarterly_5 else None
    q_period = latest_q.period if latest_q else "Unknown_Qtr"
    q_pub = _infer_publication_timestamp(q_period)
    _add(
        "quarterly_revenue_cr",
        latest_q.revenue_cr if latest_q else None,
        "Screener_Quarterly_PL",
        q_period,
        q_pub,
        "VERIFIED" if (latest_q and latest_q.revenue_cr is not None) else "MISSING",
    )
    _add(
        "quarterly_net_profit_cr",
        latest_q.net_profit_cr if latest_q else None,
        "Screener_Quarterly_PL",
        q_period,
        q_pub,
        "VERIFIED" if (latest_q and latest_q.net_profit_cr is not None) else "MISSING",
    )

    # 3. Balance Sheet & Cash Provenance
    bs_pub = _infer_publication_timestamp(cash_bs.latest_period or fy_period)
    _add(
        "total_cash_in_hand_cr",
        cash_bs.total_cash_in_hand_cr if cash_bs.history_5y else None,
        "Screener_BalanceSheet_Schedule",
        cash_bs.latest_period or fy_period,
        bs_pub,
        "VERIFIED" if cash_bs.history_5y else "MISSING",
    )
    _add(
        "operating_cash_flow_cr",
        cash_bs.latest_ocf_cr if cash_bs.history_5y else None,
        "Screener_CashFlow",
        cash_bs.latest_period or fy_period,
        bs_pub,
        "VERIFIED" if (cash_bs.history_5y and cash_bs. history_5y[-1].operating_cash_flow_cr is not None) else "MISSING",
    )

    # 4. Shareholding Provenance
    shp_sum = round(
        shp.promoters_pct + shp.fii_pct + shp.dii_pct + shp.government_pct + shp.public_pct, 1
    )
    shp_status: ValidationStatus = "MISSING"
    if shp.quarterly_history:
        shp_status = "VERIFIED" if 98.0 <= shp_sum <= 102.0 else "CONFLICTING"
    _add(
        "shareholding_total_pct",
        shp_sum if shp.quarterly_history else None,
        "BSE_NSE_Shareholding_Pattern",
        shp.latest_period or q_period,
        _infer_publication_timestamp(shp.latest_period or q_period),
        shp_status,
    )

    # 5. R&D Provenance
    rd_status: ValidationStatus = "MISSING"
    if "Yahoo Finance IFRS" in rd.rd_data_source:
        rd_status = "VERIFIED"
    elif rd.latest_rd_spend_cr > 0 and "MISSING" not in rd.rd_data_source:
        rd_status = "ESTIMATED"
    _add(
        "rd_spend_cr",
        rd.latest_rd_spend_cr if rd_status != "MISSING" else None,
        rd.rd_data_source or "Unreported",
        fy_period,
        fy_pub,
        rd_status,
    )

    return records


def evaluate_stock_data_quality(
    symbol: str,
    overview: CompanyOverview,
    annual_5: list[ReportPeriod],
    quarterly_5: list[ReportPeriod],
    cash_bs: CashAndBalanceSheet,
    shp: StockholderAnalysis,
    rd: RDAndInnovation,
    tech: TechnicalSnapshot,
    price_rows: list[dict[str, Any]],
) -> tuple[DataQualityReport, list[DataRecord]]:
    """Compute 0–100 Data Quality Score and enforce the NO TRADE gate when critical inputs fail."""
    records = build_provenance_records(
        symbol=symbol,
        overview=overview,
        annual_5=annual_5,
        quarterly_5=quarterly_5,
        cash_bs=cash_bs,
        shp=shp,
        rd=rd,
        tech=tech,
        price_rows=price_rows,
    )

    counts: dict[str, int] = {
        "VERIFIED": 0,
        "ESTIMATED": 0,
        "MISSING": 0,
        "STALE": 0,
        "CONFLICTING": 0,
    }
    metric_statuses: dict[str, ValidationStatus] = {}
    for rec in records:
        counts[rec.validation_status] = counts.get(rec.validation_status, 0) + 1
        metric_statuses[rec.metric] = rec.validation_status

    critical_failures: list[str] = []
    if tech.current_price <= 0 or metric_statuses.get("current_price") in ("MISSING", "STALE"):
        critical_failures.append("Current market price is missing or stale (>10 days old).")
    if len(price_rows) < 30:
        critical_failures.append(f"Insufficient historical price bars ({len(price_rows)} < 30 required).")
    if len(annual_5) < 3 or metric_statuses.get("annual_revenue_cr") == "MISSING":
        critical_failures.append("Missing consolidated/standalone annual financial statements.")
    if len(quarterly_5) < 3 or metric_statuses.get("quarterly_revenue_cr") == "MISSING":
        critical_failures.append("Missing quarterly earnings statements.")
    if not shp.quarterly_history or metric_statuses.get("shareholding_total_pct") in ("MISSING", "CONFLICTING"):
        critical_failures.append("Missing or conflicting shareholding pattern disclosure.")

    # Weighted quality score out of 100
    total_metrics = max(1, len(records))
    raw_points = (
        counts["VERIFIED"] * 1.0
        + counts["ESTIMATED"] * 0.75
        + counts["STALE"] * 0.25
        + counts["MISSING"] * 0.0
        + counts["CONFLICTING"] * 0.0
    )
    quality_score = round((raw_points / total_metrics) * 100.0, 1)
    if critical_failures:
        quality_score = min(quality_score, 45.0)

    trade_eligible = len(critical_failures) == 0 and quality_score >= 60.0
    forced_action = None if trade_eligible else "NO TRADE"

    report = DataQualityReport(
        symbol=symbol.upper().strip(),
        evaluated_at=datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S IST"),
        quality_score=quality_score,
        trade_eligible=trade_eligible,
        forced_action=forced_action,  # type: ignore[arg-type]
        verified_metrics_count=counts["VERIFIED"],
        estimated_metrics_count=counts["ESTIMATED"],
        missing_metrics_count=counts["MISSING"],
        stale_metrics_count=counts["STALE"],
        conflicting_metrics_count=counts["CONFLICTING"],
        critical_failures=critical_failures,
        metric_statuses=metric_statuses,
    )
    return report, records
