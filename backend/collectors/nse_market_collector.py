"""Official NSE Archive & Supplementary Price/Delivery Collector (Phase 1 — Section 5A & 5B).

Collects:
1. Official NSE Security-wise Price, Volume & Delivery Bhavcopy (`sec_bhavdata_full_{DDMMYYYY}.csv`)
   from `nsearchives.nseindia.com` — providing exchange-verified OHLC, VWAP (`AVG_PRICE`),
   Volume (`TTL_TRD_QNTY`), Turnover (`TURNOVER_LACS`), Trades (`NO_OF_TRADES`),
   Deliverable Quantity (`DELIV_QTY`), and Delivery Percentage (`DELIV_PER`).
2. Official NSE Daily Index Archive (`ind_close_all_{DDMMYYYY}.csv`) with Index OHLC,
   Volume, Turnover, P/E, P/B, and Dividend Yield.
3. Official NSE Bulk & Block Deals CSVs (`bulk.csv`, `block.csv`).
4. Multi-year historical OHLCV + Corporate Actions (splits/dividends) backfill with
   strict provenance tagging (`delivery_pct=None` and `delivery_validation_status="MISSING"`
   when official exchange delivery report is unavailable for older dates — never silent zero).
"""

from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from ..config import (
    BROWSER_USER_AGENT,
    DATA_SCHEMA_VERSION,
    IST_TZ,
    NSE_MARKET_INDICES,
    RAW_DATA_DIR,
    REQUEST_TIMEOUT,
    is_nse_trading_day,
)
from ..models import EquityDailyBar, IndexDailyBar
from ..storage.duckdb_store import ensure_storage_directories


def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": BROWSER_USER_AGENT,
            "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )
    return s


def _parse_optional_float(val: Any) -> float | None:
    """Parse numeric string or return None (never silently replace missing values with 0.0)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return None if math.isnan(float(val)) else float(val)
    s = str(val).strip().replace(",", "")
    if not s or s in ("-", "--", "NA", "N/A", "null", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_nse_date(date_str: str) -> str:
    """Normalize NSE date strings like '13-Nov-2026' or '13-11-2026' to ISO 'YYYY-MM-DD'."""
    s = (date_str or "").strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def recent_nse_trading_dates(max_days: int = 7, reference_date: date | None = None) -> list[date]:
    """Return the most recent `max_days` candidate NSE trading dates up to `reference_date`."""
    cur = reference_date or datetime.now(IST_TZ).date()
    out: list[date] = []
    steps = 0
    while len(out) < max_days and steps < 30:
        if is_nse_trading_day(cur):
            out.append(cur)
        cur -= timedelta(days=1)
        steps += 1
    return out


def fetch_nse_sec_bhavdata_full(
    trade_date: date | None = None,
    symbols_filter: set[str] | None = None,
) -> tuple[list[EquityDailyBar], dict[str, Any]]:
    """Fetch official NSE `sec_bhavdata_full_{DDMMYYYY}.csv` and compute exchange breadth.

    If `trade_date` is None, automatically probes the most recent valid NSE trading days
    until it finds the latest published Bhavcopy.
    """
    ensure_storage_directories()
    session = _nse_session()
    candidates = [trade_date] if trade_date else recent_nse_trading_dates(max_days=7)

    csv_text = ""
    resolved_date: date | None = None

    for dt in candidates:
        if dt is None:
            continue
        ddmmyyyy = dt.strftime("%d%m%Y")
        raw_path = RAW_DATA_DIR / f"sec_bhavdata_full_{ddmmyyyy}.csv"
        if raw_path.exists() and raw_path.stat().st_size > 10_000:
            csv_text = raw_path.read_text(encoding="utf-8", errors="replace")
            resolved_date = dt
            break

        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and "SYMBOL" in r.text[:200] and len(r.text) > 10_000:
                csv_text = r.text
                raw_path.write_text(csv_text, encoding="utf-8")
                resolved_date = dt
                break
        except Exception:
            continue

    if not csv_text or resolved_date is None:
        return [], {"advance_count": 0, "decline_count": 0, "unchanged_count": 0, "advance_decline_ratio": None}

    now_iso = datetime.now(IST_TZ).isoformat(timespec="seconds")
    reader = csv.DictReader(io.StringIO(csv_text))

    bars: list[EquityDailyBar] = []
    adv = 0
    dec = 0
    unch = 0

    for raw_row in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() if k}
        series = row.get("SERIES", "")
        if series not in ("EQ", "BE"):
            continue
        sym = row.get("SYMBOL", "").upper()
        if not sym:
            continue

        prev_c = _parse_optional_float(row.get("PREV_CLOSE"))
        open_p = _parse_optional_float(row.get("OPEN_PRICE"))
        high_p = _parse_optional_float(row.get("HIGH_PRICE"))
        low_p = _parse_optional_float(row.get("LOW_PRICE"))
        last_p = _parse_optional_float(row.get("LAST_PRICE"))
        close_p = _parse_optional_float(row.get("CLOSE_PRICE"))
        vwap_p = _parse_optional_float(row.get("AVG_PRICE"))
        vol = _parse_optional_float(row.get("TTL_TRD_QNTY"))
        turnover_lacs = _parse_optional_float(row.get("TURNOVER_LACS"))
        trades_f = _parse_optional_float(row.get("NO_OF_TRADES"))
        deliv_qty = _parse_optional_float(row.get("DELIV_QTY"))
        deliv_pct = _parse_optional_float(row.get("DELIV_PER"))

        # Track full NSE EQ market breadth across all traded equities
        if series == "EQ" and close_p is not None and prev_c is not None and prev_c > 0:
            if close_p > prev_c:
                adv += 1
            elif close_p < prev_c:
                dec += 1
            else:
                unch += 1

        if symbols_filter and sym not in symbols_filter:
            continue

        obs_date = _parse_nse_date(row.get("DATE1", resolved_date.strftime("%Y-%m-%d")))
        pub_ts = f"{obs_date}T18:00:00+05:30"
        turnover_cr = round(turnover_lacs / 100.0, 4) if turnover_lacs is not None else None

        bars.append(
            EquityDailyBar(
                symbol=sym,
                series=series,
                observation_date=obs_date,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                last_price=last_p,
                prev_close=prev_c,
                vwap=vwap_p,
                adj_close=close_p,
                volume=vol,
                turnover_cr=turnover_cr,
                trades_count=int(trades_f) if trades_f is not None else None,
                deliverable_qty=deliv_qty,
                delivery_pct=deliv_pct,
                source="NSE_SecBhavdata_Full",
                publication_timestamp=pub_ts,
                collection_timestamp=now_iso,
                ohlcv_validation_status="VERIFIED" if close_p is not None else "MISSING",
                delivery_validation_status="VERIFIED" if deliv_pct is not None else "MISSING",
                data_version=DATA_SCHEMA_VERSION,
            )
        )

    ad_ratio = round(adv / dec, 3) if dec > 0 else (float(adv) if adv > 0 else None)
    breadth = {
        "observation_date": resolved_date.strftime("%Y-%m-%d"),
        "advance_count": adv,
        "decline_count": dec,
        "unchanged_count": unch,
        "advance_decline_ratio": ad_ratio,
    }
    return bars, breadth


def fetch_nse_indices_daily_archive(trade_date: date | None = None) -> list[IndexDailyBar]:
    """Fetch official NSE `ind_close_all_{DDMMYYYY}.csv` for all broad, volatility & sector indices."""
    ensure_storage_directories()
    session = _nse_session()
    candidates = [trade_date] if trade_date else recent_nse_trading_dates(max_days=7)

    csv_text = ""
    resolved_date: date | None = None

    for dt in candidates:
        if dt is None:
            continue
        ddmmyyyy = dt.strftime("%d%m%Y")
        raw_path = RAW_DATA_DIR / f"ind_close_all_{ddmmyyyy}.csv"
        if raw_path.exists() and raw_path.stat().st_size > 2_000:
            csv_text = raw_path.read_text(encoding="utf-8", errors="replace")
            resolved_date = dt
            break

        url = f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and "Index Name" in r.text[:200]:
                csv_text = r.text
                raw_path.write_text(csv_text, encoding="utf-8")
                resolved_date = dt
                break
        except Exception:
            continue

    if not csv_text or resolved_date is None:
        return []

    # Map NSE Index Name (lowercased) to our canonical index_code
    name_to_code = {
        meta["nse_name"].lower(): (code, meta)
        for code, meta in NSE_MARKET_INDICES.items()
    }

    now_iso = datetime.now(IST_TZ).isoformat(timespec="seconds")
    reader = csv.DictReader(io.StringIO(csv_text))
    out: list[IndexDailyBar] = []

    for raw_row in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() if k}
        idx_name = row.get("Index Name", "")
        if not idx_name:
            continue
        match = name_to_code.get(idx_name.lower())
        if not match:
            continue

        code, meta = match
        close_val = _parse_optional_float(row.get("Closing Index Value"))
        if close_val is None:
            continue

        obs_date = _parse_nse_date(row.get("Index Date", resolved_date.strftime("%Y-%m-%d")))
        out.append(
            IndexDailyBar(
                index_code=code,
                index_name=meta["nse_name"],
                category=meta["category"],  # type: ignore[arg-type]
                observation_date=obs_date,
                open=_parse_optional_float(row.get("Open Index Value")),
                high=_parse_optional_float(row.get("High Index Value")),
                low=_parse_optional_float(row.get("Low Index Value")),
                close=close_val,
                change_pct=_parse_optional_float(row.get("Change(%)")),
                volume=_parse_optional_float(row.get("Volume")),
                turnover_cr=_parse_optional_float(row.get("Turnover (Rs. Cr.)")),
                pe_ratio=_parse_optional_float(row.get("P/E")),
                pb_ratio=_parse_optional_float(row.get("P/B")),
                div_yield=_parse_optional_float(row.get("Div Yield")),
                source="NSE_IndCloseAll_Archive",
                publication_timestamp=f"{obs_date}T18:00:00+05:30",
                collection_timestamp=now_iso,
                validation_status="VERIFIED",
                data_version=DATA_SCHEMA_VERSION,
            )
        )
    return out


def fetch_nse_bulk_and_block_deals() -> list[dict[str, Any]]:
    """Fetch latest official NSE Bulk and Block Deal disclosures from `nsearchives.nseindia.com`."""
    ensure_storage_directories()
    session = _nse_session()
    deals: list[dict[str, Any]] = []
    for deal_type, endpoint in [
        ("BULK", "https://nsearchives.nseindia.com/content/equities/bulk.csv"),
        ("BLOCK", "https://nsearchives.nseindia.com/content/equities/block.csv"),
    ]:
        try:
            r = session.get(endpoint, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200 or "Symbol" not in r.text[:200]:
                continue
            reader = csv.DictReader(io.StringIO(r.text))
            for raw_row in reader:
                row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() if k}
                sym = row.get("Symbol", "").upper()
                if not sym:
                    continue
                deals.append(
                    {
                        "deal_type": deal_type,
                        "observation_date": _parse_nse_date(row.get("Date", "")),
                        "symbol": sym,
                        "security_name": row.get("Security Name", ""),
                        "client_name": row.get("Client Name", ""),
                        "side": row.get("Buy/Sell", "").upper(),
                        "quantity": _parse_optional_float(row.get("Quantity Traded")),
                        "price": _parse_optional_float(row.get("Trade Price / Wght. Avg. Price")),
                        "source": f"NSE_{deal_type}_Archive",
                    }
                )
        except Exception:
            continue
    return deals


def backfill_equity_history_with_delivery(
    symbols: list[str],
    period: str = "2y",
    recent_bhavcopy_days: int = 5,
) -> tuple[list[EquityDailyBar], dict[str, Any]]:
    """Backfill historical daily OHLCV + corporate actions and overlay official NSE Delivery reports.

    Follows Section 5A & Section 6 rules:
    - Yahoo Finance is used only as a supplementary historical OHLCV + split/dividend source
      and explicitly tagged `source="YahooFinance_Supplementary"`, `delivery_pct=None`,
      `delivery_validation_status="MISSING"` (never silently zero!).
    - Recent trading sessions are overlaid from official NSE `sec_bhavdata_full_{DDMMYYYY}.csv`
      with `source="NSE_SecBhavdata_Full"`, `delivery_validation_status="VERIFIED"`.
    """
    clean_syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    sym_set = set(clean_syms)
    now_iso = datetime.now(IST_TZ).isoformat(timespec="seconds")

    by_sym_date: dict[tuple[str, str], EquityDailyBar] = {}

    # 1. Supplementary historical OHLCV + Corporate Actions backfill
    yf_tickers = [f"{s}.NS" for s in clean_syms]
    try:
        raw_df = yf.download(
            tickers=" ".join(yf_tickers),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=True,
            threads=True,
            progress=False,
        )
        for sym in clean_syms:
            t_key = f"{sym}.NS"
            sub = raw_df if len(clean_syms) == 1 else (raw_df[t_key] if t_key in raw_df else None)
            if sub is None or sub.empty:
                continue
            sub_clean = sub.dropna(subset=["Close"]).copy()
            prev_c: float | None = None
            for idx, r in sub_clean.iterrows():
                dt_str = str(idx)[:10]
                c_val = _parse_optional_float(r.get("Close"))
                if c_val is None or c_val <= 0:
                    continue
                o_val = _parse_optional_float(r.get("Open"))
                h_val = _parse_optional_float(r.get("High"))
                l_val = _parse_optional_float(r.get("Low"))
                adj_c = _parse_optional_float(r.get("Adj Close")) or c_val
                vol_val = _parse_optional_float(r.get("Volume"))
                div_val = _parse_optional_float(r.get("Dividends")) or 0.0
                split_val = _parse_optional_float(r.get("Stock Splits")) or 0.0

                by_sym_date[(sym, dt_str)] = EquityDailyBar(
                    symbol=sym,
                    series="EQ",
                    observation_date=dt_str,
                    open=round(o_val, 2) if o_val is not None else None,
                    high=round(h_val, 2) if h_val is not None else None,
                    low=round(l_val, 2) if l_val is not None else None,
                    close=round(c_val, 2),
                    last_price=round(c_val, 2),
                    prev_close=round(prev_c, 2) if prev_c is not None else None,
                    vwap=None,
                    adj_close=round(adj_c, 2),
                    volume=vol_val,
                    turnover_cr=None,
                    trades_count=None,
                    deliverable_qty=None,
                    delivery_pct=None,
                    dividends=round(div_val, 4),
                    stock_splits=round(split_val, 4),
                    source="YahooFinance_Supplementary",
                    publication_timestamp=f"{dt_str}T15:30:00+05:30",
                    collection_timestamp=now_iso,
                    ohlcv_validation_status="VERIFIED",
                    delivery_validation_status="MISSING",
                    data_version=DATA_SCHEMA_VERSION,
                )
                prev_c = c_val
    except Exception:
        pass

    # 2. Overlay Official NSE Bhavcopy + Deliverable Quantity/Percentage for recent trading sessions
    latest_breadth: dict[str, Any] = {
        "advance_count": 0,
        "decline_count": 0,
        "unchanged_count": 0,
        "advance_decline_ratio": None,
    }
    for t_date in recent_nse_trading_dates(max_days=recent_bhavcopy_days):
        nse_bars, breadth = fetch_nse_sec_bhavdata_full(trade_date=t_date, symbols_filter=sym_set)
        if latest_breadth.get("advance_decline_ratio") is None and breadth.get("advance_decline_ratio") is not None:
            latest_breadth = breadth
        for nb in nse_bars:
            key = (nb.symbol, nb.observation_date)
            if key in by_sym_date:
                # Preserve corporate action columns from historical series if present
                nb.dividends = by_sym_date[key].dividends
                nb.stock_splits = by_sym_date[key].stock_splits
                nb.adj_close = by_sym_date[key].adj_close or nb.close
            by_sym_date[key] = nb

    bars_sorted = sorted(by_sym_date.values(), key=lambda x: (x.symbol, x.observation_date))
    return bars_sorted, latest_breadth
