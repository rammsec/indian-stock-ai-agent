#!/usr/bin/env python3
"""Forward Paper-Trading Soak Evidence Evaluator (BHARATALPHA-FORWARD-SOAK-AUTO-001).

Scans a soak evidence directory for per-session subdirectories containing
`live_paper_state.json` (and optional `ticks/<SYM>/<DATE>/provenance.json`),
enforces the exact 7 verbatim qualification gates from
`BHARATALPHA_FORWARD_PAPER_PROTOCOL.md`, and emits:
  - `session_index.json`
  - `forward_evaluation_report.json`
  - `forward_evaluation_report.md`

HARD SAFETY INVARIANTS:
- Read-only evaluation: NEVER modifies `data/lab/sealed_holdout_registry.json`
  or any session archive.
- ALWAYS sets `real_money_authorized = False`.
- `qualification_status` is `"QUALIFIED_FOR_FURTHER_REVIEW"` ONLY if Gates 1-6
  all pass; otherwise `"NOT_QUALIFIED"`. Never authorizes real-money trading.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trader.config import IST_TZ, is_nse_trading_day

# Verbatim protocol thresholds (BHARATALPHA_FORWARD_PAPER_PROTOCOL.md)
MIN_VALID_SESSIONS: Final[int] = 30
MIN_COMPLETED_TRADES: Final[int] = 60
MIN_DISTINCT_SYMBOLS: Final[int] = 3
MIN_PROFIT_FACTOR: Final[float] = 1.25
MAX_CUMULATIVE_DRAWDOWN_PCT: Final[float] = 6.0
MAX_SINGLE_STOCK_CAP_PCT: Final[float] = 15.0
MAX_PORTFOLIO_GROSS_CAP_PCT: Final[float] = 80.0
EOD_CASH_EQUITY_TOLERANCE_INR: Final[float] = 0.05

DEFAULT_SEALED_HOLDOUT_REGISTRY: Final[Path] = ROOT / "data" / "lab" / "sealed_holdout_registry.json"
DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(20\d{2}-\d{2}-\d{2})")

SYNTHETIC_PROVIDERS: Final[set[str]] = {
    "synthetic",
    "synthetic_test",
    "mock",
    "simulated",
    "fake",
}
SYNTHETIC_SOURCE_MODES: Final[set[str]] = {
    "SYNTHETIC_TEST",
    "SYNTHETIC",
    "MOCK_TEST",
}
INCOMPLETE_STATUSES: Final[set[str]] = {
    "INTERRUPTED",
    "INCOMPLETE",
    "CRASHED",
    "FAILED",
    "ABORTED",
    "ERROR",
}


def _parse_iso_date(raw: str | None) -> date | None:
    """Extract a YYYY-MM-DD date from an ISO timestamp or string."""
    if not raw:
        return None
    m = DATE_PATTERN.search(str(raw))
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return None


def _normalize_pct(value: float | int | None) -> float | None:
    """Normalize a ratio (0..1) or percentage (0..100) into percentage points."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    if 0.0 < v <= 1.0:
        return v * 100.0
    return v


def _is_synthetic_provider(provider: str | None) -> bool:
    if not provider:
        return False
    low = str(provider).strip().lower()
    return low in SYNTHETIC_PROVIDERS or "synthetic" in low


def _is_synthetic_source_mode(source_mode: str | None) -> bool:
    if not source_mode:
        return False
    up = str(source_mode).strip().upper()
    return up in SYNTHETIC_SOURCE_MODES or "SYNTHETIC" in up


def _discover_provenance_files(session_dir: Path, state: dict[str, Any]) -> list[Path]:
    """Find provenance.json files under `<session_dir>/ticks/` or referenced in state."""
    found: dict[Path, None] = {}
    ticks_dir = session_dir / "ticks"
    if ticks_dir.exists() and ticks_dir.is_dir():
        for p in sorted(ticks_dir.rglob("provenance.json")):
            found[p.resolve()] = None

    for raw_ref in state.get("archived_provenance_files") or []:
        ref_path = Path(str(raw_ref))
        candidates = [ref_path]
        if not ref_path.is_absolute():
            candidates.append(session_dir / ref_path)
            candidates.append(ROOT / ref_path)
        for c in candidates:
            if c.exists() and c.is_file():
                found[c.resolve()] = None
                break
    return list(found.keys())


def _infer_session_date(
    session_dir: Path,
    state: dict[str, Any],
    provenance_records: list[dict[str, Any]],
) -> date | None:
    """Infer the NSE trading date for a session from state, provenance, trades, or folder name."""
    for key in ("session_date", "trading_date", "date", "session_date_ist"):
        d = _parse_iso_date(state.get(key))
        if d is not None:
            return d
    for prov in provenance_records:
        d = _parse_iso_date(prov.get("trading_date"))
        if d is not None:
            return d
    for tr in state.get("trades") or []:
        if isinstance(tr, dict):
            d = _parse_iso_date(tr.get("entry_time_ist") or tr.get("exit_time_ist"))
            if d is not None:
                return d
    return _parse_iso_date(session_dir.name)


def _inspect_sealed_holdout_registry(registry_path: Path) -> dict[str, Any]:
    """Read-only verification of `data/lab/sealed_holdout_registry.json`."""
    result: dict[str, Any] = {
        "registry_path": str(registry_path),
        "exists": False,
        "valid_json": False,
        "sealed_dates_count": 0,
        "sealed_exposures_count": 0,
        "error": None,
    }
    if not registry_path.exists() or not registry_path.is_file():
        result["error"] = f"Registry file not found at {registry_path}"
        return result
    result["exists"] = True
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = f"Invalid JSON in sealed holdout registry: {exc}"
        return result
    if not isinstance(data, dict):
        result["error"] = "Sealed holdout registry root must be a JSON object"
        return result
    result["valid_json"] = True
    sealed_dates = data.get("sealed_dates")
    sealed_exposures = data.get("sealed_exposures")
    result["sealed_dates_count"] = len(sealed_dates) if isinstance(sealed_dates, list) else 0
    result["sealed_exposures_count"] = len(sealed_exposures) if isinstance(sealed_exposures, list) else 0
    return result


def _evaluate_single_session(state_path: Path, soak_dir: Path) -> dict[str, Any]:
    """Parse and validate a single session's `live_paper_state.json` and provenance."""
    session_dir = state_path.parent
    try:
        rel_id = str(session_dir.relative_to(soak_dir))
        if rel_id in ("", "."):
            rel_id = session_dir.name
    except ValueError:
        rel_id = session_dir.name

    record: dict[str, Any] = {
        "session_id": rel_id,
        "state_path": str(state_path),
        "session_date": None,
        "valid": False,
        "exclusion_reasons": [],
        "provider": None,
        "source_modes": [],
        "synthetic_contaminated": False,
        "fabricated_timestamps": 0,
        "lookahead_violations": 0,
        "live_trading_enabled": None,
        "read_only_mode": None,
        "authority_audit_verified": False,
        "risk_audit_verified": False,
        "ledger_integrity_verified": False,
        "eod_flat_verified": False,
        "open_positions_at_eod": 0,
        "ticks_accepted": 0,
        "bars_1m_evaluated": 0,
        "trade_count": 0,
        "symbols": [],
        "strategy_profile": "UNKNOWN",
        "initial_capital_inr": None,
        "final_cash_inr": None,
        "final_equity_inr": None,
        "total_gross_pnl_inr": 0.0,
        "total_friction_inr": 0.0,
        "total_net_pnl_inr": 0.0,
        "gate_rejections_count": 0,
        "rejection_reasons": {},
        "final_risk_state": "UNKNOWN",
        "margin_call_events": 0,
        "cap_breaches": 0,
        "interrupted": False,
        "max_single_stock_exposure_pct": None,
        "max_portfolio_exposure_pct": None,
        "max_intraday_drawdown_pct": None,
        "provenance_files_count": 0,
        "risk_events": [],
        "trades": [],
    }

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        record["exclusion_reasons"].append(f"malformed_state_json:{exc}")
        return record

    if not isinstance(state, dict):
        record["exclusion_reasons"].append("state_json_not_object")
        return record

    # Load provenance files (strictly read-only)
    prov_paths = _discover_provenance_files(session_dir, state)
    prov_records: list[dict[str, Any]] = []
    source_modes: list[str] = []
    prov_providers: list[str] = []

    if state.get("source_mode"):
        source_modes.append(str(state["source_mode"]))

    for pp in prov_paths:
        try:
            pdata = json.loads(pp.read_text(encoding="utf-8"))
        except Exception as exc:
            record["exclusion_reasons"].append(f"malformed_provenance_json:{pp.name}:{exc}")
            continue
        if not isinstance(pdata, dict):
            record["exclusion_reasons"].append(f"provenance_not_object:{pp.name}")
            continue
        prov_records.append(pdata)
        if pdata.get("source_mode"):
            source_modes.append(str(pdata["source_mode"]))
        if pdata.get("provider"):
            prov_providers.append(str(pdata["provider"]))
        if pdata.get("live_trading_enabled") is not False:
            record["exclusion_reasons"].append(f"provenance_live_trading_enabled_not_false:{pp.name}")
        if pdata.get("read_only") is not True:
            record["exclusion_reasons"].append(f"provenance_read_only_not_true:{pp.name}")
        if pdata.get("fabricated_timestamps"):
            fab = int(pdata["fabricated_timestamps"]) if isinstance(pdata["fabricated_timestamps"], (int, float)) else 1
            record["fabricated_timestamps"] += max(0, fab)
        if pdata.get("lookahead_detected") or pdata.get("lookahead_violations"):
            lv = int(pdata.get("lookahead_violations", 1) or 1)
            record["lookahead_violations"] += max(1, lv)

    record["provenance_files_count"] = len(prov_records)

    provider = str(state.get("provider") or (prov_providers[0] if prov_providers else "")).strip()
    record["provider"] = provider or "unknown"
    record["source_modes"] = sorted(set(source_modes)) if source_modes else ["UNSPECIFIED"]

    # Check synthetic contamination
    synthetic_hit = _is_synthetic_provider(provider)
    for pp_prov in prov_providers:
        if _is_synthetic_provider(pp_prov):
            synthetic_hit = True
    for sm in source_modes:
        if _is_synthetic_source_mode(sm):
            synthetic_hit = True
    if state.get("synthetic") is True or state.get("is_synthetic") is True:
        synthetic_hit = True

    record["synthetic_contaminated"] = synthetic_hit
    if synthetic_hit:
        record["exclusion_reasons"].append("synthetic_data_contamination")

    # Safety flags
    live_flag = state.get("live_trading_enabled")
    ro_flag = state.get("read_only_mode")
    record["live_trading_enabled"] = live_flag
    record["read_only_mode"] = ro_flag
    if live_flag is not False:
        record["exclusion_reasons"].append("live_trading_enabled_not_false")
    if ro_flag is not True:
        record["exclusion_reasons"].append("read_only_mode_not_true")

    # Fabricated timestamps / lookahead flags on state
    if state.get("fabricated_timestamps"):
        fab_s = int(state["fabricated_timestamps"]) if isinstance(state["fabricated_timestamps"], (int, float)) else 1
        record["fabricated_timestamps"] += max(0, fab_s)
    if state.get("lookahead_detected") or state.get("lookahead_violations"):
        lv_s = int(state.get("lookahead_violations", 1) or 1)
        record["lookahead_violations"] += max(1, lv_s)

    if record["fabricated_timestamps"] > 0:
        record["exclusion_reasons"].append("fabricated_timestamps_detected")
    if record["lookahead_violations"] > 0:
        record["exclusion_reasons"].append("lookahead_violation_detected")

    # Infer session date & check NSE holiday / weekend calendar
    sess_date = _infer_session_date(session_dir, state, prov_records)
    record["session_date"] = sess_date.isoformat() if sess_date is not None else None
    if sess_date is not None and not is_nse_trading_day(sess_date):
        record["exclusion_reasons"].append(f"non_nse_trading_day_or_holiday:{sess_date.isoformat()}")

    # Completeness & interruption checks
    status_str = str(state.get("session_status") or state.get("status") or "").strip().upper()
    interrupted = bool(state.get("interrupted", False)) or (status_str in INCOMPLETE_STATUSES)
    if state.get("session_complete") is False or state.get("completed") is False:
        interrupted = True
    record["interrupted"] = interrupted
    if interrupted:
        record["exclusion_reasons"].append(f"incomplete_or_interrupted_session:{status_str or 'interrupted'}")

    ticks_accepted = int(state.get("ticks_accepted", 0) or 0)
    bars_1m = int(state.get("bars_1m_evaluated", 0) or 0)
    record["ticks_accepted"] = ticks_accepted
    record["bars_1m_evaluated"] = bars_1m
    if ticks_accepted <= 0 or bars_1m <= 0:
        record["exclusion_reasons"].append(
            f"insufficient_session_data:ticks={ticks_accepted},bars_1m={bars_1m}"
        )

    # Audit integrity checks
    auth_ok = state.get("authority_audit_verified") is True
    risk_ok = state.get("risk_audit_verified") is True
    ledger_ok = state.get("ledger_integrity_verified") is True
    record["authority_audit_verified"] = auth_ok
    record["risk_audit_verified"] = risk_ok
    record["ledger_integrity_verified"] = ledger_ok
    if not (auth_ok and risk_ok and ledger_ok):
        failed_parts = [
            name
            for name, ok in (
                ("authority", auth_ok),
                ("risk", risk_ok),
                ("ledger", ledger_ok),
            )
            if not ok
        ]
        record["exclusion_reasons"].append(f"audit_integrity_failed:{','.join(failed_parts)}")

    # EOD flat position verification
    final_cash = state.get("final_cash_inr")
    final_equity = state.get("final_equity_inr")
    record["final_cash_inr"] = float(final_cash) if final_cash is not None else None
    record["final_equity_inr"] = float(final_equity) if final_equity is not None else None

    raw_open = state.get("open_positions")
    if isinstance(raw_open, list):
        open_pos_count = len(raw_open)
    elif isinstance(raw_open, dict):
        open_pos_count = len(raw_open)
    elif isinstance(raw_open, (int, float)):
        open_pos_count = int(raw_open)
    else:
        open_pos_count = int(state.get("open_positions_count", 0) or 0)
    record["open_positions_at_eod"] = open_pos_count

    eod_flat = open_pos_count == 0
    if record["final_cash_inr"] is not None and record["final_equity_inr"] is not None:
        if abs(record["final_cash_inr"] - record["final_equity_inr"]) >= EOD_CASH_EQUITY_TOLERANCE_INR:
            eod_flat = False
    elif raw_open is None and "open_positions_count" not in state:
        # Neither open_positions nor cash/equity provided to prove flat EOD
        eod_flat = False

    record["eod_flat_verified"] = eod_flat
    if not eod_flat:
        record["exclusion_reasons"].append("eod_open_positions_or_cash_equity_mismatch")

    # Trades & financial metrics
    trades_raw = state.get("trades") or []
    trades: list[dict[str, Any]] = [t for t in trades_raw if isinstance(t, dict)]
    record["trades"] = trades
    record["trade_count"] = len(trades)
    record["strategy_profile"] = str(state.get("strategy_profile") or "UNKNOWN")

    symbols_set = {str(s).strip().upper() for s in (state.get("symbols") or []) if str(s).strip()}
    for t in trades:
        if t.get("symbol"):
            symbols_set.add(str(t["symbol"]).strip().upper())
    record["symbols"] = sorted(symbols_set)

    gross_pnl = float(state.get("total_gross_pnl_inr", sum(float(t.get("gross_pnl_inr", 0.0) or 0.0) for t in trades)))
    friction = float(state.get("total_friction_inr", sum(float(t.get("friction_inr", 0.0) or 0.0) for t in trades)))
    net_pnl = float(state.get("total_net_pnl_inr", sum(float(t.get("net_pnl_inr", 0.0) or 0.0) for t in trades)))
    record["total_gross_pnl_inr"] = round(gross_pnl, 2)
    record["total_friction_inr"] = round(friction, 2)
    record["total_net_pnl_inr"] = round(net_pnl, 2)

    init_cap = state.get("initial_capital_inr")
    if init_cap is not None:
        record["initial_capital_inr"] = float(init_cap)
    elif record["final_equity_inr"] is not None:
        inferred = record["final_equity_inr"] - net_pnl
        if inferred > 0.0:
            record["initial_capital_inr"] = round(inferred, 2)

    # Gate rejections & reasons
    record["gate_rejections_count"] = int(state.get("gate_rejections_count", 0) or 0)
    rej_reasons: Counter[str] = Counter()
    raw_rej = state.get("gate_rejection_reasons") or state.get("rejection_reasons") or {}
    if isinstance(raw_rej, dict):
        for k, v in raw_rej.items():
            rej_reasons[str(k)] += int(v or 0)
    elif isinstance(raw_rej, list):
        for item in raw_rej:
            rej_reasons[str(item)] += 1
    record["rejection_reasons"] = dict(rej_reasons)

    # Risk events, margin calls, cap breaches
    record["final_risk_state"] = str(state.get("final_risk_state") or "ACTIVE")
    risk_events_raw = state.get("risk_events") or []
    risk_events: list[str] = [str(ev) for ev in risk_events_raw]
    record["risk_events"] = risk_events

    margin_calls = 0
    if state.get("margin_call_triggered") is True:
        margin_calls += 1
    margin_calls += int(state.get("margin_call_count", 0) or 0)
    for text_item in risk_events + list(rej_reasons.keys()):
        if "MARGIN_CALL" in text_item.upper():
            margin_calls += 1
    record["margin_call_events"] = margin_calls

    cap_breaches = int(state.get("cap_breaches", 0) or 0)
    cap_breaches += int(state.get("position_cap_breaches", 0) or 0)
    cap_breaches += int(state.get("portfolio_cap_breaches", 0) or 0)

    max_stock_pct = _normalize_pct(
        state.get("max_single_stock_exposure_pct")
        if state.get("max_single_stock_exposure_pct") is not None
        else state.get("max_position_weight_pct")
    )
    max_port_pct = _normalize_pct(
        state.get("max_portfolio_exposure_pct")
        if state.get("max_portfolio_exposure_pct") is not None
        else state.get("max_gross_exposure_pct")
    )
    record["max_single_stock_exposure_pct"] = round(max_stock_pct, 4) if max_stock_pct is not None else None
    record["max_portfolio_exposure_pct"] = round(max_port_pct, 4) if max_port_pct is not None else None

    if max_stock_pct is not None and max_stock_pct > MAX_SINGLE_STOCK_CAP_PCT + 1e-6:
        cap_breaches += 1
    if max_port_pct is not None and max_port_pct > MAX_PORTFOLIO_GROSS_CAP_PCT + 1e-6:
        cap_breaches += 1
    record["cap_breaches"] = cap_breaches

    max_dd_sess = state.get("max_intraday_drawdown_pct")
    if max_dd_sess is not None:
        record["max_intraday_drawdown_pct"] = round(float(max_dd_sess), 4)

    record["valid"] = len(record["exclusion_reasons"]) == 0
    return record


def evaluate_forward_paper_soak(
    soak_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    sealed_holdout_registry: str | Path | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Evaluate all forward paper-trading sessions under `soak_dir` against the 7 protocol gates."""
    soak_path = Path(soak_dir)
    out_path = Path(output_dir) if output_dir is not None else soak_path
    registry_path = (
        Path(sealed_holdout_registry)
        if sealed_holdout_registry is not None
        else DEFAULT_SEALED_HOLDOUT_REGISTRY
    )

    state_files = sorted(soak_path.rglob("live_paper_state.json")) if soak_path.exists() else []
    session_records = [_evaluate_single_session(sf, soak_path) for sf in state_files]

    # Sort chronologically by session_date (fallback session_id)
    session_records.sort(key=lambda r: (r["session_date"] or "9999-99-99", r["session_id"]))

    valid_recs = [r for r in session_records if r["valid"]]
    invalid_recs = [r for r in session_records if not r["valid"]]

    # Aggregate across valid sessions for performance metrics
    all_valid_trades: list[dict[str, Any]] = []
    distinct_symbols: set[str] = set()
    total_gross_pnl = 0.0
    total_friction = 0.0
    total_net_pnl = 0.0
    gate_rejections_total = 0
    rejection_reasons_summary: Counter[str] = Counter()

    # Exposure & capital utilization tracking
    deployed_notional_list: list[float] = []
    utilization_pct_list: list[float] = []
    max_single_stock_obs: float | None = None
    max_portfolio_obs: float | None = None

    # Strategy profile attribution
    profile_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "gross_pnl_inr": 0.0,
            "friction_inr": 0.0,
            "net_pnl_inr": 0.0,
            "expectancy_inr": 0.0,
        }
    )

    # Cumulative drawdown curve across valid sessions
    base_capital = 50_000.0
    for r in valid_recs:
        if r["initial_capital_inr"] is not None and r["initial_capital_inr"] > 0.0:
            base_capital = float(r["initial_capital_inr"])
            break

    running_equity = base_capital
    peak_equity = base_capital
    max_cum_dd_pct = 0.0

    for r in valid_recs:
        total_gross_pnl += r["total_gross_pnl_inr"]
        total_friction += r["total_friction_inr"]
        total_net_pnl += r["total_net_pnl_inr"]
        gate_rejections_total += r["gate_rejections_count"]
        for rk, rv in r["rejection_reasons"].items():
            rejection_reasons_summary[rk] += int(rv)

        if r["max_single_stock_exposure_pct"] is not None:
            max_single_stock_obs = (
                r["max_single_stock_exposure_pct"]
                if max_single_stock_obs is None
                else max(max_single_stock_obs, r["max_single_stock_exposure_pct"])
            )
        if r["max_portfolio_exposure_pct"] is not None:
            max_portfolio_obs = (
                r["max_portfolio_exposure_pct"]
                if max_portfolio_obs is None
                else max(max_portfolio_obs, r["max_portfolio_exposure_pct"])
            )

        sess_cap = float(r["initial_capital_inr"] or base_capital)
        trades = r["trades"]
        if trades:
            for t in trades:
                all_valid_trades.append(t)
                sym = str(t.get("symbol") or "").strip().upper()
                if sym:
                    distinct_symbols.add(sym)

                entry_px = float(t.get("entry_price", 0.0) or 0.0)
                qty = int(t.get("quantity", 0) or 0)
                notional = entry_px * qty
                if notional > 0.0:
                    deployed_notional_list.append(notional)
                    if sess_cap > 0.0:
                        utilization_pct_list.append((notional / sess_cap) * 100.0)

                t_gross = float(t.get("gross_pnl_inr", 0.0) or 0.0)
                t_fric = float(t.get("friction_inr", 0.0) or 0.0)
                t_net = float(t.get("net_pnl_inr", 0.0) or 0.0)
                prof = str(t.get("strategy_profile") or r["strategy_profile"] or "UNKNOWN")

                ps = profile_stats[prof]
                ps["trades"] += 1
                if t_net > 0.0:
                    ps["wins"] += 1
                elif t_net < 0.0:
                    ps["losses"] += 1
                ps["gross_pnl_inr"] += t_gross
                ps["friction_inr"] += t_fric
                ps["net_pnl_inr"] += t_net

                running_equity += t_net
                if running_equity > peak_equity:
                    peak_equity = running_equity
                if peak_equity > 0.0:
                    dd = ((peak_equity - running_equity) / peak_equity) * 100.0
                    if dd > max_cum_dd_pct:
                        max_cum_dd_pct = dd
        else:
            running_equity += r["total_net_pnl_inr"]
            if running_equity > peak_equity:
                peak_equity = running_equity
            if peak_equity > 0.0:
                dd = ((peak_equity - running_equity) / peak_equity) * 100.0
                if dd > max_cum_dd_pct:
                    max_cum_dd_pct = dd

        if r["max_intraday_drawdown_pct"] is not None:
            max_cum_dd_pct = max(max_cum_dd_pct, float(r["max_intraday_drawdown_pct"]))

    # Finalize strategy profile attribution
    finalized_profiles: dict[str, dict[str, Any]] = {}
    for prof, ps in sorted(profile_stats.items()):
        tc = ps["trades"]
        finalized_profiles[prof] = {
            "trades": tc,
            "wins": ps["wins"],
            "losses": ps["losses"],
            "win_rate_pct": round((ps["wins"] / tc) * 100.0, 2) if tc > 0 else 0.0,
            "gross_pnl_inr": round(ps["gross_pnl_inr"], 2),
            "friction_inr": round(ps["friction_inr"], 2),
            "net_pnl_inr": round(ps["net_pnl_inr"], 2),
            "expectancy_inr": round(ps["net_pnl_inr"] / tc, 2) if tc > 0 else 0.0,
        }

    # Trade-level win rate, profit factor, expectancy
    completed_trades = len(all_valid_trades)
    winning_trades = [t for t in all_valid_trades if float(t.get("net_pnl_inr", 0.0) or 0.0) > 0.0]
    losing_trades = [t for t in all_valid_trades if float(t.get("net_pnl_inr", 0.0) or 0.0) < 0.0]
    gross_wins_net = sum(float(t.get("net_pnl_inr", 0.0) or 0.0) for t in winning_trades)
    gross_losses_net = abs(sum(float(t.get("net_pnl_inr", 0.0) or 0.0) for t in losing_trades))

    if gross_losses_net > 0.0:
        profit_factor = round(gross_wins_net / gross_losses_net, 4)
    elif gross_wins_net > 0.0:
        profit_factor = 999.99  # Finite JSON-safe sentinel for zero losing trades with positive net wins
    else:
        profit_factor = 0.0

    win_rate_pct = round((len(winning_trades) / completed_trades) * 100.0, 2) if completed_trades > 0 else 0.0
    expectancy_per_trade_inr = round(total_net_pnl / completed_trades, 2) if completed_trades > 0 else 0.0

    # Provenance & contamination summary across ALL scanned sessions
    provider_counts: Counter[str] = Counter()
    source_mode_counts: Counter[str] = Counter()
    synthetic_contamination_count = 0
    fabricated_timestamps_count = 0
    lookahead_violations_count = 0
    live_trading_flag_violations = 0
    read_only_flag_violations = 0
    provenance_files_verified = 0

    for r in session_records:
        provider_counts[str(r["provider"])] += 1
        for sm in r["source_modes"]:
            source_mode_counts[sm] += 1
        if r["synthetic_contaminated"]:
            synthetic_contamination_count += 1
        fabricated_timestamps_count += r["fabricated_timestamps"]
        lookahead_violations_count += r["lookahead_violations"]
        if r["live_trading_enabled"] is not False:
            live_trading_flag_violations += 1
        if r["read_only_mode"] is not True:
            read_only_flag_violations += 1
        provenance_files_verified += r["provenance_files_count"]

    data_provenance_summary = {
        "providers": dict(provider_counts),
        "source_modes": dict(source_mode_counts),
        "provenance_files_verified": provenance_files_verified,
        "fabricated_timestamps_count": fabricated_timestamps_count,
        "lookahead_violations_count": lookahead_violations_count,
        "live_trading_flag_violations": live_trading_flag_violations,
        "read_only_flag_violations": read_only_flag_violations,
    }

    # Risk events & interruptions across ALL scanned sessions (and valid compliance counts)
    valid_margin_calls = sum(r["margin_call_events"] for r in valid_recs)
    valid_cap_breaches = sum(r["cap_breaches"] for r in valid_recs)
    total_margin_calls = sum(r["margin_call_events"] for r in session_records)
    total_cap_breaches = sum(r["cap_breaches"] for r in session_records)
    halted_sessions_count = sum(1 for r in session_records if r["final_risk_state"] == "HALTED")
    interrupted_sessions_count = sum(1 for r in session_records if r["interrupted"])
    all_risk_events: list[dict[str, Any]] = []
    for r in session_records:
        if r["risk_events"] or r["margin_call_events"] > 0 or r["cap_breaches"] > 0 or r["interrupted"]:
            all_risk_events.append(
                {
                    "session_id": r["session_id"],
                    "session_date": r["session_date"],
                    "final_risk_state": r["final_risk_state"],
                    "interrupted": r["interrupted"],
                    "margin_call_events": r["margin_call_events"],
                    "cap_breaches": r["cap_breaches"],
                    "events": r["risk_events"],
                }
            )

    risk_events_and_interruptions = {
        "margin_call_events_count": total_margin_calls,
        "valid_session_margin_call_events": valid_margin_calls,
        "cap_breach_count": total_cap_breaches,
        "valid_session_cap_breaches": valid_cap_breaches,
        "halted_sessions_count": halted_sessions_count,
        "interrupted_sessions_count": interrupted_sessions_count,
        "events": all_risk_events,
    }

    # Audit chain integrity across valid sessions
    authority_all_ok = len(valid_recs) > 0 and all(r["authority_audit_verified"] for r in valid_recs)
    risk_all_ok = len(valid_recs) > 0 and all(r["risk_audit_verified"] for r in valid_recs)
    ledger_all_ok = len(valid_recs) > 0 and all(r["ledger_integrity_verified"] for r in valid_recs)
    eod_open_violations = sum(1 for r in valid_recs if not r["eod_flat_verified"])
    eod_flat_all_ok = len(valid_recs) > 0 and eod_open_violations == 0

    audit_chain_integrity = {
        "authority_all_ok": authority_all_ok,
        "risk_all_ok": risk_all_ok,
        "ledger_all_ok": ledger_all_ok,
        "eod_flat_all_ok": eod_flat_all_ok,
        "eod_open_position_violations": eod_open_violations,
    }

    # Exposure & capital utilization
    exposure_and_capital_utilization = {
        "avg_deployed_capital_inr": (
            round(sum(deployed_notional_list) / len(deployed_notional_list), 2)
            if deployed_notional_list
            else 0.0
        ),
        "max_deployed_capital_inr": (
            round(max(deployed_notional_list), 2) if deployed_notional_list else 0.0
        ),
        "avg_utilization_pct": (
            round(sum(utilization_pct_list) / len(utilization_pct_list), 2)
            if utilization_pct_list
            else 0.0
        ),
        "max_utilization_pct": (
            round(max(utilization_pct_list), 2) if utilization_pct_list else 0.0
        ),
        "max_single_stock_exposure_pct": max_single_stock_obs,
        "max_portfolio_exposure_pct": max_portfolio_obs,
    }

    # Gate 6: Sealed holdout governance (strictly read-only)
    sealed_holdout_info = _inspect_sealed_holdout_registry(registry_path)

    # ------------------------------------------------------------------
    # Exact 7 Verbatim Protocol Gates
    # ------------------------------------------------------------------
    valid_count = len(valid_recs)
    invalid_count = len(invalid_recs)
    distinct_symbols_sorted = sorted(distinct_symbols)

    # Gate 1: Data Provenance
    g1_passed = (
        valid_count > 0
        and synthetic_contamination_count == 0
        and fabricated_timestamps_count == 0
        and lookahead_violations_count == 0
        and all(
            (r["live_trading_enabled"] is False)
            and (r["read_only_mode"] is True)
            and (not r["synthetic_contaminated"])
            for r in valid_recs
        )
    )
    if valid_count == 0:
        g1_detail = "No valid forward sessions found to verify provenance."
    elif g1_passed:
        g1_detail = (
            f"100% genuine read-only broker ticks across {valid_count} valid session(s); "
            f"synthetic_contamination=0, fabricated_timestamps=0, lookahead=0, "
            f"live_trading_enabled=False, read_only_mode=True."
        )
    else:
        g1_detail = (
            f"Provenance check failed: synthetic_contamination={synthetic_contamination_count}, "
            f"fabricated_timestamps={fabricated_timestamps_count}, "
            f"lookahead_violations={lookahead_violations_count}, "
            f"live_trading_flag_violations={live_trading_flag_violations}, "
            f"read_only_flag_violations={read_only_flag_violations}."
        )

    # Gate 2: Minimum Forward Sample
    g2_passed = (
        valid_count >= MIN_VALID_SESSIONS
        and completed_trades >= MIN_COMPLETED_TRADES
        and len(distinct_symbols_sorted) >= MIN_DISTINCT_SYMBOLS
    )
    g2_detail = (
        f"valid_sessions={valid_count}/{MIN_VALID_SESSIONS}, "
        f"completed_trades={completed_trades}/{MIN_COMPLETED_TRADES}, "
        f"distinct_symbols={len(distinct_symbols_sorted)}/{MIN_DISTINCT_SYMBOLS} "
        f"({', '.join(distinct_symbols_sorted) or 'none'})."
    )

    # Gate 3: Net-of-Friction Expectancy
    g3_passed = (round(total_net_pnl, 2) > 0.0) and (profit_factor >= MIN_PROFIT_FACTOR)
    g3_detail = (
        f"total_net_pnl_inr={round(total_net_pnl, 2)} (req > 0.0), "
        f"profit_factor={profit_factor} (req >= {MIN_PROFIT_FACTOR}), "
        f"total_friction_inr={round(total_friction, 2)}."
    )

    # Gate 4: Drawdown & Risk Compliance
    g4_passed = (
        valid_count > 0
        and valid_cap_breaches == 0
        and valid_margin_calls == 0
        and round(max_cum_dd_pct, 4) <= MAX_CUMULATIVE_DRAWDOWN_PCT
    )
    g4_detail = (
        f"cap_breaches={valid_cap_breaches} (req 0; 15% stock / 80% portfolio), "
        f"margin_call_events={valid_margin_calls} (req 0), "
        f"max_cumulative_drawdown_pct={round(max_cum_dd_pct, 2)}% (req <= {MAX_CUMULATIVE_DRAWDOWN_PCT}%)."
    )

    # Gate 5: Cryptographic Audit Integrity
    g5_passed = (
        valid_count > 0
        and authority_all_ok
        and risk_all_ok
        and ledger_all_ok
        and eod_flat_all_ok
    )
    g5_detail = (
        f"authority_all_ok={authority_all_ok}, risk_all_ok={risk_all_ok}, "
        f"ledger_all_ok={ledger_all_ok}, eod_flat_all_ok={eod_flat_all_ok} "
        f"(eod_open_violations={eod_open_violations})."
    )

    # Gate 6: Sealed Holdout Governance
    g6_passed = bool(sealed_holdout_info["exists"] and sealed_holdout_info["valid_json"])
    g6_detail = (
        f"registry={sealed_holdout_info['registry_path']}, "
        f"exists={sealed_holdout_info['exists']}, valid_json={sealed_holdout_info['valid_json']}, "
        f"sealed_dates={sealed_holdout_info['sealed_dates_count']}, "
        f"sealed_exposures={sealed_holdout_info['sealed_exposures_count']} (read-only verified)."
    )

    # Gate 7: Real-Money Boundary (ALWAYS False; never authorizes real-money trading)
    gates_1_to_6_passed = all((g1_passed, g2_passed, g3_passed, g4_passed, g5_passed, g6_passed))
    qualification_status = (
        "QUALIFIED_FOR_FURTHER_REVIEW" if gates_1_to_6_passed else "NOT_QUALIFIED"
    )
    real_money_authorized: Final[bool] = False
    g7_passed = True
    g7_detail = (
        f"real_money_authorized={real_money_authorized} (hard invariant); "
        f"qualification_status={qualification_status}. "
        f"Real-money trading is NEVER authorized by automated soak evaluation."
    )

    gates: dict[str, dict[str, Any]] = {
        "gate_1_data_provenance": {
            "name": "Gate 1: Data Provenance",
            "passed": g1_passed,
            "detail": g1_detail,
        },
        "gate_2_minimum_forward_sample": {
            "name": "Gate 2: Minimum Forward Sample",
            "passed": g2_passed,
            "detail": g2_detail,
        },
        "gate_3_net_of_friction_expectancy": {
            "name": "Gate 3: Net-of-Friction Expectancy",
            "passed": g3_passed,
            "detail": g3_detail,
        },
        "gate_4_drawdown_and_risk_compliance": {
            "name": "Gate 4: Drawdown & Risk Compliance",
            "passed": g4_passed,
            "detail": g4_detail,
        },
        "gate_5_cryptographic_audit_integrity": {
            "name": "Gate 5: Cryptographic Audit Integrity",
            "passed": g5_passed,
            "detail": g5_detail,
        },
        "gate_6_sealed_holdout_governance": {
            "name": "Gate 6: Sealed Holdout Governance",
            "passed": g6_passed,
            "detail": g6_detail,
        },
        "gate_7_real_money_boundary": {
            "name": "Gate 7: Real-Money Boundary",
            "passed": g7_passed,
            "detail": g7_detail,
        },
    }

    evaluated_at_ist = datetime.now(IST_TZ).isoformat()

    report: dict[str, Any] = {
        "schema_version": "bharatalpha_forward_soak_eval_v1",
        "evaluated_at_ist": evaluated_at_ist,
        "soak_dir": str(soak_path),
        "real_money_authorized": real_money_authorized,
        "qualification_status": qualification_status,
        "valid_sessions": valid_count,
        "invalid_sessions": invalid_count,
        "total_sessions_scanned": len(session_records),
        "completed_trades": completed_trades,
        "distinct_symbols": distinct_symbols_sorted,
        "distinct_symbol_count": len(distinct_symbols_sorted),
        "total_gross_pnl_inr": round(total_gross_pnl, 2),
        "total_friction_inr": round(total_friction, 2),
        "total_net_pnl_inr": round(total_net_pnl, 2),
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate_pct,
        "expectancy_per_trade_inr": expectancy_per_trade_inr,
        "max_cumulative_drawdown_pct": round(max_cum_dd_pct, 2),
        "exposure_and_capital_utilization": exposure_and_capital_utilization,
        "gate_rejections_total": gate_rejections_total,
        "rejection_reasons_summary": dict(rejection_reasons_summary),
        "data_provenance_summary": data_provenance_summary,
        "synthetic_contamination_count": synthetic_contamination_count,
        "strategy_profile_attribution": finalized_profiles,
        "risk_events_and_interruptions": risk_events_and_interruptions,
        "audit_chain_integrity": audit_chain_integrity,
        "sealed_holdout_governance": sealed_holdout_info,
        "gates": gates,
    }

    session_index: dict[str, Any] = {
        "schema_version": "bharatalpha_forward_session_index_v1",
        "evaluated_at_ist": evaluated_at_ist,
        "soak_dir": str(soak_path),
        "total_sessions_scanned": len(session_records),
        "valid_sessions": valid_count,
        "invalid_sessions": invalid_count,
        "sessions": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in session_records
        ],
    }

    if write_outputs:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "session_index.json").write_text(
            json.dumps(session_index, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (out_path / "forward_evaluation_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (out_path / "forward_evaluation_report.md").write_text(
            render_markdown_report(report, session_index),
            encoding="utf-8",
        )

    return report


def render_markdown_report(report: dict[str, Any], session_index: dict[str, Any]) -> str:
    """Render a structured Markdown evaluation report for human review."""
    lines: list[str] = [
        "# BharatAlpha Forward Paper-Trading Soak Evaluation Report",
        "",
        f"- **Evaluated At (IST)**: `{report['evaluated_at_ist']}`",
        f"- **Soak Directory**: `{report['soak_dir']}`",
        f"- **Qualification Status**: **`{report['qualification_status']}`**",
        f"- **Real-Money Authorized**: **`{report['real_money_authorized']}`** *(Hard protocol boundary — never authorized automatically)*",
        "",
        "## 1. Seven-Gate Qualification Summary",
        "",
        "| Gate | Status | Detail |",
        "| :--- | :---: | :--- |",
    ]

    for gate_obj in report["gates"].values():
        status = "PASS" if gate_obj["passed"] else "FAIL"
        lines.append(f"| {gate_obj['name']} | **{status}** | {gate_obj['detail']} |")

    lines.extend(
        [
            "",
            "## 2. Core Forward Sample & Financial Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Valid Complete NSE Sessions | `{report['valid_sessions']}` (excluded/invalid: `{report['invalid_sessions']}`) |",
            f"| Completed Round-Trip Trades | `{report['completed_trades']}` |",
            f"| Distinct Symbols | `{report['distinct_symbol_count']}` ({', '.join(report['distinct_symbols']) or 'none'}) |",
            f"| Total Gross P&L (INR) | `₹{report['total_gross_pnl_inr']:,.2f}` |",
            f"| Total Statutory Friction + Slippage (INR) | `₹{report['total_friction_inr']:,.2f}` |",
            f"| Total Net P&L (INR) | `₹{report['total_net_pnl_inr']:,.2f}` |",
            f"| Profit Factor (Net) | `{report['profit_factor']}` |",
            f"| Win Rate | `{report['win_rate_pct']}%` |",
            f"| Expectancy per Trade (INR) | `₹{report['expectancy_per_trade_inr']:,.2f}` |",
            f"| Max Cumulative Drawdown | `{report['max_cumulative_drawdown_pct']}%` |",
            f"| Gate Rejections Total | `{report['gate_rejections_total']}` |",
            f"| Synthetic Contamination Count | `{report['synthetic_contamination_count']}` |",
            "",
            "## 3. Exposure & Capital Utilization",
            "",
        ]
    )

    exp = report["exposure_and_capital_utilization"]
    lines.extend(
        [
            f"- **Average Deployed Capital / Trade**: `₹{exp['avg_deployed_capital_inr']:,.2f}`",
            f"- **Maximum Deployed Capital / Trade**: `₹{exp['max_deployed_capital_inr']:,.2f}`",
            f"- **Average Capital Utilization**: `{exp['avg_utilization_pct']}%`",
            f"- **Maximum Capital Utilization**: `{exp['max_utilization_pct']}%`",
            f"- **Observed Max Single-Stock Exposure**: `{exp['max_single_stock_exposure_pct']}`",
            f"- **Observed Max Gross Portfolio Exposure**: `{exp['max_portfolio_exposure_pct']}`",
            "",
            "## 4. Strategy Profile Attribution",
            "",
        ]
    )

    if report["strategy_profile_attribution"]:
        lines.extend(
            [
                "| Strategy Profile | Trades | Wins | Losses | Win Rate | Gross P&L (INR) | Friction (INR) | Net P&L (INR) | Expectancy (INR) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for prof, stats in report["strategy_profile_attribution"].items():
            lines.append(
                f"| `{prof}` | {stats['trades']} | {stats['wins']} | {stats['losses']} | "
                f"{stats['win_rate_pct']}% | ₹{stats['gross_pnl_inr']:,.2f} | "
                f"₹{stats['friction_inr']:,.2f} | ₹{stats['net_pnl_inr']:,.2f} | "
                f"₹{stats['expectancy_inr']:,.2f} |"
            )
    else:
        lines.append("_No completed trades recorded across valid sessions._")

    prov = report["data_provenance_summary"]
    audit = report["audit_chain_integrity"]
    risk = report["risk_events_and_interruptions"]

    lines.extend(
        [
            "",
            "## 5. Data Provenance, Risk Events & Audit Integrity",
            "",
            f"- **Providers**: `{json.dumps(prov['providers'])}`",
            f"- **Source Modes**: `{json.dumps(prov['source_modes'])}`",
            f"- **Provenance Files Verified**: `{prov['provenance_files_verified']}`",
            f"- **Authority Audit All OK**: `{audit['authority_all_ok']}`",
            f"- **Risk Audit All OK**: `{audit['risk_all_ok']}`",
            f"- **Ledger Integrity All OK**: `{audit['ledger_all_ok']}`",
            f"- **EOD Flat All OK**: `{audit['eod_flat_all_ok']}` (violations: `{audit['eod_open_position_violations']}`)",
            f"- **Margin Call Events**: `{risk['margin_call_events_count']}`",
            f"- **Cap Breaches (15% stock / 80% portfolio)**: `{risk['cap_breach_count']}`",
            f"- **Halted Sessions**: `{risk['halted_sessions_count']}` | **Interrupted Sessions**: `{risk['interrupted_sessions_count']}`",
            "",
        ]
    )

    if report["rejection_reasons_summary"]:
        lines.append("### Gate Rejection Reasons Summary")
        lines.append("")
        for reason, cnt in sorted(report["rejection_reasons_summary"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{reason}`: `{cnt}`")
        lines.append("")

    invalid_sessions = [s for s in session_index["sessions"] if not s["valid"]]
    lines.append("## 6. Excluded / Invalid Sessions")
    lines.append("")
    if invalid_sessions:
        lines.extend(
            [
                "| Session ID | Date | Provider | Exclusion Reasons |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for s in invalid_sessions:
            reasons = "; ".join(s["exclusion_reasons"])
            lines.append(
                f"| `{s['session_id']}` | `{s['session_date'] or 'unknown'}` | `{s['provider']}` | `{reasons}` |"
            )
    else:
        lines.append("_Zero excluded or invalid sessions._")

    lines.append("")
    return "\n".join(lines)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BharatAlpha Forward Paper-Trading Soak Evidence Evaluator (7-Gate Protocol)"
    )
    parser.add_argument(
        "--soak-dir",
        type=Path,
        required=True,
        help="Root directory containing per-session subdirectories with live_paper_state.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write session_index.json, forward_evaluation_report.json, and .md (defaults to --soak-dir)",
    )
    parser.add_argument(
        "--sealed-holdout-registry",
        type=Path,
        default=DEFAULT_SEALED_HOLDOUT_REGISTRY,
        help="Read-only path to data/lab/sealed_holdout_registry.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    report = evaluate_forward_paper_soak(
        args.soak_dir,
        output_dir=args.output_dir,
        sealed_holdout_registry=args.sealed_holdout_registry,
        write_outputs=True,
    )
    print(
        json.dumps(
            {
                "qualification_status": report["qualification_status"],
                "real_money_authorized": report["real_money_authorized"],
                "valid_sessions": report["valid_sessions"],
                "invalid_sessions": report["invalid_sessions"],
                "completed_trades": report["completed_trades"],
                "distinct_symbol_count": report["distinct_symbol_count"],
                "total_net_pnl_inr": report["total_net_pnl_inr"],
                "profit_factor": report["profit_factor"],
                "max_cumulative_drawdown_pct": report["max_cumulative_drawdown_pct"],
                "gates_passed": sum(1 for g in report["gates"].values() if g["passed"]),
                "gates_total": len(report["gates"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
