"""Isolated Paper Portfolio & Risk Simulation Audit Module.

Provides a deterministic, zero-lookahead diagnostic engine that audits the
Phase 5.3/5.4 paper research portfolio without modifying frozen prediction or
signal-generation logic (`backend/news_intel/forward_validation.py` remains
untouched).

Key capabilities:
1. Canonical signal hygiene & look-ahead verification:
   - Excludes `is_duplicate_run == True` snapshots.
   - Detects & classifies `REJECTED_LOOKAHEAD` when a snapshot's
     `prediction_timestamp` is at or after the scheduled NSE entry open
     (`entry_date` 09:15 IST).
   - Enforces one canonical signal per `(symbol, observation_date)` when
     multiple intra-day/pre-market/post-market sessions exist on the same date,
     classifying superseded same-day sessions as `REJECTED_SAME_DAY_DUPLICATE`.
2. Full rejection-reason taxonomy:
   - `REJECTED_UNAVAILABLE_MODEL`
   - `REJECTED_NON_BUY_SIGNAL` (with `raw_reason` preserving `NO TRADE`/`WATCH`/`HOLD`)
   - `REJECTED_LOOKAHEAD`
   - `REJECTED_SAME_DAY_DUPLICATE`
   - `REJECTED_MISSING_OBSERVATION_BAR`
   - `REJECTED_PENDING_NEXT_SESSION`
   - `REJECTED_INVALID_ENTRY_PRICE`
   - `REJECTED_OVERLAPPING_POSITION`
   - `REJECTED_INSUFFICIENT_CASH`
   - `REJECTED_MAX_EXPOSURE`
   - `REJECTED_MIN_LOT_CAPITAL`
3. Cash, cost-basis vs MTM exposure, CNC delivery cost, concentration/sector,
   and drawdown auditing.
4. Fair benchmark alignment:
   - Resolves both `"NIFTY_50"` (actual stored index code) and `"NIFTY 50"`.
   - Separates `naive_momentum_1d`, `naive_momentum_5d`, `naive_momentum_10d`
     and `cash_no_trade` with identical next-open + exposure + CNC cost rules.
5. Persistence audit of `data/predictions/paper_trades.parquet` against the
   full simulated audit trail (detecting `[-60:]` truncation, legacy
   `action="REJECTED"` rows, and leaked duplicate-run rows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..config import IST_TZ
from ..news_intel.relationships import RELATIONSHIP_MAP, get_pilot_relationship

STARTING_CAPITAL_INR = 1_000_000.0
MAX_POSITION_PCT = 0.10
MAX_EXPOSURE_PCT = 0.80
MIN_LOT_CAPITAL_INR = 1_000.0
DEFAULT_CNC_ROUNDTRIP_COST_PCT = 0.347
NIFTY_INDEX_CODES = ("NIFTY_50", "NIFTY 50")

RejectionCode = Literal[
    "REJECTED_UNAVAILABLE_MODEL",
    "REJECTED_NON_BUY_SIGNAL",
    "REJECTED_LOOKAHEAD",
    "REJECTED_SAME_DAY_DUPLICATE",
    "REJECTED_MISSING_OBSERVATION_BAR",
    "REJECTED_PENDING_NEXT_SESSION",
    "REJECTED_INVALID_ENTRY_PRICE",
    "REJECTED_OVERLAPPING_POSITION",
    "REJECTED_INSUFFICIENT_CASH",
    "REJECTED_MAX_EXPOSURE",
    "REJECTED_MIN_LOT_CAPITAL",
]


@dataclass
class AuditEvent:
    """Single chronological event in the audited paper portfolio timeline."""

    strategy_horizon: Literal["1D", "5D", "10D"]
    prediction_id: str
    symbol: str
    sector: str
    signal_date: str
    event_date: str
    action: Literal[
        "SIGNAL_RECEIVED",
        "ENTRY_EXECUTED",
        "POSITION_MARKED_TO_MARKET",
        "POSITION_EXITED",
        "POSITION_OPEN",
        "SIGNAL_REJECTED",
    ]
    reason: str
    raw_reason: str = ""
    prediction_timestamp: str | None = None
    entry_timestamp: str | None = None
    entry_date: str | None = None
    exit_date: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    allocated_capital_inr: float = 0.0
    position_weight_cost_pct: float = 0.0
    position_weight_mtm_pct: float = 0.0
    gross_return_pct: float | None = None
    transaction_cost_inr: float = 0.0
    net_return_pct: float | None = None
    net_pnl_inr: float = 0.0
    realized_pnl_inr: float = 0.0
    unrealized_pnl_inr: float = 0.0
    cash_after_inr: float = STARTING_CAPITAL_INR
    invested_cost_after_inr: float = 0.0
    invested_mtm_after_inr: float = 0.0
    equity_after_inr: float = STARTING_CAPITAL_INR
    holding_sessions: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_horizon": self.strategy_horizon,
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "sector": self.sector,
            "signal_date": self.signal_date,
            "event_date": self.event_date,
            "action": self.action,
            "reason": self.reason,
            "raw_reason": self.raw_reason,
            "prediction_timestamp": self.prediction_timestamp,
            "entry_timestamp": self.entry_timestamp,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "allocated_capital_inr": self.allocated_capital_inr,
            "position_weight_cost_pct": self.position_weight_cost_pct,
            "position_weight_mtm_pct": self.position_weight_mtm_pct,
            "gross_return_pct": self.gross_return_pct,
            "transaction_cost_inr": self.transaction_cost_inr,
            "net_return_pct": self.net_return_pct,
            "net_pnl_inr": self.net_pnl_inr,
            "realized_pnl_inr": self.realized_pnl_inr,
            "unrealized_pnl_inr": self.unrealized_pnl_inr,
            "cash_after_inr": self.cash_after_inr,
            "invested_cost_after_inr": self.invested_cost_after_inr,
            "invested_mtm_after_inr": self.invested_mtm_after_inr,
            "equity_after_inr": self.equity_after_inr,
            "holding_sessions": self.holding_sessions,
        }


@dataclass
class AuditStrategyResult:
    """Audited portfolio summary and risk diagnostics for one horizon."""

    horizon: Literal["1D", "5D", "10D"]
    starting_capital_inr: float = STARTING_CAPITAL_INR
    ending_capital_inr: float = STARTING_CAPITAL_INR
    cash_inr: float = STARTING_CAPITAL_INR
    invested_cost_inr: float = 0.0
    invested_mtm_inr: float = 0.0
    realized_pnl_inr: float = 0.0
    unrealized_pnl_inr: float = 0.0
    accrued_exit_cost_est_inr: float = 0.0
    net_return_pct: float = 0.0
    gross_return_pct: float = 0.0
    total_cost_inr: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_exposure_cost_pct: float = 0.0
    peak_exposure_mtm_pct: float = 0.0
    max_symbol_concentration_pct: float = 0.0
    max_sector_concentration_pct: float = 0.0
    sector_exposure_pct: dict[str, float] = field(default_factory=dict)
    signals_received: int = 0
    unique_signal_days: int = 0
    entries_executed: int = 0
    open_positions_count: int = 0
    matured_positions_count: int = 0
    rejected_signals: int = 0
    pending_next_session_count: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    win_rate_pct: float | None = None
    accounting_invariant_ok: bool = True
    cash_conservation_error_inr: float = 0.0
    status: Literal["ACTIVE", "INSUFFICIENT_DATA"] = "INSUFFICIENT_DATA"
    daily_equity_curve: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PersistenceAuditSummary:
    """Comparison of stored `paper_trades.parquet` vs full canonical simulation."""

    parquet_rows_total: int
    legacy_action_rejected_rows: int
    duplicate_run_rows_in_parquet: int
    full_unfiltered_simulated_events: int
    truncated_recent_trades_limit: int
    dropped_by_truncation_count: int
    dropped_by_horizon: dict[str, int]
    action_counts_in_parquet: dict[str, int]


@dataclass
class PortfolioAuditReport:
    """Complete read-only diagnostic report for BharatAlpha AI paper portfolio."""

    generated_at: str
    audit_mode: Literal["strict_canonical", "production_replay"]
    total_snapshots: int
    canonical_snapshots: int
    duplicate_run_snapshots: int
    dedup_daily_session_snapshots: int
    lookahead_flagged_snapshots: int
    strategies: dict[str, AuditStrategyResult]
    benchmarks: dict[str, dict[str, Any]]
    events: list[AuditEvent]
    persistence_audit: PersistenceAuditSummary | None = None
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "audit_mode": self.audit_mode,
            "total_snapshots": self.total_snapshots,
            "canonical_snapshots": self.canonical_snapshots,
            "duplicate_run_snapshots": self.duplicate_run_snapshots,
            "dedup_daily_session_snapshots": self.dedup_daily_session_snapshots,
            "lookahead_flagged_snapshots": self.lookahead_flagged_snapshots,
            "strategies": {
                h: {
                    k: v
                    for k, v in s.__dict__.items()
                }
                for h, s in self.strategies.items()
            },
            "benchmarks": self.benchmarks,
            "persistence_audit": self.persistence_audit.__dict__ if self.persistence_audit else None,
            "findings": self.findings,
            "events_count": len(self.events),
        }


def _parse_ist_timestamp(ts: Any) -> datetime | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    s = str(ts).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(IST_TZ) if dt.tzinfo else dt.replace(tzinfo=IST_TZ)
    except Exception:
        return None


def _entry_open_timestamp_ist(entry_date: str) -> datetime:
    return datetime.fromisoformat(f"{entry_date[:10]}T09:15:00+05:30").astimezone(IST_TZ)


def _resolve_symbol_sector(symbol: str) -> str:
    sym = symbol.upper().strip()
    if sym in RELATIONSHIP_MAP:
        return str(RELATIONSHIP_MAP[sym].get("sector") or "Unknown")
    return str(get_pilot_relationship(sym).get("sector") or "Unknown")


def _compute_ca_adj_factor(grp: pd.DataFrame, entry_idx: int, exit_idx: int) -> tuple[float, bool]:
    if "adj_close" not in grp.columns:
        return 1.0, False
    entry_close = grp.loc[entry_idx, "close"]
    exit_close = grp.loc[exit_idx, "close"]
    ac_in = grp.loc[entry_idx, "adj_close"]
    ac_out = grp.loc[exit_idx, "adj_close"]
    if (
        pd.notna(ac_in)
        and pd.notna(ac_out)
        and pd.notna(entry_close)
        and pd.notna(exit_close)
        and float(ac_in) > 0
        and float(entry_close) > 0
        and float(exit_close) > 0
    ):
        ratio_in = float(ac_in) / float(entry_close)
        ratio_out = float(ac_out) / float(exit_close)
        if ratio_in > 0 and abs((ratio_out / ratio_in) - 1.0) > 1e-4:
            return float(ratio_out / ratio_in), True
    return 1.0, False


def _prepare_sym_map(equities_df: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if equities_df is None or equities_df.empty:
        return {}
    cols = [c for c in ("symbol", "observation_date", "open", "close", "adj_close") if c in equities_df.columns]
    eq = equities_df[cols].copy()
    eq["symbol"] = eq["symbol"].astype(str).str.upper()
    eq["observation_date"] = eq["observation_date"].astype(str)
    eq = eq.drop_duplicates(subset=["symbol", "observation_date"], keep="last")
    eq = eq.sort_values(["symbol", "observation_date"]).reset_index(drop=True)
    return {str(sym): grp.reset_index(drop=True) for sym, grp in eq.groupby("symbol", sort=False)}


def _classify_canonical_signals(
    snaps_df: pd.DataFrame,
    sym_map: dict[str, pd.DataFrame],
    enforce_no_lookahead: bool = True,
    dedup_same_day_sessions: bool = True,
) -> tuple[pd.DataFrame, int, int, int, int]:
    """Filter `is_duplicate_run`, flag look-ahead, and optionally pick 1 canonical signal/day."""
    if snaps_df.empty:
        return snaps_df.copy(), 0, 0, 0, 0

    df = snaps_df.copy()
    total = len(df)
    if "is_duplicate_run" in df.columns:
        dup_mask = df["is_duplicate_run"].fillna(False).astype(bool)
        canon = df[~dup_mask].copy()
        dup_cnt = int(dup_mask.sum())
    else:
        canon = df.copy()
        dup_cnt = 0

    canon["symbol"] = canon["symbol"].astype(str).str.upper()
    canon["observation_date"] = canon["observation_date"].astype(str)
    if "prediction_timestamp" not in canon.columns:
        canon["prediction_timestamp"] = None
    canon = canon.sort_values(["observation_date", "symbol", "prediction_timestamp", "prediction_id"]).reset_index(drop=True)

    # Pre-resolve scheduled entry date (t+1) and check lookahead vs prediction_timestamp
    entry_dates: list[str | None] = []
    entry_ts_strs: list[str | None] = []
    is_lookahead: list[bool] = []
    has_obs_bar: list[bool] = []

    all_eq_dates = sorted({d for grp in sym_map.values() for d in grp["observation_date"].tolist()})

    for _, r in canon.iterrows():
        sym = str(r["symbol"]).upper()
        obs_dt = str(r["observation_date"])
        grp = sym_map.get(sym)
        ent_dt: str | None = None
        obs_bar_found = False

        if grp is not None and not grp.empty:
            matches = grp.index[grp["observation_date"] == obs_dt].tolist()
            if matches:
                obs_bar_found = True
                b_idx = int(matches[0])
                if b_idx + 1 < len(grp):
                    ent_dt = str(grp.loc[b_idx + 1, "observation_date"])
            else:
                after = grp[grp["observation_date"] > obs_dt]
                if not after.empty:
                    ent_dt = str(after.iloc[0]["observation_date"])
        elif all_eq_dates:
            after_all = [d for d in all_eq_dates if d > obs_dt]
            if after_all:
                ent_dt = after_all[0]

        has_obs_bar.append(obs_bar_found)
        entry_dates.append(ent_dt)

        pred_dt = _parse_ist_timestamp(r.get("prediction_timestamp"))
        if ent_dt is not None and pred_dt is not None:
            ent_ts = _entry_open_timestamp_ist(ent_dt)
            entry_ts_strs.append(ent_ts.isoformat())
            is_lookahead.append(pred_dt >= ent_ts)
        else:
            entry_ts_strs.append(_entry_open_timestamp_ist(ent_dt).isoformat() if ent_dt else None)
            is_lookahead.append(False)

    canon["_has_obs_bar"] = has_obs_bar
    canon["_scheduled_entry_date"] = entry_dates
    canon["_scheduled_entry_ts"] = entry_ts_strs
    canon["_is_lookahead"] = is_lookahead
    lookahead_cnt = int(sum(is_lookahead))

    # Within each (symbol, observation_date), pick earliest non-lookahead canonical session if dedup enabled
    is_same_day_dup: list[bool] = [False] * len(canon)
    if dedup_same_day_sessions:
        kept_keys: set[tuple[str, str]] = set()
        for idx, r in canon.iterrows():
            key = (str(r["symbol"]), str(r["observation_date"]))
            if enforce_no_lookahead and bool(r["_is_lookahead"]):
                continue
            if key not in kept_keys:
                kept_keys.add(key)
            else:
                is_same_day_dup[idx] = True

    canon["_is_same_day_dup"] = is_same_day_dup
    effective_daily = int((~canon["_is_same_day_dup"] & (~canon["_is_lookahead"] if enforce_no_lookahead else True)).sum())
    return canon, total, len(canon), dup_cnt, effective_daily


def _simulate_audited_horizon(
    horizon: Literal["1D", "5D", "10D"],
    hold_n: int,
    canon_df: pd.DataFrame,
    sym_map: dict[str, pd.DataFrame],
    signal_getter: Any,
    enforce_no_lookahead: bool = True,
    dedup_same_day_sessions: bool = True,
    emit_events: bool = True,
) -> tuple[AuditStrategyResult, list[AuditEvent]]:
    """Run deterministic chronological audit simulation for one horizon."""
    events: list[AuditEvent] = []
    cash = STARTING_CAPITAL_INR
    realized_pnl = 0.0
    gross_pnl_total = 0.0
    cost_total = 0.0
    peak_equity = STARTING_CAPITAL_INR
    max_dd = 0.0
    peak_exposure_cost_pct = 0.0
    peak_exposure_mtm_pct = 0.0
    max_symbol_conc_pct = 0.0
    max_sector_conc_pct = 0.0
    signals_received = 0
    entries_executed = 0
    matured_cnt = 0
    rejected_cnt = 0
    pending_next_session_cnt = 0
    win_cnt = 0
    rejection_counts: dict[str, int] = {}
    daily_equity_curve: list[dict[str, Any]] = []
    seen_signal_days: set[tuple[str, str]] = set()

    open_positions: dict[str, dict[str, Any]] = {}

    eq_dates = sorted({d for grp in sym_map.values() for d in grp["observation_date"].tolist()})
    snap_dates = sorted(canon_df["observation_date"].astype(str).unique().tolist()) if not canon_df.empty else []
    all_sessions = sorted(set(eq_dates) | set(snap_dates))

    signals_by_obs: dict[str, list[dict[str, Any]]] = {}
    if not canon_df.empty:
        for _, r in canon_df.sort_values(["observation_date", "symbol", "prediction_timestamp", "prediction_id"]).iterrows():
            signals_by_obs.setdefault(str(r["observation_date"]), []).append(r.to_dict())

    pending_entries_by_date: dict[str, list[dict[str, Any]]] = {}

    def _record_rejection(
        sig: dict[str, Any],
        sess_dt: str,
        code: RejectionCode,
        raw_reason: str,
        inv_cost: float,
        inv_mtm: float,
        eq_curr: float,
    ) -> None:
        nonlocal rejected_cnt
        rejected_cnt += 1
        rejection_counts[code] = rejection_counts.get(code, 0) + 1
        if emit_events:
            sym = str(sig["symbol"]).upper()
            events.append(
                AuditEvent(
                    strategy_horizon=horizon,
                    prediction_id=str(sig["prediction_id"]),
                    symbol=sym,
                    sector=_resolve_symbol_sector(sym),
                    signal_date=str(sig["observation_date"]),
                    event_date=sess_dt,
                    action="SIGNAL_REJECTED",
                    reason=code,
                    raw_reason=raw_reason,
                    prediction_timestamp=sig.get("prediction_timestamp"),
                    entry_timestamp=sig.get("_scheduled_entry_ts"),
                    cash_after_inr=round(cash, 2),
                    invested_cost_after_inr=round(inv_cost, 2),
                    invested_mtm_after_inr=round(inv_mtm, 2),
                    equity_after_inr=round(eq_curr, 2),
                )
            )

    for sess_dt in all_sessions:
        # Phase 1: Market Open (09:15 IST) - Execute scheduled entries
        for sig in pending_entries_by_date.get(sess_dt, []):
            sym = str(sig["symbol"]).upper()
            pred_id = str(sig["prediction_id"])
            obs_dt = str(sig["observation_date"])
            rec = str(signal_getter(sig) or "UNAVAILABLE")

            inv_cost = sum(p["capital"] for p in open_positions.values())
            unrl_curr = sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
            inv_mtm = inv_cost + unrl_curr
            eq_curr = cash + inv_mtm

            if enforce_no_lookahead and bool(sig.get("_is_lookahead", False)):
                _record_rejection(
                    sig,
                    sess_dt,
                    "REJECTED_LOOKAHEAD",
                    f"prediction_timestamp={sig.get('prediction_timestamp')} >= entry_open={sig.get('_scheduled_entry_ts')}",
                    inv_cost,
                    inv_mtm,
                    eq_curr,
                )
                continue

            if dedup_same_day_sessions and bool(sig.get("_is_same_day_dup", False)):
                _record_rejection(
                    sig,
                    sess_dt,
                    "REJECTED_SAME_DAY_DUPLICATE",
                    f"superseded_session={sig.get('nse_session')}",
                    inv_cost,
                    inv_mtm,
                    eq_curr,
                )
                continue

            if rec == "UNAVAILABLE":
                _record_rejection(sig, sess_dt, "REJECTED_UNAVAILABLE_MODEL", rec, inv_cost, inv_mtm, eq_curr)
                continue

            if rec != "BUY":
                _record_rejection(sig, sess_dt, "REJECTED_NON_BUY_SIGNAL", rec, inv_cost, inv_mtm, eq_curr)
                continue

            if sym in open_positions:
                _record_rejection(
                    sig,
                    sess_dt,
                    "REJECTED_OVERLAPPING_POSITION",
                    f"open_until={open_positions[sym].get('exit_date')}",
                    inv_cost,
                    inv_mtm,
                    eq_curr,
                )
                continue

            if cash < MIN_LOT_CAPITAL_INR:
                _record_rejection(sig, sess_dt, "REJECTED_INSUFFICIENT_CASH", f"cash={cash:.2f}", inv_cost, inv_mtm, eq_curr)
                continue

            headroom = max(0.0, eq_curr * MAX_EXPOSURE_PCT - inv_cost)
            if headroom < MIN_LOT_CAPITAL_INR:
                _record_rejection(sig, sess_dt, "REJECTED_MAX_EXPOSURE", f"headroom={headroom:.2f}", inv_cost, inv_mtm, eq_curr)
                continue

            alloc_inr = min(eq_curr * MAX_POSITION_PCT, headroom, cash)
            if alloc_inr < MIN_LOT_CAPITAL_INR:
                _record_rejection(sig, sess_dt, "REJECTED_MIN_LOT_CAPITAL", f"alloc={alloc_inr:.2f}", inv_cost, inv_mtm, eq_curr)
                continue

            grp = sym_map.get(sym)
            entry_idx = sig["_entry_idx"]
            entry_open = grp.loc[entry_idx, "open"] if grp is not None and "open" in grp.columns else np.nan
            if pd.isna(entry_open) or float(entry_open) <= 0:
                _record_rejection(sig, sess_dt, "REJECTED_INVALID_ENTRY_PRICE", "missing_or_nonpositive_open", inv_cost, inv_mtm, eq_curr)
                continue

            entry_px = float(entry_open)
            meta_raw = sig.get("metadata")
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw.startswith("{") else (meta_raw or {})
            cost_pct = float(meta.get("roundtrip_delivery_cost_pct") or DEFAULT_CNC_ROUNDTRIP_COST_PCT)
            exit_idx = sig["_base_idx"] + hold_n
            exit_dt = str(grp.loc[exit_idx, "observation_date"]) if grp is not None and exit_idx < len(grp) else None

            cash -= alloc_inr
            entries_executed += 1
            weight_pct = alloc_inr / max(eq_curr, 1.0) * 100.0
            sector = _resolve_symbol_sector(sym)
            open_positions[sym] = {
                "prediction_id": pred_id,
                "symbol": sym,
                "sector": sector,
                "signal_date": obs_dt,
                "entry_date": sess_dt,
                "entry_idx": entry_idx,
                "entry_price": entry_px,
                "exit_idx": exit_idx,
                "exit_date": exit_dt,
                "capital": alloc_inr,
                "weight_pct": weight_pct,
                "cost_pct": cost_pct,
                "unrealized_pnl": 0.0,
            }

            new_inv_cost = sum(p["capital"] for p in open_positions.values())
            new_inv_mtm = new_inv_cost + sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
            eq_after = cash + new_inv_mtm
            peak_exposure_cost_pct = max(peak_exposure_cost_pct, (new_inv_cost / max(eq_after, 1.0)) * 100.0)
            peak_exposure_mtm_pct = max(peak_exposure_mtm_pct, (new_inv_mtm / max(eq_after, 1.0)) * 100.0)

            if emit_events:
                events.append(
                    AuditEvent(
                        strategy_horizon=horizon,
                        prediction_id=pred_id,
                        symbol=sym,
                        sector=sector,
                        signal_date=obs_dt,
                        event_date=sess_dt,
                        action="ENTRY_EXECUTED",
                        reason="BUY_SIGNAL_EXECUTED_AT_NEXT_OPEN",
                        prediction_timestamp=sig.get("prediction_timestamp"),
                        entry_timestamp=sig.get("_scheduled_entry_ts"),
                        entry_date=sess_dt,
                        exit_date=exit_dt,
                        entry_price=round(entry_px, 4),
                        allocated_capital_inr=round(alloc_inr, 2),
                        position_weight_cost_pct=round(weight_pct, 2),
                        position_weight_mtm_pct=round(weight_pct, 2),
                        cash_after_inr=round(cash, 2),
                        invested_cost_after_inr=round(new_inv_cost, 2),
                        invested_mtm_after_inr=round(new_inv_mtm, 2),
                        equity_after_inr=round(eq_after, 2),
                    )
                )

        # Phase 2: Market Close (15:30 IST) - Settle matured positions
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            if pos["exit_date"] is not None and sess_dt >= pos["exit_date"]:
                grp = sym_map.get(sym)
                exit_idx = pos["exit_idx"]
                if grp is not None and exit_idx < len(grp):
                    exit_px_raw = grp.loc[exit_idx, "close"]
                    if pd.notna(exit_px_raw) and float(exit_px_raw) > 0:
                        adj_factor, _ = _compute_ca_adj_factor(grp, pos["entry_idx"], exit_idx)
                        g_ret = ((float(exit_px_raw) * adj_factor) / pos["entry_price"] - 1.0) * 100.0
                        c_pct = pos["cost_pct"]
                        n_ret = g_ret - c_pct
                        g_pnl = pos["capital"] * (g_ret / 100.0)
                        c_inr = pos["capital"] * (c_pct / 100.0)
                        n_pnl = pos["capital"] * (n_ret / 100.0)

                        cash += pos["capital"] + n_pnl
                        realized_pnl += n_pnl
                        gross_pnl_total += g_pnl
                        cost_total += c_inr
                        matured_cnt += 1
                        if n_pnl > 0:
                            win_cnt += 1
                        del open_positions[sym]

                        inv_now = sum(p["capital"] for p in open_positions.values())
                        inv_mtm_now = inv_now + sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
                        eq_now = cash + inv_mtm_now
                        peak_equity = max(peak_equity, eq_now)
                        max_dd = min(max_dd, ((eq_now / max(peak_equity, 1.0)) - 1.0) * 100.0)

                        if emit_events:
                            events.append(
                                AuditEvent(
                                    strategy_horizon=horizon,
                                    prediction_id=pos["prediction_id"],
                                    symbol=sym,
                                    sector=pos["sector"],
                                    signal_date=pos["signal_date"],
                                    event_date=pos["exit_date"],
                                    action="POSITION_EXITED",
                                    reason=f"SCHEDULED_{horizon}_CLOSE_EXIT",
                                    entry_date=pos["entry_date"],
                                    exit_date=pos["exit_date"],
                                    entry_price=round(pos["entry_price"], 4),
                                    exit_price=round(float(exit_px_raw), 4),
                                    allocated_capital_inr=round(pos["capital"], 2),
                                    position_weight_cost_pct=round(pos["weight_pct"], 2),
                                    position_weight_mtm_pct=round(pos["weight_pct"], 2),
                                    gross_return_pct=round(g_ret, 4),
                                    transaction_cost_inr=round(c_inr, 2),
                                    net_return_pct=round(n_ret, 4),
                                    net_pnl_inr=round(n_pnl, 2),
                                    realized_pnl_inr=round(n_pnl, 2),
                                    cash_after_inr=round(cash, 2),
                                    invested_cost_after_inr=round(inv_now, 2),
                                    invested_mtm_after_inr=round(inv_mtm_now, 2),
                                    equity_after_inr=round(eq_now, 2),
                                    holding_sessions=hold_n,
                                )
                            )

        # Phase 3: Mark remaining open positions to market
        for sym, pos in open_positions.items():
            grp = sym_map.get(sym)
            if grp is not None:
                m_idx_list = grp.index[grp["observation_date"] == sess_dt].tolist()
                if m_idx_list:
                    m_idx = int(m_idx_list[0])
                    c_px = grp.loc[m_idx, "close"]
                    if pd.notna(c_px) and float(c_px) > 0:
                        adj_f, _ = _compute_ca_adj_factor(grp, pos["entry_idx"], m_idx)
                        unrl_ret = ((float(c_px) * adj_f) / pos["entry_price"] - 1.0) * 100.0
                        pos["unrealized_pnl"] = pos["capital"] * (unrl_ret / 100.0)

        # Phase 4: EOD Accounting & Concentration Diagnostics
        inv_cost_eod = sum(p["capital"] for p in open_positions.values())
        unrl_eod = sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
        inv_mtm_eod = inv_cost_eod + unrl_eod
        eq_eod = cash + inv_mtm_eod
        peak_equity = max(peak_equity, eq_eod)
        dd_eod = ((eq_eod / max(peak_equity, 1.0)) - 1.0) * 100.0
        max_dd = min(max_dd, dd_eod)
        peak_exposure_cost_pct = max(peak_exposure_cost_pct, (inv_cost_eod / max(eq_eod, 1.0)) * 100.0)
        peak_exposure_mtm_pct = max(peak_exposure_mtm_pct, (inv_mtm_eod / max(eq_eod, 1.0)) * 100.0)

        sec_totals: dict[str, float] = {}
        for pos in open_positions.values():
            pos_mtm = pos["capital"] + pos.get("unrealized_pnl", 0.0)
            w_sym = (pos_mtm / max(eq_eod, 1.0)) * 100.0
            max_symbol_conc_pct = max(max_symbol_conc_pct, w_sym)
            sec = pos["sector"]
            sec_totals[sec] = sec_totals.get(sec, 0.0) + pos_mtm
        for sec_val in sec_totals.values():
            max_sector_conc_pct = max(max_sector_conc_pct, (sec_val / max(eq_eod, 1.0)) * 100.0)

        if sess_dt in snap_dates or open_positions or entries_executed > 0:
            daily_equity_curve.append(
                {
                    "date": sess_dt,
                    "cash_inr": round(cash, 2),
                    "invested_cost_inr": round(inv_cost_eod, 2),
                    "invested_mtm_inr": round(inv_mtm_eod, 2),
                    "unrealized_pnl_inr": round(unrl_eod, 2),
                    "realized_pnl_inr": round(realized_pnl, 2),
                    "equity_inr": round(eq_eod, 2),
                    "drawdown_pct": round(dd_eod, 4),
                    "open_positions": len(open_positions),
                }
            )

        # Phase 5: Post-Close - Record SIGNAL_RECEIVED and schedule next open (t+1)
        for sig in signals_by_obs.get(sess_dt, []):
            signals_received += 1
            sym = str(sig["symbol"]).upper()
            seen_signal_days.add((sym, sess_dt))
            pred_id = str(sig["prediction_id"])
            rec = str(signal_getter(sig) or "UNAVAILABLE")
            if emit_events:
                events.append(
                    AuditEvent(
                        strategy_horizon=horizon,
                        prediction_id=pred_id,
                        symbol=sym,
                        sector=_resolve_symbol_sector(sym),
                        signal_date=sess_dt,
                        event_date=sess_dt,
                        action="SIGNAL_RECEIVED",
                        reason=f"SIGNAL_{rec}",
                        raw_reason=rec,
                        prediction_timestamp=sig.get("prediction_timestamp"),
                        entry_timestamp=sig.get("_scheduled_entry_ts"),
                        cash_after_inr=round(cash, 2),
                        invested_cost_after_inr=round(inv_cost_eod, 2),
                        invested_mtm_after_inr=round(inv_mtm_eod, 2),
                        equity_after_inr=round(eq_eod, 2),
                    )
                )

            grp = sym_map.get(sym)
            if grp is None or grp.empty:
                _record_rejection(
                    sig,
                    sess_dt,
                    "REJECTED_MISSING_OBSERVATION_BAR",
                    "no_market_history_for_symbol",
                    inv_cost_eod,
                    inv_mtm_eod,
                    eq_eod,
                )
                continue

            matches = grp.index[grp["observation_date"] == sess_dt].tolist()
            if matches:
                b_idx = int(matches[0])
                if b_idx + 1 < len(grp):
                    entry_dt = str(grp.loc[b_idx + 1, "observation_date"])
                    sig_copy = dict(sig)
                    sig_copy["_base_idx"] = b_idx
                    sig_copy["_entry_idx"] = b_idx + 1
                    pending_entries_by_date.setdefault(entry_dt, []).append(sig_copy)
                else:
                    pending_next_session_cnt += 1
                    rejection_counts["PENDING_NEXT_SESSION"] = rejection_counts.get("PENDING_NEXT_SESSION", 0) + 1
            else:
                after_mask = grp["observation_date"] > sess_dt
                if after_mask.any():
                    # Observation date was a holiday/missing bar; next trading bar acts as entry
                    e_idx = int(grp.index[after_mask][0])
                    entry_dt = str(grp.loc[e_idx, "observation_date"])
                    sig_copy = dict(sig)
                    sig_copy["_base_idx"] = e_idx - 1
                    sig_copy["_entry_idx"] = e_idx
                    pending_entries_by_date.setdefault(entry_dt, []).append(sig_copy)
                else:
                    pending_next_session_cnt += 1
                    rejection_counts["PENDING_NEXT_SESSION"] = rejection_counts.get("PENDING_NEXT_SESSION", 0) + 1

    inv_cost_final = sum(p["capital"] for p in open_positions.values())
    unrl_final = sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
    inv_mtm_final = inv_cost_final + unrl_final
    eq_final = cash + inv_mtm_final
    accrued_exit_cost_est = sum(p["capital"] * (p["cost_pct"] / 100.0) for p in open_positions.values())

    final_sector_pct: dict[str, float] = {}
    for pos in open_positions.values():
        pos_mtm = pos["capital"] + pos.get("unrealized_pnl", 0.0)
        sec = pos["sector"]
        final_sector_pct[sec] = round(final_sector_pct.get(sec, 0.0) + (pos_mtm / max(eq_final, 1.0)) * 100.0, 2)

    if emit_events:
        last_dt = all_sessions[-1] if all_sessions else ""
        for sym, pos in open_positions.items():
            pos_mtm = pos["capital"] + pos.get("unrealized_pnl", 0.0)
            events.append(
                AuditEvent(
                    strategy_horizon=horizon,
                    prediction_id=pos["prediction_id"],
                    symbol=sym,
                    sector=pos["sector"],
                    signal_date=pos["signal_date"],
                    event_date=last_dt or pos["entry_date"],
                    action="POSITION_OPEN",
                    reason="AWAITING_SCHEDULED_EXIT",
                    entry_date=pos["entry_date"],
                    exit_date=pos["exit_date"],
                    entry_price=round(pos["entry_price"], 4),
                    allocated_capital_inr=round(pos["capital"], 2),
                    position_weight_cost_pct=round((pos["capital"] / max(eq_final, 1.0)) * 100.0, 2),
                    position_weight_mtm_pct=round((pos_mtm / max(eq_final, 1.0)) * 100.0, 2),
                    unrealized_pnl_inr=round(pos.get("unrealized_pnl", 0.0), 2),
                    cash_after_inr=round(cash, 2),
                    invested_cost_after_inr=round(inv_cost_final, 2),
                    invested_mtm_after_inr=round(inv_mtm_final, 2),
                    equity_after_inr=round(eq_final, 2),
                )
            )

    err1 = abs(eq_final - (cash + inv_cost_final + unrl_final))
    err2 = abs(eq_final - (STARTING_CAPITAL_INR + realized_pnl + unrl_final))
    inv_err = round(max(err1, err2), 6)

    summary = AuditStrategyResult(
        horizon=horizon,
        starting_capital_inr=STARTING_CAPITAL_INR,
        ending_capital_inr=round(eq_final, 2),
        cash_inr=round(cash, 2),
        invested_cost_inr=round(inv_cost_final, 2),
        invested_mtm_inr=round(inv_mtm_final, 2),
        realized_pnl_inr=round(realized_pnl, 2),
        unrealized_pnl_inr=round(unrl_final, 2),
        accrued_exit_cost_est_inr=round(accrued_exit_cost_est, 2),
        net_return_pct=round((eq_final / STARTING_CAPITAL_INR - 1.0) * 100.0, 2),
        gross_return_pct=round(((gross_pnl_total + unrl_final) / STARTING_CAPITAL_INR) * 100.0, 2),
        total_cost_inr=round(cost_total, 2),
        max_drawdown_pct=round(max_dd, 2),
        peak_exposure_cost_pct=round(peak_exposure_cost_pct, 2),
        peak_exposure_mtm_pct=round(peak_exposure_mtm_pct, 2),
        max_symbol_concentration_pct=round(max_symbol_conc_pct, 2),
        max_sector_concentration_pct=round(max_sector_conc_pct, 2),
        sector_exposure_pct=final_sector_pct,
        signals_received=signals_received,
        unique_signal_days=len(seen_signal_days),
        entries_executed=entries_executed,
        open_positions_count=len(open_positions),
        matured_positions_count=matured_cnt,
        rejected_signals=rejected_cnt,
        pending_next_session_count=pending_next_session_cnt,
        rejection_counts=rejection_counts,
        win_rate_pct=round(win_cnt / matured_cnt * 100.0, 2) if matured_cnt > 0 else None,
        accounting_invariant_ok=inv_err < 0.05 and cash >= -1e-6,
        cash_conservation_error_inr=inv_err,
        status="ACTIVE" if entries_executed > 0 else "INSUFFICIENT_DATA",
        daily_equity_curve=daily_equity_curve,
    )
    return summary, events


def _compute_nifty_benchmark(
    indices_df: pd.DataFrame | None,
    first_signal_date: str | None,
    first_entry_date: str | None,
) -> dict[str, Any]:
    """Compute Nifty 50 buy-and-hold benchmark handling both `NIFTY_50` and `NIFTY 50`."""
    unavailable = {
        "index_code_matched": None,
        "entry_date": None,
        "end_date": None,
        "net_return_pct": None,
        "max_drawdown_pct": None,
        "ending_capital_inr": None,
        "status": "UNAVAILABLE",
    }
    if indices_df is None or indices_df.empty or not first_signal_date:
        return unavailable

    codes = indices_df["index_code"].astype(str).str.upper()
    n50 = indices_df[codes.isin(NIFTY_INDEX_CODES)].copy()
    if n50.empty:
        return unavailable

    matched_code = str(n50["index_code"].iloc[0])
    n50["observation_date"] = n50["observation_date"].astype(str)
    start_ref = first_entry_date or first_signal_date
    n50 = n50[n50["observation_date"] >= start_ref].sort_values("observation_date").reset_index(drop=True)
    if n50.empty:
        return unavailable

    # Next-open comparable entry: enter at open[first_entry_date] (fallback close if open missing)
    open_0 = pd.to_numeric(n50.get("open"), errors="coerce").iloc[0] if "open" in n50.columns else np.nan
    closes = pd.to_numeric(n50["close"], errors="coerce").dropna().values
    if len(closes) == 0:
        return unavailable
    entry_px = float(open_0) if pd.notna(open_0) and float(open_0) > 0 else float(closes[0])
    if entry_px <= 0:
        return unavailable

    invested = STARTING_CAPITAL_INR * MAX_EXPOSURE_PCT
    cash_res = STARTING_CAPITAL_INR - invested
    cost_drag = invested * (0.05 / 100.0)
    eq_curve = cash_res + (invested - cost_drag) * (closes / entry_px)
    peak = np.maximum.accumulate(np.insert(eq_curve, 0, STARTING_CAPITAL_INR))[1:]
    dd = float(((eq_curve / np.maximum(peak, 1e-9)) - 1.0).min() * 100.0)
    return {
        "index_code_matched": matched_code,
        "entry_date": str(n50["observation_date"].iloc[0]),
        "end_date": str(n50["observation_date"].iloc[-1]),
        "entry_price": round(entry_px, 2),
        "end_close": round(float(closes[-1]), 2),
        "net_return_pct": round(float((eq_curve[-1] / STARTING_CAPITAL_INR - 1.0) * 100.0), 2),
        "max_drawdown_pct": round(dd, 2),
        "ending_capital_inr": round(float(eq_curve[-1]), 2),
        "status": "ACTIVE",
    }


def audit_paper_trades_persistence(
    paper_trades_df: pd.DataFrame | None,
    snapshots_df: pd.DataFrame | None,
    production_events: list[AuditEvent],
    truncated_limit: int = 60,
) -> PersistenceAuditSummary | None:
    """Compare stored `paper_trades.parquet` rows against canonical simulation events."""
    if paper_trades_df is None:
        return None
    pt = paper_trades_df.copy()
    dup_pids: set[str] = set()
    if snapshots_df is not None and not snapshots_df.empty and "is_duplicate_run" in snapshots_df.columns:
        dup_pids = set(snapshots_df[snapshots_df["is_duplicate_run"].fillna(False).astype(bool)]["prediction_id"].astype(str))

    legacy_cnt = int((pt["action"] == "REJECTED").sum()) if not pt.empty and "action" in pt.columns else 0
    dup_in_pt = int(pt["prediction_id"].astype(str).isin(dup_pids).sum()) if not pt.empty and dup_pids else 0
    action_counts = {str(k): int(v) for k, v in pt["action"].value_counts(dropna=False).items()} if not pt.empty else {}

    total_ev = len(production_events)
    dropped = max(0, total_ev - truncated_limit)
    dropped_slice = production_events[:-truncated_limit] if total_ev > truncated_limit else []
    dropped_by_h: dict[str, int] = {"1D": 0, "5D": 0, "10D": 0}
    for e in dropped_slice:
        dropped_by_h[e.strategy_horizon] = dropped_by_h.get(e.strategy_horizon, 0) + 1

    return PersistenceAuditSummary(
        parquet_rows_total=len(pt),
        legacy_action_rejected_rows=legacy_cnt,
        duplicate_run_rows_in_parquet=dup_in_pt,
        full_unfiltered_simulated_events=total_ev,
        truncated_recent_trades_limit=truncated_limit,
        dropped_by_truncation_count=dropped,
        dropped_by_horizon=dropped_by_h,
        action_counts_in_parquet=action_counts,
    )


def run_paper_portfolio_audit(
    snapshots_df: pd.DataFrame,
    equities_df: pd.DataFrame | None = None,
    indices_df: pd.DataFrame | None = None,
    paper_trades_df: pd.DataFrame | None = None,
    audit_mode: Literal["strict_canonical", "production_replay"] = "strict_canonical",
) -> PortfolioAuditReport:
    """Execute a full, read-only audit of the paper research portfolio.

    Modes:
    - `strict_canonical`: filters `is_duplicate_run`, enforces `prediction_timestamp < entry_open`,
      and keeps only the earliest valid canonical session per `(symbol, observation_date)`.
    - `production_replay`: mirrors `forward_validation.py` signal intake (filters `is_duplicate_run`
      only, allowing multiple sessions/day and intraday look-ahead timestamps) while producing full
      rejection taxonomy, cost-basis vs MTM diagnostics, and aligned benchmarks.
    """
    enforce_no_lookahead = audit_mode == "strict_canonical"
    dedup_same_day = audit_mode == "strict_canonical"

    sym_map = _prepare_sym_map(equities_df)
    canon_df, total_snaps, canon_cnt, dup_cnt, eff_daily = _classify_canonical_signals(
        snapshots_df,
        sym_map=sym_map,
        enforce_no_lookahead=enforce_no_lookahead,
        dedup_same_day_sessions=dedup_same_day,
    )
    lookahead_cnt = int(canon_df["_is_lookahead"].sum()) if not canon_df.empty else 0

    strategies: dict[str, AuditStrategyResult] = {}
    all_events: list[AuditEvent] = []
    for horizon, hold_n in (("1D", 1), ("5D", 5), ("10D", 10)):
        rec_col = {"1D": "recommendation_1d", "5D": "recommendation_5d", "10D": "recommendation_10d"}[horizon]
        res, evts = _simulate_audited_horizon(
            horizon=horizon,  # type: ignore[arg-type]
            hold_n=hold_n,
            canon_df=canon_df,
            sym_map=sym_map,
            signal_getter=lambda s, rc=rec_col: str(s.get(rc) or "UNAVAILABLE"),
            enforce_no_lookahead=enforce_no_lookahead,
            dedup_same_day_sessions=dedup_same_day,
            emit_events=True,
        )
        strategies[horizon] = res
        all_events.extend(evts)

    # Horizon-matched Naive Momentum benchmarks
    def _mom_signal_factory(col_primary: str, col_fallback: str | None):
        def _sig(s: dict[str, Any]) -> str:
            p = s.get(col_primary)
            if (p is None or (isinstance(p, float) and np.isnan(p))) and col_fallback:
                p = s.get(col_fallback)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return "UNAVAILABLE"
            return "BUY" if float(p) >= 0.535 else "HOLD"

        return _sig

    mom_benchmarks: dict[str, dict[str, Any]] = {}
    for h, n, col, fb in (
        ("1D", 1, "benchmark_prob_1d_up", "benchmark_prob_up"),
        ("5D", 5, "benchmark_prob_5d_up", None),
        ("10D", 10, "benchmark_prob_10d_up", None),
    ):
        if canon_df.empty or (col not in canon_df.columns and (fb is None or fb not in canon_df.columns)):
            mom_benchmarks[h] = {
                "net_return_pct": None,
                "max_drawdown_pct": None,
                "ending_capital_inr": None,
                "entries_executed": 0,
                "status": "INSUFFICIENT_DATA",
            }
        else:
            m_res, _ = _simulate_audited_horizon(
                horizon=h,  # type: ignore[arg-type]
                hold_n=n,
                canon_df=canon_df,
                sym_map=sym_map,
                signal_getter=_mom_signal_factory(col, fb),
                enforce_no_lookahead=enforce_no_lookahead,
                dedup_same_day_sessions=dedup_same_day,
                emit_events=False,
            )
            ok = m_res.entries_executed > 0
            mom_benchmarks[h] = {
                "net_return_pct": m_res.net_return_pct if ok else None,
                "max_drawdown_pct": m_res.max_drawdown_pct if ok else None,
                "ending_capital_inr": m_res.ending_capital_inr if ok else None,
                "entries_executed": m_res.entries_executed,
                "open_positions_count": m_res.open_positions_count,
                "status": "ACTIVE" if ok else "INSUFFICIENT_DATA",
            }

    first_sig_dt = str(canon_df["observation_date"].min()) if not canon_df.empty else None
    first_entry_dt = None
    if not canon_df.empty and "_scheduled_entry_date" in canon_df.columns:
        valid_entries = canon_df["_scheduled_entry_date"].dropna()
        if not valid_entries.empty:
            first_entry_dt = str(valid_entries.min())

    nifty_bench = _compute_nifty_benchmark(indices_df, first_sig_dt, first_entry_dt)

    benchmarks: dict[str, dict[str, Any]] = {
        "naive_momentum_1d": mom_benchmarks["1D"],
        "naive_momentum_5d": mom_benchmarks["5D"],
        "naive_momentum_10d": mom_benchmarks["10D"],
        "nifty_50_buy_hold": nifty_bench,
        "cash_no_trade": {
            "net_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "ending_capital_inr": STARTING_CAPITAL_INR,
            "status": "ACTIVE",
        },
    }

    # Persistence audit (always check against production_replay event count if paper_trades_df provided)
    persistence_summary = None
    if paper_trades_df is not None:
        prod_canon, _, _, _, _ = _classify_canonical_signals(
            snapshots_df,
            sym_map=sym_map,
            enforce_no_lookahead=False,
            dedup_same_day_sessions=False,
        )
        prod_events: list[AuditEvent] = []
        for h, n in (("1D", 1), ("5D", 5), ("10D", 10)):
            rc = {"1D": "recommendation_1d", "5D": "recommendation_5d", "10D": "recommendation_10d"}[h]
            _, ev = _simulate_audited_horizon(
                horizon=h,  # type: ignore[arg-type]
                hold_n=n,
                canon_df=prod_canon,
                sym_map=sym_map,
                signal_getter=lambda s, rc=rc: str(s.get(rc) or "UNAVAILABLE"),
                enforce_no_lookahead=False,
                dedup_same_day_sessions=False,
                emit_events=True,
            )
            prod_events.extend(ev)
        persistence_summary = audit_paper_trades_persistence(paper_trades_df, snapshots_df, prod_events)

    findings: list[str] = []
    if lookahead_cnt > 0:
        findings.append(
            f"Look-ahead timing detected on {lookahead_cnt}/{canon_cnt} canonical snapshots where "
            f"prediction_timestamp >= scheduled 09:15 IST entry open."
        )
    if nifty_bench["status"] == "ACTIVE" and indices_df is not None:
        if (indices_df["index_code"].astype(str).str.upper() == "NIFTY_50").any():
            findings.append(
                "Production forward_validation.py queries index_code == 'NIFTY 50', missing 'NIFTY_50' rows in indices_daily.parquet."
            )
    if persistence_summary and persistence_summary.dropped_by_truncation_count > 0:
        findings.append(
            f"Production upsert_paper_trades receives only all_trades[-60:], dropping {persistence_summary.dropped_by_truncation_count} "
            f"of {persistence_summary.full_unfiltered_simulated_events} events (all 1D and most 5D)."
        )
    if persistence_summary and persistence_summary.duplicate_run_rows_in_parquet > 0:
        findings.append(
            f"paper_trades.parquet retains {persistence_summary.duplicate_run_rows_in_parquet} rows linked to is_duplicate_run=True snapshots."
        )

    return PortfolioAuditReport(
        generated_at=datetime.now(IST_TZ).isoformat(),
        audit_mode=audit_mode,
        total_snapshots=total_snaps,
        canonical_snapshots=canon_cnt,
        duplicate_run_snapshots=dup_cnt,
        dedup_daily_session_snapshots=eff_daily,
        lookahead_flagged_snapshots=lookahead_cnt,
        strategies=strategies,
        benchmarks=benchmarks,
        events=all_events,
        persistence_audit=persistence_summary,
        findings=findings,
    )


def run_audit_from_data_dir(
    data_dir: Path | str = Path("data"),
    audit_mode: Literal["strict_canonical", "production_replay"] = "strict_canonical",
) -> PortfolioAuditReport:
    """Convenience loader for running the read-only audit on persisted Parquet files."""
    d = Path(data_dir)
    snaps = pd.read_parquet(d / "predictions" / "forward_predictions.parquet")
    eq = pd.read_parquet(d / "processed" / "equities_daily.parquet")
    idx = pd.read_parquet(d / "processed" / "indices_daily.parquet")
    pt_path = d / "predictions" / "paper_trades.parquet"
    pt = pd.read_parquet(pt_path) if pt_path.exists() else None
    return run_paper_portfolio_audit(
        snapshots_df=snaps,
        equities_df=eq,
        indices_df=idx,
        paper_trades_df=pt,
        audit_mode=audit_mode,
    )
