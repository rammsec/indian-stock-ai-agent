"""Chronological, Event-Ledger-Based Paper Portfolio Simulator (Remediation Workstream 4).

Provides a deterministic, zero-lookahead, immutable-event-ledger paper portfolio
simulator (`ledger_version="ledger_v1.0.0"`) decoupled from frozen prediction and
signal-generation logic (`backend/news_intel/forward_validation.py` and
`backend/news_intel/combiner.py` are never modified).

Key Remediation Guarantees:
1. Strict prediction-before-entry & hygiene enforcement:
   - Rejects `is_duplicate_run == True` snapshots with `REJECTED_DUPLICATE_RUN`.
   - Rejects look-ahead snapshots (`prediction_timestamp >= scheduled_entry_open_0915_ist`
     or missing/unparseable timestamp) with `REJECTED_LOOKAHEAD`.
   - Deduplicates multiple valid canonical sessions on the same `(symbol, observation_date)`,
     retaining the earliest valid canonical run and rejecting superseded same-day runs
     with `REJECTED_SAME_DAY_DUPLICATE`.
2. Full immutable event ledger with zero truncation (no `[-60:]` drop):
   - Event types: `SIGNAL_RECEIVED`, `SIGNAL_REJECTED`, `ENTRY_EXECUTED`,
     `POSITION_MTM`, `POSITION_EXITED`.
   - Every event carries monotonic `event_seq`, deterministic `event_id`, and
     SHA-256 hash chain (`prev_event_hash` -> `event_hash`).
   - Full state reconstruction via `replay_strategy_from_events` and
     `replay_portfolio_from_events` reproduces exact cash, positions, MTM,
     accrued CNC exit costs, equity curves, and drawdowns purely from ledger events.
3. Exact accounting & accrued cost reconciliation at every step:
   - `cash_after_inr + mtm_value_after_inr == total_equity_after_inr`
   - `mtm_value_after_inr == invested_cost_after_inr + unrealized_pnl_total_after_inr`
   - `total_equity_after_inr == starting_capital + realized_pnl_total + unrealized_pnl_total`
   - `net_liquidation_equity_after_inr == total_equity_after_inr - accrued_exit_cost_after_inr`
   - Separates cost-basis exposure (`peak_exposure_cost_pct`) from mark-to-market
     exposure (`peak_exposure_mtm_pct`).
4. Integrated aligned benchmarks on identical entry/exit dates:
   - `nifty_50_buy_hold` and per-horizon `nifty_50_matched_windows` using
     normalized index matching (`"NIFTY_50"` and `"NIFTY 50"` -> `"NIFTY_50"`).
   - `naive_momentum_1d`, `naive_momentum_5d`, `naive_momentum_10d` simulated
     through the exact same event-ledger engine.
   - `cash_no_trade` aligned on identical session dates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import IST_TZ, is_nse_trading_day
from ..news_intel.relationships import RELATIONSHIP_MAP, get_pilot_relationship

LEDGER_VERSION = "ledger_v1.0.0"
STARTING_CAPITAL_INR = 1_000_000.0
MAX_POSITION_PCT = 0.10
MAX_EXPOSURE_PCT = 0.80
MIN_LOT_CAPITAL_INR = 1_000.0
DEFAULT_CNC_ROUNDTRIP_COST_PCT = 0.347
NIFTY_CANONICAL_CODE = "NIFTY_50"
NIFTY_INDEX_MATCH_CODES = ("NIFTY_50", "NIFTY 50")
NIFTY_BENCHMARK_COST_PCT = 0.05

LedgerAction = Literal[
    "SIGNAL_RECEIVED",
    "SIGNAL_REJECTED",
    "ENTRY_EXECUTED",
    "POSITION_MTM",
    "POSITION_EXITED",
]

HorizonType = Literal["1D", "5D", "10D"]


@dataclass(frozen=True)
class LedgerEvent:
    """Immutable, hash-chained event in the paper portfolio ledger."""

    ledger_version: str
    event_seq: int
    event_id: str
    strategy_name: str
    strategy_horizon: HorizonType
    prediction_id: str
    symbol: str
    sector: str
    signal_date: str
    event_date: str
    action: LedgerAction
    reason: str
    raw_reason: str = ""
    prediction_timestamp: str | None = None
    entry_timestamp: str | None = None
    entry_date: str | None = None
    exit_date: str | None = None
    entry_price: float | None = None
    mark_price: float | None = None
    exit_price: float | None = None
    ca_adjustment_factor: float = 1.0
    allocated_capital_inr: float = 0.0
    cost_pct: float = 0.0
    position_mtm_value_inr: float = 0.0
    position_weight_cost_pct: float = 0.0
    position_weight_mtm_pct: float = 0.0
    gross_return_pct: float | None = None
    gross_pnl_inr: float = 0.0
    transaction_cost_inr: float = 0.0
    accrued_exit_cost_inr: float = 0.0
    net_return_pct: float | None = None
    net_pnl_inr: float = 0.0
    realized_pnl_delta_inr: float = 0.0
    unrealized_pnl_inr: float = 0.0
    cash_after_inr: float = STARTING_CAPITAL_INR
    invested_cost_after_inr: float = 0.0
    mtm_value_after_inr: float = 0.0
    total_equity_after_inr: float = STARTING_CAPITAL_INR
    accrued_exit_cost_after_inr: float = 0.0
    net_liquidation_equity_after_inr: float = STARTING_CAPITAL_INR
    realized_pnl_total_after_inr: float = 0.0
    unrealized_pnl_total_after_inr: float = 0.0
    holding_sessions: int | None = None
    prev_event_hash: str = ""
    event_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyLedgerResult:
    """Reconciled portfolio state for one strategy and horizon."""

    ledger_version: str
    strategy_name: str
    horizon: HorizonType
    starting_capital_inr: float = STARTING_CAPITAL_INR
    ending_equity_inr: float = STARTING_CAPITAL_INR
    net_liquidation_equity_inr: float = STARTING_CAPITAL_INR
    cash_inr: float = STARTING_CAPITAL_INR
    invested_cost_inr: float = 0.0
    mtm_value_inr: float = 0.0
    realized_pnl_inr: float = 0.0
    unrealized_pnl_inr: float = 0.0
    accrued_exit_cost_inr: float = 0.0
    total_cost_inr: float = 0.0
    gross_pnl_inr: float = 0.0
    net_return_pct: float = 0.0
    net_liquidation_return_pct: float = 0.0
    gross_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_exposure_cost_pct: float = 0.0
    peak_exposure_mtm_pct: float = 0.0
    max_symbol_concentration_pct: float = 0.0
    max_sector_concentration_pct: float = 0.0
    total_input_signals: int = 0
    canonical_signals_received: int = 0
    entries_executed: int = 0
    open_positions_count: int = 0
    matured_positions_count: int = 0
    rejected_signals: int = 0
    pending_next_session_count: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    win_rate_pct: float | None = None
    accounting_invariant_ok: bool = True
    max_reconciliation_error_inr: float = 0.0
    ledger_chain_valid: bool = True
    status: Literal["ACTIVE", "INSUFFICIENT_DATA"] = "INSUFFICIENT_DATA"
    daily_equity_curve: list[dict[str, Any]] = field(default_factory=list)
    events: list[LedgerEvent] = field(default_factory=list)
    open_positions: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["events_count"] = len(self.events)
        d.pop("events", None)
        return d


@dataclass
class EventLedgerPortfolioReport:
    """Complete remediated paper portfolio report with benchmarks and full event ledger."""

    ledger_version: str
    generated_at: str
    strict_canonical: bool
    total_input_snapshots: int
    duplicate_run_snapshots: int
    canonical_snapshots: int
    lookahead_rejected_snapshots: int
    same_day_duplicate_snapshots: int
    strategies: dict[str, StrategyLedgerResult]
    benchmarks: dict[str, Any]
    all_events: list[LedgerEvent]
    reconciliation_checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "generated_at": self.generated_at,
            "strict_canonical": self.strict_canonical,
            "total_input_snapshots": self.total_input_snapshots,
            "duplicate_run_snapshots": self.duplicate_run_snapshots,
            "canonical_snapshots": self.canonical_snapshots,
            "lookahead_rejected_snapshots": self.lookahead_rejected_snapshots,
            "same_day_duplicate_snapshots": self.same_day_duplicate_snapshots,
            "strategies": {h: s.to_summary_dict() for h, s in self.strategies.items()},
            "benchmarks": self.benchmarks,
            "events_count": len(self.all_events),
            "reconciliation_checks": self.reconciliation_checks,
        }


def normalize_index_code(code: Any) -> str:
    """Normalize index codes so both 'NIFTY 50' and 'NIFTY_50' resolve to 'NIFTY_50'."""
    return str(code or "").strip().upper().replace(" ", "_")


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


def _next_nse_trading_day_str(obs_date: str) -> str:
    d = date.fromisoformat(obs_date[:10]) + timedelta(days=1)
    for _ in range(30):
        if is_nse_trading_day(d):
            return d.isoformat()
        d += timedelta(days=1)
    return d.isoformat()


def _resolve_symbol_sector(symbol: str) -> str:
    sym = symbol.upper().strip()
    if sym in RELATIONSHIP_MAP:
        return str(RELATIONSHIP_MAP[sym].get("sector") or "Unknown")
    return str(get_pilot_relationship(sym).get("sector") or "Unknown")


def _parse_cost_pct(sig: dict[str, Any]) -> float:
    meta_raw = sig.get("metadata")
    if isinstance(meta_raw, dict):
        meta = meta_raw
    elif isinstance(meta_raw, str) and meta_raw.strip().startswith("{"):
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
    else:
        meta = {}
    val = meta.get("roundtrip_delivery_cost_pct")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return DEFAULT_CNC_ROUNDTRIP_COST_PCT
    return float(val)


def _compute_ca_adj_factor(grp: pd.DataFrame, entry_idx: int, exit_idx: int) -> float:
    if "adj_close" not in grp.columns:
        return 1.0
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
            return float(ratio_out / ratio_in)
    return 1.0


def _prepare_sym_map(equities_df: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if equities_df is None or equities_df.empty:
        return {}
    cols = [c for c in ("symbol", "observation_date", "open", "close", "adj_close") if c in equities_df.columns]
    eq = equities_df[cols].copy()
    eq["symbol"] = eq["symbol"].astype(str).str.upper().str.strip()
    eq["observation_date"] = eq["observation_date"].astype(str).str.strip()
    eq = eq.drop_duplicates(subset=["symbol", "observation_date"], keep="last")
    eq = eq.sort_values(["symbol", "observation_date"]).reset_index(drop=True)
    return {str(sym): grp.reset_index(drop=True) for sym, grp in eq.groupby("symbol", sort=False)}


def _normalize_hash_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        return round(float(v), 10)
    return v


def compute_event_hash(event_fields: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of an immutable event payload (excluding `event_hash`)."""
    payload = {
        k: _normalize_hash_value(v)
        for k, v in event_fields.items()
        if k != "event_hash"
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_hashed_event(prev_hash: str, **kwargs: Any) -> LedgerEvent:
    fields = dict(kwargs)
    fields["prev_event_hash"] = prev_hash
    fields["event_hash"] = ""
    # Ensure float fields are float even if passed as 0
    for k in (
        "ca_adjustment_factor",
        "allocated_capital_inr",
        "cost_pct",
        "position_mtm_value_inr",
        "position_weight_cost_pct",
        "position_weight_mtm_pct",
        "gross_pnl_inr",
        "transaction_cost_inr",
        "accrued_exit_cost_inr",
        "net_pnl_inr",
        "realized_pnl_delta_inr",
        "unrealized_pnl_inr",
        "cash_after_inr",
        "invested_cost_after_inr",
        "mtm_value_after_inr",
        "total_equity_after_inr",
        "accrued_exit_cost_after_inr",
        "net_liquidation_equity_after_inr",
        "realized_pnl_total_after_inr",
        "unrealized_pnl_total_after_inr",
    ):
        if k in fields and fields[k] is not None:
            fields[k] = float(fields[k])
    e_hash = compute_event_hash(fields)
    fields["event_hash"] = e_hash
    return LedgerEvent(**fields)


def _classify_signals(
    snapshots_df: pd.DataFrame,
    sym_map: dict[str, pd.DataFrame],
    strict_canonical: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify every snapshot deterministically for duplicate runs, look-ahead, and same-day dedup."""
    stats = {
        "total_input_snapshots": len(snapshots_df),
        "duplicate_run_snapshots": 0,
        "canonical_snapshots": 0,
        "lookahead_rejected_snapshots": 0,
        "same_day_duplicate_snapshots": 0,
    }
    if snapshots_df.empty:
        return snapshots_df.copy(), stats

    df = snapshots_df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["observation_date"] = df["observation_date"].astype(str).str.strip()
    if "prediction_timestamp" not in df.columns:
        df["prediction_timestamp"] = None
    if "is_duplicate_run" not in df.columns:
        df["is_duplicate_run"] = False
    df["_is_dup_run"] = df["is_duplicate_run"].fillna(False).astype(bool)

    df = df.sort_values(
        ["observation_date", "symbol", "_is_dup_run", "prediction_timestamp", "prediction_id"],
        na_position="last",
    ).reset_index(drop=True)

    # Collapse exact duplicate prediction_id rows if caller passed raw un-deduped frames
    seen_pids: set[str] = set()
    for idx, r in df.iterrows():
        pid = str(r["prediction_id"])
        if pid in seen_pids:
            df.at[idx, "_is_dup_run"] = True
        else:
            seen_pids.add(pid)

    stats["duplicate_run_snapshots"] = int(df["_is_dup_run"].sum())
    stats["canonical_snapshots"] = int((~df["_is_dup_run"]).sum())

    all_eq_dates = sorted({d for grp in sym_map.values() for d in grp["observation_date"].tolist()})

    sched_entry_dates: list[str | None] = []
    sched_entry_ts: list[str | None] = []
    has_obs_bar: list[bool] = []
    has_entry_bar: list[bool] = []
    is_lookahead: list[bool] = []
    lookahead_reason: list[str] = []

    for _, r in df.iterrows():
        sym = str(r["symbol"])
        obs_dt = str(r["observation_date"])
        grp = sym_map.get(sym)

        obs_found = False
        ent_dt: str | None = None
        ent_bar_found = False

        if grp is not None and not grp.empty:
            matches = grp.index[grp["observation_date"] == obs_dt].tolist()
            if matches:
                obs_found = True
                b_idx = int(matches[0])
                if b_idx + 1 < len(grp):
                    ent_dt = str(grp.loc[b_idx + 1, "observation_date"])
                    ent_bar_found = True
            else:
                after = grp[grp["observation_date"] > obs_dt]
                if not after.empty:
                    ent_dt = str(after.iloc[0]["observation_date"])
                    ent_bar_found = True
        elif all_eq_dates:
            after_all = [d for d in all_eq_dates if d > obs_dt]
            if after_all:
                ent_dt = after_all[0]

        est_ent_dt = ent_dt or _next_nse_trading_day_str(obs_dt)
        ent_ts = _entry_open_timestamp_ist(est_ent_dt)

        pred_dt = _parse_ist_timestamp(r.get("prediction_timestamp"))
        if pred_dt is None:
            la = True
            la_msg = "missing_or_unparseable_prediction_timestamp"
        elif pred_dt >= ent_ts:
            la = True
            la_msg = f"prediction_timestamp={pred_dt.isoformat()} >= entry_open={ent_ts.isoformat()}"
        else:
            la = False
            la_msg = ""

        has_obs_bar.append(obs_found)
        has_entry_bar.append(ent_bar_found)
        sched_entry_dates.append(ent_dt)
        sched_entry_ts.append(ent_ts.isoformat())
        is_lookahead.append(la)
        lookahead_reason.append(la_msg)

    df["_has_obs_bar"] = has_obs_bar
    df["_has_entry_bar"] = has_entry_bar
    df["_scheduled_entry_date"] = sched_entry_dates
    df["_scheduled_entry_ts"] = sched_entry_ts
    df["_is_lookahead"] = is_lookahead
    df["_lookahead_reason"] = lookahead_reason

    is_same_day_dup: list[bool] = [False] * len(df)
    if strict_canonical:
        kept_keys: set[tuple[str, str]] = set()
        for idx, r in df.iterrows():
            if bool(r["_is_dup_run"]):
                continue
            if bool(r["_is_lookahead"]):
                continue
            key = (str(r["symbol"]), str(r["observation_date"]))
            if key not in kept_keys:
                kept_keys.add(key)
            else:
                is_same_day_dup[idx] = True

    df["_is_same_day_dup"] = is_same_day_dup
    stats["lookahead_rejected_snapshots"] = int((~df["_is_dup_run"] & df["_is_lookahead"]).sum()) if strict_canonical else 0
    stats["same_day_duplicate_snapshots"] = int(sum(is_same_day_dup))
    return df, stats


def _reconciliation_error(
    cash: float,
    inv_cost: float,
    mtm_val: float,
    unrl_pnl: float,
    realized_pnl: float,
    total_eq: float,
    accrued_cost: float,
    net_liq_eq: float,
    starting_capital: float,
) -> float:
    err1 = abs((cash + mtm_val) - total_eq)
    err2 = abs((inv_cost + unrl_pnl) - mtm_val)
    err3 = abs((starting_capital + realized_pnl + unrl_pnl) - total_eq)
    err4 = abs((total_eq - accrued_cost) - net_liq_eq)
    return max(err1, err2, err3, err4)


def _simulate_horizon_ledger(
    strategy_name: str,
    horizon: HorizonType,
    hold_n: int,
    classified_df: pd.DataFrame,
    sym_map: dict[str, pd.DataFrame],
    signal_getter: Callable[[dict[str, Any]], str],
    strict_canonical: bool = True,
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> StrategyLedgerResult:
    """Run chronological event-ledger simulation for one strategy & horizon."""
    events: list[LedgerEvent] = []
    event_seq = 0
    prev_hash = "GENESIS"

    cash = float(starting_capital_inr)
    realized_pnl = 0.0
    gross_pnl_total = 0.0
    cost_total = 0.0
    total_input_signals = 0
    canonical_signals_received = 0
    entries_executed = 0
    matured_cnt = 0
    rejected_cnt = 0
    pending_next_session_cnt = 0
    win_cnt = 0
    rejection_counts: dict[str, int] = {}

    peak_equity = float(starting_capital_inr)
    max_dd = 0.0
    peak_exposure_cost_pct = 0.0
    peak_exposure_mtm_pct = 0.0
    max_symbol_conc_pct = 0.0
    max_sector_conc_pct = 0.0
    max_recon_err = 0.0
    daily_equity_curve: list[dict[str, Any]] = []

    open_positions: dict[str, dict[str, Any]] = {}

    eq_dates = sorted({d for grp in sym_map.values() for d in grp["observation_date"].tolist()})
    snap_dates = sorted(classified_df["observation_date"].astype(str).unique().tolist()) if not classified_df.empty else []
    all_sessions = sorted(set(eq_dates) | set(snap_dates))

    signals_by_obs: dict[str, list[dict[str, Any]]] = {}
    if not classified_df.empty:
        for _, r in classified_df.iterrows():
            signals_by_obs.setdefault(str(r["observation_date"]), []).append(r.to_dict())

    pending_entries_by_date: dict[str, list[dict[str, Any]]] = {}

    def _portfolio_snapshot() -> tuple[float, float, float, float, float, float]:
        inv_cost = sum(p["capital"] for p in open_positions.values())
        mtm_val = sum(p["mtm_value"] for p in open_positions.values())
        unrl_pnl = mtm_val - inv_cost
        accrued_cost = sum(p["accrued_exit_cost"] for p in open_positions.values())
        total_eq = cash + mtm_val
        net_liq = total_eq - accrued_cost
        nonlocal max_recon_err
        err = _reconciliation_error(
            cash, inv_cost, mtm_val, unrl_pnl, realized_pnl, total_eq, accrued_cost, net_liq, starting_capital_inr
        )
        if err > max_recon_err:
            max_recon_err = err
        return inv_cost, mtm_val, unrl_pnl, total_eq, accrued_cost, net_liq

    def _emit(
        *,
        prediction_id: str,
        symbol: str,
        sector: str,
        signal_date: str,
        event_date: str,
        action: LedgerAction,
        reason: str,
        raw_reason: str = "",
        prediction_timestamp: str | None = None,
        entry_timestamp: str | None = None,
        entry_date: str | None = None,
        exit_date: str | None = None,
        entry_price: float | None = None,
        mark_price: float | None = None,
        exit_price: float | None = None,
        ca_adjustment_factor: float = 1.0,
        allocated_capital_inr: float = 0.0,
        cost_pct: float = 0.0,
        position_mtm_value_inr: float = 0.0,
        position_weight_cost_pct: float = 0.0,
        position_weight_mtm_pct: float = 0.0,
        gross_return_pct: float | None = None,
        gross_pnl_inr: float = 0.0,
        transaction_cost_inr: float = 0.0,
        accrued_exit_cost_inr: float = 0.0,
        net_return_pct: float | None = None,
        net_pnl_inr: float = 0.0,
        realized_pnl_delta_inr: float = 0.0,
        unrealized_pnl_inr: float = 0.0,
        holding_sessions: int | None = None,
    ) -> None:
        nonlocal event_seq, prev_hash
        event_seq += 1
        inv_cost, mtm_val, unrl_pnl, total_eq, accrued_cost, net_liq = _portfolio_snapshot()
        ev_id = f"{strategy_name}:{horizon}:{event_seq}:{action}:{prediction_id}:{event_date}"
        ev = _make_hashed_event(
            prev_hash,
            ledger_version=LEDGER_VERSION,
            event_seq=event_seq,
            event_id=ev_id,
            strategy_name=strategy_name,
            strategy_horizon=horizon,
            prediction_id=prediction_id,
            symbol=symbol,
            sector=sector,
            signal_date=signal_date,
            event_date=event_date,
            action=action,
            reason=reason,
            raw_reason=raw_reason,
            prediction_timestamp=prediction_timestamp,
            entry_timestamp=entry_timestamp,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            mark_price=mark_price,
            exit_price=exit_price,
            ca_adjustment_factor=ca_adjustment_factor,
            allocated_capital_inr=allocated_capital_inr,
            cost_pct=cost_pct,
            position_mtm_value_inr=position_mtm_value_inr,
            position_weight_cost_pct=position_weight_cost_pct,
            position_weight_mtm_pct=position_weight_mtm_pct,
            gross_return_pct=gross_return_pct,
            gross_pnl_inr=gross_pnl_inr,
            transaction_cost_inr=transaction_cost_inr,
            accrued_exit_cost_inr=accrued_exit_cost_inr,
            net_return_pct=net_return_pct,
            net_pnl_inr=net_pnl_inr,
            realized_pnl_delta_inr=realized_pnl_delta_inr,
            unrealized_pnl_inr=unrealized_pnl_inr,
            cash_after_inr=cash,
            invested_cost_after_inr=inv_cost,
            mtm_value_after_inr=mtm_val,
            total_equity_after_inr=total_eq,
            accrued_exit_cost_after_inr=accrued_cost,
            net_liquidation_equity_after_inr=net_liq,
            realized_pnl_total_after_inr=realized_pnl,
            unrealized_pnl_total_after_inr=unrl_pnl,
            holding_sessions=holding_sessions,
        )
        prev_hash = ev.event_hash
        events.append(ev)

    def _reject(sig: dict[str, Any], sess_dt: str, code: str, raw_reason: str) -> None:
        nonlocal rejected_cnt
        rejected_cnt += 1
        rejection_counts[code] = rejection_counts.get(code, 0) + 1
        sym = str(sig["symbol"]).upper()
        _emit(
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
            entry_date=sig.get("_scheduled_entry_date"),
        )

    global_max_eq_date = eq_dates[-1] if eq_dates else ""

    for sess_dt in all_sessions:
        sess_had_event = False

        # Phase 1: Market Open (09:15 IST) - Execute scheduled entries for sess_dt
        for sig in pending_entries_by_date.get(sess_dt, []):
            sess_had_event = True
            sym = str(sig["symbol"]).upper()
            pred_id = str(sig["prediction_id"])
            obs_dt = str(sig["observation_date"])
            rec = str(signal_getter(sig) or "UNAVAILABLE")

            inv_cost, mtm_val, _, eq_curr, _, _ = _portfolio_snapshot()

            if rec == "UNAVAILABLE":
                _reject(sig, sess_dt, "REJECTED_UNAVAILABLE_MODEL", rec)
                continue
            if rec != "BUY":
                _reject(sig, sess_dt, "REJECTED_NON_BUY_SIGNAL", rec)
                continue
            if sym in open_positions:
                _reject(sig, sess_dt, "REJECTED_OVERLAPPING_POSITION", f"open_until={open_positions[sym].get('exit_date')}")
                continue
            if cash < MIN_LOT_CAPITAL_INR:
                _reject(sig, sess_dt, "REJECTED_INSUFFICIENT_CASH", f"cash={cash:.2f}")
                continue

            headroom = max(0.0, eq_curr * MAX_EXPOSURE_PCT - inv_cost)
            if headroom < MIN_LOT_CAPITAL_INR:
                _reject(sig, sess_dt, "REJECTED_MAX_EXPOSURE", f"headroom={headroom:.2f}")
                continue

            alloc_inr = min(eq_curr * MAX_POSITION_PCT, headroom, cash)
            if alloc_inr < MIN_LOT_CAPITAL_INR:
                _reject(sig, sess_dt, "REJECTED_MIN_LOT_CAPITAL", f"alloc={alloc_inr:.2f}")
                continue

            grp = sym_map.get(sym)
            entry_idx = int(sig["_entry_idx"])
            entry_open = grp.loc[entry_idx, "open"] if grp is not None and "open" in grp.columns else np.nan
            if pd.isna(entry_open) or float(entry_open) <= 0:
                _reject(sig, sess_dt, "REJECTED_INVALID_ENTRY_PRICE", "missing_or_nonpositive_open")
                continue

            entry_px = float(entry_open)
            cost_pct = _parse_cost_pct(sig)
            accrued_exit_cost = alloc_inr * (cost_pct / 100.0)
            exit_idx = int(sig["_base_idx"]) + hold_n
            exit_dt = str(grp.loc[exit_idx, "observation_date"]) if grp is not None and exit_idx < len(grp) else None
            sector = _resolve_symbol_sector(sym)

            cash -= alloc_inr
            entries_executed += 1
            open_positions[sym] = {
                "prediction_id": pred_id,
                "symbol": sym,
                "sector": sector,
                "signal_date": obs_dt,
                "entry_date": sess_dt,
                "entry_idx": entry_idx,
                "entry_price": entry_px,
                "last_price": entry_px,
                "ca_factor": 1.0,
                "exit_idx": exit_idx,
                "exit_date": exit_dt,
                "capital": alloc_inr,
                "mtm_value": alloc_inr,
                "unrealized_pnl": 0.0,
                "cost_pct": cost_pct,
                "accrued_exit_cost": accrued_exit_cost,
                "sessions_held": 0,
            }

            new_inv_cost, new_mtm_val, _, eq_after, _, _ = _portfolio_snapshot()
            weight_cost_pct = (alloc_inr / max(eq_after, 1e-9)) * 100.0
            peak_exposure_cost_pct = max(peak_exposure_cost_pct, (new_inv_cost / max(eq_after, 1e-9)) * 100.0)
            peak_exposure_mtm_pct = max(peak_exposure_mtm_pct, (new_mtm_val / max(eq_after, 1e-9)) * 100.0)

            _emit(
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
                entry_price=entry_px,
                mark_price=entry_px,
                allocated_capital_inr=alloc_inr,
                cost_pct=cost_pct,
                position_mtm_value_inr=alloc_inr,
                position_weight_cost_pct=weight_cost_pct,
                position_weight_mtm_pct=weight_cost_pct,
                accrued_exit_cost_inr=accrued_exit_cost,
            )

        # Phase 2: Market Close (15:30 IST) - Settle matured positions
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            if pos["exit_date"] is not None and sess_dt >= pos["exit_date"]:
                grp = sym_map.get(sym)
                if grp is None or grp.empty:
                    continue

                exit_idx = pos["exit_idx"]
                exit_px_raw = np.nan
                actual_exit_idx = exit_idx
                exit_reason = f"SCHEDULED_{horizon}_CLOSE_EXIT"

                if exit_idx < len(grp):
                    cand_px = grp.loc[exit_idx, "close"]
                    if pd.notna(cand_px) and float(cand_px) > 0:
                        exit_px_raw = float(cand_px)
                    else:
                        # Fallback to current session close if scheduled close was NaN/non-positive
                        m_list = grp.index[grp["observation_date"] == sess_dt].tolist()
                        if m_list:
                            fb_idx = int(m_list[0])
                            fb_px = grp.loc[fb_idx, "close"]
                            if pd.notna(fb_px) and float(fb_px) > 0:
                                exit_px_raw = float(fb_px)
                                actual_exit_idx = fb_idx
                                exit_reason = "FALLBACK_NEXT_VALID_CLOSE_EXIT"

                if pd.notna(exit_px_raw) and float(exit_px_raw) > 0:
                    sess_had_event = True
                    adj_factor = _compute_ca_adj_factor(grp, pos["entry_idx"], actual_exit_idx)
                    g_ret = ((float(exit_px_raw) * adj_factor) / pos["entry_price"] - 1.0) * 100.0
                    c_pct = pos["cost_pct"]
                    n_ret = g_ret - c_pct
                    g_pnl = pos["capital"] * (g_ret / 100.0)
                    c_inr = pos["capital"] * (c_pct / 100.0)
                    n_pnl = g_pnl - c_inr

                    cash += pos["capital"] + n_pnl
                    realized_pnl += n_pnl
                    gross_pnl_total += g_pnl
                    cost_total += c_inr
                    matured_cnt += 1
                    if n_pnl > 0:
                        win_cnt += 1
                    del open_positions[sym]

                    _, _, _, eq_now, _, _ = _portfolio_snapshot()
                    peak_equity = max(peak_equity, eq_now)
                    max_dd = min(max_dd, ((eq_now / max(peak_equity, 1e-9)) - 1.0) * 100.0)

                    actual_hold = actual_exit_idx - pos["entry_idx"] + 1
                    _emit(
                        prediction_id=pos["prediction_id"],
                        symbol=sym,
                        sector=pos["sector"],
                        signal_date=pos["signal_date"],
                        event_date=sess_dt,
                        action="POSITION_EXITED",
                        reason=exit_reason,
                        entry_date=pos["entry_date"],
                        exit_date=sess_dt,
                        entry_price=pos["entry_price"],
                        exit_price=float(exit_px_raw),
                        ca_adjustment_factor=adj_factor,
                        allocated_capital_inr=pos["capital"],
                        cost_pct=c_pct,
                        position_mtm_value_inr=0.0,
                        position_weight_cost_pct=(pos["capital"] / max(eq_now, 1e-9)) * 100.0,
                        position_weight_mtm_pct=0.0,
                        gross_return_pct=g_ret,
                        gross_pnl_inr=g_pnl,
                        transaction_cost_inr=c_inr,
                        accrued_exit_cost_inr=0.0,
                        net_return_pct=n_ret,
                        net_pnl_inr=n_pnl,
                        realized_pnl_delta_inr=n_pnl,
                        unrealized_pnl_inr=0.0,
                        holding_sessions=actual_hold,
                    )

        # Phase 3: Mark remaining open positions to market (emit POSITION_MTM for every open position)
        for sym, pos in open_positions.items():
            grp = sym_map.get(sym)
            if grp is not None and not grp.empty:
                m_idx_list = grp.index[grp["observation_date"] == sess_dt].tolist()
                if m_idx_list:
                    m_idx = int(m_idx_list[0])
                    c_px = grp.loc[m_idx, "close"]
                    if pd.notna(c_px) and float(c_px) > 0:
                        adj_f = _compute_ca_adj_factor(grp, pos["entry_idx"], m_idx)
                        pos["last_price"] = float(c_px)
                        pos["ca_factor"] = adj_f
                        pos["mtm_value"] = pos["capital"] * ((float(c_px) * adj_f) / pos["entry_price"])
                        pos["unrealized_pnl"] = pos["mtm_value"] - pos["capital"]
                        pos["sessions_held"] = m_idx - pos["entry_idx"] + 1
                    else:
                        pos["sessions_held"] = m_idx - pos["entry_idx"] + 1

            sess_had_event = True
            _, _, _, eq_now, _, _ = _portfolio_snapshot()
            w_cost = (pos["capital"] / max(eq_now, 1e-9)) * 100.0
            w_mtm = (pos["mtm_value"] / max(eq_now, 1e-9)) * 100.0
            unrl_ret = ((pos["mtm_value"] / pos["capital"]) - 1.0) * 100.0 if pos["capital"] > 0 else 0.0

            _emit(
                prediction_id=pos["prediction_id"],
                symbol=sym,
                sector=pos["sector"],
                signal_date=pos["signal_date"],
                event_date=sess_dt,
                action="POSITION_MTM",
                reason="EOD_MARK_TO_MARKET",
                entry_date=pos["entry_date"],
                exit_date=pos["exit_date"],
                entry_price=pos["entry_price"],
                mark_price=pos["last_price"],
                ca_adjustment_factor=pos["ca_factor"],
                allocated_capital_inr=pos["capital"],
                cost_pct=pos["cost_pct"],
                position_mtm_value_inr=pos["mtm_value"],
                position_weight_cost_pct=w_cost,
                position_weight_mtm_pct=w_mtm,
                gross_return_pct=unrl_ret,
                accrued_exit_cost_inr=pos["accrued_exit_cost"],
                unrealized_pnl_inr=pos["unrealized_pnl"],
                holding_sessions=pos["sessions_held"],
            )

        # Phase 4: Post-Close - Process signals on sess_dt
        for sig in signals_by_obs.get(sess_dt, []):
            sess_had_event = True
            total_input_signals += 1
            sym = str(sig["symbol"]).upper()
            pred_id = str(sig["prediction_id"])
            sector = _resolve_symbol_sector(sym)
            is_dup_run = bool(sig.get("_is_dup_run", False))
            if not is_dup_run:
                canonical_signals_received += 1

            rec = str(signal_getter(sig) or "UNAVAILABLE")
            _emit(
                prediction_id=pred_id,
                symbol=sym,
                sector=sector,
                signal_date=sess_dt,
                event_date=sess_dt,
                action="SIGNAL_RECEIVED",
                reason=f"SIGNAL_{rec}",
                raw_reason="DUPLICATE_RUN" if is_dup_run else rec,
                prediction_timestamp=sig.get("prediction_timestamp"),
                entry_timestamp=sig.get("_scheduled_entry_ts"),
                entry_date=sig.get("_scheduled_entry_date"),
            )

            if is_dup_run:
                _reject(sig, sess_dt, "REJECTED_DUPLICATE_RUN", f"prediction_run_id={sig.get('prediction_run_id')}")
                continue

            if strict_canonical and bool(sig.get("_is_lookahead", False)):
                _reject(sig, sess_dt, "REJECTED_LOOKAHEAD", str(sig.get("_lookahead_reason") or "lookahead"))
                continue

            if strict_canonical and bool(sig.get("_is_same_day_dup", False)):
                _reject(
                    sig,
                    sess_dt,
                    "REJECTED_SAME_DAY_DUPLICATE",
                    f"superseded_session={sig.get('nse_session')}",
                )
                continue

            grp = sym_map.get(sym)
            if grp is None or grp.empty:
                _reject(sig, sess_dt, "REJECTED_MISSING_OBSERVATION_BAR", "no_market_history_for_symbol")
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
            else:
                after_mask = grp["observation_date"] > sess_dt
                if after_mask.any():
                    e_idx = int(grp.index[after_mask][0])
                    entry_dt = str(grp.loc[e_idx, "observation_date"])
                    sig_copy = dict(sig)
                    sig_copy["_base_idx"] = e_idx - 1
                    sig_copy["_entry_idx"] = e_idx
                    pending_entries_by_date.setdefault(entry_dt, []).append(sig_copy)
                else:
                    sym_max_dt = str(grp["observation_date"].max())
                    if global_max_eq_date and global_max_eq_date > sess_dt and sym_max_dt <= sess_dt:
                        _reject(sig, sess_dt, "REJECTED_MISSING_OBSERVATION_BAR", "no_subsequent_bar_while_market_advanced")
                    else:
                        pending_next_session_cnt += 1

        # Phase 5: EOD Accounting & Concentration Diagnostics
        if sess_had_event:
            inv_cost_eod, mtm_eod, unrl_eod, eq_eod, accrued_eod, net_liq_eod = _portfolio_snapshot()
            peak_equity = max(peak_equity, eq_eod)
            dd_eod = ((eq_eod / max(peak_equity, 1e-9)) - 1.0) * 100.0
            max_dd = min(max_dd, dd_eod)
            peak_exposure_cost_pct = max(peak_exposure_cost_pct, (inv_cost_eod / max(eq_eod, 1e-9)) * 100.0)
            peak_exposure_mtm_pct = max(peak_exposure_mtm_pct, (mtm_eod / max(eq_eod, 1e-9)) * 100.0)

            sec_totals: dict[str, float] = {}
            for pos in open_positions.values():
                w_sym = (pos["mtm_value"] / max(eq_eod, 1e-9)) * 100.0
                max_symbol_conc_pct = max(max_symbol_conc_pct, w_sym)
                sec = pos["sector"]
                sec_totals[sec] = sec_totals.get(sec, 0.0) + pos["mtm_value"]
            for sec_val in sec_totals.values():
                max_sector_conc_pct = max(max_sector_conc_pct, (sec_val / max(eq_eod, 1e-9)) * 100.0)

            daily_equity_curve.append(
                {
                    "date": sess_dt,
                    "cash_inr": round(cash, 2),
                    "invested_cost_inr": round(inv_cost_eod, 2),
                    "mtm_value_inr": round(mtm_eod, 2),
                    "unrealized_pnl_inr": round(unrl_eod, 2),
                    "realized_pnl_inr": round(realized_pnl, 2),
                    "accrued_exit_cost_inr": round(accrued_eod, 2),
                    "total_equity_inr": round(eq_eod, 2),
                    "net_liquidation_equity_inr": round(net_liq_eod, 2),
                    "drawdown_pct": round(dd_eod, 4),
                    "open_positions": len(open_positions),
                }
            )

    inv_cost_f, mtm_f, unrl_f, eq_f, accrued_f, net_liq_f = _portfolio_snapshot()
    chain_ok, _ = verify_ledger_chain(events, starting_capital_inr=starting_capital_inr)

    open_pos_list = [
        {
            "prediction_id": p["prediction_id"],
            "symbol": p["symbol"],
            "sector": p["sector"],
            "signal_date": p["signal_date"],
            "entry_date": p["entry_date"],
            "exit_date": p["exit_date"],
            "entry_price": round(p["entry_price"], 4),
            "last_price": round(p["last_price"], 4),
            "allocated_capital_inr": round(p["capital"], 2),
            "mtm_value_inr": round(p["mtm_value"], 2),
            "unrealized_pnl_inr": round(p["unrealized_pnl"], 2),
            "accrued_exit_cost_inr": round(p["accrued_exit_cost"], 2),
        }
        for p in open_positions.values()
    ]

    return StrategyLedgerResult(
        ledger_version=LEDGER_VERSION,
        strategy_name=strategy_name,
        horizon=horizon,
        starting_capital_inr=starting_capital_inr,
        ending_equity_inr=round(eq_f, 2),
        net_liquidation_equity_inr=round(net_liq_f, 2),
        cash_inr=round(cash, 2),
        invested_cost_inr=round(inv_cost_f, 2),
        mtm_value_inr=round(mtm_f, 2),
        realized_pnl_inr=round(realized_pnl, 2),
        unrealized_pnl_inr=round(unrl_f, 2),
        accrued_exit_cost_inr=round(accrued_f, 2),
        total_cost_inr=round(cost_total, 2),
        gross_pnl_inr=round(gross_pnl_total, 2),
        net_return_pct=round((eq_f / starting_capital_inr - 1.0) * 100.0, 2),
        net_liquidation_return_pct=round((net_liq_f / starting_capital_inr - 1.0) * 100.0, 2),
        gross_return_pct=round(((gross_pnl_total + unrl_f) / starting_capital_inr) * 100.0, 2),
        max_drawdown_pct=round(max_dd, 2),
        peak_exposure_cost_pct=round(peak_exposure_cost_pct, 2),
        peak_exposure_mtm_pct=round(peak_exposure_mtm_pct, 2),
        max_symbol_concentration_pct=round(max_symbol_conc_pct, 2),
        max_sector_concentration_pct=round(max_sector_conc_pct, 2),
        total_input_signals=total_input_signals,
        canonical_signals_received=canonical_signals_received,
        entries_executed=entries_executed,
        open_positions_count=len(open_positions),
        matured_positions_count=matured_cnt,
        rejected_signals=rejected_cnt,
        pending_next_session_count=pending_next_session_cnt,
        rejection_counts=rejection_counts,
        win_rate_pct=round(win_cnt / matured_cnt * 100.0, 2) if matured_cnt > 0 else None,
        accounting_invariant_ok=(max_recon_err < 1e-5) and (cash >= -1e-6),
        max_reconciliation_error_inr=round(max_recon_err, 8),
        ledger_chain_valid=chain_ok,
        status="ACTIVE" if entries_executed > 0 else "INSUFFICIENT_DATA",
        daily_equity_curve=daily_equity_curve,
        events=events,
        open_positions=open_pos_list,
    )


def verify_ledger_chain(
    events: list[LedgerEvent],
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> tuple[bool, list[str]]:
    """Verify hash-chain integrity, monotonic sequence, and accounting invariants on every event."""
    errors: list[str] = []
    expected_prev = "GENESIS"
    prev_seq = 0
    for ev in events:
        if ev.event_seq != prev_seq + 1:
            errors.append(f"Non-monotonic event_seq={ev.event_seq} (expected {prev_seq + 1})")
        prev_seq = ev.event_seq

        if ev.prev_event_hash != expected_prev:
            errors.append(f"Broken prev_event_hash at seq={ev.event_seq}")

        expected_hash = compute_event_hash(ev.to_dict())
        if ev.event_hash != expected_hash:
            errors.append(f"Tampered event_hash at seq={ev.event_seq}")
        expected_prev = ev.event_hash

        err = _reconciliation_error(
            ev.cash_after_inr,
            ev.invested_cost_after_inr,
            ev.mtm_value_after_inr,
            ev.unrealized_pnl_total_after_inr,
            ev.realized_pnl_total_after_inr,
            ev.total_equity_after_inr,
            ev.accrued_exit_cost_after_inr,
            ev.net_liquidation_equity_after_inr,
            starting_capital_inr,
        )
        if err > 1e-5:
            errors.append(f"Accounting invariant error={err:.8f} at seq={ev.event_seq} ({ev.action})")
        if ev.cash_after_inr < -1e-6:
            errors.append(f"Negative cash={ev.cash_after_inr} at seq={ev.event_seq}")

    return len(errors) == 0, errors


def events_to_dataframe(events: list[LedgerEvent]) -> pd.DataFrame:
    """Serialize immutable LedgerEvent objects to a flat DataFrame (zero truncation)."""
    if not events:
        return pd.DataFrame(columns=list(LedgerEvent.__dataclass_fields__.keys()))
    return pd.DataFrame([e.to_dict() for e in events])


def dataframe_to_events(df: pd.DataFrame) -> list[LedgerEvent]:
    """Deserialize LedgerEvent objects from a DataFrame / Parquet round-trip."""
    if df.empty:
        return []
    out: list[LedgerEvent] = []
    for rec in df.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                cleaned[k] = None
            elif pd.isna(v) if not isinstance(v, (str, int, float, bool, type(None))) else False:
                cleaned[k] = None
            else:
                cleaned[k] = v
        if cleaned.get("holding_sessions") is not None:
            cleaned["holding_sessions"] = int(cleaned["holding_sessions"])
        if cleaned.get("event_seq") is not None:
            cleaned["event_seq"] = int(cleaned["event_seq"])
        out.append(LedgerEvent(**cleaned))
    return out


def persist_ledger_parquet(events: list[LedgerEvent], target_path: Path | str) -> int:
    """Atomically write the FULL immutable event ledger to Parquet with zero truncation."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = events_to_dataframe(events)
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp_path = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(path)
    return len(df)


def replay_strategy_from_events(
    events: list[LedgerEvent] | pd.DataFrame,
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> StrategyLedgerResult:
    """Reconstruct exact strategy portfolio state purely by replaying immutable ledger events."""
    ev_list = dataframe_to_events(events) if isinstance(events, pd.DataFrame) else list(events)
    if not ev_list:
        raise ValueError("Cannot replay strategy from empty event list.")

    ev_list = sorted(ev_list, key=lambda e: e.event_seq)
    chain_ok, chain_errors = verify_ledger_chain(ev_list, starting_capital_inr=starting_capital_inr)
    if not chain_ok:
        raise ValueError(f"Ledger verification failed during replay: {'; '.join(chain_errors[:3])}")

    first = ev_list[0]
    strategy_name = first.strategy_name
    horizon = first.strategy_horizon

    cash = float(starting_capital_inr)
    realized_pnl = 0.0
    gross_pnl_total = 0.0
    cost_total = 0.0
    total_input_signals = 0
    canonical_signals_received = 0
    entries_executed = 0
    matured_cnt = 0
    rejected_cnt = 0
    win_cnt = 0
    rejection_counts: dict[str, int] = {}
    open_positions: dict[str, dict[str, Any]] = {}
    peak_equity = float(starting_capital_inr)
    max_dd = 0.0
    peak_exposure_cost_pct = 0.0
    peak_exposure_mtm_pct = 0.0
    max_symbol_conc_pct = 0.0
    max_sector_conc_pct = 0.0
    max_recon_err = 0.0
    daily_equity_curve: list[dict[str, Any]] = []

    def _current_totals() -> tuple[float, float, float, float, float, float]:
        inv_cost = sum(p["capital"] for p in open_positions.values())
        mtm_val = sum(p["mtm_value"] for p in open_positions.values())
        unrl_pnl = mtm_val - inv_cost
        accrued_cost = sum(p["accrued_exit_cost"] for p in open_positions.values())
        total_eq = cash + mtm_val
        net_liq = total_eq - accrued_cost
        return inv_cost, mtm_val, unrl_pnl, total_eq, accrued_cost, net_liq

    def _flush_eod(dt_str: str) -> None:
        nonlocal peak_equity, max_dd, peak_exposure_cost_pct, peak_exposure_mtm_pct
        nonlocal max_symbol_conc_pct, max_sector_conc_pct
        inv_cost, mtm_val, unrl_pnl, total_eq, accrued_cost, net_liq = _current_totals()
        peak_equity = max(peak_equity, total_eq)
        dd_eod = ((total_eq / max(peak_equity, 1e-9)) - 1.0) * 100.0
        max_dd = min(max_dd, dd_eod)
        peak_exposure_cost_pct = max(peak_exposure_cost_pct, (inv_cost / max(total_eq, 1e-9)) * 100.0)
        peak_exposure_mtm_pct = max(peak_exposure_mtm_pct, (mtm_val / max(total_eq, 1e-9)) * 100.0)

        sec_totals: dict[str, float] = {}
        for pos in open_positions.values():
            w_sym = (pos["mtm_value"] / max(total_eq, 1e-9)) * 100.0
            max_symbol_conc_pct = max(max_symbol_conc_pct, w_sym)
            sec_totals[pos["sector"]] = sec_totals.get(pos["sector"], 0.0) + pos["mtm_value"]
        for sv in sec_totals.values():
            max_sector_conc_pct = max(max_sector_conc_pct, (sv / max(total_eq, 1e-9)) * 100.0)

        daily_equity_curve.append(
            {
                "date": dt_str,
                "cash_inr": round(cash, 2),
                "invested_cost_inr": round(inv_cost, 2),
                "mtm_value_inr": round(mtm_val, 2),
                "unrealized_pnl_inr": round(unrl_pnl, 2),
                "realized_pnl_inr": round(realized_pnl, 2),
                "accrued_exit_cost_inr": round(accrued_cost, 2),
                "total_equity_inr": round(total_eq, 2),
                "net_liquidation_equity_inr": round(net_liq, 2),
                "drawdown_pct": round(dd_eod, 4),
                "open_positions": len(open_positions),
            }
        )

    current_date: str | None = None
    seen_pending_candidates: set[str] = set()
    resolved_candidates: set[str] = set()

    for ev in ev_list:
        if current_date is not None and ev.event_date != current_date:
            _flush_eod(current_date)
        current_date = ev.event_date

        if ev.action == "SIGNAL_RECEIVED":
            total_input_signals += 1
            if ev.raw_reason != "DUPLICATE_RUN":
                canonical_signals_received += 1
                seen_pending_candidates.add(ev.prediction_id)
        elif ev.action == "SIGNAL_REJECTED":
            rejected_cnt += 1
            rejection_counts[ev.reason] = rejection_counts.get(ev.reason, 0) + 1
            resolved_candidates.add(ev.prediction_id)
        elif ev.action == "ENTRY_EXECUTED":
            entries_executed += 1
            resolved_candidates.add(ev.prediction_id)
            cash -= ev.allocated_capital_inr
            accrued = ev.allocated_capital_inr * (ev.cost_pct / 100.0)
            open_positions[ev.symbol] = {
                "prediction_id": ev.prediction_id,
                "symbol": ev.symbol,
                "sector": ev.sector,
                "signal_date": ev.signal_date,
                "entry_date": ev.entry_date or ev.event_date,
                "exit_date": ev.exit_date,
                "entry_price": float(ev.entry_price or 0.0),
                "last_price": float(ev.mark_price or ev.entry_price or 0.0),
                "capital": ev.allocated_capital_inr,
                "mtm_value": ev.allocated_capital_inr,
                "unrealized_pnl": 0.0,
                "cost_pct": ev.cost_pct,
                "accrued_exit_cost": accrued,
            }
            inv_c, mtm_v, _, eq_a, _, _ = _current_totals()
            peak_exposure_cost_pct = max(peak_exposure_cost_pct, (inv_c / max(eq_a, 1e-9)) * 100.0)
            peak_exposure_mtm_pct = max(peak_exposure_mtm_pct, (mtm_v / max(eq_a, 1e-9)) * 100.0)
        elif ev.action == "POSITION_MTM":
            pos = open_positions[ev.symbol]
            pos["last_price"] = float(ev.mark_price or pos["last_price"])
            pos["mtm_value"] = ev.position_mtm_value_inr
            pos["unrealized_pnl"] = ev.position_mtm_value_inr - pos["capital"]
        elif ev.action == "POSITION_EXITED":
            pos = open_positions.pop(ev.symbol)
            cash += pos["capital"] + ev.net_pnl_inr
            realized_pnl += ev.net_pnl_inr
            gross_pnl_total += ev.gross_pnl_inr
            cost_total += ev.transaction_cost_inr
            matured_cnt += 1
            if ev.net_pnl_inr > 0:
                win_cnt += 1
            _, _, _, eq_now, _, _ = _current_totals()
            peak_equity = max(peak_equity, eq_now)
            max_dd = min(max_dd, ((eq_now / max(peak_equity, 1e-9)) - 1.0) * 100.0)

        inv_c, mtm_v, unrl_p, total_eq, accrued_c, net_liq = _current_totals()
        step_err = max(
            abs(cash - ev.cash_after_inr),
            abs(inv_c - ev.invested_cost_after_inr),
            abs(mtm_v - ev.mtm_value_after_inr),
            abs(total_eq - ev.total_equity_after_inr),
            abs(accrued_c - ev.accrued_exit_cost_after_inr),
            abs(net_liq - ev.net_liquidation_equity_after_inr),
        )
        max_recon_err = max(max_recon_err, step_err)
        if step_err > 1e-5:
            raise ValueError(f"Replay state mismatch at seq={ev.event_seq}: err={step_err:.8f}")

    if current_date is not None:
        _flush_eod(current_date)

    inv_f, mtm_f, unrl_f, eq_f, accrued_f, net_liq_f = _current_totals()
    pending_cnt = len(seen_pending_candidates - resolved_candidates)

    open_pos_list = [
        {
            "prediction_id": p["prediction_id"],
            "symbol": p["symbol"],
            "sector": p["sector"],
            "signal_date": p["signal_date"],
            "entry_date": p["entry_date"],
            "exit_date": p["exit_date"],
            "entry_price": round(p["entry_price"], 4),
            "last_price": round(p["last_price"], 4),
            "allocated_capital_inr": round(p["capital"], 2),
            "mtm_value_inr": round(p["mtm_value"], 2),
            "unrealized_pnl_inr": round(p["unrealized_pnl"], 2),
            "accrued_exit_cost_inr": round(p["accrued_exit_cost"], 2),
        }
        for p in open_positions.values()
    ]

    return StrategyLedgerResult(
        ledger_version=LEDGER_VERSION,
        strategy_name=strategy_name,
        horizon=horizon,
        starting_capital_inr=starting_capital_inr,
        ending_equity_inr=round(eq_f, 2),
        net_liquidation_equity_inr=round(net_liq_f, 2),
        cash_inr=round(cash, 2),
        invested_cost_inr=round(inv_f, 2),
        mtm_value_inr=round(mtm_f, 2),
        realized_pnl_inr=round(realized_pnl, 2),
        unrealized_pnl_inr=round(unrl_f, 2),
        accrued_exit_cost_inr=round(accrued_f, 2),
        total_cost_inr=round(cost_total, 2),
        gross_pnl_inr=round(gross_pnl_total, 2),
        net_return_pct=round((eq_f / starting_capital_inr - 1.0) * 100.0, 2),
        net_liquidation_return_pct=round((net_liq_f / starting_capital_inr - 1.0) * 100.0, 2),
        gross_return_pct=round(((gross_pnl_total + unrl_f) / starting_capital_inr) * 100.0, 2),
        max_drawdown_pct=round(max_dd, 2),
        peak_exposure_cost_pct=round(peak_exposure_cost_pct, 2),
        peak_exposure_mtm_pct=round(peak_exposure_mtm_pct, 2),
        max_symbol_concentration_pct=round(max_symbol_conc_pct, 2),
        max_sector_concentration_pct=round(max_sector_conc_pct, 2),
        total_input_signals=total_input_signals,
        canonical_signals_received=canonical_signals_received,
        entries_executed=entries_executed,
        open_positions_count=len(open_positions),
        matured_positions_count=matured_cnt,
        rejected_signals=rejected_cnt,
        pending_next_session_count=pending_cnt,
        rejection_counts=rejection_counts,
        win_rate_pct=round(win_cnt / matured_cnt * 100.0, 2) if matured_cnt > 0 else None,
        accounting_invariant_ok=(max_recon_err < 1e-5) and (cash >= -1e-6),
        max_reconciliation_error_inr=round(max_recon_err, 8),
        ledger_chain_valid=chain_ok,
        status="ACTIVE" if entries_executed > 0 else "INSUFFICIENT_DATA",
        daily_equity_curve=daily_equity_curve,
        events=ev_list,
        open_positions=open_pos_list,
    )


def replay_portfolio_from_events(
    events: list[LedgerEvent] | pd.DataFrame,
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> dict[str, StrategyLedgerResult]:
    """Replay all `(strategy_name, strategy_horizon)` streams from a combined event ledger."""
    ev_list = dataframe_to_events(events) if isinstance(events, pd.DataFrame) else list(events)
    grouped: dict[str, list[LedgerEvent]] = {}
    for ev in ev_list:
        key = ev.strategy_horizon if ev.strategy_name == "bharat_alpha" else f"{ev.strategy_name}_{ev.strategy_horizon}"
        grouped.setdefault(key, []).append(ev)
    return {k: replay_strategy_from_events(v, starting_capital_inr=starting_capital_inr) for k, v in grouped.items()}


def _prepare_nifty_frame(indices_df: pd.DataFrame | None) -> pd.DataFrame:
    if indices_df is None or indices_df.empty or "index_code" not in indices_df.columns:
        return pd.DataFrame()
    norm = indices_df["index_code"].apply(normalize_index_code)
    n50 = indices_df[norm == NIFTY_CANONICAL_CODE].copy()
    if n50.empty:
        return n50
    n50["observation_date"] = n50["observation_date"].astype(str).str.strip()
    n50 = n50.sort_values("observation_date").drop_duplicates(subset=["observation_date"], keep="last").reset_index(drop=True)
    return n50


def _compute_nifty_buy_hold_benchmark(
    indices_df: pd.DataFrame | None,
    entry_start_date: str | None,
    end_date: str | None,
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> dict[str, Any]:
    unavailable = {
        "index_code_normalized": NIFTY_CANONICAL_CODE,
        "index_code_matched": None,
        "entry_date": None,
        "end_date": None,
        "entry_price": None,
        "end_close": None,
        "ending_equity_inr": None,
        "net_return_pct": None,
        "max_drawdown_pct": None,
        "daily_equity_curve": [],
        "status": "UNAVAILABLE",
    }
    n50 = _prepare_nifty_frame(indices_df)
    if n50.empty or not entry_start_date:
        return unavailable

    matched_raw = str(n50["index_code"].iloc[0])
    n50 = n50[n50["observation_date"] >= entry_start_date].copy()
    if end_date:
        n50 = n50[n50["observation_date"] <= end_date].copy()
    n50 = n50.sort_values("observation_date").reset_index(drop=True)
    if n50.empty:
        return unavailable

    open_0 = pd.to_numeric(n50.get("open"), errors="coerce").iloc[0] if "open" in n50.columns else np.nan
    closes = pd.to_numeric(n50["close"], errors="coerce").values
    dates = n50["observation_date"].tolist()
    valid_mask = ~np.isnan(closes) & (closes > 0)
    if not valid_mask.any():
        return unavailable

    closes = closes[valid_mask]
    dates = [d for d, ok in zip(dates, valid_mask) if ok]
    entry_px = float(open_0) if pd.notna(open_0) and float(open_0) > 0 else float(closes[0])
    if entry_px <= 0:
        return unavailable

    invested = starting_capital_inr * MAX_EXPOSURE_PCT
    cash_res = starting_capital_inr - invested
    cost_drag = invested * (NIFTY_BENCHMARK_COST_PCT / 100.0)
    net_invested = invested - cost_drag

    curve: list[dict[str, Any]] = []
    peak = starting_capital_inr
    max_dd = 0.0
    for dt_s, cl_px in zip(dates, closes):
        mtm = net_invested * (float(cl_px) / entry_px)
        eq = cash_res + mtm
        peak = max(peak, eq)
        dd = ((eq / max(peak, 1e-9)) - 1.0) * 100.0
        max_dd = min(max_dd, dd)
        curve.append(
            {
                "date": dt_s,
                "cash_inr": round(cash_res, 2),
                "mtm_value_inr": round(mtm, 2),
                "total_equity_inr": round(eq, 2),
                "drawdown_pct": round(dd, 4),
            }
        )

    eq_end = float(cash_res + net_invested * (float(closes[-1]) / entry_px))
    return {
        "index_code_normalized": NIFTY_CANONICAL_CODE,
        "index_code_matched": matched_raw,
        "entry_date": dates[0],
        "end_date": dates[-1],
        "entry_price": round(entry_px, 2),
        "end_close": round(float(closes[-1]), 2),
        "ending_equity_inr": round(eq_end, 2),
        "net_return_pct": round((eq_end / starting_capital_inr - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "daily_equity_curve": curve,
        "status": "ACTIVE",
    }


def _compute_nifty_matched_windows_benchmark(
    strategy_res: StrategyLedgerResult,
    indices_df: pd.DataFrame | None,
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> dict[str, Any]:
    """Replicate a strategy's exact entry/exit dates & lot sizes into NIFTY_50."""
    n50 = _prepare_nifty_frame(indices_df)
    if n50.empty:
        return {
            "horizon": strategy_res.horizon,
            "index_code_normalized": NIFTY_CANONICAL_CODE,
            "matched_entries": 0,
            "ending_equity_inr": None,
            "net_return_pct": None,
            "max_drawdown_pct": None,
            "status": "UNAVAILABLE",
        }

    if strategy_res.entries_executed == 0:
        return {
            "horizon": strategy_res.horizon,
            "index_code_normalized": NIFTY_CANONICAL_CODE,
            "matched_entries": 0,
            "ending_equity_inr": round(starting_capital_inr, 2),
            "net_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "status": "NO_TRADES_MATCHED",
        }

    n50_by_date: dict[str, tuple[float | None, float | None]] = {}
    for _, r in n50.iterrows():
        op = float(r["open"]) if "open" in n50.columns and pd.notna(r.get("open")) else None
        cl = float(r["close"]) if "close" in n50.columns and pd.notna(r.get("close")) else None
        n50_by_date[str(r["observation_date"])] = (op, cl)

    cash = float(starting_capital_inr)
    active_lots: dict[str, dict[str, Any]] = {}
    peak = float(starting_capital_inr)
    max_dd = 0.0
    matched_entries = 0

    events_by_date: dict[str, list[LedgerEvent]] = {}
    for ev in strategy_res.events:
        events_by_date.setdefault(ev.event_date, []).append(ev)

    for dt_s in sorted(events_by_date.keys()):
        day_evs = events_by_date[dt_s]
        n_op, n_cl = n50_by_date.get(dt_s, (None, None))
        for ev in day_evs:
            if ev.action == "ENTRY_EXECUTED":
                px_in = n_op if (n_op is not None and n_op > 0) else n_cl
                if px_in is not None and px_in > 0:
                    cash -= ev.allocated_capital_inr
                    matched_entries += 1
                    active_lots[ev.prediction_id] = {
                        "capital": ev.allocated_capital_inr,
                        "entry_px": px_in,
                        "mtm_value": ev.allocated_capital_inr,
                    }
            elif ev.action == "POSITION_EXITED" and ev.prediction_id in active_lots:
                lot = active_lots.pop(ev.prediction_id)
                if n_cl is not None and n_cl > 0:
                    g_ret = (n_cl / lot["entry_px"]) - 1.0
                    c_inr = lot["capital"] * (NIFTY_BENCHMARK_COST_PCT / 100.0)
                    cash += lot["capital"] * (1.0 + g_ret) - c_inr
                else:
                    cash += lot["mtm_value"]

        if n_cl is not None and n_cl > 0:
            for lot in active_lots.values():
                lot["mtm_value"] = lot["capital"] * (n_cl / lot["entry_px"])

        eq = cash + sum(l["mtm_value"] for l in active_lots.values())
        peak = max(peak, eq)
        max_dd = min(max_dd, ((eq / max(peak, 1e-9)) - 1.0) * 100.0)

    eq_final = cash + sum(l["mtm_value"] for l in active_lots.values())
    return {
        "horizon": strategy_res.horizon,
        "index_code_normalized": NIFTY_CANONICAL_CODE,
        "matched_entries": matched_entries,
        "ending_equity_inr": round(eq_final, 2),
        "net_return_pct": round((eq_final / starting_capital_inr - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "status": "ACTIVE" if matched_entries > 0 else "UNAVAILABLE",
    }


def _compute_cash_no_trade_benchmark(
    session_dates: list[str],
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> dict[str, Any]:
    return {
        "entry_date": session_dates[0] if session_dates else None,
        "end_date": session_dates[-1] if session_dates else None,
        "ending_equity_inr": round(starting_capital_inr, 2),
        "net_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "daily_equity_curve": [
            {
                "date": dt_s,
                "cash_inr": round(starting_capital_inr, 2),
                "mtm_value_inr": 0.0,
                "total_equity_inr": round(starting_capital_inr, 2),
                "drawdown_pct": 0.0,
            }
            for dt_s in session_dates
        ],
        "status": "ACTIVE",
    }


def run_event_ledger_portfolio(
    snapshots_df: pd.DataFrame,
    equities_df: pd.DataFrame | None = None,
    indices_df: pd.DataFrame | None = None,
    strict_canonical: bool = True,
    starting_capital_inr: float = STARTING_CAPITAL_INR,
) -> EventLedgerPortfolioReport:
    """Execute the remediated chronological event-ledger paper portfolio simulator."""
    sym_map = _prepare_sym_map(equities_df)
    classified_df, hy_stats = _classify_signals(snapshots_df, sym_map=sym_map, strict_canonical=strict_canonical)

    strategies: dict[str, StrategyLedgerResult] = {}
    all_events: list[LedgerEvent] = []

    for horizon, hold_n in (("1D", 1), ("5D", 5), ("10D", 10)):
        rec_col = {"1D": "recommendation_1d", "5D": "recommendation_5d", "10D": "recommendation_10d"}[horizon]
        res = _simulate_horizon_ledger(
            strategy_name="bharat_alpha",
            horizon=horizon,  # type: ignore[arg-type]
            hold_n=hold_n,
            classified_df=classified_df,
            sym_map=sym_map,
            signal_getter=lambda s, rc=rec_col: str(s.get(rc) or "UNAVAILABLE"),
            strict_canonical=strict_canonical,
            starting_capital_inr=starting_capital_inr,
        )
        strategies[horizon] = res
        all_events.extend(res.events)

    # Integrated Naive Momentum benchmarks (horizon-matched, identical hygiene & ledger rules)
    def _mom_signal_factory(col_primary: str, col_fallback: str | None) -> Callable[[dict[str, Any]], str]:
        def _sig(s: dict[str, Any]) -> str:
            p = s.get(col_primary)
            if (p is None or (isinstance(p, float) and np.isnan(p))) and col_fallback:
                p = s.get(col_fallback)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return "UNAVAILABLE"
            return "BUY" if float(p) >= 0.535 else "HOLD"

        return _sig

    naive_momentum: dict[str, dict[str, Any]] = {}
    for h, n, col, fb in (
        ("1D", 1, "benchmark_prob_1d_up", "benchmark_prob_up"),
        ("5D", 5, "benchmark_prob_5d_up", None),
        ("10D", 10, "benchmark_prob_10d_up", None),
    ):
        if classified_df.empty or (col not in classified_df.columns and (fb is None or fb not in classified_df.columns)):
            naive_momentum[h] = {
                "horizon": h,
                "ending_equity_inr": None,
                "net_liquidation_equity_inr": None,
                "net_return_pct": None,
                "max_drawdown_pct": None,
                "entries_executed": 0,
                "status": "INSUFFICIENT_DATA",
            }
        else:
            m_res = _simulate_horizon_ledger(
                strategy_name="naive_momentum",
                horizon=h,  # type: ignore[arg-type]
                hold_n=n,
                classified_df=classified_df,
                sym_map=sym_map,
                signal_getter=_mom_signal_factory(col, fb),
                strict_canonical=strict_canonical,
                starting_capital_inr=starting_capital_inr,
            )
            ok = m_res.entries_executed > 0
            naive_momentum[h] = {
                "horizon": h,
                "ending_equity_inr": m_res.ending_equity_inr if ok else None,
                "net_liquidation_equity_inr": m_res.net_liquidation_equity_inr if ok else None,
                "accrued_exit_cost_inr": m_res.accrued_exit_cost_inr if ok else 0.0,
                "net_return_pct": m_res.net_return_pct if ok else None,
                "net_liquidation_return_pct": m_res.net_liquidation_return_pct if ok else None,
                "max_drawdown_pct": m_res.max_drawdown_pct if ok else None,
                "entries_executed": m_res.entries_executed,
                "open_positions_count": m_res.open_positions_count,
                "matured_positions_count": m_res.matured_positions_count,
                "accounting_invariant_ok": m_res.accounting_invariant_ok,
                "ledger_chain_valid": m_res.ledger_chain_valid,
                "status": m_res.status,
            }

    # Determine aligned entry/exit calendar window across executed trades or valid canonical signals
    executed_entry_dates = [
        ev.entry_date
        for s in strategies.values()
        for ev in s.events
        if ev.action == "ENTRY_EXECUTED" and ev.entry_date
    ]
    valid_sched_entries: list[str] = []
    if not classified_df.empty:
        valid_mask = (~classified_df["_is_dup_run"]) & (
            (~classified_df["_is_lookahead"]) if strict_canonical else True
        )
        valid_sched_entries = (
            classified_df.loc[valid_mask, "_scheduled_entry_date"].dropna().astype(str).tolist()
        )

    entry_start_date = (
        min(executed_entry_dates)
        if executed_entry_dates
        else (min(valid_sched_entries) if valid_sched_entries else None)
    )
    all_eq_dates = sorted({d for grp in sym_map.values() for d in grp["observation_date"].tolist()})
    end_date = all_eq_dates[-1] if all_eq_dates else None

    nifty_buy_hold = _compute_nifty_buy_hold_benchmark(
        indices_df,
        entry_start_date=entry_start_date,
        end_date=end_date,
        starting_capital_inr=starting_capital_inr,
    )
    nifty_matched_windows = {
        h: _compute_nifty_matched_windows_benchmark(s, indices_df, starting_capital_inr=starting_capital_inr)
        for h, s in strategies.items()
    }
    cash_dates = [d for d in all_eq_dates if (not entry_start_date or d >= entry_start_date)]
    cash_no_trade = _compute_cash_no_trade_benchmark(cash_dates, starting_capital_inr=starting_capital_inr)

    benchmarks = {
        "nifty_50_buy_hold": nifty_buy_hold,
        "nifty_50_matched_windows": nifty_matched_windows,
        "naive_momentum_1d": naive_momentum["1D"],
        "naive_momentum_5d": naive_momentum["5D"],
        "naive_momentum_10d": naive_momentum["10D"],
        "cash_no_trade": cash_no_trade,
    }

    # Verify full replay reproducibility on every strategy stream
    replay_ok = True
    max_err = 0.0
    for h, s in strategies.items():
        replayed = replay_strategy_from_events(s.events, starting_capital_inr=starting_capital_inr)
        max_err = max(max_err, s.max_reconciliation_error_inr, replayed.max_reconciliation_error_inr)
        if (
            replayed.ending_equity_inr != s.ending_equity_inr
            or replayed.cash_inr != s.cash_inr
            or replayed.mtm_value_inr != s.mtm_value_inr
            or replayed.accrued_exit_cost_inr != s.accrued_exit_cost_inr
            or not replayed.accounting_invariant_ok
            or not replayed.ledger_chain_valid
        ):
            replay_ok = False

    reconciliation_checks = {
        "all_invariants_ok": all(s.accounting_invariant_ok for s in strategies.values()),
        "all_chains_valid": all(s.ledger_chain_valid for s in strategies.values()),
        "replay_exact_match": replay_ok,
        "max_reconciliation_error_inr": round(max_err, 8),
        "zero_truncation_events_count": len(all_events),
    }

    return EventLedgerPortfolioReport(
        ledger_version=LEDGER_VERSION,
        generated_at=datetime.now(IST_TZ).isoformat(),
        strict_canonical=strict_canonical,
        total_input_snapshots=hy_stats["total_input_snapshots"],
        duplicate_run_snapshots=hy_stats["duplicate_run_snapshots"],
        canonical_snapshots=hy_stats["canonical_snapshots"],
        lookahead_rejected_snapshots=hy_stats["lookahead_rejected_snapshots"],
        same_day_duplicate_snapshots=hy_stats["same_day_duplicate_snapshots"],
        strategies=strategies,
        benchmarks=benchmarks,
        all_events=all_events,
        reconciliation_checks=reconciliation_checks,
    )


def run_event_ledger_from_data_dir(
    data_dir: Path | str = Path("data"),
    strict_canonical: bool = True,
) -> EventLedgerPortfolioReport:
    """Convenience loader for running the remediated event-ledger simulator on persisted Parquet data."""
    d = Path(data_dir)
    snaps = pd.read_parquet(d / "predictions" / "forward_predictions.parquet")
    eq = pd.read_parquet(d / "processed" / "equities_daily.parquet")
    idx = pd.read_parquet(d / "processed" / "indices_daily.parquet")
    return run_event_ledger_portfolio(
        snapshots_df=snaps,
        equities_df=eq,
        indices_df=idx,
        strict_canonical=strict_canonical,
    )
