"""CYCLE-009C — Lookahead, Causal Cutoff, MTF Alignment & Recursive Convergence Bias Auditor."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from trader.config import IST_TZ
from trader.intelligence.data_contract import (
    LookaheadViolationError,
    ProvenanceTier,
    _coerce_ts_series_ist,
    _ensure_tz_aware_ist,
    build_ohlcv_evidence_frame,
)
from trader.intelligence.mtf_aligner import (
    align_htf_to_ltf_ist,
    expected_5m_sub_bars_for_bucket,
)


@dataclass(frozen=True)
class BiasViolation:
    """Single lookahead, causality, alignment, or convergence violation."""

    check_name: str
    code: str
    message: str
    offending_timestamp: datetime | str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BiasAuditResult:
    """Structured result emitted by each CYCLE-009C gate auditor."""

    passed: bool
    check_name: str
    violations: tuple[BiasViolation, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    checked_cutoffs: tuple[datetime, ...] = ()


def _default_convergence_rtol(span_or_period: int) -> float:
    if span_or_period <= 14:
        return 1e-4
    if span_or_period <= 30:
        return 5e-4
    return 5e-3


def _resolve_ts_close_series(df: pd.DataFrame, label: str = "ts_close_ist") -> pd.Series:
    for col in ("ts_close_ist", "ts_event_ist", "timestamp"):
        if col in df.columns:
            return _coerce_ts_series_ist(df[col], f"{label}:{col}")
    raise ValueError(
        f"DataFrame missing timestamp column ('ts_close_ist', 'ts_event_ist', or 'timestamp') for {label}"
    )


def _slice_output_by_mask(output: Any, mask: np.ndarray) -> Any:
    if isinstance(output, (pd.Series, pd.DataFrame)):
        if len(output) == len(mask):
            return output.iloc[mask]
        if len(output) == int(mask.sum()):
            return output
        raise ValueError(
            f"Output length {len(output)} does not match full ({len(mask)}) or masked ({int(mask.sum())}) length"
        )
    if isinstance(output, np.ndarray):
        if output.shape[0] == len(mask):
            return output[mask]
        if output.shape[0] == int(mask.sum()):
            return output
        raise ValueError(
            f"Array length {output.shape[0]} does not match full ({len(mask)}) or masked ({int(mask.sum())}) length"
        )
    return output


def _to_numeric_array_and_extras(output: Any) -> tuple[np.ndarray, dict[str, list[Any]]]:
    if isinstance(output, pd.DataFrame):
        num_cols = output.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        other_cols = [c for c in output.columns if c not in num_cols]
        num_arr = (
            output[num_cols].to_numpy(dtype=float)
            if num_cols
            else np.empty((len(output), 0), dtype=float)
        )
        extras = {str(c): output[c].tolist() for c in other_cols}
        return num_arr, extras
    if isinstance(output, pd.Series):
        if pd.api.types.is_numeric_dtype(output) or pd.api.types.is_bool_dtype(output):
            return output.to_numpy(dtype=float).reshape(-1, 1), {}
        return np.empty((len(output), 0), dtype=float), {"_series": output.tolist()}
    if isinstance(output, np.ndarray):
        arr = np.asarray(output, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, {}
    if isinstance(output, (int, float, np.number)):
        return np.array([[float(output)]], dtype=float), {}
    raise TypeError(f"Unsupported feature output type for numerical comparison: {type(output).__name__}")


def _compare_outputs(
    baseline: Any,
    candidate: Any,
    *,
    atol: float,
) -> tuple[bool, float, str]:
    """Compare two outputs for NaN-mask identity and atol invariance."""
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        if set(baseline.keys()) != set(candidate.keys()):
            return False, float("inf"), "Mapping keys mismatch"
        max_diff = 0.0
        for key in baseline:
            ok, diff, reason = _compare_outputs(baseline[key], candidate[key], atol=atol)
            max_diff = max(max_diff, diff)
            if not ok:
                return False, max_diff, f"Key {key!r}: {reason}"
        return True, max_diff, "ok"

    base_arr, base_extras = _to_numeric_array_and_extras(baseline)
    cand_arr, cand_extras = _to_numeric_array_and_extras(candidate)

    if base_arr.shape != cand_arr.shape:
        return (
            False,
            float("inf"),
            f"Shape mismatch: baseline {base_arr.shape} vs candidate {cand_arr.shape}",
        )

    if base_extras != cand_extras:
        return False, float("inf"), "Non-numeric column values differ"

    if base_arr.size == 0:
        return True, 0.0, "ok"

    base_nan = np.isnan(base_arr)
    cand_nan = np.isnan(cand_arr)
    if not np.array_equal(base_nan, cand_nan):
        return False, float("inf"), "NaN-mask mismatch between baseline and candidate"

    finite_mask = ~base_nan
    if not finite_mask.any():
        return True, 0.0, "ok"

    diffs = np.abs(base_arr[finite_mask] - cand_arr[finite_mask])
    max_diff = float(np.max(diffs)) if diffs.size else 0.0
    if not np.isfinite(max_diff) or max_diff > atol:
        return False, max_diff, f"Max absolute diff {max_diff:.3e} exceeds atol={atol:.3e}"
    return True, max_diff, "ok"


def _perturb_ohlcv_rows(df: pd.DataFrame, mask: np.ndarray, factor: float = 5.0) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            vals = out.loc[mask, col].astype(float) * factor
            out.loc[mask, col] = vals
    return out


def _extract_scalar_last(output: Any) -> float | None:
    if output is None:
        return None
    if isinstance(output, (pd.Series, pd.DataFrame)):
        if output.empty:
            return None
        val = output.iloc[-1] if isinstance(output, pd.Series) else output.iloc[-1, -1]
    elif isinstance(output, np.ndarray):
        if output.size == 0:
            return None
        val = output.reshape(-1)[-1]
    else:
        val = output
    if val is None or pd.isna(val):
        return None
    fval = float(val)
    return fval if math.isfinite(fval) else None


def audit_bar_index_lookahead(
    df: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], Any],
    cutoffs_ist: Sequence[datetime],
    atol: float = 1e-12,
) -> BiasAuditResult:
    """Audit bar-index lookahead via truncation, 5x future perturbation, and unclosed-bar rejection."""
    violations: list[BiasViolation] = []
    checked_cutoffs: list[datetime] = []
    max_diff_seen = 0.0

    ts_close = _resolve_ts_close_series(df, "audit_bar_index_lookahead")
    out_full = feature_fn(df.copy())

    for raw_cutoff in cutoffs_ist:
        cutoff = _ensure_tz_aware_ist(raw_cutoff, "cutoff_ist")
        checked_cutoffs.append(cutoff)

        past_mask = (ts_close <= cutoff).to_numpy(dtype=bool)
        future_mask = (ts_close > cutoff).to_numpy(dtype=bool)

        df_trunc = df.loc[past_mask].copy()
        if df_trunc.empty:
            violations.append(
                BiasViolation(
                    check_name="bar_index_lookahead",
                    code="EMPTY_TRUNCATED_SLICE",
                    message=f"No bars with ts_close_ist <= {cutoff.isoformat()}",
                    offending_timestamp=cutoff,
                )
            )
            continue

        out_trunc = feature_fn(df_trunc)
        try:
            full_past = _slice_output_by_mask(out_full, past_mask)
            ok_trunc, diff_trunc, reason_trunc = _compare_outputs(out_trunc, full_past, atol=atol)
        except Exception as exc:
            ok_trunc, diff_trunc, reason_trunc = False, float("inf"), str(exc)

        max_diff_seen = max(max_diff_seen, diff_trunc if math.isfinite(diff_trunc) else 0.0)
        if not ok_trunc:
            violations.append(
                BiasViolation(
                    check_name="bar_index_lookahead",
                    code="TRUNCATION_LOOKAHEAD",
                    message=f"Output at <= {cutoff.isoformat()} changed when future bars were included: {reason_trunc}",
                    offending_timestamp=cutoff,
                    details={"max_abs_diff": diff_trunc, "atol": atol},
                )
            )

        if future_mask.any():
            df_pert = _perturb_ohlcv_rows(df, future_mask, factor=5.0)
            out_pert = feature_fn(df_pert)
            try:
                pert_past = _slice_output_by_mask(out_pert, past_mask)
                ok_pert, diff_pert, reason_pert = _compare_outputs(out_trunc, pert_past, atol=atol)
            except Exception as exc:
                ok_pert, diff_pert, reason_pert = False, float("inf"), str(exc)

            max_diff_seen = max(max_diff_seen, diff_pert if math.isfinite(diff_pert) else 0.0)
            if not ok_pert:
                violations.append(
                    BiasViolation(
                        check_name="bar_index_lookahead",
                        code="FUTURE_PERTURBATION_LEAKAGE",
                        message=f"Output at <= {cutoff.isoformat()} changed under 5x future perturbation: {reason_pert}",
                        offending_timestamp=cutoff,
                        details={"max_abs_diff": diff_pert, "atol": atol},
                    )
                )

    # Probe unclosed bar rejection (ts_open < as_of < ts_close) via build_ohlcv_evidence_frame
    probe_as_of = checked_cutoffs[0] if checked_cutoffs else datetime(2026, 1, 15, 10, 30, tzinfo=IST_TZ)
    unclosed_df = pd.DataFrame(
        {
            "ts_open_ist": [probe_as_of - timedelta(minutes=15)],
            "ts_close_ist": [probe_as_of + timedelta(minutes=45)],
            "ts_init_ist": [probe_as_of + timedelta(minutes=45)],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000.0],
        }
    )
    unclosed_rejected = False
    try:
        build_ohlcv_evidence_frame(
            unclosed_df,
            symbol="PROBE_UNCLOSED",
            timeframe="1h",
            as_of_ist=probe_as_of,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="bias_auditor_probe",
        )
    except LookaheadViolationError:
        unclosed_rejected = True

    if not unclosed_rejected:
        violations.append(
            BiasViolation(
                check_name="bar_index_lookahead",
                code="UNCLOSED_BAR_NOT_REJECTED",
                message="build_ohlcv_evidence_frame failed to reject unclosed bar (ts_open < as_of < ts_close)",
                offending_timestamp=probe_as_of,
            )
        )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="bar_index_lookahead",
        violations=tuple(violations),
        metrics={
            "max_abs_diff": max_diff_seen,
            "unclosed_bar_rejected": unclosed_rejected,
            "cutoffs_checked": len(checked_cutoffs),
        },
        checked_cutoffs=tuple(checked_cutoffs),
    )


def _extract_panel_target(output: Any, target_sym: str) -> Any:
    if isinstance(output, Mapping) and target_sym in output:
        return output[target_sym]
    return output


def audit_cross_sectional_lookahead(
    panel_by_sym: Mapping[str, pd.DataFrame],
    target_sym: str,
    panel_feature_fn: Callable[[dict[str, pd.DataFrame]], Any],
    cutoffs_ist: Sequence[datetime],
    atol: float = 1e-12,
) -> BiasAuditResult:
    """Audit cross-sectional lookahead via 5x peer future perturbation and S_2 current-bar drop."""
    if target_sym not in panel_by_sym:
        raise ValueError(f"target_sym {target_sym!r} not found in panel_by_sym")
    peer_syms = [s for s in panel_by_sym if s != target_sym]
    if not peer_syms:
        raise ValueError("panel_by_sym must contain at least one peer symbol besides target_sym")

    violations: list[BiasViolation] = []
    checked_cutoffs: list[datetime] = []
    max_diff_seen = 0.0

    target_ts = _resolve_ts_close_series(panel_by_sym[target_sym], f"panel:{target_sym}")
    base_out = _extract_panel_target(
        panel_feature_fn({sym: df.copy() for sym, df in panel_by_sym.items()}),
        target_sym,
    )

    for raw_cutoff in cutoffs_ist:
        cutoff = _ensure_tz_aware_ist(raw_cutoff, "cutoff_ist")
        checked_cutoffs.append(cutoff)
        target_past_mask = (target_ts <= cutoff).to_numpy(dtype=bool)
        if not target_past_mask.any():
            violations.append(
                BiasViolation(
                    check_name="cross_sectional_lookahead",
                    code="EMPTY_TARGET_SLICE",
                    message=f"No {target_sym} bars <= {cutoff.isoformat()}",
                    offending_timestamp=cutoff,
                )
            )
            continue

        base_past = _slice_output_by_mask(base_out, target_past_mask)

        # 1. Perturb S_2..S_m future bars by 5x
        pert_panel: dict[str, pd.DataFrame] = {}
        for sym, df in panel_by_sym.items():
            if sym == target_sym:
                pert_panel[sym] = df.copy()
            else:
                sym_ts = _resolve_ts_close_series(df, f"panel:{sym}")
                fut_mask = (sym_ts > cutoff).to_numpy(dtype=bool)
                pert_panel[sym] = _perturb_ohlcv_rows(df, fut_mask, factor=5.0) if fut_mask.any() else df.copy()

        pert_out = _extract_panel_target(panel_feature_fn(pert_panel), target_sym)
        try:
            pert_past = _slice_output_by_mask(pert_out, target_past_mask)
            ok_pert, diff_pert, reason_pert = _compare_outputs(base_past, pert_past, atol=atol)
        except Exception as exc:
            ok_pert, diff_pert, reason_pert = False, float("inf"), str(exc)

        max_diff_seen = max(max_diff_seen, diff_pert if math.isfinite(diff_pert) else 0.0)
        if not ok_pert:
            violations.append(
                BiasViolation(
                    check_name="cross_sectional_lookahead",
                    code="PEER_FUTURE_PERTURBATION_LEAKAGE",
                    message=f"{target_sym} output <= {cutoff.isoformat()} changed on 5x peer future perturbation: {reason_pert}",
                    offending_timestamp=cutoff,
                    details={"max_abs_diff": diff_pert, "atol": atol},
                )
            )

        # 2. Drop S_2's current bar while keeping S_2's future bars
        s2 = peer_syms[0]
        s2_df = panel_by_sym[s2]
        s2_ts = _resolve_ts_close_series(s2_df, f"panel:{s2}")
        exact_mask = s2_ts == cutoff
        if exact_mask.any():
            drop_mask = exact_mask
        else:
            le_indices = s2_ts[s2_ts <= cutoff].index
            drop_mask = s2_df.index == le_indices[-1] if len(le_indices) > 0 else pd.Series(False, index=s2_df.index)

        drop_panel = {sym: df.copy() for sym, df in panel_by_sym.items()}
        drop_panel[s2] = s2_df.loc[~drop_mask].copy()

        drop_out = _extract_panel_target(panel_feature_fn(drop_panel), target_sym)
        try:
            drop_past = _slice_output_by_mask(drop_out, target_past_mask)
            ok_drop, diff_drop, reason_drop = _compare_outputs(base_past, drop_past, atol=atol)
        except Exception as exc:
            ok_drop, diff_drop, reason_drop = False, float("inf"), str(exc)

        max_diff_seen = max(max_diff_seen, diff_drop if math.isfinite(diff_drop) else 0.0)
        if not ok_drop:
            violations.append(
                BiasViolation(
                    check_name="cross_sectional_lookahead",
                    code="PEER_CURRENT_DROP_FUTURE_LEAKAGE",
                    message=f"{target_sym} output <= {cutoff.isoformat()} changed when {s2} current bar dropped with future bars retained: {reason_drop}",
                    offending_timestamp=cutoff,
                    details={"peer_symbol": s2, "max_abs_diff": diff_drop, "atol": atol},
                )
            )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="cross_sectional_lookahead",
        violations=tuple(violations),
        metrics={"max_abs_diff": max_diff_seen, "cutoffs_checked": len(checked_cutoffs)},
        checked_cutoffs=tuple(checked_cutoffs),
    )


def audit_ts_init_cutoff(
    df: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], Any],
    as_of_ist: datetime,
    atol: float = 1e-12,
) -> BiasAuditResult:
    """Audit ts_init causal cutoff invisibility and fail-closed rejection of impossible causality & naive timestamps."""
    as_of_norm = _ensure_tz_aware_ist(as_of_ist, "as_of_ist")
    violations: list[BiasViolation] = []

    base_df = df.copy()
    base_out = feature_fn(base_df)

    # 1. Inject delayed-arrival record: ts_event <= as_of < ts_init
    ts_close = _resolve_ts_close_series(base_df, "audit_ts_init_cutoff")
    last_row = base_df.iloc[[-1]].copy()
    delayed_event_ts = min(as_of_norm, ts_close.iloc[-1].to_pydatetime())
    if "ts_open_ist" in last_row.columns:
        last_row["ts_open_ist"] = delayed_event_ts - timedelta(hours=1)
    if "ts_close_ist" in last_row.columns:
        last_row["ts_close_ist"] = delayed_event_ts
    elif "timestamp" in last_row.columns:
        last_row["timestamp"] = delayed_event_ts
    last_row["ts_init_ist"] = as_of_norm + timedelta(minutes=15)
    for col in ("open", "high", "low", "close", "volume"):
        if col in last_row.columns:
            last_row[col] = last_row[col].astype(float) * 5.0

    work_df = base_df.copy()
    if "ts_init_ist" not in work_df.columns:
        work_df["ts_init_ist"] = ts_close
    delayed_df = pd.concat([work_df, last_row], ignore_index=True)

    delayed_out = feature_fn(delayed_df)
    if isinstance(delayed_out, (pd.Series, pd.DataFrame, np.ndarray)) and len(delayed_out) == len(delayed_df):
        delayed_cmp = _slice_output_by_mask(
            delayed_out, np.array([True] * len(base_df) + [False], dtype=bool)
        )
    else:
        delayed_cmp = delayed_out

    try:
        ok_delay, diff_delay, reason_delay = _compare_outputs(base_out, delayed_cmp, atol=atol)
    except Exception as exc:
        ok_delay, diff_delay, reason_delay = False, float("inf"), str(exc)

    if not ok_delay:
        violations.append(
            BiasViolation(
                check_name="ts_init_cutoff",
                code="DELAYED_TS_INIT_VISIBLE",
                message=f"Delayed-arrival record (ts_event <= as_of < ts_init) leaked into output: {reason_delay}",
                offending_timestamp=as_of_norm,
                details={"max_abs_diff": diff_delay, "atol": atol},
            )
        )

    # 2. Impossible causality (ts_init < ts_event - 2s) must fail closed
    impossible_df = pd.DataFrame(
        {
            "ts_open_ist": [as_of_norm - timedelta(hours=1)],
            "ts_close_ist": [as_of_norm],
            "ts_init_ist": [as_of_norm - timedelta(seconds=5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        }
    )
    impossible_rejected = False
    try:
        build_ohlcv_evidence_frame(
            impossible_df,
            symbol="PROBE_IMPOSSIBLE_CAUSALITY",
            timeframe="1h",
            as_of_ist=as_of_norm,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="bias_auditor_probe",
        )
    except (LookaheadViolationError, ValueError):
        impossible_rejected = True

    if not impossible_rejected:
        violations.append(
            BiasViolation(
                check_name="ts_init_cutoff",
                code="IMPOSSIBLE_CAUSALITY_NOT_REJECTED",
                message="Impossible causality (ts_init < ts_event - 2s) was not rejected",
                offending_timestamp=as_of_norm,
            )
        )

    # 3. Naive timestamps must fail closed
    naive_as_of_rejected = False
    try:
        build_ohlcv_evidence_frame(
            impossible_df.assign(ts_init_ist=[as_of_norm]),
            symbol="PROBE_NAIVE_AS_OF",
            timeframe="1h",
            as_of_ist=as_of_norm.replace(tzinfo=None),
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="bias_auditor_probe",
        )
    except (LookaheadViolationError, ValueError):
        naive_as_of_rejected = True

    naive_series_rejected = False
    try:
        naive_df = pd.DataFrame(
            {
                "ts_open_ist": [(as_of_norm - timedelta(hours=1)).replace(tzinfo=None)],
                "ts_close_ist": [as_of_norm.replace(tzinfo=None)],
                "ts_init_ist": [as_of_norm.replace(tzinfo=None)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )
        build_ohlcv_evidence_frame(
            naive_df,
            symbol="PROBE_NAIVE_SERIES",
            timeframe="1h",
            as_of_ist=as_of_norm,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name="bias_auditor_probe",
        )
    except (LookaheadViolationError, ValueError):
        naive_series_rejected = True

    if not (naive_as_of_rejected and naive_series_rejected):
        violations.append(
            BiasViolation(
                check_name="ts_init_cutoff",
                code="NAIVE_TIMESTAMP_NOT_REJECTED",
                message="Timezone-naive timestamp was not rejected fail-closed",
                offending_timestamp=as_of_norm,
                details={
                    "naive_as_of_rejected": naive_as_of_rejected,
                    "naive_series_rejected": naive_series_rejected,
                },
            )
        )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="ts_init_cutoff",
        violations=tuple(violations),
        metrics={
            "delayed_max_abs_diff": diff_delay if math.isfinite(diff_delay) else None,
            "impossible_causality_rejected": impossible_rejected,
            "naive_timestamp_rejected": naive_as_of_rejected and naive_series_rejected,
        },
        checked_cutoffs=(as_of_norm,),
    )


def audit_recursive_convergence(
    series: pd.Series,
    indicator_fn: Callable[[pd.Series], Any],
    span_or_period: int,
    windows: Sequence[int] = (50, 100, 200, 500),
    ref_window: int = 500,
    rtol: float | None = None,
) -> BiasAuditResult:
    """Audit recursive indicator convergence across truncation windows vs ref_window."""
    tol = float(rtol) if rtol is not None else _default_convergence_rtol(span_or_period)
    violations: list[BiasViolation] = []
    drifts: dict[int, float] = {}

    required_len = max(max(windows, default=0), ref_window)
    if len(series) < required_len:
        violations.append(
            BiasViolation(
                check_name="recursive_convergence",
                code="INSUFFICIENT_SERIES_LENGTH",
                message=f"Series length {len(series)} < required {required_len}",
                details={"series_len": len(series), "required_len": required_len},
            )
        )
        return BiasAuditResult(
            passed=False,
            check_name="recursive_convergence",
            violations=tuple(violations),
            metrics={"tolerance": tol, "drifts_by_window": drifts},
        )

    ref_val = _extract_scalar_last(indicator_fn(series.iloc[-ref_window:].copy()))
    if ref_val is None:
        violations.append(
            BiasViolation(
                check_name="recursive_convergence",
                code="NON_FINITE_REFERENCE",
                message=f"Reference window W={ref_window} produced None/NaN",
            )
        )
        return BiasAuditResult(
            passed=False,
            check_name="recursive_convergence",
            violations=tuple(violations),
            metrics={"tolerance": tol, "drifts_by_window": drifts},
        )

    denom = max(abs(ref_val), 1e-8)
    for w in windows:
        val_w = _extract_scalar_last(indicator_fn(series.iloc[-w:].copy()))
        if val_w is None:
            drifts[int(w)] = float("inf")
            continue
        drifts[int(w)] = abs(val_w - ref_val) / denom

    target_w = 200 if 200 in drifts else max((w for w in drifts if w < ref_window), default=ref_window)
    target_drift = drifts.get(target_w, float("inf"))
    if not math.isfinite(target_drift) or target_drift > tol:
        violations.append(
            BiasViolation(
                check_name="recursive_convergence",
                code="CONVERGENCE_DRIFT_EXCEEDED",
                message=(
                    f"Relative drift at W={target_w} is {target_drift:.3e} > tolerance {tol:.3e} "
                    f"(span_or_period={span_or_period})"
                ),
                details={
                    "target_window": target_w,
                    "drift": target_drift,
                    "tolerance": tol,
                    "ref_window": ref_window,
                },
            )
        )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="recursive_convergence",
        violations=tuple(violations),
        metrics={
            "tolerance": tol,
            "target_window": target_w,
            "target_drift": target_drift,
            "drifts_by_window": drifts,
            "ref_value": ref_val,
        },
    )


def audit_warmup_sensitivity(
    series: pd.Series,
    indicator_fn: Callable[[pd.Series], Any],
    min_warmup_bars: int,
    *,
    ref_window: int = 500,
    max_drift: float = 0.02,
    below_probe_bars: int | None = None,
    above_windows: Sequence[int] | None = None,
) -> BiasAuditResult:
    """Audit warm-up sensitivity: below min_warmup_bars must emit None/NaN; above stays within drift bounds."""
    if min_warmup_bars < 1:
        raise ValueError(f"min_warmup_bars must be >= 1, got {min_warmup_bars}")

    violations: list[BiasViolation] = []
    probe_below = below_probe_bars if below_probe_bars is not None else max(1, min_warmup_bars - 1)
    if min_warmup_bars > 1 and probe_below < min_warmup_bars:
        below_slice = series.iloc[-probe_below:].copy() if len(series) >= probe_below else series.copy()
        below_val = _extract_scalar_last(indicator_fn(below_slice))
        if below_val is not None:
            violations.append(
                BiasViolation(
                    check_name="warmup_sensitivity",
                    code="WARMUP_SILENT_FLOAT_BELOW_MIN",
                    message=(
                        f"Indicator returned finite float {below_val} with {len(below_slice)} bars "
                        f"(< min_warmup_bars={min_warmup_bars}); expected None/NaN"
                    ),
                    details={"bars": len(below_slice), "min_warmup_bars": min_warmup_bars, "value": below_val},
                )
            )

    ref_len = min(len(series), max(ref_window, min_warmup_bars))
    if ref_len < min_warmup_bars:
        violations.append(
            BiasViolation(
                check_name="warmup_sensitivity",
                code="INSUFFICIENT_SERIES_FOR_WARMUP",
                message=f"Series length {len(series)} < min_warmup_bars={min_warmup_bars}",
            )
        )
        return BiasAuditResult(
            passed=False,
            check_name="warmup_sensitivity",
            violations=tuple(violations),
            metrics={"min_warmup_bars": min_warmup_bars},
        )

    ref_val = _extract_scalar_last(indicator_fn(series.iloc[-ref_len:].copy()))
    if ref_val is None:
        violations.append(
            BiasViolation(
                check_name="warmup_sensitivity",
                code="WARMUP_REF_NONE",
                message=f"Indicator returned None/NaN at reference length {ref_len} >= min_warmup_bars",
            )
        )
        return BiasAuditResult(
            passed=False,
            check_name="warmup_sensitivity",
            violations=tuple(violations),
            metrics={"min_warmup_bars": min_warmup_bars},
        )

    if above_windows is None:
        candidates = [min_warmup_bars, min(len(series), min_warmup_bars + 50), min(len(series), min_warmup_bars + 150)]
        above_list = sorted({w for w in candidates if min_warmup_bars <= w <= len(series)})
    else:
        above_list = sorted({int(w) for w in above_windows if min_warmup_bars <= int(w) <= len(series)})

    denom = max(abs(ref_val), 1e-8)
    above_drifts: dict[int, float] = {}
    for w in above_list:
        val_w = _extract_scalar_last(indicator_fn(series.iloc[-w:].copy()))
        if val_w is None:
            violations.append(
                BiasViolation(
                    check_name="warmup_sensitivity",
                    code="WARMUP_MISSING_ABOVE_MIN",
                    message=f"Indicator returned None/NaN at W={w} >= min_warmup_bars={min_warmup_bars}",
                    details={"window": w, "min_warmup_bars": min_warmup_bars},
                )
            )
            above_drifts[w] = float("inf")
            continue
        drift = abs(val_w - ref_val) / denom
        above_drifts[w] = drift
        if drift > max_drift:
            violations.append(
                BiasViolation(
                    check_name="warmup_sensitivity",
                    code="WARMUP_DRIFT_EXCEEDED",
                    message=f"Warm-up drift at W={w} is {drift:.3e} > max_drift={max_drift:.3e}",
                    details={"window": w, "drift": drift, "max_drift": max_drift},
                )
            )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="warmup_sensitivity",
        violations=tuple(violations),
        metrics={
            "min_warmup_bars": min_warmup_bars,
            "max_drift_limit": max_drift,
            "above_drifts": above_drifts,
            "ref_value": ref_val,
        },
    )


def audit_mtf_backward_alignment(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_cols: Sequence[str],
    cutoffs_ist: Sequence[datetime],
    atol: float = 1e-12,
) -> BiasAuditResult:
    """Audit MTF backward alignment: truncation invariance, in-progress HTF perturbation, and 15:15-15:30 bar visibility."""
    violations: list[BiasViolation] = []
    checked_cutoffs: list[datetime] = []
    max_diff_seen = 0.0

    ltf_ts = _coerce_ts_series_ist(ltf_df["ts_close_ist"], "ltf.ts_close_ist")
    htf_ts = _coerce_ts_series_ist(htf_df["ts_close_ist"], "htf.ts_close_ist")

    aligned_full = align_htf_to_ltf_ist(ltf_df, htf_df, htf_cols)
    aligned_cols = [c for c in aligned_full.columns if c not in ltf_df.columns or c in htf_cols]
    # Specifically check the newly added/aligned HTF columns
    htf_out_cols = [c for c in aligned_full.columns if c != "ts_close_ist" and (c in htf_cols or c.endswith("_htf"))]
    if not htf_out_cols:
        htf_out_cols = list(htf_cols)

    for raw_cutoff in cutoffs_ist:
        cutoff = _ensure_tz_aware_ist(raw_cutoff, "cutoff_ist")
        checked_cutoffs.append(cutoff)

        ltf_past_mask = (ltf_ts <= cutoff).to_numpy(dtype=bool)
        htf_past_mask = (htf_ts <= cutoff).to_numpy(dtype=bool)
        htf_fut_mask = (htf_ts > cutoff).to_numpy(dtype=bool)

        ltf_cut = ltf_df.loc[ltf_past_mask].copy()
        htf_cut = htf_df.loc[htf_past_mask].copy()
        if ltf_cut.empty or htf_cut.empty:
            continue

        aligned_cut = align_htf_to_ltf_ist(ltf_cut, htf_cut, htf_cols)
        full_past = aligned_full.loc[ltf_ts <= cutoff, htf_out_cols].reset_index(drop=True)
        cut_sub = aligned_cut[htf_out_cols].reset_index(drop=True)

        ok_trunc, diff_trunc, reason_trunc = _compare_outputs(cut_sub, full_past, atol=atol)
        max_diff_seen = max(max_diff_seen, diff_trunc if math.isfinite(diff_trunc) else 0.0)
        if not ok_trunc:
            violations.append(
                BiasViolation(
                    check_name="mtf_backward_alignment",
                    code="MTF_TRUNCATION_LEAKAGE",
                    message=f"HTF->LTF alignment <= {cutoff.isoformat()} changed on truncation: {reason_trunc}",
                    offending_timestamp=cutoff,
                    details={"max_abs_diff": diff_trunc, "atol": atol},
                )
            )

        if htf_fut_mask.any():
            htf_pert = htf_df.copy()
            for col in htf_cols:
                if col != "ts_close_ist" and pd.api.types.is_numeric_dtype(htf_pert[col]):
                    htf_pert.loc[htf_fut_mask, col] = htf_pert.loc[htf_fut_mask, col].astype(float) * 5.0 + 100.0
            aligned_pert = align_htf_to_ltf_ist(ltf_df, htf_pert, htf_cols)
            pert_sub = aligned_pert.loc[ltf_ts <= cutoff, htf_out_cols].reset_index(drop=True)
            ok_pert, diff_pert, reason_pert = _compare_outputs(cut_sub, pert_sub, atol=atol)
            max_diff_seen = max(max_diff_seen, diff_pert if math.isfinite(diff_pert) else 0.0)
            if not ok_pert:
                violations.append(
                    BiasViolation(
                        check_name="mtf_backward_alignment",
                        code="MTF_IN_PROGRESS_HTF_LEAKAGE",
                        message=f"In-progress/future HTF bar leaked into <= {cutoff.isoformat()}: {reason_pert}",
                        offending_timestamp=cutoff,
                        details={"max_abs_diff": diff_pert, "atol": atol},
                    )
                )

    # Check 15:15-15:30 short closing bar rule and visibility if present
    closing_sub_bars = expected_5m_sub_bars_for_bucket(
        datetime(2026, 1, 15, 15, 15, tzinfo=IST_TZ), 60
    )
    if closing_sub_bars != 3:
        violations.append(
            BiasViolation(
                check_name="mtf_backward_alignment",
                code="CLOSING_BAR_SUB_BAR_COUNT_WRONG",
                message=f"Expected 3 sub-bars for 15:15-15:30 60m bucket, got {closing_sub_bars}",
            )
        )

    htf_1530_mask = (htf_ts.dt.hour == 15) & (htf_ts.dt.minute == 30)
    if htf_1530_mask.any():
        closing_ts = htf_ts.loc[htf_1530_mask].iloc[0].to_pydatetime()
        t_1525 = closing_ts - timedelta(minutes=5)
        ltf_1525_mask = ltf_ts == t_1525
        ltf_1530_mask = ltf_ts == closing_ts
        if ltf_1525_mask.any():
            htf_pert_close = htf_df.copy()
            for col in htf_cols:
                if col != "ts_close_ist" and pd.api.types.is_numeric_dtype(htf_pert_close[col]):
                    htf_pert_close.loc[htf_1530_mask, col] = (
                        htf_pert_close.loc[htf_1530_mask, col].astype(float) * 5.0 + 999.0
                    )
            aligned_pert_close = align_htf_to_ltf_ist(ltf_df, htf_pert_close, htf_cols)
            base_1525 = aligned_full.loc[ltf_1525_mask, htf_out_cols].reset_index(drop=True)
            pert_1525 = aligned_pert_close.loc[ltf_1525_mask, htf_out_cols].reset_index(drop=True)
            ok_1525, diff_1525, reason_1525 = _compare_outputs(base_1525, pert_1525, atol=atol)
            if not ok_1525:
                violations.append(
                    BiasViolation(
                        check_name="mtf_backward_alignment",
                        code="CLOSING_BAR_VISIBLE_AT_1525",
                        message=f"15:30 closing bar leaked into 15:25 LTF bar: {reason_1525}",
                        offending_timestamp=t_1525,
                    )
                )
        if ltf_1530_mask.any():
            # At 15:30, the 15:30 HTF bar MUST be visible
            htf_row_1530 = htf_df.loc[htf_1530_mask].iloc[[0]]
            for orig_col, out_col in zip(
                [c for c in htf_cols if c != "ts_close_ist"], htf_out_cols
            ):
                if pd.api.types.is_numeric_dtype(htf_row_1530[orig_col]):
                    expected_val = float(htf_row_1530[orig_col].iloc[0])
                    actual_val = float(aligned_full.loc[ltf_1530_mask, out_col].iloc[0])
                    if not math.isfinite(actual_val) or abs(actual_val - expected_val) > atol:
                        violations.append(
                            BiasViolation(
                                check_name="mtf_backward_alignment",
                                code="CLOSING_BAR_INVISIBLE_AT_1530",
                                message=f"15:30 closing bar {orig_col} not visible at 15:30 (expected {expected_val}, got {actual_val})",
                                offending_timestamp=closing_ts,
                            )
                        )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="mtf_backward_alignment",
        violations=tuple(violations),
        metrics={
            "max_abs_diff": max_diff_seen,
            "closing_bar_sub_bars": closing_sub_bars,
            "cutoffs_checked": len(checked_cutoffs),
        },
        checked_cutoffs=tuple(checked_cutoffs),
    )


def _call_event_filter(
    event_filter_fn: Callable[..., Any],
    df: pd.DataFrame,
    as_of_ist: datetime,
) -> Any:
    sig = inspect.signature(event_filter_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_varargs or len(positional) >= 2:
        return event_filter_fn(df, as_of_ist)
    return event_filter_fn(df)


def audit_future_event_exclusion(
    events_df: pd.DataFrame,
    event_filter_fn: Callable[..., Any],
    as_of_ist: datetime,
    atol: float = 1e-12,
) -> BiasAuditResult:
    """Audit future news/event exclusion across 4 adversarial event classes."""
    as_of_norm = _ensure_tz_aware_ist(as_of_ist, "as_of_ist")
    violations: list[BiasViolation] = []

    pub_col = next(
        (c for c in ("published_at_ist", "published_at", "ts_event_ist", "timestamp") if c in events_df.columns),
        "published_at_ist",
    )
    col_col = next(
        (c for c in ("collected_at_ist", "collected_at", "ts_init_ist") if c in events_df.columns),
        "collected_at_ist",
    )
    sent_col = next(
        (c for c in ("sentiment_score", "sentiment", "score") if c in events_df.columns),
        None,
    )

    base_out = _call_event_filter(event_filter_fn, events_df.copy(), as_of_norm)

    template = events_df.iloc[0].to_dict() if not events_df.empty else {}

    def _make_row(pub_val: Any, col_val: Any, sentiment_val: float) -> dict[str, Any]:
        row = dict(template)
        row[pub_col] = pub_val
        row[col_col] = col_val
        if sent_col is not None:
            row[sent_col] = sentiment_val
        return row

    adversarial_rows = [
        # 1. pub > as_of
        _make_row(as_of_norm + timedelta(minutes=10), as_of_norm + timedelta(minutes=11), 99.0),
        # 2. pub <= as_of < col
        _make_row(as_of_norm - timedelta(minutes=5), as_of_norm + timedelta(minutes=5), -99.0),
        # 3. missing / naive timestamp
        _make_row(None, as_of_norm.replace(tzinfo=None), 50.0),
        # 4. extreme sentiment with future timestamp
        _make_row(as_of_norm + timedelta(hours=1), as_of_norm + timedelta(hours=2), 1e6),
    ]

    adv_df = pd.concat([events_df.copy(), pd.DataFrame(adversarial_rows)], ignore_index=True)
    adv_out = _call_event_filter(event_filter_fn, adv_df, as_of_norm)

    excluded_count = len(adversarial_rows)
    if isinstance(base_out, pd.DataFrame) and isinstance(adv_out, pd.DataFrame):
        excluded_count = len(adv_df) - len(adv_out)
        if excluded_count <= 0 or len(adv_out) != len(base_out):
            violations.append(
                BiasViolation(
                    check_name="future_event_exclusion",
                    code="ADVERSARIAL_EVENTS_NOT_EXCLUDED",
                    message=(
                        f"Expected {len(base_out)} events after excluding 4 adversarial classes, "
                        f"got {len(adv_out)} (excluded_count={excluded_count})"
                    ),
                    offending_timestamp=as_of_norm,
                )
            )
        else:
            ok, diff, reason = _compare_outputs(
                base_out.reset_index(drop=True), adv_out.reset_index(drop=True), atol=atol
            )
            if not ok:
                violations.append(
                    BiasViolation(
                        check_name="future_event_exclusion",
                        code="ADVERSARIAL_EVENT_OUTPUT_DRIFT",
                        message=f"Filtered event output drifted after adversarial injection: {reason}",
                        offending_timestamp=as_of_norm,
                        details={"max_abs_diff": diff, "atol": atol},
                    )
                )
    else:
        try:
            ok, diff, reason = _compare_outputs(base_out, adv_out, atol=atol)
        except Exception as exc:
            ok, diff, reason = False, float("inf"), str(exc)
        if not ok:
            violations.append(
                BiasViolation(
                    check_name="future_event_exclusion",
                    code="ADVERSARIAL_EVENT_LEAKAGE",
                    message=f"Event output changed under 4 adversarial event classes: {reason}",
                    offending_timestamp=as_of_norm,
                    details={"max_abs_diff": diff, "atol": atol},
                )
            )

    return BiasAuditResult(
        passed=len(violations) == 0,
        check_name="future_event_exclusion",
        violations=tuple(violations),
        metrics={
            "injected_count": len(adversarial_rows),
            "excluded_count": excluded_count,
        },
        checked_cutoffs=(as_of_norm,),
    )
