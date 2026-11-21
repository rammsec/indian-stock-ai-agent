"""Phase 11.3: Strict Sealed Holdout Ownership, Lifecycle & Split Validation Guard."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from ..config import IST_TZ

HoldoutState = Literal["AVAILABLE", "SEALED", "EXPOSED", "RETIRED"]
ReservationStatus = Literal[
    "ACTIVE_RESERVATION",
    "EVALUATION_IN_PROGRESS",
    "EVALUATION_COMPLETED",
    "FAILED_EVALUATION",
    "RELEASED_RESERVATION",
]
REGISTRY_VERSION = "11.3.3"
SEALED_HOLDOUT_REGISTRY_PATH = Path("data/lab/sealed_holdout_registry.json")


def _get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


class SealedHoldoutGuard:
    """Manages strict AVAILABLE -> SEALED -> EXPOSED -> RETIRED holdout lifecycle with cycle ownership."""

    def __init__(self, registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH) -> None:
        self.registry_path = registry_path

    @contextmanager
    def _locked_registry(self) -> Iterator[None]:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.registry_path.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(reg, dict):
                raise ValueError("Corrupted holdout registry: root must be a JSON object")
        else:
            reg = {
                "registry_version": REGISTRY_VERSION,
                "sealed_exposures": [],
                "sealed_dates": [],
                "records": {},
                "recovery_log": [],
            }

        reg["registry_version"] = REGISTRY_VERSION
        reg.setdefault("sealed_exposures", [])
        reg.setdefault("sealed_dates", [])
        reg.setdefault("records", {})
        reg.setdefault("recovery_log", [])
        self._migrate_legacy_exposures(reg)
        return reg

    def _migrate_legacy_exposures(self, reg: dict[str, Any]) -> None:
        records: dict[str, dict[str, Any]] = reg["records"]
        for exp in reg.get("sealed_exposures", []):
            cid = str(exp.get("cycle_id", "LEGACY"))
            ts = str(exp.get("recorded_at_ist", ""))
            for d_str in exp.get("unseen_dates", []):
                if d_str not in records:
                    records[d_str] = {
                        "date": d_str,
                        "state": "EXPOSED",
                        "reservation_status": "EVALUATION_COMPLETED",
                        "owning_cycle_id": cid,
                        "reservation_timestamp": ts,
                        "first_sealed_cycle": cid,
                        "first_exposed_cycle": cid,
                        "exposure_timestamp": ts,
                        "dataset_fingerprint": str(exp.get("dataset_fingerprint", "")),
                        "strategy_ids_evaluated": list(exp.get("strategy_ids_evaluated", [])),
                        "previous_exposure_count": 1,
                        "data_accessed": True,
                        "reason_for_retirement": None,
                        "registry_version": REGISTRY_VERSION,
                    }
                else:
                    rec = records[d_str]
                    rec.setdefault("owning_cycle_id", rec.get("first_sealed_cycle") or cid)
                    rec.setdefault("reservation_timestamp", rec.get("exposure_timestamp") or ts)
                    rec.setdefault(
                        "reservation_status",
                        "EVALUATION_COMPLETED" if rec.get("state") == "EXPOSED" else "ACTIVE_RESERVATION",
                    )
                    rec.setdefault("data_accessed", rec.get("state") == "EXPOSED")
                    rec["registry_version"] = REGISTRY_VERSION

    def save_registry(self, reg: dict[str, Any]) -> None:
        reg["registry_version"] = REGISTRY_VERSION
        records = reg.get("records", {})
        reg["sealed_dates"] = sorted(
            d for d, rec in records.items() if rec.get("state") in ("SEALED", "EXPOSED", "RETIRED")
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.registry_path)

    def get_date_state(self, d: date | str) -> HoldoutState:
        d_str = d.isoformat() if isinstance(d, date) else str(d)
        reg = self.load_registry()
        rec = reg["records"].get(d_str)
        if not rec:
            return "AVAILABLE"
        return rec.get("state", "AVAILABLE")  # type: ignore[return-value]

    def seal_dates(
        self,
        cycle_id: str,
        dates: list[date],
        dataset_fingerprint: str = "",
    ) -> dict[str, Any]:
        """Atomically reserve AVAILABLE dates as SEALED with explicit owning_cycle_id."""
        with self._locked_registry():
            if len(dates) != len(set(dates)):
                return {
                    "cycle_id": cycle_id,
                    "sealed_count": 0,
                    "blocked_dates": [{"date": "DUPLICATE", "state": "DUPLICATE_IN_REQUEST", "owning_cycle_id": cycle_id}],
                    "ownership_valid": False,
                    "independent_holdout_valid": False,
                    "reservation_status": "REJECTED_DUPLICATE_DATES",
                }
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            blocked: list[dict[str, str]] = []
            sealed_now: list[str] = []

            # Atomic pre-check: reject entire reservation if any requested date is not AVAILABLE
            for d in dates:
                d_str = d.isoformat()
                rec = records.get(d_str)
                if rec and rec.get("state") != "AVAILABLE":
                    blocked.append({
                        "date": d_str,
                        "state": rec["state"],
                        "owning_cycle_id": str(rec.get("owning_cycle_id") or ""),
                    })

            if blocked or not dates:
                return {
                    "cycle_id": cycle_id,
                    "sealed_count": 0,
                    "blocked_dates": blocked,
                    "ownership_valid": False,
                    "independent_holdout_valid": False,
                    "reservation_status": "ACTIVE_RESERVATION" if not blocked else "REJECTED_RESERVATION",
                }

            for d in dates:
                d_str = d.isoformat()
                records[d_str] = {
                    "date": d_str,
                    "state": "SEALED",
                    "reservation_status": "ACTIVE_RESERVATION",
                    "owning_cycle_id": cycle_id,
                    "reservation_timestamp": now_ist,
                    "first_sealed_cycle": cycle_id,
                    "first_exposed_cycle": None,
                    "exposure_timestamp": None,
                    "dataset_fingerprint": dataset_fingerprint,
                    "strategy_ids_evaluated": [],
                    "previous_exposure_count": 0,
                    "reason_for_retirement": None,
                    "registry_version": REGISTRY_VERSION,
                }
                sealed_now.append(d_str)

            self.save_registry(reg)
            return {
                "cycle_id": cycle_id,
                "sealed_count": len(sealed_now),
                "blocked_dates": [],
                "ownership_valid": True,
                "independent_holdout_valid": True,
                "reservation_timestamp": now_ist,
                "reservation_status": "ACTIVE_RESERVATION",
            }

    def record_exposure(
        self,
        cycle_id: str,
        unseen_dates: list[date],
        notes: str = "",
        dataset_fingerprint: str = "",
        strategy_ids: list[str] | None = None,
        evaluation_ref: str = "",
    ) -> dict[str, Any]:
        """Record final-test exposure. Enforces ownership on SEALED dates and flags EXPOSED/RETIRED reuse."""
        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            iso_dates = [d.isoformat() for d in unseen_dates]
            strat_list = list(strategy_ids or [])

            reused_exposed: list[str] = []
            ownership_violations: list[str] = []
            fingerprint_mismatches: list[str] = []
            newly_exposed: list[str] = []

            for d_str in iso_dates:
                rec = records.get(d_str)
                if rec is None:
                    records[d_str] = {
                        "date": d_str,
                        "state": "EXPOSED",
                        "reservation_status": "EVALUATION_COMPLETED",
                        "owning_cycle_id": cycle_id,
                        "reservation_timestamp": now_ist,
                        "first_sealed_cycle": cycle_id,
                        "first_exposed_cycle": cycle_id,
                        "exposure_timestamp": now_ist,
                        "dataset_fingerprint": dataset_fingerprint,
                        "strategy_ids_evaluated": strat_list,
                        "previous_exposure_count": 1,
                        "reason_for_retirement": None,
                        "registry_version": REGISTRY_VERSION,
                    }
                    newly_exposed.append(d_str)
                else:
                    prev_state = rec.get("state", "AVAILABLE")
                    owner = rec.get("owning_cycle_id") or rec.get("first_sealed_cycle")
                    if prev_state in ("EXPOSED", "RETIRED"):
                        reused_exposed.append(d_str)
                        rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 1)) + 1
                    elif prev_state == "SEALED":
                        if owner and owner != cycle_id:
                            ownership_violations.append(d_str)
                        else:
                            rec["state"] = "EXPOSED"
                            rec["reservation_status"] = "EVALUATION_COMPLETED"
                            rec["first_exposed_cycle"] = rec.get("first_exposed_cycle") or cycle_id
                            rec["exposure_timestamp"] = rec.get("exposure_timestamp") or now_ist
                            rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 0)) + 1
                            newly_exposed.append(d_str)
                    else:
                        rec["state"] = "EXPOSED"
                        rec["reservation_status"] = "EVALUATION_COMPLETED"
                        rec["owning_cycle_id"] = cycle_id
                        rec["first_exposed_cycle"] = cycle_id
                        rec["exposure_timestamp"] = now_ist
                        rec["previous_exposure_count"] = 1
                        newly_exposed.append(d_str)

                    if dataset_fingerprint and rec.get("dataset_fingerprint") and rec["dataset_fingerprint"] != dataset_fingerprint:
                        fingerprint_mismatches.append(d_str)
                    if dataset_fingerprint and not rec.get("dataset_fingerprint"):
                        rec["dataset_fingerprint"] = dataset_fingerprint
                    existing_strats = set(rec.get("strategy_ids_evaluated", []))
                    rec["strategy_ids_evaluated"] = sorted(existing_strats | set(strat_list))
                    rec["registry_version"] = REGISTRY_VERSION

        leakage = len(reused_exposed) > 0 or len(fingerprint_mismatches) > 0 or len(ownership_violations) > 0
        independent_valid = (len(iso_dates) > 0) and not leakage

        reg["sealed_exposures"].append({
            "cycle_id": cycle_id,
            "recorded_at_ist": now_ist,
            "commit_sha": _get_git_commit_sha(),
            "unseen_dates": iso_dates,
            "newly_sealed_count": len(newly_exposed),
            "previously_sealed_count": len(reused_exposed),
            "ownership_violations": ownership_violations,
            "leakage_detected": leakage,
            "independent_holdout_valid": independent_valid,
            "fingerprint_mismatches": fingerprint_mismatches,
            "dataset_fingerprint": dataset_fingerprint,
            "strategy_ids_evaluated": strat_list,
            "notes": notes,
        })
        self.save_registry(reg)

        status = "VALID_UNSEEN"
        if ownership_violations:
            status = "HOLDOUT_OWNERSHIP_VIOLATION"
        elif reused_exposed:
            status = "EXPOSED_HOLDOUT_REUSED"
        elif fingerprint_mismatches:
            status = "DATASET_FINGERPRINT_MISMATCH"
        elif not iso_dates:
            status = "INSUFFICIENT_UNSEEN_DATA"

        return {
            "total_sealed_dates": len(reg["sealed_dates"]),
            "current_unseen_dates": len(iso_dates),
            "newly_exposed_dates": len(newly_exposed),
            "previously_exposed_dates": len(reused_exposed),
            "reused_dates": reused_exposed,
            "ownership_violations": ownership_violations,
            "fingerprint_mismatches": fingerprint_mismatches,
            "ownership_valid": len(ownership_violations) == 0,
            "leakage_detected": leakage,
            "independent_holdout_valid": independent_valid,
            "status": status,
        }

    def check_tuning_leakage(self, tuning_dates: list[date]) -> list[str]:
        """Return any dates in tuning_dates that are SEALED, EXPOSED, or RETIRED."""
        reg = self.load_registry()
        records = reg["records"]
        return [
            d.isoformat()
            for d in tuning_dates
            if records.get(d.isoformat(), {}).get("state", "AVAILABLE") in ("SEALED", "EXPOSED", "RETIRED")
        ]

    def validate_cycle_splits(
        self,
        train_dates: list[date],
        val_dates: list[date],
        unseen_dates: list[date],
        cycle_id: str | None = None,
        dataset_fingerprint: str = "",
    ) -> dict[str, Any]:
        """Validate train, validation, and unseen splits against the holdout registry & ownership rules."""
        try:
            reg = self.load_registry()
        except Exception as exc:
            return {
                "independent_holdout_valid": False,
                "ownership_valid": False,
                "split_valid": False,
                "dataset_fingerprint_valid": False,
                "status": "REGISTRY_CORRUPTED",
                "error": str(exc),
                "leakage_detected": True,
            }

        records = reg["records"]
        train_set = {d.isoformat() for d in train_dates}
        val_set = {d.isoformat() for d in val_dates}
        unseen_set = {d.isoformat() for d in unseen_dates}

        overlap_tv = sorted(train_set & val_set)
        overlap_tu = sorted(train_set & unseen_set)
        overlap_vu = sorted(val_set & unseen_set)
        split_valid = not (overlap_tv or overlap_tu or overlap_vu) and bool(train_dates) and bool(unseen_dates)

        train_violations = [
            d.isoformat() for d in train_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") != "AVAILABLE"
        ]
        val_violations = [
            d.isoformat() for d in val_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") != "AVAILABLE"
        ]
        unseen_reused = [
            d.isoformat() for d in unseen_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") in ("EXPOSED", "RETIRED")
        ]
        ownership_violations = [
            d.isoformat()
            for d in unseen_dates
            if records.get(d.isoformat(), {}).get("state") == "SEALED"
            and cycle_id is not None
            and records.get(d.isoformat(), {}).get("owning_cycle_id") != cycle_id
        ]
        fp_mismatches = [
            d.isoformat()
            for d in unseen_dates
            if dataset_fingerprint
            and records.get(d.isoformat(), {}).get("dataset_fingerprint")
            and records.get(d.isoformat(), {}).get("dataset_fingerprint") != dataset_fingerprint
        ]
        untouched_unseen = [
            d.isoformat()
            for d in unseen_dates
            if records.get(d.isoformat(), {}).get("state", "AVAILABLE") == "AVAILABLE"
            or (
                records.get(d.isoformat(), {}).get("state") == "SEALED"
                and (cycle_id is None or records.get(d.isoformat(), {}).get("owning_cycle_id") == cycle_id)
            )
        ]

        ownership_valid = len(ownership_violations) == 0
        fp_valid = len(fp_mismatches) == 0

        if ownership_violations:
            status = "HOLDOUT_OWNERSHIP_VIOLATION"
            independent_valid = False
        elif not unseen_dates or not untouched_unseen:
            status = "EXPOSED_HOLDOUT_REUSED" if unseen_reused else "INSUFFICIENT_UNSEEN_DATA"
            independent_valid = False
        elif unseen_reused or train_violations or val_violations or not split_valid or not fp_valid:
            status = "EXPOSED_HOLDOUT_REUSED" if unseen_reused else ("DATASET_FINGERPRINT_MISMATCH" if not fp_valid else "SPLIT_VIOLATION")
            independent_valid = False
        else:
            status = "VALID_UNSEEN"
            independent_valid = True

        return {
            "independent_holdout_valid": independent_valid,
            "ownership_valid": ownership_valid,
            "split_valid": split_valid,
            "dataset_fingerprint_valid": fp_valid,
            "status": status,
            "train_violations": train_violations,
            "validation_violations": val_violations,
            "unseen_reused_dates": unseen_reused,
            "ownership_violations": ownership_violations,
            "fingerprint_mismatches": fp_mismatches,
            "overlap_dates": sorted(set(overlap_tv + overlap_tu + overlap_vu)),
            "untouched_unseen_dates": untouched_unseen,
            "leakage_detected": bool(train_violations or val_violations or unseen_reused or ownership_violations or not fp_valid),
        }

    def retire_dates(self, dates: list[date], reason: str) -> int:
        """Mark dates as RETIRED so they cannot be used for future independent validation."""
        reg = self.load_registry()
        records = reg["records"]
        count = 0
        for d in dates:
            d_str = d.isoformat()
            rec = records.setdefault(
                d_str,
                {
                    "date": d_str,
                    "state": "RETIRED",
                    "owning_cycle_id": None,
                    "reservation_timestamp": None,
                    "first_sealed_cycle": None,
                    "first_exposed_cycle": None,
                    "exposure_timestamp": None,
                    "dataset_fingerprint": "",
                    "strategy_ids_evaluated": [],
                    "previous_exposure_count": 0,
                    "reason_for_retirement": reason,
                    "registry_version": REGISTRY_VERSION,
                },
            )
            rec["state"] = "RETIRED"
            rec["reason_for_retirement"] = reason
            rec["registry_version"] = REGISTRY_VERSION
            count += 1
        self.save_registry(reg)
        return count

    def mark_evaluation_in_progress(
        self,
        cycle_id: str,
        dates: list[date],
        dataset_fingerprint: str = "",
        evaluation_type: str = "FORMAL_QUALIFICATION",
    ) -> dict[str, Any]:
        """Atomically validate ownership + fingerprint and transition SEALED -> EVALUATION_IN_PROGRESS."""
        if not dates or len(dates) != len(set(dates)):
            return {"ok": False, "updated": [], "rejected": ["EMPTY_OR_DUPLICATE_DATES"], "reason": "Empty or duplicate dates"}

        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            rejected: list[dict[str, str]] = []

            for d in dates:
                d_str = d.isoformat()
                rec = records.get(d_str)
                if not rec or rec.get("state") != "SEALED":
                    rejected.append({"date": d_str, "reason": f"Illegal state: {rec.get('state') if rec else 'UNRESERVED'}"})
                elif rec.get("owning_cycle_id") != cycle_id:
                    rejected.append({"date": d_str, "reason": f"Owned by {rec.get('owning_cycle_id')}, not {cycle_id}"})
                elif dataset_fingerprint and rec.get("dataset_fingerprint") and rec["dataset_fingerprint"] != dataset_fingerprint:
                    rejected.append({"date": d_str, "reason": "Dataset fingerprint mismatch"})

            if rejected:
                return {"ok": False, "updated": [], "rejected": rejected, "reason": rejected[0]["reason"]}

            updated: list[str] = []
            for d in dates:
                d_str = d.isoformat()
                rec = records[d_str]
                rec["reservation_status"] = "EVALUATION_IN_PROGRESS"
                rec["evaluation_type"] = evaluation_type
                rec["evaluation_started_at"] = now_ist
                rec["data_accessed"] = True
                rec["registry_version"] = REGISTRY_VERSION
                updated.append(d_str)

            self.save_registry(reg)
            return {"ok": True, "updated": updated, "rejected": [], "started_at": now_ist}

    def complete_evaluation(
        self,
        cycle_id: str,
        evaluation_id: str,
        evaluation_type: str,
        consumed_dates: list[date],
        strategy_ids: list[str],
        dataset_fingerprint: str = "",
        evaluation_ref: str = "",
    ) -> dict[str, Any]:
        """Atomically validate lifecycle & transition consumed SEALED dates to EXPOSED under single lock."""
        now_ist = datetime.now(tz=IST_TZ).isoformat()
        strat_list = list(strategy_ids or [])

        # Check for empty dates or duplicates in the request
        has_duplicates = len(consumed_dates) != len(set(consumed_dates))
        is_empty = len(consumed_dates) == 0

        if is_empty or has_duplicates:
            reason = "No consumed dates provided in completion request" if is_empty else "Duplicate consumed dates in completion request"
            err_detail = {"date": "EMPTY" if is_empty else "DUPLICATE", "reason": reason}
            return {
                "ok": False,
                "completion_success": False,
                "evaluation_id": evaluation_id,
                "cycle_id": cycle_id,
                "evaluation_type": evaluation_type,
                "holdout_consumed": False,
                "holdout_state": "SEALED",
                "completed_at": now_ist,
                "invalid_dates": [err_detail],
                "failure_reasons": [reason],
                "ownership_violations": [],
                "fingerprint_mismatches": [],
                "lifecycle_violations": ["DUPLICATE_DATES" if has_duplicates else "EMPTY_DATES"],
                "reused_dates": [],
                "ownership_valid": not has_duplicates,
                "independent_holdout_valid": False,
                "status": "DUPLICATE_CONSUMED_DATES" if has_duplicates else "INSUFFICIENT_UNSEEN_DATA",
            }

        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            iso_dates = [d.isoformat() for d in consumed_dates]
            normalized_dates = sorted(iso_dates)
            normalized_strats = sorted(strat_list)

            # Idempotency check: if evaluation_id already exists in evaluation_history,
            # either replay the original authoritative result (exact identity match) or
            # reject with EVALUATION_ID_CONFLICT without mutating the registry.
            existing_hist = next(
                (h for h in reg.get("evaluation_history", []) if h.get("evaluation_id") == evaluation_id),
                None,
            )
            if existing_hist is not None:
                conflict_fields: list[str] = []
                if existing_hist.get("cycle_id") != cycle_id:
                    conflict_fields.append("cycle_id")
                if existing_hist.get("evaluation_type") != evaluation_type:
                    conflict_fields.append("evaluation_type")
                if sorted(existing_hist.get("consumed_dates", [])) != normalized_dates:
                    conflict_fields.append("consumed_dates")
                if existing_hist.get("dataset_fingerprint", "") != dataset_fingerprint:
                    conflict_fields.append("dataset_fingerprint")
                if sorted(existing_hist.get("strategy_ids", [])) != normalized_strats:
                    conflict_fields.append("strategy_ids")

                if conflict_fields:
                    conflict_msg = (
                        f"EVALUATION_ID_CONFLICT: evaluation_id '{evaluation_id}' already exists with different "
                        f"parameters ({', '.join(conflict_fields)})"
                    )
                    return {
                        "ok": False,
                        "completion_success": False,
                        "evaluation_id": evaluation_id,
                        "cycle_id": cycle_id,
                        "evaluation_type": evaluation_type,
                        "holdout_consumed": False,
                        "holdout_state": existing_hist.get("holdout_state")
                        or existing_hist.get("final_lifecycle_state", "SEALED"),
                        "exposed_dates": [],
                        "newly_exposed_dates": 0,
                        "completed_at": now_ist,
                        "invalid_dates": [{"date": "CONFLICT", "reason": conflict_msg}],
                        "failure_reasons": [conflict_msg],
                        "ownership_violations": [],
                        "fingerprint_mismatches": [],
                        "lifecycle_violations": ["EVALUATION_ID_CONFLICT"],
                        "reused_dates": [],
                        "ownership_valid": False,
                        "independent_holdout_valid": False,
                        "leakage_detected": True,
                        "idempotent_replay": False,
                        "conflict_fields": conflict_fields,
                        "status": "EVALUATION_ID_CONFLICT",
                    }

                orig_ok = bool(existing_hist.get("ok", existing_hist.get("completion_success", False)))
                orig_completion_success = bool(existing_hist.get("completion_success", orig_ok))
                orig_eval_status = existing_hist.get(
                    "evaluation_status", "EVALUATION_COMPLETED" if orig_ok else "FAILED_EVALUATION"
                )
                orig_holdout_state = (
                    existing_hist.get("holdout_state")
                    or existing_hist.get("final_lifecycle_state")
                    or ("EXPOSED" if orig_ok else "SEALED")
                )
                orig_exposed_dates = list(
                    existing_hist.get(
                        "exposed_dates",
                        existing_hist.get("consumed_dates", []) if orig_ok else [],
                    )
                )
                orig_holdout_consumed = bool(
                    existing_hist.get("holdout_consumed", orig_ok or bool(orig_exposed_dates))
                )
                orig_failure_reasons = list(existing_hist.get("failure_reasons", []))
                orig_completed_at = existing_hist.get("completed_at", now_ist)
                orig_status = existing_hist.get("status") or ("VALID_UNSEEN" if orig_ok else "FAILED_EVALUATION")
                orig_invalid_dates = list(existing_hist.get("invalid_dates", []))
                orig_ownership_violations = list(existing_hist.get("ownership_violations", []))
                orig_fingerprint_mismatches = list(existing_hist.get("fingerprint_mismatches", []))
                orig_lifecycle_violations = list(existing_hist.get("lifecycle_violations", []))
                orig_reused_dates = list(existing_hist.get("reused_dates", []))

                return {
                    "ok": orig_ok,
                    "completion_success": orig_completion_success,
                    "evaluation_id": evaluation_id,
                    "cycle_id": cycle_id,
                    "evaluation_type": evaluation_type,
                    "evaluation_status": orig_eval_status,
                    "holdout_consumed": orig_holdout_consumed,
                    "holdout_state": orig_holdout_state,
                    "exposed_dates": orig_exposed_dates,
                    "newly_exposed_dates": len(orig_exposed_dates),
                    "completed_at": orig_completed_at,
                    "total_sealed_dates": len(reg.get("sealed_dates", [])),
                    "current_unseen_dates": len(iso_dates),
                    "previously_exposed_dates": 0 if orig_ok else len(orig_reused_dates),
                    "reused_dates": orig_reused_dates,
                    "ownership_violations": orig_ownership_violations,
                    "fingerprint_mismatches": orig_fingerprint_mismatches,
                    "lifecycle_violations": orig_lifecycle_violations,
                    "invalid_dates": orig_invalid_dates,
                    "failure_reasons": orig_failure_reasons,
                    "ownership_valid": len(orig_ownership_violations) == 0,
                    "leakage_detected": not orig_ok,
                    "independent_holdout_valid": orig_ok,
                    "idempotent_replay": True,
                    "status": orig_status,
                }

            invalid_dates: list[dict[str, str]] = []
            failure_reasons: list[str] = []
            ownership_violations: list[str] = []
            fingerprint_mismatches: list[str] = []
            lifecycle_violations: list[str] = []
            reused_dates: list[str] = []

            # 1-4. Validate every consumed date: exists, SEALED, owner match, EVALUATION_IN_PROGRESS, fingerprint match, not exposed/retired
            for d_str in iso_dates:
                rec = records.get(d_str)
                if rec is None:
                    msg = f"Date {d_str} does not exist in registry"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    lifecycle_violations.append(d_str)
                    continue

                curr_state = rec.get("state")
                owner = rec.get("owning_cycle_id")
                res_status = rec.get("reservation_status")
                rec_fp = rec.get("dataset_fingerprint", "")

                # Must not be already EXPOSED or RETIRED
                if curr_state in ("EXPOSED", "RETIRED"):
                    msg = f"Date {d_str} has already been {curr_state}"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    reused_dates.append(d_str)
                elif curr_state != "SEALED":
                    msg = f"Date {d_str} is in illegal state '{curr_state}' (expected SEALED)"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    lifecycle_violations.append(d_str)

                # owning_cycle_id must match requesting cycle
                if owner != cycle_id:
                    msg = f"Ownership mismatch for {d_str}: owned by '{owner}', requested by '{cycle_id}'"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    ownership_violations.append(d_str)

                # reservation_status must be exactly EVALUATION_IN_PROGRESS
                if res_status != "EVALUATION_IN_PROGRESS":
                    msg = f"Reservation status for {d_str} is '{res_status}' (expected 'EVALUATION_IN_PROGRESS')"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    lifecycle_violations.append(d_str)

                # dataset_fingerprint must match reservation fingerprint
                if dataset_fingerprint and rec_fp and rec_fp != dataset_fingerprint:
                    msg = f"Dataset fingerprint mismatch for {d_str}: reservation='{rec_fp}', request='{dataset_fingerprint}'"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    fingerprint_mismatches.append(d_str)
                elif rec_fp and not dataset_fingerprint:
                    msg = f"Dataset fingerprint missing in completion request for {d_str} (expected '{rec_fp}')"
                    invalid_dates.append({"date": d_str, "reason": msg})
                    failure_reasons.append(msg)
                    fingerprint_mismatches.append(d_str)

            # All-or-nothing: if ANY date fails validation, reject completion without partial EVALUATION_COMPLETED.
            # However, if owned dates already entered EVALUATION_IN_PROGRESS / data_accessed=True, conservatively
            # transition those accessed owned dates to EXPOSED (FAILED_EVALUATION) so they can never be reused.
            if invalid_dates:
                newly_exposed_on_fail: list[str] = []
                for d_str in iso_dates:
                    rec = records.get(d_str)
                    if (
                        rec is not None
                        and rec.get("state") == "SEALED"
                        and rec.get("owning_cycle_id") == cycle_id
                        and (bool(rec.get("data_accessed")) or rec.get("reservation_status") == "EVALUATION_IN_PROGRESS")
                    ):
                        rec["state"] = "EXPOSED"
                        rec["reservation_status"] = "FAILED_EVALUATION"
                        rec["data_accessed"] = True
                        rec["first_exposed_cycle"] = rec.get("first_exposed_cycle") or cycle_id
                        rec["exposure_timestamp"] = rec.get("exposure_timestamp") or now_ist
                        rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 0)) + 1
                        existing_strats = set(rec.get("strategy_ids_evaluated", []))
                        rec["strategy_ids_evaluated"] = sorted(existing_strats | set(strat_list))
                        rec["registry_version"] = REGISTRY_VERSION
                        newly_exposed_on_fail.append(d_str)

                # Determine actual final lifecycle state from persisted records
                if newly_exposed_on_fail:
                    final_state = "EXPOSED"
                else:
                    existing_states = {records[d_str].get("state") for d_str in iso_dates if d_str in records}
                    if "SEALED" in existing_states:
                        final_state = "SEALED"
                    elif "EXPOSED" in existing_states:
                        final_state = "EXPOSED"
                    elif "RETIRED" in existing_states:
                        final_state = "RETIRED"
                    else:
                        final_state = "AVAILABLE"

                # Extract started_at from reservation record if available
                started_at = now_ist
                for d_str in iso_dates:
                    if d_str in records and records[d_str].get("evaluation_started_at"):
                        started_at = records[d_str]["evaluation_started_at"]
                        break

                if newly_exposed_on_fail:
                    reg.setdefault("recovery_log", []).append({
                        "action": "FAILED_COMPLETION_CONSERVATIVE_EXPOSURE",
                        "cycle_id": cycle_id,
                        "evaluation_id": evaluation_id,
                        "dates": iso_dates,
                        "exposed_due_to_data_access": newly_exposed_on_fail,
                        "reason": "; ".join(failure_reasons),
                        "timestamp": now_ist,
                    })

                if ownership_violations:
                    status = "HOLDOUT_OWNERSHIP_VIOLATION"
                elif reused_dates:
                    status = "EXPOSED_HOLDOUT_REUSED"
                elif fingerprint_mismatches:
                    status = "DATASET_FINGERPRINT_MISMATCH"
                else:
                    status = "LIFECYCLE_STATE_VIOLATION"

                # Append failed evaluation history record reflecting ACTUAL registry state
                reg.setdefault("evaluation_history", []).append({
                    "evaluation_id": evaluation_id,
                    "cycle_id": cycle_id,
                    "evaluation_type": evaluation_type,
                    "evaluation_status": "FAILED_EVALUATION",
                    "consumed_dates": iso_dates,
                    "strategy_ids": strat_list,
                    "dataset_fingerprint": dataset_fingerprint,
                    "started_at": started_at,
                    "completed_at": now_ist,
                    "final_lifecycle_state": final_state,
                    "holdout_state": final_state,
                    "holdout_consumed": bool(newly_exposed_on_fail),
                    "completion_success": False,
                    "failure_reasons": failure_reasons,
                    "exposed_dates": newly_exposed_on_fail,
                    "invalid_dates": invalid_dates,
                    "ownership_violations": ownership_violations,
                    "fingerprint_mismatches": fingerprint_mismatches,
                    "lifecycle_violations": lifecycle_violations,
                    "reused_dates": reused_dates,
                    "status": status,
                    "ok": False,
                })
                self.save_registry(reg)

                return {
                    "ok": False,
                    "completion_success": False,
                    "evaluation_id": evaluation_id,
                    "cycle_id": cycle_id,
                    "evaluation_type": evaluation_type,
                    "holdout_consumed": bool(newly_exposed_on_fail),
                    "holdout_state": final_state,
                    "exposed_dates": newly_exposed_on_fail,
                    "newly_exposed_dates": len(newly_exposed_on_fail),
                    "completed_at": now_ist,
                    "invalid_dates": invalid_dates,
                    "failure_reasons": failure_reasons,
                    "ownership_violations": ownership_violations,
                    "fingerprint_mismatches": fingerprint_mismatches,
                    "lifecycle_violations": lifecycle_violations,
                    "reused_dates": reused_dates,
                    "ownership_valid": len(ownership_violations) == 0,
                    "independent_holdout_valid": False,
                    "leakage_detected": True,
                    "status": status,
                }

            # 5. Transition consumed dates from SEALED to EXPOSED
            # 6. Update reservation status to EVALUATION_COMPLETED
            # 7. Record exposure timestamps and first-exposure cycle
            started_at = now_ist
            for d_str in iso_dates:
                rec = records[d_str]
                if rec.get("evaluation_started_at"):
                    started_at = rec["evaluation_started_at"]
                rec["state"] = "EXPOSED"
                rec["reservation_status"] = "EVALUATION_COMPLETED"
                rec["first_exposed_cycle"] = rec.get("first_exposed_cycle") or cycle_id
                rec["exposure_timestamp"] = rec.get("exposure_timestamp") or now_ist
                rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 0)) + 1
                rec["data_accessed"] = True
                if dataset_fingerprint and not rec.get("dataset_fingerprint"):
                    rec["dataset_fingerprint"] = dataset_fingerprint
                existing_strats = set(rec.get("strategy_ids_evaluated", []))
                rec["strategy_ids_evaluated"] = sorted(existing_strats | set(strat_list))
                rec["registry_version"] = REGISTRY_VERSION

            # 8. Record evaluation metadata in sealed_exposures
            reg.setdefault("sealed_exposures", []).append({
                "cycle_id": cycle_id,
                "recorded_at_ist": now_ist,
                "commit_sha": _get_git_commit_sha(),
                "unseen_dates": iso_dates,
                "newly_sealed_count": len(iso_dates),
                "previously_sealed_count": 0,
                "ownership_violations": [],
                "leakage_detected": False,
                "independent_holdout_valid": True,
                "fingerprint_mismatches": [],
                "dataset_fingerprint": dataset_fingerprint,
                "strategy_ids_evaluated": strat_list,
                "notes": f"{evaluation_type} completed ({evaluation_id})",
            })

            # 9. Append evaluation history
            reg.setdefault("evaluation_history", []).append({
                "evaluation_id": evaluation_id,
                "cycle_id": cycle_id,
                "evaluation_type": evaluation_type,
                "evaluation_status": "EVALUATION_COMPLETED",
                "consumed_dates": iso_dates,
                "strategy_ids": strat_list,
                "dataset_fingerprint": dataset_fingerprint,
                "started_at": started_at,
                "completed_at": now_ist,
                "final_lifecycle_state": "EXPOSED",
                "holdout_state": "EXPOSED",
                "holdout_consumed": True,
                "exposed_dates": iso_dates,
                "completion_success": True,
                "failure_reasons": [],
                "status": "VALID_UNSEEN",
                "ok": True,
            })

            # 10. Save the registry atomically
            self.save_registry(reg)

            return {
                "ok": True,
                "completion_success": True,
                "evaluation_id": evaluation_id,
                "cycle_id": cycle_id,
                "evaluation_type": evaluation_type,
                "holdout_consumed": True,
                "holdout_state": "EXPOSED",
                "completed_at": now_ist,
                "total_sealed_dates": len(reg["sealed_dates"]),
                "current_unseen_dates": len(iso_dates),
                "newly_exposed_dates": len(iso_dates),
                "previously_exposed_dates": 0,
                "reused_dates": [],
                "ownership_violations": [],
                "fingerprint_mismatches": [],
                "invalid_dates": [],
                "failure_reasons": [],
                "ownership_valid": True,
                "leakage_detected": False,
                "independent_holdout_valid": True,
                "status": "VALID_UNSEEN",
            }

    def mark_evaluation_failed(
        self,
        cycle_id: str,
        dates: list[date],
        reason: str,
        data_accessed: bool = True,
    ) -> dict[str, Any]:
        """Record FAILED_EVALUATION. Conservatively transitions to EXPOSED if data access occurred."""
        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            marked: list[str] = []
            exposed_on_fail: list[str] = []
            for d in dates:
                d_str = d.isoformat()
                rec = records.get(d_str)
                if rec and rec.get("state") == "SEALED" and rec.get("owning_cycle_id") == cycle_id:
                    rec["reservation_status"] = "FAILED_EVALUATION"
                    accessed = data_accessed or bool(rec.get("data_accessed", False))
                    rec["data_accessed"] = accessed
                    if accessed:
                        rec["state"] = "EXPOSED"
                        rec["first_exposed_cycle"] = rec.get("first_exposed_cycle") or cycle_id
                        rec["exposure_timestamp"] = rec.get("exposure_timestamp") or now_ist
                        rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 0)) + 1
                        exposed_on_fail.append(d_str)
                    rec["registry_version"] = REGISTRY_VERSION
                    marked.append(d_str)
            reg["recovery_log"].append({
                "action": "FAILED_EVALUATION",
                "cycle_id": cycle_id,
                "dates": marked,
                "exposed_due_to_data_access": exposed_on_fail,
                "data_accessed": data_accessed,
                "reason": reason,
                "timestamp": now_ist,
            })
            if marked:
                reg.setdefault("evaluation_history", []).append({
                    "evaluation_id": f"failed_{cycle_id}_{now_ist[:19]}",
                    "cycle_id": cycle_id,
                    "evaluation_type": "FAILED_EVALUATION",
                    "evaluation_status": "FAILED_EVALUATION",
                    "consumed_dates": marked,
                    "strategy_ids": [],
                    "dataset_fingerprint": records[marked[0]].get("dataset_fingerprint", "") if marked else "",
                    "started_at": records[marked[0]].get("evaluation_started_at", now_ist) if marked else now_ist,
                    "completed_at": now_ist,
                    "final_lifecycle_state": "EXPOSED" if exposed_on_fail else "SEALED",
                    "completion_success": False,
                    "failure_reasons": [reason],
                    "ok": False,
                })
            self.save_registry(reg)
            return {
                "marked_failed": marked,
                "exposed_dates": exposed_on_fail,
                "holdout_consumed": bool(exposed_on_fail),
                "timestamp": now_ist,
            }

    def recover_stale_reservations(self, cycle_id: str | None = None) -> dict[str, Any]:
        """Recover any EVALUATION_IN_PROGRESS reservations conservatively as EXPOSED (never AVAILABLE)."""
        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            recovered: list[str] = []
            for d_str, rec in records.items():
                if rec.get("reservation_status") == "EVALUATION_IN_PROGRESS":
                    if cycle_id is not None and rec.get("owning_cycle_id") != cycle_id:
                        continue
                    rec["state"] = "EXPOSED"
                    rec["reservation_status"] = "FAILED_EVALUATION"
                    rec["data_accessed"] = True
                    rec["first_exposed_cycle"] = rec.get("first_exposed_cycle") or rec.get("owning_cycle_id")
                    rec["exposure_timestamp"] = rec.get("exposure_timestamp") or now_ist
                    rec["previous_exposure_count"] = max(1, int(rec.get("previous_exposure_count", 0)))
                    rec["registry_version"] = REGISTRY_VERSION
                    recovered.append(d_str)
            if recovered:
                reg["recovery_log"].append({
                    "action": "STALE_IN_PROGRESS_RECOVERED_AS_EXPOSED",
                    "cycle_id": cycle_id or "ALL",
                    "recovered_dates": recovered,
                    "reason": "Previous process terminated during EVALUATION_IN_PROGRESS; data access cannot be ruled out.",
                    "timestamp": now_ist,
                })
                self.save_registry(reg)
            return {"recovered_dates": recovered, "count": len(recovered), "timestamp": now_ist}

    def release_reservation(self, cycle_id: str, dates: list[date], reason: str) -> dict[str, Any]:
        """Safely release unused, unaccessed SEALED reservations back to AVAILABLE (owning cycle only)."""
        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            released: list[str] = []
            denied: list[dict[str, str]] = []

            for d in dates:
                d_str = d.isoformat()
                rec = records.get(d_str)
                if not rec:
                    continue
                state = rec.get("state")
                owner = rec.get("owning_cycle_id")
                if state in ("EXPOSED", "RETIRED"):
                    denied.append({"date": d_str, "reason": f"Cannot release {state} date"})
                elif rec.get("data_accessed"):
                    denied.append({"date": d_str, "reason": "Cannot release date after holdout data access"})
                elif state == "SEALED" and owner != cycle_id:
                    denied.append({"date": d_str, "reason": f"Owned by {owner}, not {cycle_id}"})
                elif state == "SEALED" and owner == cycle_id:
                    rec["state"] = "AVAILABLE"
                    rec["reservation_status"] = "RELEASED_RESERVATION"
                    rec["owning_cycle_id"] = None
                    rec["registry_version"] = REGISTRY_VERSION
                    released.append(d_str)

            reg["recovery_log"].append({
                "action": "RELEASE_RESERVATION",
                "cycle_id": cycle_id,
                "released_dates": released,
                "denied_dates": denied,
                "reason": reason,
                "timestamp": now_ist,
            })
            self.save_registry(reg)
            return {"released": released, "denied": denied, "ok": len(denied) == 0, "timestamp": now_ist}


