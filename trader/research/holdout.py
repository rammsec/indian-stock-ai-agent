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
REGISTRY_VERSION = "11.3.1"
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

    def mark_evaluation_in_progress(self, cycle_id: str, dates: list[date]) -> dict[str, Any]:
        """Transition owned SEALED dates from ACTIVE_RESERVATION to EVALUATION_IN_PROGRESS."""
        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            updated: list[str] = []
            rejected: list[str] = []
            for d in dates:
                d_str = d.isoformat()
                rec = records.get(d_str)
                if rec and rec.get("state") == "SEALED" and rec.get("owning_cycle_id") == cycle_id:
                    rec["reservation_status"] = "EVALUATION_IN_PROGRESS"
                    rec["registry_version"] = REGISTRY_VERSION
                    updated.append(d_str)
                else:
                    rejected.append(d_str)
            self.save_registry(reg)
            return {"ok": len(rejected) == 0, "updated": updated, "rejected": rejected}

    def mark_evaluation_failed(self, cycle_id: str, dates: list[date], reason: str) -> dict[str, Any]:
        """Mark evaluation as FAILED_EVALUATION without exposing SEALED dates."""
        with self._locked_registry():
            reg = self.load_registry()
            records = reg["records"]
            now_ist = datetime.now(tz=IST_TZ).isoformat()
            marked: list[str] = []
            for d in dates:
                d_str = d.isoformat()
                rec = records.get(d_str)
                if rec and rec.get("state") == "SEALED" and rec.get("owning_cycle_id") == cycle_id:
                    rec["reservation_status"] = "FAILED_EVALUATION"
                    rec["registry_version"] = REGISTRY_VERSION
                    marked.append(d_str)
            reg["recovery_log"].append({
                "action": "FAILED_EVALUATION",
                "cycle_id": cycle_id,
                "dates": marked,
                "reason": reason,
                "timestamp": now_ist,
            })
            self.save_registry(reg)
            return {"marked_failed": marked, "timestamp": now_ist}

    def release_reservation(self, cycle_id: str, dates: list[date], reason: str) -> dict[str, Any]:
        """Safely release unused SEALED reservations back to AVAILABLE (owning cycle only; never EXPOSED/RETIRED)."""
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

