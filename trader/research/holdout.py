"""Phase 11.2: Strict Sealed Holdout Lifecycle & Leakage Prevention Guard."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from ..config import IST_TZ

HoldoutState = Literal["AVAILABLE", "SEALED", "EXPOSED", "RETIRED"]
SEALED_HOLDOUT_REGISTRY_PATH = Path("data/lab/sealed_holdout_registry.json")


def _get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


class SealedHoldoutGuard:
    """Manages strict AVAILABLE -> SEALED -> EXPOSED -> RETIRED holdout date lifecycle."""

    def __init__(self, registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH) -> None:
        self.registry_path = registry_path

    def load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
        else:
            reg = {"sealed_exposures": [], "sealed_dates": [], "records": {}}

        reg.setdefault("sealed_exposures", [])
        reg.setdefault("sealed_dates", [])
        reg.setdefault("records", {})
        # Migrate legacy sealed_exposures entries into per-date records without deleting history
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
                        "first_sealed_cycle": cid,
                        "first_exposed_cycle": cid,
                        "exposure_timestamp": ts,
                        "dataset_fingerprint": str(exp.get("dataset_fingerprint", "")),
                        "strategy_ids_evaluated": list(exp.get("strategy_ids_evaluated", [])),
                        "previous_exposure_count": 1,
                        "reason_for_retirement": None,
                    }

    def save_registry(self, reg: dict[str, Any]) -> None:
        records = reg.get("records", {})
        reg["sealed_dates"] = sorted(
            d for d, rec in records.items() if rec.get("state") in ("SEALED", "EXPOSED", "RETIRED")
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

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
        """Reserve AVAILABLE dates as SEALED for final evaluation."""
        reg = self.load_registry()
        records = reg["records"]
        blocked: list[dict[str, str]] = []
        sealed_now: list[str] = []

        for d in dates:
            d_str = d.isoformat()
            rec = records.get(d_str)
            if rec and rec.get("state") in ("EXPOSED", "RETIRED", "SEALED"):
                blocked.append({"date": d_str, "state": rec["state"]})
            else:
                records[d_str] = {
                    "date": d_str,
                    "state": "SEALED",
                    "first_sealed_cycle": cycle_id,
                    "first_exposed_cycle": None,
                    "exposure_timestamp": None,
                    "dataset_fingerprint": dataset_fingerprint,
                    "strategy_ids_evaluated": [],
                    "previous_exposure_count": 0,
                    "reason_for_retirement": None,
                }
                sealed_now.append(d_str)

        self.save_registry(reg)
        return {
            "cycle_id": cycle_id,
            "sealed_count": len(sealed_now),
            "blocked_dates": blocked,
            "independent_holdout_valid": len(blocked) == 0 and len(sealed_now) > 0,
        }

    def record_exposure(
        self,
        cycle_id: str,
        unseen_dates: list[date],
        notes: str = "",
        dataset_fingerprint: str = "",
        strategy_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record final-test exposure. Reusing EXPOSED/RETIRED dates flags leakage & invalidates independence."""
        reg = self.load_registry()
        records = reg["records"]
        now_ist = datetime.now(tz=IST_TZ).isoformat()
        iso_dates = [d.isoformat() for d in unseen_dates]
        strat_list = list(strategy_ids or [])

        reused_exposed: list[str] = []
        fingerprint_mismatches: list[str] = []
        newly_exposed: list[str] = []

        for d_str in iso_dates:
            rec = records.get(d_str)
            if rec is None:
                records[d_str] = {
                    "date": d_str,
                    "state": "EXPOSED",
                    "first_sealed_cycle": cycle_id,
                    "first_exposed_cycle": cycle_id,
                    "exposure_timestamp": now_ist,
                    "dataset_fingerprint": dataset_fingerprint,
                    "strategy_ids_evaluated": strat_list,
                    "previous_exposure_count": 1,
                    "reason_for_retirement": None,
                }
                newly_exposed.append(d_str)
            else:
                prev_state = rec.get("state", "AVAILABLE")
                if prev_state in ("EXPOSED", "RETIRED"):
                    reused_exposed.append(d_str)
                    rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 1)) + 1
                else:
                    # Transition SEALED -> EXPOSED
                    rec["state"] = "EXPOSED"
                    rec["first_exposed_cycle"] = rec.get("first_exposed_cycle") or cycle_id
                    rec["exposure_timestamp"] = rec.get("exposure_timestamp") or now_ist
                    rec["previous_exposure_count"] = int(rec.get("previous_exposure_count", 0)) + 1
                    newly_exposed.append(d_str)

                if dataset_fingerprint and rec.get("dataset_fingerprint") and rec["dataset_fingerprint"] != dataset_fingerprint:
                    fingerprint_mismatches.append(d_str)
                if dataset_fingerprint and not rec.get("dataset_fingerprint"):
                    rec["dataset_fingerprint"] = dataset_fingerprint
                existing_strats = set(rec.get("strategy_ids_evaluated", []))
                rec["strategy_ids_evaluated"] = sorted(existing_strats | set(strat_list))

        leakage = len(reused_exposed) > 0 or len(fingerprint_mismatches) > 0
        independent_valid = (len(iso_dates) > 0) and not leakage

        reg["sealed_exposures"].append({
            "cycle_id": cycle_id,
            "recorded_at_ist": now_ist,
            "commit_sha": _get_git_commit_sha(),
            "unseen_dates": iso_dates,
            "newly_sealed_count": len(newly_exposed),
            "previously_sealed_count": len(reused_exposed),
            "leakage_detected": leakage,
            "independent_holdout_valid": independent_valid,
            "fingerprint_mismatches": fingerprint_mismatches,
            "dataset_fingerprint": dataset_fingerprint,
            "strategy_ids_evaluated": strat_list,
            "notes": notes,
        })
        self.save_registry(reg)

        return {
            "total_sealed_dates": len(reg["sealed_dates"]),
            "current_unseen_dates": len(iso_dates),
            "newly_exposed_dates": len(newly_exposed),
            "previously_exposed_dates": len(reused_exposed),
            "reused_dates": reused_exposed,
            "fingerprint_mismatches": fingerprint_mismatches,
            "leakage_detected": leakage,
            "independent_holdout_valid": independent_valid,
            "status": "VALID_UNSEEN" if independent_valid else ("EXPOSED_HOLDOUT_REUSED" if reused_exposed else "INSUFFICIENT_UNSEEN_DATA"),
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
    ) -> dict[str, Any]:
        """Validate train, validation, and unseen splits against the holdout registry."""
        reg = self.load_registry()
        records = reg["records"]

        train_violations = [
            d.isoformat() for d in train_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") != "AVAILABLE"
        ]
        val_violations = [
            d.isoformat() for d in val_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") != "AVAILABLE"
        ]
        unseen_reused = [
            d.isoformat() for d in unseen_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") in ("EXPOSED", "RETIRED")
        ]
        untouched_unseen = [
            d.isoformat() for d in unseen_dates if records.get(d.isoformat(), {}).get("state", "AVAILABLE") in ("AVAILABLE", "SEALED")
        ]

        if not unseen_dates or not untouched_unseen:
            status = "EXPOSED_HOLDOUT_REUSED" if unseen_reused else "INSUFFICIENT_UNSEEN_DATA"
            independent_valid = False
        elif unseen_reused or train_violations or val_violations:
            status = "EXPOSED_HOLDOUT_REUSED"
            independent_valid = False
        else:
            status = "VALID_UNSEEN"
            independent_valid = True

        return {
            "independent_holdout_valid": independent_valid,
            "status": status,
            "train_violations": train_violations,
            "validation_violations": val_violations,
            "unseen_reused_dates": unseen_reused,
            "untouched_unseen_dates": untouched_unseen,
            "leakage_detected": bool(train_violations or val_violations or unseen_reused),
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
                    "first_sealed_cycle": None,
                    "first_exposed_cycle": None,
                    "exposure_timestamp": None,
                    "dataset_fingerprint": "",
                    "strategy_ids_evaluated": [],
                    "previous_exposure_count": 0,
                    "reason_for_retirement": reason,
                },
            )
            rec["state"] = "RETIRED"
            rec["reason_for_retirement"] = reason
            count += 1
        self.save_registry(reg)
        return count
