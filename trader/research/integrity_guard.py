"""Phase 11.3: Central Research Integrity Guard (Fail-Closed Provenance, Holdout & Split Governance)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import IST_TZ
from .holdout import SEALED_HOLDOUT_REGISTRY_PATH, SealedHoldoutGuard

GUARD_VERSION = "11.3.0"


class ResearchIntegrityGuard:
    """Single authoritative fail-closed gatekeeper for provenance, holdout ownership, and split validity."""

    def __init__(self, registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH) -> None:
        self.holdout_guard = SealedHoldoutGuard(registry_path=registry_path)

    def validate(
        self,
        cycle_id: str,
        training_dates: list[date],
        validation_dates: list[date],
        unseen_dates: list[date],
        dataset_fingerprint: str = "",
        provenance_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate all mandatory integrity checks; fails closed on any missing or invalid input."""
        now_ist = datetime.now(tz=IST_TZ).isoformat()
        failure_reasons: list[str] = []

        # 1. Provenance check (never defaults to True)
        if not provenance_result or not isinstance(provenance_result, dict):
            prov_status = "FAILED"
            prov_valid = False
            failure_reasons.append("Missing provenance verification result.")
        else:
            prov_status = str(
                provenance_result.get("final_provenance_status")
                or provenance_result.get("status")
                or "FAILED"
            )
            prov_valid = (prov_status == "VERIFIED") and bool(provenance_result.get("verified", False))
            if not prov_valid:
                failure_reasons.append(f"Provenance status is {prov_status} (requires VERIFIED).")

        # 2. Holdout ownership & split validation (fail closed on exception)
        try:
            split_res = self.holdout_guard.validate_cycle_splits(
                train_dates=training_dates,
                val_dates=validation_dates,
                unseen_dates=unseen_dates,
                cycle_id=cycle_id,
                dataset_fingerprint=dataset_fingerprint,
            )
            holdout_status = str(split_res.get("status", "FAILED"))
            ownership_valid = bool(split_res.get("ownership_valid", False))
            split_valid = bool(split_res.get("split_valid", False))
            fp_valid = bool(split_res.get("dataset_fingerprint_valid", False))
            holdout_valid = bool(split_res.get("independent_holdout_valid", False))

            if not ownership_valid:
                failure_reasons.append(
                    f"Holdout ownership violation: {split_res.get('ownership_violations', [])}"
                )
            if not split_valid:
                failure_reasons.append(
                    f"Split validation failed (overlaps={split_res.get('overlap_dates', [])})."
                )
            if not fp_valid:
                failure_reasons.append(
                    f"Dataset fingerprint mismatch: {split_res.get('fingerprint_mismatches', [])}"
                )
            if not holdout_valid:
                failure_reasons.append(f"Holdout integrity failed: {holdout_status}.")
        except Exception as exc:
            holdout_status = "REGISTRY_ERROR"
            ownership_valid = False
            split_valid = False
            fp_valid = False
            holdout_valid = False
            failure_reasons.append(f"Holdout validation exception (fail-closed): {exc}")

        integrity_passed = bool(
            prov_valid and holdout_valid and ownership_valid and split_valid and fp_valid
        )

        return {
            "cycle_id": cycle_id,
            "integrity_passed": integrity_passed,
            "qualification_allowed": integrity_passed,
            "provenance_status": prov_status,
            "holdout_status": holdout_status,
            "ownership_valid": ownership_valid,
            "split_valid": split_valid,
            "dataset_fingerprint_valid": fp_valid,
            "failure_reasons": failure_reasons,
            "validation_timestamp": now_ist,
            "guard_version": GUARD_VERSION,
        }
