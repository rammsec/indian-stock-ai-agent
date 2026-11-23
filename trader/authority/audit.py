"""Immutable SHA-256 Hash-Chained Authority Audit Ledger.

Records:
- Original human instruction
- Central Agent instruction
- Candle Agent recommendation
- Any disagreement
- Permission evaluation
- Risk Engine decision
- Final execution decision
- Rejection reason
- Instruction timestamps
All records are strictly immutable and cryptographically chained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from .instructions import ActionType, AuthorityInstruction


@dataclass(frozen=True)
class AuthorityAuditRecord:
    """Immutable audit record for a single authority execution evaluation."""

    audit_id: str
    timestamp_ist: str
    symbol: str
    action_type: str
    human_instruction: dict[str, Any] | None
    central_instruction: dict[str, Any] | None
    candle_recommendation: str | None
    candle_bias: str | None
    disagreement_explanation: str | None
    permission_evaluation_passed: bool
    risk_evaluation_passed: bool
    final_execution_decision: Literal["APPROVED", "REJECTED"]
    rejection_reasons: tuple[str, ...]
    prev_hash: str
    record_hash: str

    @classmethod
    def create(
        cls,
        *,
        audit_id: str,
        timestamp_ist: datetime,
        symbol: str,
        action_type: ActionType,
        human_instruction: AuthorityInstruction | None,
        central_instruction: AuthorityInstruction | None,
        candle_recommendation: str | None,
        candle_bias: str | None,
        disagreement_explanation: str | None,
        permission_evaluation_passed: bool,
        risk_evaluation_passed: bool,
        final_execution_decision: Literal["APPROVED", "REJECTED"],
        rejection_reasons: tuple[str, ...],
        prev_hash: str,
    ) -> AuthorityAuditRecord:
        ts_str = timestamp_ist.isoformat()
        payload = {
            "audit_id": audit_id,
            "timestamp_ist": ts_str,
            "symbol": symbol.strip().upper(),
            "action_type": action_type.value,
            "human_instruction": human_instruction.to_dict() if human_instruction else None,
            "central_instruction": central_instruction.to_dict() if central_instruction else None,
            "candle_recommendation": candle_recommendation,
            "candle_bias": candle_bias,
            "disagreement_explanation": disagreement_explanation,
            "permission_evaluation_passed": permission_evaluation_passed,
            "risk_evaluation_passed": risk_evaluation_passed,
            "final_execution_decision": final_execution_decision,
            "rejection_reasons": list(rejection_reasons),
            "prev_hash": prev_hash,
        }
        canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h = hashlib.sha256(canonical_bytes).hexdigest()

        return cls(
            audit_id=audit_id,
            timestamp_ist=ts_str,
            symbol=symbol.strip().upper(),
            action_type=action_type.value,
            human_instruction=payload["human_instruction"],
            central_instruction=payload["central_instruction"],
            candle_recommendation=candle_recommendation,
            candle_bias=candle_bias,
            disagreement_explanation=disagreement_explanation,
            permission_evaluation_passed=permission_evaluation_passed,
            risk_evaluation_passed=risk_evaluation_passed,
            final_execution_decision=final_execution_decision,
            rejection_reasons=rejection_reasons,
            prev_hash=prev_hash,
            record_hash=h,
        )

    def verify_hash(self) -> bool:
        """Verify that record_hash matches the canonical content."""
        payload = {
            "audit_id": self.audit_id,
            "timestamp_ist": self.timestamp_ist,
            "symbol": self.symbol,
            "action_type": self.action_type,
            "human_instruction": self.human_instruction,
            "central_instruction": self.central_instruction,
            "candle_recommendation": self.candle_recommendation,
            "candle_bias": self.candle_bias,
            "disagreement_explanation": self.disagreement_explanation,
            "permission_evaluation_passed": self.permission_evaluation_passed,
            "risk_evaluation_passed": self.risk_evaluation_passed,
            "final_execution_decision": self.final_execution_decision,
            "rejection_reasons": list(self.rejection_reasons),
            "prev_hash": self.prev_hash,
        }
        canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        computed = hashlib.sha256(canonical_bytes).hexdigest()
        return computed == self.record_hash
