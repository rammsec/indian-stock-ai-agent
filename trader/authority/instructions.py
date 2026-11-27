"""Central Agent and Human Authority Control Instruction Interface.

Defines deterministic permission types, action types, and validated immutable
authority instructions with strict valid-from and expiry timestamps.
Expired, missing, invalid or contradictory instructions fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class PermissionType(str, Enum):
    """Supported trading permissions for Human and Central Agent authority."""

    ALLOW_LONG = "ALLOW_LONG"
    ALLOW_SHORT = "ALLOW_SHORT"
    ALLOW_BOTH = "ALLOW_BOTH"
    NO_NEW_ENTRIES = "NO_NEW_ENTRIES"
    EXIT_ONLY = "EXIT_ONLY"
    PAUSE = "PAUSE"
    STOP_ALL = "STOP_ALL"


class ActionType(str, Enum):
    """Trading action classifications distinguishing entries, exits, and risk actions."""

    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    RISK_EXIT = "RISK_EXIT"

    @property
    def is_new_entry(self) -> bool:
        """True if action creates new market exposure."""
        return self in (ActionType.ENTRY_LONG, ActionType.ENTRY_SHORT)

    @property
    def is_exit(self) -> bool:
        """True if action reduces or closes open market exposure."""
        return self in (ActionType.EXIT_LONG, ActionType.EXIT_SHORT, ActionType.RISK_EXIT)


class AuthoritySource(str, Enum):
    """Authority source hierarchy."""

    HUMAN = "HUMAN"
    CENTRAL_AGENT = "CENTRAL_AGENT"


@dataclass(frozen=True)
class AuthorityInstruction:
    """Immutable, timestamped authority instruction issued by Human or Central Agent."""

    instruction_id: str
    authority: AuthoritySource
    scope: str  # Symbol (e.g. "TCS") or "*" / "PORTFOLIO"
    permission: PermissionType
    valid_from: datetime
    expires_at: datetime
    reason: str
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.instruction_id or not self.instruction_id.strip():
            raise ValueError("AuthorityInstruction.instruction_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("AuthorityInstruction.reason must be non-empty")
        if not self.version or not self.version.strip():
            raise ValueError("AuthorityInstruction.version must be non-empty")

        # Timezone-awareness requirement: fail closed on naive datetimes
        if self.valid_from.tzinfo is None:
            raise ValueError(f"AuthorityInstruction.valid_from must be timezone-aware; got {self.valid_from!r}")
        if self.expires_at.tzinfo is None:
            raise ValueError(f"AuthorityInstruction.expires_at must be timezone-aware; got {self.expires_at!r}")
        if self.valid_from >= self.expires_at:
            raise ValueError(
                f"AuthorityInstruction valid_from ({self.valid_from.isoformat()}) must be strictly "
                f"before expires_at ({self.expires_at.isoformat()})"
            )

    @property
    def normalized_scope(self) -> str:
        s = self.scope.strip().upper()
        return "*" if s in ("*", "PORTFOLIO", "ALL") else s

    def applies_to_symbol(self, symbol: str) -> bool:
        """Check if instruction applies to the given symbol."""
        s = symbol.strip().upper()
        return self.normalized_scope in ("*", s)

    def is_valid_at(self, current_time: datetime) -> bool:
        """Check if instruction is active (valid_from <= current_time <= expires_at)."""
        if current_time.tzinfo is None:
            raise ValueError(f"current_time must be timezone-aware; got {current_time!r}")
        return self.valid_from <= current_time <= self.expires_at

    def allows_action(self, action: ActionType) -> tuple[bool, str]:
        """Check whether this instruction permits the specified action.

        Returns (allowed: bool, reason: str).
        """
        # STOP_ALL blocks all new entries; permits protective/unwind exits
        if self.permission == PermissionType.STOP_ALL:
            if action.is_new_entry:
                return False, f"{self.authority.value} permission STOP_ALL blocks all new entries"
            return True, "STOP_ALL permits protective exit/unwind"

        # PAUSE blocks all new entries AND discretionary strategy exits; preserves protective RISK_EXIT
        if self.permission == PermissionType.PAUSE:
            if action.is_new_entry:
                return False, f"{self.authority.value} permission PAUSE blocks all new entries"
            if action == ActionType.RISK_EXIT:
                return True, "PAUSE preserves deterministic protective RISK_EXIT"
            return False, f"{self.authority.value} permission PAUSE suspends discretionary {action.value}"

        # NO_NEW_ENTRIES prohibits new entries but permits position exits
        if self.permission == PermissionType.NO_NEW_ENTRIES:
            if action.is_new_entry:
                return False, f"{self.authority.value} permission NO_NEW_ENTRIES blocks new entries"
            return True, "NO_NEW_ENTRIES permits risk-reducing exits"

        # EXIT_ONLY prohibits new exposure while allowing position reductions
        if self.permission == PermissionType.EXIT_ONLY:
            if action.is_new_entry:
                return False, f"{self.authority.value} permission EXIT_ONLY prohibits new exposure"
            return True, "EXIT_ONLY allows position reductions and closures"

        # Directional permissions (protective exits are always permitted so mid-session flips never trap open positions)
        if self.permission == PermissionType.ALLOW_LONG:
            if action == ActionType.ENTRY_SHORT:
                return False, f"{self.authority.value} permission ALLOW_LONG prohibits short entries"
            if action.is_exit:
                return True, f"ALLOW_LONG permits protective exit ({action.value})"
            return True, "ALLOW_LONG permits long actions"

        if self.permission == PermissionType.ALLOW_SHORT:
            if action == ActionType.ENTRY_LONG:
                return False, f"{self.authority.value} permission ALLOW_SHORT prohibits long entries"
            if action.is_exit:
                return True, f"ALLOW_SHORT permits protective exit ({action.value})"
            return True, "ALLOW_SHORT permits short actions"

        if self.permission == PermissionType.ALLOW_BOTH:
            return True, "ALLOW_BOTH permits all actions"

        return False, f"Unhandled permission: {self.permission.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "authority": self.authority.value,
            "scope": self.scope,
            "permission": self.permission.value,
            "valid_from": self.valid_from.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "reason": self.reason,
            "version": self.version,
        }
