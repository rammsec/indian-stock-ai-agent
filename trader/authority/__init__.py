"""Authority Control and Mandatory Execution Gate Package.

Implements Central Agent & Human Authority Control addendum:
- Subordination of Candle Agent to Human, Central Agent, and Risk Engine.
- Multi-tier permission interface (ALLOW_LONG, ALLOW_SHORT, NO_NEW_ENTRIES, EXIT_ONLY, PAUSE, STOP_ALL).
- Immediate human override at all stages.
- Atomic revalidation of the 7-point execution gate immediately prior to execution.
- Cryptographically hash-chained immutable audit records.
"""

from .audit import AuthorityAuditRecord
from .gate import (
    AuthorityGateRejectionError,
    GateEvaluationOutcome,
    MandatoryExecutionGate,
)
from .instructions import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)

__all__ = [
    "ActionType",
    "AuthorityAuditRecord",
    "AuthorityGateRejectionError",
    "AuthorityInstruction",
    "AuthoritySource",
    "GateEvaluationOutcome",
    "MandatoryExecutionGate",
    "PermissionType",
]
