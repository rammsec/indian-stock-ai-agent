"""CYCLE-008 — Historical Out-of-Sample Evaluation & Strategy Validation."""

from trader.evaluation.bias_auditor import (
    BiasAuditResult,
    BiasViolation,
    audit_bar_index_lookahead,
    audit_cross_sectional_lookahead,
    audit_future_event_exclusion,
    audit_mtf_backward_alignment,
    audit_recursive_convergence,
    audit_ts_init_cutoff,
    audit_warmup_sensitivity,
)
from trader.evaluation.data_readiness import (
    DEFAULT_HOLDOUT_REGISTRY_PATH,
    audit_historical_data_readiness,
)
from trader.evaluation.metrics import TradeRecord, compute_strategy_metrics
from trader.evaluation.walk_forward import (
    CYCLE008_IMPLEMENTATION_MODULES,
    DEFAULT_ARCHIVED_DIAGNOSTIC_SESSIONS,
    Cycle008Config,
    Cycle008WalkForwardEvaluator,
    build_chronological_splits,
)

__all__ = [
    "BiasAuditResult",
    "BiasViolation",
    "CYCLE008_IMPLEMENTATION_MODULES",
    "Cycle008Config",
    "Cycle008WalkForwardEvaluator",
    "DEFAULT_ARCHIVED_DIAGNOSTIC_SESSIONS",
    "DEFAULT_HOLDOUT_REGISTRY_PATH",
    "TradeRecord",
    "audit_bar_index_lookahead",
    "audit_cross_sectional_lookahead",
    "audit_future_event_exclusion",
    "audit_historical_data_readiness",
    "audit_mtf_backward_alignment",
    "audit_recursive_convergence",
    "audit_ts_init_cutoff",
    "audit_warmup_sensitivity",
    "build_chronological_splits",
    "compute_strategy_metrics",
]
