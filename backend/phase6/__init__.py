"""Phase 6: shadow-mode multi-agent research intelligence (core package).

Strictly additive: never modifies the frozen quantitative model, features,
news-intelligence pipeline, or stored predictions. See ``frozen_guard``.
"""
from backend.phase6.contracts import PHASE6_SCHEMA_VERSION

__all__ = ["PHASE6_SCHEMA_VERSION"]
