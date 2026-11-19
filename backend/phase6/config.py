"""Phase 6 configuration (env-loadable via ``PHASE6_*``)."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# ESTIMATES ONLY (USD per 1M tokens: input, output). User-editable; verify against
# your provider's current pricing before relying on cost figures.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-flash": (0.10, 0.40),
    "claude-sonnet": (3.00, 15.00),
    "mock": (0.0, 0.0),
}

AGENT_NAMES = ["fundamental", "technical", "news", "promoter", "competitor", "quantitative",
               "bull", "bear", "research_manager", "risk_manager"]


class Phase6Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "openai_compatible"
    quick_think_provider: Optional[str] = None   # per-tier provider override
    deep_think_provider: Optional[str] = None
    quick_think_model: str = "gpt-4o-mini"
    deep_think_model: str = "gpt-4o"
    max_retries: int = 2
    timeout_s: float = 60.0
    debate_rounds: int = 1
    namespace_dir: str = "data/phase6"
    prompt_versions: dict[str, str] = Field(
        default_factory=lambda: {a: "v1" for a in AGENT_NAMES})
    model_prices: dict[str, tuple[float, float]] = Field(default_factory=lambda: dict(MODEL_PRICES))

    def provider_for(self, tier: str) -> str:
        return (self.quick_think_provider if tier == "quick" else self.deep_think_provider) or self.provider

    def model_for(self, tier: str) -> str:
        return self.quick_think_model if tier == "quick" else self.deep_think_model

    def snapshot(self) -> dict:
        """JSON-safe snapshot (no secrets exist in config)."""
        return json.loads(self.model_dump_json())

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "Phase6Config":
        """Load from PHASE6_PROVIDER, PHASE6_QUICK_MODEL, PHASE6_DEEP_MODEL,
        PHASE6_QUICK_PROVIDER, PHASE6_DEEP_PROVIDER, PHASE6_MAX_RETRIES,
        PHASE6_TIMEOUT_S, PHASE6_DEBATE_ROUNDS, PHASE6_NAMESPACE_DIR."""
        e = os.environ if env is None else env
        kw: dict = {}
        m = {"PHASE6_PROVIDER": ("provider", str), "PHASE6_QUICK_MODEL": ("quick_think_model", str),
             "PHASE6_DEEP_MODEL": ("deep_think_model", str),
             "PHASE6_QUICK_PROVIDER": ("quick_think_provider", str),
             "PHASE6_DEEP_PROVIDER": ("deep_think_provider", str),
             "PHASE6_MAX_RETRIES": ("max_retries", int), "PHASE6_TIMEOUT_S": ("timeout_s", float),
             "PHASE6_DEBATE_ROUNDS": ("debate_rounds", int),
             "PHASE6_NAMESPACE_DIR": ("namespace_dir", str)}
        for var, (field, typ) in m.items():
            if e.get(var):
                kw[field] = typ(e[var])
        return cls(**kw)
