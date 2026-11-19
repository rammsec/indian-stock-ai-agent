"""Token / estimated-cost accounting per agent."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional

from backend.phase6.config import MODEL_PRICES


class CostTracker:
    """Thread-safe tracker. Prices are ESTIMATES per 1M tokens (input, output)."""

    def __init__(self, prices: Optional[dict[str, tuple[float, float]]] = None):
        self.prices = dict(prices if prices is not None else MODEL_PRICES)
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usd": 0.0})
        self.unpriced_models: set[str] = set()

    def record(self, agent: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = self.prices.get(model, (0.0, 0.0))
        if model not in self.prices:
            self.unpriced_models.add(model)
        usd = (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
        with self._lock:
            r = self._rows[agent]
            r["prompt_tokens"] += prompt_tokens
            r["completion_tokens"] += completion_tokens
            r["calls"] += 1
            r["usd"] += usd
        return usd

    def summary(self) -> dict:
        with self._lock:
            per = {k: dict(v) for k, v in self._rows.items()}
        return {
            "per_agent": per,
            "total_prompt_tokens": sum(v["prompt_tokens"] for v in per.values()),
            "total_completion_tokens": sum(v["completion_tokens"] for v in per.values()),
            "total_estimated_usd": sum(v["usd"] for v in per.values()),
            "unpriced_models": sorted(self.unpriced_models),
            "note": "USD figures are estimates from a user-editable price table.",
        }
