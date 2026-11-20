"""Dynamic 4-level model routing, token budgeting, and cost optimization engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

# Standard Google Gemini API Pricing (USD per 1M tokens)
GEMINI_FLASH_INPUT_PRICE_PER_M = 0.075
GEMINI_FLASH_OUTPUT_PRICE_PER_M = 0.30
GEMINI_PRO_INPUT_PRICE_PER_M = 1.25
GEMINI_PRO_OUTPUT_PRICE_PER_M = 5.00


@dataclass
class AgentUsageStats:
    agent_name: str
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


class TokenBudgetManager:
    """Monitors token consumption and enforces hard financial budget circuit breakers."""

    def __init__(
        self,
        daily_budget_usd: float = 0.50,
        monthly_budget_usd: float = 10.00,
    ) -> None:
        self.daily_budget_usd = daily_budget_usd
        self.monthly_budget_usd = monthly_budget_usd
        self.current_daily_spend_usd: float = 0.0
        self.current_monthly_spend_usd: float = 0.0
        self.agent_stats: dict[str, AgentUsageStats] = {}
        self.is_daily_budget_exhausted: bool = False

    def record_usage(
        self,
        agent_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record usage and compute cost in USD."""
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = AgentUsageStats(agent_name=agent_name)

        stats = self.agent_stats[agent_name]
        stats.total_calls += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens

        # Calculate cost
        if "pro" in model_name.lower():
            in_cost = (input_tokens / 1_000_000.0) * GEMINI_PRO_INPUT_PRICE_PER_M
            out_cost = (output_tokens / 1_000_000.0) * GEMINI_PRO_OUTPUT_PRICE_PER_M
        else:
            in_cost = (input_tokens / 1_000_000.0) * GEMINI_FLASH_INPUT_PRICE_PER_M
            out_cost = (output_tokens / 1_000_000.0) * GEMINI_FLASH_OUTPUT_PRICE_PER_M

        call_cost = in_cost + out_cost
        stats.total_cost_usd += call_cost
        self.current_daily_spend_usd += call_cost
        self.current_monthly_spend_usd += call_cost

        if self.current_daily_spend_usd >= self.daily_budget_usd:
            self.is_daily_budget_exhausted = True

        return round(call_cost, 6)

    def can_invoke_llm(self) -> bool:
        """Check if remaining daily and monthly budgets allow LLM execution."""
        return not self.is_daily_budget_exhausted and (self.current_monthly_spend_usd < self.monthly_budget_usd)

    def get_summary(self) -> dict[str, Any]:
        """Return token and cost statistics for dashboard display."""
        total_tokens = sum(
            s.total_input_tokens + s.total_output_tokens for s in self.agent_stats.values()
        )
        return {
            "daily_budget_usd": self.daily_budget_usd,
            "current_daily_spend_usd": round(self.current_daily_spend_usd, 4),
            "remaining_daily_budget_usd": max(0.0, round(self.daily_budget_usd - self.current_daily_spend_usd, 4)),
            "budget_exhausted": self.is_daily_budget_exhausted,
            "total_tokens_consumed": total_tokens,
            "total_calls_count": sum(s.total_calls for s in self.agent_stats.values()),
            "agents": {
                name: {
                    "calls": s.total_calls,
                    "input_tokens": s.total_input_tokens,
                    "output_tokens": s.total_output_tokens,
                    "cost_usd": round(s.total_cost_usd, 4),
                }
                for name, s in self.agent_stats.items()
            },
        }


class ModelRouter:
    """Routes agent execution requests across the 4-level model hierarchy."""

    def __init__(self, budget_manager: TokenBudgetManager | None = None) -> None:
        self.budget_manager = budget_manager or TokenBudgetManager()

    def select_model(
        self,
        task_level: Literal[0, 1, 2, 3],
        agent_name: str,
        force_deterministic: bool = False,
    ) -> str:
        """Determine model class to use based on hierarchy and budget availability."""
        # Level 0 is always deterministic Python
        if task_level == 0 or force_deterministic:
            return "deterministic_python"

        # Check if budget is exhausted; fallback to deterministic logic
        if not self.budget_manager.can_invoke_llm():
            return "deterministic_fallback"

        if task_level == 1:
            return "gemini-2.5-flash"
        elif task_level == 2:
            return "gemini-2.5-pro"
        elif task_level == 3:
            return "gemini-2.5-pro-reasoning"

        return "deterministic_python"
