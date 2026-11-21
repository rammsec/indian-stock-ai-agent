"""Dynamic 4-level model routing, token budgeting, and cost optimization engine."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

import httpx

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
    llm_calls: int = 0
    fallback_calls: int = 0


@dataclass(frozen=True)
class ProviderInvocationResult:
    """Structured result from an agent model invocation."""

    text: str
    model_used: str
    execution_mode: Literal["LLM", "DETERMINISTIC", "DETERMINISTIC_FALLBACK"]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    fallback_reason: str | None = None


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
        is_llm: bool = True,
    ) -> float:
        """Record usage and compute cost in USD."""
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = AgentUsageStats(agent_name=agent_name)

        stats = self.agent_stats[agent_name]
        stats.total_calls += 1
        if is_llm:
            stats.llm_calls += 1
        else:
            stats.fallback_calls += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens

        # Calculate cost
        if not is_llm or "deterministic" in model_name.lower():
            call_cost = 0.0
        elif "pro" in model_name.lower():
            in_cost = (input_tokens / 1_000_000.0) * GEMINI_PRO_INPUT_PRICE_PER_M
            out_cost = (output_tokens / 1_000_000.0) * GEMINI_PRO_OUTPUT_PRICE_PER_M
            call_cost = in_cost + out_cost
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
            "llm_calls_count": sum(s.llm_calls for s in self.agent_stats.values()),
            "fallback_calls_count": sum(s.fallback_calls for s in self.agent_stats.values()),
            "agents": {
                name: {
                    "calls": s.total_calls,
                    "llm_calls": s.llm_calls,
                    "fallback_calls": s.fallback_calls,
                    "input_tokens": s.total_input_tokens,
                    "output_tokens": s.total_output_tokens,
                    "cost_usd": round(s.total_cost_usd, 4),
                }
                for name, s in self.agent_stats.items()
            },
        }


class GeminiProvider:
    """Provider abstraction for Google Gemini REST API with strict timeout and structured usage extraction."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(self, model_name: str, prompt: str, system_instruction: str | None = None) -> tuple[str, int, int]:
        """Invoke Gemini generateContent API and return (text, input_tokens, output_tokens)."""
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY not configured")

        api_model = model_name.replace("-reasoning", "")
        url = f"{self.BASE_URL}/{api_model}:generateContent?key={self.api_key}"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        in_tok = int(usage.get("promptTokenCount", max(1, len(prompt) // 4)))
        out_tok = int(usage.get("candidatesTokenCount", max(1, len(text) // 4)))
        return text, in_tok, out_tok


class ModelRouter:
    """Routes agent execution requests across the 4-level model hierarchy."""

    def __init__(
        self,
        budget_manager: TokenBudgetManager | None = None,
        provider: GeminiProvider | None = None,
    ) -> None:
        self.budget_manager = budget_manager or TokenBudgetManager()
        self.provider = provider or GeminiProvider()

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

    def invoke_agent(
        self,
        task_level: Literal[0, 1, 2, 3],
        agent_name: str,
        prompt: str,
        deterministic_fn: Callable[[], str],
        system_instruction: str | None = None,
        force_deterministic: bool = False,
    ) -> ProviderInvocationResult:
        """Invoke LLM via GeminiProvider if permitted and available, else fallback deterministically."""
        selected = self.select_model(task_level, agent_name, force_deterministic=force_deterministic)

        if selected == "deterministic_python":
            out = deterministic_fn()
            self.budget_manager.record_usage(agent_name, selected, 0, 0, is_llm=False)
            return ProviderInvocationResult(
                text=out,
                model_used=selected,
                execution_mode="DETERMINISTIC",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )

        if selected == "deterministic_fallback":
            out = deterministic_fn()
            self.budget_manager.record_usage(agent_name, selected, 0, 0, is_llm=False)
            return ProviderInvocationResult(
                text=out,
                model_used=selected,
                execution_mode="DETERMINISTIC_FALLBACK",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                fallback_reason="Daily or monthly token budget exhausted",
            )

        if not self.provider.is_available():
            out = deterministic_fn()
            self.budget_manager.record_usage(agent_name, "deterministic_fallback", 0, 0, is_llm=False)
            return ProviderInvocationResult(
                text=out,
                model_used="deterministic_fallback",
                execution_mode="DETERMINISTIC_FALLBACK",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                fallback_reason="GEMINI_API_KEY not configured",
            )

        try:
            text, in_tok, out_tok = self.provider.generate(
                model_name=selected,
                prompt=prompt,
                system_instruction=system_instruction,
            )
            cost = self.budget_manager.record_usage(agent_name, selected, in_tok, out_tok, is_llm=True)
            return ProviderInvocationResult(
                text=text,
                model_used=selected,
                execution_mode="LLM",
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
            )
        except Exception as exc:
            out = deterministic_fn()
            self.budget_manager.record_usage(agent_name, "deterministic_fallback", 0, 0, is_llm=False)
            return ProviderInvocationResult(
                text=out,
                model_used="deterministic_fallback",
                execution_mode="DETERMINISTIC_FALLBACK",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                fallback_reason=f"Provider error: {exc}",
            )

