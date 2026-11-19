"""Real LLM provider adapters. NEVER exercised by tests (no network in CI).

API keys are read from the environment variable named in config; never hardcoded.
Only ``openai_compatible`` (and ``openai`` via the same wire format) is implemented.
"""
from __future__ import annotations

import os
from typing import Optional

from backend.phase6.llm import LLMClient, LLMProviderError, LLMResponse


class OpenAICompatibleClient:
    """Chat-completions adapter for any OpenAI-compatible HTTP endpoint."""

    def __init__(self, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", provider: str = "openai_compatible"):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.provider = provider

    def complete(self, system, user, *, model, timeout_s=60.0, response_schema=None,
                 agent: str = "") -> LLMResponse:
        import requests  # local import: keeps module import cheap/offline-safe

        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMProviderError(f"API key env var '{self.api_key_env}' is not set")
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=timeout_s,
            )
            r.raise_for_status()
            d = r.json()
            usage = d.get("usage", {})
            return LLMResponse(text=d["choices"][0]["message"]["content"],
                               prompt_tokens=int(usage.get("prompt_tokens", 0)),
                               completion_tokens=int(usage.get("completion_tokens", 0)),
                               model=model, provider=self.provider)
        except Exception as e:  # requests errors, bad payloads
            raise LLMProviderError(f"{type(e).__name__}: {e}") from e


def _not_impl(name: str):
    def factory(**_kw) -> LLMClient:
        raise NotImplementedError(
            f"Provider '{name}' is a stub and not implemented. "
            "Use 'openai_compatible' with a base_url instead."
        )
    return factory


def _openai(**kw):
    return OpenAICompatibleClient(provider="openai", **kw)


PROVIDER_REGISTRY = {
    "openai_compatible": lambda **kw: OpenAICompatibleClient(**kw),
    "openai": _openai,
    "google": _not_impl("google"),
    "anthropic": _not_impl("anthropic"),
    "ollama": _not_impl("ollama"),
}


def get_provider(name: str, **kwargs) -> LLMClient:
    """Instantiate a provider adapter by name."""
    if name not in PROVIDER_REGISTRY:
        raise KeyError(f"Unknown provider '{name}'. Known: {sorted(PROVIDER_REGISTRY)}")
    return PROVIDER_REGISTRY[name](**kwargs)
