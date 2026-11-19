"""LLM client abstraction, deterministic mock, and structured-call helper."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class AgentTimeout(Exception):
    """LLM call exceeded the per-call timeout (after retries)."""


class InvalidStructuredOutput(Exception):
    """Model output was not valid JSON / did not satisfy the schema (after repair)."""


class LLMProviderError(Exception):
    """Transport / provider failure (after retries)."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""


class LLMClient(Protocol):
    """Minimal LLM interface. ``agent`` is an optional routing/accounting label."""

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        timeout_s: float,
        response_schema: Optional[type] = None,
        agent: str = "",
    ) -> LLMResponse: ...


Canned = Union[str, dict, list, Callable[[str, str], Any]]


@dataclass
class _Behavior:
    response: Canned = "{}"
    transient_failures: int = 0
    invalid_first: int = 0
    delay_s: float = 0.0
    calls: int = 0


class MockLLMClient:
    """Deterministic client keyed by agent name; never touches the network.

    Per-agent behaviours: canned response (str/dict/callable(system,user)),
    ``transient_failures`` (raise LLMProviderError N times), ``invalid_first``
    (return invalid JSON N times), ``delay_s`` (simulate slow call/timeout).
    """

    provider = "mock"

    def __init__(self, responses: Optional[dict[str, Canned]] = None):
        self._b: dict[str, _Behavior] = {}
        self.call_log: list[dict[str, Any]] = []
        for k, v in (responses or {}).items():
            self.set(k, v)

    def set(self, agent: str, response: Canned = "{}", *, transient_failures: int = 0,
            invalid_first: int = 0, delay_s: float = 0.0) -> None:
        self._b[agent] = _Behavior(response, transient_failures, invalid_first, delay_s)

    def calls(self, agent: str) -> int:
        return self._b[agent].calls if agent in self._b else 0

    def complete(self, system, user, *, model="mock", timeout_s=30.0,
                 response_schema=None, agent: str = "") -> LLMResponse:
        b = self._b.get(agent)
        if b is None:
            raise LLMProviderError(f"MockLLMClient: no behaviour for agent '{agent}'")
        b.calls += 1
        self.call_log.append({"agent": agent, "system": system, "user": user})
        if b.delay_s:
            time.sleep(b.delay_s)
        if b.calls <= b.transient_failures:
            raise LLMProviderError("simulated transient failure")
        if b.calls <= b.transient_failures + b.invalid_first:
            text = "this is not json {"
        else:
            r = b.response(system, user) if callable(b.response) else b.response
            text = r if isinstance(r, str) else json.dumps(r)
        return LLMResponse(text=text, prompt_tokens=max(1, (len(system) + len(user)) // 4),
                           completion_tokens=max(1, len(text) // 4), model=model,
                           provider=self.provider)


def _extract_json(text: str) -> Any:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    return json.loads(t.strip())


def _complete_with_timeout(client, system, user, model, timeout_s, schema, agent):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(client.complete, system, user, model=model, timeout_s=timeout_s,
                    response_schema=schema, agent=agent)
    try:
        return fut.result(timeout=timeout_s)
    except FutTimeout:
        raise AgentTimeout(f"LLM call for '{agent}' exceeded {timeout_s}s") from None
    finally:
        ex.shutdown(wait=False)


def call_structured(
    client: LLMClient,
    schema_cls: Type[T],
    system: str,
    user: str,
    *,
    model: str,
    agent: str = "",
    timeout_s: float = 30.0,
    max_retries: int = 2,
    backoff_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    cost_tracker: Any = None,
) -> T:
    """Call the LLM and return a validated ``schema_cls`` instance.

    * Provider errors / timeouts are retried up to ``max_retries`` times with
      exponential backoff (``sleep`` is injectable); then the last typed error is raised.
    * Invalid JSON / schema failures trigger exactly one repair retry, then
      :class:`InvalidStructuredOutput`.
    """
    schema_hint = json.dumps(schema_cls.model_json_schema())
    full_system = f"{system}\n\nRespond ONLY with JSON matching this schema:\n{schema_hint}"
    cur_user = user
    repaired = False
    attempt = 0
    while True:
        try:
            resp = _complete_with_timeout(client, full_system, cur_user, model, timeout_s,
                                          schema_cls, agent)
        except (AgentTimeout, LLMProviderError) as e:
            if attempt >= max_retries:
                if isinstance(e, AgentTimeout):
                    raise
                raise LLMProviderError(str(e)) from e
            sleep(backoff_s * (2 ** attempt))
            attempt += 1
            continue
        except Exception as e:  # unexpected client failure -> typed
            if attempt >= max_retries:
                raise LLMProviderError(f"{type(e).__name__}: {e}") from e
            sleep(backoff_s * (2 ** attempt))
            attempt += 1
            continue
        if cost_tracker is not None:
            cost_tracker.record(agent or "unknown", resp.model or model,
                                resp.prompt_tokens, resp.completion_tokens)
        try:
            return schema_cls.model_validate(_extract_json(resp.text))
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            if repaired:
                raise InvalidStructuredOutput(f"{agent}: {e}") from e
            repaired = True
            cur_user = (f"{user}\n\nYour previous output was invalid ({str(e)[:500]}). "
                        "Return corrected JSON only.")
