import json

import pytest
from pydantic import BaseModel

from backend.phase6.config import Phase6Config
from backend.phase6.cost import CostTracker
from backend.phase6.llm import (AgentTimeout, InvalidStructuredOutput, LLMProviderError,
                                MockLLMClient, call_structured)
from backend.phase6.prompts import PROMPTS, RULE, get_prompt, prompt_hash
from backend.phase6.config import AGENT_NAMES
from backend.phase6.providers import get_provider


class Out(BaseModel):
    answer: int


def run(client, **kw):
    sleeps = []
    kw.setdefault("sleep", sleeps.append)
    out = call_structured(client, Out, "sys", "usr", model="mock", agent="a", **kw)
    return out, sleeps


def test_success():
    out, _ = run(MockLLMClient({"a": {"answer": 7}}))
    assert out.answer == 7


def test_transient_failure_retried_with_exponential_backoff():
    c = MockLLMClient()
    c.set("a", {"answer": 1}, transient_failures=2)
    out, sleeps = run(c, max_retries=3, backoff_s=0.1)
    assert out.answer == 1 and c.calls("a") == 3
    assert sleeps == [0.1, 0.2]


def test_retries_exhausted_raises_provider_error():
    c = MockLLMClient()
    c.set("a", {"answer": 1}, transient_failures=5)
    with pytest.raises(LLMProviderError):
        run(c, max_retries=2)
    assert c.calls("a") == 3


def test_timeout_raises_agent_timeout():
    c = MockLLMClient()
    c.set("a", {"answer": 1}, delay_s=0.5)
    with pytest.raises(AgentTimeout):
        run(c, timeout_s=0.05, max_retries=1)
    assert c.calls("a") == 2


def test_invalid_json_gets_one_repair_retry_then_succeeds():
    c = MockLLMClient()
    c.set("a", {"answer": 3}, invalid_first=1)
    out, _ = run(c)
    assert out.answer == 3 and c.calls("a") == 2
    assert "invalid" in c.call_log[1]["user"]


def test_invalid_json_twice_raises():
    c = MockLLMClient()
    c.set("a", {"answer": 3}, invalid_first=2)
    with pytest.raises(InvalidStructuredOutput):
        run(c)
    assert c.calls("a") == 2


def test_schema_violation_is_invalid_output():
    with pytest.raises(InvalidStructuredOutput):
        run(MockLLMClient({"a": {"answer": "NaN"}}))


def test_fenced_json_accepted():
    out, _ = run(MockLLMClient({"a": "```json\n" + json.dumps({"answer": 9}) + "\n```"}))
    assert out.answer == 9


def test_cost_tracker_math_and_integration():
    ct = CostTracker({"m": (2.0, 10.0)})
    usd = ct.record("bull", "m", 1_000_000, 500_000)
    assert usd == pytest.approx(2.0 + 5.0)
    ct.record("bull", "m", 1_000_000, 0)
    ct.record("bear", "unknown", 10, 10)
    s = ct.summary()
    assert s["per_agent"]["bull"]["calls"] == 2
    assert s["total_estimated_usd"] == pytest.approx(9.0)
    assert s["unpriced_models"] == ["unknown"]
    json.dumps(s)
    ct2 = CostTracker({"mock": (1.0, 1.0)})
    run(MockLLMClient({"a": {"answer": 1}}), cost_tracker=ct2)
    assert ct2.summary()["total_prompt_tokens"] > 0


def test_prompt_registry_hashing_and_rule():
    assert {a for a, _ in PROMPTS} == set(AGENT_NAMES)
    for a in AGENT_NAMES:
        assert "Never invent numbers" in get_prompt(a, "v1")
        h = prompt_hash(a, "v1")
        assert len(h) == 64 and h == prompt_hash(a, "v1")
    assert len({prompt_hash(a) for a in AGENT_NAMES}) == len(AGENT_NAMES)
    with pytest.raises(KeyError):
        get_prompt("bull", "v99")


def test_config_from_env_and_hash():
    c = Phase6Config.from_env({"PHASE6_PROVIDER": "ollama", "PHASE6_MAX_RETRIES": "5",
                               "PHASE6_DEEP_MODEL": "x"})
    assert (c.provider, c.max_retries, c.deep_think_model) == ("ollama", 5, "x")
    assert c.debate_rounds == 1 and c.namespace_dir == "data/phase6"
    assert c.config_hash() != Phase6Config().config_hash()


def test_provider_stubs_raise_not_implemented_and_no_network():
    for n in ("google", "anthropic", "ollama"):
        with pytest.raises(NotImplementedError, match=n):
            get_provider(n)
    with pytest.raises(KeyError):
        get_provider("nope")
    cl = get_provider("openai_compatible", api_key_env="P6_DEFINITELY_UNSET_KEY")
    with pytest.raises(LLMProviderError, match="not set"):
        cl.complete("s", "u", model="m", timeout_s=1)
