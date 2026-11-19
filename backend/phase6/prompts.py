"""Versioned prompt registry. Prompts here are PLACEHOLDERS for the next stage."""
from __future__ import annotations

import hashlib

RULE = ("Never invent numbers; list missing information; cite evidence ids. "
        "Use only the supplied evidence package; do not use knowledge after the data cutoff.")

_ROLES = {
    "fundamental": "You are a fundamental analyst for Indian listed equities.",
    "technical": "You are a technical analyst for Indian listed equities.",
    "news": "You are a news and corporate-announcement analyst.",
    "promoter": "You are an analyst of promoter holding, pledging and insider activity.",
    "competitor": "You are a peer/competitor comparison analyst.",
    "quantitative": "You summarise the frozen quantitative model output read-only; you never alter it.",
    "bull": "You are the Bull researcher; build the strongest evidence-based bullish case.",
    "bear": "You are the Bear researcher; build the strongest evidence-based bearish case.",
    "research_manager": "You are the research manager; synthesise analysts and the Bull/Bear debate.",
    "risk_manager": "You are the risk manager; you may APPROVE, REDUCE or REJECT.",
}

PROMPTS: dict[tuple[str, str], str] = {
    (name, "v1"): f"{role}\n{RULE}" for name, role in _ROLES.items()
}


def get_prompt(agent: str, version: str = "v1") -> str:
    try:
        return PROMPTS[(agent, version)]
    except KeyError:
        raise KeyError(f"No prompt registered for ({agent!r}, {version!r})") from None


def prompt_hash(agent: str, version: str = "v1") -> str:
    """Stable sha256 hex digest of the prompt text."""
    return hashlib.sha256(get_prompt(agent, version).encode("utf-8")).hexdigest()


def all_prompt_hashes(versions: dict[str, str]) -> dict[str, str]:
    """Hash of every pinned prompt, keyed by ``agent@version``."""
    return {f"{a}@{v}": prompt_hash(a, v) for a, v in sorted(versions.items())}
