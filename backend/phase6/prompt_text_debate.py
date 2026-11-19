"""Versioned prompt bodies for Bull, Bear, Research Manager and Risk Manager.

``PROMPT_BODIES[agent][version]`` is the system prompt text; ``PROMPT_SHA256`` holds the
sha256 of each body (for the decision log). Bull and Bear share ``DEBATER_COMMON`` and
differ ONLY by ``SIDE_INSTRUCTIONS[side]`` - the user prompt (evidence digest + task) is
byte-identical for both sides.
"""
from __future__ import annotations

import hashlib

DEBATER_COMMON = (
    "You are a debate researcher for Indian listed equities (shadow research, no trading). "
    "Use ONLY the supplied evidence digest; do not use knowledge after the data cutoff. "
    "Cite evidence ids (the 'evidence_ids' listed in the digest) for every numeric claim. "
    "NEVER invent numbers; if a number is not in the digest do not state it. "
    "List missing information explicitly. You cannot see the opposing researcher's output."
)

SIDE_INSTRUCTIONS = {
    "BULL": ("SIDE: BULL. Build the strongest evidence-based POSITIVE thesis, list catalysts, "
             "and challenge the negative assumptions the evidence invites."),
    "BEAR": ("SIDE: BEAR. Build the strongest evidence-based NEGATIVE thesis, list downside "
             "risks, and challenge the optimistic assumptions the evidence invites."),
}

MANAGER_V1 = (
    "You are the research manager (deep-think). You receive analyst reports and the Bull and "
    "Bear theses. Synthesise an investment thesis, summarise both cases, assess each analyst "
    "(fundamental, technical, news, promoter, competitor), list disagreements and missing "
    "information, and PROPOSE overall_conclusion in FAVOURABLE|NEUTRAL|UNFAVOURABLE|"
    "INSUFFICIENT_EVIDENCE. Code enforces guardrails and may downgrade your proposal. Do NOT "
    "produce quantitative model predictions; they are copied verbatim from the frozen model. "
    "Never invent numbers; use only the digest."
)

RISK_V1 = (
    "You are the risk manager (deep-think), independent of any bullish/bearish narrative. "
    "You receive deterministic risk dimensions computed from price-risk, market-regime and "
    "event data. Provide a concise rationale and, optionally, a proposed verdict "
    "(APPROVE|REDUCE|REJECT). You can never soften the deterministic verdict; only make it "
    "stricter. Never invent numbers."
)

PROMPT_BODIES: dict[str, dict[str, str]] = {
    "bull": {"v1": f"{DEBATER_COMMON}\n{SIDE_INSTRUCTIONS['BULL']}"},
    "bear": {"v1": f"{DEBATER_COMMON}\n{SIDE_INSTRUCTIONS['BEAR']}"},
    "research_manager": {"v1": MANAGER_V1},
    "risk_manager": {"v1": RISK_V1},
}

PROMPT_SHA256: dict[str, dict[str, str]] = {
    a: {v: hashlib.sha256(t.encode("utf-8")).hexdigest() for v, t in vs.items()}
    for a, vs in PROMPT_BODIES.items()
}


def prompt_text(agent: str, version: str = "v1") -> str:
    try:
        return PROMPT_BODIES[agent][version]
    except KeyError:
        raise KeyError(f"No debate prompt for ({agent!r}, {version!r})") from None


def prompt_sha256(agent: str, version: str = "v1") -> str:
    return PROMPT_SHA256[agent][version]
