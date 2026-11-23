"""CYCLE-007 — Stage 1: Central Directional Agent.

Consumes strictly validated, timezone-aware pre-cutoff predictions, Phase 5
intelligence, and market regime inputs (reusing CYCLE-006B freshness/lookahead
validation semantics) and emits a deterministic, frozen ``DirectionalHypothesis``
for each symbol in the target universe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any

from trader.config import IST_TZ
from trader.intelligence.parsing import (
    load_json_strict,
    parse_finite_float,
    parse_iso_to_ist,
    parse_obs_date,
)
from trader.paper.cycle006b import SUPPORTED_PHASE5_SCHEMA_PREFIX


CONFIDENCE_CALIBRATION_STATUS = "UNCALIBRATED"
CONFIDENCE_CALIBRATION_NOTE = (
    "Heuristic directional confidence score; not a calibrated probability of a profitable trade."
)


class DirectionalBias(str, Enum):
    """Stage 1 directional hypothesis emitted by the Central Directional Agent."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class CentralDirectionalConfig:
    """Deterministic thresholds and freshness rules for the Central Directional Agent."""

    bullish_prob_threshold: float = 0.55
    bullish_return_threshold_pct: float = 0.25
    bearish_prob_threshold: float = 0.45
    bearish_return_threshold_pct: float = -0.25
    min_bullish_news_sentiment: float = 0.0
    max_staleness_hours: float = 72.0
    max_input_spread_hours: float = 48.0
    block_bullish_in_bear_regime: bool = True

    def __post_init__(self) -> None:
        if not (0.5 <= self.bullish_prob_threshold <= 1.0):
            raise ValueError("bullish_prob_threshold must be in [0.5, 1.0]")
        if not (0.0 <= self.bearish_prob_threshold <= 0.5):
            raise ValueError("bearish_prob_threshold must be in [0.0, 0.5]")
        if self.bullish_return_threshold_pct < 0.0:
            raise ValueError("bullish_return_threshold_pct must be >= 0")
        if self.bearish_return_threshold_pct > 0.0:
            raise ValueError("bearish_return_threshold_pct must be <= 0")
        if self.max_staleness_hours <= 0.0:
            raise ValueError("max_staleness_hours must be > 0")
        if self.max_input_spread_hours < 0.0:
            raise ValueError("max_input_spread_hours must be >= 0")


@dataclass(frozen=True)
class DirectionalHypothesis:
    """Frozen Stage 1 output for a single symbol."""

    symbol: str
    direction: DirectionalBias
    confidence: float
    prob_next_day_up: float
    expected_next_day_return_pct: float
    news_sentiment_score: float
    market_regime: str
    observation_date: date
    as_of_ist: datetime
    rationale: tuple[str, ...]
    confidence_calibration_status: str = CONFIDENCE_CALIBRATION_STATUS
    confidence_calibration_note: str = CONFIDENCE_CALIBRATION_NOTE

    def __post_init__(self) -> None:
        if self.confidence_calibration_status != CONFIDENCE_CALIBRATION_STATUS:
            raise ValueError(
                "DirectionalHypothesis.confidence_calibration_status must be "
                f"{CONFIDENCE_CALIBRATION_STATUS!r}; got {self.confidence_calibration_status!r}"
            )
        if (
            not isinstance(self.confidence_calibration_note, str)
            or not self.confidence_calibration_note.strip()
        ):
            raise ValueError(
                "DirectionalHypothesis.confidence_calibration_note must be a non-empty string"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "confidence_calibration_status": self.confidence_calibration_status,
            "confidence_calibration_note": self.confidence_calibration_note,
            "prob_next_day_up": round(self.prob_next_day_up, 4),
            "expected_next_day_return_pct": round(self.expected_next_day_return_pct, 4),
            "news_sentiment_score": round(self.news_sentiment_score, 4),
            "market_regime": self.market_regime,
            "observation_date": self.observation_date.isoformat(),
            "as_of_ist": self.as_of_ist.isoformat(),
            "rationale": list(self.rationale),
        }

    def to_authority_instruction(self, expires_after_hours: float = 8.0) -> Any:
        from ..authority.instructions import AuthorityInstruction, AuthoritySource, PermissionType
        perm = (
            PermissionType.ALLOW_LONG
            if self.direction == DirectionalBias.BULLISH
            else (
                PermissionType.ALLOW_SHORT
                if self.direction == DirectionalBias.BEARISH
                else PermissionType.NO_NEW_ENTRIES
            )
        )
        return AuthorityInstruction(
            instruction_id=f"CENTRAL-{self.symbol}-{self.observation_date.isoformat()}",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope=self.symbol,
            permission=perm,
            valid_from=self.as_of_ist,
            expires_at=self.as_of_ist + timedelta(hours=expires_after_hours),
            reason=f"Central Agent {self.direction.value} hypothesis (confidence: {self.confidence:.2f})",
            version="1.0",
        )

@dataclass(frozen=True)
class CentralDirectionalBatchResult:
    """Validated Stage 1 output across all requested symbols plus freshness audit."""

    as_of_ist: datetime
    market_regime: str
    hypotheses: dict[str, DirectionalHypothesis]
    freshness: dict[str, Any]
    input_file_hashes: dict[str, str]


def _ensure_ist(dt: datetime, label: str) -> datetime:
    if not isinstance(dt, datetime):
        raise ValueError(f"Expected datetime for {label}, got {type(dt).__name__}")
    if dt.tzinfo is None:
        raise ValueError(f"Timezone-naive datetime rejected for {label}: {dt!r}")
    return dt.astimezone(IST_TZ)


class CentralDirectionalAgent:
    """Stage 1 deterministic directional hypothesis agent (fail-closed, zero lookahead)."""

    def __init__(self, config: CentralDirectionalConfig | None = None) -> None:
        self.config = config or CentralDirectionalConfig()

    def evaluate_from_files(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        as_of_ist: datetime,
        predictions_path: Path,
        intelligence_path: Path,
        regime_path: Path,
    ) -> CentralDirectionalBatchResult:
        """Load JSON inputs strictly from disk, validate freshness/integrity, and emit hypotheses."""
        pred_raw, pred_hash = load_json_strict(predictions_path, "latest_predictions")
        intel_raw, intel_hash = load_json_strict(intelligence_path, "phase5_intelligence")
        regime_raw, regime_hash = load_json_strict(regime_path, "regime")

        file_hashes = {
            str(predictions_path): pred_hash,
            str(intelligence_path): intel_hash,
            str(regime_path): regime_hash,
        }
        return self.evaluate_from_payloads(
            symbols=symbols,
            as_of_ist=as_of_ist,
            predictions_payload=pred_raw,
            intelligence_payload=intel_raw,
            regime_payload=regime_raw,
            input_file_hashes=file_hashes,
        )

    def evaluate_from_payloads(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        as_of_ist: datetime,
        predictions_payload: dict[str, Any],
        intelligence_payload: dict[str, Any],
        regime_payload: dict[str, Any],
        input_file_hashes: dict[str, str] | None = None,
    ) -> CentralDirectionalBatchResult:
        """Strictly validate in-memory payloads and emit frozen ``DirectionalHypothesis`` objects."""
        cutoff_ist = _ensure_ist(as_of_ist, "as_of_ist")
        session_date = cutoff_ist.date()
        cfg = self.config

        if not symbols:
            raise ValueError("symbols must not be empty")
        norm_symbols = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
        if len(norm_symbols) != len(symbols) or len(set(norm_symbols)) != len(norm_symbols):
            raise ValueError("symbols must be non-empty, unique instrument identifiers")

        if not isinstance(predictions_payload, dict):
            raise ValueError("predictions_payload must be a dict")
        if not isinstance(intelligence_payload, dict):
            raise ValueError("intelligence_payload must be a dict")
        if not isinstance(regime_payload, dict):
            raise ValueError("regime_payload must be a dict")

        # 1. Validate predictions payload
        pred_ts = parse_iso_to_ist(
            predictions_payload.get("generated_at"), "latest_predictions.generated_at"
        )
        pred_list = predictions_payload.get("predictions")
        if not isinstance(pred_list, list) or not pred_list:
            raise ValueError("latest_predictions has missing or empty 'predictions' list")

        pred_by_sym: dict[str, dict[str, Any]] = {}
        for row in pred_list:
            if not isinstance(row, dict) or not row.get("symbol"):
                raise ValueError("Prediction record missing required 'symbol' identifier")
            sym = str(row["symbol"]).strip().upper()
            if sym in pred_by_sym:
                raise ValueError(f"Duplicate prediction record for symbol {sym}")
            pred_by_sym[sym] = row

        # 2. Validate Phase 5 intelligence payload
        model_ver = str(intelligence_payload.get("model_version", ""))
        if not model_ver.startswith(SUPPORTED_PHASE5_SCHEMA_PREFIX):
            raise ValueError(
                f"Incompatible phase5_intelligence schema/model_version: {model_ver!r} "
                f"(expected prefix {SUPPORTED_PHASE5_SCHEMA_PREFIX!r})"
            )
        intel_ts = parse_iso_to_ist(
            intelligence_payload.get("generated_at"), "phase5_intelligence.generated_at"
        )
        intel_stocks = intelligence_payload.get("stocks")
        if not isinstance(intel_stocks, dict) or not intel_stocks:
            raise ValueError("phase5_intelligence has missing or empty 'stocks' map")

        intel_by_sym: dict[str, dict[str, Any]] = {}
        for raw_k, val in intel_stocks.items():
            norm_k = str(raw_k).strip().upper()
            if norm_k in intel_by_sym:
                raise ValueError(f"Duplicate intelligence record for symbol {norm_k}")
            if isinstance(val, dict):
                intel_by_sym[norm_k] = val

        # 3. Validate regime payload
        regime_name = regime_payload.get("regime")
        if not isinstance(regime_name, str) or not regime_name.strip():
            raise ValueError("regime payload missing required 'regime' label")
        regime_label = regime_name.strip()
        regime_obs_date = parse_obs_date(
            regime_payload.get("observation_date"), "regime.observation_date"
        )
        if regime_payload.get("generated_at"):
            regime_ts = parse_iso_to_ist(regime_payload["generated_at"], "regime.generated_at")
        else:
            regime_ts = datetime.combine(regime_obs_date, dt_time(15, 30), tzinfo=IST_TZ)

        # 4. Strict as-of timestamp, staleness & cross-source spread checks
        freshness: dict[str, Any] = {
            "as_of_ist": cutoff_ist.isoformat(),
            "max_staleness_hours": cfg.max_staleness_hours,
            "max_input_spread_hours": cfg.max_input_spread_hours,
            "sources": {},
        }
        source_timestamps: dict[str, datetime] = {
            "latest_predictions": pred_ts,
            "phase5_intelligence": intel_ts,
            "regime": regime_ts,
        }
        for name, ts in source_timestamps.items():
            age_hours = round((cutoff_ist - ts).total_seconds() / 3600.0, 4)
            freshness["sources"][name] = {
                "timestamp_ist": ts.isoformat(),
                "age_hours": age_hours,
            }
            if ts > cutoff_ist:
                raise ValueError(
                    f"Future-dated input rejected ({name}): {ts.isoformat()} > as_of {cutoff_ist.isoformat()}"
                )
            if age_hours > cfg.max_staleness_hours:
                raise ValueError(
                    f"Stale input rejected ({name}): age={age_hours:.2f}h > max_staleness_hours={cfg.max_staleness_hours}"
                )

        regime_obs_dt = datetime.combine(regime_obs_date, dt_time(15, 30), tzinfo=IST_TZ)
        regime_obs_age_hours = round((cutoff_ist - regime_obs_dt).total_seconds() / 3600.0, 4)
        if regime_obs_date > session_date or regime_obs_dt > cutoff_ist:
            raise ValueError(
                f"Future-dated regime observation_date rejected: {regime_obs_date} "
                f"(EOD {regime_obs_dt.isoformat()} > as_of {cutoff_ist.isoformat()})"
            )
        if regime_obs_age_hours > cfg.max_staleness_hours:
            raise ValueError(
                f"Stale regime observation_date rejected: {regime_obs_date} "
                f"(age={regime_obs_age_hours:.2f}h > max_staleness_hours={cfg.max_staleness_hours})"
            )

        min_ts = min(source_timestamps.values())
        max_ts = max(source_timestamps.values())
        spread_hours = round((max_ts - min_ts).total_seconds() / 3600.0, 4)
        freshness["cross_source_spread_hours"] = spread_hours
        if spread_hours > cfg.max_input_spread_hours:
            raise ValueError(
                f"Conflicting input timestamps rejected: spread={spread_hours:.2f}h > "
                f"max_input_spread_hours={cfg.max_input_spread_hours}"
            )

        obs_freshness: dict[str, Any] = {
            "regime": {
                "observation_date": regime_obs_date.isoformat(),
                "eod_ist": regime_obs_dt.isoformat(),
                "age_hours": regime_obs_age_hours,
            },
            "predictions": {},
            "intelligence": {},
        }

        hypotheses: dict[str, DirectionalHypothesis] = {}
        is_bear_regime = "bear" in regime_label.lower()

        # 5. Per-symbol validation & directional hypothesis construction
        for sym in norm_symbols:
            if sym not in pred_by_sym:
                raise ValueError(f"Mismatched instruments: {sym} missing from latest_predictions")
            if sym not in intel_by_sym:
                raise ValueError(f"Mismatched instruments: {sym} missing from phase5_intelligence")

            p_row = pred_by_sym[sym]
            i_row = intel_by_sym[sym]

            p_obs = parse_obs_date(p_row.get("observation_date"), f"prediction[{sym}].observation_date")
            i_obs = parse_obs_date(i_row.get("observation_date"), f"intelligence[{sym}].observation_date")
            p_obs_dt = datetime.combine(p_obs, dt_time(15, 30), tzinfo=IST_TZ)
            i_obs_dt = datetime.combine(i_obs, dt_time(15, 30), tzinfo=IST_TZ)
            p_obs_age = round((cutoff_ist - p_obs_dt).total_seconds() / 3600.0, 4)
            i_obs_age = round((cutoff_ist - i_obs_dt).total_seconds() / 3600.0, 4)

            if p_obs > session_date or p_obs_dt > cutoff_ist:
                raise ValueError(
                    f"Future-dated prediction observation_date for {sym}: {p_obs} "
                    f"(EOD {p_obs_dt.isoformat()} > as_of {cutoff_ist.isoformat()})"
                )
            if p_obs_age > cfg.max_staleness_hours:
                raise ValueError(
                    f"Stale prediction observation_date rejected for {sym}: {p_obs} "
                    f"(age={p_obs_age:.2f}h > max_staleness_hours={cfg.max_staleness_hours})"
                )
            if i_obs > session_date or i_obs_dt > cutoff_ist:
                raise ValueError(
                    f"Future-dated intelligence observation_date for {sym}: {i_obs} "
                    f"(EOD {i_obs_dt.isoformat()} > as_of {cutoff_ist.isoformat()})"
                )
            if i_obs_age > cfg.max_staleness_hours:
                raise ValueError(
                    f"Stale intelligence observation_date rejected for {sym}: {i_obs} "
                    f"(age={i_obs_age:.2f}h > max_staleness_hours={cfg.max_staleness_hours})"
                )

            obs_freshness["predictions"][sym] = {
                "observation_date": p_obs.isoformat(),
                "eod_ist": p_obs_dt.isoformat(),
                "age_hours": p_obs_age,
            }
            obs_freshness["intelligence"][sym] = {
                "observation_date": i_obs.isoformat(),
                "eod_ist": i_obs_dt.isoformat(),
                "age_hours": i_obs_age,
            }

            prob_up = parse_finite_float(
                p_row.get("prob_next_day_up"), f"prediction[{sym}].prob_next_day_up"
            )
            if not (0.0 <= prob_up <= 1.0):
                raise ValueError(
                    f"Invalid probability prob_next_day_up for {sym}: {prob_up} (expected [0, 1])"
                )
            exp_ret = parse_finite_float(
                p_row.get("expected_next_day_return_pct"),
                f"prediction[{sym}].expected_next_day_return_pct",
            )
            news_sent = parse_finite_float(
                i_row.get("news_sentiment_score"),
                f"intelligence[{sym}].news_sentiment_score",
            )

            direction, confidence, rationale = self._classify_direction(
                prob_up=prob_up,
                exp_ret=exp_ret,
                news_sent=news_sent,
                regime_label=regime_label,
                is_bear_regime=is_bear_regime,
            )

            hypotheses[sym] = DirectionalHypothesis(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                prob_next_day_up=prob_up,
                expected_next_day_return_pct=exp_ret,
                news_sentiment_score=news_sent,
                market_regime=regime_label,
                observation_date=p_obs,
                as_of_ist=cutoff_ist,
                rationale=tuple(rationale),
            )

        freshness["observation_freshness"] = obs_freshness
        return CentralDirectionalBatchResult(
            as_of_ist=cutoff_ist,
            market_regime=regime_label,
            hypotheses=hypotheses,
            freshness=freshness,
            input_file_hashes=dict(input_file_hashes or {}),
        )

    def _classify_direction(
        self,
        *,
        prob_up: float,
        exp_ret: float,
        news_sent: float,
        regime_label: str,
        is_bear_regime: bool,
    ) -> tuple[DirectionalBias, float, list[str]]:
        cfg = self.config
        rationale: list[str] = []

        # Deterministic confidence component from probability distance + return strength
        prob_edge = min(1.0, abs(prob_up - 0.5) * 4.0)  # 0.75 -> 1.0, 0.55 -> 0.20
        ret_edge = min(1.0, abs(exp_ret) / 2.0)         # 2.0% -> 1.0
        raw_conf = max(0.0, min(1.0, 0.65 * prob_edge + 0.35 * ret_edge))

        if prob_up >= cfg.bullish_prob_threshold and exp_ret >= cfg.bullish_return_threshold_pct:
            if cfg.block_bullish_in_bear_regime and is_bear_regime:
                rationale.append(
                    f"Bullish model signal (prob_up={prob_up:.4f}, exp_ret={exp_ret:+.4f}%) "
                    f"gated to NEUTRAL by bearish regime ({regime_label})"
                )
                return DirectionalBias.NEUTRAL, round(min(raw_conf, 0.40), 4), rationale
            if news_sent < cfg.min_bullish_news_sentiment:
                rationale.append(
                    f"Bullish model signal (prob_up={prob_up:.4f}, exp_ret={exp_ret:+.4f}%) "
                    f"gated to NEUTRAL by negative news sentiment ({news_sent:+.4f} < {cfg.min_bullish_news_sentiment:.2f})"
                )
                return DirectionalBias.NEUTRAL, round(min(raw_conf, 0.45), 4), rationale

            # Boost confidence modestly if news sentiment confirms
            sent_bonus = 0.10 if news_sent > 0.1 else 0.0
            conf = max(0.0, min(1.0, raw_conf + sent_bonus))
            rationale.append(
                f"BULLISH hypothesis: prob_up={prob_up:.4f} >= {cfg.bullish_prob_threshold:.2f}, "
                f"exp_ret={exp_ret:+.4f}% >= {cfg.bullish_return_threshold_pct:+.2f}%, "
                f"news_sent={news_sent:+.4f}, regime={regime_label}"
            )
            return DirectionalBias.BULLISH, round(conf, 4), rationale

        if prob_up <= cfg.bearish_prob_threshold and exp_ret <= cfg.bearish_return_threshold_pct:
            sent_bonus = 0.10 if news_sent < -0.1 else 0.0
            regime_bonus = 0.10 if is_bear_regime else 0.0
            conf = max(0.0, min(1.0, raw_conf + sent_bonus + regime_bonus))
            rationale.append(
                f"BEARISH hypothesis: prob_up={prob_up:.4f} <= {cfg.bearish_prob_threshold:.2f}, "
                f"exp_ret={exp_ret:+.4f}% <= {cfg.bearish_return_threshold_pct:+.2f}%, "
                f"news_sent={news_sent:+.4f}, regime={regime_label}"
            )
            return DirectionalBias.BEARISH, round(conf, 4), rationale

        rationale.append(
            f"NEUTRAL hypothesis: thresholds not met (prob_up={prob_up:.4f}, "
            f"exp_ret={exp_ret:+.4f}%, news_sent={news_sent:+.4f}, regime={regime_label})"
        )
        # Neutral confidence stays bounded low
        neutral_conf = round(max(0.0, min(0.5, 0.5 - prob_edge * 0.5)), 4)
        if not math.isfinite(neutral_conf):
            neutral_conf = 0.0
        return DirectionalBias.NEUTRAL, neutral_conf, rationale
