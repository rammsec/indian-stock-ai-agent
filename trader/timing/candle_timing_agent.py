"""CYCLE-007 — Stage 2: Independent Candle Timing Agent.

Independently evaluates numerical candlestick anatomy, patterns, and price-action
structure via ``CandleIntelligenceEngine`` BEFORE comparing with the Stage 1
``DirectionalHypothesis``.

Hard independence rule:
- The Candle Timing Agent MUST NOT blindly agree with the Central Directional Agent.
- It derives its own ``candle_bias``, ``timing_quality_score``, structural
  ``invalidation_price``, and ``target_price``, and can emit:
  - ``ENTER_NOW`` (``CONFIRMED``)
  - ``WAIT`` (``UNCONFIRMED``)
  - ``NO_TRADE`` (``NEUTRAL_CENTRAL``)
  - ``CONTRADICT_CENTRAL`` (``CONTRADICTED``)
  - ``REJECT`` (``REJECTED`` — fail closed on invalid/corrupt data or unacceptable risk/volatility)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pandas as pd

from trader.config import IST_TZ
from trader.intelligence.structure import StructureSnapshot, analyze_market_structure
from trader.portfolio.ledger import round_to_nse_tick
from trader.timing.candle_intelligence import (
    BEARISH_PATTERNS,
    BULLISH_PATTERNS,
    INDECISION_PATTERNS,
    CandleBias,
    CandleIntelligenceEngine,
    CandleIntelligenceReport,
    CandlePattern,
    TrendStructure,
)
from trader.timing.central_directional_agent import (
    CONFIDENCE_CALIBRATION_STATUS,
    DirectionalBias,
    DirectionalHypothesis,
)


class TimingDecision(str, Enum):
    """Canonical Stage 2 timing decisions."""

    ENTER_NOW = "ENTER_NOW"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"
    CONTRADICT_CENTRAL = "CONTRADICT_CENTRAL"
    REJECT = "REJECT"


class AlignmentStatus(str, Enum):
    """Alignment between independent candle evidence and Central Directional Hypothesis."""

    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    NEUTRAL_CENTRAL = "NEUTRAL_CENTRAL"
    REJECTED = "REJECTED"


class AgentCall(str, Enum):
    """Canonical 9-call continuous intraday monitoring decision taxonomy.

    Explicitly separates opening a short (OPEN_SHORT) from exiting a long (EXIT_LONG),
    and opening a long (OPEN_LONG) from covering a short (COVER_SHORT).
    """

    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    WAIT = "WAIT"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    COVER_SHORT = "COVER_SHORT"
    NO_TRADE = "NO_TRADE"
    CANCEL_SETUP = "CANCEL_SETUP"
    REJECT = "REJECT"


@dataclass(frozen=True)
class PositionContext:
    """Active open position state passed into stateful bar-by-bar lifecycle evaluation."""

    side: Literal["LONG", "SHORT"]
    entry_price: float
    current_stop_loss: float
    target_price: float | None = None
    highest_price: float | None = None
    lowest_price: float | None = None
    bars_held: int = 0

    def __post_init__(self) -> None:
        if self.side not in ("LONG", "SHORT"):
            raise ValueError(f"PositionContext.side must be 'LONG' or 'SHORT'; got {self.side!r}")


@dataclass(frozen=True)
class ActiveSetupContext:
    """Watched setup state while FLAT awaiting trigger confirmation or invalidation."""

    direction: Literal["LONG", "SHORT"]
    setup_type: str
    trigger_price: float
    invalidation_price: float
    bars_watching: int = 0
    max_watch_bars: int = 6

    def __post_init__(self) -> None:
        if self.direction not in ("LONG", "SHORT"):
            raise ValueError(f"ActiveSetupContext.direction must be 'LONG' or 'SHORT'; got {self.direction!r}")


@dataclass(frozen=True)
class LifecycleAssessment:
    """Stateful bar-by-bar evaluation output emitting one of the 9 canonical AgentCall values."""

    call: AgentCall
    timing_assessment: TimingAssessment
    structure: StructureSnapshot
    recommended_trailing_stop: float | None
    updated_setup: ActiveSetupContext | None
    exit_reason: str | None
    cancel_reason: str | None
    rationale: tuple[str, ...]
    central_confidence_calibration_status: str = CONFIDENCE_CALIBRATION_STATUS

    def __post_init__(self) -> None:
        if self.central_confidence_calibration_status != CONFIDENCE_CALIBRATION_STATUS:
            raise ValueError(
                "LifecycleAssessment.central_confidence_calibration_status must be "
                f"{CONFIDENCE_CALIBRATION_STATUS!r}; got {self.central_confidence_calibration_status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.value,
            "timing_assessment": self.timing_assessment.to_dict(),
            "structure": self.structure.to_dict(),
            "recommended_trailing_stop": (
                round(self.recommended_trailing_stop, 2)
                if self.recommended_trailing_stop is not None
                else None
            ),
            "updated_setup": (
                {
                    "direction": self.updated_setup.direction,
                    "setup_type": self.updated_setup.setup_type,
                    "trigger_price": round(self.updated_setup.trigger_price, 2),
                    "invalidation_price": round(self.updated_setup.invalidation_price, 2),
                    "bars_watching": self.updated_setup.bars_watching,
                    "max_watch_bars": self.updated_setup.max_watch_bars,
                }
                if self.updated_setup is not None
                else None
            ),
            "exit_reason": self.exit_reason,
            "cancel_reason": self.cancel_reason,
            "rationale": list(self.rationale),
            "central_confidence_calibration_status": self.central_confidence_calibration_status,
        }


@dataclass(frozen=True)
class CandleTimingConfig:
    """Deterministic thresholds and structural risk parameters for Stage 2 timing."""

    entry_quality_threshold: float = 0.60
    contradiction_score_threshold: float = 0.20
    min_risk_reward_ratio: float = 1.5
    target_rr_multiplier: float = 2.0
    max_atr_pct: float = 6.0
    min_stop_distance_pct: float = 0.002
    max_stop_distance_pct: float = 0.06
    atr_stop_buffer_multiplier: float = 0.25
    min_relative_volume_for_entry: float = 0.50

    def __post_init__(self) -> None:
        if not (0.0 < self.entry_quality_threshold <= 1.0):
            raise ValueError("entry_quality_threshold must be in (0, 1]")
        if not (0.0 < self.contradiction_score_threshold <= 1.0):
            raise ValueError("contradiction_score_threshold must be in (0, 1]")
        if self.min_risk_reward_ratio < 1.0:
            raise ValueError("min_risk_reward_ratio must be >= 1.0")
        if self.target_rr_multiplier < self.min_risk_reward_ratio:
            raise ValueError("target_rr_multiplier must be >= min_risk_reward_ratio")
        if self.max_atr_pct <= 0.0:
            raise ValueError("max_atr_pct must be > 0")
        if not (0.0 < self.min_stop_distance_pct < self.max_stop_distance_pct < 1.0):
            raise ValueError("stop distance bounds must satisfy 0 < min < max < 1")
        if self.atr_stop_buffer_multiplier < 0.0:
            raise ValueError("atr_stop_buffer_multiplier must be >= 0")
        if self.min_relative_volume_for_entry < 0.0:
            raise ValueError("min_relative_volume_for_entry must be >= 0")


@dataclass(frozen=True)
class TimingAssessment:
    """Frozen Stage 2 independent timing evaluation output."""

    symbol: str
    as_of_ist: datetime
    decision: TimingDecision
    alignment_status: AlignmentStatus
    central_direction: DirectionalBias
    candle_bias: CandleBias
    timing_quality_score: float
    price_action_score: float
    entry_price: float | None
    invalidation_price: float | None
    target_price: float | None
    risk_reward_ratio: float | None
    detected_patterns: tuple[CandlePattern, ...]
    timeframe_statuses: dict[str, str]
    contradiction_reasons: tuple[str, ...]
    rationale: tuple[str, ...]
    intelligence_report: CandleIntelligenceReport | None = None
    central_confidence_calibration_status: str = CONFIDENCE_CALIBRATION_STATUS

    def __post_init__(self) -> None:
        if self.central_confidence_calibration_status != CONFIDENCE_CALIBRATION_STATUS:
            raise ValueError(
                "TimingAssessment.central_confidence_calibration_status must be "
                f"{CONFIDENCE_CALIBRATION_STATUS!r}; got "
                f"{self.central_confidence_calibration_status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of_ist": self.as_of_ist.isoformat(),
            "decision": self.decision.value,
            "alignment_status": self.alignment_status.value,
            "central_direction": self.central_direction.value,
            "central_confidence_calibration_status": self.central_confidence_calibration_status,
            "candle_bias": self.candle_bias.value,
            "timing_quality_score": round(self.timing_quality_score, 4),
            "price_action_score": round(self.price_action_score, 4),
            "entry_price": round(self.entry_price, 2) if self.entry_price is not None else None,
            "invalidation_price": (
                round(self.invalidation_price, 2)
                if self.invalidation_price is not None
                else None
            ),
            "target_price": round(self.target_price, 2) if self.target_price is not None else None,
            "risk_reward_ratio": (
                round(self.risk_reward_ratio, 3)
                if self.risk_reward_ratio is not None
                else None
            ),
            "detected_patterns": [p.value for p in self.detected_patterns],
            "timeframe_statuses": dict(self.timeframe_statuses),
            "contradiction_reasons": list(self.contradiction_reasons),
            "rationale": list(self.rationale),
            "intelligence_report": (
                self.intelligence_report.to_dict()
                if self.intelligence_report is not None
                else None
            ),
        }


class CandleTimingAgent:
    """Stage 2 Independent Candle Timing Agent (numerical-only, deterministic, fail-closed)."""

    def __init__(
        self,
        *,
        config: CandleTimingConfig | None = None,
        intelligence_engine: CandleIntelligenceEngine | None = None,
    ) -> None:
        self.config = config or CandleTimingConfig()
        self.engine = intelligence_engine or CandleIntelligenceEngine()

    def evaluate(
        self,
        *,
        hypothesis: DirectionalHypothesis,
        bars_primary: pd.DataFrame | None,
        primary_timeframe: str = "1h",
        bars_15m: pd.DataFrame | None = None,
        bars_5m: pd.DataFrame | None = None,
        as_of_ist: datetime | None = None,
    ) -> TimingAssessment:
        """Independently analyze candles first, then compare with ``hypothesis``.

        Never raises on corrupt/missing market data; fails closed with
        ``TimingDecision.REJECT``.
        """
        eval_ts = as_of_ist if as_of_ist is not None else hypothesis.as_of_ist
        if not isinstance(eval_ts, datetime) or eval_ts.tzinfo is None:
            return self._reject(
                symbol=hypothesis.symbol,
                as_of_ist=datetime.now(tz=IST_TZ),
                central_direction=hypothesis.direction,
                reason="Timezone-naive or invalid as_of_ist rejected (fail closed)",
            )
        eval_ist = eval_ts.astimezone(IST_TZ)

        if not isinstance(hypothesis.as_of_ist, datetime) or hypothesis.as_of_ist.tzinfo is None:
            return self._reject(
                symbol=hypothesis.symbol,
                as_of_ist=eval_ist,
                central_direction=hypothesis.direction,
                reason="Timezone-naive or invalid hypothesis.as_of_ist rejected (fail closed)",
            )
        hyp_ist = hypothesis.as_of_ist.astimezone(IST_TZ)
        if hyp_ist > eval_ist:
            return self._reject(
                symbol=hypothesis.symbol,
                as_of_ist=eval_ist,
                central_direction=hypothesis.direction,
                reason=(
                    f"Lookahead violation: hypothesis.as_of_ist ({hyp_ist.isoformat()}) "
                    f"> evaluation as_of_ist ({eval_ist.isoformat()}) (fail closed)"
                ),
            )

        # ------------------------------------------------------------------
        # STEP 1: INDEPENDENT CANDLE & PRICE-ACTION EVALUATION (BEFORE CENTRAL COMPARISON)
        # ------------------------------------------------------------------
        if bars_primary is None:
            return self._reject(
                symbol=hypothesis.symbol,
                as_of_ist=eval_ist,
                central_direction=hypothesis.direction,
                reason=f"Missing primary ({primary_timeframe}) OHLCV bars; failed closed",
            )

        try:
            report = self.engine.analyze(
                symbol=hypothesis.symbol,
                as_of_ist=eval_ist,
                bars_primary=bars_primary,
                primary_timeframe=primary_timeframe,
                bars_15m=bars_15m,
                bars_5m=bars_5m,
            )
        except Exception as exc:
            return self._reject(
                symbol=hypothesis.symbol,
                as_of_ist=eval_ist,
                central_direction=hypothesis.direction,
                reason=f"OHLCV validation failed closed: {exc}",
            )

        primary = report.primary
        latest = primary.latest_candle
        if latest is None:
            return self._reject(
                symbol=hypothesis.symbol,
                as_of_ist=eval_ist,
                central_direction=hypothesis.direction,
                reason="Primary timeframe produced no latest candle; failed closed",
                report=report,
            )

        tf_statuses = {primary.timeframe: primary.timeframe_status}
        for tf_name, tf_analysis in report.confirming_timeframes.items():
            tf_statuses[tf_name] = tf_analysis.timeframe_status

        candle_bias = report.directional_lean
        pa_score = report.composite_price_action_score
        patterns = primary.detected_patterns
        entry_price = float(latest.close)

        # Independent volatility safety check
        if primary.atr_pct is not None and primary.atr_pct > self.config.max_atr_pct:
            return TimingAssessment(
                symbol=report.symbol,
                as_of_ist=eval_ist,
                decision=TimingDecision.REJECT,
                alignment_status=AlignmentStatus.REJECTED,
                central_direction=hypothesis.direction,
                candle_bias=candle_bias,
                timing_quality_score=0.0,
                price_action_score=pa_score,
                entry_price=entry_price,
                invalidation_price=None,
                target_price=None,
                risk_reward_ratio=None,
                detected_patterns=patterns,
                timeframe_statuses=tf_statuses,
                contradiction_reasons=(),
                rationale=(
                    f"Volatility guard rejected entry: ATR%={primary.atr_pct:.2f}% > "
                    f"max_atr_pct={self.config.max_atr_pct:.2f}%",
                ),
                intelligence_report=report,
            )

        # ------------------------------------------------------------------
        # STEP 2: COMPARE INDEPENDENT EVIDENCE AGAINST CENTRAL HYPOTHESIS
        # ------------------------------------------------------------------
        if hypothesis.direction == DirectionalBias.NEUTRAL:
            return TimingAssessment(
                symbol=report.symbol,
                as_of_ist=eval_ist,
                decision=TimingDecision.NO_TRADE,
                alignment_status=AlignmentStatus.NEUTRAL_CENTRAL,
                central_direction=hypothesis.direction,
                candle_bias=candle_bias,
                timing_quality_score=round(abs(pa_score) * 0.5, 4),
                price_action_score=pa_score,
                entry_price=entry_price,
                invalidation_price=None,
                target_price=None,
                risk_reward_ratio=None,
                detected_patterns=patterns,
                timeframe_statuses=tf_statuses,
                contradiction_reasons=(),
                rationale=(
                    f"Central Directional Agent is NEUTRAL; standing aside (independent candle_bias={candle_bias.value}, score={pa_score:+.3f}).",
                ),
                intelligence_report=report,
            )

        # Check active contradiction BEFORE checking confirmation
        contradictions = self._detect_contradictions(
            central_direction=hypothesis.direction,
            candle_bias=candle_bias,
            pa_score=pa_score,
            report=report,
        )
        if contradictions:
            return TimingAssessment(
                symbol=report.symbol,
                as_of_ist=eval_ist,
                decision=TimingDecision.CONTRADICT_CENTRAL,
                alignment_status=AlignmentStatus.CONTRADICTED,
                central_direction=hypothesis.direction,
                candle_bias=candle_bias,
                timing_quality_score=0.0,
                price_action_score=pa_score,
                entry_price=entry_price,
                invalidation_price=None,
                target_price=None,
                risk_reward_ratio=None,
                detected_patterns=patterns,
                timeframe_statuses=tf_statuses,
                contradiction_reasons=tuple(contradictions),
                rationale=tuple(
                    [
                        f"Candle Timing Agent actively contradicts Central {hypothesis.direction.value} hypothesis "
                        f"(independent candle_bias={candle_bias.value}, score={pa_score:+.3f})."
                    ]
                    + contradictions
                ),
                intelligence_report=report,
            )

        # ------------------------------------------------------------------
        # STEP 3: STRUCTURAL INVALIDATION, TARGET & RISK-REWARD CALCULATION
        # ------------------------------------------------------------------
        invalidation_price, target_price, rr_ratio, level_error = self._compute_trade_levels(
            direction=hypothesis.direction,
            entry_price=entry_price,
            report=report,
        )
        if level_error is not None or invalidation_price is None or target_price is None or rr_ratio is None:
            return TimingAssessment(
                symbol=report.symbol,
                as_of_ist=eval_ist,
                decision=TimingDecision.REJECT,
                alignment_status=AlignmentStatus.REJECTED,
                central_direction=hypothesis.direction,
                candle_bias=candle_bias,
                timing_quality_score=0.0,
                price_action_score=pa_score,
                entry_price=entry_price,
                invalidation_price=invalidation_price,
                target_price=target_price,
                risk_reward_ratio=rr_ratio,
                detected_patterns=patterns,
                timeframe_statuses=tf_statuses,
                contradiction_reasons=(),
                rationale=(level_error or "Invalid structural stop/target levels; failed closed",),
                intelligence_report=report,
            )

        if rr_ratio < self.config.min_risk_reward_ratio:
            return TimingAssessment(
                symbol=report.symbol,
                as_of_ist=eval_ist,
                decision=TimingDecision.REJECT,
                alignment_status=AlignmentStatus.REJECTED,
                central_direction=hypothesis.direction,
                candle_bias=candle_bias,
                timing_quality_score=0.0,
                price_action_score=pa_score,
                entry_price=entry_price,
                invalidation_price=invalidation_price,
                target_price=target_price,
                risk_reward_ratio=rr_ratio,
                detected_patterns=patterns,
                timeframe_statuses=tf_statuses,
                contradiction_reasons=(),
                rationale=(
                    f"Risk-reward ratio {rr_ratio:.2f} below minimum {self.config.min_risk_reward_ratio:.2f}; failed closed",
                ),
                intelligence_report=report,
            )

        # ------------------------------------------------------------------
        # STEP 4: INDEPENDENT TIMING QUALITY SCORING & ENTER_NOW vs WAIT
        # ------------------------------------------------------------------
        quality_score, quality_notes, has_wait_blocker = self._compute_timing_quality(
            direction=hypothesis.direction,
            report=report,
        )

        expected_bias = (
            CandleBias.BULLISH
            if hypothesis.direction == DirectionalBias.BULLISH
            else CandleBias.BEARISH
        )

        if (
            candle_bias == expected_bias
            and quality_score >= self.config.entry_quality_threshold
            and not has_wait_blocker
        ):
            return TimingAssessment(
                symbol=report.symbol,
                as_of_ist=eval_ist,
                decision=TimingDecision.ENTER_NOW,
                alignment_status=AlignmentStatus.CONFIRMED,
                central_direction=hypothesis.direction,
                candle_bias=candle_bias,
                timing_quality_score=quality_score,
                price_action_score=pa_score,
                entry_price=entry_price,
                invalidation_price=invalidation_price,
                target_price=target_price,
                risk_reward_ratio=rr_ratio,
                detected_patterns=patterns,
                timeframe_statuses=tf_statuses,
                contradiction_reasons=(),
                rationale=tuple(
                    [
                        f"Independent candle evidence CONFIRMS Central {hypothesis.direction.value} "
                        f"(quality={quality_score:.3f} >= {self.config.entry_quality_threshold:.2f}, "
                        f"R:R={rr_ratio:.2f})."
                    ]
                    + quality_notes
                ),
                intelligence_report=report,
            )

        wait_reasons = [
            f"Awaiting stronger candle confirmation for Central {hypothesis.direction.value} "
            f"(candle_bias={candle_bias.value}, quality={quality_score:.3f}, "
            f"threshold={self.config.entry_quality_threshold:.2f})."
        ] + quality_notes

        return TimingAssessment(
            symbol=report.symbol,
            as_of_ist=eval_ist,
            decision=TimingDecision.WAIT,
            alignment_status=AlignmentStatus.UNCONFIRMED,
            central_direction=hypothesis.direction,
            candle_bias=candle_bias,
            timing_quality_score=quality_score,
            price_action_score=pa_score,
            entry_price=entry_price,
            invalidation_price=invalidation_price,
            target_price=target_price,
            risk_reward_ratio=rr_ratio,
            detected_patterns=patterns,
            timeframe_statuses=tf_statuses,
            contradiction_reasons=(),
            rationale=tuple(wait_reasons),
            intelligence_report=report,
        )

    def _detect_contradictions(
        self,
        *,
        central_direction: DirectionalBias,
        candle_bias: CandleBias,
        pa_score: float,
        report: CandleIntelligenceReport,
    ) -> list[str]:
        reasons: list[str] = []
        primary = report.primary
        patterns_set = set(primary.detected_patterns)

        if central_direction == DirectionalBias.BULLISH:
            if candle_bias == CandleBias.BEARISH:
                reasons.append(
                    f"Independent candle_bias is BEARISH (price_action_score={pa_score:+.3f})"
                )
            elif pa_score <= -self.config.contradiction_score_threshold:
                reasons.append(
                    f"Composite price_action_score={pa_score:+.3f} breaches bearish contradiction threshold "
                    f"(-{self.config.contradiction_score_threshold:.2f})"
                )

            bearish_hits = sorted(p.value for p in (patterns_set & BEARISH_PATTERNS))
            if bearish_hits and (
                primary.swing_structure == TrendStructure.DOWNTREND
                or primary.ema_alignment == "BEARISH"
                or primary.breakout_state in ("BREAKDOWN_LOW", "REJECTION_HIGH")
                or CandlePattern.BEARISH_ENGULFING in patterns_set
                or CandlePattern.SHOOTING_STAR in patterns_set
            ):
                reasons.append(
                    f"Bearish reversal/momentum pattern(s) detected against BULLISH hypothesis: {', '.join(bearish_hits)}"
                )

            if (
                primary.swing_structure == TrendStructure.DOWNTREND
                and primary.ema_alignment == "BEARISH"
                and primary.breakout_state in ("BREAKDOWN_LOW", "REJECTION_HIGH")
            ):
                reasons.append(
                    f"Bearish structure ({primary.swing_structure.value}), EMA alignment (BEARISH), "
                    f"and {primary.breakout_state} oppose BULLISH hypothesis"
                )

        elif central_direction == DirectionalBias.BEARISH:
            if candle_bias == CandleBias.BULLISH:
                reasons.append(
                    f"Independent candle_bias is BULLISH (price_action_score={pa_score:+.3f})"
                )
            elif pa_score >= self.config.contradiction_score_threshold:
                reasons.append(
                    f"Composite price_action_score={pa_score:+.3f} breaches bullish contradiction threshold "
                    f"(+{self.config.contradiction_score_threshold:.2f})"
                )

            bullish_hits = sorted(p.value for p in (patterns_set & BULLISH_PATTERNS))
            if bullish_hits and (
                primary.swing_structure == TrendStructure.UPTREND
                or primary.ema_alignment == "BULLISH"
                or primary.breakout_state in ("BREAKOUT_HIGH", "REJECTION_LOW")
                or CandlePattern.BULLISH_ENGULFING in patterns_set
                or CandlePattern.HAMMER in patterns_set
            ):
                reasons.append(
                    f"Bullish reversal/momentum pattern(s) detected against BEARISH hypothesis: {', '.join(bullish_hits)}"
                )

            if (
                primary.swing_structure == TrendStructure.UPTREND
                and primary.ema_alignment == "BULLISH"
                and primary.breakout_state in ("BREAKOUT_HIGH", "REJECTION_LOW")
            ):
                reasons.append(
                    f"Bullish structure ({primary.swing_structure.value}), EMA alignment (BULLISH), "
                    f"and {primary.breakout_state} oppose BEARISH hypothesis"
                )

        return reasons

    def _compute_trade_levels(
        self,
        *,
        direction: DirectionalBias,
        entry_price: float,
        report: CandleIntelligenceReport,
    ) -> tuple[float | None, float | None, float | None, str | None]:
        primary = report.primary
        latest = primary.latest_candle
        if latest is None or entry_price <= 0.0 or not math.isfinite(entry_price):
            return None, None, None, "Invalid entry price for structural level calculation"

        atr = primary.atr if (primary.atr is not None and primary.atr > 0.0) else entry_price * 0.01
        buffer = atr * self.config.atr_stop_buffer_multiplier

        if direction == DirectionalBias.BULLISH:
            structural_low = min(
                latest.low,
                primary.recent_swing_low if primary.recent_swing_low is not None else latest.low,
            )
            raw_stop = structural_low - buffer
            risk_per_share = entry_price - raw_stop
            min_risk = entry_price * self.config.min_stop_distance_pct
            max_risk = entry_price * self.config.max_stop_distance_pct
            if risk_per_share < min_risk:
                raw_stop = entry_price - min_risk
                risk_per_share = min_risk
            if risk_per_share > max_risk:
                return (
                    round(raw_stop, 2),
                    None,
                    None,
                    f"Structural stop distance ({(risk_per_share / entry_price) * 100:.2f}%) exceeds "
                    f"max_stop_distance_pct ({self.config.max_stop_distance_pct * 100:.2f}%)",
                )
            if raw_stop <= 0.0 or raw_stop >= entry_price:
                return None, None, None, "Computed invalidation_price is non-positive or >= entry_price"
            target_price = entry_price + (risk_per_share * self.config.target_rr_multiplier)
            rr = (target_price - entry_price) / risk_per_share
            return round(raw_stop, 2), round(target_price, 2), round(rr, 4), None

        if direction == DirectionalBias.BEARISH:
            structural_high = max(
                latest.high,
                primary.recent_swing_high if primary.recent_swing_high is not None else latest.high,
            )
            raw_stop = structural_high + buffer
            risk_per_share = raw_stop - entry_price
            min_risk = entry_price * self.config.min_stop_distance_pct
            max_risk = entry_price * self.config.max_stop_distance_pct
            if risk_per_share < min_risk:
                raw_stop = entry_price + min_risk
                risk_per_share = min_risk
            if risk_per_share > max_risk:
                return (
                    round(raw_stop, 2),
                    None,
                    None,
                    f"Structural stop distance ({(risk_per_share / entry_price) * 100:.2f}%) exceeds "
                    f"max_stop_distance_pct ({self.config.max_stop_distance_pct * 100:.2f}%)",
                )
            target_price = entry_price - (risk_per_share * self.config.target_rr_multiplier)
            if target_price <= 0.0:
                return None, None, None, "Computed short target_price is non-positive"
            rr = (entry_price - target_price) / risk_per_share
            return round(raw_stop, 2), round(target_price, 2), round(rr, 4), None

        return None, None, None, "Unsupported direction for trade level computation"

    def _compute_timing_quality(
        self,
        *,
        direction: DirectionalBias,
        report: CandleIntelligenceReport,
    ) -> tuple[float, list[str], bool]:
        primary = report.primary
        patterns_set = set(primary.detected_patterns)
        notes: list[str] = []
        has_wait_blocker = False

        # Base aligned price-action magnitude (0.0 .. 0.45)
        aligned_score = (
            report.composite_price_action_score
            if direction == DirectionalBias.BULLISH
            else -report.composite_price_action_score
        )
        quality = max(0.0, min(0.45, aligned_score * 0.55))

        # Pattern confirmation (+0.20)
        if direction == DirectionalBias.BULLISH:
            supportive_pats = sorted(p.value for p in (patterns_set & BULLISH_PATTERNS))
            if supportive_pats:
                quality += 0.20
                notes.append(f"Supportive bullish pattern(s): {', '.join(supportive_pats)}")
            if primary.swing_structure == TrendStructure.UPTREND:
                quality += 0.12
                notes.append("Swing structure supports UPTREND (HH/HL)")
            if primary.ema_alignment == "BULLISH":
                quality += 0.12
                notes.append("EMA alignment supports BULLISH trend")
            if primary.vwap_position == "ABOVE":
                quality += 0.08
                notes.append("Price trading ABOVE session VWAP")
            if primary.breakout_state in ("BREAKOUT_HIGH", "REJECTION_LOW"):
                quality += 0.08
                notes.append(f"Price-action breakout/rejection supportive: {primary.breakout_state}")
        else:
            supportive_pats = sorted(p.value for p in (patterns_set & BEARISH_PATTERNS))
            if supportive_pats:
                quality += 0.20
                notes.append(f"Supportive bearish pattern(s): {', '.join(supportive_pats)}")
            if primary.swing_structure == TrendStructure.DOWNTREND:
                quality += 0.12
                notes.append("Swing structure supports DOWNTREND (LH/LL)")
            if primary.ema_alignment == "BEARISH":
                quality += 0.12
                notes.append("EMA alignment supports BEARISH trend")
            if primary.vwap_position == "BELOW":
                quality += 0.08
                notes.append("Price trading BELOW session VWAP")
            if primary.breakout_state in ("BREAKDOWN_LOW", "REJECTION_HIGH"):
                quality += 0.08
                notes.append(f"Price-action breakdown/rejection supportive: {primary.breakout_state}")

        # Relative volume confirmation / low-volume wait guard
        if primary.relative_volume_ratio is not None:
            if primary.relative_volume_ratio >= 1.1:
                quality += 0.08
                notes.append(f"Relative volume supportive ({primary.relative_volume_ratio:.2f}x)")
            elif primary.relative_volume_ratio < self.config.min_relative_volume_for_entry:
                quality -= 0.18
                has_wait_blocker = True
                notes.append(
                    f"Low relative volume ({primary.relative_volume_ratio:.2f}x < "
                    f"{self.config.min_relative_volume_for_entry:.2f}x) — awaiting participation"
                )

        # Indecision candle penalty (DOJI / INSIDE_BAR without strong directional breakout)
        indecision_hits = sorted(p.value for p in (patterns_set & INDECISION_PATTERNS))
        if indecision_hits:
            quality -= 0.22
            if not supportive_pats:
                has_wait_blocker = True
            notes.append(
                f"Indecision pattern detected ({', '.join(indecision_hits)}) — awaiting breakout confirmation"
            )

        quality = round(max(0.0, min(1.0, quality)), 4)
        return quality, notes, has_wait_blocker

    @staticmethod
    def _reject(
        *,
        symbol: str,
        as_of_ist: datetime,
        central_direction: DirectionalBias,
        reason: str,
        report: CandleIntelligenceReport | None = None,
    ) -> TimingAssessment:
        return TimingAssessment(
            symbol=symbol,
            as_of_ist=as_of_ist,
            decision=TimingDecision.REJECT,
            alignment_status=AlignmentStatus.REJECTED,
            central_direction=central_direction,
            candle_bias=CandleBias.NEUTRAL,
            timing_quality_score=0.0,
            price_action_score=0.0,
            entry_price=None,
            invalidation_price=None,
            target_price=None,
            risk_reward_ratio=None,
            detected_patterns=(),
            timeframe_statuses={},
            contradiction_reasons=(),
            rationale=(reason,),
            intelligence_report=report,
        )

    def evaluate_bar_lifecycle(
        self,
        *,
        hypothesis: DirectionalHypothesis,
        bars_primary: pd.DataFrame | None,
        primary_timeframe: str = "1h",
        bars_15m: pd.DataFrame | None = None,
        bars_5m: pd.DataFrame | None = None,
        as_of_ist: datetime | None = None,
        position: PositionContext | None = None,
        active_setup: ActiveSetupContext | None = None,
    ) -> LifecycleAssessment:
        """Stateful bar-by-bar evaluation emitting one of the 9 canonical AgentCall values.

        - When `position` is open (LONG or SHORT): evaluates protective exits (EXIT_LONG / COVER_SHORT)
          or HOLD with monotonic ATR trailing stop ratcheting. Never confuses EXIT_LONG with OPEN_SHORT
          or COVER_SHORT with OPEN_LONG.
        - When FLAT with `active_setup`: confirms entry (OPEN_LONG / OPEN_SHORT), continues waiting (WAIT),
          or cancels invalidated/expired/contradicted setups (CANCEL_SETUP).
        - When FLAT without `active_setup`: emits OPEN_LONG, OPEN_SHORT, WAIT (registering a watched setup),
          NO_TRADE, or REJECT.
        """
        base = self.evaluate(
            hypothesis=hypothesis,
            bars_primary=bars_primary,
            primary_timeframe=primary_timeframe,
            bars_15m=bars_15m,
            bars_5m=bars_5m,
            as_of_ist=as_of_ist,
        )

        # Fail closed BEFORE structure analysis or position management whenever temporal/data
        # integrity validation rejected the input (e.g., lookahead violation, naive datetime, corrupt OHLCV).
        if (
            bars_primary is None
            or bars_primary.empty
            or base.intelligence_report is None
            or base.intelligence_report.primary.latest_candle is None
        ):
            return LifecycleAssessment(
                call=AgentCall.REJECT,
                timing_assessment=base,
                structure=analyze_market_structure(None, max_atr_pct=self.config.max_atr_pct),
                recommended_trailing_stop=position.current_stop_loss if position else None,
                updated_setup=None,
                exit_reason=None,
                cancel_reason="REJECTED_BY_TIMING_GUARD" if active_setup is not None else None,
                rationale=tuple(base.rationale),
            )

        structure = analyze_market_structure(bars_primary, max_atr_pct=self.config.max_atr_pct)

        rationale: list[str] = list(base.rationale)
        if structure.volume_regime == "VOLUME_UNREPORTED_BY_PROVIDER":
            rationale.append(
                "Provider reported volume=0 on active bar (VOLUME_UNREPORTED_BY_PROVIDER); "
                "relative volume unavailable rather than zero participation."
            )

        curr_close = float(bars_primary["close"].iloc[-1])
        curr_low = float(bars_primary["low"].iloc[-1])
        curr_high = float(bars_primary["high"].iloc[-1])
        primary_tf_report = base.intelligence_report.primary
        atr_val = (
            primary_tf_report.atr
            if (primary_tf_report is not None and primary_tf_report.atr is not None and primary_tf_report.atr > 0)
            else max(0.05, curr_close * 0.01)
        )

        # ------------------------------------------------------------------
        # BRANCH 1: Managing an Open LONG or SHORT Position
        # ------------------------------------------------------------------
        if position is not None:
            if position.side == "LONG":
                exit_reason: str | None = None
                if curr_close <= position.current_stop_loss:
                    exit_reason = f"STOP_LOSS_BREACHED:close={curr_close:.2f}<=stop={position.current_stop_loss:.2f}"
                elif position.target_price is not None and curr_close >= position.target_price:
                    exit_reason = f"TARGET_REACHED:close={curr_close:.2f}>=target={position.target_price:.2f}"
                elif structure.chart_pattern == "HEAD_AND_SHOULDERS_BREAKDOWN":
                    exit_reason = "STRUCTURAL_EXIT:HEAD_AND_SHOULDERS_BREAKDOWN"
                elif structure.reversal_event == "BEARISH_REVERSAL":
                    exit_reason = "STRUCTURAL_EXIT:BEARISH_REVERSAL_CHOCH"
                elif structure.breakout_event in ("BEARISH_BREAKDOWN", "FALSE_BREAKOUT"):
                    exit_reason = f"STRUCTURAL_EXIT:{structure.breakout_event}"
                elif base.candle_bias == CandleBias.BEARISH and base.price_action_score <= -0.35:
                    exit_reason = (
                        f"BEARISH_CONTRADICTION_EXIT:pa_score={base.price_action_score:.2f}"
                    )

                if exit_reason is not None:
                    rationale.append(f"Open LONG position exit triggered: {exit_reason}")
                    return LifecycleAssessment(
                        call=AgentCall.EXIT_LONG,
                        timing_assessment=base,
                        structure=structure,
                        recommended_trailing_stop=position.current_stop_loss,
                        updated_setup=None,
                        exit_reason=exit_reason,
                        cancel_reason=None,
                        rationale=tuple(rationale),
                    )

                # HOLD with monotonic upward trailing stop ratchet once trade moves >= 0.5*ATR in favor
                trailed_stop = position.current_stop_loss
                if curr_close >= position.entry_price + 0.5 * atr_val:
                    candidate = round_to_nse_tick(curr_close - 1.5 * atr_val)
                    max_safe_stop = round_to_nse_tick(curr_close - 0.25 * atr_val)
                    candidate = min(candidate, max_safe_stop)
                    if candidate > trailed_stop:
                        trailed_stop = candidate
                        rationale.append(
                            f"Trailing stop ratcheted upward: {position.current_stop_loss:.2f} -> {trailed_stop:.2f}"
                        )
                rationale.append(f"HOLD open LONG (close={curr_close:.2f}, stop={trailed_stop:.2f})")
                return LifecycleAssessment(
                    call=AgentCall.HOLD,
                    timing_assessment=base,
                    structure=structure,
                    recommended_trailing_stop=trailed_stop,
                    updated_setup=None,
                    exit_reason=None,
                    cancel_reason=None,
                    rationale=tuple(rationale),
                )

            # position.side == "SHORT"
            cover_reason: str | None = None
            if curr_close >= position.current_stop_loss:
                cover_reason = f"STOP_LOSS_BREACHED:close={curr_close:.2f}>=stop={position.current_stop_loss:.2f}"
            elif position.target_price is not None and curr_close <= position.target_price:
                cover_reason = f"TARGET_REACHED:close={curr_close:.2f}<=target={position.target_price:.2f}"
            elif structure.chart_pattern == "INVERSE_HEAD_AND_SHOULDERS_BREAKOUT":
                cover_reason = "STRUCTURAL_EXIT:INVERSE_HEAD_AND_SHOULDERS_BREAKOUT"
            elif structure.reversal_event == "BULLISH_REVERSAL":
                cover_reason = "STRUCTURAL_EXIT:BULLISH_REVERSAL_CHOCH"
            elif structure.breakout_event in ("BULLISH_BREAKOUT", "FAILED_BREAKDOWN"):
                cover_reason = f"STRUCTURAL_EXIT:{structure.breakout_event}"
            elif base.candle_bias == CandleBias.BULLISH and base.price_action_score >= 0.35:
                cover_reason = f"BULLISH_CONTRADICTION_COVER:pa_score={base.price_action_score:.2f}"

            if cover_reason is not None:
                rationale.append(f"Open SHORT position cover triggered: {cover_reason}")
                return LifecycleAssessment(
                    call=AgentCall.COVER_SHORT,
                    timing_assessment=base,
                    structure=structure,
                    recommended_trailing_stop=position.current_stop_loss,
                    updated_setup=None,
                    exit_reason=cover_reason,
                    cancel_reason=None,
                    rationale=tuple(rationale),
                )

            # HOLD SHORT with monotonic downward trailing stop ratchet
            trailed_stop = position.current_stop_loss
            if curr_close <= position.entry_price - 0.5 * atr_val:
                candidate = round_to_nse_tick(curr_close + 1.5 * atr_val)
                min_safe_stop = round_to_nse_tick(curr_close + 0.25 * atr_val)
                candidate = max(candidate, min_safe_stop)
                if candidate < trailed_stop:
                    trailed_stop = candidate
                    rationale.append(
                        f"Short trailing stop ratcheted downward: {position.current_stop_loss:.2f} -> {trailed_stop:.2f}"
                    )
            rationale.append(f"HOLD open SHORT (close={curr_close:.2f}, stop={trailed_stop:.2f})")
            return LifecycleAssessment(
                call=AgentCall.HOLD,
                timing_assessment=base,
                structure=structure,
                recommended_trailing_stop=trailed_stop,
                updated_setup=None,
                exit_reason=None,
                cancel_reason=None,
                rationale=tuple(rationale),
            )

        # ------------------------------------------------------------------
        # BRANCH 2: FLAT — Evaluating Active Setup or Fresh Entry Candidate
        # ------------------------------------------------------------------
        if base.decision == TimingDecision.REJECT:
            return LifecycleAssessment(
                call=AgentCall.REJECT,
                timing_assessment=base,
                structure=structure,
                recommended_trailing_stop=None,
                updated_setup=None,
                exit_reason=None,
                cancel_reason="REJECTED_BY_TIMING_GUARD" if active_setup is not None else None,
                rationale=tuple(rationale),
            )

        if base.decision == TimingDecision.NO_TRADE:
            if active_setup is not None:
                cancel_reason = "CENTRAL_DIRECTION_NEUTRAL"
                rationale.append(f"Active setup cancelled: {cancel_reason}")
                return LifecycleAssessment(
                    call=AgentCall.CANCEL_SETUP,
                    timing_assessment=base,
                    structure=structure,
                    recommended_trailing_stop=None,
                    updated_setup=None,
                    exit_reason=None,
                    cancel_reason=cancel_reason,
                    rationale=tuple(rationale),
                )
            return LifecycleAssessment(
                call=AgentCall.NO_TRADE,
                timing_assessment=base,
                structure=structure,
                recommended_trailing_stop=None,
                updated_setup=None,
                exit_reason=None,
                cancel_reason=None,
                rationale=tuple(rationale),
            )

        # Check active setup cancellation rules (direction flip, invalidation breach, trap/contradiction, timeout)
        if active_setup is not None:
            cancel_reason: str | None = None
            if (
                active_setup.direction == "LONG"
                and hypothesis.direction == DirectionalBias.BEARISH
            ) or (
                active_setup.direction == "SHORT"
                and hypothesis.direction == DirectionalBias.BULLISH
            ):
                cancel_reason = "CENTRAL_DIRECTION_REVERSED"
            elif active_setup.direction == "LONG":
                if curr_low <= active_setup.invalidation_price or curr_close < active_setup.invalidation_price:
                    cancel_reason = (
                        f"SETUP_INVALIDATED:low={curr_low:.2f}<=invalidation={active_setup.invalidation_price:.2f}"
                    )
                elif base.decision == TimingDecision.CONTRADICT_CENTRAL:
                    cancel_reason = "SETUP_CONTRADICTED_BY_CANDLE_EVIDENCE"
                elif structure.breakout_event == "FALSE_BREAKOUT" or structure.orb_event == "FAILED_ORB_HIGH":
                    cancel_reason = f"SETUP_TRAP_DETECTED:{structure.breakout_event}/{structure.orb_event}"
                elif structure.ema_cross_event == "FAILED_BULLISH_CROSSOVER":
                    cancel_reason = "SETUP_FAILED_BULLISH_CROSSOVER"
            else:
                if curr_high >= active_setup.invalidation_price or curr_close > active_setup.invalidation_price:
                    cancel_reason = (
                        f"SETUP_INVALIDATED:high={curr_high:.2f}>=invalidation={active_setup.invalidation_price:.2f}"
                    )
                elif base.decision == TimingDecision.CONTRADICT_CENTRAL:
                    cancel_reason = "SETUP_CONTRADICTED_BY_CANDLE_EVIDENCE"
                elif structure.breakout_event == "FAILED_BREAKDOWN" or structure.orb_event == "FAILED_ORB_LOW":
                    cancel_reason = f"SETUP_TRAP_DETECTED:{structure.breakout_event}/{structure.orb_event}"
                elif structure.ema_cross_event == "FAILED_BEARISH_CROSSOVER":
                    cancel_reason = "SETUP_FAILED_BEARISH_CROSSOVER"

            if cancel_reason is None and active_setup.bars_watching >= active_setup.max_watch_bars:
                cancel_reason = f"SETUP_EXPIRED:bars_watching={active_setup.bars_watching}>={active_setup.max_watch_bars}"

            if cancel_reason is not None:
                rationale.append(f"Active setup cancelled: {cancel_reason}")
                return LifecycleAssessment(
                    call=AgentCall.CANCEL_SETUP,
                    timing_assessment=base,
                    structure=structure,
                    recommended_trailing_stop=None,
                    updated_setup=None,
                    exit_reason=None,
                    cancel_reason=cancel_reason,
                    rationale=tuple(rationale),
                )

        # Entry Confirmation (with structural trap veto)
        if base.decision == TimingDecision.ENTER_NOW:
            if hypothesis.direction == DirectionalBias.BULLISH:
                if (
                    structure.breakout_event == "FALSE_BREAKOUT"
                    or structure.orb_event == "FAILED_ORB_HIGH"
                    or structure.vwap_event == "VWAP_REJECTION"
                ):
                    rationale.append(
                        f"Bullish entry vetoed by structural trap ({structure.breakout_event}/{structure.orb_event}/{structure.vwap_event})"
                    )
                    return LifecycleAssessment(
                        call=AgentCall.CANCEL_SETUP if active_setup is not None else AgentCall.WAIT,
                        timing_assessment=base,
                        structure=structure,
                        recommended_trailing_stop=None,
                        updated_setup=None,
                        exit_reason=None,
                        cancel_reason="STRUCTURAL_TRAP_VETO" if active_setup is not None else None,
                        rationale=tuple(rationale),
                    )
                return LifecycleAssessment(
                    call=AgentCall.OPEN_LONG,
                    timing_assessment=base,
                    structure=structure,
                    recommended_trailing_stop=base.invalidation_price,
                    updated_setup=None,
                    exit_reason=None,
                    cancel_reason=None,
                    rationale=tuple(rationale),
                )

            if hypothesis.direction == DirectionalBias.BEARISH:
                if (
                    structure.breakout_event == "FAILED_BREAKDOWN"
                    or structure.orb_event == "FAILED_ORB_LOW"
                    or structure.vwap_event == "VWAP_RECLAIM"
                ):
                    rationale.append(
                        f"Bearish entry vetoed by structural trap ({structure.breakout_event}/{structure.orb_event}/{structure.vwap_event})"
                    )
                    return LifecycleAssessment(
                        call=AgentCall.CANCEL_SETUP if active_setup is not None else AgentCall.WAIT,
                        timing_assessment=base,
                        structure=structure,
                        recommended_trailing_stop=None,
                        updated_setup=None,
                        exit_reason=None,
                        cancel_reason="STRUCTURAL_TRAP_VETO" if active_setup is not None else None,
                        rationale=tuple(rationale),
                    )
                return LifecycleAssessment(
                    call=AgentCall.OPEN_SHORT,
                    timing_assessment=base,
                    structure=structure,
                    recommended_trailing_stop=base.invalidation_price,
                    updated_setup=None,
                    exit_reason=None,
                    cancel_reason=None,
                    rationale=tuple(rationale),
                )

        # Contradicted or unconfirmed WAIT -> track or update watched setup
        updated_setup: ActiveSetupContext | None = None
        if base.decision == TimingDecision.WAIT and hypothesis.direction in (
            DirectionalBias.BULLISH,
            DirectionalBias.BEARISH,
        ):
            if active_setup is not None:
                updated_setup = ActiveSetupContext(
                    direction=active_setup.direction,
                    setup_type=active_setup.setup_type,
                    trigger_price=active_setup.trigger_price,
                    invalidation_price=active_setup.invalidation_price,
                    bars_watching=active_setup.bars_watching + 1,
                    max_watch_bars=active_setup.max_watch_bars,
                )
            elif base.invalidation_price is not None:
                setup_dir: Literal["LONG", "SHORT"] = (
                    "LONG" if hypothesis.direction == DirectionalBias.BULLISH else "SHORT"
                )
                trigger_px = round_to_nse_tick(curr_high if setup_dir == "LONG" else curr_low)
                updated_setup = ActiveSetupContext(
                    direction=setup_dir,
                    setup_type=structure.active_scenarios[0] if structure.active_scenarios else "TREND_SETUP",
                    trigger_price=trigger_px,
                    invalidation_price=round_to_nse_tick(base.invalidation_price),
                    bars_watching=1,
                )

        return LifecycleAssessment(
            call=AgentCall.WAIT,
            timing_assessment=base,
            structure=structure,
            recommended_trailing_stop=None,
            updated_setup=updated_setup,
            exit_reason=None,
            cancel_reason=None,
            rationale=tuple(rationale),
        )
