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
from typing import Any

import pandas as pd

from trader.config import IST_TZ
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
