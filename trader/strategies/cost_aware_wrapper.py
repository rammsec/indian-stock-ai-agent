"""Phase 9: Cost-Aware Strategy Filter Wrapper (Versioned Experiment Candidate)."""

from __future__ import annotations

from datetime import time
import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class CostAwareFilterWrapper(BaseStrategy):
    """Wraps a BaseStrategy with a minimum expected-move cost hurdle and VWAP/trend gate.

    Designed for ₹50,000 portfolios where flat ₹20/order brokerage + STT + slippage
    require higher expected move per trade to achieve positive net expectancy.
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        version_tag: str = "v2.0-cost-aware",
        min_expected_move_pct: float = 0.012,
        require_above_vwap: bool = True,
        require_ema_alignment: bool = True,
        entry_start: time = time(9, 30),
        entry_end: time = time(14, 0),
    ) -> None:
        self.base_strategy = base_strategy
        self.version_tag = version_tag
        self.min_expected_move_pct = min_expected_move_pct
        self.require_above_vwap = require_above_vwap
        self.require_ema_alignment = require_ema_alignment
        self.entry_start = entry_start
        self.entry_end = entry_end

        base_meta = base_strategy.metadata
        meta = StrategyMetadata(
            strategy_id=f"{base_meta.strategy_id}-{version_tag.upper()}",
            name=f"{base_meta.name} ({version_tag})",
            origin_source=base_meta.origin_source,
            source_url=base_meta.source_url,
            license_type=base_meta.license_type,
            timeframe=base_meta.timeframe,
            required_indicators=base_meta.required_indicators + ["VWAP", "EMA 9", "EMA 21"],
            known_limitations=base_meta.known_limitations,
            validation_status="IN_VALIDATION",
        )
        super().__init__(meta)

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        sig = self.base_strategy.evaluate_bar(symbol, features_df, current_bar_idx)
        if sig.action != "ENTER_LONG":
            return sig

        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        t = ts.time()
        close = float(row["close"])
        vwap = float(row["vwap"])
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])

        # 1. Time window filter
        if not (self.entry_start <= t <= self.entry_end):
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="NO_SIGNAL",
                confidence=0.0,
                rationale=[f"Filtered by time window ({t.strftime('%H:%M')} outside 09:30-14:00)"],
            )

        # 2. VWAP & EMA trend alignment
        if self.require_above_vwap and close <= vwap:
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="NO_SIGNAL",
                confidence=0.0,
                rationale=[f"Filtered: Close ₹{close:.2f} <= VWAP ₹{vwap:.2f}"],
            )
        if self.require_ema_alignment and ema9 < ema21:
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="NO_SIGNAL",
                confidence=0.0,
                rationale=[f"Filtered: EMA9 ({ema9:.2f}) < EMA21 ({ema21:.2f})"],
            )

        # 3. Minimum expected move vs round-trip cost hurdle
        if sig.target is not None and close > 0:
            expected_move_pct = (sig.target - close) / close
            if expected_move_pct < self.min_expected_move_pct:
                return StrategySignal(
                    strategy_id=self.metadata.strategy_id,
                    symbol=symbol,
                    timestamp=ts,
                    action="NO_SIGNAL",
                    confidence=0.0,
                    rationale=[
                        f"Filtered: Expected move {expected_move_pct*100:.2f}% < cost hurdle {self.min_expected_move_pct*100:.2f}%"
                    ],
                )

        return StrategySignal(
            strategy_id=self.metadata.strategy_id,
            symbol=symbol,
            timestamp=ts,
            action="ENTER_LONG",
            entry_price=sig.entry_price,
            stop_loss=sig.stop_loss,
            target=sig.target,
            confidence=sig.confidence,
            rationale=sig.rationale + [f"Passed cost-aware hurdle (>={self.min_expected_move_pct*100:.1f}%)"],
        )
