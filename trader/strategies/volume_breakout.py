"""Strategy 6: Institutional Volume Breakout Intraday Strategy."""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class VolumeBreakoutStrategy(BaseStrategy):
    """Captures high-volume (>1.8x 20-bar avg) price breakouts above recent 10-bar high."""

    def __init__(self, volume_mult: float = 1.8, atr_sl_mult: float = 1.0, rr_ratio: float = 2.0) -> None:
        self.volume_mult = volume_mult
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id="STRAT-006-VOLUME-BREAKOUT",
            name="Volume Breakout",
            origin_source="Volume Spread Analysis",
            source_url="internal://volume-breakout-5m",
            license_type="MIT",
            required_indicators=["vwap", "volume_ratio", "atr_14"],
            validation_status="APPROVED_PAPER",
        )

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        ts = features_df.iloc[current_bar_idx]["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if current_bar_idx < 20 or current_bar_idx >= len(features_df):
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="NO_SIGNAL",
                confidence=0.0,
                rationale=["Warming up volume baseline"],
            )

        curr = features_df.iloc[current_bar_idx]
        window = features_df.iloc[current_bar_idx - 10 : current_bar_idx]
        close = float(curr["close"])
        vwap = float(curr["vwap"])
        vol_ratio = float(curr.get("volume_ratio", 1.0))
        atr = max(float(curr["atr_14"]), close * 0.002)
        recent_high = float(window["high"].max())

        if close > recent_high and vol_ratio >= self.volume_mult and close > vwap:
            sl = round(close - (self.atr_sl_mult * atr), 2)
            tp = round(close + (self.atr_sl_mult * self.rr_ratio * atr), 2)
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="ENTER_LONG",
                entry_price=close,
                stop_loss=sl,
                target=tp,
                confidence=0.78,
                rationale=[f"10-bar high breakout (₹{close:.2f} > ₹{recent_high:.2f}) on {vol_ratio:.1f}x volume"],
            )

        if close < vwap and vol_ratio >= 1.5:
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="EXIT",
                confidence=0.68,
                rationale=[f"High-volume breakdown below VWAP (₹{close:.2f} < ₹{vwap:.2f})"],
            )

        return StrategySignal(
            strategy_id=self.metadata.strategy_id,
            symbol=symbol,
            timestamp=ts,
            action="HOLD",
            confidence=0.5,
            rationale=["No volume breakout"],
        )
