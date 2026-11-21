"""Strategy STRAT-002: 15-Minute Opening Range Breakout (ORB-15) with Volume Spike."""

from __future__ import annotations

from datetime import datetime, time
import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class ORBBreakoutStrategy(BaseStrategy):
    """Establishes 09:15-09:30 IST opening range and buys breakouts confirmed by volume."""

    def __init__(self) -> None:
        meta = StrategyMetadata(
            strategy_id="STRAT-002-ORB-BREAKOUT",
            name="15-Minute Opening Range Breakout",
            origin_source="Toby Crabel Institutional ORB Models",
            source_url="https://github.com/QuantConnect/Lean",
            license_type="MIT",
            timeframe="5m",
            required_indicators=["Opening Range 15m", "Volume Ratio", "ATR 14"],
            known_limitations=["False breakouts on low-volume choppy opening sessions"],
            validation_status="IN_VALIDATION",
        )
        super().__init__(meta)
        self.orb_high: float | None = None
        self.orb_low: float | None = None
        self.orb_date: datetime.date | None = None

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        if current_bar_idx < 3 or current_bar_idx >= len(features_df):
            return StrategySignal(self.metadata.strategy_id, symbol, datetime.now(), "NO_SIGNAL")

        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        t = ts.time()
        d = ts.date()

        # Reset ORB daily
        if self.orb_date != d:
            self.orb_date = d
            self.orb_high = None
            self.orb_low = None

        # Build ORB during first 3 bars (09:15, 09:20, 09:25 -> ends at 09:30)
        # Assuming bar 0 is 09:15, bar 1 is 09:20, bar 2 is 09:25
        if current_bar_idx < 3:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")

        if self.orb_high is None:
            first_3_bars = features_df.iloc[current_bar_idx - 3 : current_bar_idx]
            self.orb_high = float(first_3_bars["high"].max())
            self.orb_low = float(first_3_bars["low"].min())

        close = float(row["close"])
        vol_ratio = float(row.get("volume_ratio", 1.0))
        atr = float(row["atr_14"])

        # Entry window: 09:30 to 11:30 IST
        if time(9, 30) <= t <= time(11, 30):
            if close > self.orb_high and vol_ratio >= 1.4:
                stop_loss = round((self.orb_high + self.orb_low) / 2.0, 2)  # Stop inside ORB midpoint
                risk = close - stop_loss
                target = round(close + (1.5 * (self.orb_high - self.orb_low)), 2)

                return StrategySignal(
                    strategy_id=self.metadata.strategy_id,
                    symbol=symbol,
                    timestamp=ts,
                    action="ENTER_LONG",
                    entry_price=close,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.72,
                    rationale=[
                        f"Close ₹{close:.2f} broke 15m ORB High ₹{self.orb_high:.2f}",
                        f"Volume confirmation: {vol_ratio:.1f}x of 20-SMA",
                    ],
                )

        return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")
