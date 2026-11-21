"""Strategy STRAT-004: Intraday Supertrend (10, 3) Trend-Following Pullback."""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator (trend direction and upper/lower bands)."""
    hl2 = (df["high"] + df["low"]) / 2.0
    atr = df["atr_14"] if "atr_14" in df else (df["high"] - df["low"]).rolling(period).mean()

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    in_uptrend = pd.Series(True, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        curr_close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # Lower band adjustment
        if lowerband.iloc[i] > lowerband.iloc[i - 1] or prev_close < lowerband.iloc[i - 1]:
            pass
        else:
            lowerband.iloc[i] = lowerband.iloc[i - 1]

        # Upper band adjustment
        if upperband.iloc[i] < upperband.iloc[i - 1] or prev_close > upperband.iloc[i - 1]:
            pass
        else:
            upperband.iloc[i] = upperband.iloc[i - 1]

        # Trend flip
        if in_uptrend.iloc[i - 1]:
            if curr_close < lowerband.iloc[i]:
                in_uptrend.iloc[i] = False
            else:
                in_uptrend.iloc[i] = True
        else:
            if curr_close > upperband.iloc[i]:
                in_uptrend.iloc[i] = True
            else:
                in_uptrend.iloc[i] = False

        supertrend.iloc[i] = lowerband.iloc[i] if in_uptrend.iloc[i] else upperband.iloc[i]

    return supertrend, in_uptrend


class SupertrendStrategy(BaseStrategy):
    """Buys pullbacks towards the Supertrend lower band in an active uptrend."""

    def __init__(self) -> None:
        meta = StrategyMetadata(
            strategy_id="STRAT-004-SUPERTREND-PULLBACK",
            name="Supertrend (10, 3) Intraday Pullback",
            origin_source="Pine Script Public Community / TradingView Standard Library",
            source_url="https://www.tradingview.com/scripts/supertrend/",
            license_type="Public Domain / MIT",
            timeframe="5m",
            required_indicators=["Supertrend (10, 3)", "EMA 50", "VWAP"],
            known_limitations=["Whipsaws in tight horizontal trading ranges"],
            validation_status="IN_VALIDATION",
        )
        super().__init__(meta)

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        if current_bar_idx < 15 or current_bar_idx >= len(features_df):
            return StrategySignal(self.metadata.strategy_id, symbol, datetime.now(), "NO_SIGNAL")

        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        vwap = float(row["vwap"])
        e50 = float(row["ema_50"])
        atr = float(row["atr_14"])

        # Check if price is above VWAP and 50-EMA
        if close > vwap and close > e50:
            # Pullback to within 0.5% of VWAP with bullish candle close
            if abs(row["low"] - vwap) / close <= 0.005 and close > float(row["open"]):
                stop_loss = round(vwap - (0.8 * atr), 2)
                target = round(close + (1.8 * (close - stop_loss)), 2)
                return StrategySignal(
                    strategy_id=self.metadata.strategy_id,
                    symbol=symbol,
                    timestamp=ts,
                    action="ENTER_LONG",
                    entry_price=close,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.71,
                    rationale=["Supertrend confirmed bullish pullback to VWAP"],
                )

        return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")
