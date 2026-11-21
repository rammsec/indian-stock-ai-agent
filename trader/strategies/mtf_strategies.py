"""CYCLE-002: Lower-Frequency 15m & 30m Multi-Timeframe Candidate Strategies."""

from __future__ import annotations

from datetime import time
import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class MTF15mRelativeStrengthLeaderStrategy(BaseStrategy):
    """Strategy 1: 15m Relative Strength Leader (VWAP + EMA9>21 + Strong Intraday Gain + Vol)."""

    def __init__(self, min_session_gain_pct: float = 0.6, atr_sl_mult: float = 1.2, rr_ratio: float = 2.0) -> None:
        self.min_session_gain_pct = min_session_gain_pct
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-008-15M-RS-LEADER",
                name="15m Relative Strength Leader",
                origin_source="CYCLE-002 Multi-Timeframe Research",
                source_url="internal://15m-rs-leader",
                license_type="MIT",
                timeframe="15m",
                required_indicators=["vwap", "ema_9", "ema_21", "atr_14", "volume_ratio"],
                validation_status="IN_VALIDATION",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        if current_bar_idx < 10:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")

        close = float(row["close"])
        vwap = float(row["vwap"])
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        atr = max(float(row["atr_14"]), close * 0.003)
        vol_ratio = float(row.get("volume_ratio", 1.0))

        # Session open price (lookback only within same day)
        curr_date = ts.date()
        day_rows = features_df.iloc[: current_bar_idx + 1]
        same_day = day_rows[day_rows["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date == curr_date] if hasattr(day_rows["timestamp"].dt, "tz_convert") else day_rows
        day_open = float(same_day.iloc[0]["open"]) if not same_day.empty else close
        gain_pct = ((close - day_open) / day_open) * 100.0 if day_open > 0 else 0.0

        if time(9, 45) <= ts.time() <= time(13, 30):
            if gain_pct >= self.min_session_gain_pct and close > vwap and ema9 > ema21 and vol_ratio >= 1.1:
                sl = round(close - self.atr_sl_mult * atr, 2)
                tp = round(close + self.atr_sl_mult * self.rr_ratio * atr, 2)
                return StrategySignal(
                    self.metadata.strategy_id, symbol, ts, "ENTER_LONG", close, sl, tp, 0.76,
                    [f"15m RS leader (+{gain_pct:.2f}% session), Close>VWAP, EMA9>21, Vol={vol_ratio:.1f}x"],
                )

        if close < ema21:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", close, confidence=0.7, rationale=["Lost 15m EMA21 trend"])
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", close, confidence=0.5)


class MTF15mVWAPTrendContinuationStrategy(BaseStrategy):
    """Strategy 2: 15m VWAP Trend Continuation with cost-hurdle target."""

    def __init__(self, atr_sl_mult: float = 1.2, rr_ratio: float = 2.0) -> None:
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-009-15M-VWAP-TREND",
                name="15m VWAP Trend Continuation",
                origin_source="CYCLE-002 Multi-Timeframe Research",
                source_url="internal://15m-vwap-trend",
                license_type="MIT",
                timeframe="15m",
                required_indicators=["vwap", "ema_9", "ema_21", "atr_14"],
                validation_status="IN_VALIDATION",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        if current_bar_idx < 12:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")

        prev = features_df.iloc[current_bar_idx - 1]
        close = float(row["close"])
        low = float(row["low"])
        vwap = float(row["vwap"])
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        atr = max(float(row["atr_14"]), close * 0.003)

        # Higher high + higher low pullback bounce off VWAP/EMA9
        hh_hl = close > float(prev["high"]) and low >= float(prev["low"]) * 0.999
        if time(9, 45) <= ts.time() <= time(13, 45) and close > vwap and ema9 > ema21 and hh_hl:
            sl = round(close - self.atr_sl_mult * atr, 2)
            tp = round(close + self.atr_sl_mult * self.rr_ratio * atr, 2)
            if (tp - close) / close >= 0.008:
                return StrategySignal(
                    self.metadata.strategy_id, symbol, ts, "ENTER_LONG", close, sl, tp, 0.74,
                    ["15m higher-high continuation above VWAP & EMA9>21"],
                )

        if close < vwap:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", close, confidence=0.7, rationale=["15m close below VWAP"])
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", close, confidence=0.5)


class MTF30mEMATrendStrategy(BaseStrategy):
    """Strategy 3: 30m EMA 20/50 Trend Continuation."""

    def __init__(self, atr_sl_mult: float = 1.5, rr_ratio: float = 2.0) -> None:
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-010-30M-EMA-TREND",
                name="30m EMA Trend Continuation",
                origin_source="CYCLE-002 Multi-Timeframe Research",
                source_url="internal://30m-ema-trend",
                license_type="MIT",
                timeframe="30m",
                required_indicators=["ema_21", "ema_50", "rsi_14", "atr_14"],
                validation_status="IN_VALIDATION",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        if current_bar_idx < 10:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")

        prev = features_df.iloc[current_bar_idx - 1]
        close = float(row["close"])
        e21 = float(row["ema_21"])
        e50 = float(row["ema_50"])
        prev_e21 = float(prev["ema_21"])
        prev_e50 = float(prev["ema_50"])
        rsi = float(row["rsi_14"])
        atr = max(float(row["atr_14"]), close * 0.004)

        if (prev_e21 <= prev_e50 and e21 > e50) or (close > e21 > e50 and 55.0 <= rsi <= 70.0 and float(prev["close"]) <= float(prev["ema_21"])):
            sl = round(close - self.atr_sl_mult * atr, 2)
            tp = round(close + self.atr_sl_mult * self.rr_ratio * atr, 2)
            return StrategySignal(
                self.metadata.strategy_id, symbol, ts, "ENTER_LONG", close, sl, tp, 0.75,
                [f"30m EMA21>EMA50 trend alignment (RSI={rsi:.1f})"],
            )

        if e21 < e50:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", close, confidence=0.7, rationale=["30m EMA21 crossed below EMA50"])
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", close, confidence=0.5)


class MTF30mBreakoutVolumeStrategy(BaseStrategy):
    """Strategy 4: 30m Breakout with Volume Confirmation."""

    def __init__(self, vol_mult: float = 1.4, atr_sl_mult: float = 1.2, rr_ratio: float = 2.0) -> None:
        self.vol_mult = vol_mult
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-011-30M-BREAKOUT-VOL",
                name="30m Breakout w/ Volume",
                origin_source="CYCLE-002 Multi-Timeframe Research",
                source_url="internal://30m-breakout-vol",
                license_type="MIT",
                timeframe="30m",
                required_indicators=["vwap", "volume_ratio", "atr_14"],
                validation_status="IN_VALIDATION",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        if current_bar_idx < 8:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")

        window = features_df.iloc[current_bar_idx - 6 : current_bar_idx]
        prior_high = float(window["high"].max())
        close = float(row["close"])
        vwap = float(row["vwap"])
        vol_ratio = float(row.get("volume_ratio", 1.0))
        atr = max(float(row["atr_14"]), close * 0.004)

        if time(9, 45) <= ts.time() <= time(13, 30) and close > prior_high and close > vwap and vol_ratio >= self.vol_mult:
            sl = round(close - self.atr_sl_mult * atr, 2)
            tp = round(close + self.atr_sl_mult * self.rr_ratio * atr, 2)
            return StrategySignal(
                self.metadata.strategy_id, symbol, ts, "ENTER_LONG", close, sl, tp, 0.77,
                [f"30m breakout ₹{close:.2f} > 6-bar high ₹{prior_high:.2f} on {vol_ratio:.1f}x vol"],
            )

        if close < vwap:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", close, confidence=0.68, rationale=["30m close fell below VWAP"])
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", close, confidence=0.5)


class LowFrequencyTrendFollowingStrategy(BaseStrategy):
    """Strategy 5: Lower-Frequency 30m Trend Persistence with ATR Trailing Stop."""

    def __init__(self, atr_sl_mult: float = 1.8, rr_ratio: float = 2.5) -> None:
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-012-LOW-FREQ-TREND",
                name="Lower-Frequency 30m Trend",
                origin_source="CYCLE-002 Multi-Timeframe Research",
                source_url="internal://low-freq-trend",
                license_type="MIT",
                timeframe="30m",
                required_indicators=["ema_9", "ema_21", "ema_50", "macd", "macd_signal", "atr_14"],
                validation_status="IN_VALIDATION",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        if current_bar_idx < 12:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")

        close = float(row["close"])
        e9 = float(row["ema_9"])
        e21 = float(row["ema_21"])
        e50 = float(row["ema_50"])
        macd = float(row["macd"])
        sig = float(row["macd_signal"])
        atr = max(float(row["atr_14"]), close * 0.004)

        if time(9, 45) <= ts.time() <= time(12, 30) and e9 > e21 > e50 and macd > sig:
            sl = round(close - self.atr_sl_mult * atr, 2)
            tp = round(close + self.atr_sl_mult * self.rr_ratio * atr, 2)
            return StrategySignal(
                self.metadata.strategy_id, symbol, ts, "ENTER_LONG", close, sl, tp, 0.78,
                ["30m full EMA9>21>50 + MACD trend persistence"],
            )

        if e9 < e21 and macd < sig:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", close, confidence=0.72, rationale=["30m trend momentum reversed"])
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", close, confidence=0.5)


class CashNoTradeBenchmarkStrategy(BaseStrategy):
    """Strategy 6: Cash / No-Trade Benchmark (capital preservation baseline)."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="BENCHMARK-CASH-NO-TRADE",
                name="Cash / No-Trade Benchmark",
                origin_source="Capital Preservation Baseline",
                source_url="internal://cash-benchmark",
                license_type="MIT",
                timeframe="15m",
                required_indicators=[],
                validation_status="APPROVED_PAPER",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL", confidence=1.0, rationale=["Cash preservation"])


class MTF30mConcentratedBreakoutStrategy(BaseStrategy):
    """CYCLE-003 Strategy: Concentrated 30m Breakout + Volume + EMA alignment with 2.2x ATR target."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-013-30M-CONC-BREAKOUT",
                name="30m Concentrated Breakout Volume",
                origin_source="BharatAlpha CYCLE-003 Research",
                source_url="internal://cycle-003",
                license_type="MIT",
                timeframe="30m",
                required_indicators=["ema_9", "ema_21", "vwap", "rsi_14", "atr_14", "volume"],
                validation_status="EXPERIMENTAL",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        if current_bar_idx < 10:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5, rationale=["Warmup"])

        hhmm = ts.strftime("%H:%M")
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        vwap = float(row["vwap"])
        rsi = float(row["rsi_14"])
        atr = max(float(row["atr_14"]), close * 0.0025)
        vol = float(row["volume"])
        avg_vol = float(features_df.iloc[max(0, current_bar_idx - 10):current_bar_idx]["volume"].mean())
        prev_high = float(features_df.iloc[max(0, current_bar_idx - 4):current_bar_idx]["high"].max())

        if ema9 < ema21 or close < vwap * 0.997:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", confidence=0.75, rationale=["30m trend/VWAP breakdown"])

        if "09:45" <= hhmm <= "13:45" and close > prev_high and close > vwap and ema9 > ema21 and 52.0 <= rsi <= 72.0 and vol >= 1.5 * max(1.0, avg_vol):
            sl = round(close - 1.2 * atr, 2)
            tp = round(close + 2.4 * atr, 2)
            if (tp - close) / close >= 0.012:
                return StrategySignal(
                    self.metadata.strategy_id, symbol, ts, "BUY", confidence=0.82,
                    stop_loss=sl, target=tp, rationale=["30m concentrated high-RR breakout + 1.5x vol"]
                )
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5)


class MTF30mVWAPSqueezeStrategy(BaseStrategy):
    """CYCLE-003 Strategy: 30m Volatility Contraction + VWAP Momentum Breakout."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-014-30M-VWAP-SQUEEZE",
                name="30m VWAP Volatility Squeeze",
                origin_source="BharatAlpha CYCLE-003 Research",
                source_url="internal://cycle-003",
                license_type="MIT",
                timeframe="30m",
                required_indicators=["ema_9", "ema_21", "vwap", "rsi_14", "atr_14", "volume"],
                validation_status="EXPERIMENTAL",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        if current_bar_idx < 12:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5, rationale=["Warmup"])

        hhmm = ts.strftime("%H:%M")
        vwap = float(row["vwap"])
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        rsi = float(row["rsi_14"])
        atr = max(float(row["atr_14"]), close * 0.0025)
        recent_ranges = (features_df.iloc[current_bar_idx - 4:current_bar_idx]["high"] - features_df.iloc[current_bar_idx - 4:current_bar_idx]["low"]).mean()
        curr_range = float(row["high"]) - float(row["low"])
        vol = float(row["volume"])
        avg_vol = float(features_df.iloc[current_bar_idx - 10:current_bar_idx]["volume"].mean())

        if close < vwap * 0.996:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", confidence=0.72, rationale=["Lost VWAP support"])

        if "10:15" <= hhmm <= "13:45" and curr_range > 1.25 * max(1e-4, float(recent_ranges)) and close > vwap and ema9 > ema21 and 54.0 <= rsi <= 72.0 and vol >= 1.4 * max(1.0, avg_vol):
            sl = round(close - 1.1 * atr, 2)
            tp = round(close + 2.2 * atr, 2)
            return StrategySignal(
                self.metadata.strategy_id, symbol, ts, "BUY", confidence=0.78,
                stop_loss=sl, target=tp, rationale=["30m range expansion above VWAP"]
            )
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5)


class MTF60mTrendBreakoutStrategy(BaseStrategy):
    """CYCLE-003 Strategy: 60m (1H) Trend-Aligned 4-Bar High Breakout (240-session research)."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-015-60M-TREND-BREAKOUT",
                name="60m Trend Breakout",
                origin_source="BharatAlpha CYCLE-003 1H Research",
                source_url="internal://cycle-003",
                license_type="MIT",
                timeframe="60m",
                required_indicators=["ema_9", "ema_21", "vwap", "rsi_14", "atr_14", "volume"],
                validation_status="EXPERIMENTAL",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        if current_bar_idx < 12:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5, rationale=["Warmup"])

        hhmm = ts.strftime("%H:%M")
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        rsi = float(row["rsi_14"])
        atr = max(float(row["atr_14"]), close * 0.004)
        vol = float(row["volume"])
        avg_vol = float(features_df.iloc[current_bar_idx - 10:current_bar_idx]["volume"].mean())
        prev_4h_high = float(features_df.iloc[current_bar_idx - 4:current_bar_idx]["high"].max())

        if ema9 < ema21:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", confidence=0.75, rationale=["60m EMA9 < EMA21"])

        if "09:15" <= hhmm <= "13:15" and close > prev_4h_high and ema9 > ema21 and 50.0 <= rsi <= 72.0 and vol >= 1.25 * max(1.0, avg_vol):
            sl = round(close - 1.2 * atr, 2)
            tp = round(close + 2.4 * atr, 2)
            return StrategySignal(
                self.metadata.strategy_id, symbol, ts, "BUY", confidence=0.80,
                stop_loss=sl, target=tp, rationale=["60m 4-bar high breakout with EMA & volume confirmation"]
            )
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5)


class MTF60mVWAPMomentumStrategy(BaseStrategy):
    """CYCLE-003 Strategy: 60m (1H) VWAP + RSI Momentum Continuation."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-016-60M-VWAP-MOMENTUM",
                name="60m VWAP Momentum",
                origin_source="BharatAlpha CYCLE-003 1H Research",
                source_url="internal://cycle-003",
                license_type="MIT",
                timeframe="60m",
                required_indicators=["ema_9", "ema_21", "vwap", "rsi_14", "atr_14"],
                validation_status="EXPERIMENTAL",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        if current_bar_idx < 10:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5, rationale=["Warmup"])

        hhmm = ts.strftime("%H:%M")
        vwap = float(row["vwap"])
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        rsi = float(row["rsi_14"])
        atr = max(float(row["atr_14"]), close * 0.004)

        if close < vwap * 0.996:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", confidence=0.72, rationale=["60m close below VWAP"])

        if "10:15" <= hhmm <= "13:15" and close > vwap and ema9 > ema21 and 52.0 <= rsi <= 68.0:
            sl = round(close - 1.1 * atr, 2)
            tp = round(close + 2.2 * atr, 2)
            return StrategySignal(
                self.metadata.strategy_id, symbol, ts, "BUY", confidence=0.76,
                stop_loss=sl, target=tp, rationale=["60m VWAP + EMA9>EMA21 momentum"]
            )
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5)


class MTF60mORBFilteredStrategy(BaseStrategy):
    """CYCLE-003 Strategy: 60m (1H) Opening-Hour Range Breakout with Trend Filter."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-017-60M-ORB-FILTERED",
                name="60m Opening Range Breakout",
                origin_source="BharatAlpha CYCLE-003 1H Research",
                source_url="internal://cycle-003",
                license_type="MIT",
                timeframe="60m",
                required_indicators=["ema_9", "ema_21", "atr_14", "volume"],
                validation_status="EXPERIMENTAL",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        if current_bar_idx < 10:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5, rationale=["Warmup"])

        hhmm = ts.strftime("%H:%M")
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        atr = max(float(row["atr_14"]), close * 0.004)
        vol = float(row["volume"])
        avg_vol = float(features_df.iloc[current_bar_idx - 10:current_bar_idx]["volume"].mean())

        if ema9 < ema21:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", confidence=0.72, rationale=["60m trend reversal"])

        # Compare against today's 09:15 bar high
        today = ts.date()
        same_day = features_df.iloc[:current_bar_idx]
        same_day_bars = same_day[same_day["timestamp"].dt.date == today]
        if hhmm in ("10:15", "11:15", "12:15") and not same_day_bars.empty:
            orb_high = float(same_day_bars.iloc[0]["high"])
            if close > orb_high and ema9 > ema21 and vol >= 1.15 * max(1.0, avg_vol):
                sl = round(close - 1.2 * atr, 2)
                tp = round(close + 2.4 * atr, 2)
                return StrategySignal(
                    self.metadata.strategy_id, symbol, ts, "BUY", confidence=0.79,
                    stop_loss=sl, target=tp, rationale=["60m 1st-hour ORB breakout + EMA trend"]
                )
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5)


class MTF60mORBSelectiveStrategy(BaseStrategy):
    """CYCLE-004 Strategy: Ultra-Selective 60m ORB with 1.5x Volume, RSI 55-68 & 1.5% Move Hurdle."""

    def __init__(self) -> None:
        super().__init__(
            StrategyMetadata(
                strategy_id="STRAT-018-60M-ORB-SELECTIVE",
                name="60m Ultra-Selective ORB",
                origin_source="BharatAlpha CYCLE-004 Research",
                source_url="internal://cycle-004",
                license_type="MIT",
                timeframe="60m",
                required_indicators=["ema_9", "ema_21", "rsi_14", "atr_14", "volume"],
                validation_status="EXPERIMENTAL",
            )
        )

    def evaluate_bar(self, symbol: str, features_df: pd.DataFrame, current_bar_idx: int) -> StrategySignal:
        row = features_df.iloc[current_bar_idx]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        close = float(row["close"])
        if current_bar_idx < 12:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5, rationale=["Warmup"])

        hhmm = ts.strftime("%H:%M")
        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        rsi = float(row["rsi_14"])
        atr = max(float(row["atr_14"]), close * 0.004)
        vol = float(row["volume"])
        avg_vol = float(features_df.iloc[current_bar_idx - 10:current_bar_idx]["volume"].mean())

        if ema9 < ema21:
            return StrategySignal(self.metadata.strategy_id, symbol, ts, "EXIT", confidence=0.75, rationale=["60m trend reversal"])

        today = ts.date()
        same_day = features_df.iloc[:current_bar_idx]
        same_day_bars = same_day[same_day["timestamp"].dt.date == today]
        if hhmm in ("10:15", "11:15") and not same_day_bars.empty:
            orb_high = float(same_day_bars.iloc[0]["high"])
            if close > orb_high and ema9 > ema21 and 55.0 <= rsi <= 68.0 and vol >= 1.5 * max(1.0, avg_vol):
                sl = round(close - 1.2 * atr, 2)
                tp = round(close + 3.0 * atr, 2)
                if (tp - close) / close >= 0.015:
                    return StrategySignal(
                        self.metadata.strategy_id, symbol, ts, "BUY", confidence=0.84,
                        stop_loss=sl, target=tp, rationale=["60m ultra-selective ORB + 1.5x vol + 1.5% hurdle"]
                    )
        return StrategySignal(self.metadata.strategy_id, symbol, ts, "HOLD", confidence=0.5)
