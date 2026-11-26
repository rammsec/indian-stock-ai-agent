"""CYCLE-020: Full Multi-Month Walk-Forward 1-Minute Profit Optimizer (Zero-Lookahead).

Provides:
1. `precompute_causal_1m_arrays`: Strictly left-to-right causal NumPy indicator & overlay
   computation (EMA9/21, ATR14, RSI14, ADX14/+DI/-DI, Supertrend(10,3), Session VWAP ±1σ/±2σ,
   ORB 15m, PDH/PDL, Causal 5m MTF trend, Liquidity Sweeps, FVG, AUTO 4-vote live bias).
2. `simulate_fast_1m_session`: Bar-by-bar paper simulator implementing CYCLE-012..019 rules
   (3x NSE friction gate, 5-bar cooldown, 4/day cap, 2.5R TP, Chandelier ATR trailing stop,
   +1.2R fee-aware breakeven lock, EOD 15:15 IST square-off, exact NSE statutory taxes via
   `calculate_intraday_costs`, and optional CYCLE-022 ML probability filter).
3. `run_walkforward_optimization`: Strictly chronological Train / unseen OOS Test split
   (`max(train_dates) < min(test_dates)`) across multiple NSE symbols, strategy profiles,
   and capital tiers (e.g. ₹50,000 vs ₹2,00,000).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from trader.config import IST_TZ
from trader.data.loader_1m import (
    DEFAULT_1M_DIR,
    DEFAULT_SYMBOLS_1M,
    OneMinuteSessionBundle,
    load_1m_sessions,
)
from trader.portfolio.cost_calculator import calculate_intraday_costs
from trader.intelligence.chart_overlays import StrategyProfileLiteral

# Strictly paper-only evaluation module
LIVE_TRADING_ENABLED: bool = False

ENTRY_START_IST = time(9, 20)
ENTRY_END_IST = time(14, 45)
EOD_SQUAREOFF_IST = time(15, 15)
ORB_END_IST = time(9, 30)

ALL_STRATEGY_PROFILES: tuple[StrategyProfileLiteral, ...] = (
    "BALANCED_ALL",
    "ORB_BREAKOUT",
    "SUPERTREND_ADX",
    "LIQUIDITY_REVERSAL",
)


@dataclass(frozen=True)
class FastSessionConfig:
    """Configuration for a single fast causal 1m paper session."""

    strategy_profile: StrategyProfileLiteral = "BALANCED_ALL"
    initial_capital_inr: float = 50_000.0
    risk_pct: float = 0.01
    rr_target: float = 2.5
    adx_min: float = 20.0
    cooldown_bars: int = 5
    max_entries_per_day: int = 4
    min_reward_to_friction: float = 3.0
    chandelier_atr_mult: float = 2.0
    breakeven_r_trigger: float = 1.2
    breakeven_fee_buffer_pct: float = 0.0012
    auto_direction: bool = True
    fixed_direction: str = "AUTO"  # "AUTO" | "BULLISH" | "BEARISH" | "NEUTRAL"
    min_ml_win_prob: float | None = None


@dataclass(frozen=True)
class FastTradeRecord:
    """Single closed paper trade with causal entry feature snapshot."""

    symbol: str
    trading_date: str
    side: str  # "BUY" (long) or "SELL" (short)
    entry_time_ist: str
    exit_time_ist: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl_inr: float
    friction_inr: float
    net_pnl_inr: float
    exit_reason: str
    r_multiple: float
    strategy_profile: str
    entry_features: dict[str, float] = field(default_factory=dict)
    ml_win_prob: float | None = None


@dataclass(frozen=True)
class FastSessionSummary:
    """Aggregate results of one fast causal 1m session."""

    symbol: str
    trading_date: str
    strategy_profile: str
    initial_capital_inr: float
    bars_processed: int
    trades: tuple[FastTradeRecord, ...]
    total_gross_pnl_inr: float
    total_friction_inr: float
    total_net_pnl_inr: float
    wins: int
    losses: int
    max_drawdown_inr: float
    blocked_by_ml: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trading_date": self.trading_date,
            "strategy_profile": self.strategy_profile,
            "initial_capital_inr": round(self.initial_capital_inr, 2),
            "bars_processed": self.bars_processed,
            "trade_count": len(self.trades),
            "wins": self.wins,
            "losses": self.losses,
            "total_gross_pnl_inr": round(self.total_gross_pnl_inr, 2),
            "total_friction_inr": round(self.total_friction_inr, 2),
            "total_net_pnl_inr": round(self.total_net_pnl_inr, 2),
            "max_drawdown_inr": round(self.max_drawdown_inr, 2),
            "blocked_by_ml": self.blocked_by_ml,
        }


def _compute_ema_array(values: np.ndarray, period: int) -> np.ndarray:
    """Strictly causal EMA computed left-to-right."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _compute_rsi_array(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Strictly causal Wilder RSI(period)."""
    n = len(closes)
    out = np.full(n, 50.0, dtype=np.float64)
    if n <= period:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    if avg_loss == 0.0:
        out[period] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss <= 1e-12:
            out[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _compute_adx_atr_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strictly causal Wilder ATR(14), ADX(14), +DI(14), -DI(14)."""
    n = len(closes)
    atr = np.full(n, np.nan, dtype=np.float64)
    adx = np.full(n, np.nan, dtype=np.float64)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return atr, adx, plus_di, minus_di

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0.0) else 0.0
        minus_dm[i] = down if (down > up and down > 0.0) else 0.0

    if n <= period:
        return atr, adx, plus_di, minus_di

    tr_s = float(np.sum(tr[1 : period + 1]))
    pdm_s = float(np.sum(plus_dm[1 : period + 1]))
    mdm_s = float(np.sum(minus_dm[1 : period + 1]))
    atr[period] = tr_s / period
    dx_vals = np.full(n, np.nan, dtype=np.float64)
    if tr_s > 1e-12:
        pdi = 100.0 * pdm_s / tr_s
        mdi = 100.0 * mdm_s / tr_s
        plus_di[period] = pdi
        minus_di[period] = mdi
        denom = pdi + mdi
        dx_vals[period] = 100.0 * abs(pdi - mdi) / denom if denom > 1e-12 else 0.0

    for i in range(period + 1, n):
        tr_s = tr_s - (tr_s / period) + tr[i]
        pdm_s = pdm_s - (pdm_s / period) + plus_dm[i]
        mdm_s = mdm_s - (mdm_s / period) + minus_dm[i]
        atr[i] = tr_s / period
        if tr_s > 1e-12:
            pdi = 100.0 * pdm_s / tr_s
            mdi = 100.0 * mdm_s / tr_s
            plus_di[i] = pdi
            minus_di[i] = mdi
            denom = pdi + mdi
            dx_vals[i] = 100.0 * abs(pdi - mdi) / denom if denom > 1e-12 else 0.0

    first_adx_idx = 2 * period - 1
    if first_adx_idx < n:
        seed_slice = dx_vals[period : first_adx_idx + 1]
        valid_seed = seed_slice[~np.isnan(seed_slice)]
        if len(valid_seed) > 0:
            adx[first_adx_idx] = float(np.mean(valid_seed))
            for i in range(first_adx_idx + 1, n):
                if not np.isnan(dx_vals[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx_vals[i]) / period
                else:
                    adx[i] = adx[i - 1]

    return atr, adx, plus_di, minus_di


def _compute_supertrend_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly causal Supertrend(10, 3.0). Direction: +1=BULLISH, -1=BEARISH, 0=NEUTRAL."""
    n = len(closes)
    st_val = np.full(n, np.nan, dtype=np.float64)
    st_dir = np.zeros(n, dtype=np.int8)
    if n <= period:
        return st_val, st_dir

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    atr = np.full(n, np.nan, dtype=np.float64)
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hl2 = 0.5 * (highs[i] + lows[i])
        basic_upper = hl2 + multiplier * atr[i]
        basic_lower = hl2 - multiplier * atr[i]
        if i == period - 1:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            if closes[i] >= hl2:
                st_dir[i] = 1
                st_val[i] = final_lower[i]
            else:
                st_dir[i] = -1
                st_val[i] = final_upper[i]
            continue

        prev_u = final_upper[i - 1]
        prev_l = final_lower[i - 1]
        final_upper[i] = basic_upper if (basic_upper < prev_u or closes[i - 1] > prev_u) else prev_u
        final_lower[i] = basic_lower if (basic_lower > prev_l or closes[i - 1] < prev_l) else prev_l

        prev_dir = st_dir[i - 1]
        if prev_dir == 1:
            if closes[i] < final_lower[i]:
                st_dir[i] = -1
                st_val[i] = final_upper[i]
            else:
                st_dir[i] = 1
                st_val[i] = final_lower[i]
        else:
            if closes[i] > final_upper[i]:
                st_dir[i] = 1
                st_val[i] = final_lower[i]
            else:
                st_dir[i] = -1
                st_val[i] = final_upper[i]

    return st_val, st_dir


def precompute_causal_1m_arrays(bundle: OneMinuteSessionBundle) -> dict[str, Any]:
    """Compute all strictly causal indicator & overlay arrays for one trading day.

    Warmup bars precede session bars; every value at index `k` uses ONLY data <= `k`.
    """
    all_bars = list(bundle.warmup_bars) + list(bundle.session_bars)
    n_warmup = len(bundle.warmup_bars)
    n_total = len(all_bars)
    n_sess = len(bundle.session_bars)

    opens = np.array([b.open for b in all_bars], dtype=np.float64)
    highs = np.array([b.high for b in all_bars], dtype=np.float64)
    lows = np.array([b.low for b in all_bars], dtype=np.float64)
    closes = np.array([b.close for b in all_bars], dtype=np.float64)
    volumes = np.array([b.volume for b in all_bars], dtype=np.float64)
    timestamps = [b.timestamp.astimezone(IST_TZ) for b in all_bars]

    ema9 = _compute_ema_array(closes, 9)
    ema21 = _compute_ema_array(closes, 21)
    rsi14 = _compute_rsi_array(closes, 14)
    atr14, adx14, plus_di14, minus_di14 = _compute_adx_atr_arrays(highs, lows, closes, 14)
    st_val, st_dir = _compute_supertrend_arrays(highs, lows, closes, 10, 3.0)

    # Previous Day High / Low (strictly from warmup bars prior to bundle.trading_date)
    prior_highs = [highs[i] for i in range(n_warmup) if timestamps[i].date() < bundle.trading_date]
    prior_lows = [lows[i] for i in range(n_warmup) if timestamps[i].date() < bundle.trading_date]
    pdh = float(max(prior_highs)) if prior_highs else np.nan
    pdl = float(min(prior_lows)) if prior_lows else np.nan

    # Session-only causal VWAP ± 1σ/2σ, ORB 15m, and causal 5m MTF trend
    vwap = np.full(n_sess, np.nan, dtype=np.float64)
    vwap_sd = np.full(n_sess, np.nan, dtype=np.float64)
    orb_high = np.full(n_sess, np.nan, dtype=np.float64)
    orb_low = np.full(n_sess, np.nan, dtype=np.float64)
    orb_ready = np.zeros(n_sess, dtype=bool)
    mtf_5m_dir = np.zeros(n_sess, dtype=np.int8)  # +1=BULLISH, -1=BEARISH, 0=NEUTRAL
    sweep_signal = np.zeros(n_sess, dtype=np.int8)  # +1=BULLISH_SWEEP, -1=BEARISH_SWEEP
    fvg_signal = np.zeros(n_sess, dtype=np.int8)  # +1=BULLISH_FVG, -1=BEARISH_FVG

    # Build causal 5m closed buckets across warmup + session
    five_m_closes: list[float] = []
    bucket_close: float | None = None
    bucket_key: tuple[date, int, int] | None = None
    mtf_all = np.zeros(n_total, dtype=np.int8)
    last_mtf: int = 0

    for i in range(n_total):
        ts = timestamps[i]
        b_min = (ts.minute // 5) * 5
        key = (ts.date(), ts.hour, b_min)
        if bucket_key is not None and key != bucket_key and bucket_close is not None:
            five_m_closes.append(bucket_close)
            if len(five_m_closes) >= 21:
                arr5 = np.array(five_m_closes, dtype=np.float64)
                e9_5 = _compute_ema_array(arr5, 9)[-1]
                e21_5 = _compute_ema_array(arr5, 21)[-1]
                if e9_5 > e21_5:
                    last_mtf = 1
                elif e9_5 < e21_5:
                    last_mtf = -1
                else:
                    last_mtf = 0
        bucket_key = key
        bucket_close = closes[i]
        # If this 1m bar is the 5th minute of the bucket (minute % 5 == 4), bucket just closed!
        if ts.minute % 5 == 4:
            five_m_closes.append(closes[i])
            if len(five_m_closes) >= 21:
                arr5 = np.array(five_m_closes, dtype=np.float64)
                e9_5 = _compute_ema_array(arr5, 9)[-1]
                e21_5 = _compute_ema_array(arr5, 21)[-1]
                last_mtf = 1 if e9_5 > e21_5 else (-1 if e9_5 < e21_5 else 0)
            bucket_close = None
        mtf_all[i] = last_mtf

    cum_pv = 0.0
    cum_v = 0.0
    cum_pv2 = 0.0
    cur_orb_hi = -np.inf
    cur_orb_lo = np.inf
    orb_bars_seen = 0

    for j in range(n_sess):
        idx = n_warmup + j
        ts = timestamps[idx]
        tp = (highs[idx] + lows[idx] + closes[idx]) / 3.0
        vol = max(volumes[idx], 1.0)
        cum_pv += tp * vol
        cum_v += vol
        cum_pv2 += (tp * tp) * vol
        v_mean = cum_pv / cum_v
        var = max(0.0, (cum_pv2 / cum_v) - (v_mean * v_mean))
        vwap[j] = v_mean
        vwap_sd[j] = float(np.sqrt(var))

        if ts.time() < ORB_END_IST:
            cur_orb_hi = max(cur_orb_hi, highs[idx])
            cur_orb_lo = min(cur_orb_lo, lows[idx])
            orb_bars_seen += 1
            orb_high[j] = cur_orb_hi
            orb_low[j] = cur_orb_lo
            orb_ready[j] = False
        else:
            if orb_bars_seen > 0:
                orb_high[j] = cur_orb_hi
                orb_low[j] = cur_orb_lo
                orb_ready[j] = True

        mtf_5m_dir[j] = mtf_all[idx]

        # Causal Liquidity Sweep (previous 10 bars excluding current bar)
        if idx >= 10:
            prev_lo = float(np.min(lows[idx - 10 : idx]))
            prev_hi = float(np.max(highs[idx - 10 : idx]))
            body_lo = min(opens[idx], closes[idx])
            body_hi = max(opens[idx], closes[idx])
            bar_rng = max(highs[idx] - lows[idx], 1e-6)
            lower_wick = body_lo - lows[idx]
            upper_wick = highs[idx] - body_hi
            if lows[idx] < prev_lo and closes[idx] > prev_lo and lower_wick >= 0.4 * bar_rng:
                sweep_signal[j] = 1
            elif highs[idx] > prev_hi and closes[idx] < prev_hi and upper_wick >= 0.4 * bar_rng:
                sweep_signal[j] = -1

        # Causal 3-candle Fair Value Gap (idx-2, idx-1, idx)
        if idx >= 2:
            if lows[idx] > highs[idx - 2]:
                fvg_signal[j] = 1
            elif highs[idx] < lows[idx - 2]:
                fvg_signal[j] = -1

    return {
        "n_warmup": n_warmup,
        "n_sess": n_sess,
        "opens": opens[n_warmup:],
        "highs": highs[n_warmup:],
        "lows": lows[n_warmup:],
        "closes": closes[n_warmup:],
        "volumes": volumes[n_warmup:],
        "timestamps": timestamps[n_warmup:],
        "ema9": ema9[n_warmup:],
        "ema21": ema21[n_warmup:],
        "rsi14": rsi14[n_warmup:],
        "atr14": atr14[n_warmup:],
        "adx14": adx14[n_warmup:],
        "plus_di14": plus_di14[n_warmup:],
        "minus_di14": minus_di14[n_warmup:],
        "st_val": st_val[n_warmup:],
        "st_dir": st_dir[n_warmup:],
        "vwap": vwap,
        "vwap_sd": vwap_sd,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_ready": orb_ready,
        "pdh": pdh,
        "pdl": pdl,
        "mtf_5m_dir": mtf_5m_dir,
        "sweep_signal": sweep_signal,
        "fvg_signal": fvg_signal,
    }


def _infer_auto_direction_fast(
    st_dir: int,
    mtf_dir: int,
    close: float,
    vwap: float,
    plus_di: float,
    minus_di: float,
    adx: float,
) -> str:
    """Fast scalar equivalent of `infer_auto_market_direction` (4-vote causal consensus)."""
    bull = 0
    bear = 0
    if st_dir == 1:
        bull += 1
    elif st_dir == -1:
        bear += 1
    if mtf_dir == 1:
        bull += 1
    elif mtf_dir == -1:
        bear += 1
    if not np.isnan(vwap) and vwap > 0.0:
        if close > vwap * 1.0005:
            bull += 1
        elif close < vwap * 0.9995:
            bear += 1
    if not np.isnan(plus_di) and not np.isnan(minus_di):
        if plus_di > minus_di + 2.0:
            bull += 1
        elif minus_di > plus_di + 2.0:
            bear += 1
    if not np.isnan(adx) and adx < 16.0:
        return "NEUTRAL"
    if bull >= 3:
        return "BULLISH"
    if bear >= 3:
        return "BEARISH"
    return "NEUTRAL"


def simulate_fast_1m_session(
    bundle: OneMinuteSessionBundle,
    config: FastSessionConfig | None = None,
    *,
    precomputed: dict[str, Any] | None = None,
    ml_scorer: Callable[[dict[str, float]], float] | None = None,
) -> FastSessionSummary:
    """Run a strictly causal 1m paper-trading session with NSE statutory taxes."""
    cfg = config or FastSessionConfig()
    arr = precomputed if precomputed is not None else precompute_causal_1m_arrays(bundle)

    n = arr["n_sess"]
    opens = arr["opens"]
    highs = arr["highs"]
    lows = arr["lows"]
    closes = arr["closes"]
    volumes = arr["volumes"]
    timestamps = arr["timestamps"]
    ema9 = arr["ema9"]
    ema21 = arr["ema21"]
    rsi14 = arr["rsi14"]
    atr14 = arr["atr14"]
    adx14 = arr["adx14"]
    plus_di14 = arr["plus_di14"]
    minus_di14 = arr["minus_di14"]
    st_dir = arr["st_dir"]
    vwap = arr["vwap"]
    vwap_sd = arr["vwap_sd"]
    orb_high = arr["orb_high"]
    orb_low = arr["orb_low"]
    orb_ready = arr["orb_ready"]
    pdh = arr["pdh"]
    pdl = arr["pdl"]
    mtf_5m_dir = arr["mtf_5m_dir"]
    sweep_signal = arr["sweep_signal"]
    fvg_signal = arr["fvg_signal"]

    trades: list[FastTradeRecord] = []
    entries_today = 0
    cooldown_left = 0
    blocked_by_ml = 0

    # Open position state
    pos_side: str | None = None  # "BUY" or "SELL"
    pos_qty = 0
    pos_entry = 0.0
    pos_entry_time = ""
    pos_initial_sl = 0.0
    pos_active_sl = 0.0
    pos_tp = 0.0
    pos_peak = 0.0
    pos_trough = 0.0
    pos_entry_features: dict[str, float] = {}
    pos_ml_prob: float | None = None

    equity = cfg.initial_capital_inr
    peak_equity = equity
    max_dd = 0.0

    def _close_position(exit_px: float, exit_ts_str: str, reason: str) -> None:
        nonlocal pos_side, pos_qty, pos_entry, pos_entry_time, pos_initial_sl, pos_active_sl
        nonlocal pos_tp, pos_peak, pos_trough, cooldown_left, equity, peak_equity, max_dd
        nonlocal pos_entry_features, pos_ml_prob

        if pos_side == "BUY":
            gross = (exit_px - pos_entry) * pos_qty
            buy_cost = calculate_intraday_costs(price=pos_entry, quantity=pos_qty, side="BUY").total_friction_inr
            sell_cost = calculate_intraday_costs(price=exit_px, quantity=pos_qty, side="SELL").total_friction_inr
            risk_per_sh = max(pos_entry - pos_initial_sl, 1e-6)
            r_mult = (exit_px - pos_entry) / risk_per_sh
        else:
            gross = (pos_entry - exit_px) * pos_qty
            # Exact SHORT leg ordering: SELL at entry, BUY at close
            buy_cost = calculate_intraday_costs(price=pos_entry, quantity=pos_qty, side="SELL").total_friction_inr
            sell_cost = calculate_intraday_costs(price=exit_px, quantity=pos_qty, side="BUY").total_friction_inr
            risk_per_sh = max(pos_initial_sl - pos_entry, 1e-6)
            r_mult = (pos_entry - exit_px) / risk_per_sh

        friction = buy_cost + sell_cost
        net = gross - friction
        equity += net
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, peak_equity - equity)

        trades.append(
            FastTradeRecord(
                symbol=bundle.symbol,
                trading_date=bundle.trading_date.isoformat(),
                side=pos_side or "BUY",
                entry_time_ist=pos_entry_time,
                exit_time_ist=exit_ts_str,
                entry_price=round(pos_entry, 2),
                exit_price=round(exit_px, 2),
                quantity=pos_qty,
                gross_pnl_inr=round(gross, 2),
                friction_inr=round(friction, 2),
                net_pnl_inr=round(net, 2),
                exit_reason=reason,
                r_multiple=round(r_mult, 3),
                strategy_profile=cfg.strategy_profile,
                entry_features=dict(pos_entry_features),
                ml_win_prob=pos_ml_prob,
            )
        )
        pos_side = None
        pos_qty = 0
        cooldown_left = cfg.cooldown_bars

    for i in range(n):
        ts = timestamps[i]
        t_ist = ts.time()
        c = closes[i]
        h = highs[i]
        l = lows[i]
        atr = atr14[i]
        adx = adx14[i]
        if np.isnan(atr) or atr <= 0.0:
            atr = max(c * 0.002, 0.5)

        # 1. Manage open position (EOD square-off, SL/TP, Chandelier Trail + Breakeven Lock)
        if pos_side is not None:
            if t_ist >= EOD_SQUAREOFF_IST or i == n - 1:
                _close_position(c, ts.isoformat(), "EOD_SQUARE_OFF")
                continue

            if pos_side == "BUY":
                if l <= pos_active_sl:
                    reason = "TRAILING_STOP_HIT" if pos_active_sl > pos_initial_sl else "STOP_LOSS_HIT"
                    _close_position(pos_active_sl, ts.isoformat(), reason)
                    continue
                if h >= pos_tp:
                    _close_position(pos_tp, ts.isoformat(), "TARGET_HIT")
                    continue
                # Causal Chandelier + Breakeven update at bar close
                pos_peak = max(pos_peak, h)
                risk_unit = max(pos_entry - pos_initial_sl, 1e-6)
                cand_sl = pos_active_sl
                if (pos_peak - pos_entry) >= cfg.breakeven_r_trigger * risk_unit:
                    be_sl = pos_entry * (1.0 + cfg.breakeven_fee_buffer_pct)
                    cand_sl = max(cand_sl, be_sl)
                chand_sl = pos_peak - cfg.chandelier_atr_mult * atr
                cand_sl = max(cand_sl, chand_sl)
                if cand_sl < c:
                    pos_active_sl = cand_sl
            else:
                if h >= pos_active_sl:
                    reason = "TRAILING_STOP_HIT" if pos_active_sl < pos_initial_sl else "STOP_LOSS_HIT"
                    _close_position(pos_active_sl, ts.isoformat(), reason)
                    continue
                if l <= pos_tp:
                    _close_position(pos_tp, ts.isoformat(), "TARGET_HIT")
                    continue
                pos_trough = min(pos_trough, l)
                risk_unit = max(pos_initial_sl - pos_entry, 1e-6)
                cand_sl = pos_active_sl
                if (pos_entry - pos_trough) >= cfg.breakeven_r_trigger * risk_unit:
                    be_sl = pos_entry * (1.0 - cfg.breakeven_fee_buffer_pct)
                    cand_sl = min(cand_sl, be_sl)
                chand_sl = pos_trough + cfg.chandelier_atr_mult * atr
                cand_sl = min(cand_sl, chand_sl)
                if cand_sl > c:
                    pos_active_sl = cand_sl
            continue

        # 2. Entry checks (flat)
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if entries_today >= cfg.max_entries_per_day:
            continue
        if t_ist < ENTRY_START_IST or t_ist > ENTRY_END_IST:
            continue

        # Determine causal market direction
        if cfg.auto_direction or cfg.fixed_direction == "AUTO":
            eff_dir = _infer_auto_direction_fast(
                int(st_dir[i]),
                int(mtf_5m_dir[i]),
                c,
                vwap[i],
                plus_di14[i],
                minus_di14[i],
                adx,
            )
        else:
            eff_dir = cfg.fixed_direction

        if eff_dir not in ("BULLISH", "BEARISH"):
            continue

        # ADX chop filter (skip for LIQUIDITY_REVERSAL which can fade sweeps in ranging markets)
        adx_val = 0.0 if np.isnan(adx) else float(adx)
        if cfg.strategy_profile != "LIQUIDITY_REVERSAL" and adx_val < cfg.adx_min:
            continue

        # VWAP ±2σ anti-chase stretch guard
        v = vwap[i]
        vsd = vwap_sd[i]
        if not np.isnan(v) and not np.isnan(vsd) and vsd > 0.0:
            if eff_dir == "BULLISH" and c > v + 2.0 * vsd:
                continue
            if eff_dir == "BEARISH" and c < v - 2.0 * vsd:
                continue

        # Profile-specific causal entry trigger
        want_side: str | None = None
        if cfg.strategy_profile == "ORB_BREAKOUT":
            if orb_ready[i]:
                if eff_dir == "BULLISH" and c > orb_high[i] and st_dir[i] == 1:
                    want_side = "BUY"
                elif eff_dir == "BEARISH" and c < orb_low[i] and st_dir[i] == -1:
                    want_side = "SELL"
        elif cfg.strategy_profile == "SUPERTREND_ADX":
            if eff_dir == "BULLISH" and st_dir[i] == 1 and ema9[i] > ema21[i] and c > v:
                want_side = "BUY"
            elif eff_dir == "BEARISH" and st_dir[i] == -1 and ema9[i] < ema21[i] and c < v:
                want_side = "SELL"
        elif cfg.strategy_profile == "LIQUIDITY_REVERSAL":
            if eff_dir == "BULLISH" and (sweep_signal[i] == 1 or fvg_signal[i] == 1) and rsi14[i] < 65.0:
                want_side = "BUY"
            elif eff_dir == "BEARISH" and (sweep_signal[i] == -1 or fvg_signal[i] == -1) and rsi14[i] > 35.0:
                want_side = "SELL"
        else:  # BALANCED_ALL
            if eff_dir == "BULLISH":
                trend_ok = st_dir[i] == 1 and ema9[i] > ema21[i] and c > v
                trigger_ok = (
                    (orb_ready[i] and c > orb_high[i])
                    or sweep_signal[i] == 1
                    or fvg_signal[i] == 1
                    or (rsi14[i] >= 52.0 and rsi14[i] <= 68.0)
                )
                if trend_ok and trigger_ok:
                    want_side = "BUY"
            elif eff_dir == "BEARISH":
                trend_ok = st_dir[i] == -1 and ema9[i] < ema21[i] and c < v
                trigger_ok = (
                    (orb_ready[i] and c < orb_low[i])
                    or sweep_signal[i] == -1
                    or fvg_signal[i] == -1
                    or (rsi14[i] <= 48.0 and rsi14[i] >= 32.0)
                )
                if trend_ok and trigger_ok:
                    want_side = "SELL"

        if want_side is None:
            continue

        # Risk sizing & stop/target geometry
        stop_dist = max(1.5 * atr, c * 0.0025)
        target_dist = cfg.rr_target * stop_dist
        risk_budget = equity * cfg.risk_pct
        max_notional_qty = int((equity * 0.90) // c)
        risk_qty = int(risk_budget // stop_dist)
        qty = min(max_notional_qty, risk_qty)
        if qty <= 0:
            continue

        # 3x NSE statutory friction-to-reward gate
        if want_side == "BUY":
            est_tp = c + target_dist
            est_sl = c - stop_dist
            rt_cost = (
                calculate_intraday_costs(price=c, quantity=qty, side="BUY").total_friction_inr
                + calculate_intraday_costs(price=est_tp, quantity=qty, side="SELL").total_friction_inr
            )
        else:
            est_tp = c - target_dist
            est_sl = c + stop_dist
            rt_cost = (
                calculate_intraday_costs(price=c, quantity=qty, side="SELL").total_friction_inr
                + calculate_intraday_costs(price=est_tp, quantity=qty, side="BUY").total_friction_inr
            )
        expected_gross_reward = target_dist * qty
        if rt_cost <= 0.0 or (expected_gross_reward / rt_cost) < cfg.min_reward_to_friction:
            continue

        # Strictly causal entry feature vector (for CYCLE-022 ML scorer & auditability)
        bar_rng = max(h - l, 1e-6)
        body_ratio = abs(c - opens[i]) / bar_rng
        vwap_z = (c - v) / vsd if (not np.isnan(v) and not np.isnan(vsd) and vsd > 1e-6) else 0.0
        ema_spread_pct = (ema9[i] - ema21[i]) / c if c > 0 else 0.0
        atr_pct = atr / c if c > 0 else 0.0
        side_sign = 1.0 if want_side == "BUY" else -1.0
        minutes_from_open = float((t_ist.hour - 9) * 60 + (t_ist.minute - 15))
        features: dict[str, float] = {
            "side_sign": side_sign,
            "adx_14": adx_val,
            "di_diff_aligned": side_sign
            * (
                (0.0 if np.isnan(plus_di14[i]) else float(plus_di14[i]))
                - (0.0 if np.isnan(minus_di14[i]) else float(minus_di14[i]))
            ),
            "rsi_14_centered": side_sign * (float(rsi14[i]) - 50.0),
            "ema_spread_aligned_bps": side_sign * ema_spread_pct * 10_000.0,
            "vwap_z_aligned": side_sign * float(np.clip(vwap_z, -4.0, 4.0)),
            "atr_pct_bps": atr_pct * 10_000.0,
            "body_ratio": float(body_ratio),
            "mtf_5m_aligned": 1.0 if (mtf_5m_dir[i] * side_sign > 0) else 0.0,
            "orb_breakout_aligned": 1.0
            if (
                orb_ready[i]
                and ((want_side == "BUY" and c > orb_high[i]) or (want_side == "SELL" and c < orb_low[i]))
            )
            else 0.0,
            "sweep_aligned": 1.0 if (sweep_signal[i] * side_sign > 0) else 0.0,
            "fvg_aligned": 1.0 if (fvg_signal[i] * side_sign > 0) else 0.0,
            "minutes_from_open": minutes_from_open,
            "reward_to_friction": float(expected_gross_reward / max(rt_cost, 1e-6)),
            "near_pdhl": 1.0
            if (
                (not np.isnan(pdh) and abs(c - pdh) / c <= 0.0025)
                or (not np.isnan(pdl) and abs(c - pdl) / c <= 0.0025)
            )
            else 0.0,
        }

        ml_prob: float | None = None
        if ml_scorer is not None:
            ml_prob = float(ml_scorer(features))
            if cfg.min_ml_win_prob is not None and ml_prob < cfg.min_ml_win_prob:
                blocked_by_ml += 1
                continue

        # Open paper position
        pos_side = want_side
        pos_qty = qty
        pos_entry = c
        pos_entry_time = ts.isoformat()
        pos_initial_sl = est_sl
        pos_active_sl = est_sl
        pos_tp = est_tp
        pos_peak = c
        pos_trough = c
        pos_entry_features = features
        pos_ml_prob = ml_prob
        entries_today += 1

    total_gross = sum(t.gross_pnl_inr for t in trades)
    total_friction = sum(t.friction_inr for t in trades)
    total_net = sum(t.net_pnl_inr for t in trades)
    wins = sum(1 for t in trades if t.net_pnl_inr > 0.0)
    losses = sum(1 for t in trades if t.net_pnl_inr <= 0.0)

    return FastSessionSummary(
        symbol=bundle.symbol,
        trading_date=bundle.trading_date.isoformat(),
        strategy_profile=cfg.strategy_profile,
        initial_capital_inr=cfg.initial_capital_inr,
        bars_processed=n,
        trades=tuple(trades),
        total_gross_pnl_inr=round(total_gross, 2),
        total_friction_inr=round(total_friction, 2),
        total_net_pnl_inr=round(total_net, 2),
        wins=wins,
        losses=losses,
        max_drawdown_inr=round(max_dd, 2),
        blocked_by_ml=blocked_by_ml,
    )


def aggregate_session_metrics(sessions: Sequence[FastSessionSummary]) -> dict[str, Any]:
    """Compute honest portfolio metrics across a collection of fast 1m sessions."""
    all_trades = [t for s in sessions for t in s.trades]
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.net_pnl_inr > 0.0)
    losses = total_trades - wins
    gross_pnl = sum(t.gross_pnl_inr for t in all_trades)
    friction = sum(t.friction_inr for t in all_trades)
    net_pnl = sum(t.net_pnl_inr for t in all_trades)
    gross_wins = sum(t.net_pnl_inr for t in all_trades if t.net_pnl_inr > 0.0)
    gross_losses = abs(sum(t.net_pnl_inr for t in all_trades if t.net_pnl_inr <= 0.0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 1e-9 else (float("inf") if gross_wins > 0 else 0.0)
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0

    # Chronological equity curve drawdown across sessions
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for s in sorted(sessions, key=lambda x: (x.trading_date, x.symbol)):
        eq += s.total_net_pnl_inr
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    per_symbol: dict[str, dict[str, Any]] = {}
    for sym in sorted({s.symbol for s in sessions}):
        sym_trades = [t for s in sessions if s.symbol == sym for t in s.trades]
        sym_wins = sum(1 for t in sym_trades if t.net_pnl_inr > 0.0)
        sym_net = sum(t.net_pnl_inr for t in sym_trades)
        sym_fric = sum(t.friction_inr for t in sym_trades)
        per_symbol[sym] = {
            "sessions": sum(1 for s in sessions if s.symbol == sym),
            "trades": len(sym_trades),
            "wins": sym_wins,
            "win_rate": round(sym_wins / len(sym_trades), 4) if sym_trades else 0.0,
            "net_pnl_inr": round(sym_net, 2),
            "friction_inr": round(sym_fric, 2),
        }

    return {
        "sessions_evaluated": len(sessions),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "total_gross_pnl_inr": round(gross_pnl, 2),
        "total_friction_inr": round(friction, 2),
        "total_net_pnl_inr": round(net_pnl, 2),
        "avg_net_pnl_per_trade_inr": round(net_pnl / total_trades, 2) if total_trades else 0.0,
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else 999.0,
        "max_drawdown_inr": round(max_dd, 2),
        "blocked_by_ml": sum(s.blocked_by_ml for s in sessions),
        "per_symbol": per_symbol,
    }


def split_bundles_chronologically(
    bundles_by_symbol: dict[str, list[OneMinuteSessionBundle]],
    train_ratio: float = 0.70,
) -> tuple[list[date], list[date], list[OneMinuteSessionBundle], list[OneMinuteSessionBundle]]:
    """Split sessions strictly chronologically so `max(train_dates) < min(test_dates)`."""
    if not (0.1 < train_ratio < 0.95):
        raise ValueError(f"train_ratio must be in (0.1, 0.95), got {train_ratio}")

    all_dates = sorted({b.trading_date for bs in bundles_by_symbol.values() for b in bs})
    if len(all_dates) < 2:
        raise ValueError(f"Need at least 2 distinct trading dates for walk-forward split, got {len(all_dates)}")

    split_idx = max(1, min(len(all_dates) - 1, int(len(all_dates) * train_ratio)))
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]
    assert max(train_dates) < min(test_dates), "Chronological leakage violation!"

    train_set = set(train_dates)
    test_set = set(test_dates)
    train_bundles: list[OneMinuteSessionBundle] = []
    test_bundles: list[OneMinuteSessionBundle] = []
    for sym in sorted(bundles_by_symbol.keys()):
        for b in bundles_by_symbol[sym]:
            if b.trading_date in train_set:
                train_bundles.append(b)
            elif b.trading_date in test_set:
                test_bundles.append(b)

    return train_dates, test_dates, train_bundles, test_bundles


def run_walkforward_optimization(
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS_1M,
    base_dir: str | Path = DEFAULT_1M_DIR,
    profiles: Sequence[StrategyProfileLiteral] = ALL_STRATEGY_PROFILES,
    capital_tiers: Sequence[float] = (50_000.0, 200_000.0),
    train_ratio: float = 0.70,
    max_days: int | None = None,
    bundles_override: dict[str, list[OneMinuteSessionBundle]] | None = None,
) -> dict[str, Any]:
    """Run full strictly chronological Train → unseen OOS Test optimization across stocks & profiles."""
    if bundles_override is not None:
        bundles_by_symbol = bundles_override
    else:
        bundles_by_symbol = {}
        for sym in symbols:
            bundles_by_symbol[sym] = load_1m_sessions(sym, base_dir=base_dir, max_days=max_days)

    train_dates, test_dates, train_bundles, test_bundles = split_bundles_chronologically(
        bundles_by_symbol, train_ratio=train_ratio
    )

    # Precompute causal arrays once per bundle (strictly per-day causal, reused across configs)
    precomp_train = [precompute_causal_1m_arrays(b) for b in train_bundles]
    precomp_test = [precompute_causal_1m_arrays(b) for b in test_bundles]

    tier_results: dict[str, Any] = {}
    for cap in capital_tiers:
        cap_key = f"INR_{int(cap)}"
        profile_matrix: dict[str, Any] = {}
        best_profile: str | None = None
        best_train_net = -float("inf")

        for prof in profiles:
            cfg = FastSessionConfig(
                strategy_profile=prof,
                initial_capital_inr=float(cap),
                auto_direction=True,
            )
            tr_res = [
                simulate_fast_1m_session(b, cfg, precomputed=pc)
                for b, pc in zip(train_bundles, precomp_train)
            ]
            te_res = [
                simulate_fast_1m_session(b, cfg, precomputed=pc)
                for b, pc in zip(test_bundles, precomp_test)
            ]
            tr_metrics = aggregate_session_metrics(tr_res)
            te_metrics = aggregate_session_metrics(te_res)
            profile_matrix[prof] = {
                "train": tr_metrics,
                "oos_test": te_metrics,
            }
            if tr_metrics["total_net_pnl_inr"] > best_train_net:
                best_train_net = tr_metrics["total_net_pnl_inr"]
                best_profile = prof

        tier_results[cap_key] = {
            "initial_capital_inr": float(cap),
            "train_selected_profile": best_profile,
            "train_selected_oos_metrics": (
                profile_matrix[best_profile]["oos_test"] if best_profile else None
            ),
            "profiles": profile_matrix,
        }

    return {
        "schema_version": "cycle020_walkforward_optimizer_v1",
        "live_trading_enabled": False,
        "zero_lookahead_verified": max(train_dates) < min(test_dates),
        "symbols": sorted(bundles_by_symbol.keys()),
        "train_date_range": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "oos_test_date_range": [test_dates[0].isoformat(), test_dates[-1].isoformat()],
        "train_days": len(train_dates),
        "oos_test_days": len(test_dates),
        "train_sessions": len(train_bundles),
        "oos_test_sessions": len(test_bundles),
        "capital_tiers": tier_results,
    }
