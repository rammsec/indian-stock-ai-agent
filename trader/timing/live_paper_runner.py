"""CYCLE-021: Live Market-Hours Read-Only Paper-Trading Runner.

Connects any `ReadOnlyTickAdapter` (Zerodha KiteTicker, Angel One SmartAPI WebSocket v2,
or `OfflineJsonlTickAdapter` for deterministic offline replay) through `CausalTickAggregator`
to a live paper-trading evaluator with simultaneous Parquet tick/bar archival.

HARD SAFETY INVARIANTS:
- Strictly READ-ONLY market-data ingestion (`LIVE_TRADING_ENABLED = False`, `READ_ONLY_MODE = True`).
- Zero broker order placement, modification, cancellation, or live execution routing.
- Every closed 1-minute bar is evaluated strictly causally at `ts_close_ist <= watermark_ist`.
- All simulated paper fills apply exact Indian NSE intraday statutory taxes & slippage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Final, Iterable

import pandas as pd

from trader.config import DEFAULT_INITIAL_CAPITAL_INR, IST_TZ
from trader.data.tick_recorder import (
    CausalTickAggregator,
    ReadOnlyTickAdapter,
    RecordedBar,
    TickEvent,
    TickParquetArchive,
    TimeframeName,
    assert_read_only_recorder_config,
)
from trader.intelligence.chart_overlays import (
    StrategyProfileLiteral,
    compute_chart_overlays,
    infer_auto_market_direction,
)
from trader.portfolio.cost_calculator import calculate_intraday_costs

LIVE_TRADING_ENABLED: Final[bool] = False
READ_ONLY_MODE: Final[bool] = True

ENTRY_START_IST: Final[time] = time(9, 20)
ENTRY_END_IST: Final[time] = time(14, 45)
EOD_SQUAREOFF_IST: Final[time] = time(15, 15)


def assert_paper_only_live_runner(
    *,
    live_trading_enabled: bool = False,
    read_only: bool = True,
) -> None:
    """Fail-closed safety guard ensuring paper-only, read-only execution at all times."""
    assert_read_only_recorder_config(
        live_trading_enabled=live_trading_enabled,
        read_only=read_only,
    )
    if live_trading_enabled or LIVE_TRADING_ENABLED:
        raise RuntimeError(
            "SAFETY VIOLATION: LivePaperTradingRunner is strictly paper-only (LIVE_TRADING_ENABLED=False)."
        )
    if not read_only or not READ_ONLY_MODE:
        raise RuntimeError(
            "SAFETY VIOLATION: LivePaperTradingRunner requires READ_ONLY_MODE=True."
        )


@dataclass(frozen=True)
class LivePaperConfig:
    """Runtime parameters for the read-only live paper-trading runner."""

    strategy_profile: StrategyProfileLiteral = "BALANCED_ALL"
    initial_capital_inr: float = DEFAULT_INITIAL_CAPITAL_INR
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
    min_ml_win_prob: float | None = None
    archive_timeframes: tuple[TimeframeName, ...] = ("1s", "1m", "5m")


@dataclass
class LivePaperTrade:
    """Completed paper trade emitted during a live or offline-replayed tick session."""

    symbol: str
    side: str  # "BUY" or "SELL"
    entry_time_ist: str
    exit_time_ist: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl_inr: float
    friction_inr: float
    net_pnl_inr: float
    exit_reason: str
    strategy_profile: str
    ml_win_prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_time_ist": self.entry_time_ist,
            "exit_time_ist": self.exit_time_ist,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "quantity": self.quantity,
            "gross_pnl_inr": round(self.gross_pnl_inr, 2),
            "friction_inr": round(self.friction_inr, 2),
            "net_pnl_inr": round(self.net_pnl_inr, 2),
            "exit_reason": self.exit_reason,
            "strategy_profile": self.strategy_profile,
            "ml_win_prob": round(self.ml_win_prob, 4) if self.ml_win_prob is not None else None,
        }


@dataclass
class _OpenPaperPosition:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    entry_time_ist: str
    initial_sl: float
    active_sl: float
    take_profit: float
    peak_price: float
    trough_price: float
    ml_win_prob: float | None = None


@dataclass
class LivePaperSessionSummary:
    """Final audit summary of a read-only live paper-trading session."""

    provider: str
    symbols: list[str]
    strategy_profile: str
    live_trading_enabled: bool
    read_only_mode: bool
    ticks_accepted: int
    bars_1m_evaluated: int
    trades: list[LivePaperTrade] = field(default_factory=list)
    total_gross_pnl_inr: float = 0.0
    total_friction_inr: float = 0.0
    total_net_pnl_inr: float = 0.0
    archived_provenance_files: list[str] = field(default_factory=list)
    state_json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cycle021_live_paper_session_v1",
            "provider": self.provider,
            "symbols": self.symbols,
            "strategy_profile": self.strategy_profile,
            "live_trading_enabled": False,
            "read_only_mode": True,
            "ticks_accepted": self.ticks_accepted,
            "bars_1m_evaluated": self.bars_1m_evaluated,
            "trade_count": len(self.trades),
            "total_gross_pnl_inr": round(self.total_gross_pnl_inr, 2),
            "total_friction_inr": round(self.total_friction_inr, 2),
            "total_net_pnl_inr": round(self.total_net_pnl_inr, 2),
            "trades": [t.to_dict() for t in self.trades],
            "archived_provenance_files": self.archived_provenance_files,
            "state_json_path": self.state_json_path,
        }


class LivePaperTradingRunner:
    """Strictly read-only live tick -> 1m bar paper-trading runner + Parquet archiver."""

    def __init__(
        self,
        adapter: ReadOnlyTickAdapter,
        *,
        output_dir: str | Path = "data/live_paper",
        config: LivePaperConfig | None = None,
        ml_scorer: Callable[[dict[str, float]], float] | None = None,
        on_bar_evaluated: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        assert_paper_only_live_runner()
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.config = config or LivePaperConfig()
        self.ml_scorer = ml_scorer
        self.on_bar_evaluated = on_bar_evaluated

        tfs: tuple[TimeframeName, ...] = (
            self.config.archive_timeframes
            if "1m" in self.config.archive_timeframes
            else tuple(list(self.config.archive_timeframes) + ["1m"])
        )
        self.timeframes: tuple[TimeframeName, ...] = tfs
        self.aggregator = CausalTickAggregator(timeframes=self.timeframes)
        self.archive = TickParquetArchive(base_dir=self.output_dir / "ticks")

        # Per-symbol causal state
        self._raw_ticks_by_symbol_day: dict[tuple[str, date], list[TickEvent]] = defaultdict(list)
        self._archived_bars: dict[tuple[str, date], dict[TimeframeName, list[RecordedBar]]] = defaultdict(
            lambda: {tf: [] for tf in self.timeframes}
        )
        self._bars_1m_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._open_positions: dict[str, _OpenPaperPosition | None] = defaultdict(lambda: None)
        self._cooldown_left: dict[str, int] = defaultdict(int)
        self._entries_today: dict[tuple[str, date], int] = defaultdict(int)
        self._equity_inr: float = float(self.config.initial_capital_inr)
        self._trades: list[LivePaperTrade] = []
        self._bars_1m_evaluated: int = 0
        self._last_direction_by_symbol: dict[str, str] = {}
        self._last_watermark_ist: datetime | None = None

    def _close_position(
        self,
        sym: str,
        exit_price: float,
        exit_ts: datetime,
        reason: str,
    ) -> None:
        pos = self._open_positions.get(sym)
        if pos is None:
            return
        if pos.side == "BUY":
            gross = (exit_price - pos.entry_price) * pos.quantity
            c1 = calculate_intraday_costs(price=pos.entry_price, quantity=pos.quantity, side="BUY").total_friction_inr
            c2 = calculate_intraday_costs(price=exit_price, quantity=pos.quantity, side="SELL").total_friction_inr
        else:
            gross = (pos.entry_price - exit_price) * pos.quantity
            c1 = calculate_intraday_costs(price=pos.entry_price, quantity=pos.quantity, side="SELL").total_friction_inr
            c2 = calculate_intraday_costs(price=exit_price, quantity=pos.quantity, side="BUY").total_friction_inr
        friction = c1 + c2
        net = gross - friction
        self._equity_inr += net
        self._trades.append(
            LivePaperTrade(
                symbol=sym,
                side=pos.side,
                entry_time_ist=pos.entry_time_ist,
                exit_time_ist=exit_ts.astimezone(IST_TZ).isoformat(),
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                gross_pnl_inr=gross,
                friction_inr=friction,
                net_pnl_inr=net,
                exit_reason=reason,
                strategy_profile=self.config.strategy_profile,
                ml_win_prob=pos.ml_win_prob,
            )
        )
        self._open_positions[sym] = None
        self._cooldown_left[sym] = self.config.cooldown_bars

    def _evaluate_closed_1m_bar(self, bar: RecordedBar) -> None:
        """Strictly causal evaluation of one newly closed 1-minute NSE bar."""
        sym = bar.symbol
        ts_open = bar.ts_open_ist.astimezone(IST_TZ)
        ts_close = bar.ts_close_ist.astimezone(IST_TZ)
        t_ist = ts_open.time()
        day = ts_open.date()

        self._bars_1m_history[sym].append(
            {
                "timestamp": ts_open,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
        # Keep bounded causal history (up to 450 bars ~ 1.2 NSE sessions)
        if len(self._bars_1m_history[sym]) > 450:
            self._bars_1m_history[sym] = self._bars_1m_history[sym][-450:]

        self._bars_1m_evaluated += 1
        df = pd.DataFrame(self._bars_1m_history[sym])
        overlays = compute_chart_overlays(df, session_date=day)

        # Estimate ATR causally from recent bars
        if len(df) >= 2:
            hi = df["high"].to_numpy(dtype=float)
            lo = df["low"].to_numpy(dtype=float)
            cl = df["close"].to_numpy(dtype=float)
            tr = [hi[0] - lo[0]] + [
                max(hi[k] - lo[k], abs(hi[k] - cl[k - 1]), abs(lo[k] - cl[k - 1]))
                for k in range(1, len(df))
            ]
            atr = float(sum(tr[-14:]) / min(14, len(tr)))
        else:
            atr = max(bar.close * 0.002, 0.5)
        if atr <= 0.0:
            atr = max(bar.close * 0.002, 0.5)

        eff_dir = infer_auto_market_direction(
            mtf_5m_trend="NEUTRAL",
            overlays=overlays,
            current_close=bar.close,
        )
        self._last_direction_by_symbol[sym] = eff_dir

        # 1. Manage open paper position
        pos = self._open_positions.get(sym)
        if pos is not None:
            if t_ist >= EOD_SQUAREOFF_IST:
                self._close_position(sym, bar.close, ts_close, "EOD_SQUARE_OFF")
            elif pos.side == "BUY":
                if bar.low <= pos.active_sl:
                    reason = "TRAILING_STOP_HIT" if pos.active_sl > pos.initial_sl else "STOP_LOSS_HIT"
                    self._close_position(sym, pos.active_sl, ts_close, reason)
                elif bar.high >= pos.take_profit:
                    self._close_position(sym, pos.take_profit, ts_close, "TARGET_HIT")
                else:
                    pos.peak_price = max(pos.peak_price, bar.high)
                    risk_u = max(pos.entry_price - pos.initial_sl, 1e-6)
                    cand = pos.active_sl
                    if (pos.peak_price - pos.entry_price) >= self.config.breakeven_r_trigger * risk_u:
                        cand = max(cand, pos.entry_price * (1.0 + self.config.breakeven_fee_buffer_pct))
                    cand = max(cand, pos.peak_price - self.config.chandelier_atr_mult * atr)
                    if cand < bar.close:
                        pos.active_sl = cand
            else:
                if bar.high >= pos.active_sl:
                    reason = "TRAILING_STOP_HIT" if pos.active_sl < pos.initial_sl else "STOP_LOSS_HIT"
                    self._close_position(sym, pos.active_sl, ts_close, reason)
                elif bar.low <= pos.take_profit:
                    self._close_position(sym, pos.take_profit, ts_close, "TARGET_HIT")
                else:
                    pos.trough_price = min(pos.trough_price, bar.low)
                    risk_u = max(pos.initial_sl - pos.entry_price, 1e-6)
                    cand = pos.active_sl
                    if (pos.entry_price - pos.trough_price) >= self.config.breakeven_r_trigger * risk_u:
                        cand = min(cand, pos.entry_price * (1.0 - self.config.breakeven_fee_buffer_pct))
                    cand = min(cand, pos.trough_price + self.config.chandelier_atr_mult * atr)
                    if cand > bar.close:
                        pos.active_sl = cand
            if self.on_bar_evaluated is not None:
                self.on_bar_evaluated(self._build_bar_event(sym, bar, eff_dir, overlays.to_dict()))
            return

        # 2. Entry evaluation when flat
        if self._cooldown_left[sym] > 0:
            self._cooldown_left[sym] -= 1
        elif (
            self._entries_today[(sym, day)] < self.config.max_entries_per_day
            and ENTRY_START_IST <= t_ist <= ENTRY_END_IST
            and len(df) >= 20
            and eff_dir in ("BULLISH", "BEARISH")
        ):
            adx_val = overlays.adx_14 if overlays.adx_14 is not None else 0.0
            adx_ok = (self.config.strategy_profile == "LIQUIDITY_REVERSAL") or (adx_val >= self.config.adx_min)
            stretch_ok = not (
                (eff_dir == "BULLISH" and overlays.vwap_stretch == "OVEREXTENDED_ABOVE_2SD")
                or (eff_dir == "BEARISH" and overlays.vwap_stretch == "OVEREXTENDED_BELOW_2SD")
            )
            if adx_ok and stretch_ok:
                want_side: str | None = None
                if self.config.strategy_profile == "ORB_BREAKOUT":
                    if eff_dir == "BULLISH" and overlays.orb_state == "ORB_BREAKOUT_UP" and overlays.supertrend_direction == "BULLISH":
                        want_side = "BUY"
                    elif eff_dir == "BEARISH" and overlays.orb_state == "ORB_BREAKOUT_DOWN" and overlays.supertrend_direction == "BEARISH":
                        want_side = "SELL"
                elif self.config.strategy_profile == "LIQUIDITY_REVERSAL":
                    if eff_dir == "BULLISH" and overlays.liquidity_sweep == "BULLISH_SWEEP":
                        want_side = "BUY"
                    elif eff_dir == "BEARISH" and overlays.liquidity_sweep == "BEARISH_SWEEP":
                        want_side = "SELL"
                else:
                    if eff_dir == "BULLISH" and overlays.supertrend_direction == "BULLISH":
                        want_side = "BUY"
                    elif eff_dir == "BEARISH" and overlays.supertrend_direction == "BEARISH":
                        want_side = "SELL"

                if want_side is not None:
                    stop_dist = max(1.5 * atr, bar.close * 0.0025)
                    target_dist = self.config.rr_target * stop_dist
                    qty = min(
                        int((self._equity_inr * 0.90) // bar.close),
                        int((self._equity_inr * self.config.risk_pct) // stop_dist),
                    )
                    if qty > 0:
                        if want_side == "BUY":
                            tp = bar.close + target_dist
                            sl = bar.close - stop_dist
                            rt_cost = (
                                calculate_intraday_costs(price=bar.close, quantity=qty, side="BUY").total_friction_inr
                                + calculate_intraday_costs(price=tp, quantity=qty, side="SELL").total_friction_inr
                            )
                        else:
                            tp = bar.close - target_dist
                            sl = bar.close + stop_dist
                            rt_cost = (
                                calculate_intraday_costs(price=bar.close, quantity=qty, side="SELL").total_friction_inr
                                + calculate_intraday_costs(price=tp, quantity=qty, side="BUY").total_friction_inr
                            )
                        if rt_cost > 0.0 and ((target_dist * qty) / rt_cost) >= self.config.min_reward_to_friction:
                            ml_prob: float | None = None
                            allow_entry = True
                            if self.ml_scorer is not None:
                                side_sign = 1.0 if want_side == "BUY" else -1.0
                                feat = {
                                    "side_sign": side_sign,
                                    "adx_14": float(adx_val),
                                    "di_diff_aligned": side_sign
                                    * ((overlays.plus_di_14 or 0.0) - (overlays.minus_di_14 or 0.0)),
                                    "rsi_14_centered": 0.0,
                                    "ema_spread_aligned_bps": 0.0,
                                    "vwap_z_aligned": 0.0,
                                    "atr_pct_bps": (atr / bar.close) * 10_000.0,
                                    "body_ratio": abs(bar.close - bar.open) / max(bar.high - bar.low, 1e-6),
                                    "mtf_5m_aligned": 0.0,
                                    "orb_breakout_aligned": 1.0
                                    if (
                                        (want_side == "BUY" and overlays.orb_state == "ORB_BREAKOUT_UP")
                                        or (want_side == "SELL" and overlays.orb_state == "ORB_BREAKOUT_DOWN")
                                    )
                                    else 0.0,
                                    "sweep_aligned": 1.0
                                    if (
                                        (want_side == "BUY" and overlays.liquidity_sweep == "BULLISH_SWEEP")
                                        or (want_side == "SELL" and overlays.liquidity_sweep == "BEARISH_SWEEP")
                                    )
                                    else 0.0,
                                    "fvg_aligned": 1.0 if overlays.recent_fvg is not None else 0.0,
                                    "minutes_from_open": float((t_ist.hour - 9) * 60 + (t_ist.minute - 15)),
                                    "reward_to_friction": float((target_dist * qty) / rt_cost),
                                    "near_pdhl": 1.0 if overlays.near_key_level else 0.0,
                                }
                                ml_prob = float(self.ml_scorer(feat))
                                if self.config.min_ml_win_prob is not None and ml_prob < self.config.min_ml_win_prob:
                                    allow_entry = False

                            if allow_entry:
                                self._open_positions[sym] = _OpenPaperPosition(
                                    symbol=sym,
                                    side=want_side,
                                    quantity=qty,
                                    entry_price=bar.close,
                                    entry_time_ist=ts_close.isoformat(),
                                    initial_sl=sl,
                                    active_sl=sl,
                                    take_profit=tp,
                                    peak_price=bar.high,
                                    trough_price=bar.low,
                                    ml_win_prob=ml_prob,
                                )
                                self._entries_today[(sym, day)] += 1

        if self.on_bar_evaluated is not None:
            self.on_bar_evaluated(self._build_bar_event(sym, bar, eff_dir, overlays.to_dict()))

    def _build_bar_event(
        self,
        sym: str,
        bar: RecordedBar,
        eff_dir: str,
        overlay_dict: dict[str, Any],
    ) -> dict[str, Any]:
        pos = self._open_positions.get(sym)
        return {
            "symbol": sym,
            "ts_open_ist": bar.ts_open_ist.isoformat(),
            "ts_close_ist": bar.ts_close_ist.isoformat(),
            "close": bar.close,
            "auto_direction": eff_dir,
            "open_position": (
                {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "active_sl": round(pos.active_sl, 2),
                    "take_profit": round(pos.take_profit, 2),
                }
                if pos is not None
                else None
            ),
            "equity_inr": round(self._equity_inr, 2),
            "overlays": overlay_dict,
        }

    def _handle_tick(self, tick: TickEvent) -> None:
        """Ingest one read-only tick, flush any causally closed bars, and evaluate 1m closes."""
        if not self.aggregator.ingest_tick(tick):
            return
        ts_ist = tick.timestamp_ist.astimezone(IST_TZ)
        self._raw_ticks_by_symbol_day[(tick.symbol, ts_ist.date())].append(tick)
        if self._last_watermark_ist is None or ts_ist > self._last_watermark_ist:
            self._last_watermark_ist = ts_ist

        closed_bars = self.aggregator.flush_closed_bars(self._last_watermark_ist)
        for bar in closed_bars:
            day = bar.ts_open_ist.astimezone(IST_TZ).date()
            self._archived_bars[(bar.symbol, day)][bar.timeframe].append(bar)
            if bar.timeframe == "1m":
                self._evaluate_closed_1m_bar(bar)

    def run(self, symbols: Iterable[str]) -> LivePaperSessionSummary:
        """Stream read-only ticks, evaluate closed 1m bars in paper mode, and archive Parquet + state JSON."""
        assert_paper_only_live_runner()
        clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not clean_symbols:
            raise ValueError("At least one NSE symbol is required")

        try:
            self.adapter.stream(clean_symbols, self._handle_tick)
        finally:
            remaining = self.aggregator.finalize_all()
            for bar in remaining:
                day = bar.ts_open_ist.astimezone(IST_TZ).date()
                self._archived_bars[(bar.symbol, day)][bar.timeframe].append(bar)
                if bar.timeframe == "1m":
                    self._evaluate_closed_1m_bar(bar)

            # Force paper EOD close of any open position on runner shutdown
            for sym, pos in list(self._open_positions.items()):
                if pos is not None and self._bars_1m_history[sym]:
                    last_bar = self._bars_1m_history[sym][-1]
                    self._close_position(
                        sym,
                        float(last_bar["close"]),
                        last_bar["timestamp"],
                        "SESSION_SHUTDOWN_SQUARE_OFF",
                    )

            prov_files: list[str] = []
            for (sym, day), raw_ticks in sorted(self._raw_ticks_by_symbol_day.items()):
                tf_map = self._archived_bars.get((sym, day), {tf: [] for tf in self.timeframes})
                day_dir, _ = self.archive.write_day_archive(
                    symbol=sym,
                    trading_date=day,
                    provider=self.adapter.provider_name,
                    raw_ticks=raw_ticks,
                    bars_by_timeframe=tf_map,
                    out_of_order_ticks=self.aggregator.out_of_order_ticks,
                    rejected_outside_hours_ticks=self.aggregator.rejected_outside_hours_ticks,
                )
                prov_files.append(str(day_dir / "provenance.json"))

            summary = LivePaperSessionSummary(
                provider=self.adapter.provider_name,
                symbols=clean_symbols,
                strategy_profile=self.config.strategy_profile,
                live_trading_enabled=False,
                read_only_mode=True,
                ticks_accepted=self.aggregator.accepted_ticks,
                bars_1m_evaluated=self._bars_1m_evaluated,
                trades=list(self._trades),
                total_gross_pnl_inr=sum(t.gross_pnl_inr for t in self._trades),
                total_friction_inr=sum(t.friction_inr for t in self._trades),
                total_net_pnl_inr=sum(t.net_pnl_inr for t in self._trades),
                archived_provenance_files=prov_files,
            )

            self.output_dir.mkdir(parents=True, exist_ok=True)
            state_path = self.output_dir / "live_paper_state.json"
            state_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
            summary.state_json_path = str(state_path)

        return summary
