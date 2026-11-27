"""CYCLE-021 / REMED-01: Governed Live Market-Hours Read-Only Paper-Trading Runner.

Connects any `ReadOnlyTickAdapter` (Zerodha KiteTicker, Angel One SmartAPI WebSocket v2,
or `OfflineJsonlTickAdapter` for deterministic offline replay) through `CausalTickAggregator`
to the **mandatory 5-tier BharatAlpha governed paper execution stack**:
  1. Human Authority (strict fail-closed if missing/invalid for new entries)
  2. Central Agent directional instructions & AUTO regime mode (yields to explicit overrides)
  3. Candle Timing / Overlay recommendation (advisory only; never self-authorizing)
  4. Deterministic Risk Engine (15% per-stock cap, 80% portfolio cap, CashReservationLedger,
     2.5% daily / 3.0% MTM drawdown HALTED -> EXIT_ONLY)
  5. IntradayPaperPortfolio double-entry ledger with ₹0.05 NSE tick rounding & SHA-256 audit chains

HARD SAFETY INVARIANTS:
- Strictly READ-ONLY market-data ingestion (`LIVE_TRADING_ENABLED = False`, `READ_ONLY_MODE = True`).
- Zero broker order placement, modification, cancellation, or live execution routing.
- ZERO bypasses of `MandatoryExecutionGate`, `DeterministicRiskEngine`, or `IntradayPaperPortfolio`.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Final, Iterable

import pandas as pd

from trader.authority.gate import InstructionRegistrationResult, MandatoryExecutionGate
from trader.authority.instructions import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)
from trader.config import (
    DEFAULT_INITIAL_CAPITAL_INR,
    INTRADAY_REDUCING_WIND_DOWN_TIME,
    IST_TZ,
)
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
from trader.portfolio.ledger import IntradayPaperPortfolio, round_to_nse_tick
from trader.risk.engine import DeterministicRiskEngine, RiskState
from trader.timing.candle_timing_agent import (
    AlignmentStatus,
    TimingAssessment,
    TimingDecision,
)
from trader.timing.central_directional_agent import DirectionalBias
from trader.timing.monitoring_session import evaluate_causal_5m_mtf_trend

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
    """Runtime parameters for the governed read-only live paper-trading runner."""

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
    requested_quantity: int = 50
    min_ml_win_prob: float | None = None
    archive_timeframes: tuple[TimeframeName, ...] = ("1s", "1m", "5m")


@dataclass
class LivePaperTrade:
    """Completed paper trade booked through IntradayPaperPortfolio."""

    symbol: str
    side: str
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
class LivePaperSessionSummary:
    """Final audit summary of a governed read-only live paper-trading session."""

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
    authority_audit_verified: bool = False
    risk_audit_verified: bool = False
    ledger_integrity_verified: bool = False
    gate_rejections_count: int = 0
    final_risk_state: str = "ACTIVE"
    final_cash_inr: float = 0.0
    final_equity_inr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cycle021_remed01_live_paper_session_v2",
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
            "final_cash_inr": round(self.final_cash_inr, 2),
            "final_equity_inr": round(self.final_equity_inr, 2),
            "final_risk_state": self.final_risk_state,
            "gate_rejections_count": self.gate_rejections_count,
            "authority_audit_verified": self.authority_audit_verified,
            "risk_audit_verified": self.risk_audit_verified,
            "ledger_integrity_verified": self.ledger_integrity_verified,
            "trades": [t.to_dict() for t in self.trades],
            "archived_provenance_files": self.archived_provenance_files,
            "state_json_path": self.state_json_path,
        }


def _build_advisory_timing_assessment(
    *,
    symbol: str,
    ts_close: datetime,
    want_side: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    central_dir: str,
) -> TimingAssessment:
    """Build a typed advisory TimingAssessment for MandatoryExecutionGate Gate 3."""
    from trader.timing.candle_intelligence import CandleBias

    candle_bias = CandleBias.BULLISH if want_side == "BUY" else CandleBias.BEARISH
    central_bias = (
        DirectionalBias.BULLISH
        if central_dir == "BULLISH"
        else (DirectionalBias.BEARISH if central_dir == "BEARISH" else DirectionalBias.NEUTRAL)
    )
    alignment = (
        AlignmentStatus.CONFIRMED
        if (central_bias.value == candle_bias.value or central_bias == DirectionalBias.NEUTRAL)
        else AlignmentStatus.CONTRADICTED
    )
    risk_unit = max(abs(entry_price - stop_loss), 1e-6)
    rr = abs(target - entry_price) / risk_unit
    return TimingAssessment(
        symbol=symbol,
        as_of_ist=ts_close,
        decision=TimingDecision.ENTER_NOW,
        alignment_status=alignment,
        central_direction=central_bias,
        candle_bias=candle_bias,
        timing_quality_score=0.75,
        price_action_score=0.75,
        entry_price=entry_price,
        invalidation_price=stop_loss,
        target_price=target,
        risk_reward_ratio=round(rr, 2),
        detected_patterns=(),
        timeframe_statuses={"1m": "VALID"},
        contradiction_reasons=() if alignment != AlignmentStatus.CONTRADICTED else ("Opposing central bias",),
        rationale=(f"LivePaperRunner advisory {want_side} @ {entry_price:.2f}",),
    )


class LivePaperTradingRunner:
    """Strictly read-only live tick -> 1m bar paper runner governed by MandatoryExecutionGate."""

    def __init__(
        self,
        adapter: ReadOnlyTickAdapter,
        *,
        output_dir: str | Path = "data/live_paper",
        config: LivePaperConfig | None = None,
        portfolio: IntradayPaperPortfolio | None = None,
        risk_engine: DeterministicRiskEngine | None = None,
        execution_gate: MandatoryExecutionGate | None = None,
        human_instruction: AuthorityInstruction | None = None,
        central_instruction: AuthorityInstruction | None = None,
        ml_scorer: Callable[[dict[str, float]], float] | None = None,
        on_bar_evaluated: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        assert_paper_only_live_runner()
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.config = config or LivePaperConfig()
        self.ml_scorer = ml_scorer
        self.on_bar_evaluated = on_bar_evaluated

        # Mandatory 5-tier governed stack (NO private _open_positions or _equity_inr bypass!)
        self.portfolio = portfolio or IntradayPaperPortfolio(initial_capital=float(self.config.initial_capital_inr))
        self.risk_engine = risk_engine or DeterministicRiskEngine(
            max_risk_per_trade_pct=self.config.risk_pct,
        )
        self.execution_gate = execution_gate or MandatoryExecutionGate(risk_engine=self.risk_engine)

        # Decision 1: NO implicit Human permission. If caller omits human_instruction, Gate 1 fails closed!
        self._explicit_central_scopes: set[str] = set()
        if human_instruction is not None:
            self.execution_gate.set_human_instruction(human_instruction)
        if central_instruction is not None:
            reg = self.execution_gate.set_central_instruction(central_instruction)
            if reg.accepted:
                self._explicit_central_scopes.add(central_instruction.normalized_scope)

        tfs: tuple[TimeframeName, ...] = (
            self.config.archive_timeframes
            if "1m" in self.config.archive_timeframes
            else tuple(list(self.config.archive_timeframes) + ["1m"])
        )
        self.timeframes: tuple[TimeframeName, ...] = tfs
        self.aggregator = CausalTickAggregator(timeframes=self.timeframes)
        self.archive = TickParquetArchive(base_dir=self.output_dir / "ticks")

        # Per-symbol causal tracking (position state lives strictly in self.portfolio.positions)
        self._raw_ticks_by_symbol_day: dict[tuple[str, date], list[TickEvent]] = defaultdict(list)
        self._archived_bars: dict[tuple[str, date], dict[TimeframeName, list[RecordedBar]]] = defaultdict(
            lambda: {tf: [] for tf in self.timeframes}
        )
        self._bars_1m_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._pos_meta: dict[str, dict[str, Any]] = {}
        self._cooldown_left: dict[str, int] = defaultdict(int)
        self._entries_today: dict[tuple[str, date], int] = defaultdict(int)
        self._trades: list[LivePaperTrade] = []
        self._bars_1m_evaluated: int = 0
        self._gate_rejections_count: int = 0
        self._last_direction_by_symbol: dict[str, str] = {}
        self._last_watermark_ist: datetime | None = None

    # ------------------------------------------------------------------
    # Governed Authority Controls (Decision 1, 2, 3)
    # ------------------------------------------------------------------

    def set_human_instruction(
        self,
        instruction: AuthorityInstruction,
        *,
        current_time: datetime | None = None,
    ) -> InstructionRegistrationResult:
        """Register Human Authority instruction via MandatoryExecutionGate (triggers safe mgmt on STOP_ALL)."""
        res = self.execution_gate.set_human_instruction(instruction, current_time=current_time)
        if res.accepted and instruction.permission == PermissionType.STOP_ALL:
            self.execution_gate.release_reservations_for_scope(instruction.normalized_scope)
            ts = current_time or self._last_watermark_ist or datetime.now(IST_TZ)
            self.execution_gate.invoke_safe_position_management(
                symbol=instruction.normalized_scope,
                portfolio=self.portfolio,
                current_time=ts,
            )
            self._sync_closed_trades_from_portfolio("HUMAN_STOP_ALL_SAFE_MANAGEMENT")
        return res

    def set_central_instruction(
        self,
        instruction: AuthorityInstruction,
        *,
        current_time: datetime | None = None,
    ) -> InstructionRegistrationResult:
        """Register explicit Central Agent instruction (takes precedence over AUTO mode per Decision 3)."""
        res = self.execution_gate.set_central_instruction(instruction, current_time=current_time)
        if res.accepted:
            self._explicit_central_scopes.add(instruction.normalized_scope)
        return res

    def emergency_human_stop(self, *, reason: str = "HUMAN_EMERGENCY_STOP", current_time: datetime) -> AuthorityInstruction:
        """Issue immediate Human STOP_ALL and flatten open positions via audited RISK_EXIT."""
        instr = self.execution_gate.emergency_human_stop(
            reason=reason,
            current_time=current_time,
            scope="*",
            portfolio=self.portfolio,
        )
        self._sync_closed_trades_from_portfolio("HUMAN_EMERGENCY_STOP")
        return instr

    def _sync_closed_trades_from_portfolio(self, exit_reason: str) -> None:
        """Mirror newly closed trades from self.portfolio.closed_trades into self._trades."""
        while len(self._trades) < len(self.portfolio.closed_trades):
            ct = self.portfolio.closed_trades[len(self._trades)]
            sym = str(ct.get("symbol", "")).upper()
            meta = self._pos_meta.pop(sym, {})
            self._trades.append(
                LivePaperTrade(
                    symbol=sym,
                    side="SELL" if str(ct.get("side", "LONG")).upper() == "SHORT" else "BUY",
                    entry_time_ist=str(ct.get("entry_time", meta.get("entry_time_ist", ""))),
                    exit_time_ist=str(ct.get("exit_time", "")),
                    entry_price=float(ct.get("entry_price", 0.0)),
                    exit_price=float(ct.get("exit_price", 0.0)),
                    quantity=int(ct.get("quantity", 0)),
                    gross_pnl_inr=float(ct.get("gross_pnl_inr", 0.0)),
                    friction_inr=float(ct.get("total_friction_inr", 0.0)),
                    net_pnl_inr=float(ct.get("net_pnl_inr", 0.0)),
                    exit_reason=exit_reason,
                    strategy_profile=self.config.strategy_profile,
                    ml_win_prob=meta.get("ml_win_prob"),
                )
            )
            self._cooldown_left[sym] = self.config.cooldown_bars

    def _execute_governed_exit(
        self,
        sym: str,
        exit_price: float,
        exit_ts: datetime,
        reason: str,
        *,
        is_risk_exit: bool = True,
    ) -> bool:
        pos = self.portfolio.positions.get(sym)
        if pos is None or pos.quantity <= 0:
            return False
        if is_risk_exit:
            act = ActionType.RISK_EXIT
        else:
            act = ActionType.EXIT_SHORT if getattr(pos, "side", "LONG") == "SHORT" else ActionType.EXIT_LONG
        outcome, executed = self.execution_gate.execute_with_gate_protection(
            symbol=sym,
            action_type=act,
            current_time=exit_ts,
            portfolio=self.portfolio,
            requested_quantity=pos.quantity,
            price=round_to_nse_tick(exit_price),
        )
        if executed:
            self._sync_closed_trades_from_portfolio(reason)
        else:
            self._gate_rejections_count += 1
        return executed

    def _evaluate_closed_1m_bar(self, bar: RecordedBar) -> None:
        """Strictly causal evaluation of one newly closed 1-minute NSE bar via MandatoryExecutionGate."""
        sym = bar.symbol.strip().upper()
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
        if len(self._bars_1m_history[sym]) > 450:
            self._bars_1m_history[sym] = self._bars_1m_history[sym][-450:]

        self._bars_1m_evaluated += 1
        df = pd.DataFrame(self._bars_1m_history[sym])
        overlays = compute_chart_overlays(df, session_date=day)

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

        mtf_5m_trend = evaluate_causal_5m_mtf_trend(df)

        # Decision 3: Explicit Central instruction takes precedence over AUTO regime mode.
        has_explicit_central = (sym in self._explicit_central_scopes) or ("*" in self._explicit_central_scopes)
        active_central = self.execution_gate.get_active_central_instruction(sym, ts_close)

        if self.config.auto_direction and not has_explicit_central:
            eff_dir = infer_auto_market_direction(
                mtf_5m_trend=mtf_5m_trend,
                overlays=overlays,
                current_close=bar.close,
            )
            perm = (
                PermissionType.ALLOW_LONG
                if eff_dir == "BULLISH"
                else (PermissionType.ALLOW_SHORT if eff_dir == "BEARISH" else PermissionType.NO_NEW_ENTRIES)
            )
            if active_central is None or active_central.permission != perm:
                auto_instr = AuthorityInstruction(
                    instruction_id=f"CENTRAL-AUTO-{sym}-{uuid.uuid4().hex[:8]}",
                    authority=AuthoritySource.CENTRAL_AGENT,
                    scope=sym,
                    permission=perm,
                    valid_from=ts_close,
                    expires_at=ts_close + timedelta(hours=8),
                    reason=f"AUTO Central-authorized regime mode: {eff_dir}",
                    version="1.0",
                )
                self.execution_gate.set_central_instruction(auto_instr, current_time=ts_close)
        else:
            if active_central is None:
                eff_dir = "NEUTRAL"
            elif active_central.permission == PermissionType.ALLOW_LONG:
                eff_dir = "BULLISH"
            elif active_central.permission == PermissionType.ALLOW_SHORT:
                eff_dir = "BEARISH"
            elif active_central.permission == PermissionType.ALLOW_BOTH:
                eff_dir = infer_auto_market_direction(
                    mtf_5m_trend=mtf_5m_trend,
                    overlays=overlays,
                    current_close=bar.close,
                )
            else:
                eff_dir = "NEUTRAL"
        self._last_direction_by_symbol[sym] = eff_dir

        # Update MTM price & check drawdown circuit breaker (Decision 4: HALTED -> EXIT_ONLY)
        pos = self.portfolio.positions.get(sym)
        if pos is None:
            for p in self.portfolio.positions.values():
                if p.symbol.strip().upper() == sym:
                    pos = p
                    break
        if pos is not None and pos.quantity > 0:
            pos.current_price = round_to_nse_tick(bar.close)
        self.risk_engine.check_daily_loss_limit(self.portfolio)
        if self.risk_engine.state == RiskState.ACTIVE and ts_close.time() >= INTRADAY_REDUCING_WIND_DOWN_TIME:
            self.risk_engine.state = RiskState.REDUCING

        # 1. Manage existing open position (always allowed under EXIT_ONLY / HALTED / PAUSE for RISK_EXIT)
        if pos is not None and pos.quantity > 0:
            is_short_pos = getattr(pos, "side", "LONG") == "SHORT"
            default_sl = round_to_nse_tick(bar.close * (1.015 if is_short_pos else 0.985))
            default_tp = round_to_nse_tick(bar.close * (0.97 if is_short_pos else 1.03))
            meta = self._pos_meta.setdefault(
                sym,
                {
                    "initial_sl": pos.stop_loss or default_sl,
                    "active_sl": pos.stop_loss or default_sl,
                    "take_profit": pos.target or default_tp,
                    "peak_price": pos.entry_price,
                    "trough_price": pos.entry_price,
                    "entry_time_ist": ts_close.isoformat(),
                    "ml_win_prob": None,
                },
            )
            if ts_close.time() >= EOD_SQUAREOFF_IST:
                self._execute_governed_exit(sym, bar.close, ts_close, "EOD_SQUARE_OFF", is_risk_exit=True)
            elif getattr(pos, "side", "LONG") == "LONG":
                if bar.low <= meta["active_sl"]:
                    reason = "TRAILING_STOP_HIT" if meta["active_sl"] > meta["initial_sl"] else "STOP_LOSS_HIT"
                    self._execute_governed_exit(sym, meta["active_sl"], ts_close, reason, is_risk_exit=True)
                elif bar.high >= meta["take_profit"]:
                    self._execute_governed_exit(sym, meta["take_profit"], ts_close, "TARGET_HIT", is_risk_exit=True)
                else:
                    meta["peak_price"] = max(meta["peak_price"], bar.high)
                    risk_u = max(pos.entry_price - meta["initial_sl"], 1e-6)
                    cand = meta["active_sl"]
                    if (meta["peak_price"] - pos.entry_price) >= self.config.breakeven_r_trigger * risk_u:
                        cand = max(cand, pos.entry_price * (1.0 + self.config.breakeven_fee_buffer_pct))
                    cand = max(cand, meta["peak_price"] - self.config.chandelier_atr_mult * atr)
                    cand_tick = round_to_nse_tick(cand)
                    if cand_tick < bar.close and cand_tick > meta["active_sl"]:
                        meta["active_sl"] = cand_tick
                        pos.stop_loss = cand_tick
            else:
                if bar.high >= meta["active_sl"]:
                    reason = "TRAILING_STOP_HIT" if meta["active_sl"] < meta["initial_sl"] else "STOP_LOSS_HIT"
                    self._execute_governed_exit(sym, meta["active_sl"], ts_close, reason, is_risk_exit=True)
                elif bar.low <= meta["take_profit"]:
                    self._execute_governed_exit(sym, meta["take_profit"], ts_close, "TARGET_HIT", is_risk_exit=True)
                else:
                    meta["trough_price"] = min(meta["trough_price"], bar.low)
                    risk_u = max(meta["initial_sl"] - pos.entry_price, 1e-6)
                    cand = meta["active_sl"]
                    if (pos.entry_price - meta["trough_price"]) >= self.config.breakeven_r_trigger * risk_u:
                        cand = min(cand, pos.entry_price * (1.0 - self.config.breakeven_fee_buffer_pct))
                    cand = min(cand, meta["trough_price"] + self.config.chandelier_atr_mult * atr)
                    cand_tick = round_to_nse_tick(cand)
                    if cand_tick > bar.close and cand_tick < meta["active_sl"]:
                        meta["active_sl"] = cand_tick
                        pos.stop_loss = cand_tick
            if self.on_bar_evaluated is not None:
                self.on_bar_evaluated(self._build_bar_event(sym, bar, eff_dir, overlays.to_dict()))
            return

        # 2. Entry evaluation when flat (MUST pass MandatoryExecutionGate!)
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
                    entry_px = round_to_nse_tick(bar.close)
                    stop_dist = max(1.5 * atr, entry_px * 0.0025)
                    target_dist = self.config.rr_target * stop_dist
                    if want_side == "BUY":
                        sl = round_to_nse_tick(entry_px - stop_dist)
                        tp = round_to_nse_tick(entry_px + target_dist)
                        act = ActionType.ENTRY_LONG
                    else:
                        sl = round_to_nse_tick(entry_px + stop_dist)
                        tp = round_to_nse_tick(entry_px - target_dist)
                        act = ActionType.ENTRY_SHORT

                    advisory = _build_advisory_timing_assessment(
                        symbol=sym,
                        ts_close=ts_close,
                        want_side=want_side,
                        entry_price=entry_px,
                        stop_loss=sl,
                        target=tp,
                        central_dir=eff_dir,
                    )

                    # Dry-run gate check to obtain Risk-Engine-approved quantity (respects 15% cap, 80% cap, cash)
                    pre_gate = self.execution_gate.evaluate_gate(
                        symbol=sym,
                        action_type=act,
                        current_time=ts_close,
                        portfolio=self.portfolio,
                        candle_assessment=advisory,
                        requested_quantity=self.config.requested_quantity,
                        entry_price=entry_px,
                        stop_loss_price=sl,
                        target_price=tp,
                        dry_run=True,
                    )
                    if not pre_gate.approved or pre_gate.risk_evaluation is None:
                        self._gate_rejections_count += 1
                    else:
                        qty = pre_gate.risk_evaluation.approved_quantity
                        if qty > 0:
                            if want_side == "BUY":
                                rt_cost = (
                                    calculate_intraday_costs(price=entry_px, quantity=qty, side="BUY").total_friction_inr
                                    + calculate_intraday_costs(price=tp, quantity=qty, side="SELL").total_friction_inr
                                )
                            else:
                                rt_cost = (
                                    calculate_intraday_costs(price=entry_px, quantity=qty, side="SELL").total_friction_inr
                                    + calculate_intraday_costs(price=tp, quantity=qty, side="BUY").total_friction_inr
                                )
                            if rt_cost > 0.0 and ((target_dist * qty) / rt_cost) >= self.config.min_reward_to_friction:
                                ml_prob: float | None = None
                                allow_entry = True
                                if self.ml_scorer is not None:
                                    side_sign = 1.0 if want_side == "BUY" else -1.0
                                    want_dir = "BULLISH" if want_side == "BUY" else "BEARISH"
                                    close_s = df["close"].astype(float)
                                    ema9_val = float(close_s.ewm(span=9, adjust=False).mean().iloc[-1])
                                    ema21_val = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
                                    ema_spread_bps = ((ema9_val - ema21_val) / bar.close) * 10_000.0
                                    deltas = close_s.diff().dropna()
                                    gains = deltas.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
                                    losses = (-deltas.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
                                    if len(losses) > 0 and float(losses.iloc[-1]) > 1e-12:
                                        rs_val = float(gains.iloc[-1]) / float(losses.iloc[-1])
                                        rsi_val = 100.0 - 100.0 / (1.0 + rs_val)
                                    else:
                                        rsi_val = 50.0
                                    vwap_sd_est = (
                                        (overlays.vwap_upper_1sd - overlays.vwap)
                                        if (overlays.vwap is not None and overlays.vwap_upper_1sd is not None)
                                        else 0.0
                                    )
                                    vwap_z_val = (
                                        max(-4.0, min(4.0, (bar.close - overlays.vwap) / vwap_sd_est))
                                        if (overlays.vwap is not None and vwap_sd_est > 1e-6)
                                        else 0.0
                                    )
                                    feat = {
                                        "side_sign": side_sign,
                                        "adx_14": float(adx_val),
                                        "di_diff_aligned": side_sign
                                        * ((overlays.plus_di_14 or 0.0) - (overlays.minus_di_14 or 0.0)),
                                        "rsi_14_centered": side_sign * (rsi_val - 50.0),
                                        "ema_spread_aligned_bps": side_sign * ema_spread_bps,
                                        "vwap_z_aligned": side_sign * vwap_z_val,
                                        "atr_pct_bps": (atr / bar.close) * 10_000.0,
                                        "body_ratio": abs(bar.close - bar.open) / max(bar.high - bar.low, 1e-6),
                                        "mtf_5m_aligned": 1.0 if mtf_5m_trend == want_dir else 0.0,
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
                                        "fvg_aligned": 1.0
                                        if (
                                            overlays.recent_fvg is not None
                                            and overlays.recent_fvg.direction == want_dir
                                        )
                                        else 0.0,
                                        "minutes_from_open": float((t_ist.hour - 9) * 60 + (t_ist.minute - 15)),
                                        "reward_to_friction": float((target_dist * qty) / rt_cost),
                                        "near_pdhl": 1.0 if overlays.near_key_level else 0.0,
                                    }
                                    ml_prob = float(self.ml_scorer(feat))
                                    if self.config.min_ml_win_prob is not None and ml_prob < self.config.min_ml_win_prob:
                                        allow_entry = False

                                if allow_entry:
                                    res_id = f"LIVE-{sym}-{uuid.uuid4().hex[:8]}"
                                    self.execution_gate.reserve_cash_for_pending_order(res_id, sym, entry_px * qty)
                                    outcome, executed = self.execution_gate.execute_with_gate_protection(
                                        symbol=sym,
                                        action_type=act,
                                        current_time=ts_close,
                                        portfolio=self.portfolio,
                                        requested_quantity=qty,
                                        price=entry_px,
                                        candle_assessment=advisory,
                                        stop_loss_price=sl,
                                        target_price=tp,
                                        reservation_order_id=res_id,
                                    )
                                    if executed:
                                        self._pos_meta[sym] = {
                                            "initial_sl": sl,
                                            "active_sl": sl,
                                            "take_profit": tp,
                                            "peak_price": entry_px,
                                            "trough_price": entry_px,
                                            "entry_time_ist": ts_close.isoformat(),
                                            "ml_win_prob": ml_prob,
                                        }
                                        self._entries_today[(sym, day)] += 1
                                    else:
                                        self._gate_rejections_count += 1

        if self.on_bar_evaluated is not None:
            self.on_bar_evaluated(self._build_bar_event(sym, bar, eff_dir, overlays.to_dict()))

    def _build_bar_event(
        self,
        sym: str,
        bar: RecordedBar,
        eff_dir: str,
        overlay_dict: dict[str, Any],
    ) -> dict[str, Any]:
        pos = self.portfolio.positions.get(sym)
        meta = self._pos_meta.get(sym, {})
        return {
            "symbol": sym,
            "ts_open_ist": bar.ts_open_ist.isoformat(),
            "ts_close_ist": bar.ts_close_ist.isoformat(),
            "close": bar.close,
            "auto_direction": eff_dir,
            "risk_state": self.risk_engine.state.value,
            "open_position": (
                {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "active_sl": round(meta.get("active_sl", pos.stop_loss or 0.0), 2),
                    "take_profit": round(meta.get("take_profit", pos.target or 0.0), 2),
                }
                if pos is not None
                else None
            ),
            "cash_inr": round(self.portfolio.cash, 2),
            "equity_inr": round(self.portfolio.total_equity, 2),
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
        """Stream read-only ticks, evaluate closed 1m bars via MandatoryExecutionGate, and archive."""
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

            # Release any remaining cash reservations & close all open positions via audited RISK_EXIT
            self.execution_gate.clear_cash_reservations()
            for pos in list(self.portfolio.positions.values()):
                if pos is not None and pos.quantity > 0:
                    sym = pos.symbol.strip().upper()
                    hist = self._bars_1m_history.get(sym)
                    if hist:
                        last_bar = hist[-1]
                        exit_px = float(last_bar["close"])
                        exit_ts = last_bar["timestamp"] + timedelta(minutes=1)
                    else:
                        exit_px = float(getattr(pos, "current_price", pos.entry_price) or pos.entry_price)
                        exit_ts = self._last_watermark_ist or datetime.now(IST_TZ)
                    self._execute_governed_exit(
                        sym,
                        exit_px,
                        exit_ts,
                        "SESSION_SHUTDOWN_SQUARE_OFF",
                        is_risk_exit=True,
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

            ledger_ok, _ = self.portfolio.verify_ledger_integrity()
            risk_ok = bool(self.risk_engine.verify_audit_chain())
            auth_ok = self.execution_gate.verify_audit_chain()

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
                authority_audit_verified=auth_ok,
                risk_audit_verified=risk_ok,
                ledger_integrity_verified=ledger_ok,
                gate_rejections_count=self._gate_rejections_count,
                final_risk_state=self.risk_engine.state.value,
                final_cash_inr=self.portfolio.cash,
                final_equity_inr=self.portfolio.total_equity,
            )

            self.output_dir.mkdir(parents=True, exist_ok=True)
            state_path = self.output_dir / "live_paper_state.json"
            state_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
            summary.state_json_path = str(state_path)

        return summary
