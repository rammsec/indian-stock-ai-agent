"""CYCLE-008 Phase 2 — Deterministic Walk-Forward Evaluation Engine.

Evaluates 5 strategies strictly offline under ``OfflineExecutionGuard`` with:
- Strictly chronological 3-way split (train / validation / dev_test_oos) over the
  197 non-holdout 1h trading days (zero shuffling, zero overlap, zero exposed holdout
  dates, zero parameter tuning on dev_test_oos).
- Strict pre-cutoff `< as_of_ist` feature slicing (zero lookahead).
- Realistic intraday execution via ``DeterministicRiskEngine`` + ``IntradayPaperPortfolio``
  (5 bps slippage, statutory charges, intraday bar-by-bar SL/target/EOD square-off,
  MAE/MFE tracking).
- Archived diagnostic runs of ``CentralDirectionalAgent`` and ``Cycle007Pipeline``
  over the 3 committed archive sessions (`2026-11-17`, `2026-11-18`, `2026-11-19`),
  explicitly tagged ``EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY`` and ``INSUFFICIENT_EVIDENCE``.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.evaluation.data_readiness import (
    DEFAULT_HOLDOUT_REGISTRY_PATH,
    audit_historical_data_readiness,
)
from trader.evaluation.metrics import TradeRecord, compute_strategy_metrics
from trader.paper.cycle006b import (
    DEFAULT_CACHE_DIR,
    ExecutionInstrumentation,
    OfflineExecutionGuard,
    sha256_file,
)
from trader.paper.cycle006c_snapshot import (
    DEFAULT_FORWARD_PREDICTIONS_PATH,
    DEFAULT_MARKET_REGIMES_PATH,
    DEFAULT_PREDICTION_LOG_PATH,
    DEFAULT_STRUCTURED_EVENTS_PATH,
    reconstruct_as_of_snapshot,
)
from trader.portfolio.cost_calculator import CostConfig
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus, SimulatedOrder
from trader.risk.engine import DeterministicRiskEngine
from trader.timing.candle_intelligence import CandleIntelligenceEngine
from trader.timing.candle_timing_agent import (
    CandleTimingAgent,
    CandleTimingConfig,
    TimingDecision,
)
from trader.timing.central_directional_agent import (
    CONFIDENCE_CALIBRATION_NOTE,
    CONFIDENCE_CALIBRATION_STATUS,
    CentralDirectionalAgent,
    CentralDirectionalConfig,
    DirectionalBias,
    DirectionalHypothesis,
)
from trader.timing.cycle007_pipeline import (
    REPORT_COMMIT_DISTINCTION,
    Cycle007Config,
    Cycle007Pipeline,
)

CYCLE008_IMPLEMENTATION_MODULES: tuple[str, ...] = (
    "trader/evaluation/__init__.py",
    "trader/evaluation/data_readiness.py",
    "trader/evaluation/metrics.py",
    "trader/evaluation/walk_forward.py",
    "scripts/run_cycle_008_walk_forward_eval.py",
)

DEFAULT_ARCHIVED_DIAGNOSTIC_SESSIONS: tuple[str, ...] = (
    "2026-11-17",
    "2026-11-18",
    "2026-11-19",
)


def _collect_cycle008_execution_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Capture git revision, working tree cleanliness, and SHA-256 of CYCLE-008 modules."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit_sha = "UNKNOWN"

    try:
        raw_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty_paths = [line[3:].strip() for line in raw_status.splitlines() if line.strip()]
        working_tree_clean = len(dirty_paths) == 0
    except Exception:
        working_tree_clean = False
        dirty_paths = ["GIT_STATUS_UNAVAILABLE"]

    implementation_file_hashes: dict[str, str] = {}
    for rel_path in CYCLE008_IMPLEMENTATION_MODULES:
        abs_path = root / rel_path
        if abs_path.exists():
            implementation_file_hashes[rel_path] = sha256_file(abs_path)

    return {
        "implementation_commit_sha": commit_sha,
        "working_tree_clean_at_execution": working_tree_clean,
        "dirty_paths_at_execution": dirty_paths,
        "implementation_file_hashes": implementation_file_hashes,
        "report_commit_distinction": REPORT_COMMIT_DISTINCTION,
    }


@dataclass(frozen=True)
class Cycle008Config:
    """Configuration for CYCLE-008 historical walk-forward evaluation."""

    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE
    initial_capital_inr: float = 50000.0
    slippage_bps: float = 5.0
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    min_history_bars: int = 25
    min_bars_per_session: int = 6
    as_of_time_ist: dt_time = dt_time(9, 15)
    cache_dir: Path = DEFAULT_CACHE_DIR
    holdout_registry_path: Path = DEFAULT_HOLDOUT_REGISTRY_PATH
    prediction_log_path: Path = DEFAULT_PREDICTION_LOG_PATH
    forward_predictions_path: Path = DEFAULT_FORWARD_PREDICTIONS_PATH
    market_regimes_path: Path = DEFAULT_MARKET_REGIMES_PATH
    structured_events_path: Path = DEFAULT_STRUCTURED_EVENTS_PATH
    archived_diagnostic_sessions: tuple[str, ...] = DEFAULT_ARCHIVED_DIAGNOSTIC_SESSIONS
    central_config: CentralDirectionalConfig = field(default_factory=CentralDirectionalConfig)
    timing_config: CandleTimingConfig = field(default_factory=CandleTimingConfig)
    max_risk_per_trade_pct: float = 0.005
    max_position_weight_pct: float = 0.40
    max_portfolio_exposure_pct: float = 0.90
    max_daily_loss_pct: float = 0.02
    max_concurrent_positions: int = 3

    def __post_init__(self) -> None:
        if self.initial_capital_inr <= 0:
            raise ValueError("initial_capital_inr must be > 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("train_ratio, val_ratio, test_ratio must all be > 0")
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.min_history_bars < 10:
            raise ValueError("min_history_bars must be >= 10")
        if self.min_bars_per_session < 1:
            raise ValueError("min_bars_per_session must be >= 1")
        if not self.symbols:
            raise ValueError("symbols must not be empty")


def _extract_order_costs(order: SimulatedOrder | None) -> tuple[float, float, float, float]:
    """Return (brokerage_inr, statutory_charges_inr, slippage_inr, total_friction_inr)."""
    if order is None or order.cost_breakdown is None:
        return 0.0, 0.0, 0.0, 0.0
    cb = order.cost_breakdown
    brokerage = float(cb.brokerage_inr)
    statutory = (
        float(cb.stt_inr)
        + float(cb.exchange_charges_inr)
        + float(cb.sebi_fee_inr)
        + float(cb.stamp_duty_inr)
        + float(cb.gst_inr)
    )
    slippage = float(cb.slippage_inr)
    total = float(cb.total_friction_inr)
    return brokerage, statutory, slippage, total


def _extract_dict_order_costs(order_dict: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not order_dict or not order_dict.get("cost_breakdown"):
        return 0.0, 0.0, 0.0, 0.0
    cb = order_dict["cost_breakdown"]
    brokerage = float(cb.get("brokerage_inr", 0.0))
    statutory = (
        float(cb.get("stt_inr", 0.0))
        + float(cb.get("exchange_charges_inr", 0.0))
        + float(cb.get("sebi_fee_inr", 0.0))
        + float(cb.get("stamp_duty_inr", 0.0))
        + float(cb.get("gst_inr", 0.0))
    )
    slippage = float(cb.get("slippage_inr", 0.0))
    total = float(cb.get("total_friction_inr", 0.0))
    return brokerage, statutory, slippage, total


def _realized_session_direction(day_df: pd.DataFrame) -> str:
    """Return realized session direction comparing final close vs opening bar open."""
    if day_df.empty:
        return "FLAT"
    open_px = float(day_df["open"].iloc[0])
    close_px = float(day_df["close"].iloc[-1])
    if close_px > open_px:
        return "BULLISH"
    if close_px < open_px:
        return "BEARISH"
    return "FLAT"


def _compute_pre_cutoff_regime_proxy(
    pre_frames_by_symbol: dict[str, pd.DataFrame],
    lookback_bars: int = 20,
) -> str:
    """Deterministic strictly pre-cutoff 1h trend-regime proxy across the 5-stock universe."""
    rets: list[float] = []
    for df in pre_frames_by_symbol.values():
        if len(df) >= lookback_bars + 1:
            c_now = float(df["close"].iloc[-1])
            c_prev = float(df["close"].iloc[-(lookback_bars + 1)])
            if c_prev > 0 and math.isfinite(c_now) and math.isfinite(c_prev):
                rets.append(((c_now - c_prev) / c_prev) * 100.0)
    if not rets:
        return "UNKNOWN_1H_PROXY"
    med = float(np.median(rets))
    if med > 1.0:
        return "TRENDING_UP_1H_PROXY"
    if med < -1.0:
        return "TRENDING_DOWN_1H_PROXY"
    return "SIDEWAYS_1H_PROXY"


def build_chronological_splits(
    non_holdout_dates: list[str],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> dict[str, list[str]]:
    """Build strictly chronological 3-way split (train / validation / dev_test_oos)."""
    if not (
        math.isfinite(train_ratio)
        and math.isfinite(val_ratio)
        and math.isfinite(test_ratio)
    ):
        raise ValueError("train_ratio, val_ratio, test_ratio must all be finite")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, test_ratio must all be > 0")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    sorted_dates = sorted(set(non_holdout_dates))
    n = len(sorted_dates)
    if n < 3:
        raise ValueError(f"Need at least 3 non-holdout dates for 3-way split; got {n}")
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_dates = sorted_dates[:n_train]
    val_dates = sorted_dates[n_train : n_train + n_val]
    test_dates = sorted_dates[n_train + n_val :]

    if not test_dates:
        raise ValueError("dev_test_oos split is empty")
    if max(train_dates) >= min(val_dates) or max(val_dates) >= min(test_dates):
        raise ValueError("Chronological ordering violated in walk-forward split")

    return {
        "train": train_dates,
        "validation": val_dates,
        "dev_test_oos": test_dates,
    }


def _load_validated_1h_frames(
    symbols: tuple[str, ...],
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load and strictly validate committed 1h parquet files for all symbols."""
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_u = sym.strip().upper()
        pq_path = cache_dir / f"{sym_u}_1h.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing 1h archive for {sym_u}: {pq_path}")
        df = pd.read_parquet(pq_path)
        req = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {pq_path}: {sorted(missing)}")
        ts = pd.to_datetime(df["timestamp"])
        if ts.isna().any():
            raise ValueError(f"NaT timestamps in {pq_path}")
        if ts.dt.tz is None:
            raise ValueError(f"Timezone-naive timestamps rejected in {pq_path}")
        ts_ist = ts.dt.tz_convert(IST_TZ)
        if ts_ist.duplicated().any():
            raise ValueError(f"Duplicate timestamps in {pq_path}")
        work = df.copy()
        work["timestamp"] = ts_ist
        work = work.sort_values("timestamp").reset_index(drop=True)
        work["session_date"] = work["timestamp"].dt.date
        frames[sym_u] = work
    return frames


class Cycle008WalkForwardEvaluator:
    """Deterministic, offline, lookahead-free CYCLE-008 strategy evaluation engine."""

    def __init__(self, config: Cycle008Config | None = None) -> None:
        self.config = config or Cycle008Config()
        self.instrumentation = ExecutionInstrumentation()

    def _create_portfolio_and_risk(self) -> tuple[IntradayPaperPortfolio, DeterministicRiskEngine]:
        cfg = self.config
        portfolio = IntradayPaperPortfolio(
            initial_capital=cfg.initial_capital_inr,
            cost_config=CostConfig(base_slippage_bps=cfg.slippage_bps),
        )
        risk_engine = DeterministicRiskEngine(
            max_risk_per_trade_pct=cfg.max_risk_per_trade_pct,
            max_position_weight_pct=cfg.max_position_weight_pct,
            max_portfolio_exposure_pct=cfg.max_portfolio_exposure_pct,
            max_daily_loss_pct=cfg.max_daily_loss_pct,
            max_concurrent_positions=cfg.max_concurrent_positions,
        )
        return portfolio, risk_engine

    def _simulate_strategy_on_dates(
        self,
        *,
        dates_iso: list[str],
        frames: dict[str, pd.DataFrame],
        signal_fn: Callable[[str, date, datetime, pd.DataFrame, float], dict[str, Any]],
        leakage_tracker: dict[str, int],
        enforce_timing_gap_rules: bool = False,
    ) -> dict[str, Any]:
        """Run realistic bar-by-bar intraday simulation over a chronological date list."""
        cfg = self.config
        portfolio, risk_engine = self._create_portfolio_and_risk()
        trades: list[TradeRecord] = []
        equity_curve: list[dict[str, Any]] = []
        directional_outcomes: list[dict[str, Any]] = []
        decision_counts: dict[str, int] = {}
        skipped_symbol_sessions = 0
        risk_rejections = 0
        gap_rejections = 0

        for d_str in dates_iso:
            session_d = date.fromisoformat(d_str)
            as_of_ist = datetime.combine(session_d, cfg.as_of_time_ist, tzinfo=IST_TZ)
            as_of_ts = pd.Timestamp(as_of_ist)

            portfolio.reset_daily_session(session_d)
            risk_engine.is_daily_halted = False

            pre_by_sym: dict[str, pd.DataFrame] = {}
            day_by_sym: dict[str, pd.DataFrame] = {}

            for sym in cfg.symbols:
                sym_u = sym.strip().upper()
                df = frames[sym_u]
                pre_df = df[df["timestamp"] < as_of_ts]
                day_df = df[df["session_date"] == session_d]
                if len(pre_df) < cfg.min_history_bars or len(day_df) < cfg.min_bars_per_session:
                    skipped_symbol_sessions += 1
                    continue

                # Strict zero-lookahead assertion on every single evaluated symbol-session
                max_pre_ts = pre_df["timestamp"].max()
                leakage_tracker["evaluated_symbol_sessions_checked"] += 1
                if not (max_pre_ts < as_of_ts):
                    leakage_tracker["lookahead_bar_violations"] += 1
                    raise RuntimeError(
                        f"Lookahead violation for {sym_u} on {d_str}: "
                        f"max_pre_ts={max_pre_ts} >= as_of_ist={as_of_ts}"
                    )

                pre_by_sym[sym_u] = pre_df
                day_by_sym[sym_u] = day_df

            regime_label = _compute_pre_cutoff_regime_proxy(pre_by_sym)

            # Track open positions for this session to compute exact MAE/MFE & costs
            open_meta: dict[str, dict[str, Any]] = {}

            for sym_u, pre_df in pre_by_sym.items():
                day_df = day_by_sym[sym_u]
                eligible_bars = day_df[day_df["timestamp"] >= as_of_ts]
                if eligible_bars.empty:
                    skipped_symbol_sessions += 1
                    continue

                entry_bar = eligible_bars.iloc[0]
                entry_ts = entry_bar["timestamp"].to_pydatetime()
                entry_price = float(entry_bar["open"])
                realized_dir = _realized_session_direction(eligible_bars)

                sig = signal_fn(sym_u, session_d, as_of_ist, pre_df, entry_price)
                dec_label = str(sig.get("decision_label", "NO_TRADE"))
                decision_counts[dec_label] = decision_counts.get(dec_label, 0) + 1

                predicted_dir = str(sig.get("predicted_direction", "NEUTRAL")).upper()
                traded = False

                if sig.get("enter_long", False):
                    stop_loss = sig.get("stop_loss")
                    target = sig.get("target")
                    valid_levels = (
                        stop_loss is not None
                        and target is not None
                        and entry_price > 0.0
                        and float(stop_loss) < entry_price < float(target)
                    )
                    if valid_levels and enforce_timing_gap_rules:
                        realized_stop_dist = (entry_price - float(stop_loss)) / entry_price
                        realized_rr = (float(target) - entry_price) / (entry_price - float(stop_loss))
                        if realized_stop_dist > cfg.timing_config.max_stop_distance_pct:
                            valid_levels = False
                        elif realized_rr < cfg.timing_config.min_risk_reward_ratio:
                            valid_levels = False

                    if not valid_levels:
                        gap_rejections += 1
                    else:
                        sl_f = float(stop_loss)
                        tgt_f = float(target)
                        risk_per_share = max(entry_price - sl_f, 0.01)
                        max_risk_inr = portfolio.total_equity * cfg.max_risk_per_trade_pct
                        req_qty = max(1, int(max_risk_inr // risk_per_share))

                        risk_eval = risk_engine.evaluate_order(
                            symbol=sym_u,
                            side=OrderSide.BUY.value,
                            requested_quantity=req_qty,
                            entry_price=entry_price,
                            stop_loss_price=sl_f,
                            portfolio=portfolio,
                            current_time=entry_ts,
                        )
                        if risk_eval.approved and risk_eval.approved_quantity > 0:
                            buy_order = portfolio.submit_order(
                                symbol=sym_u,
                                side=OrderSide.BUY,
                                quantity=risk_eval.approved_quantity,
                                price=entry_price,
                                stop_loss=sl_f,
                                target=tgt_f,
                                timestamp=entry_ts,
                            )
                            if buy_order.status in (
                                OrderStatus.FILLED,
                                OrderStatus.PARTIALLY_FILLED,
                            ):
                                traded = True
                                open_meta[sym_u] = {
                                    "entry_order": buy_order,
                                    "entry_price": entry_price,
                                    "quantity": int(buy_order.filled_quantity),
                                    "min_low": entry_price,
                                    "max_high": entry_price,
                                    "regime_label": regime_label,
                                    "session_date": d_str,
                                }
                            else:
                                risk_rejections += 1
                        else:
                            risk_rejections += 1

                directional_outcomes.append(
                    {
                        "symbol": sym_u,
                        "session_date": d_str,
                        "predicted_direction": predicted_dir,
                        "realized_direction": realized_dir,
                        "traded": traded,
                        "regime_label": regime_label,
                    }
                )

            # Intraday bar-by-bar SL / target / MTM simulation
            all_ts = sorted(
                {
                    ts.to_pydatetime()
                    for df in day_by_sym.values()
                    for ts in df["timestamp"]
                    if ts >= as_of_ts
                }
            )
            latest_prices: dict[str, float] = {}
            for ts in all_ts:
                for sym_u, day_df in day_by_sym.items():
                    row_df = day_df[day_df["timestamp"] == pd.Timestamp(ts)]
                    if row_df.empty:
                        continue
                    row = row_df.iloc[0]
                    bar_open = float(row["open"])
                    bar_high = float(row["high"])
                    bar_low = float(row["low"])
                    bar_close = float(row["close"])
                    latest_prices[sym_u] = bar_close

                    if sym_u not in portfolio.positions or sym_u not in open_meta:
                        continue
                    pos = portfolio.positions[sym_u]
                    if ts < pos.entry_time:
                        continue

                    meta = open_meta[sym_u]
                    meta["min_low"] = min(meta["min_low"], bar_low)
                    meta["max_high"] = max(meta["max_high"], bar_high)

                    exit_orders: list[SimulatedOrder] = []
                    exit_reason: str | None = None

                    # Conservative intraday check: stop-loss checked before target
                    if pos.stop_loss is not None and bar_low <= pos.stop_loss:
                        sl_fill = min(bar_open, pos.stop_loss)
                        exit_orders = portfolio.update_market_price(sym_u, sl_fill, ts)
                        exit_reason = "STOP_LOSS"
                    elif pos.target is not None and bar_high >= pos.target:
                        exit_orders = portfolio.update_market_price(sym_u, pos.target, ts)
                        exit_reason = "TARGET"
                    else:
                        exit_orders = portfolio.update_market_price(sym_u, bar_close, ts)
                        if exit_orders:
                            exit_reason = "INTRADAY_AUTO_EXIT"

                    if exit_orders and exit_reason is not None:
                        self._append_closed_trade_record(
                            trades=trades,
                            sym_u=sym_u,
                            meta=meta,
                            exit_order=exit_orders[-1],
                            exit_reason=exit_reason,
                        )
                        open_meta.pop(sym_u, None)

            # Mandatory end-of-session square-off
            if portfolio.positions and all_ts:
                sq_ts = all_ts[-1]
                sq_orders = portfolio.square_off_all_positions(latest_prices, sq_ts)
                for so in sq_orders:
                    sym_u = so.symbol.strip().upper()
                    meta = open_meta.pop(sym_u, None)
                    if meta is not None:
                        self._append_closed_trade_record(
                            trades=trades,
                            sym_u=sym_u,
                            meta=meta,
                            exit_order=so,
                            exit_reason="EOD_SQUARE_OFF",
                        )

            equity_curve.append(
                {
                    "date": d_str,
                    "equity_inr": round(float(portfolio.total_equity), 2),
                }
            )

        ledger_ok, ledger_msg = portfolio.verify_ledger_integrity()
        metrics = compute_strategy_metrics(
            trades=trades,
            equity_curve=equity_curve,
            directional_outcomes=directional_outcomes,
            initial_capital_inr=cfg.initial_capital_inr,
        )
        return {
            "date_count": len(dates_iso),
            "date_min": dates_iso[0] if dates_iso else None,
            "date_max": dates_iso[-1] if dates_iso else None,
            "metrics": metrics,
            "decision_counts": decision_counts,
            "skipped_symbol_sessions": skipped_symbol_sessions,
            "risk_rejections": risk_rejections,
            "gap_rejections": gap_rejections,
            "ledger_reconciliation": {
                "sha256_chain_valid": bool(ledger_ok),
                "verification_message": ledger_msg,
                "ledger_event_count": len(portfolio.event_ledger),
                "open_positions_at_end": len(portfolio.positions),
            },
            "equity_curve_points": len(equity_curve),
        }

    @staticmethod
    def _append_closed_trade_record(
        *,
        trades: list[TradeRecord],
        sym_u: str,
        meta: dict[str, Any],
        exit_order: SimulatedOrder,
        exit_reason: str,
    ) -> None:
        if exit_order.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            return
        entry_order: SimulatedOrder = meta["entry_order"]
        entry_price = float(meta["entry_price"])
        exit_price = float(exit_order.filled_price if exit_order.filled_price is not None else exit_order.price)
        qty = int(meta["quantity"])

        b_in, s_in, sl_in, tot_in = _extract_order_costs(entry_order)
        b_out, s_out, sl_out, tot_out = _extract_order_costs(exit_order)

        brokerage = round(b_in + b_out, 4)
        statutory = round(s_in + s_out, 4)
        slippage = round(sl_in + sl_out, 4)
        total_friction = round(tot_in + tot_out, 4)

        gross_pnl = round((exit_price - entry_price) * qty, 2)
        net_pnl = round(gross_pnl - total_friction, 2)
        cost_basis = entry_price * qty
        return_pct = round((net_pnl / cost_basis) * 100.0, 4) if cost_basis > 0 else 0.0
        mae_pct = round(min(0.0, ((float(meta["min_low"]) - entry_price) / entry_price) * 100.0), 4)
        mfe_pct = round(max(0.0, ((float(meta["max_high"]) - entry_price) / entry_price) * 100.0), 4)

        trades.append(
            TradeRecord(
                symbol=sym_u,
                session_date=str(meta["session_date"]),
                side="BUY",
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                quantity=qty,
                gross_pnl_inr=gross_pnl,
                brokerage_inr=brokerage,
                statutory_charges_inr=statutory,
                slippage_inr=slippage,
                total_friction_inr=total_friction,
                net_pnl_inr=net_pnl,
                return_pct=return_pct,
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
                regime_label=str(meta["regime_label"]),
                exit_reason=exit_reason,
            )
        )

    def _make_independent_candle_timing_signal_fn(
        self,
    ) -> Callable[[str, date, datetime, pd.DataFrame, float], dict[str, Any]]:
        """Strategy 1: Stage 2 standalone timing selectivity with fixed BULLISH probe."""
        cfg = self.config
        timing_agent = CandleTimingAgent(
            config=cfg.timing_config,
            intelligence_engine=CandleIntelligenceEngine(min_bars=5),
        )

        def _signal(
            sym_u: str,
            session_d: date,
            as_of_ist: datetime,
            pre_df: pd.DataFrame,
            entry_price: float,
        ) -> dict[str, Any]:
            # Synthetic BULLISH probe used ONLY to exercise Stage 2's independent
            # ENTER_NOW vs WAIT / CONTRADICT_CENTRAL / REJECT gating on real 1h bars.
            probe_hyp = DirectionalHypothesis(
                symbol=sym_u,
                direction=DirectionalBias.BULLISH,
                confidence=0.65,
                prob_next_day_up=0.60,
                expected_next_day_return_pct=0.50,
                news_sentiment_score=0.0,
                market_regime="PROBE_STAGE2_ONLY",
                observation_date=session_d,
                as_of_ist=as_of_ist,
                rationale=(
                    "Fixed BULLISH probe for standalone Stage 2 CandleTimingAgent selectivity evaluation; "
                    "does NOT represent Stage 1 CentralDirectionalAgent predictions.",
                ),
            )
            assessment = timing_agent.evaluate(
                hypothesis=probe_hyp,
                bars_primary=pre_df,
                primary_timeframe="1h",
                bars_15m=None,
                bars_5m=None,
                as_of_ist=as_of_ist,
            )
            enter_long = assessment.decision == TimingDecision.ENTER_NOW
            pred_dir = "BULLISH" if enter_long else "NEUTRAL"
            return {
                "enter_long": enter_long,
                "predicted_direction": pred_dir,
                "decision_label": assessment.decision.value,
                "stop_loss": assessment.invalidation_price,
                "target": assessment.target_price,
            }

        return _signal

    @staticmethod
    def _make_simple_technical_baseline_signal_fn(
        stop_loss_pct: float = 0.015,
        target_pct: float = 0.030,
    ) -> Callable[[str, date, datetime, pd.DataFrame, float], dict[str, Any]]:
        """Strategy 2: 1h EMA(9) > EMA(21) AND prior close > prior 5-bar high breakout."""

        def _signal(
            sym_u: str,
            session_d: date,
            as_of_ist: datetime,
            pre_df: pd.DataFrame,
            entry_price: float,
        ) -> dict[str, Any]:
            if len(pre_df) < 22:
                return {
                    "enter_long": False,
                    "predicted_direction": "NEUTRAL",
                    "decision_label": "INSUFFICIENT_BARS",
                    "stop_loss": None,
                    "target": None,
                }
            closes = pre_df["close"].astype(float)
            ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
            prior_close = float(closes.iloc[-1])
            prior_5_high = float(pre_df["high"].astype(float).iloc[-6:-1].max())

            if ema9 > ema21 and prior_close > prior_5_high:
                return {
                    "enter_long": True,
                    "predicted_direction": "BULLISH",
                    "decision_label": "BREAKOUT_LONG",
                    "stop_loss": round(entry_price * (1.0 - stop_loss_pct), 2),
                    "target": round(entry_price * (1.0 + target_pct), 2),
                }
            return {
                "enter_long": False,
                "predicted_direction": "NEUTRAL",
                "decision_label": "NO_SIGNAL",
                "stop_loss": None,
                "target": None,
            }

        return _signal

    @staticmethod
    def _make_no_trade_baseline_signal_fn() -> (
        Callable[[str, date, datetime, pd.DataFrame, float], dict[str, Any]]
    ):
        """Strategy 3: 0 trades, 100% cash baseline."""

        def _signal(
            sym_u: str,
            session_d: date,
            as_of_ist: datetime,
            pre_df: pd.DataFrame,
            entry_price: float,
        ) -> dict[str, Any]:
            return {
                "enter_long": False,
                "predicted_direction": "NEUTRAL",
                "decision_label": "CASH_HOLD",
                "stop_loss": None,
                "target": None,
            }

        return _signal

    def _run_central_directional_archived_diagnostic(
        self,
        frames: dict[str, pd.DataFrame],
        leakage_tracker: dict[str, int],
    ) -> dict[str, Any]:
        """Strategy 4: Genuine CentralDirectionalAgent on the 3 committed archive sessions."""
        cfg = self.config
        portfolio, risk_engine = self._create_portfolio_and_risk()
        central_agent = CentralDirectionalAgent(cfg.central_config)

        trades: list[TradeRecord] = []
        equity_curve: list[dict[str, Any]] = []
        directional_outcomes: list[dict[str, Any]] = []
        per_session_diagnostics: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="cycle008_central_diag_") as tmp_root:
            for d_str in cfg.archived_diagnostic_sessions:
                session_d = date.fromisoformat(d_str)
                as_of_ist = datetime.combine(session_d, cfg.as_of_time_ist, tzinfo=IST_TZ)
                as_of_ts = pd.Timestamp(as_of_ist)
                snap_dir = Path(tmp_root) / d_str

                portfolio.reset_daily_session(session_d)
                risk_engine.is_daily_halted = False

                try:
                    manifest = reconstruct_as_of_snapshot(
                        session_date=session_d,
                        output_dir=snap_dir,
                        as_of_time_ist=cfg.as_of_time_ist,
                        symbols=cfg.symbols,
                        prediction_log_path=cfg.prediction_log_path,
                        market_regimes_path=cfg.market_regimes_path,
                        forward_predictions_path=cfg.forward_predictions_path,
                        structured_events_path=cfg.structured_events_path,
                    )
                    out = manifest["output_files"]
                    batch = central_agent.evaluate_from_files(
                        symbols=cfg.symbols,
                        as_of_ist=as_of_ist,
                        predictions_path=Path(out["predictions_path"]),
                        intelligence_path=Path(out["intelligence_path"]),
                        regime_path=Path(out["regime_path"]),
                    )
                except Exception as exc:
                    per_session_diagnostics.append(
                        {
                            "session_date": d_str,
                            "status": "BLOCKED_FAIL_CLOSED",
                            "reason": str(exc),
                            "market_regime": None,
                            "signals": {},
                        }
                    )
                    equity_curve.append(
                        {"date": d_str, "equity_inr": round(float(portfolio.total_equity), 2)}
                    )
                    continue

                regime_label = batch.market_regime
                day_by_sym: dict[str, pd.DataFrame] = {}
                open_meta: dict[str, dict[str, Any]] = {}
                session_signals: dict[str, str] = {}

                for sym in cfg.symbols:
                    sym_u = sym.strip().upper()
                    hyp = batch.hypotheses[sym_u]
                    session_signals[sym_u] = hyp.direction.value

                    df = frames[sym_u]
                    pre_df = df[df["timestamp"] < as_of_ts]
                    day_df = df[df["session_date"] == session_d]
                    if not pre_df.empty:
                        leakage_tracker["evaluated_symbol_sessions_checked"] += 1
                        if not (pre_df["timestamp"].max() < as_of_ts):
                            leakage_tracker["lookahead_bar_violations"] += 1
                            raise RuntimeError(f"Lookahead violation in diagnostic for {sym_u} on {d_str}")

                    eligible_bars = day_df[day_df["timestamp"] >= as_of_ts]
                    if eligible_bars.empty:
                        continue
                    day_by_sym[sym_u] = eligible_bars

                    entry_bar = eligible_bars.iloc[0]
                    entry_ts = entry_bar["timestamp"].to_pydatetime()
                    entry_price = float(entry_bar["open"])
                    realized_dir = _realized_session_direction(eligible_bars)
                    traded = False

                    if hyp.direction == DirectionalBias.BULLISH and entry_price > 0:
                        sl_f = round(entry_price * 0.985, 2)
                        tgt_f = round(entry_price * 1.030, 2)
                        risk_per_share = max(entry_price - sl_f, 0.01)
                        req_qty = max(
                            1, int((portfolio.total_equity * cfg.max_risk_per_trade_pct) // risk_per_share)
                        )
                        risk_eval = risk_engine.evaluate_order(
                            symbol=sym_u,
                            side=OrderSide.BUY.value,
                            requested_quantity=req_qty,
                            entry_price=entry_price,
                            stop_loss_price=sl_f,
                            portfolio=portfolio,
                            current_time=entry_ts,
                        )
                        if risk_eval.approved and risk_eval.approved_quantity > 0:
                            buy_order = portfolio.submit_order(
                                symbol=sym_u,
                                side=OrderSide.BUY,
                                quantity=risk_eval.approved_quantity,
                                price=entry_price,
                                stop_loss=sl_f,
                                target=tgt_f,
                                timestamp=entry_ts,
                            )
                            if buy_order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                                traded = True
                                open_meta[sym_u] = {
                                    "entry_order": buy_order,
                                    "entry_price": entry_price,
                                    "quantity": int(buy_order.filled_quantity),
                                    "min_low": entry_price,
                                    "max_high": entry_price,
                                    "regime_label": regime_label,
                                    "session_date": d_str,
                                }

                    directional_outcomes.append(
                        {
                            "symbol": sym_u,
                            "session_date": d_str,
                            "predicted_direction": hyp.direction.value,
                            "realized_direction": realized_dir,
                            "traded": traded,
                            "regime_label": regime_label,
                        }
                    )

                # Bar-by-bar intraday exit simulation
                all_ts = sorted(
                    {ts.to_pydatetime() for df in day_by_sym.values() for ts in df["timestamp"]}
                )
                latest_prices: dict[str, float] = {}
                for ts in all_ts:
                    for sym_u, day_df in day_by_sym.items():
                        row_df = day_df[day_df["timestamp"] == pd.Timestamp(ts)]
                        if row_df.empty:
                            continue
                        row = row_df.iloc[0]
                        bar_open = float(row["open"])
                        bar_high = float(row["high"])
                        bar_low = float(row["low"])
                        bar_close = float(row["close"])
                        latest_prices[sym_u] = bar_close

                        if sym_u not in portfolio.positions or sym_u not in open_meta:
                            continue
                        pos = portfolio.positions[sym_u]
                        meta = open_meta[sym_u]
                        meta["min_low"] = min(meta["min_low"], bar_low)
                        meta["max_high"] = max(meta["max_high"], bar_high)

                        exit_orders: list[SimulatedOrder] = []
                        exit_reason: str | None = None
                        if pos.stop_loss is not None and bar_low <= pos.stop_loss:
                            exit_orders = portfolio.update_market_price(
                                sym_u, min(bar_open, pos.stop_loss), ts
                            )
                            exit_reason = "STOP_LOSS"
                        elif pos.target is not None and bar_high >= pos.target:
                            exit_orders = portfolio.update_market_price(sym_u, pos.target, ts)
                            exit_reason = "TARGET"
                        else:
                            exit_orders = portfolio.update_market_price(sym_u, bar_close, ts)
                            if exit_orders:
                                exit_reason = "INTRADAY_AUTO_EXIT"

                        if exit_orders and exit_reason is not None:
                            self._append_closed_trade_record(
                                trades=trades,
                                sym_u=sym_u,
                                meta=meta,
                                exit_order=exit_orders[-1],
                                exit_reason=exit_reason,
                            )
                            open_meta.pop(sym_u, None)

                if portfolio.positions and all_ts:
                    sq_orders = portfolio.square_off_all_positions(latest_prices, all_ts[-1])
                    for so in sq_orders:
                        sym_u = so.symbol.strip().upper()
                        meta = open_meta.pop(sym_u, None)
                        if meta is not None:
                            self._append_closed_trade_record(
                                trades=trades,
                                sym_u=sym_u,
                                meta=meta,
                                exit_order=so,
                                exit_reason="EOD_SQUARE_OFF",
                            )

                equity_curve.append(
                    {"date": d_str, "equity_inr": round(float(portfolio.total_equity), 2)}
                )
                per_session_diagnostics.append(
                    {
                        "session_date": d_str,
                        "status": "COMPLETED",
                        "reason": None,
                        "market_regime": regime_label,
                        "signals": session_signals,
                    }
                )

        ledger_ok, ledger_msg = portfolio.verify_ledger_integrity()
        metrics = compute_strategy_metrics(
            trades=trades,
            equity_curve=equity_curve,
            directional_outcomes=directional_outcomes,
            initial_capital_inr=cfg.initial_capital_inr,
        )
        return {
            "strategy_id": "central_directional_agent_archived_diagnostic",
            "evaluation_scope": "EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY",
            "verdict": "INSUFFICIENT_EVIDENCE",
            "sessions_requested": list(cfg.archived_diagnostic_sessions),
            "sessions_completed": sum(1 for s in per_session_diagnostics if s["status"] == "COMPLETED"),
            "sessions_blocked": sum(1 for s in per_session_diagnostics if s["status"] != "COMPLETED"),
            "per_session_diagnostics": per_session_diagnostics,
            "metrics": metrics,
            "ledger_reconciliation": {
                "sha256_chain_valid": bool(ledger_ok),
                "verification_message": ledger_msg,
                "ledger_event_count": len(portfolio.event_ledger),
            },
            "note": (
                "Evaluates genuine CentralDirectionalAgent strictly on the 3 committed archive "
                "sessions (all inside the exposed holdout window). Cannot be used to claim "
                "unseen OOS generalization."
            ),
        }

    def _run_combined_cycle007_archived_diagnostic(
        self,
        frames: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Strategy 5: Genuine Cycle007Pipeline on the 3 committed archive sessions."""
        cfg = self.config
        trades: list[TradeRecord] = []
        directional_outcomes: list[dict[str, Any]] = []
        per_session_diagnostics: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        running_equity = float(cfg.initial_capital_inr)
        all_ledgers_valid = True

        with tempfile.TemporaryDirectory(prefix="cycle008_c007_diag_") as tmp_root:
            for d_str in cfg.archived_diagnostic_sessions:
                session_d = date.fromisoformat(d_str)
                as_of_ist = datetime.combine(session_d, cfg.as_of_time_ist, tzinfo=IST_TZ)
                as_of_ts = pd.Timestamp(as_of_ist)
                snap_dir = Path(tmp_root) / d_str

                c007_cfg = Cycle007Config(
                    session_date=session_d,
                    symbols=cfg.symbols,
                    initial_capital_inr=running_equity,
                    slippage_bps=cfg.slippage_bps,
                    as_of_time_ist=cfg.as_of_time_ist,
                    min_bars_per_session=cfg.min_bars_per_session,
                    cache_dir=cfg.cache_dir,
                    snapshot_dir=snap_dir,
                    holdout_registry_path=cfg.holdout_registry_path,
                    central_config=cfg.central_config,
                    timing_config=cfg.timing_config,
                    max_risk_per_trade_pct=cfg.max_risk_per_trade_pct,
                    max_position_weight_pct=cfg.max_position_weight_pct,
                    max_portfolio_exposure_pct=cfg.max_portfolio_exposure_pct,
                    max_daily_loss_pct=cfg.max_daily_loss_pct,
                    max_concurrent_positions=cfg.max_concurrent_positions,
                )
                pipeline = Cycle007Pipeline(c007_cfg)
                rep = pipeline.run()

                if not rep.get("ledger_reconciliation", {}).get("sha256_chain_valid", False):
                    all_ledgers_valid = False

                run_status = str(rep.get("run_status", "UNKNOWN"))
                regime_label = str(rep.get("market_regime") or "UNKNOWN")

                # Extract directional outcomes & closed trades from the genuine Cycle007Pipeline run
                orders = rep.get("orders", [])
                filled_buys_by_sym: dict[str, dict[str, Any]] = {}
                filled_sells_by_sym: dict[str, dict[str, Any]] = {}
                for o in orders:
                    if o.get("status") not in ("FILLED", "PARTIALLY_FILLED"):
                        continue
                    s_u = str(o.get("symbol", "")).strip().upper()
                    if o.get("side") == "BUY":
                        filled_buys_by_sym[s_u] = o
                    elif o.get("side") == "SELL":
                        filled_sells_by_sym[s_u] = o

                for row in rep.get("per_symbol_decisions", []):
                    sym_u = str(row["symbol"]).strip().upper()
                    s1_dir = str(row["stage1_central"]["direction"]).upper()
                    s2_dec = str(row["stage2_timing"]["decision"]).upper()
                    df = frames[sym_u]
                    day_df = df[(df["session_date"] == session_d) & (df["timestamp"] >= as_of_ts)]
                    realized_dir = _realized_session_direction(day_df)
                    traded = sym_u in filled_buys_by_sym
                    directional_outcomes.append(
                        {
                            "symbol": sym_u,
                            "session_date": d_str,
                            "predicted_direction": s1_dir if s2_dec == "ENTER_NOW" else "NEUTRAL",
                            "realized_direction": realized_dir,
                            "traded": traded,
                            "regime_label": regime_label,
                        }
                    )

                    if sym_u in filled_buys_by_sym and sym_u in filled_sells_by_sym:
                        buy_o = filled_buys_by_sym[sym_u]
                        sell_o = filled_sells_by_sym[sym_u]
                        entry_px = float(buy_o.get("filled_price") or buy_o.get("price") or 0.0)
                        exit_px = float(sell_o.get("filled_price") or sell_o.get("price") or 0.0)
                        qty = int(buy_o.get("filled_quantity") or 0)
                        b_in, s_in, sl_in, tot_in = _extract_dict_order_costs(buy_o)
                        b_out, s_out, sl_out, tot_out = _extract_dict_order_costs(sell_o)
                        total_fric = round(tot_in + tot_out, 4)
                        gross_pnl = round((exit_px - entry_px) * qty, 2)
                        net_pnl = round(gross_pnl - total_fric, 2)
                        basis = entry_px * qty
                        ret_pct = round((net_pnl / basis) * 100.0, 4) if basis > 0 else 0.0
                        trade_df = day_df
                        sell_filled_at = sell_o.get("filled_at")
                        if sell_filled_at and not day_df.empty:
                            exit_ts = pd.Timestamp(sell_filled_at)
                            if exit_ts.tzinfo is None:
                                exit_ts = exit_ts.tz_localize(IST_TZ)
                            else:
                                exit_ts = exit_ts.tz_convert(IST_TZ)
                            held_df = day_df[day_df["timestamp"] <= exit_ts]
                            if not held_df.empty:
                                trade_df = held_df
                        min_low = float(trade_df["low"].min()) if not trade_df.empty else entry_px
                        max_high = float(trade_df["high"].max()) if not trade_df.empty else entry_px
                        mae_pct = round(min(0.0, ((min_low - entry_px) / entry_px) * 100.0), 4) if entry_px > 0 else 0.0
                        mfe_pct = round(max(0.0, ((max_high - entry_px) / entry_px) * 100.0), 4) if entry_px > 0 else 0.0
                        trades.append(
                            TradeRecord(
                                symbol=sym_u,
                                session_date=d_str,
                                side="BUY",
                                entry_price=round(entry_px, 4),
                                exit_price=round(exit_px, 4),
                                quantity=qty,
                                gross_pnl_inr=gross_pnl,
                                brokerage_inr=round(b_in + b_out, 4),
                                statutory_charges_inr=round(s_in + s_out, 4),
                                slippage_inr=round(sl_in + sl_out, 4),
                                total_friction_inr=total_fric,
                                net_pnl_inr=net_pnl,
                                return_pct=ret_pct,
                                mae_pct=mae_pct,
                                mfe_pct=mfe_pct,
                                regime_label=regime_label,
                                exit_reason="CYCLE007_PIPELINE_EXIT",
                            )
                        )

                ending_eq = float(rep.get("portfolio_results", {}).get("ending_equity_inr", running_equity))
                running_equity = ending_eq
                equity_curve.append({"date": d_str, "equity_inr": round(running_equity, 2)})

                per_session_diagnostics.append(
                    {
                        "session_date": d_str,
                        "run_status": run_status,
                        "market_regime": rep.get("market_regime"),
                        "order_count": int(rep.get("order_count", 0)),
                        "filled_order_count": int(rep.get("filled_order_count", 0)),
                        "closed_trades_count": int(rep.get("closed_trades_count", 0)),
                        "net_pnl_inr": float(rep.get("portfolio_results", {}).get("net_pnl_inr", 0.0)),
                        "rejection_reasons": list(rep.get("rejection_reasons", [])),
                    }
                )

        metrics = compute_strategy_metrics(
            trades=trades,
            equity_curve=equity_curve,
            directional_outcomes=directional_outcomes,
            initial_capital_inr=cfg.initial_capital_inr,
        )
        return {
            "strategy_id": "combined_cycle007_pipeline_archived_diagnostic",
            "evaluation_scope": "EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY",
            "verdict": "INSUFFICIENT_EVIDENCE",
            "sessions_requested": list(cfg.archived_diagnostic_sessions),
            "sessions_completed": sum(
                1 for s in per_session_diagnostics if s["run_status"] == "COMPLETED"
            ),
            "sessions_blocked": sum(
                1 for s in per_session_diagnostics if s["run_status"] != "COMPLETED"
            ),
            "per_session_diagnostics": per_session_diagnostics,
            "metrics": metrics,
            "ledger_reconciliation": {
                "all_session_sha256_chains_valid": all_ledgers_valid,
            },
            "note": (
                "Evaluates genuine Cycle007Pipeline (Stage 1 + Stage 2 + Risk + Ledger) strictly "
                "on the 3 committed archive sessions (all inside the exposed holdout window). "
                "Cannot be used to claim unseen OOS generalization."
            ),
        }

    def run(self) -> dict[str, Any]:
        """Execute full CYCLE-008 historical walk-forward evaluation under OfflineExecutionGuard."""
        started_at = datetime.now(tz=IST_TZ)
        cfg = self.config
        reg_path = cfg.holdout_registry_path
        if not reg_path.exists():
            raise FileNotFoundError(f"Required holdout registry missing: {reg_path}")
        holdout_sha_before = sha256_file(reg_path)

        leakage_tracker = {
            "evaluated_symbol_sessions_checked": 0,
            "lookahead_bar_violations": 0,
        }

        with OfflineExecutionGuard(self.instrumentation):
            # Phase 1: Historical Data Readiness & Provenance Audit
            data_readiness = audit_historical_data_readiness(
                symbols=cfg.symbols,
                cache_dir_1h=cfg.cache_dir,
                prediction_log_path=cfg.prediction_log_path,
                forward_predictions_path=cfg.forward_predictions_path,
                market_regimes_path=cfg.market_regimes_path,
                structured_events_path=cfg.structured_events_path,
                holdout_registry_path=cfg.holdout_registry_path,
            )

            holdout_audit = data_readiness["holdout_registry_audit"]
            if not holdout_audit.get("exists", False):
                raise FileNotFoundError(f"Required holdout registry missing: {reg_path}")
            non_holdout_dates: list[str] = list(holdout_audit["non_holdout_dates"])
            exposed_holdout_dates_set = set(holdout_audit["exposed_holdout_dates"])

            # Phase 2: Strictly Chronological 3-Way Split (train / validation / dev_test_oos)
            splits = build_chronological_splits(
                non_holdout_dates=non_holdout_dates,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
            )

            # Leakage guard: verify zero exposed holdout dates and zero split overlap
            all_wf_dates = splits["train"] + splits["validation"] + splits["dev_test_oos"]
            exposed_overlap = sorted(set(all_wf_dates) & exposed_holdout_dates_set)
            split_overlap_count = (
                len(set(splits["train"]) & set(splits["validation"]))
                + len(set(splits["train"]) & set(splits["dev_test_oos"]))
                + len(set(splits["validation"]) & set(splits["dev_test_oos"]))
            )
            if exposed_overlap:
                raise RuntimeError(
                    f"Holdout leakage detected in walk-forward splits: {exposed_overlap}"
                )
            if split_overlap_count != 0:
                raise RuntimeError(f"Split overlap detected: {split_overlap_count}")

            # Load validated 1h frames once
            frames = _load_validated_1h_frames(cfg.symbols, cfg.cache_dir)

            # Strategy 1: independent_candle_timing_agent
            s1_fn = self._make_independent_candle_timing_signal_fn()
            s1_dev_test = self._simulate_strategy_on_dates(
                dates_iso=splits["dev_test_oos"],
                frames=frames,
                signal_fn=s1_fn,
                leakage_tracker=leakage_tracker,
                enforce_timing_gap_rules=True,
            )
            s1_full_wf = self._simulate_strategy_on_dates(
                dates_iso=non_holdout_dates,
                frames=frames,
                signal_fn=s1_fn,
                leakage_tracker=leakage_tracker,
                enforce_timing_gap_rules=True,
            )

            # Strategy 2: simple_technical_baseline
            s2_fn = self._make_simple_technical_baseline_signal_fn()
            s2_dev_test = self._simulate_strategy_on_dates(
                dates_iso=splits["dev_test_oos"],
                frames=frames,
                signal_fn=s2_fn,
                leakage_tracker=leakage_tracker,
                enforce_timing_gap_rules=False,
            )
            s2_full_wf = self._simulate_strategy_on_dates(
                dates_iso=non_holdout_dates,
                frames=frames,
                signal_fn=s2_fn,
                leakage_tracker=leakage_tracker,
                enforce_timing_gap_rules=False,
            )

            # Strategy 3: no_trade_baseline
            s3_fn = self._make_no_trade_baseline_signal_fn()
            s3_dev_test = self._simulate_strategy_on_dates(
                dates_iso=splits["dev_test_oos"],
                frames=frames,
                signal_fn=s3_fn,
                leakage_tracker=leakage_tracker,
                enforce_timing_gap_rules=False,
            )
            s3_full_wf = self._simulate_strategy_on_dates(
                dates_iso=non_holdout_dates,
                frames=frames,
                signal_fn=s3_fn,
                leakage_tracker=leakage_tracker,
                enforce_timing_gap_rules=False,
            )

            # Strategy 4: central_directional_agent_archived_diagnostic
            s4_diag = self._run_central_directional_archived_diagnostic(
                frames=frames,
                leakage_tracker=leakage_tracker,
            )

            # Strategy 5: combined_cycle007_pipeline_archived_diagnostic
            s5_diag = self._run_combined_cycle007_archived_diagnostic(frames=frames)

        completed_at = datetime.now(tz=IST_TZ)
        holdout_sha_after = sha256_file(reg_path) if reg_path.exists() else None
        holdout_unchanged = (
            holdout_sha_before is not None and holdout_sha_before == holdout_sha_after
        )

        execution_provenance = _collect_cycle008_execution_provenance()

        insufficient_evidence_reasons = [
            "STAGE1_HISTORICAL_ARCHIVE_ONLY_3_OBSERVATION_DATES",
            "ZERO_AVAILABLE_SEALED_HOLDOUT_DATES_48_EXPOSED",
            "MARKET_DATA_1H_PROVENANCE_PARTIALLY_VERIFIED_YAHOO_FINANCE",
            "COMMITTED_15M_AND_5M_CANDLES_UNAVAILABLE",
            "DIRECTIONAL_CONFIDENCE_UNCALIBRATED",
        ]

        data_leakage_audit = {
            "exposed_holdout_dates_in_walk_forward": len(exposed_overlap),
            "lookahead_bar_violations": int(leakage_tracker["lookahead_bar_violations"]),
            "evaluated_symbol_sessions_checked": int(
                leakage_tracker["evaluated_symbol_sessions_checked"]
            ),
            "strict_pre_cutoff_rule": "max_feature_timestamp < as_of_ist (09:15 IST)",
            "parameter_tuning_on_dev_test_oos": 0,
            "synthetic_candles_used": 0,
            "fabricated_timeframes": 0,
            "strictly_chronological_splits": True,
            "split_overlap_count": split_overlap_count,
            "passed": (
                len(exposed_overlap) == 0
                and leakage_tracker["lookahead_bar_violations"] == 0
                and split_overlap_count == 0
            ),
        }

        split_summary = {
            name: {
                "date_count": len(d_list),
                "date_min": d_list[0] if d_list else None,
                "date_max": d_list[-1] if d_list else None,
            }
            for name, d_list in splits.items()
        }

        report: dict[str, Any] = {
            "cycle_id": "CYCLE-008",
            "phase": "HISTORICAL_OUT_OF_SAMPLE_EVALUATION_AND_STRATEGY_VALIDATION",
            "run_mode": "HISTORICAL_WALK_FORWARD_EVALUATION_OFFLINE",
            "run_status": "COMPLETED",
            "overall_classification": "INSUFFICIENT_EVIDENCE",
            "insufficient_evidence_reasons": insufficient_evidence_reasons,
            "confidence_calibration_status": CONFIDENCE_CALIBRATION_STATUS,
            "confidence_calibration_note": CONFIDENCE_CALIBRATION_NOTE,
            "qualification_allowed": False,
            "unseen_evaluation_status": "NOT_EVALUATED",
            "unseen_return_pct": None,
            "live_trading_enabled": False,
            "real_money_trading": False,
            "visual_ai_enabled": False,
            "synthetic_candles_used": 0,
            "broker_connections": self.instrumentation.broker_connections,
            "external_api_calls": self.instrumentation.external_api_calls,
            "paid_api_cost": self.instrumentation.paid_api_cost_usd,
            "instrumentation": {
                "network_guard_active": self.instrumentation.network_guard_active,
                "network_guard_scope": self.instrumentation.network_guard_scope,
                "intercepted_network_attempts": self.instrumentation.external_api_calls,
                "blocked_network_attempts": list(self.instrumentation.blocked_network_attempts),
                "broker_adapters_configured": self.instrumentation.broker_adapters_configured,
                "broker_connections_verification": self.instrumentation.broker_connections_verification,
                "paid_llm_providers_configured": self.instrumentation.paid_llm_providers_configured,
                "paid_api_cost_verification": self.instrumentation.paid_api_cost_verification,
                "market_data_generation_mode": "READ_ONLY_COMMITTED_1H_PARQUET_NO_SYNTHETIC_CANDLES",
            },
            "started_at_ist": started_at.isoformat(),
            "completed_at_ist": completed_at.isoformat(),
            "code_commit_sha": execution_provenance["implementation_commit_sha"],
            "execution_provenance": execution_provenance,
            "config": {
                "symbols": list(cfg.symbols),
                "initial_capital_inr": cfg.initial_capital_inr,
                "slippage_bps": cfg.slippage_bps,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "min_history_bars": cfg.min_history_bars,
                "min_bars_per_session": cfg.min_bars_per_session,
                "as_of_time_ist": cfg.as_of_time_ist.isoformat(),
                "archived_diagnostic_sessions": list(cfg.archived_diagnostic_sessions),
            },
            "phase1_data_readiness": data_readiness,
            "phase2_walk_forward_splits": {
                "non_holdout_total_days": len(non_holdout_dates),
                "exposed_holdout_excluded_days": len(exposed_holdout_dates_set),
                "splits": split_summary,
                "regime_proxy_note": (
                    "Non-holdout walk-forward regime breakdowns use a strictly pre-cutoff 1h "
                    "20-bar universe median return proxy because committed market_regimes.parquet "
                    "covers only 3 archive dates (2026-11-16..2026-11-18)."
                ),
            },
            "data_leakage_audit": data_leakage_audit,
            "strategies": {
                "independent_candle_timing_agent": {
                    "strategy_id": "independent_candle_timing_agent",
                    "evaluation_scope": "NON_HOLDOUT_WALK_FORWARD_STAGE2_SELECTIVITY",
                    "strategy_note": (
                        "Evaluates Stage 2 (CandleIntelligenceEngine + CandleTimingAgent) "
                        "standalone timing selectivity on real 1h bars using a fixed BULLISH probe "
                        "to test ENTER_NOW vs WAIT / CONTRADICT_CENTRAL / REJECT gating. "
                        "Does NOT evaluate Stage 1 CentralDirectionalAgent."
                    ),
                    "dev_test_oos": s1_dev_test,
                    "full_non_holdout": s1_full_wf,
                },
                "simple_technical_baseline": {
                    "strategy_id": "simple_technical_baseline",
                    "evaluation_scope": "NON_HOLDOUT_WALK_FORWARD_BASELINE",
                    "strategy_note": (
                        "Deterministic 1h EMA(9) > EMA(21) and prior close > prior 5-bar high "
                        "breakout baseline with 1.5% SL / 3.0% target under identical "
                        "DeterministicRiskEngine + IntradayPaperPortfolio."
                    ),
                    "dev_test_oos": s2_dev_test,
                    "full_non_holdout": s2_full_wf,
                },
                "no_trade_baseline": {
                    "strategy_id": "no_trade_baseline",
                    "evaluation_scope": "NON_HOLDOUT_WALK_FORWARD_CASH_BASELINE",
                    "strategy_note": "Zero-trade, 100% cash baseline (constant ₹50,000 equity).",
                    "dev_test_oos": s3_dev_test,
                    "full_non_holdout": s3_full_wf,
                },
                "central_directional_agent_archived_diagnostic": s4_diag,
                "combined_cycle007_pipeline_archived_diagnostic": s5_diag,
            },
            "holdout_registry_integrity": {
                "path": str(reg_path),
                "sha256_before": holdout_sha_before,
                "sha256_after": holdout_sha_after,
                "unchanged": holdout_unchanged,
            },
        }

        canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        report["output_report_sha256"] = hashlib.sha256(canonical).hexdigest()
        return report
