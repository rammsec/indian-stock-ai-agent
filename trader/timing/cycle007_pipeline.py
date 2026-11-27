"""CYCLE-007 — Fail-Closed Paper-Trading Integration Pipeline (Numerical Phase).

Connects:
    CentralDirectionalAgent (Stage 1)
    -> CandleTimingAgent (Stage 2, independent numerical candlestick evaluation)
    -> DeterministicRiskEngine (Hard-coded risk gatekeeper)
    -> IntradayPaperPortfolio (Double-entry paper ledger with SHA-256 chain)

Runs strictly inside ``OfflineExecutionGuard`` with zero network/API calls,
zero live broker connections, zero visual/multimodal AI calls, zero synthetic
runtime candles, and dynamic before/after SHA-256 verification of
``data/lab/sealed_holdout_registry.json``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, LabRiskConfig
from trader.data.provenance_1h import validate_1h_provenance
from trader.paper.cycle006b import (
    DEFAULT_CACHE_DIR,
    ExecutionInstrumentation,
    OfflineExecutionGuard,
    sha256_file,
)
from trader.paper.cycle006c_snapshot import (
    classify_timeframe_provenance,
    reconstruct_as_of_snapshot,
)
from trader.portfolio.cost_calculator import CostConfig
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from trader.risk.engine import DeterministicRiskEngine, RiskEvaluationResult
from trader.timing.candle_intelligence import CandleIntelligenceEngine
from trader.timing.candle_timing_agent import (
    CandleTimingAgent,
    CandleTimingConfig,
    TimingAssessment,
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

if TYPE_CHECKING:
    from trader.authority.gate import MandatoryExecutionGate
    from trader.authority.instructions import AuthorityInstruction, PermissionType

DEFAULT_HOLDOUT_REGISTRY_PATH = Path("data/lab/sealed_holdout_registry.json")
DEFAULT_CYCLE007_SNAPSHOT_DIR = Path("/tmp/cycle_007_snapshot")
REPORT_COMMIT_DISTINCTION = (
    "implementation_commit_sha identifies the committed source revision executed to produce "
    "this report; the subsequent commit recording updated JSON/Markdown report files is "
    "necessarily a descendant commit."
)
CYCLE007_IMPLEMENTATION_MODULES: tuple[str, ...] = (
    "trader/timing/__init__.py",
    "trader/timing/candle_intelligence.py",
    "trader/timing/central_directional_agent.py",
    "trader/timing/candle_timing_agent.py",
    "trader/timing/cycle007_pipeline.py",
    "scripts/run_cycle_007_timing_dry_run.py",
)


def _collect_execution_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Capture git revision, cleanliness state, and SHA-256 hashes of executed CYCLE-007 code."""
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
        dirty_paths = [
            line[3:].strip()
            for line in raw_status.splitlines()
            if line.strip()
        ]
        working_tree_clean = len(dirty_paths) == 0
    except Exception:
        working_tree_clean = False
        dirty_paths = ["GIT_STATUS_UNAVAILABLE"]

    implementation_file_hashes: dict[str, str] = {}
    for rel_path in CYCLE007_IMPLEMENTATION_MODULES:
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
class Cycle007Config:
    """Configuration for CYCLE-007 fail-closed numerical timing dry-run pipeline."""

    session_date: date
    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE
    initial_capital_inr: float = LabRiskConfig().initial_capital_inr
    slippage_bps: float = 5.0
    as_of_time_ist: dt_time = dt_time(9, 15)
    min_bars_per_session: int = 6
    min_history_bars: int = 10
    cache_dir: Path = DEFAULT_CACHE_DIR
    predictions_path: Path | None = None
    intelligence_path: Path | None = None
    regime_path: Path | None = None
    snapshot_dir: Path = DEFAULT_CYCLE007_SNAPSHOT_DIR
    holdout_registry_path: Path = DEFAULT_HOLDOUT_REGISTRY_PATH
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
        if self.min_bars_per_session < 1:
            raise ValueError("min_bars_per_session must be >= 1")
        if self.min_history_bars < 5:
            raise ValueError("min_history_bars must be >= 5")
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        norm = [s.strip().upper() for s in self.symbols if isinstance(s, str) and s.strip()]
        if len(norm) != len(self.symbols) or len(set(norm)) != len(norm):
            raise ValueError("symbols must be non-empty, unique instrument identifiers")


class Cycle007Pipeline:
    """Deterministic, fail-closed CYCLE-007 paper-trading dry-run pipeline."""

    def __init__(
        self,
        config: Cycle007Config,
        *,
        central_agent: CentralDirectionalAgent | None = None,
        timing_agent: CandleTimingAgent | None = None,
        risk_engine: DeterministicRiskEngine | None = None,
        execution_gate: MandatoryExecutionGate | None = None,
        human_instruction: AuthorityInstruction | None = None,
        human_permission: PermissionType | None = None,
    ) -> None:
        from trader.authority.gate import MandatoryExecutionGate as _MandatoryExecutionGate
        from trader.authority.instructions import (
            AuthorityInstruction as _AuthorityInstruction,
            AuthoritySource as _AuthoritySource,
        )

        self.config = config
        self.as_of_ist = datetime.combine(
            config.session_date, config.as_of_time_ist, tzinfo=IST_TZ
        )
        self.central_agent = central_agent or CentralDirectionalAgent(config.central_config)
        self.timing_agent = timing_agent or CandleTimingAgent(
            config=config.timing_config,
            intelligence_engine=CandleIntelligenceEngine(min_bars=min(5, config.min_history_bars)),
        )
        if execution_gate is not None:
            self.execution_gate = execution_gate
            self.risk_engine = risk_engine if risk_engine is not None else execution_gate.risk_engine
        else:
            self.risk_engine = risk_engine or DeterministicRiskEngine(
                max_risk_per_trade_pct=config.max_risk_per_trade_pct,
                max_position_weight_pct=config.max_position_weight_pct,
                max_portfolio_exposure_pct=config.max_portfolio_exposure_pct,
                max_daily_loss_pct=config.max_daily_loss_pct,
                max_concurrent_positions=config.max_concurrent_positions,
            )
            self.execution_gate = _MandatoryExecutionGate(risk_engine=self.risk_engine)

        # PY-04 Strict Fail-Closed: NEVER fabricate an implicit ALLOW_LONG Human instruction.
        session_open_ist = datetime.combine(config.session_date, dt_time(9, 15), tzinfo=IST_TZ)
        session_close_ist = datetime.combine(config.session_date, dt_time(15, 30), tzinfo=IST_TZ)
        if human_instruction is not None:
            self.execution_gate.set_human_instruction(human_instruction, current_time=session_open_ist)
        elif human_permission is not None:
            h_instr = _AuthorityInstruction(
                instruction_id=f"HUMAN-CYCLE007-{config.session_date.isoformat()}",
                authority=_AuthoritySource.HUMAN,
                scope="*",
                permission=human_permission,
                valid_from=session_open_ist - timedelta(minutes=15),
                expires_at=session_close_ist + timedelta(minutes=15),
                reason=f"Explicit caller session authority: {human_permission.value}",
            )
            self.execution_gate.set_human_instruction(h_instr, current_time=session_open_ist)

        self.portfolio = IntradayPaperPortfolio(
            initial_capital=config.initial_capital_inr,
            cost_config=CostConfig(base_slippage_bps=config.slippage_bps),
        )
        self.instrumentation = ExecutionInstrumentation()

    def _load_and_split_1h_archives(
        self,
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, pd.DataFrame],
        dict[str, Any],
        dict[str, str],
    ]:
        """Load committed 1h parquet archives and split strictly into pre-cutoff & session frames."""
        cfg = self.config
        prov_verified, prov_report = validate_1h_provenance(
            symbols=cfg.symbols,
            cache_dir=cfg.cache_dir,
            min_bars_per_session=cfg.min_bars_per_session,
        )
        prov_status = str(prov_report.get("final_provenance_status", "FAILED"))
        if prov_status not in ("VERIFIED", "PARTIALLY_VERIFIED"):
            raise ValueError(
                f"Historical 1h market data provenance rejected: status={prov_status}"
            )

        pre_cutoff_frames: dict[str, pd.DataFrame] = {}
        session_frames: dict[str, pd.DataFrame] = {}
        per_symbol_meta: dict[str, Any] = {}
        file_hashes: dict[str, str] = {}

        for sym in cfg.symbols:
            sym_norm = sym.strip().upper()
            pq_path = cfg.cache_dir / f"{sym_norm}_1h.parquet"
            if not pq_path.exists():
                raise FileNotFoundError(f"Missing 1h historical archive for {sym}: {pq_path}")
            file_hashes[str(pq_path)] = sha256_file(pq_path)

            df = pd.read_parquet(pq_path)
            req_cols = {"timestamp", "open", "high", "low", "close", "volume"}
            missing_cols = req_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing OHLCV columns for {sym}: {sorted(missing_cols)}")
            if df.empty:
                raise ValueError(f"Empty 1h parquet archive for {sym}")

            ts_series = pd.to_datetime(df["timestamp"])
            if ts_series.isna().any():
                raise ValueError(f"NaT/null timestamps detected in {sym}_1h.parquet")
            if ts_series.dt.tz is None:
                raise ValueError(f"Timezone-naive timestamps rejected in {sym}_1h.parquet")
            ts_ist = ts_series.dt.tz_convert(IST_TZ)

            if ts_ist.duplicated().any():
                raise ValueError(f"Duplicate timestamps detected in {sym}_1h.parquet")
            if not ts_ist.is_monotonic_increasing:
                raise ValueError(f"Non-monotonic timestamps detected in {sym}_1h.parquet")

            work = df.copy()
            work["timestamp"] = ts_ist

            # Strict zero-lookahead split:
            # - pre_cutoff_df contains ONLY completed bars strictly before as_of_ist
            # - day_df contains session_date bars for simulated execution & MTM
            pre_df = work[work["timestamp"] < pd.Timestamp(self.as_of_ist)].copy()
            day_df = work[work["timestamp"].dt.date == cfg.session_date].copy()

            if len(pre_df) < cfg.min_history_bars:
                raise ValueError(
                    f"Insufficient pre-cutoff 1h history for {sym} before {self.as_of_ist.isoformat()}: "
                    f"got {len(pre_df)} bars, required >= {cfg.min_history_bars}"
                )
            if len(day_df) < cfg.min_bars_per_session:
                raise ValueError(
                    f"Incomplete 1h session for {sym} on {cfg.session_date}: "
                    f"got {len(day_df)} bars, required >= {cfg.min_bars_per_session}"
                )

            # Validate OHLCV integrity on session bars as well (pre_df is also validated by CandleIntelligenceEngine)
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            if day_df[ohlcv_cols].isna().any().any():
                raise ValueError(f"NaN OHLCV values detected for {sym} on {cfg.session_date}")
            arr = day_df[ohlcv_cols].to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite OHLCV values detected for {sym} on {cfg.session_date}")
            if (arr[:, :4] <= 0).any():
                raise ValueError(f"Non-positive OHLC prices detected for {sym} on {cfg.session_date}")
            if (arr[:, 4] < 0).any():
                raise ValueError(f"Negative volume detected for {sym} on {cfg.session_date}")
            if (day_df["high"] < day_df[["open", "close", "low"]].max(axis=1)).any():
                raise ValueError(f"OHLC high invariant violated for {sym} on {cfg.session_date}")
            if (day_df["low"] > day_df[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError(f"OHLC low invariant violated for {sym} on {cfg.session_date}")

            pre_cutoff_frames[sym] = pre_df.reset_index(drop=True)
            session_frames[sym] = day_df.reset_index(drop=True)
            per_symbol_meta[sym] = {
                "file": pq_path.name,
                "sha256": file_hashes[str(pq_path)],
                "pre_cutoff_bars": int(len(pre_df)),
                "last_pre_cutoff_candle_ist": pre_df["timestamp"].iloc[-1].isoformat(),
                "session_bars": int(len(day_df)),
                "first_session_candle_ist": day_df["timestamp"].iloc[0].isoformat(),
                "latest_session_candle_ist": day_df["timestamp"].iloc[-1].isoformat(),
                "timezone": str(day_df["timestamp"].dt.tz),
            }

        timeframe_prov = classify_timeframe_provenance(
            session_date=cfg.session_date,
            symbols=cfg.symbols,
            cache_dir_1h=cfg.cache_dir,
            cache_dir_5m=None,
        )
        market_meta = {
            "data_classification": "HISTORICAL_1H_PARQUET_ARCHIVE_NOT_LIVE",
            "provenance_status": prov_status,
            "provenance_verified": bool(prov_verified),
            "dataset_fingerprint": prov_report.get("dataset_fingerprint"),
            "session_date": cfg.session_date.isoformat(),
            "timeframe_provenance": timeframe_prov,
            "symbols": per_symbol_meta,
        }
        return pre_cutoff_frames, session_frames, market_meta, file_hashes

    def _resolve_snapshot_paths(self) -> tuple[Path, Path, Path, dict[str, Any] | None]:
        """Use caller-provided snapshot paths or reconstruct strictly pre-cutoff snapshot."""
        cfg = self.config
        if (
            cfg.predictions_path is not None
            and cfg.intelligence_path is not None
            and cfg.regime_path is not None
        ):
            return cfg.predictions_path, cfg.intelligence_path, cfg.regime_path, None

        manifest = reconstruct_as_of_snapshot(
            session_date=cfg.session_date,
            output_dir=cfg.snapshot_dir,
            as_of_time_ist=cfg.as_of_time_ist,
            symbols=cfg.symbols,
        )
        out = manifest["output_files"]
        return (
            Path(out["predictions_path"]),
            Path(out["intelligence_path"]),
            Path(out["regime_path"]),
            manifest,
        )

    def run(self) -> dict[str, Any]:
        """Execute CYCLE-007 fail-closed paper-trading dry run under OfflineExecutionGuard."""
        started_at = datetime.now(tz=IST_TZ)
        cfg = self.config

        reg_path = cfg.holdout_registry_path
        holdout_sha_before = sha256_file(reg_path) if reg_path.exists() else None

        rejection_reasons: list[str] = []
        risk_decisions: list[dict[str, Any]] = []
        orders_recorded: list[dict[str, Any]] = []
        input_file_hashes: dict[str, str] = {}
        market_meta: dict[str, Any] = {}
        freshness_meta: dict[str, Any] = {}
        snapshot_manifest: dict[str, Any] | None = None
        regime_label: str | None = None
        hypotheses: dict[str, DirectionalHypothesis] = {}
        timing_assessments: dict[str, TimingAssessment] = {}
        per_symbol_decisions: list[dict[str, Any]] = []
        run_status = "COMPLETED"

        with OfflineExecutionGuard(self.instrumentation):
            try:
                pre_cutoff_frames, session_frames, market_meta, mkt_hashes = (
                    self._load_and_split_1h_archives()
                )
                input_file_hashes.update(mkt_hashes)

                pred_path, intel_path, reg_json_path, snapshot_manifest = (
                    self._resolve_snapshot_paths()
                )
                stage1_batch = self.central_agent.evaluate_from_files(
                    symbols=cfg.symbols,
                    as_of_ist=self.as_of_ist,
                    predictions_path=pred_path,
                    intelligence_path=intel_path,
                    regime_path=reg_json_path,
                )
                hypotheses = stage1_batch.hypotheses
                freshness_meta = stage1_batch.freshness
                regime_label = stage1_batch.market_regime
                input_file_hashes.update(stage1_batch.input_file_hashes)
            except Exception as exc:
                run_status = "BLOCKED_INPUT_VALIDATION"
                rejection_reasons.append(str(exc))
                completed_at = datetime.now(tz=IST_TZ)
                holdout_sha_after = sha256_file(reg_path) if reg_path.exists() else None
                return self._build_report(
                    run_status=run_status,
                    started_at=started_at,
                    completed_at=completed_at,
                    market_meta=market_meta,
                    freshness_meta=freshness_meta,
                    snapshot_manifest=snapshot_manifest,
                    regime_label=regime_label,
                    hypotheses=hypotheses,
                    timing_assessments=timing_assessments,
                    per_symbol_decisions=per_symbol_decisions,
                    risk_decisions=risk_decisions,
                    orders_recorded=orders_recorded,
                    rejection_reasons=rejection_reasons,
                    input_file_hashes=input_file_hashes,
                    holdout_sha_before=holdout_sha_before,
                    holdout_sha_after=holdout_sha_after,
                )

            # Reset portfolio session cleanly
            self.portfolio.reset_daily_session(cfg.session_date)

            # Stage 2 (Independent Candle Timing Agent) -> Stage 3 (DeterministicRiskEngine)
            for sym in cfg.symbols:
                norm_sym = sym.strip().upper()
                hyp = hypotheses[norm_sym]
                pre_bars = pre_cutoff_frames[sym]

                assessment = self.timing_agent.evaluate(
                    hypothesis=hyp,
                    bars_primary=pre_bars,
                    primary_timeframe="1h",
                    bars_15m=None,  # Committed 15m is UNAVAILABLE; never fabricated
                    bars_5m=None,   # Committed 5m is UNAVAILABLE; never fabricated
                    as_of_ist=self.as_of_ist,
                )
                timing_assessments[norm_sym] = assessment

                day_df = session_frames[sym]
                eligible_bars = day_df[day_df["timestamp"] >= pd.Timestamp(self.as_of_ist)]

                symbol_risk_decision: dict[str, Any]
                if assessment.decision != TimingDecision.ENTER_NOW:
                    symbol_risk_decision = {
                        "symbol": norm_sym,
                        "stage": "STAGE2_TIMING_GATE",
                        "approved": False,
                        "status": f"TIMING_{assessment.decision.value}",
                        "approved_quantity": 0,
                        "requested_quantity": 0,
                        "rejection_reasons": list(assessment.rationale),
                        "timestamp_ist": self.as_of_ist.isoformat(),
                    }
                    risk_decisions.append(symbol_risk_decision)
                elif hyp.direction != DirectionalBias.BULLISH:
                    reason = (
                        f"BEARISH timing confirmation ({norm_sym}) skipped: "
                        "IntradayPaperPortfolio supports long-only equity execution"
                    )
                    symbol_risk_decision = {
                        "symbol": norm_sym,
                        "stage": "EXECUTION_CAPABILITY_GATE",
                        "approved": False,
                        "status": "SHORT_NOT_SUPPORTED_LONG_ONLY_LEDGER",
                        "approved_quantity": 0,
                        "requested_quantity": 0,
                        "rejection_reasons": [reason],
                        "timestamp_ist": self.as_of_ist.isoformat(),
                    }
                    risk_decisions.append(symbol_risk_decision)
                elif eligible_bars.empty:
                    reason = (
                        f"No eligible session bars at or after as_of_ist "
                        f"({self.as_of_ist.isoformat()}) for {norm_sym}"
                    )
                    rejection_reasons.append(reason)
                    symbol_risk_decision = {
                        "symbol": norm_sym,
                        "stage": "EXECUTION_BAR_GATE",
                        "approved": False,
                        "status": "NO_ELIGIBLE_BAR_AFTER_AS_OF",
                        "approved_quantity": 0,
                        "requested_quantity": 0,
                        "rejection_reasons": [reason],
                        "timestamp_ist": self.as_of_ist.isoformat(),
                    }
                    risk_decisions.append(symbol_risk_decision)
                else:
                    entry_bar = eligible_bars.iloc[0]
                    entry_ts = entry_bar["timestamp"].to_pydatetime()
                    entry_price = float(entry_bar["open"])

                    # Validate structural stop & target against actual entry open (fail closed on gap violation)
                    stop_loss = assessment.invalidation_price
                    target = assessment.target_price
                    if (
                        stop_loss is None
                        or target is None
                        or entry_price <= 0.0
                        or stop_loss >= entry_price
                        or target <= entry_price
                    ):
                        reason = (
                            f"Opening gap invalidated structural levels for {norm_sym}: "
                            f"open={entry_price:.2f}, stop={stop_loss}, target={target}"
                        )
                        rejection_reasons.append(reason)
                        symbol_risk_decision = {
                            "symbol": norm_sym,
                            "stage": "OPEN_GAP_VALIDATION",
                            "approved": False,
                            "status": "REJECTED_INVALID_LEVELS_AT_OPEN",
                            "approved_quantity": 0,
                            "requested_quantity": 0,
                            "rejection_reasons": [reason],
                            "timestamp_ist": entry_ts.isoformat(),
                        }
                        risk_decisions.append(symbol_risk_decision)
                    elif (
                        (entry_price - stop_loss) / entry_price
                        > self.config.timing_config.max_stop_distance_pct
                    ):
                        realized_stop_dist_pct = (entry_price - stop_loss) / entry_price
                        reason = (
                            f"Opening gap exceeded max stop distance for {norm_sym}: "
                            f"realized={realized_stop_dist_pct:.4f} > "
                            f"max={self.config.timing_config.max_stop_distance_pct:.4f} "
                            f"(open={entry_price:.2f}, stop={stop_loss:.2f})"
                        )
                        rejection_reasons.append(reason)
                        symbol_risk_decision = {
                            "symbol": norm_sym,
                            "stage": "OPEN_GAP_VALIDATION",
                            "approved": False,
                            "status": "OPEN_GAP_STOP_DISTANCE_EXCEEDED",
                            "approved_quantity": 0,
                            "requested_quantity": 0,
                            "rejection_reasons": [reason],
                            "timestamp_ist": entry_ts.isoformat(),
                        }
                        risk_decisions.append(symbol_risk_decision)
                    elif (
                        (target - entry_price) / (entry_price - stop_loss)
                        < self.config.timing_config.min_risk_reward_ratio
                    ):
                        realized_rr = (target - entry_price) / (entry_price - stop_loss)
                        reason = (
                            f"Opening gap degraded risk-reward ratio for {norm_sym}: "
                            f"realized={realized_rr:.3f} < "
                            f"min={self.config.timing_config.min_risk_reward_ratio:.3f} "
                            f"(open={entry_price:.2f}, stop={stop_loss:.2f}, target={target:.2f})"
                        )
                        rejection_reasons.append(reason)
                        symbol_risk_decision = {
                            "symbol": norm_sym,
                            "stage": "OPEN_GAP_VALIDATION",
                            "approved": False,
                            "status": "OPEN_GAP_RR_DEGRADED",
                            "approved_quantity": 0,
                            "requested_quantity": 0,
                            "rejection_reasons": [reason],
                            "timestamp_ist": entry_ts.isoformat(),
                        }
                        risk_decisions.append(symbol_risk_decision)
                    else:
                        from trader.authority.instructions import ActionType as _ActionType

                        risk_per_share = max(entry_price - stop_loss, 0.01)
                        max_risk_inr = self.portfolio.total_equity * cfg.max_risk_per_trade_pct
                        requested_qty = max(1, int(max_risk_inr // risk_per_share))

                        if self.execution_gate.get_active_central_instruction(norm_sym, entry_ts) is None:
                            self.execution_gate.set_central_instruction(
                                hyp.to_authority_instruction(expires_after_hours=8.0),
                                current_time=entry_ts,
                            )

                        orders_before = len(self.portfolio.orders)
                        try:
                            gate_outcome, _ = self.execution_gate.execute_with_gate_protection(
                                symbol=norm_sym,
                                action_type=_ActionType.ENTRY_LONG,
                                current_time=entry_ts,
                                portfolio=self.portfolio,
                                requested_quantity=requested_qty,
                                price=entry_price,
                                candle_assessment=assessment,
                                stop_loss_price=stop_loss,
                                target_price=target,
                            )
                        except AttributeError as exc:
                            if "'NoneType' object has no attribute 'approved'" in str(exc):
                                reason = f"Missing risk decision for {norm_sym}; failed closed"
                                rejection_reasons.append(reason)
                                symbol_risk_decision = {
                                    "symbol": norm_sym,
                                    "stage": "RISK_ENGINE",
                                    "approved": False,
                                    "status": "FAIL_CLOSED_MISSING_DECISION",
                                    "approved_quantity": 0,
                                    "requested_quantity": requested_qty,
                                    "rejection_reasons": [reason],
                                    "timestamp_ist": entry_ts.isoformat(),
                                }
                                risk_decisions.append(symbol_risk_decision)
                                gate_outcome = None
                            else:
                                raise

                        if gate_outcome is not None:
                            risk_eval = gate_outcome.risk_evaluation
                            combined_reasons = (
                                list(gate_outcome.rejection_reasons)
                                if not gate_outcome.approved
                                else (list(risk_eval.rejection_reasons) if risk_eval is not None else [])
                            )
                            if risk_eval is None:
                                symbol_risk_decision = {
                                    "symbol": norm_sym,
                                    "stage": "RISK_ENGINE",
                                    "approved": False,
                                    "status": str(gate_outcome.status),
                                    "approved_quantity": 0,
                                    "requested_quantity": requested_qty,
                                    "rejection_reasons": combined_reasons,
                                    "max_risk_amount_inr": 0.0,
                                    "estimated_trade_risk_inr": 0.0,
                                    "gate_status": str(gate_outcome.status),
                                    "gate_approved": False,
                                    "timestamp_ist": entry_ts.isoformat(),
                                }
                                risk_decisions.append(symbol_risk_decision)
                                rejection_reasons.extend(combined_reasons)
                            else:
                                symbol_risk_decision = {
                                    "symbol": norm_sym,
                                    "stage": "RISK_ENGINE",
                                    "approved": bool(gate_outcome.approved and risk_eval.approved),
                                    "status": (
                                        str(risk_eval.status)
                                        if (gate_outcome.approved or not risk_eval.approved)
                                        else str(gate_outcome.status)
                                    ),
                                    "approved_quantity": int(risk_eval.approved_quantity) if gate_outcome.approved else 0,
                                    "requested_quantity": int(risk_eval.requested_quantity),
                                    "rejection_reasons": combined_reasons,
                                    "max_risk_amount_inr": float(risk_eval.max_risk_amount_inr),
                                    "estimated_trade_risk_inr": float(risk_eval.estimated_trade_risk_inr),
                                    "gate_status": str(gate_outcome.status),
                                    "gate_approved": bool(gate_outcome.approved),
                                    "timestamp_ist": entry_ts.isoformat(),
                                }
                                risk_decisions.append(symbol_risk_decision)

                                if gate_outcome.approved and risk_eval.approved and risk_eval.approved_quantity > 0:
                                    for order in self.portfolio.orders[orders_before:]:
                                        orders_recorded.append(self._serialize_order(order))
                                        if order.status == OrderStatus.REJECTED and order.rejection_reason:
                                            rejection_reasons.append(order.rejection_reason)
                                else:
                                    rejection_reasons.extend(combined_reasons)

                per_symbol_decisions.append(
                    {
                        "symbol": norm_sym,
                        "confidence_calibration_status": hyp.confidence_calibration_status,
                        "confidence_calibration_note": hyp.confidence_calibration_note,
                        "stage1_central": hyp.to_dict(),
                        "stage2_timing": {
                            "decision": assessment.decision.value,
                            "alignment_status": assessment.alignment_status.value,
                            "central_confidence_calibration_status": (
                                assessment.central_confidence_calibration_status
                            ),
                            "candle_bias": assessment.candle_bias.value,
                            "timing_quality_score": round(assessment.timing_quality_score, 4),
                            "price_action_score": round(assessment.price_action_score, 4),
                            "entry_price": (
                                round(float(assessment.entry_price), 2)
                                if assessment.entry_price is not None
                                else None
                            ),
                            "invalidation_price": assessment.invalidation_price,
                            "target_price": assessment.target_price,
                            "risk_reward_ratio": assessment.risk_reward_ratio,
                            "detected_patterns": [p.value for p in assessment.detected_patterns],
                            "timeframe_statuses": dict(assessment.timeframe_statuses),
                            "contradiction_reasons": list(assessment.contradiction_reasons),
                            "rationale": list(assessment.rationale),
                        },
                        "risk_decision": symbol_risk_decision,
                    }
                )

            # Chronologically update market prices across session bars; route all SL/TP exits via RISK_EXIT
            from trader.authority.instructions import ActionType as _ExitActionType

            all_timestamps = sorted(
                {
                    ts.to_pydatetime()
                    for df in session_frames.values()
                    for ts in df["timestamp"]
                }
            )
            latest_prices: dict[str, float] = {}
            for ts in all_timestamps:
                for sym, df in session_frames.items():
                    row = df[df["timestamp"] == ts]
                    if not row.empty:
                        close_px = float(row.iloc[0]["close"])
                        norm_sym = sym.strip().upper()
                        latest_prices[norm_sym] = close_px
                        if norm_sym in self.portfolio.positions:
                            pos = self.portfolio.positions[norm_sym]
                            if ts >= pos.entry_time:
                                pos.current_price = close_px
                                pos.highest_price = max(pos.highest_price, close_px)
                                pos.lowest_price = min(
                                    pos.lowest_price if pos.lowest_price is not None else close_px,
                                    close_px,
                                )
                                exit_triggered = False
                                if pos.side == "SHORT":
                                    if (pos.stop_loss is not None and close_px >= pos.stop_loss) or (
                                        pos.target is not None and close_px <= pos.target
                                    ):
                                        exit_triggered = True
                                else:
                                    if (pos.stop_loss is not None and close_px <= pos.stop_loss) or (
                                        pos.target is not None and close_px >= pos.target
                                    ):
                                        exit_triggered = True
                                if exit_triggered:
                                    orders_before = len(self.portfolio.orders)
                                    exit_out, _ = self.execution_gate.execute_with_gate_protection(
                                        symbol=pos.symbol,
                                        action_type=_ExitActionType.RISK_EXIT,
                                        current_time=ts,
                                        portfolio=self.portfolio,
                                        requested_quantity=pos.quantity,
                                        price=close_px,
                                    )
                                    for ao in self.portfolio.orders[orders_before:]:
                                        orders_recorded.append(self._serialize_order(ao))
                                        if ao.status == OrderStatus.REJECTED and ao.rejection_reason:
                                            rejection_reasons.append(ao.rejection_reason)
                                    if not exit_out.approved and exit_out.rejection_reasons:
                                        rejection_reasons.extend(exit_out.rejection_reasons)

            # Mandatory end-of-session square-off on final bar timestamp routed via ActionType.RISK_EXIT
            if self.portfolio.positions and all_timestamps:
                sq_ts = all_timestamps[-1]
                for pos in list(self.portfolio.positions.values()):
                    sq_px = latest_prices.get(
                        pos.symbol,
                        latest_prices.get(pos.symbol.strip().upper(), pos.current_price),
                    )
                    orders_before = len(self.portfolio.orders)
                    sq_out, _ = self.execution_gate.execute_with_gate_protection(
                        symbol=pos.symbol,
                        action_type=_ExitActionType.RISK_EXIT,
                        current_time=sq_ts,
                        portfolio=self.portfolio,
                        requested_quantity=pos.quantity,
                        price=sq_px,
                    )
                    for so in self.portfolio.orders[orders_before:]:
                        orders_recorded.append(self._serialize_order(so))
                        if so.status == OrderStatus.REJECTED and so.rejection_reason:
                            rejection_reasons.append(so.rejection_reason)
                    if not sq_out.approved and sq_out.rejection_reasons:
                        rejection_reasons.extend(sq_out.rejection_reasons)

        completed_at = datetime.now(tz=IST_TZ)
        holdout_sha_after = sha256_file(reg_path) if reg_path.exists() else None
        return self._build_report(
            run_status=run_status,
            started_at=started_at,
            completed_at=completed_at,
            market_meta=market_meta,
            freshness_meta=freshness_meta,
            snapshot_manifest=snapshot_manifest,
            regime_label=regime_label,
            hypotheses=hypotheses,
            timing_assessments=timing_assessments,
            per_symbol_decisions=per_symbol_decisions,
            risk_decisions=risk_decisions,
            orders_recorded=orders_recorded,
            rejection_reasons=rejection_reasons,
            input_file_hashes=input_file_hashes,
            holdout_sha_before=holdout_sha_before,
            holdout_sha_after=holdout_sha_after,
        )

    @staticmethod
    def _serialize_order(order: Any) -> dict[str, Any]:
        cb = order.cost_breakdown
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "exchange": order.exchange,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "quantity": int(order.quantity),
            "filled_quantity": int(order.filled_quantity),
            "price": float(order.price),
            "filled_price": float(order.filled_price) if order.filled_price is not None else None,
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            "stop_loss": order.stop_loss,
            "target": order.target,
            "timestamp_ist": order.timestamp.isoformat() if order.timestamp else None,
            "rejection_reason": order.rejection_reason,
            "cost_breakdown": asdict(cb) if cb is not None else None,
        }

    def _build_report(
        self,
        *,
        run_status: str,
        started_at: datetime,
        completed_at: datetime,
        market_meta: dict[str, Any],
        freshness_meta: dict[str, Any],
        snapshot_manifest: dict[str, Any] | None,
        regime_label: str | None,
        hypotheses: dict[str, DirectionalHypothesis],
        timing_assessments: dict[str, TimingAssessment],
        per_symbol_decisions: list[dict[str, Any]],
        risk_decisions: list[dict[str, Any]],
        orders_recorded: list[dict[str, Any]],
        rejection_reasons: list[str],
        input_file_hashes: dict[str, str],
        holdout_sha_before: str | None,
        holdout_sha_after: str | None,
    ) -> dict[str, Any]:
        ledger_ok, ledger_msg = self.portfolio.verify_ledger_integrity()
        summary = self.portfolio.get_portfolio_summary()

        total_brokerage = 0.0
        total_statutory = 0.0
        total_slippage = 0.0
        total_friction = 0.0
        executed_statuses = (OrderStatus.FILLED.value, OrderStatus.PARTIALLY_FILLED.value)
        for o in orders_recorded:
            if o.get("status") not in executed_statuses:
                continue
            cb = o.get("cost_breakdown")
            if cb:
                total_brokerage += float(cb.get("brokerage_inr", 0.0))
                total_statutory += (
                    float(cb.get("stt_inr", 0.0))
                    + float(cb.get("exchange_charges_inr", 0.0))
                    + float(cb.get("sebi_fee_inr", 0.0))
                    + float(cb.get("stamp_duty_inr", 0.0))
                    + float(cb.get("gst_inr", 0.0))
                )
                total_slippage += float(cb.get("slippage_inr", 0.0))
                total_friction += float(cb.get("total_friction_inr", 0.0))

        init_cap = float(self.portfolio.initial_capital)
        net_pnl_inr = float(self.portfolio.total_equity) - init_cap
        gross_pnl_inr = round(net_pnl_inr + total_friction, 2)
        gross_return_pct = round((gross_pnl_inr / init_cap) * 100.0, 4) if init_cap > 0 else 0.0
        net_return_pct = round((net_pnl_inr / init_cap) * 100.0, 4) if init_cap > 0 else 0.0

        reconciled_equity = round(
            float(summary["cash_inr"]) + float(summary["invested_capital_inr"]), 2
        )
        equity_diff = round(abs(reconciled_equity - float(summary["total_equity_inr"])), 2)

        holdout_unchanged = (
            holdout_sha_before is not None and holdout_sha_before == holdout_sha_after
        )
        execution_provenance = _collect_execution_provenance()

        report: dict[str, Any] = {
            "cycle_id": "CYCLE-007",
            "phase": "DIRECTIONAL_PREDICTION_AND_INTELLIGENT_ENTRY_TIMING_NUMERICAL",
            "run_mode": "HISTORICAL_PAPER_TRADING_DRY_RUN",
            "run_status": run_status,
            "qualification_allowed": False,
            "unseen_evaluation_status": "NOT_EVALUATED",
            "unseen_return_pct": None,
            "live_trading_enabled": False,
            "real_money_trading": False,
            "visual_ai_enabled": False,
            "synthetic_candles_used": 0,
            "confidence_calibration_status": CONFIDENCE_CALIBRATION_STATUS,
            "confidence_calibration_note": CONFIDENCE_CALIBRATION_NOTE,
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
            "session_date": self.config.session_date.isoformat(),
            "as_of_ist": self.as_of_ist.isoformat(),
            "symbols": list(self.config.symbols),
            "market_regime": regime_label,
            "market_data_provenance": market_meta,
            "input_freshness": freshness_meta,
            "snapshot_manifest": snapshot_manifest,
            "input_file_hashes": input_file_hashes,
            "stage1_central_hypotheses": {
                sym: hyp.to_dict() for sym, hyp in hypotheses.items()
            },
            "stage2_timing_assessments": {
                sym: assmt.to_dict() for sym, assmt in timing_assessments.items()
            },
            "per_symbol_decisions": per_symbol_decisions,
            "risk_decisions": risk_decisions,
            "orders": orders_recorded,
            "order_count": len(orders_recorded),
            "filled_order_count": sum(
                1 for o in orders_recorded if o.get("status") in executed_statuses
            ),
            "open_positions_count": int(summary["open_positions_count"]),
            "closed_trades_count": int(summary["closed_trades_count"]),
            "portfolio_results": {
                "initial_capital_inr": init_cap,
                "ending_cash_inr": float(summary["cash_inr"]),
                "invested_capital_inr": float(summary["invested_capital_inr"]),
                "ending_equity_inr": float(summary["total_equity_inr"]),
                "gross_pnl_inr": gross_pnl_inr,
                "net_pnl_inr": round(net_pnl_inr, 2),
                "gross_return_pct": gross_return_pct,
                "net_return_pct": net_return_pct,
            },
            "cost_accounting": {
                "brokerage_inr": round(total_brokerage, 4),
                "statutory_charges_inr": round(total_statutory, 4),
                "slippage_inr": round(total_slippage, 4),
                "total_friction_inr": round(total_friction, 4),
            },
            "ledger_reconciliation": {
                "sha256_chain_valid": bool(ledger_ok),
                "verification_message": ledger_msg,
                "ledger_event_count": len(self.portfolio.event_ledger),
                "cash_plus_invested_matches_equity": equity_diff <= 0.02,
                "equity_reconciliation_diff_inr": equity_diff,
            },
            "holdout_registry_integrity": {
                "path": str(self.config.holdout_registry_path),
                "sha256_before": holdout_sha_before,
                "sha256_after": holdout_sha_after,
                "unchanged": holdout_unchanged,
            },
            "rejection_reasons": rejection_reasons,
        }

        canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        report["output_report_sha256"] = hashlib.sha256(canonical).hexdigest()
        return report
