"""CYCLE-006B — Isolated Historical Paper-Trading Dry-Run Infrastructure.

Wires existing BharatAlpha components into a deterministic, fail-closed,
auditable historical paper-trading pipeline without live broker connections,
external API calls, synthetic candles, or holdout registry mutations.

Classification: HISTORICAL_PAPER_TRADING_DRY_RUN (infrastructure validation only;
not strategy profitability validation or formal qualification).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, LabRiskConfig
from trader.data.provenance_1h import validate_1h_provenance
from trader.intelligence.parsing import (
    get_git_commit_sha,
    load_json_strict,
    parse_finite_float,
    parse_iso_to_ist,
    parse_obs_date,
    sha256_file,
    write_report_atomic,
)
from trader.portfolio.cost_calculator import CostConfig
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from trader.risk.engine import DeterministicRiskEngine, RiskEvaluationResult

if TYPE_CHECKING:
    from trader.authority.gate import MandatoryExecutionGate
    from trader.authority.instructions import AuthorityInstruction, PermissionType

DEFAULT_CACHE_DIR = Path("data/intraday")
DEFAULT_PREDICTIONS_PATH = Path("data/predictions/latest_predictions.json")
DEFAULT_INTELLIGENCE_PATH = Path("data/predictions/phase5_intelligence.json")
DEFAULT_REGIME_PATH = Path("docs/data/regime.json")
SUPPORTED_PHASE5_SCHEMA_PREFIX = "5."


class ExternalNetworkBlockedError(RuntimeError):
    """Raised if any outbound network or API call is attempted during dry-run execution."""


@dataclass
class ExecutionInstrumentation:
    """Actual runtime instrumentation counters and explicit verification semantics.

    Distinguishes actively intercepted Python socket/urllib network attempts
    from structural absence of broker adapters and paid API clients in the
    CYCLE-006B execution path.
    """

    network_guard_active: bool = False
    network_guard_scope: str = "python_stdlib_socket_and_urllib_only_not_os_sandbox"
    external_api_calls: int = 0
    broker_adapters_configured: int = 0
    broker_connections: int = 0
    broker_connections_verification: str = "NO_BROKER_ADAPTER_IN_EXECUTION_PATH"
    paid_llm_providers_configured: int = 0
    paid_api_cost_usd: float = 0.0
    paid_api_cost_verification: str = "NO_PAID_API_CLIENT_IN_EXECUTION_PATH_AND_PYTHON_NET_GUARD_ACTIVE"
    blocked_network_attempts: list[str] = field(default_factory=list)


class OfflineExecutionGuard:
    """Intercepts selected Python stdlib socket/urllib calls (not a full OS network sandbox)."""

    def __init__(self, instrumentation: ExecutionInstrumentation) -> None:
        self._inst = instrumentation
        self._orig_create_connection: Any = None
        self._orig_urlopen: Any = None
        self._orig_socket_connect: Any = None
        self._orig_socket_connect_ex: Any = None

    def __enter__(self) -> ExecutionInstrumentation:
        self._orig_create_connection = socket.create_connection
        self._orig_urlopen = urllib.request.urlopen
        self._orig_socket_connect = socket.socket.connect
        self._orig_socket_connect_ex = socket.socket.connect_ex
        inst = self._inst
        orig_connect = self._orig_socket_connect
        orig_connect_ex = self._orig_socket_connect_ex

        def _blocked_connect(address: Any, *args: Any, **kwargs: Any) -> Any:
            inst.external_api_calls += 1
            inst.blocked_network_attempts.append(f"socket.create_connection({address!r})")
            raise ExternalNetworkBlockedError(
                f"Outbound network connection blocked in CYCLE-006B dry-run: {address!r}"
            )

        def _blocked_urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
            inst.external_api_calls += 1
            inst.blocked_network_attempts.append(f"urllib.request.urlopen({url!r})")
            raise ExternalNetworkBlockedError(
                f"Outbound HTTP request blocked in CYCLE-006B dry-run: {url!r}"
            )

        def _blocked_socket_connect(sock_self: socket.socket, address: Any) -> Any:
            if sock_self.family in (socket.AF_INET, socket.AF_INET6):
                inst.external_api_calls += 1
                inst.blocked_network_attempts.append(f"socket.socket.connect({address!r})")
                raise ExternalNetworkBlockedError(
                    f"Outbound socket connect blocked in CYCLE-006B dry-run: {address!r}"
                )
            return orig_connect(sock_self, address)

        def _blocked_socket_connect_ex(sock_self: socket.socket, address: Any) -> Any:
            if sock_self.family in (socket.AF_INET, socket.AF_INET6):
                inst.external_api_calls += 1
                inst.blocked_network_attempts.append(f"socket.socket.connect_ex({address!r})")
                raise ExternalNetworkBlockedError(
                    f"Outbound socket connect_ex blocked in CYCLE-006B dry-run: {address!r}"
                )
            return orig_connect_ex(sock_self, address)

        socket.create_connection = _blocked_connect  # type: ignore[assignment]
        urllib.request.urlopen = _blocked_urlopen  # type: ignore[assignment]
        socket.socket.connect = _blocked_socket_connect  # type: ignore[assignment]
        socket.socket.connect_ex = _blocked_socket_connect_ex  # type: ignore[assignment]
        inst.network_guard_active = True
        return inst

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._orig_create_connection is not None:
            socket.create_connection = self._orig_create_connection  # type: ignore[assignment]
        if self._orig_urlopen is not None:
            urllib.request.urlopen = self._orig_urlopen  # type: ignore[assignment]
        if self._orig_socket_connect is not None:
            socket.socket.connect = self._orig_socket_connect  # type: ignore[assignment]
        if self._orig_socket_connect_ex is not None:
            socket.socket.connect_ex = self._orig_socket_connect_ex  # type: ignore[assignment]


@dataclass(frozen=True)
class Cycle006BConfig:
    """Explicit configuration for CYCLE-006B historical paper-trading dry run."""

    session_date: date
    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE
    initial_capital_inr: float = LabRiskConfig().initial_capital_inr
    slippage_bps: float = 5.0
    max_staleness_hours: float = 72.0
    max_input_spread_hours: float = 48.0
    min_bars_per_session: int = 6
    min_prob_up_buy: float = 0.55
    min_expected_return_pct: float = 0.25
    stop_loss_pct: float = 0.015
    target_pct: float = 0.03
    as_of_time_ist: dt_time = dt_time(9, 15)
    cache_dir: Path = DEFAULT_CACHE_DIR
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH
    intelligence_path: Path = DEFAULT_INTELLIGENCE_PATH
    regime_path: Path = DEFAULT_REGIME_PATH
    max_risk_per_trade_pct: float = 0.005
    max_position_weight_pct: float = 0.40
    max_portfolio_exposure_pct: float = 0.90
    max_daily_loss_pct: float = 0.02
    max_concurrent_positions: int = 3


# Backward-compatible shims for existing tests/callers (CYCLE-009B)
_get_git_commit_sha = get_git_commit_sha
_parse_iso_to_ist = parse_iso_to_ist
_parse_obs_date = parse_obs_date
_parse_finite_float = parse_finite_float
_load_json_strict = load_json_strict


class Cycle006BPaperTradingRunner:
    """Deterministic, fail-closed historical paper-trading runner (CYCLE-006B)."""

    def __init__(
        self,
        config: Cycle006BConfig,
        risk_engine: DeterministicRiskEngine | None = None,
        *,
        execution_gate: MandatoryExecutionGate | None = None,
        human_instruction: AuthorityInstruction | None = None,
        human_permission: PermissionType | None = None,
    ) -> None:
        if config.initial_capital_inr <= 0:
            raise ValueError("initial_capital_inr must be > 0")
        if config.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if config.max_staleness_hours <= 0:
            raise ValueError("max_staleness_hours must be > 0")
        if config.max_input_spread_hours < 0:
            raise ValueError("max_input_spread_hours must be >= 0")
        if config.min_bars_per_session < 1:
            raise ValueError("min_bars_per_session must be >= 1")
        if config.stop_loss_pct <= 0 or config.stop_loss_pct >= 1:
            raise ValueError("stop_loss_pct must be in (0, 1)")
        if config.target_pct <= 0:
            raise ValueError("target_pct must be > 0")
        if not config.symbols:
            raise ValueError("symbols must not be empty")
        normalized_syms = [s.strip().upper() for s in config.symbols if isinstance(s, str) and s.strip()]
        if len(normalized_syms) != len(config.symbols) or len(set(normalized_syms)) != len(normalized_syms):
            raise ValueError("symbols must be non-empty, unique instrument identifiers")

        from trader.authority.gate import MandatoryExecutionGate as _MandatoryExecutionGate
        from trader.authority.instructions import (
            AuthorityInstruction as _AuthorityInstruction,
            AuthoritySource as _AuthoritySource,
        )

        self.config = config
        self.as_of_ist = datetime.combine(
            config.session_date, config.as_of_time_ist, tzinfo=IST_TZ
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
        # Only register Human authority if explicitly provided by the caller.
        session_open_ist = datetime.combine(config.session_date, dt_time(9, 15), tzinfo=IST_TZ)
        session_close_ist = datetime.combine(config.session_date, dt_time(15, 30), tzinfo=IST_TZ)
        if human_instruction is not None:
            self.execution_gate.set_human_instruction(human_instruction, current_time=session_open_ist)
        elif human_permission is not None:
            h_instr = _AuthorityInstruction(
                instruction_id=f"HUMAN-CYCLE006B-{config.session_date.isoformat()}",
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

    def _validate_and_load_market_data(self) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, str]]:
        """Load historical 1h parquet archives with strict provenance & OHLCV validation."""
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

        session_frames: dict[str, pd.DataFrame] = {}
        per_symbol_meta: dict[str, Any] = {}
        file_hashes: dict[str, str] = {}

        for sym in cfg.symbols:
            pq_path = cfg.cache_dir / f"{sym}_1h.parquet"
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
            day_df = work[work["timestamp"].dt.date == cfg.session_date].copy()
            if len(day_df) < cfg.min_bars_per_session:
                raise ValueError(
                    f"Incomplete 1h session for {sym} on {cfg.session_date}: "
                    f"got {len(day_df)} bars, required >= {cfg.min_bars_per_session}"
                )

            # Strict OHLCV integrity on session bars (never silently fill NaNs or allow non-finite values)
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            if day_df[ohlcv_cols].isna().any().any():
                raise ValueError(f"NaN OHLCV values detected for {sym} on {cfg.session_date}")
            try:
                ohlcv_arr = day_df[ohlcv_cols].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric OHLCV values detected for {sym} on {cfg.session_date}") from exc
            if not np.isfinite(ohlcv_arr).all():
                raise ValueError(f"Non-finite (Inf/-Inf) OHLCV values detected for {sym} on {cfg.session_date}")
            if (
                (day_df["open"] <= 0).any()
                or (day_df["high"] <= 0).any()
                or (day_df["close"] <= 0).any()
                or (day_df["low"] <= 0).any()
            ):
                raise ValueError(f"Non-positive prices detected for {sym} on {cfg.session_date}")
            if (day_df["volume"] < 0).any():
                raise ValueError(f"Negative volume detected for {sym} on {cfg.session_date}")
            if (day_df["high"] < day_df[["open", "close", "low"]].max(axis=1)).any():
                raise ValueError(f"OHLC high invariant violated for {sym} on {cfg.session_date}")
            if (day_df["low"] > day_df[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError(f"OHLC low invariant violated for {sym} on {cfg.session_date}")

            day_df = day_df.sort_values("timestamp").reset_index(drop=True)
            session_frames[sym] = day_df
            per_symbol_meta[sym] = {
                "file": pq_path.name,
                "sha256": file_hashes[str(pq_path)],
                "session_bars": int(len(day_df)),
                "first_candle_ist": day_df["timestamp"].iloc[0].isoformat(),
                "latest_candle_ist": day_df["timestamp"].iloc[-1].isoformat(),
                "timezone": str(day_df["timestamp"].dt.tz),
            }

        market_meta = {
            "data_classification": "HISTORICAL_1H_PARQUET_ARCHIVE_NOT_LIVE",
            "provenance_status": prov_status,
            "provenance_verified": bool(prov_verified),
            "dataset_fingerprint": prov_report.get("dataset_fingerprint"),
            "session_date": cfg.session_date.isoformat(),
            "symbols": per_symbol_meta,
        }
        return session_frames, market_meta, file_hashes

    def _validate_and_load_inputs(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
        """Validate predictions, Phase 5 intelligence, and regime JSONs with strict as-of alignment."""
        cfg = self.config
        file_hashes: dict[str, str] = {}

        # 1. Predictions JSON
        pred_raw, pred_hash = _load_json_strict(cfg.predictions_path, "latest_predictions")
        file_hashes[str(cfg.predictions_path)] = pred_hash
        pred_ts = _parse_iso_to_ist(pred_raw.get("generated_at"), "latest_predictions.generated_at")
        pred_list = pred_raw.get("predictions")
        if not isinstance(pred_list, list) or not pred_list:
            raise ValueError("latest_predictions.json has missing or empty 'predictions' list")

        pred_by_sym: dict[str, dict[str, Any]] = {}
        for row in pred_list:
            if not isinstance(row, dict) or not row.get("symbol"):
                raise ValueError("Prediction record missing required 'symbol' identifier")
            sym = str(row["symbol"]).strip().upper()
            if sym in pred_by_sym:
                raise ValueError(f"Duplicate prediction record for symbol {sym}")
            pred_by_sym[sym] = row

        # 2. Phase 5 Intelligence JSON
        intel_raw, intel_hash = _load_json_strict(cfg.intelligence_path, "phase5_intelligence")
        file_hashes[str(cfg.intelligence_path)] = intel_hash
        model_ver = str(intel_raw.get("model_version", ""))
        if not model_ver.startswith(SUPPORTED_PHASE5_SCHEMA_PREFIX):
            raise ValueError(
                f"Incompatible phase5_intelligence schema/model_version: {model_ver!r} "
                f"(expected prefix {SUPPORTED_PHASE5_SCHEMA_PREFIX!r})"
            )
        intel_ts = _parse_iso_to_ist(intel_raw.get("generated_at"), "phase5_intelligence.generated_at")
        intel_stocks = intel_raw.get("stocks")
        if not isinstance(intel_stocks, dict) or not intel_stocks:
            raise ValueError("phase5_intelligence.json has missing or empty 'stocks' map")

        # 3. Regime JSON
        regime_raw, regime_hash = _load_json_strict(cfg.regime_path, "regime")
        file_hashes[str(cfg.regime_path)] = regime_hash
        regime_name = regime_raw.get("regime")
        if not isinstance(regime_name, str) or not regime_name.strip():
            raise ValueError("regime.json missing required 'regime' label")
        regime_obs_date = _parse_obs_date(regime_raw.get("observation_date"), "regime.observation_date")
        # Regime observation_date is end-of-day IST (15:30) if no explicit generated_at
        if regime_raw.get("generated_at"):
            regime_ts = _parse_iso_to_ist(regime_raw["generated_at"], "regime.generated_at")
        else:
            regime_ts = datetime.combine(regime_obs_date, dt_time(15, 30), tzinfo=IST_TZ)

        # 4. Strict as-of timestamp checks (reject future-dated & stale inputs)
        freshness: dict[str, Any] = {
            "as_of_ist": self.as_of_ist.isoformat(),
            "max_staleness_hours": cfg.max_staleness_hours,
            "max_input_spread_hours": cfg.max_input_spread_hours,
            "sources": {},
        }
        source_timestamps: dict[str, datetime] = {
            "latest_predictions": pred_ts,
            "phase5_intelligence": intel_ts,
            "regime": regime_ts,
        }
        for name, ts in source_timestamps.items():
            age_hours = round((self.as_of_ist - ts).total_seconds() / 3600.0, 4)
            freshness["sources"][name] = {
                "timestamp_ist": ts.isoformat(),
                "age_hours": age_hours,
            }
            if ts > self.as_of_ist:
                raise ValueError(
                    f"Future-dated input rejected ({name}): {ts.isoformat()} > as_of {self.as_of_ist.isoformat()}"
                )
            if age_hours > cfg.max_staleness_hours:
                raise ValueError(
                    f"Stale input rejected ({name}): age={age_hours:.2f}h > max_staleness_hours={cfg.max_staleness_hours}"
                )

        regime_obs_dt = datetime.combine(regime_obs_date, dt_time(15, 30), tzinfo=IST_TZ)
        regime_obs_age_hours = round((self.as_of_ist - regime_obs_dt).total_seconds() / 3600.0, 4)
        if regime_obs_date > cfg.session_date or regime_obs_dt > self.as_of_ist:
            raise ValueError(
                f"Future-dated regime observation_date rejected: {regime_obs_date} "
                f"(EOD {regime_obs_dt.isoformat()} > as_of {self.as_of_ist.isoformat()})"
            )
        if regime_obs_age_hours > cfg.max_staleness_hours:
            raise ValueError(
                f"Stale regime observation_date rejected: {regime_obs_date} "
                f"(age={regime_obs_age_hours:.2f}h > max_staleness_hours={cfg.max_staleness_hours})"
            )

        # Check cross-source timestamp spread (conflicting timestamps)
        min_ts = min(source_timestamps.values())
        max_ts = max(source_timestamps.values())
        spread_hours = round((max_ts - min_ts).total_seconds() / 3600.0, 4)
        freshness["cross_source_spread_hours"] = spread_hours
        if spread_hours > cfg.max_input_spread_hours:
            raise ValueError(
                f"Conflicting input timestamps rejected: spread={spread_hours:.2f}h > "
                f"max_input_spread_hours={cfg.max_input_spread_hours}"
            )

        intel_by_sym: dict[str, dict[str, Any]] = {}
        for raw_k, val in intel_stocks.items():
            norm_k = str(raw_k).strip().upper()
            if norm_k in intel_by_sym:
                raise ValueError(f"Duplicate intelligence record for symbol {norm_k}")
            if isinstance(val, dict):
                intel_by_sym[norm_k] = val

        # 5. Instrument coverage, independent observation-date freshness & field integrity
        matched_preds: dict[str, dict[str, Any]] = {}
        matched_intel: dict[str, dict[str, Any]] = {}
        obs_freshness: dict[str, Any] = {
            "regime": {
                "observation_date": regime_obs_date.isoformat(),
                "eod_ist": regime_obs_dt.isoformat(),
                "age_hours": regime_obs_age_hours,
            },
            "predictions": {},
            "intelligence": {},
        }
        for sym in cfg.symbols:
            norm_sym = sym.strip().upper()
            if norm_sym not in pred_by_sym:
                raise ValueError(f"Mismatched instruments: {sym} missing from latest_predictions.json")
            if norm_sym not in intel_by_sym:
                raise ValueError(f"Mismatched instruments: {sym} missing from phase5_intelligence.json")

            p_row = pred_by_sym[norm_sym]
            i_row = intel_by_sym[norm_sym]

            p_obs = _parse_obs_date(p_row.get("observation_date"), f"prediction[{sym}].observation_date")
            i_obs = _parse_obs_date(i_row.get("observation_date"), f"intelligence[{sym}].observation_date")

            p_obs_dt = datetime.combine(p_obs, dt_time(15, 30), tzinfo=IST_TZ)
            i_obs_dt = datetime.combine(i_obs, dt_time(15, 30), tzinfo=IST_TZ)
            p_obs_age = round((self.as_of_ist - p_obs_dt).total_seconds() / 3600.0, 4)
            i_obs_age = round((self.as_of_ist - i_obs_dt).total_seconds() / 3600.0, 4)

            if p_obs > cfg.session_date or p_obs_dt > self.as_of_ist:
                raise ValueError(
                    f"Future-dated prediction observation_date for {sym}: {p_obs} "
                    f"(EOD {p_obs_dt.isoformat()} > as_of {self.as_of_ist.isoformat()})"
                )
            if p_obs_age > cfg.max_staleness_hours:
                raise ValueError(
                    f"Stale prediction observation_date rejected for {sym}: {p_obs} "
                    f"(age={p_obs_age:.2f}h > max_staleness_hours={cfg.max_staleness_hours})"
                )
            if i_obs > cfg.session_date or i_obs_dt > self.as_of_ist:
                raise ValueError(
                    f"Future-dated intelligence observation_date for {sym}: {i_obs} "
                    f"(EOD {i_obs_dt.isoformat()} > as_of {self.as_of_ist.isoformat()})"
                )
            if i_obs_age > cfg.max_staleness_hours:
                raise ValueError(
                    f"Stale intelligence observation_date rejected for {sym}: {i_obs} "
                    f"(age={i_obs_age:.2f}h > max_staleness_hours={cfg.max_staleness_hours})"
                )

            obs_freshness["predictions"][norm_sym] = {
                "observation_date": p_obs.isoformat(),
                "eod_ist": p_obs_dt.isoformat(),
                "age_hours": p_obs_age,
            }
            obs_freshness["intelligence"][norm_sym] = {
                "observation_date": i_obs.isoformat(),
                "eod_ist": i_obs_dt.isoformat(),
                "age_hours": i_obs_age,
            }

            prob_up = _parse_finite_float(p_row.get("prob_next_day_up"), f"prediction[{sym}].prob_next_day_up")
            if not (0.0 <= prob_up <= 1.0):
                raise ValueError(f"Invalid probability prob_next_day_up for {sym}: {prob_up} (expected [0, 1])")
            _parse_finite_float(
                p_row.get("expected_next_day_return_pct"),
                f"prediction[{sym}].expected_next_day_return_pct",
            )
            _parse_finite_float(
                i_row.get("news_sentiment_score"),
                f"intelligence[{sym}].news_sentiment_score",
            )

            matched_preds[sym] = p_row
            matched_intel[sym] = i_row

        freshness["observation_freshness"] = obs_freshness
        return matched_preds, matched_intel, regime_raw, freshness, file_hashes

    def run(self) -> dict[str, Any]:
        """Execute bounded historical paper-trading dry run under offline instrumentation."""
        started_at = datetime.now(tz=IST_TZ)
        cfg = self.config
        rejection_reasons: list[str] = []
        risk_decisions: list[dict[str, Any]] = []
        orders_recorded: list[dict[str, Any]] = []
        input_file_hashes: dict[str, str] = {}
        market_meta: dict[str, Any] = {}
        freshness_meta: dict[str, Any] = {}
        regime_label: str | None = None
        run_status = "COMPLETED"

        with OfflineExecutionGuard(self.instrumentation):
            try:
                session_frames, market_meta, mkt_hashes = self._validate_and_load_market_data()
                input_file_hashes.update(mkt_hashes)

                preds, intel, regime_raw, freshness_meta, in_hashes = self._validate_and_load_inputs()
                input_file_hashes.update(in_hashes)
                regime_label = str(regime_raw["regime"])
            except Exception as exc:
                run_status = "BLOCKED_INPUT_VALIDATION"
                rejection_reasons.append(str(exc))
                completed_at = datetime.now(tz=IST_TZ)
                return self._build_report(
                    run_status=run_status,
                    started_at=started_at,
                    completed_at=completed_at,
                    market_meta=market_meta,
                    freshness_meta=freshness_meta,
                    regime_label=regime_label,
                    risk_decisions=risk_decisions,
                    orders_recorded=orders_recorded,
                    rejection_reasons=rejection_reasons,
                    input_file_hashes=input_file_hashes,
                )

            # Reset portfolio session cleanly
            self.portfolio.reset_daily_session(cfg.session_date)

            # Evaluate candidate entries at the first session bar at or after as_of_ist (fills at bar open).
            # Zero future leakage: entry signal uses ONLY as-of pre-market inputs (predictions, intelligence, regime).
            entry_candidates: list[tuple[str, float, float, float, datetime]] = []
            for sym in cfg.symbols:
                p_row = preds[sym]
                i_row = intel[sym]
                prob_up = float(p_row["prob_next_day_up"])
                exp_ret = float(p_row["expected_next_day_return_pct"])
                news_sent = float(i_row["news_sentiment_score"])

                df = session_frames[sym]
                eligible_bars = df[df["timestamp"] >= self.as_of_ist]
                if eligible_bars.empty:
                    reason = (
                        f"No eligible session bars at or after as_of_ist "
                        f"({self.as_of_ist.isoformat()}) for {sym}"
                    )
                    rejection_reasons.append(reason)
                    risk_decisions.append({
                        "symbol": sym,
                        "stage": "SIGNAL_FILTER",
                        "approved": False,
                        "status": "NO_ELIGIBLE_BAR_AFTER_AS_OF",
                        "approved_quantity": 0,
                        "requested_quantity": 0,
                        "rejection_reasons": [reason],
                        "timestamp_ist": self.as_of_ist.isoformat(),
                    })
                    continue

                entry_bar = eligible_bars.iloc[0]
                entry_ts = entry_bar["timestamp"].to_pydatetime()
                entry_price = float(entry_bar["open"])

                if (
                    prob_up >= cfg.min_prob_up_buy
                    and exp_ret >= cfg.min_expected_return_pct
                    and news_sent >= 0.0
                    and "bear" not in regime_label.lower()
                ):
                    stop_loss = round(entry_price * (1.0 - cfg.stop_loss_pct), 2)
                    target = round(entry_price * (1.0 + cfg.target_pct), 2)
                    entry_candidates.append((sym, entry_price, stop_loss, target, entry_ts))
                else:
                    risk_decisions.append({
                        "symbol": sym,
                        "stage": "SIGNAL_FILTER",
                        "approved": False,
                        "status": "NO_TRADE_SIGNAL",
                        "approved_quantity": 0,
                        "requested_quantity": 0,
                        "rejection_reasons": [
                            f"Signal criteria not met (prob_up={prob_up:.3f}, exp_ret={exp_ret:.3f}%, "
                            f"news_sent={news_sent:.3f}, regime={regime_label})"
                        ],
                        "timestamp_ist": entry_ts.isoformat(),
                    })

            # PY-04: Mandatory execution gate for every entry, SL/TP exit, and EOD square-off
            from trader.authority.instructions import (
                ActionType as _ActionType,
                AuthorityInstruction as _AuthorityInstruction,
                AuthoritySource as _AuthoritySource,
                PermissionType as _PermissionType,
            )
            from trader.timing.candle_intelligence import CandleBias as _CandleBias
            from trader.timing.candle_timing_agent import (
                AlignmentStatus as _AlignmentStatus,
                TimingAssessment as _TimingAssessment,
                TimingDecision as _TimingDecision,
            )
            from trader.timing.central_directional_agent import DirectionalBias as _DirectionalBias

            session_close_ist = datetime.combine(cfg.session_date, dt_time(15, 30), tzinfo=IST_TZ)
            for sym, entry_price, stop_loss, target, entry_ts in entry_candidates:
                sym_u = sym.strip().upper()
                risk_per_share = max(entry_price - stop_loss, 0.01)
                max_risk_inr = self.portfolio.total_equity * cfg.max_risk_per_trade_pct
                requested_qty = max(1, int(max_risk_inr // risk_per_share))

                if self.execution_gate.get_active_central_instruction(sym_u, entry_ts) is None:
                    self.execution_gate.set_central_instruction(
                        _AuthorityInstruction(
                            instruction_id=f"CENTRAL-CYCLE006B-{sym_u}-{cfg.session_date.isoformat()}",
                            authority=_AuthoritySource.CENTRAL_AGENT,
                            scope=sym_u,
                            permission=_PermissionType.ALLOW_LONG,
                            valid_from=entry_ts - timedelta(minutes=15),
                            expires_at=session_close_ist + timedelta(minutes=15),
                            reason="CYCLE-006B directional signal filter approved long candidate",
                        ),
                        current_time=entry_ts,
                    )

                legacy_assessment = _TimingAssessment(
                    symbol=sym_u,
                    as_of_ist=entry_ts,
                    decision=_TimingDecision.ENTER_NOW,
                    alignment_status=_AlignmentStatus.CONFIRMED,
                    central_direction=_DirectionalBias.BULLISH,
                    candle_bias=_CandleBias.BULLISH,
                    timing_quality_score=1.0,
                    price_action_score=1.0,
                    entry_price=entry_price,
                    invalidation_price=stop_loss,
                    target_price=target,
                    risk_reward_ratio=round((target - entry_price) / risk_per_share, 4),
                    detected_patterns=(),
                    timeframe_statuses={"1h": "CYCLE006B_SIGNAL_ENTRY"},
                    contradiction_reasons=(),
                    rationale=("CYCLE-006B legacy 1h bar entry routed through MandatoryExecutionGate",),
                )

                orders_before = len(self.portfolio.orders)
                try:
                    gate_outcome, _ = self.execution_gate.execute_with_gate_protection(
                        symbol=sym_u,
                        action_type=_ActionType.ENTRY_LONG,
                        current_time=entry_ts,
                        portfolio=self.portfolio,
                        requested_quantity=requested_qty,
                        price=entry_price,
                        candle_assessment=legacy_assessment,
                        stop_loss_price=stop_loss,
                        target_price=target,
                    )
                except AttributeError as exc:
                    if "'NoneType' object has no attribute 'approved'" in str(exc):
                        reason = f"Missing risk decision for {sym}; failed closed"
                        rejection_reasons.append(reason)
                        risk_decisions.append({
                            "symbol": sym,
                            "stage": "RISK_ENGINE",
                            "approved": False,
                            "status": "FAIL_CLOSED_MISSING_DECISION",
                            "approved_quantity": 0,
                            "requested_quantity": requested_qty,
                            "rejection_reasons": [reason],
                            "timestamp_ist": entry_ts.isoformat(),
                        })
                        continue
                    raise

                risk_eval = gate_outcome.risk_evaluation
                if risk_eval is None and not gate_outcome.approved:
                    combined_reasons = list(gate_outcome.rejection_reasons)
                    risk_decisions.append({
                        "symbol": sym,
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
                    })
                    rejection_reasons.extend(combined_reasons)
                    continue

                assert risk_eval is not None
                combined_reasons = list(gate_outcome.rejection_reasons) if not gate_outcome.approved else list(risk_eval.rejection_reasons)
                decision_record = {
                    "symbol": sym,
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
                risk_decisions.append(decision_record)

                if not gate_outcome.approved or not risk_eval.approved or risk_eval.approved_quantity <= 0:
                    rejection_reasons.extend(combined_reasons)
                    continue

                for order in self.portfolio.orders[orders_before:]:
                    orders_recorded.append(self._serialize_order(order))
                    if order.status == OrderStatus.REJECTED and order.rejection_reason:
                        rejection_reasons.append(order.rejection_reason)

            # Chronologically update market prices across session bars; route all SL/TP exits via RISK_EXIT
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
                        latest_prices[sym] = close_px
                        if sym in self.portfolio.positions:
                            pos = self.portfolio.positions[sym]
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
                                        action_type=_ActionType.RISK_EXIT,
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
                        action_type=_ActionType.RISK_EXIT,
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
        return self._build_report(
            run_status=run_status,
            started_at=started_at,
            completed_at=completed_at,
            market_meta=market_meta,
            freshness_meta=freshness_meta,
            regime_label=regime_label,
            risk_decisions=risk_decisions,
            orders_recorded=orders_recorded,
            rejection_reasons=rejection_reasons,
            input_file_hashes=input_file_hashes,
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
        regime_label: str | None,
        risk_decisions: list[dict[str, Any]],
        orders_recorded: list[dict[str, Any]],
        rejection_reasons: list[str],
        input_file_hashes: dict[str, str],
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

        # Explicit cash + invested == total_equity reconciliation
        reconciled_equity = round(float(summary["cash_inr"]) + float(summary["invested_capital_inr"]), 2)
        equity_diff = round(abs(reconciled_equity - float(summary["total_equity_inr"])), 2)

        report: dict[str, Any] = {
            "cycle_id": "CYCLE-006B",
            "run_mode": "HISTORICAL_PAPER_TRADING_DRY_RUN",
            "run_status": run_status,
            "qualification_allowed": False,
            "unseen_evaluation_status": "NOT_EVALUATED",
            "unseen_return_pct": None,
            "real_money_trading": False,
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
                "market_data_generation_mode": "READ_ONLY_EXISTING_PARQUET_NO_SYNTHETIC_CANDLE_GENERATION",
            },
            "started_at_ist": started_at.isoformat(),
            "completed_at_ist": completed_at.isoformat(),
            "code_commit_sha": _get_git_commit_sha(),
            "session_date": self.config.session_date.isoformat(),
            "symbols": list(self.config.symbols),
            "market_regime": regime_label,
            "market_data_provenance": market_meta,
            "input_freshness": freshness_meta,
            "input_file_hashes": input_file_hashes,
            "risk_decisions": risk_decisions,
            "orders": orders_recorded,
            "order_count": len(orders_recorded),
            "filled_order_count": sum(1 for o in orders_recorded if o.get("status") in executed_statuses),
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
            "rejection_reasons": rejection_reasons,
        }

        canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        report["output_report_sha256"] = hashlib.sha256(canonical).hexdigest()
        return report
