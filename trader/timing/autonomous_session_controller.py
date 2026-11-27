"""BHARATALPHA-FORWARD-SOAK-AUTO-001: Autonomous Long-Only Session Controller.

Manages the deterministic lifecycle of an unattended, strictly paper-only, long-only
NSE intraday forward-soak session:
  - Stage A (`PREFLIGHT`): fail-closed safety, calendar, credential, disk, and lock checks.
  - Stage B (`OBSERVING_NO_NEW_ENTRIES`): 09:15-09:30 IST observation with `NO_NEW_ENTRIES`.
  - Stage C (`AUTONOMOUS_ALLOW_LONG`): automatic 09:30 IST transition to standing `ALLOW_LONG`
    if and only if all 8 mandatory preconditions pass.
  - Stage D (`AUTONOMOUS_ALLOW_LONG`): governed long-only trading with checkpointing.
  - Stage E (`EOD_WIND_DOWN` -> `COMPLETED`): 15:05 wind-down and 15:15 mandatory square-off.
  - `HALTED_OR_INTERRUPTED`: fail-closed halt / disconnect downgrade to `NO_NEW_ENTRIES`
    (preserving safe exits).

HARD SAFETY INVARIANTS:
- Strictly paper-only (`LIVE_TRADING_ENABLED = False`).
- Standing permission is hard-coded to `PermissionType.ALLOW_LONG` (NEVER `ALLOW_SHORT` or `ALLOW_BOTH`).
- Secret values from environment variables are NEVER logged, stored, or returned.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable

from trader.authority.gate import MandatoryExecutionGate
from trader.authority.instructions import (
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)
from trader.config import (
    INTRADAY_REDUCING_WIND_DOWN_TIME,
    IST_TZ,
    is_nse_trading_day,
)
from trader.data.tick_recorder import LIVE_TRADING_ENABLED as RECORDER_LIVE_TRADING_ENABLED
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine, RiskState

LIVE_TRADING_ENABLED: Final[bool] = False
_BLOCKED_HUMAN_OVERRIDE_PERMS: Final[frozenset[PermissionType]] = frozenset(
    {PermissionType.STOP_ALL, PermissionType.PAUSE, PermissionType.EXIT_ONLY}
)
_ANGEL_ONE_ENV_KEYS: Final[tuple[str, ...]] = (
    "ANGEL_AUTH_TOKEN",
    "ANGEL_API_KEY",
    "ANGEL_CLIENT_CODE",
    "ANGEL_FEED_TOKEN",
)
_ZERODHA_ENV_KEYS: Final[tuple[str, ...]] = (
    "KITE_API_KEY",
    "KITE_ACCESS_TOKEN",
)


class SessionStage(str, Enum):
    """Lifecycle stages for an autonomous long-only paper-trading session."""

    PREFLIGHT = "PREFLIGHT"
    OBSERVING_NO_NEW_ENTRIES = "OBSERVING_NO_NEW_ENTRIES"
    AUTONOMOUS_ALLOW_LONG = "AUTONOMOUS_ALLOW_LONG"
    EOD_WIND_DOWN = "EOD_WIND_DOWN"
    HALTED_OR_INTERRUPTED = "HALTED_OR_INTERRUPTED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class PreflightCheckResult:
    """Structured Stage A preflight verification report (contains zero secrets)."""

    passed: bool
    session_date: date
    checked_at_ist: datetime
    checks: dict[str, bool]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "session_date": self.session_date.isoformat(),
            "checked_at_ist": self.checked_at_ist.astimezone(IST_TZ).isoformat(),
            "checks": dict(self.checks),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class TransitionDecision:
    """Audit record for every Stage C transition evaluation or safety downgrade."""

    transitioned: bool
    from_stage: str
    to_stage: str
    timestamp_ist: datetime
    instruction_id: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transitioned": self.transitioned,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "timestamp_ist": self.timestamp_ist.astimezone(IST_TZ).isoformat(),
            "instruction_id": self.instruction_id,
            "reasons": list(self.reasons),
        }


def _to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST_TZ)
    return dt.astimezone(IST_TZ)


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class AutonomousLongOnlySessionController:
    """Fail-closed controller for autonomous long-only paper-trading soak sessions."""

    def __init__(
        self,
        *,
        session_date: date,
        provider: str,
        symbols: Iterable[str],
        output_dir: str | Path,
        min_observation_bars: int = 10,
        max_stale_seconds: float = 180.0,
        min_free_disk_mb: float = 100.0,
        transition_time_ist: time = time(9, 30),
    ) -> None:
        self.session_date: date = session_date
        self.provider: str = str(provider).strip().lower()
        self.symbols: tuple[str, ...] = tuple(
            s.strip().upper() for s in symbols if str(s).strip()
        )
        self.output_dir: Path = Path(output_dir)
        self.min_observation_bars: int = int(min_observation_bars)
        self.max_stale_seconds: float = float(max_stale_seconds)
        self.min_free_disk_mb: float = float(min_free_disk_mb)
        self.transition_time_ist: time = transition_time_ist

        # Hard-coded standing permission: strictly ALLOW_LONG (never ALLOW_SHORT / ALLOW_BOTH)
        self.standing_permission: Final[PermissionType] = PermissionType.ALLOW_LONG

        self.stage: SessionStage = SessionStage.PREFLIGHT
        self.preflight_result: PreflightCheckResult | None = None
        self.ticks_observed: int = 0
        self.completed_1m_bars: int = 0
        self.last_tick_ts_ist: datetime | None = None
        self.interrupted: bool = False
        self.critical_errors: list[str] = []
        self.transition_audit_log: list[TransitionDecision] = []

        self._lock_file: Path | None = None
        self._lock_acquired: bool = False

    # ------------------------------------------------------------------
    # Stage A: Preflight Verification
    # ------------------------------------------------------------------

    def _check_credentials(self, credentials_present: bool | None) -> bool:
        if credentials_present is not None:
            return bool(credentials_present)
        if self.provider == "offline_jsonl":
            return True
        if self.provider == "zerodha":
            return all(bool(os.environ.get(k, "").strip()) for k in _ZERODHA_ENV_KEYS)
        if self.provider == "angel_one":
            return all(bool(os.environ.get(k, "").strip()) for k in _ANGEL_ONE_ENV_KEYS)
        return False

    def _check_storage_and_disk(self) -> bool:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            probe = self.output_dir / f".preflight_write_probe_{os.getpid()}.tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            usage = shutil.disk_usage(self.output_dir)
            free_mb = usage.free / (1024.0 * 1024.0)
            return free_mb >= self.min_free_disk_mb
        except OSError:
            return False

    def _acquire_single_instance_lock(self, lock_file: Path | None) -> bool:
        if lock_file is None:
            return True
        lock_path = Path(lock_file)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            my_pid = os.getpid()
            if lock_path.exists():
                try:
                    raw_pid = lock_path.read_text(encoding="utf-8").strip()
                    existing_pid = int(raw_pid) if raw_pid else -1
                except (OSError, ValueError):
                    existing_pid = -1
                if existing_pid == my_pid:
                    self._lock_file = lock_path
                    self._lock_acquired = True
                    return True
                if existing_pid > 0 and _is_pid_running(existing_pid):
                    return False
                # Stale PID lock file; remove before re-acquiring
                lock_path.unlink(missing_ok=True)

            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"{my_pid}\n".encode("utf-8"))
            finally:
                os.close(fd)
            self._lock_file = lock_path
            self._lock_acquired = True
            return True
        except FileExistsError:
            return False
        except OSError:
            return False

    def release_lock(self) -> None:
        """Release the single-instance PID lock file if owned by this controller."""
        if self._lock_acquired and self._lock_file is not None:
            try:
                if self._lock_file.exists():
                    raw_pid = self._lock_file.read_text(encoding="utf-8").strip()
                    if raw_pid == str(os.getpid()):
                        self._lock_file.unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                self._lock_acquired = False

    def run_preflight(
        self,
        *,
        risk_engine: DeterministicRiskEngine,
        portfolio: IntradayPaperPortfolio,
        credentials_present: bool | None = None,
        connectivity_ok: bool = True,
        clock_synced: bool = True,
        lock_file: Path | None = None,
    ) -> PreflightCheckResult:
        """Run all Stage A preflight safety checks (fail-closed on any failure)."""
        now_ist = datetime.now(IST_TZ)

        runner_live_disabled = True
        try:
            from trader.timing.live_paper_runner import (
                LIVE_TRADING_ENABLED as RUNNER_LIVE_TRADING_ENABLED,
            )

            runner_live_disabled = RUNNER_LIVE_TRADING_ENABLED is False
        except Exception:
            runner_live_disabled = False

        live_trading_disabled = (
            (LIVE_TRADING_ENABLED is False)
            and (RECORDER_LIVE_TRADING_ENABLED is False)
            and runner_live_disabled
        )
        tz_key = getattr(IST_TZ, "key", str(IST_TZ))
        timezone_ist_valid = (
            tz_key == "Asia/Kolkata"
            and now_ist.utcoffset() == timedelta(hours=5, minutes=30)
        )
        nse_trading_day = is_nse_trading_day(self.session_date)
        credentials_valid = self._check_credentials(credentials_present)
        risk_engine_active = (
            risk_engine is not None
            and risk_engine.state == RiskState.ACTIVE
            and portfolio is not None
        )
        storage_ok = self._check_storage_and_disk()
        lock_ok = self._acquire_single_instance_lock(lock_file)

        checks: dict[str, bool] = {
            "live_trading_disabled": live_trading_disabled,
            "timezone_ist_valid": timezone_ist_valid,
            "nse_trading_day": nse_trading_day,
            "credentials_valid": credentials_valid,
            "connectivity_ok": bool(connectivity_ok),
            "clock_synced": bool(clock_synced),
            "risk_engine_active": risk_engine_active,
            "storage_writable_and_disk_ok": storage_ok,
            "single_instance_lock": lock_ok,
        }
        failure_reasons = tuple(
            f"PREFLIGHT_FAILED:{name}" for name, ok in checks.items() if not ok
        )
        passed = len(failure_reasons) == 0

        result = PreflightCheckResult(
            passed=passed,
            session_date=self.session_date,
            checked_at_ist=now_ist,
            checks=checks,
            failure_reasons=failure_reasons,
        )
        self.preflight_result = result
        if not passed:
            self.stage = SessionStage.HALTED_OR_INTERRUPTED
        return result

    # ------------------------------------------------------------------
    # Stage B: Observation (09:15-09:30 IST, NO_NEW_ENTRIES)
    # ------------------------------------------------------------------

    def start_observation(
        self,
        execution_gate: MandatoryExecutionGate,
        current_time: datetime,
    ) -> AuthorityInstruction:
        """Enter Stage B observation by registering `NO_NEW_ENTRIES` on the execution gate."""
        now_ist = _to_ist(current_time)
        expires_at = datetime.combine(self.session_date, time(16, 0), tzinfo=IST_TZ)
        if expires_at <= now_ist:
            expires_at = now_ist + timedelta(hours=8)

        instr = AuthorityInstruction(
            instruction_id=f"AUTONOMOUS-OBS-{self.session_date.isoformat()}",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.NO_NEW_ENTRIES,
            valid_from=now_ist,
            expires_at=expires_at,
            reason="AUTONOMOUS_STAGE_B_OBSERVATION:NO_NEW_ENTRIES",
            version="1.0",
        )
        execution_gate.set_human_instruction(instr, current_time=now_ist)
        self.stage = SessionStage.OBSERVING_NO_NEW_ENTRIES
        return instr

    # ------------------------------------------------------------------
    # Feed Health & Error Tracking
    # ------------------------------------------------------------------

    def record_tick_health(
        self,
        ts_ist: datetime,
        *,
        execution_gate: MandatoryExecutionGate | None = None,
    ) -> None:
        """Track tick continuity; detect stale feed gaps > `max_stale_seconds`."""
        ts = _to_ist(ts_ist)
        if self.last_tick_ts_ist is not None:
            gap_seconds = (ts - self.last_tick_ts_ist).total_seconds()
            if gap_seconds > self.max_stale_seconds and self.stage in (
                SessionStage.OBSERVING_NO_NEW_ENTRIES,
                SessionStage.AUTONOMOUS_ALLOW_LONG,
            ):
                self.interrupted = True
                if "STALE_FEED_INTERRUPTION" not in self.critical_errors:
                    self.critical_errors.append("STALE_FEED_INTERRUPTION")
                if execution_gate is not None:
                    self.handle_disconnect(
                        execution_gate,
                        ts,
                        "STALE_FEED_INTERRUPTION",
                    )
                else:
                    from_stage = self.stage.value
                    self.stage = SessionStage.HALTED_OR_INTERRUPTED
                    self.transition_audit_log.append(
                        TransitionDecision(
                            transitioned=True,
                            from_stage=from_stage,
                            to_stage=self.stage.value,
                            timestamp_ist=ts,
                            instruction_id=None,
                            reasons=("STALE_FEED_INTERRUPTION",),
                        )
                    )
        self.ticks_observed += 1
        if self.last_tick_ts_ist is None or ts > self.last_tick_ts_ist:
            self.last_tick_ts_ist = ts

    def record_completed_1m_bar(self, ts_close_ist: datetime) -> None:
        """Record a causally closed 1-minute bar and track EOD wind-down transition."""
        ts = _to_ist(ts_close_ist)
        self.completed_1m_bars += 1
        if (
            self.stage == SessionStage.AUTONOMOUS_ALLOW_LONG
            and ts.time() >= INTRADAY_REDUCING_WIND_DOWN_TIME
        ):
            self.stage = SessionStage.EOD_WIND_DOWN

    def record_critical_error(self, reason: str) -> None:
        """Record a critical runtime error that blocks Stage C transition."""
        self.critical_errors.append(str(reason))

    # ------------------------------------------------------------------
    # Stage C: 09:30 IST Autonomous Transition to Standing ALLOW_LONG
    # ------------------------------------------------------------------

    def evaluate_0930_transition(
        self,
        *,
        execution_gate: MandatoryExecutionGate,
        risk_engine: DeterministicRiskEngine,
        portfolio: IntradayPaperPortfolio,
        current_time: datetime,
    ) -> TransitionDecision:
        """Evaluate all 8 mandatory preconditions and transition to `ALLOW_LONG` if all pass."""
        now_ist = _to_ist(current_time)

        if self.stage == SessionStage.AUTONOMOUS_ALLOW_LONG:
            return TransitionDecision(
                transitioned=False,
                from_stage=self.stage.value,
                to_stage=self.stage.value,
                timestamp_ist=now_ist,
                instruction_id=None,
                reasons=("ALREADY_IN_AUTONOMOUS_ALLOW_LONG",),
            )

        failure_reasons: list[str] = []

        # 1. Preflight passed
        if self.preflight_result is None or not self.preflight_result.passed:
            failure_reasons.append("PRECONDITION_1_FAILED:PREFLIGHT_NOT_PASSED")

        # 2. NSE trading day
        if not is_nse_trading_day(self.session_date):
            failure_reasons.append("PRECONDITION_2_FAILED:NOT_NSE_TRADING_DAY")

        # 3. Time >= transition_time_ist AND observation bars >= min AND not stale/interrupted
        if now_ist.time() < self.transition_time_ist:
            failure_reasons.append(
                f"PRECONDITION_3_FAILED:BEFORE_TRANSITION_TIME:{now_ist.time().isoformat()}"
            )
        if self.completed_1m_bars < self.min_observation_bars:
            failure_reasons.append(
                f"PRECONDITION_3_FAILED:INSUFFICIENT_OBSERVATION_BARS:{self.completed_1m_bars}<{self.min_observation_bars}"
            )
        if self.interrupted:
            failure_reasons.append("PRECONDITION_3_FAILED:FEED_INTERRUPTED_OR_STALE")

        # 4. Zero critical errors
        if len(self.critical_errors) > 0:
            failure_reasons.append(
                f"PRECONDITION_4_FAILED:CRITICAL_ERRORS:{','.join(self.critical_errors)}"
            )

        # 5. Risk engine ACTIVE
        if risk_engine.state != RiskState.ACTIVE:
            failure_reasons.append(
                f"PRECONDITION_5_FAILED:RISK_ENGINE_NOT_ACTIVE:{risk_engine.state.value}"
            )

        # 6. Portfolio cash > 0, no margin call, cash reservation ledger valid
        if float(portfolio.cash) <= 0.0:
            failure_reasons.append("PRECONDITION_6_FAILED:PORTFOLIO_CASH_NON_POSITIVE")
        if bool(getattr(portfolio, "margin_call_triggered", False)):
            failure_reasons.append("PRECONDITION_6_FAILED:MARGIN_CALL_TRIGGERED")
        if float(risk_engine.cash_ledger.total_reserved_inr) < 0.0:
            failure_reasons.append("PRECONDITION_6_FAILED:NEGATIVE_CASH_RESERVATION")

        # 7. Active Human instruction is NOT STOP_ALL / PAUSE / EXIT_ONLY AND stage is OBSERVING
        if self.stage != SessionStage.OBSERVING_NO_NEW_ENTRIES:
            failure_reasons.append(
                f"PRECONDITION_7_FAILED:STAGE_NOT_OBSERVING:{self.stage.value}"
            )
        scopes_to_check = ("*",) + self.symbols
        for scope in scopes_to_check:
            active_human = execution_gate.get_active_human_instruction(scope, now_ist)
            if (
                active_human is not None
                and active_human.permission in _BLOCKED_HUMAN_OVERRIDE_PERMS
            ):
                failure_reasons.append(
                    f"PRECONDITION_7_FAILED:HUMAN_OVERRIDE_ACTIVE:{active_human.permission.value}"
                )
                break

        # 8. Standing permission strictly ALLOW_LONG
        if self.standing_permission != PermissionType.ALLOW_LONG:
            failure_reasons.append(
                f"PRECONDITION_8_FAILED:INVALID_STANDING_PERMISSION:{self.standing_permission.value}"
            )

        if failure_reasons:
            decision = TransitionDecision(
                transitioned=False,
                from_stage=self.stage.value,
                to_stage=self.stage.value,
                timestamp_ist=now_ist,
                instruction_id=None,
                reasons=tuple(failure_reasons),
            )
            self.transition_audit_log.append(decision)
            return decision

        expires_at = datetime.combine(self.session_date, time(16, 0), tzinfo=IST_TZ)
        if expires_at <= now_ist:
            expires_at = now_ist + timedelta(hours=8)

        instr = AuthorityInstruction(
            instruction_id=f"AUTONOMOUS-LONG-{self.session_date.isoformat()}",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=now_ist,
            expires_at=expires_at,
            reason="STANDING_LONG_ONLY_0930_TRANSITION:all_8_preconditions_verified",
            version="1.0",
        )
        reg = execution_gate.set_human_instruction(instr, current_time=now_ist)
        if not reg.accepted:
            decision = TransitionDecision(
                transitioned=False,
                from_stage=self.stage.value,
                to_stage=self.stage.value,
                timestamp_ist=now_ist,
                instruction_id=None,
                reasons=(f"GATE_REGISTRATION_REJECTED:{reg.reason}",),
            )
            self.transition_audit_log.append(decision)
            return decision

        from_stage = self.stage.value
        self.stage = SessionStage.AUTONOMOUS_ALLOW_LONG
        decision = TransitionDecision(
            transitioned=True,
            from_stage=from_stage,
            to_stage=self.stage.value,
            timestamp_ist=now_ist,
            instruction_id=instr.instruction_id,
            reasons=("ALL_8_PRECONDITIONS_VERIFIED",),
        )
        self.transition_audit_log.append(decision)
        return decision

    # ------------------------------------------------------------------
    # Disconnect / Interruption Safe Downgrade
    # ------------------------------------------------------------------

    def handle_disconnect(
        self,
        execution_gate: MandatoryExecutionGate,
        current_time: datetime,
        reason: str,
    ) -> AuthorityInstruction:
        """Downgrade Human permission to `NO_NEW_ENTRIES` while preserving safe exits."""
        now_ist = _to_ist(current_time)
        self.interrupted = True
        from_stage = self.stage.value
        self.stage = SessionStage.HALTED_OR_INTERRUPTED

        existing = execution_gate.get_active_human_instruction("*", now_ist)
        valid_from = now_ist
        if existing is not None and existing.valid_from > valid_from:
            valid_from = existing.valid_from

        expires_at = datetime.combine(self.session_date, time(16, 0), tzinfo=IST_TZ)
        if expires_at <= valid_from:
            expires_at = valid_from + timedelta(hours=8)

        instr = AuthorityInstruction(
            instruction_id=f"AUTONOMOUS-DISC-{self.session_date.isoformat()}-{uuid.uuid4().hex[:8]}",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.NO_NEW_ENTRIES,
            valid_from=valid_from,
            expires_at=expires_at,
            reason=f"DISCONNECT_SAFE_DOWNGRADE:{reason}",
            version="1.0",
        )
        execution_gate.set_human_instruction(instr, current_time=now_ist)
        self.transition_audit_log.append(
            TransitionDecision(
                transitioned=True,
                from_stage=from_stage,
                to_stage=self.stage.value,
                timestamp_ist=now_ist,
                instruction_id=instr.instruction_id,
                reasons=(f"DISCONNECT:{reason}",),
            )
        )
        return instr

    # ------------------------------------------------------------------
    # Atomic Checkpoint Persistence & Restart Recovery
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        portfolio: IntradayPaperPortfolio,
        executed_signal_keys: set[str],
        trade_count: int,
    ) -> Path:
        """Atomically persist session checkpoint via `os.replace` for restart recovery."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.output_dir / "session_checkpoint.json"
        tmp_path = self.output_dir / f".session_checkpoint.{os.getpid()}.tmp"

        open_positions: dict[str, dict[str, Any]] = {}
        for sym, pos in portfolio.positions.items():
            if pos is not None and pos.quantity > 0:
                open_positions[sym] = {
                    "symbol": pos.symbol,
                    "side": getattr(pos, "side", "LONG"),
                    "quantity": int(pos.quantity),
                    "entry_price": float(pos.entry_price),
                    "stop_loss": float(pos.stop_loss) if pos.stop_loss is not None else None,
                    "target": float(pos.target) if pos.target is not None else None,
                }

        payload: dict[str, Any] = {
            "schema_version": "autonomous_long_only_checkpoint_v1",
            "session_date": self.session_date.isoformat(),
            "provider": self.provider,
            "stage": self.stage.value,
            "interrupted": self.interrupted,
            "ticks_observed": self.ticks_observed,
            "completed_1m_bars": self.completed_1m_bars,
            "trade_count": int(trade_count),
            "cash_inr": round(float(portfolio.cash), 2),
            "executed_signal_keys": sorted(str(k) for k in executed_signal_keys),
            "open_positions": open_positions,
            "saved_at_ist": datetime.now(IST_TZ).isoformat(),
        }
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, target_path)
        return target_path

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load today's checkpoint if present and matching `self.session_date`."""
        target_path = self.output_dir / "session_checkpoint.json"
        if not target_path.exists():
            return None
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("session_date") != self.session_date.isoformat():
            return None
        return data

    # ------------------------------------------------------------------
    # Summary Export for live_paper_state.json
    # ------------------------------------------------------------------

    def summary_dict(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date.isoformat(),
            "provider": self.provider,
            "stage": self.stage.value,
            "standing_permission": self.standing_permission.value,
            "interrupted": self.interrupted,
            "ticks_observed": self.ticks_observed,
            "completed_1m_bars": self.completed_1m_bars,
            "critical_errors": list(self.critical_errors),
            "preflight": (
                self.preflight_result.to_dict()
                if self.preflight_result is not None
                else None
            ),
            "transitions": [t.to_dict() for t in self.transition_audit_log],
        }
