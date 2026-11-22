"""Strict deterministic parsing, hashing, and atomic I/O utilities for BharatAlpha."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from trader.config import IST_TZ


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming 64 KiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    """Return current HEAD commit SHA, or 'UNKNOWN' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def parse_iso_to_ist(raw: Any, label: str) -> datetime:
    """Parse an ISO-8601 timestamp string to IST, rejecting empty or timezone-naive values."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing or empty timestamp for {label}")
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"Timezone-naive timestamp rejected for {label}: {raw!r}")
    return dt.astimezone(IST_TZ)


def parse_obs_date(raw: Any, label: str) -> date:
    """Parse observation_date safely: accept strict YYYY-MM-DD or tz-aware ISO timestamp normalized to IST."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing or empty observation_date for {label}")
    text = raw.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return date.fromisoformat(text)
    return parse_iso_to_ist(text, label).date()


def parse_finite_float(raw: Any, label: str) -> float:
    """Parse a finite float, rejecting None, bool, non-numeric, NaN, and +/-Inf."""
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"Missing or invalid numeric field for {label}: {raw!r}")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric field for {label}: {raw!r}") from exc
    if not math.isfinite(val):
        raise ValueError(f"Non-finite numeric value for {label}: {raw!r}")
    return val


def _reject_non_standard_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON numeric constant rejected: {token!r}")


def load_json_strict(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Load a JSON object strictly (rejecting NaN/Infinity constants) and return (payload, sha256)."""
    if not path.exists():
        raise FileNotFoundError(f"Required input missing ({label}): {path}")
    file_hash = sha256_file(path)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_standard_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed JSON in {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid root JSON type in {label} ({path}): expected object")
    return payload, file_hash


def write_report_atomic(report: dict[str, Any], output_path: Path) -> Path:
    """Atomically write report JSON using a PID/TID-unique temp file and os.replace."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(
        f"{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def compute_candle_series_hash(candles: Sequence[Any]) -> str:
    """Compute deterministic SHA-256 checksum across candle timestamps and OHLCV."""
    if not candles:
        return "EMPTY_DATASET_HASH_0000000000000000"
    raw_str = ";".join(
        f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
        for c in candles
    )
    return hashlib.sha256(raw_str.encode()).hexdigest()
