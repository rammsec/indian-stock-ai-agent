"""Immutable, append-only, hash-chained JSONL decision log.

Records (``record_type`` = ``decision`` | ``settlement``) are chained via
``prev_hash`` / ``record_hash`` (sha256 of the canonical JSON of the record
minus ``record_hash``). Existing lines are never rewritten. Self-contained.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

from backend.phase6.contracts import PHASE6_SCHEMA_VERSION

GENESIS = "0" * 64
IST = ZoneInfo("Asia/Kolkata")
HORIZONS = (1, 5, 10)


class DecisionLogError(Exception):
    pass


class SettlementError(DecisionLogError):
    pass


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash_record(rec: dict) -> str:
    return hashlib.sha256(_canon({k: v for k, v in rec.items() if k != "record_hash"}).encode()).hexdigest()


def _iso(v) -> str:
    return v.isoformat() if isinstance(v, (datetime, date)) else str(v)


def dedup_key(symbol: str, data_cutoff: Any, input_hash: str, prompt_hashes: dict,
              config_snapshot: dict) -> str:
    """Key over (symbol, data-cutoff date, input hash, prompt+config hash)."""
    cutoff_date = _iso(data_cutoff)[:10]
    pc = hashlib.sha256(_canon({"p": prompt_hashes, "c": config_snapshot}).encode()).hexdigest()
    return f"{symbol}|{cutoff_date}|{input_hash}|{pc}"


class DecisionLog:
    def __init__(self, directory: str | Path = "data/phase6/decisions"):
        self.dir = Path(directory)
        self.path = self.dir / "decisions.jsonl"
        self._lock = threading.Lock()

    # ---- reading ----
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """Return (ok, error). Detects edits, deletions, reordering."""
        prev = GENESIS
        try:
            recs = self.records()
        except json.JSONDecodeError as e:
            return False, f"unparseable record: {e}"
        for i, r in enumerate(recs):
            if r.get("prev_hash") != prev:
                return False, f"record {i}: prev_hash mismatch (chain broken)"
            if _hash_record(r) != r.get("record_hash"):
                return False, f"record {i}: record_hash mismatch (content tampered)"
            prev = r["record_hash"]
        return True, None

    # ---- writing ----
    def _append_raw(self, rec: dict) -> dict:
        recs = self.records()
        rec["prev_hash"] = recs[-1]["record_hash"] if recs else GENESIS
        rec["record_hash"] = _hash_record(rec)
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:  # append-only
            f.write(_canon(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def append(self, *, run_id: str, symbol: str, as_of: str, analysis_timestamp,
               data_cutoff_timestamp, config_snapshot: dict, prompt_hashes: dict,
               input_hash: str, report) -> tuple[dict, bool]:
        """Append a decision. Returns ``(record, created)``.

        A duplicate (same dedup key) returns the existing record with
        ``created=False``. Reusing a ``run_id`` for different content raises.
        """
        key = dedup_key(symbol, data_cutoff_timestamp, input_hash, prompt_hashes, config_snapshot)
        report_json = json.loads(report.model_dump_json()) if hasattr(report, "model_dump_json") else report
        with self._lock:
            for r in self.records():
                if r.get("record_type") != "decision":
                    continue
                if r["dedup_key"] == key:
                    return r, False
                if r["run_id"] == run_id:
                    raise DecisionLogError(f"run_id '{run_id}' already exists with different content; "
                                           "refusing to overwrite")
            rec = {
                "record_type": "decision", "schema_version": PHASE6_SCHEMA_VERSION,
                "run_id": run_id, "symbol": symbol, "as_of": as_of,
                "analysis_timestamp": _iso(analysis_timestamp),
                "data_cutoff_timestamp": _iso(data_cutoff_timestamp),
                "config_snapshot": config_snapshot, "prompt_hashes": prompt_hashes,
                "input_hash": input_hash, "dedup_key": key, "report": report_json,
            }
            return self._append_raw(rec), True

    def settle(self, record: dict, price_series, horizons: Sequence[int] = HORIZONS) -> dict:
        """Append a settlement record for ``record`` (original never mutated).

        ``price_series``: iterable of ``{"date", "open", "close"}`` dicts (or a
        DataFrame with those columns). Entry = open of the first trading date
        strictly AFTER the decision date (IST); exit for horizon H = close of the
        H-th trading day counting the entry day as day 1. Rows on/before the
        decision date are ignored. Horizons without enough data are ``PENDING``.
        """
        if record.get("record_type") != "decision":
            raise SettlementError("can only settle decision records")
        if hasattr(price_series, "to_dict"):
            price_series = price_series.to_dict("records")
        dec_ts = datetime.fromisoformat(record["analysis_timestamp"])
        dec_date = dec_ts.astimezone(IST).date()
        rows = sorted(
            ({"date": date.fromisoformat(str(r["date"])[:10]), "open": float(r["open"]),
              "close": float(r["close"])} for r in price_series), key=lambda r: r["date"])
        post = [r for r in rows if r["date"] > dec_date]
        if not post:
            raise SettlementError("no price rows strictly after the decision date")
        entry = post[0]
        outcomes = {}
        for h in horizons:
            if len(post) >= h:
                ex = post[h - 1]
                outcomes[f"{h}D"] = {
                    "status": "SETTLED", "entry_date": entry["date"].isoformat(),
                    "entry_open": entry["open"], "exit_date": ex["date"].isoformat(),
                    "exit_close": ex["close"], "return": ex["close"] / entry["open"] - 1.0}
            else:
                outcomes[f"{h}D"] = {"status": "PENDING"}
        with self._lock:
            for r in self.records():
                if (r.get("record_type") == "settlement" and r["decision_hash"] == record["record_hash"]
                        and r["outcomes"] == outcomes):
                    return r
            return self._append_raw({
                "record_type": "settlement", "schema_version": PHASE6_SCHEMA_VERSION,
                "decision_hash": record["record_hash"], "run_id": record["run_id"],
                "symbol": record["symbol"], "decision_date": dec_date.isoformat(),
                "outcomes": outcomes})
