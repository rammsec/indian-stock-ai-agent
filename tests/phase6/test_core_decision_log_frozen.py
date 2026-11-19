import copy
from datetime import datetime, timezone

import pytest

from backend.phase6.config import Phase6Config
from backend.phase6.contracts import ResearchManagerReport
from backend.phase6.decision_log import DecisionLog, DecisionLogError, SettlementError
from backend.phase6.frozen_guard import (FrozenFileChanged, FROZEN_PATHS, assert_unchanged,
                                         snapshot_frozen_files)
from p6helpers import ANA, CUT, base


def report():
    return ResearchManagerReport(**base("research_manager"), overall_conclusion="NEUTRAL")


def add(log, run_id="r1", input_hash="h1", ts=ANA):
    cfg = Phase6Config().snapshot()
    return log.append(run_id=run_id, symbol="TCS", as_of="2025-01-15", analysis_timestamp=ts,
                      data_cutoff_timestamp=CUT, config_snapshot=cfg,
                      prompt_hashes={"bull@v1": "abc"}, input_hash=input_hash, report=report())


def test_append_chain_and_verify(tmp_path):
    log = DecisionLog(tmp_path)
    r1, c1 = add(log, "r1", "h1")
    r2, c2 = add(log, "r2", "h2")
    assert c1 and c2 and r2["prev_hash"] == r1["record_hash"]
    assert r1["report"]["overall_conclusion"] == "NEUTRAL" and r1["schema_version"] == "p6-1.0.0"
    assert log.verify_chain() == (True, None)


def test_duplicate_returns_existing(tmp_path):
    log = DecisionLog(tmp_path)
    r1, _ = add(log, "r1", "h1")
    r1b, created = add(log, "r-other", "h1")
    assert not created and r1b["record_hash"] == r1["record_hash"] and len(log.records()) == 1


def test_run_id_reuse_with_different_content_refused(tmp_path):
    log = DecisionLog(tmp_path)
    add(log, "r1", "h1")
    with pytest.raises(DecisionLogError, match="overwrite"):
        add(log, "r1", "h2")
    assert len(log.records()) == 1


def test_tamper_detected(tmp_path):
    log = DecisionLog(tmp_path)
    add(log, "r1", "h1"); add(log, "r2", "h2")
    txt = log.path.read_text()
    log.path.write_text(txt.replace("NEUTRAL", "FAVOURABLE", 1))
    ok, err = log.verify_chain()
    assert not ok and "tampered" in err


def test_deletion_detected(tmp_path):
    log = DecisionLog(tmp_path)
    add(log, "r1", "h1"); add(log, "r2", "h2"); add(log, "r3", "h3")
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    ok, err = log.verify_chain()
    assert not ok and "chain broken" in err


def series(days, start=100.0):
    return [{"date": d, "open": start + i, "close": start + i + 0.5} for i, d in enumerate(days)]


DAYS = ["2025-01-13", "2025-01-14", "2025-01-15",  # on/before decision (2025-01-15 IST)
        "2025-01-16", "2025-01-17", "2025-01-20", "2025-01-21", "2025-01-22", "2025-01-23"]


def test_settlement_next_open_to_close_and_no_mutation(tmp_path):
    log = DecisionLog(tmp_path)
    rec, _ = add(log)
    before = copy.deepcopy(rec)
    s = log.settle(rec, series(DAYS))
    assert rec == before and log.records()[0] == before            # original untouched
    o = s["outcomes"]
    assert o["1D"]["entry_date"] == "2025-01-16" and o["1D"]["entry_open"] == 103.0
    assert o["1D"]["exit_close"] == 103.5
    assert o["5D"]["exit_date"] == "2025-01-22"
    assert o["5D"]["return"] == pytest.approx(107.5 / 103.0 - 1)
    assert o["10D"]["status"] == "PENDING"
    assert len(log.records()) == 2 and log.verify_chain()[0]
    assert log.settle(rec, series(DAYS))["record_hash"] == s["record_hash"]  # idempotent


def test_settlement_ignores_pre_decision_rows_and_rejects_none_after(tmp_path):
    log = DecisionLog(tmp_path)
    rec, _ = add(log)
    pre = series(DAYS)
    for r in pre[:3]:
        r["open"] = r["close"] = 9999.0                              # poison pre-decision rows
    o = log.settle(rec, pre)["outcomes"]
    assert o["1D"]["entry_open"] == 103.0
    with pytest.raises(SettlementError):
        log.settle(rec, series(DAYS[:3]))
    with pytest.raises(SettlementError):
        log.settle({"record_type": "settlement"}, series(DAYS))


def test_settlement_uses_ist_date_of_decision(tmp_path):
    log = DecisionLog(tmp_path)
    # 2025-01-15 20:00 UTC == 2025-01-16 01:30 IST -> decision date is the 16th
    rec, _ = add(log, ts=datetime(2025, 1, 15, 20, tzinfo=timezone.utc))
    assert log.settle(rec, series(DAYS))["outcomes"]["1D"]["entry_date"] == "2025-01-17"


def test_frozen_guard_detects_modification(tmp_path):
    (tmp_path / "backend/prediction").mkdir(parents=True)
    f = tmp_path / "backend/prediction/m.py"
    f.write_text("x = 1\n")
    (tmp_path / "requirements.txt").write_text("a\n")
    snap = snapshot_frozen_files(root=tmp_path)
    assert set(snap) == {"backend/prediction/m.py", "requirements.txt"}
    assert_unchanged(snap, root=tmp_path)
    f.write_text("x = 2\n")
    with pytest.raises(FrozenFileChanged, match="modified: backend/prediction/m.py"):
        assert_unchanged(snap, root=tmp_path)
    f.write_text("x = 1\n")
    assert_unchanged(snap, root=tmp_path)
    f.unlink()
    with pytest.raises(FrozenFileChanged, match="deleted"):
        assert_unchanged(snap, root=tmp_path)


def test_frozen_default_paths_cover_repo():
    assert "requirements.txt" in FROZEN_PATHS and "backend/prediction/*.py" in FROZEN_PATHS
    snap = snapshot_frozen_files()          # repo root (cwd = rootdir)
    assert any(k.startswith("backend/prediction/") for k in snap)
    assert_unchanged(snap)
