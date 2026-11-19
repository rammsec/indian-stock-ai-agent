import json

from backend.phase6.config import Phase6Config
from backend.phase6.decision_log import DecisionLog
from backend.phase6.graph import ShadowResearchGraph
from p6helpers import StubManager, loader, registry


def graph(tmp_path, reg, **kw):
    cfg = Phase6Config(namespace_dir=str(tmp_path / "ns"))
    return ShadowResearchGraph(reg, cfg, evidence_loader=loader, **kw)


def test_happy_path_complete_and_checkpoints(tmp_path):
    g = graph(tmp_path, registry())
    s = g.run("TCS", "2025-01-15")
    assert s.status == "COMPLETE"
    assert s.completed_nodes == ["load_evidence", "analysts", "debate", "research_manager",
                                 "risk_manager", "persist"]
    assert len(s.analyst_reports) == 6 and s.final_report.overall_conclusion == "FAVOURABLE"
    cp = json.loads(g.checkpoint_path("TCS-2025-01-15").read_text())
    assert cp["status"] == "COMPLETE"


def test_analyst_failure_is_isolated(tmp_path):
    s = graph(tmp_path, registry(failing=("news", "promoter"))).run("TCS", "2025-01-15")
    assert s.status == "PARTIAL"
    assert sorted(s.missing_analysts) == ["news", "promoter"]
    assert set(s.analyst_reports) == {"fundamental", "technical", "competitor", "quantitative"}
    assert any("news" in w and "boom" in w for w in s.warnings)
    assert s.final_report is not None


def test_all_analysts_fail_is_failed(tmp_path):
    names = ("fundamental", "technical", "news", "promoter", "competitor", "quantitative")
    s = graph(tmp_path, registry(failing=names)).run("TCS", "2025-01-15")
    assert s.status == "FAILED" and s.final_report is None and s.bull is None


def test_bull_bear_identical_input_and_no_cross_visibility(tmp_path):
    reg = registry()
    s = graph(tmp_path, reg).run("TCS", "2025-01-15")
    assert s.bull_input_hash == s.bear_input_hash is not None
    assert reg.bull.seen is reg.bear.seen                       # same object
    assert reg.bull.seen.input_hash() == s.bull_input_hash
    assert not hasattr(reg.bull.seen, "bull") and not hasattr(reg.bear.seen, "bear")
    blob = reg.bull.seen.model_dump_json()
    assert "BULL thesis" not in blob and "BEAR thesis" not in blob
    assert len(reg.bull.seen.analyst_reports) == 6
    # manager does see both
    assert reg.manager.seen.bull is not None and reg.manager.seen.bear is not None


def test_debater_failure_degrades_but_continues(tmp_path):
    s = graph(tmp_path, registry(bull_fail=True)).run("TCS", "2025-01-15")
    assert s.status == "PARTIAL" and s.bull is None and s.bear is not None
    assert s.final_report is not None


def test_risk_reject_overrides_manager(tmp_path):
    s = graph(tmp_path, registry(verdict="REJECT", conclusion="FAVOURABLE")).run("TCS", "2025-01-15")
    assert s.manager_report.overall_conclusion == "FAVOURABLE"
    assert s.final_report.overall_conclusion == "REJECTED_BY_RISK"
    assert s.final_report.risk_assessment.rejection_reasons == ["too risky"]


def test_risk_approve_keeps_conclusion_and_risk_failure_downgrades(tmp_path):
    s = graph(tmp_path, registry(verdict="APPROVE")).run("TCS", "2025-01-15")
    assert s.final_report.overall_conclusion == "FAVOURABLE"
    s2 = graph(tmp_path, registry(risk_fail=True)).run("TCS", "2025-01-16", run_id="r2")
    assert s2.final_report.overall_conclusion == "INSUFFICIENT_EVIDENCE" and s2.status == "PARTIAL"


def test_checkpoint_resume_skips_completed_nodes(tmp_path):
    calls = {"n": 0}
    reg = registry()
    orig = reg.analysts["news"].run

    def counting(pkg, ctx):
        calls["n"] += 1
        return orig(pkg, ctx)
    reg.analysts["news"].run = counting

    class Exploding(StubManager):
        def run(self, pkg, ctx):
            raise RuntimeError("manager crash")
    good = reg.manager
    reg.manager = Exploding()
    g = graph(tmp_path, reg)
    s1 = g.run("TCS", "2025-01-15", run_id="r")
    assert s1.status == "FAILED" and "research_manager" not in s1.completed_nodes
    assert "debate" in g.load_checkpoint("r").completed_nodes
    reg.manager = good
    s2 = graph(tmp_path, reg).run("TCS", "2025-01-15", run_id="r")
    assert s2.status == "COMPLETE" and calls["n"] == 1   # analysts NOT re-run
    assert s2.final_report is not None


def test_persist_to_decision_log_and_duplicate(tmp_path):
    log = DecisionLog(tmp_path / "dec")
    s = graph(tmp_path, registry(), decision_log=log).run("TCS", "2025-01-15", run_id="a")
    assert s.decision_record_hash and len(log.records()) == 1
    s2 = graph(tmp_path, registry(), decision_log=log).run("TCS", "2025-01-15", run_id="b")
    assert len(log.records()) == 1 and s2.decision_record_hash == s.decision_record_hash
    assert log.verify_chain() == (True, None)
