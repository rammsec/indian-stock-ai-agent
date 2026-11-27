# BharatAlpha — Continuous QA & Monitoring Design (`BHARATALPHA-AUTO-QA-REMEDIATION-001`)

## 1. Configured & Verified Workflows (Read-Only, Event-Triggered & Scheduled)

Continuous monitoring is **actively configured** via two GitHub Actions workflows (both strictly read-only with `permissions: contents: read`, zero automatic commits, zero automatic merges, and zero uncontrolled repair loops):

### A. Pull Request & Push Gate — `.github/workflows/intraday-lab-tests.yml`
- **Triggers**: `pull_request` (to `main`), `push` (to `main` and feature branches), `workflow_dispatch`.
- **Checks Executed**:
  1. Python 3.12 + Node 20 environment setup.
  2. JavaScript syntax checks (`node -c docs/candle-agent/engine.js docs/trader/app.js docs/lab/app.js`).
  3. Full `pytest tests/intraday/ tests/phase6/` (459+ tests including REMED-01 authority/risk/concurrency/parity suite and the 26-test autonomous QA remediation suite).
  4. **Post-Test Clean-Tree & Sealed-Holdout Immutability Gate**:
     - `git diff --exit-code data/lab/sealed_holdout_registry.json`
     - `git diff --exit-code data/lab/ docs/lab/data/`

### B. Scheduled Daily Pre-Market QA Sentinel — `.github/workflows/bharatalpha-continuous-qa.yml`
- **Triggers**: Daily cron `30 2 * * 1-5` (`02:30 UTC` / `08:00 IST`, weekdays pre-market) + manual `workflow_dispatch`.
- **Permissions**: `contents: read` (strictly read-only; never mutates repo or opens unapproved PRs).
- **Checks Executed**:
  1. JavaScript syntax & inline `<script>` verification across `docs/candle-agent/`, `docs/trader/`, `docs/lab/`.
  2. Authority, risk, concurrency, Python↔JS parity & AST non-bypass regression (`tests/intraday/test_remed01_authority_risk_capital.py` + `tests/intraday/test_autonomous_qa_remediation.py`).
  3. Sealed holdout & tracked lab JSON immutability check.
  4. Paper-only safety scan (confirming zero broker order-placement endpoints).
  5. Uploads structured JSON + Markdown QA health summary as a GitHub Actions artifact (`bharatalpha-continuous-qa-report`).

---

## 2. Incident Triage & Remediation Escalation Protocol

When either workflow flags a failure:
1. Capture failing run URL, commit SHA, and reproduction trace from the uploaded QA artifact.
2. Classify severity (`P0` crash/bypass/exposure, `P1` lifecycle/risk/data integrity, `P2` functional/parity, `P3` hygiene) and check `BHARATALPHA_DEFECT_REGISTER.md` for duplicates.
3. Require an explicit, scoped remediation task authorization before code changes are pushed.
4. Enforce the **Implement → Targeted & Full Regression → Independent Quality Guardian Audit → Human Merge Approval** loop — never self-certify or auto-merge.
