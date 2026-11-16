"""Phase 5.3 Forward Validation & Paper Research Portfolio Runner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.news_intel.forward_validation import (
    build_forward_validation_payload,
    evaluate_forward_outcomes,
)
from backend.storage.duckdb_store import DuckDBStore


def main() -> None:
    st = DuckDBStore()
    evaluated = evaluate_forward_outcomes(st)
    payload = build_forward_validation_payload(store=st, persist=True)
    print(
        f"Phase 5.3 Forward Validation updated: "
        f"news(valid={payload.valid_news_events}, quarantined={payload.quarantined_news_events}), "
        f"snapshots={payload.total_immutable_snapshots}, "
        f"outcomes(matured={payload.matured_outcomes_total}, pending={payload.pending_outcomes_total}), "
        f"evaluated_records={len(evaluated)}"
    )


if __name__ == "__main__":
    main()
