"""Parquet + DuckDB Analytical Storage Engine (Phase 1 — Section 6).

Manages the 5 separated storage layers:
- `/data/raw/`
- `/data/processed/`
- `/data/features/`
- `/data/predictions/`
- `/data/backtests/`

Stores normalized datasets in ZSTD-compressed Apache Parquet files and registers
DuckDB SQL views for zero-cost analytical queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ..config import (
    BACKTESTS_DATA_DIR,
    DUCKDB_PATH,
    FEATURES_DATA_DIR,
    PREDICTIONS_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from ..models import (
    DataQualityReport,
    DataRecord,
    EquityDailyBar,
    IndexDailyBar,
    MarketRegimeSnapshot,
)


def ensure_storage_directories() -> dict[str, Path]:
    """Create the 5 required `/data/*` storage directories and `.gitkeep` markers."""
    dirs = {
        "raw": RAW_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "features": FEATURES_DATA_DIR,
        "predictions": PREDICTIONS_DATA_DIR,
        "backtests": BACKTESTS_DATA_DIR,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("")
    return dirs


class DuckDBStore:
    """Analytical Parquet + DuckDB manager for Indian equity market data."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        features_dir: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        ensure_storage_directories()
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = features_dir or FEATURES_DATA_DIR
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or DUCKDB_PATH

        self.equities_parquet = self.processed_dir / "equities_daily.parquet"
        self.indices_parquet = self.processed_dir / "indices_daily.parquet"
        self.provenance_parquet = self.processed_dir / "provenance_records.parquet"
        self.regime_parquet = self.processed_dir / "market_regimes.parquet"
        self.quality_parquet = self.processed_dir / "data_quality.parquet"
        self.features_parquet = self.features_dir / "daily_features.parquet"
        self.feature_catalog_json = self.features_dir / "feature_catalog.json"

    @staticmethod
    def _upsert_parquet(
        parquet_path: Path,
        new_rows: list[dict[str, Any]],
        dedup_cols: list[str],
        sort_cols: list[str],
    ) -> int:
        if not new_rows:
            return 0
        df_new = pd.DataFrame(new_rows)
        if parquet_path.exists():
            try:
                df_existing = pd.read_parquet(parquet_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
        df_combined = df_combined.sort_values(sort_cols).reset_index(drop=True)
        df_combined.to_parquet(parquet_path, index=False, compression="zstd")
        return len(df_combined)

    def upsert_equity_bars(self, bars: list[EquityDailyBar]) -> int:
        """Upsert daily equity OHLCV + delivery bars into `equities_daily.parquet`."""
        rows = [b.model_dump() for b in bars]
        return self._upsert_parquet(
            self.equities_parquet,
            rows,
            dedup_cols=["symbol", "observation_date"],
            sort_cols=["symbol", "observation_date"],
        )

    def upsert_index_bars(self, bars: list[IndexDailyBar]) -> int:
        """Upsert daily market & sector index bars into `indices_daily.parquet`."""
        rows = [b.model_dump() for b in bars]
        return self._upsert_parquet(
            self.indices_parquet,
            rows,
            dedup_cols=["index_code", "observation_date"],
            sort_cols=["index_code", "observation_date"],
        )

    def upsert_provenance_records(self, records: list[DataRecord]) -> int:
        """Upsert canonical provenance records into `provenance_records.parquet`."""
        rows = []
        for r in records:
            d = r.model_dump()
            d["value"] = str(d["value"]) if d["value"] is not None else None
            rows.append(d)
        return self._upsert_parquet(
            self.provenance_parquet,
            rows,
            dedup_cols=["symbol", "metric", "observation_date"],
            sort_cols=["symbol", "observation_date", "metric"],
        )

    def upsert_market_regime(self, snapshot: MarketRegimeSnapshot) -> int:
        """Upsert market regime snapshot into `market_regimes.parquet`."""
        d = snapshot.model_dump()
        # Store complex list fields as JSON strings for clean Parquet schema
        import json

        d["regime_rationale"] = json.dumps(d["regime_rationale"])
        d["sector_performances"] = json.dumps(d["sector_performances"])
        return self._upsert_parquet(
            self.regime_parquet,
            [d],
            dedup_cols=["observation_date"],
            sort_cols=["observation_date"],
        )

    def upsert_data_quality(self, report: DataQualityReport) -> int:
        """Upsert per-stock data quality report into `data_quality.parquet`."""
        import json

        d = report.model_dump()
        d["critical_failures"] = json.dumps(d["critical_failures"])
        d["metric_statuses"] = json.dumps(d["metric_statuses"])
        return self._upsert_parquet(
            self.quality_parquet,
            [d],
            dedup_cols=["symbol"],
            sort_cols=["symbol"],
        )

    def upsert_feature_matrix(self, df: pd.DataFrame) -> int:
        """Upsert historical + current feature rows into `/data/features/daily_features.parquet`."""
        if df.empty:
            return 0
        if self.features_parquet.exists():
            try:
                df_existing = pd.read_parquet(self.features_parquet)
                df_combined = pd.concat([df_existing, df], ignore_index=True)
            except Exception:
                df_combined = df.copy()
        else:
            df_combined = df.copy()

        df_combined = df_combined.drop_duplicates(subset=["symbol", "observation_date"], keep="last")
        df_combined = df_combined.sort_values(["symbol", "observation_date"]).reset_index(drop=True)
        df_combined.to_parquet(self.features_parquet, index=False, compression="zstd")
        return len(df_combined)

    def query_df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Execute an analytical SQL query over the Parquet views using in-memory DuckDB."""
        con = duckdb.connect(":memory:")
        try:
            if self.equities_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_equities_daily AS SELECT * FROM read_parquet('{self.equities_parquet.as_posix()}')"
                )
            if self.indices_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_indices_daily AS SELECT * FROM read_parquet('{self.indices_parquet.as_posix()}')"
                )
            if self.provenance_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_provenance_records AS SELECT * FROM read_parquet('{self.provenance_parquet.as_posix()}')"
                )
            if self.regime_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_market_regimes AS SELECT * FROM read_parquet('{self.regime_parquet.as_posix()}')"
                )
            if self.quality_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_data_quality AS SELECT * FROM read_parquet('{self.quality_parquet.as_posix()}')"
                )
            if self.features_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_daily_features AS SELECT * FROM read_parquet('{self.features_parquet.as_posix()}')"
                )
            if params:
                return con.execute(sql, params).df()
            return con.execute(sql).df()
        finally:
            con.close()
