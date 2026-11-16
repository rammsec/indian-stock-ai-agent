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

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
        predictions_dir: Path | None = None,
        backtests_dir: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        ensure_storage_directories()
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = features_dir or FEATURES_DATA_DIR
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = predictions_dir or PREDICTIONS_DATA_DIR
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.backtests_dir = backtests_dir or BACKTESTS_DATA_DIR
        self.backtests_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or DUCKDB_PATH

        self.equities_parquet = self.processed_dir / "equities_daily.parquet"
        self.indices_parquet = self.processed_dir / "indices_daily.parquet"
        self.provenance_parquet = self.processed_dir / "provenance_records.parquet"
        self.regime_parquet = self.processed_dir / "market_regimes.parquet"
        self.quality_parquet = self.processed_dir / "data_quality.parquet"
        self.features_parquet = self.features_dir / "daily_features.parquet"
        self.feature_catalog_json = self.features_dir / "feature_catalog.json"
        self.predictions_parquet = self.predictions_dir / "prediction_log.parquet"
        self.latest_predictions_json = self.predictions_dir / "latest_predictions.json"
        self.backtest_predictions_parquet = self.backtests_dir / "walk_forward_predictions.parquet"
        self.backtest_summary_json = self.backtests_dir / "walk_forward_summary.json"
        self.corporate_intel_parquet = self.processed_dir / "corporate_intelligence.parquet"
        self.bulk_block_parquet = self.processed_dir / "nse_bulk_block_deals.parquet"
        self.events_parquet = self.predictions_dir / "structured_events.parquet"
        self.cross_stock_parquet = self.predictions_dir / "cross_stock_impacts.parquet"

    @staticmethod
    def _upsert_parquet(
        parquet_path: Path,
        new_rows: list[dict[str, Any]],
        dedup_cols: list[str],
        sort_cols: list[str],
        keep: Literal["first", "last"] = "last",
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

        df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep=keep)
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

    def upsert_predictions(self, records: list[Any]) -> int:
        """Append immutable prediction records into `/data/predictions/prediction_log.parquet`."""
        import json

        rows: list[dict[str, Any]] = []
        for rec in records:
            d = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
            d["top_bullish_drivers"] = json.dumps(d.get("top_bullish_drivers", []))
            d["top_bearish_drivers"] = json.dumps(d.get("top_bearish_drivers", []))
            d["feature_snapshot"] = json.dumps(d.get("feature_snapshot", {}))
            rows.append(d)
        return self._upsert_parquet(
            self.predictions_parquet,
            rows,
            dedup_cols=["prediction_id"],
            sort_cols=["observation_date", "symbol", "generated_at"],
            keep="first",
        )

    def evaluate_realized_outcomes(self, equities_df: pd.DataFrame) -> int:
        """Match historical predictions to realized 1D/5D outcomes strictly after target session closes."""
        if not self.predictions_parquet.exists() or equities_df.empty:
            return 0
        pred_df = pd.read_parquet(self.predictions_parquet)
        if pred_df.empty:
            return 0

        for col in ("realized_next_day_return_pct", "realized_5d_return_pct", "outcome_evaluated_at"):
            if col not in pred_df.columns:
                pred_df[col] = None

        eq = equities_df[["symbol", "observation_date", "close"]].dropna(subset=["close"]).copy()
        eq["symbol"] = eq["symbol"].astype(str).str.upper()
        eq["observation_date"] = eq["observation_date"].astype(str)
        eq = eq.sort_values(["symbol", "observation_date"]).reset_index(drop=True)

        sym_dates: dict[str, list[str]] = {}
        sym_closes: dict[str, list[float]] = {}
        for sym, grp in eq.groupby("symbol", sort=False):
            sym_dates[str(sym)] = grp["observation_date"].tolist()
            sym_closes[str(sym)] = grp["close"].astype(float).tolist()

        updated_count = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        for idx in pred_df.index:
            sym = str(pred_df.loc[idx, "symbol"]).upper()
            obs_dt = str(pred_df.loc[idx, "observation_date"])
            if sym not in sym_dates:
                continue
            dates_list = sym_dates[sym]
            closes_list = sym_closes[sym]

            # Find observation_date index (or latest trading date <= observation_date)
            base_idx = None
            for i, d_str in enumerate(dates_list):
                if d_str == obs_dt:
                    base_idx = i
                    break
            if base_idx is None:
                continue

            base_close = closes_list[base_idx]
            if base_close <= 0:
                continue

            changed = False
            if pd.isna(pred_df.loc[idx, "realized_next_day_return_pct"]) and base_idx + 1 < len(closes_list):
                r1 = (closes_list[base_idx + 1] / base_close - 1.0) * 100.0
                pred_df.loc[idx, "realized_next_day_return_pct"] = round(float(r1), 4)
                changed = True

            if pd.isna(pred_df.loc[idx, "realized_5d_return_pct"]) and base_idx + 5 < len(closes_list):
                r5 = (closes_list[base_idx + 5] / base_close - 1.0) * 100.0
                pred_df.loc[idx, "realized_5d_return_pct"] = round(float(r5), 4)
                changed = True

            if changed:
                pred_df.loc[idx, "outcome_evaluated_at"] = now_iso
                updated_count += 1

        if updated_count > 0:
            pred_df.to_parquet(self.predictions_parquet, index=False, compression="zstd")
        return updated_count

    def upsert_backtest_predictions(self, df: pd.DataFrame) -> int:
        """Persist walk-forward out-of-sample predictions into `/data/backtests/walk_forward_predictions.parquet`."""
        if df.empty:
            return 0
        out = df.copy()
        if "prob_bin" in out.columns:
            out = out.drop(columns=["prob_bin"])
        out = out.sort_values(["model_name", "symbol", "observation_date"]).reset_index(drop=True)
        out.to_parquet(self.backtest_predictions_parquet, index=False, compression="zstd")
        return len(out)

    def upsert_corporate_intelligence(self, summaries: list[Any]) -> int:
        """Upsert Phase 4 `CorporateIntelligenceSummary` records into `corporate_intelligence.parquet`."""
        import json

        rows: list[dict[str, Any]] = []
        for s in summaries:
            d = s.model_dump() if hasattr(s, "model_dump") else dict(s)
            d["bulk_block_deals"] = json.dumps(d.get("bulk_block_deals", []))
            d["recent_corporate_actions"] = json.dumps(d.get("recent_corporate_actions", []))
            d["key_signals"] = json.dumps(d.get("key_signals", []))
            rows.append(d)
        return self._upsert_parquet(
            self.corporate_intel_parquet,
            rows,
            dedup_cols=["symbol"],
            sort_cols=["symbol"],
        )

    def upsert_bulk_block_deals(self, deals: list[Any]) -> int:
        """Upsert official NSE Bulk & Block deal records into `nse_bulk_block_deals.parquet`."""
        rows = [d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in deals]
        return self._upsert_parquet(
            self.bulk_block_parquet,
            rows,
            dedup_cols=["deal_type", "deal_date", "symbol", "client_name", "side", "quantity"],
            sort_cols=["deal_date", "symbol", "deal_type"],
        )

    def upsert_structured_events(self, events: list[Any]) -> int:
        """Append immutable structured news events into `structured_events.parquet`."""
        import json

        rows: list[dict[str, Any]] = []
        for e in events:
            d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
            d["related_companies"] = json.dumps(d.get("related_companies", []))
            d["related_sectors"] = json.dumps(d.get("related_sectors", []))
            rows.append(d)
        return self._upsert_parquet(
            self.events_parquet,
            rows,
            dedup_cols=["event_id", "ticker"],
            sort_cols=["publication_timestamp", "ticker", "event_id"],
            keep="first",
        )

    def upsert_cross_stock_impacts(self, impacts: list[Any]) -> int:
        """Append cross-stock relationship impact records into `cross_stock_impacts.parquet`."""
        rows = [i.model_dump() if hasattr(i, "model_dump") else dict(i) for i in impacts]
        return self._upsert_parquet(
            self.cross_stock_parquet,
            rows,
            dedup_cols=["affected_company", "originating_event_id", "relationship_type"],
            sort_cols=["affected_company", "originating_event_id"],
            keep="first",
        )

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
            if self.predictions_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_predictions AS SELECT * FROM read_parquet('{self.predictions_parquet.as_posix()}')"
                )
            if self.backtest_predictions_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_backtest_predictions AS SELECT * FROM read_parquet('{self.backtest_predictions_parquet.as_posix()}')"
                )
            if self.corporate_intel_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_corporate_intel AS SELECT * FROM read_parquet('{self.corporate_intel_parquet.as_posix()}')"
                )
            if self.bulk_block_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_nse_bulk_block_deals AS SELECT * FROM read_parquet('{self.bulk_block_parquet.as_posix()}')"
                )
            if self.events_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_structured_events AS SELECT * FROM read_parquet('{self.events_parquet.as_posix()}')"
                )
            if self.cross_stock_parquet.exists():
                con.execute(
                    f"CREATE VIEW v_cross_stock_impacts AS SELECT * FROM read_parquet('{self.cross_stock_parquet.as_posix()}')"
                )
            if params:
                return con.execute(sql, params).df()
            return con.execute(sql).df()
        finally:
            con.close()
