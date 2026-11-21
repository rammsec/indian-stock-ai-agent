"""Persistent SQLite Historical Decision Database for the AI Trading Lab."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/lab/historical_lab.db")


class HistoricalLabDatabase:
    """Stores BacktestRuns, AgentDecisions, SimulatedOrders, Trades, PortfolioSnapshots, and StrategyExperiments."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    selected_date TEXT NOT NULL,
                    starting_capital REAL NOT NULL,
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    model_config_json TEXT NOT NULL,
                    data_provenance_json TEXT NOT NULL,
                    execution_config_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    final_portfolio_value REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    sortino_ratio REAL NOT NULL,
                    win_rate_pct REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    total_fees REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_decisions (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision_timestamp TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning_summary TEXT NOT NULL,
                    indicator_snapshot_json TEXT NOT NULL,
                    council_decision TEXT NOT NULL,
                    risk_approval TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS simulated_orders (
                    order_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    expected_price REAL NOT NULL,
                    execution_price REAL,
                    fill_quantity INTEGER NOT NULL,
                    slippage REAL NOT NULL,
                    fees REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_order_id TEXT NOT NULL,
                    exit_order_id TEXT NOT NULL,
                    entry_timestamp TEXT NOT NULL,
                    exit_timestamp TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    gross_pnl REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    fees REAL NOT NULL,
                    holding_minutes REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    cash REAL NOT NULL,
                    invested_capital REAL NOT NULL,
                    unrealised_pnl REAL NOT NULL,
                    realised_pnl REAL NOT NULL,
                    total_equity REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS strategy_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    parent_strategy_version TEXT NOT NULL,
                    modified_parameters_json TEXT NOT NULL,
                    validation_period TEXT NOT NULL,
                    performance_metrics_json TEXT NOT NULL,
                    suggestion_summary TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_response_cache (
                    input_hash TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    request_timestamp TEXT NOT NULL,
                    response_timestamp TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    actual_cost_usd REAL NOT NULL,
                    response_text TEXT NOT NULL,
                    input_snapshot_json TEXT NOT NULL
                );
                """
            )

    def save_run_bundle(self, run_data: dict[str, Any]) -> None:
        """Persist a complete BacktestRun along with its decisions, orders, trades, and snapshots."""
        run = run_data["run"]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO backtest_runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run["run_id"],
                    run["selected_date"],
                    run["starting_capital"],
                    run["strategy_id"],
                    run["strategy_version"],
                    run["mode"],
                    json.dumps(run["model_config"]),
                    json.dumps(run["data_provenance"]),
                    json.dumps(run["execution_config"]),
                    run["started_at"],
                    run["completed_at"],
                    run["final_portfolio_value"],
                    run["net_pnl"],
                    run["total_return_pct"],
                    run["max_drawdown_pct"],
                    run["sharpe_ratio"],
                    run["sortino_ratio"],
                    run["win_rate_pct"],
                    run["profit_factor"],
                    run["total_trades"],
                    run["total_fees"],
                ),
            )

            for dec in run_data.get("decisions", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_decisions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        dec["decision_id"],
                        run["run_id"],
                        dec["symbol"],
                        dec["decision_timestamp"],
                        dec["agent_name"],
                        dec["model_name"],
                        dec["model_version"],
                        dec["execution_mode"],
                        dec["recommendation"],
                        dec["confidence"],
                        dec["reasoning_summary"],
                        json.dumps(dec["indicator_snapshot"]),
                        dec["council_decision"],
                        dec["risk_approval"],
                    ),
                )

            for ord_item in run_data.get("orders", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO simulated_orders VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        ord_item["order_id"],
                        run["run_id"],
                        ord_item["symbol"],
                        ord_item["side"],
                        ord_item["quantity"],
                        ord_item["order_type"],
                        ord_item["expected_price"],
                        ord_item["execution_price"],
                        ord_item["fill_quantity"],
                        ord_item["slippage"],
                        ord_item["fees"],
                        ord_item["status"],
                        ord_item["timestamp"],
                    ),
                )

            for tr in run_data.get("trades", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trades VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        tr["trade_id"],
                        run["run_id"],
                        tr["symbol"],
                        tr["entry_order_id"],
                        tr["exit_order_id"],
                        tr["entry_timestamp"],
                        tr["exit_timestamp"],
                        tr["quantity"],
                        tr["entry_price"],
                        tr["exit_price"],
                        tr["gross_pnl"],
                        tr["net_pnl"],
                        tr["fees"],
                        tr["holding_minutes"],
                        tr["exit_reason"],
                    ),
                )

            conn.execute("DELETE FROM portfolio_snapshots WHERE run_id = ?", (run["run_id"],))
            for snap in run_data.get("snapshots", []):
                conn.execute(
                    """
                    INSERT INTO portfolio_snapshots (
                        run_id, timestamp, cash, invested_capital, unrealised_pnl, realised_pnl, total_equity, drawdown_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run["run_id"],
                        snap["timestamp"],
                        snap["cash"],
                        snap["invested_capital"],
                        snap["unrealised_pnl"],
                        snap["realised_pnl"],
                        snap["total_equity"],
                        snap["drawdown_pct"],
                    ),
                )

    def save_experiment(self, exp: dict[str, Any]) -> None:
        """Persist a Stage C improvement experiment without mutating production strategies."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp["experiment_id"],
                    exp["parent_strategy_version"],
                    json.dumps(exp["modified_parameters"]),
                    exp["validation_period"],
                    json.dumps(exp["performance_metrics"]),
                    exp["suggestion_summary"],
                    exp["approval_status"],
                    exp["created_at"],
                ),
            )

    def list_runs(self, selected_date: str | None = None) -> list[dict[str, Any]]:
        """Query stored backtest runs, optionally filtered by date."""
        with self._connect() as conn:
            if selected_date:
                rows = conn.execute(
                    "SELECT * FROM backtest_runs WHERE selected_date = ? ORDER BY total_return_pct DESC",
                    (selected_date,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM backtest_runs ORDER BY selected_date DESC, total_return_pct DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def list_experiments(self) -> list[dict[str, Any]]:
        """Return all recorded strategy improvement experiments."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM strategy_experiments ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_cached_ai_response(self, input_hash: str) -> dict[str, Any] | None:
        """Retrieve cached AI provider response by reproducible input hash."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_response_cache WHERE input_hash = ?", (input_hash,)
            ).fetchone()
            return dict(row) if row else None

    def save_cached_ai_response(self, entry: dict[str, Any]) -> None:
        """Persist genuine AI provider response keyed by reproducible input hash."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ai_response_cache VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    entry["input_hash"],
                    entry["provider"],
                    entry["model_id"],
                    entry["model_version"],
                    entry["execution_mode"],
                    entry["request_timestamp"],
                    entry["response_timestamp"],
                    entry["input_tokens"],
                    entry["output_tokens"],
                    entry["actual_cost_usd"],
                    entry["response_text"],
                    json.dumps(entry["input_snapshot"]),
                ),
            )

