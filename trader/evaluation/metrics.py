"""CYCLE-008 Phase 3 — Deterministic Performance & Risk Metrics.

Computes auditable, cost-adjusted performance, risk, drawdown, MAE/MFE,
per-stock, and per-regime metrics for historical walk-forward evaluations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TradeRecord:
    """Immutable record of a single closed intraday paper trade."""

    symbol: str
    session_date: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl_inr: float
    brokerage_inr: float
    statutory_charges_inr: float
    slippage_inr: float
    total_friction_inr: float
    net_pnl_inr: float
    return_pct: float
    mae_pct: float
    mfe_pct: float
    regime_label: str
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_equity_series(
    equity_curve: Sequence[dict[str, Any] | float],
    initial_capital_inr: float,
) -> list[float]:
    values: list[float] = []
    for pt in equity_curve:
        if isinstance(pt, dict):
            values.append(float(pt["equity_inr"]))
        else:
            values.append(float(pt))
    if not values:
        return [float(initial_capital_inr)]
    return values


def _compute_drawdown(
    equity_values: list[float],
    initial_capital_inr: float,
) -> tuple[float, float]:
    """Return (max_drawdown_pct, max_drawdown_inr) as non-negative magnitudes."""
    peak = float(initial_capital_inr)
    max_dd_inr = 0.0
    max_dd_pct = 0.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd_inr = max(0.0, peak - eq)
        dd_pct = (dd_inr / peak) * 100.0 if peak > 0 else 0.0
        if dd_inr > max_dd_inr:
            max_dd_inr = dd_inr
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return round(max_dd_pct, 4), round(max_dd_inr, 2)


def _compute_daily_returns(
    equity_values: list[float],
    initial_capital_inr: float,
) -> list[float]:
    """Compute sequential period returns starting from initial_capital_inr."""
    if not equity_values:
        return []
    returns: list[float] = []
    prev = float(initial_capital_inr)
    for eq in equity_values:
        if prev > 0:
            returns.append((eq - prev) / prev)
        prev = eq
    return returns


def _compute_sharpe_and_sortino(
    daily_returns: list[float],
) -> tuple[float | None, float | None]:
    """Annualized Sharpe & Sortino (sqrt(252)); returns None if insufficient data or zero vol."""
    n = len(daily_returns)
    if n < 2:
        return None, None

    mean_r = sum(daily_returns) / n
    var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.0
    sharpe = round((mean_r / std_r) * math.sqrt(TRADING_DAYS_PER_YEAR), 4) if std_r > 0 else None

    downside_sq = [min(0.0, r) ** 2 for r in daily_returns]
    downside_dev = math.sqrt(sum(downside_sq) / n)
    sortino = (
        round((mean_r / downside_dev) * math.sqrt(TRADING_DAYS_PER_YEAR), 4)
        if downside_dev > 0
        else None
    )
    return sharpe, sortino


def _compute_directional_accuracy(
    directional_outcomes: Sequence[dict[str, Any]],
) -> tuple[float | None, int]:
    """Fraction of directional (BULLISH/BEARISH) signals matching realized session direction."""
    directional = [
        o
        for o in directional_outcomes
        if str(o.get("predicted_direction", "")).upper() in ("BULLISH", "BEARISH")
    ]
    if not directional:
        return None, 0
    hits = sum(
        1
        for o in directional
        if str(o.get("predicted_direction", "")).upper()
        == str(o.get("realized_direction", "")).upper()
    )
    return round(hits / len(directional), 4), len(directional)


def _compute_subset_metrics(
    trades: Sequence[TradeRecord],
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute core metrics for a per-stock or per-regime slice."""
    total_trades = len(trades)
    wins = [t for t in trades if t.net_pnl_inr > 0]
    losses = [t for t in trades if t.net_pnl_inr < 0]

    gross_wins = sum(t.net_pnl_inr for t in wins)
    gross_losses = sum(t.net_pnl_inr for t in losses)
    profit_factor = round(gross_wins / abs(gross_losses), 4) if gross_losses < 0 else None

    dir_acc, dir_signals = _compute_directional_accuracy(outcomes)
    eval_count = len(outcomes) if outcomes else total_trades
    trade_cov = round((total_trades / eval_count) * 100.0, 4) if eval_count > 0 else 0.0

    return {
        "evaluated_symbol_sessions": eval_count,
        "directional_signals": dir_signals,
        "directional_accuracy": dir_acc,
        "trade_coverage_pct": trade_cov,
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / total_trades, 4) if total_trades > 0 else None,
        "avg_win_inr": round(sum(t.net_pnl_inr for t in wins) / len(wins), 2) if wins else None,
        "avg_win_pct": round(sum(t.return_pct for t in wins) / len(wins), 4) if wins else None,
        "avg_loss_inr": (
            round(sum(t.net_pnl_inr for t in losses) / len(losses), 2) if losses else None
        ),
        "avg_loss_pct": (
            round(sum(t.return_pct for t in losses) / len(losses), 4) if losses else None
        ),
        "net_expectancy_inr": (
            round(sum(t.net_pnl_inr for t in trades) / total_trades, 2)
            if total_trades > 0
            else None
        ),
        "net_expectancy_pct": (
            round(sum(t.return_pct for t in trades) / total_trades, 4)
            if total_trades > 0
            else None
        ),
        "profit_factor": profit_factor,
        "avg_mae_pct": (
            round(sum(t.mae_pct for t in trades) / total_trades, 4) if total_trades > 0 else None
        ),
        "avg_mfe_pct": (
            round(sum(t.mfe_pct for t in trades) / total_trades, 4) if total_trades > 0 else None
        ),
        "gross_pnl_inr": round(sum(t.gross_pnl_inr for t in trades), 2),
        "total_friction_inr": round(sum(t.total_friction_inr for t in trades), 2),
        "net_pnl_inr": round(sum(t.net_pnl_inr for t in trades), 2),
    }


def compute_strategy_metrics(
    trades: Sequence[TradeRecord],
    equity_curve: Sequence[dict[str, Any] | float],
    directional_outcomes: Sequence[dict[str, Any]],
    initial_capital_inr: float,
) -> dict[str, Any]:
    """Compute complete Phase 3 deterministic performance & risk metrics."""
    if initial_capital_inr <= 0:
        raise ValueError("initial_capital_inr must be > 0")

    total_trades = len(trades)
    wins = [t for t in trades if t.net_pnl_inr > 0]
    losses = [t for t in trades if t.net_pnl_inr < 0]
    breakevens = [t for t in trades if t.net_pnl_inr == 0]

    directional_accuracy, directional_signals = _compute_directional_accuracy(directional_outcomes)
    evaluated_sessions = len(directional_outcomes) if directional_outcomes else total_trades
    signal_frequency = (
        round(directional_signals / evaluated_sessions, 4) if evaluated_sessions > 0 else 0.0
    )
    trade_coverage_pct = (
        round((total_trades / evaluated_sessions) * 100.0, 4) if evaluated_sessions > 0 else 0.0
    )

    gross_wins = sum(t.net_pnl_inr for t in wins)
    gross_losses = sum(t.net_pnl_inr for t in losses)
    profit_factor = round(gross_wins / abs(gross_losses), 4) if gross_losses < 0 else None

    equity_values = _extract_equity_series(equity_curve, initial_capital_inr)
    max_dd_pct, max_dd_inr = _compute_drawdown(equity_values, initial_capital_inr)
    daily_returns = _compute_daily_returns(equity_values, initial_capital_inr)
    sharpe_ratio, sortino_ratio = _compute_sharpe_and_sortino(daily_returns)

    brokerage_inr = round(sum(t.brokerage_inr for t in trades), 4)
    statutory_charges_inr = round(sum(t.statutory_charges_inr for t in trades), 4)
    slippage_inr = round(sum(t.slippage_inr for t in trades), 4)
    total_friction_inr = round(sum(t.total_friction_inr for t in trades), 4)
    gross_pnl_inr = round(sum(t.gross_pnl_inr for t in trades), 2)
    net_pnl_inr = round(sum(t.net_pnl_inr for t in trades), 2)
    ending_equity_inr = round(equity_values[-1], 2) if equity_values else round(initial_capital_inr, 2)
    net_return_pct = round(((ending_equity_inr - initial_capital_inr) / initial_capital_inr) * 100.0, 4)

    # Per-stock breakdown
    all_symbols = sorted(
        {t.symbol for t in trades}
        | {str(o.get("symbol", "")).strip().upper() for o in directional_outcomes if o.get("symbol")}
    )
    by_stock: dict[str, Any] = {}
    for sym in all_symbols:
        sym_trades = [t for t in trades if t.symbol == sym]
        sym_outcomes = [
            o for o in directional_outcomes if str(o.get("symbol", "")).strip().upper() == sym
        ]
        by_stock[sym] = _compute_subset_metrics(sym_trades, sym_outcomes)

    # Per-regime breakdown
    all_regimes = sorted(
        {t.regime_label for t in trades if t.regime_label}
        | {
            str(o.get("regime_label", "")).strip()
            for o in directional_outcomes
            if o.get("regime_label")
        }
    )
    by_regime: dict[str, Any] = {}
    for reg in all_regimes:
        reg_trades = [t for t in trades if t.regime_label == reg]
        reg_outcomes = [
            o for o in directional_outcomes if str(o.get("regime_label", "")).strip() == reg
        ]
        by_regime[reg] = _compute_subset_metrics(reg_trades, reg_outcomes)

    return {
        "evaluated_symbol_sessions": evaluated_sessions,
        "directional_signals": directional_signals,
        "directional_accuracy": directional_accuracy,
        "signal_frequency": signal_frequency,
        "trade_coverage_pct": trade_coverage_pct,
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "breakeven_trades": len(breakevens),
        "win_rate": round(len(wins) / total_trades, 4) if total_trades > 0 else None,
        "avg_win_inr": round(sum(t.net_pnl_inr for t in wins) / len(wins), 2) if wins else None,
        "avg_win_pct": round(sum(t.return_pct for t in wins) / len(wins), 4) if wins else None,
        "avg_loss_inr": (
            round(sum(t.net_pnl_inr for t in losses) / len(losses), 2) if losses else None
        ),
        "avg_loss_pct": (
            round(sum(t.return_pct for t in losses) / len(losses), 4) if losses else None
        ),
        "net_expectancy_inr": (
            round(sum(t.net_pnl_inr for t in trades) / total_trades, 2)
            if total_trades > 0
            else None
        ),
        "net_expectancy_pct": (
            round(sum(t.return_pct for t in trades) / total_trades, 4)
            if total_trades > 0
            else None
        ),
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_inr": max_dd_inr,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_mae_pct": (
            round(sum(t.mae_pct for t in trades) / total_trades, 4) if total_trades > 0 else None
        ),
        "avg_mfe_pct": (
            round(sum(t.mfe_pct for t in trades) / total_trades, 4) if total_trades > 0 else None
        ),
        "initial_capital_inr": round(float(initial_capital_inr), 2),
        "ending_equity_inr": ending_equity_inr,
        "cost_breakdown": {
            "brokerage_inr": brokerage_inr,
            "statutory_charges_inr": statutory_charges_inr,
            "slippage_inr": slippage_inr,
            "total_friction_inr": total_friction_inr,
            "gross_pnl_inr": gross_pnl_inr,
            "net_pnl_inr": net_pnl_inr,
            "net_return_pct": net_return_pct,
        },
        "by_stock": by_stock,
        "by_regime": by_regime,
    }
