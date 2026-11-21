"""Chronological validation, Purged K-Fold with embargo, and baseline comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..data.schema import Candle5M
from .base import BaseStrategy
from .backtester import BacktestResult, IntradayBacktester


@dataclass(frozen=True)
class ChronologicalSplit:
    """Strict chronological split keeping unseen out-of-sample data completely isolated."""

    train_candles: list[Candle5M]
    validation_candles: list[Candle5M]
    unseen_test_candles: list[Candle5M]


@dataclass(frozen=True)
class BaselineMetrics:
    """Buy-and-Hold baseline performance over the same candle series."""

    symbol: str
    total_candles: int
    initial_capital_inr: float
    ending_capital_inr: float
    total_return_pct: float
    max_drawdown_pct: float


@dataclass(frozen=True)
class StrategyComparisonResult:
    """Side-by-side comparison of an intraday strategy against Buy & Hold baseline."""

    strategy_result: BacktestResult
    baseline: BaselineMetrics
    alpha_vs_baseline_pct: float
    outperforms_baseline: bool


def create_chronological_split(
    candles: list[Candle5M],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> ChronologicalSplit:
    """Split historical 5m candles strictly chronologically into Train, Validation, and Unseen Test."""
    if not candles:
        raise ValueError("Cannot split empty candle series")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave room for unseen test set (< 1.0)")

    n = len(candles)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return ChronologicalSplit(
        train_candles=candles[:train_end],
        validation_candles=candles[train_end:val_end],
        unseen_test_candles=candles[val_end:],
    )


class PurgedKFold:
    """Time-series Purged K-Fold cross-validator with embargo buffer (Lopez de Prado)."""

    def __init__(self, n_splits: int = 5, purge_bars: int = 6, embargo_bars: int = 6) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, n_samples: int) -> Iterator[tuple[list[int], list[int]]]:
        """Yield (train_indices, test_indices) with purge before test and embargo after test."""
        if n_samples < self.n_splits * 2:
            raise ValueError(f"Insufficient samples ({n_samples}) for {self.n_splits} splits")

        fold_size = n_samples // self.n_splits
        all_indices = list(range(n_samples))

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n_samples
            test_idx = all_indices[test_start:test_end]

            # Purge train samples immediately preceding test_start
            purge_start = max(0, test_start - self.purge_bars)
            # Embargo train samples immediately following test_end
            embargo_end = min(n_samples, test_end + self.embargo_bars)

            train_idx = [
                i for i in all_indices
                if i < purge_start or i >= embargo_end
            ]
            yield train_idx, test_idx


def compute_buy_and_hold_baseline(
    candles: list[Candle5M],
    initial_capital: float = 1_000_000.0,
) -> BaselineMetrics:
    """Compute realistic Buy & Hold benchmark over the exact same candle window."""
    if not candles:
        raise ValueError("Cannot compute baseline on empty candle list")

    entry_price = candles[0].open
    shares = int(initial_capital / entry_price) if entry_price > 0 else 0
    leftover_cash = initial_capital - (shares * entry_price)

    peak = initial_capital
    max_dd_pct = 0.0
    for c in candles:
        eq = leftover_cash + (shares * c.close)
        peak = max(peak, eq)
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd)

    ending_equity = leftover_cash + (shares * candles[-1].close)
    ret_pct = ((ending_equity - initial_capital) / initial_capital) * 100.0

    return BaselineMetrics(
        symbol=candles[0].symbol,
        total_candles=len(candles),
        initial_capital_inr=round(initial_capital, 2),
        ending_capital_inr=round(ending_equity, 2),
        total_return_pct=round(ret_pct, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
    )


def compare_strategy_vs_baseline(
    strategy: BaseStrategy,
    candles: list[Candle5M],
    initial_capital: float = 1_000_000.0,
    execution_mode: str = "NEXT_BAR_OPEN",
) -> StrategyComparisonResult:
    """Run strategy backtest and compare against Buy & Hold baseline."""
    bt = IntradayBacktester(
        initial_capital=initial_capital,
        execution_mode=execution_mode,  # type: ignore[arg-type]
    )
    strat_res = bt.run_backtest(strategy=strategy, candles=candles)
    baseline = compute_buy_and_hold_baseline(candles=candles, initial_capital=initial_capital)
    alpha = round(strat_res.total_return_pct - baseline.total_return_pct, 2)

    return StrategyComparisonResult(
        strategy_result=strat_res,
        baseline=baseline,
        alpha_vs_baseline_pct=alpha,
        outperforms_baseline=alpha > 0.0,
    )
