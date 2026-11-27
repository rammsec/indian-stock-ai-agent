"""CYCLE-011: Strictly Walk-Forward (Zero-Lookahead) 1-Minute NSE Calibration & Evaluation.

Runs the full Phase 1-3 continuous monitoring engine (`IntradayCandleMonitoringSession` +
`PostSessionOutcomeEvaluator`) across genuine historical 1-minute NSE trading sessions,
splitting strictly chronologically into Train vs Out-of-Sample Test dates:
  max(train_dates) < min(test_dates)

Computes honest, net-of-NSE-tax metrics (win rate, profit factor, net P&L in INR,
friction paid, empirical pattern expectancy) with ZERO lookahead and ZERO synthetic fallback.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Sequence

import pandas as pd

from trader.authority import PermissionType
from trader.config import DEFAULT_INITIAL_CAPITAL_INR, IST_TZ
from trader.data.loader_1m import OneMinuteSessionBundle
from trader.data.market_feed import BaseMarketFeed, TimeframeLiteral
from trader.intelligence.data_contract import OHLCVEvidenceFrame, ProvenanceTier, build_ohlcv_evidence_frame
from trader.timing.central_directional_agent import DirectionalBias
from trader.timing.monitoring_session import IntradayCandleMonitoringSession, MonitoringSessionResult


class InMemory1MReplayFeed(BaseMarketFeed):
    """Strictly causal in-memory 1m market feed backed by a OneMinuteSessionBundle."""

    def __init__(self, bundle: OneMinuteSessionBundle) -> None:
        self._bundle = bundle
        self._provider = "+".join(bundle.source_providers) or "verified_1m_archive"
        rows = []
        for bar in list(bundle.warmup_bars) + list(bundle.session_bars):
            ts_open = bar.timestamp.astimezone(IST_TZ)
            ts_close = ts_open + pd.Timedelta(minutes=1)
            rows.append(
                {
                    "ts_open_ist": ts_open,
                    "ts_close_ist": ts_close,
                    "ts_init_ist": ts_close,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
        self._df = pd.DataFrame(rows)

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def is_authoritative(self) -> bool:
        return False

    @property
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        return ("1m",)

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        if timeframe != "1m":
            raise ValueError(f"InMemory1MReplayFeed supports only '1m', got {timeframe!r}")
        as_of = as_of_ist.astimezone(IST_TZ)
        causal = self._df.loc[self._df["ts_close_ist"] <= as_of].tail(lookback_bars).reset_index(drop=True)
        if causal.empty:
            return build_ohlcv_evidence_frame(
                symbol=symbol.strip().upper(),
                timeframe="1m",
                df=None,
                as_of_ist=as_of,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=self._provider,
                exchange="NSE",
                missing_reason="NO_CAUSAL_BARS_BEFORE_AS_OF",
            )
        return build_ohlcv_evidence_frame(
            symbol=symbol.strip().upper(),
            timeframe="1m",
            df=causal,
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name=self._provider,
            exchange="NSE",
        )


@dataclass(frozen=True)
class PatternCalibrationEntry:
    """Empirical performance bucket learned strictly on the chronological Train split."""

    key: str
    sample_count: int
    wins: int
    losses: int
    win_rate: float
    avg_net_pnl_inr: float
    total_net_pnl_inr: float
    calibration_status: str  # "EMPIRICAL_WALKFORWARD_CALIBRATED" or "UNCALIBRATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sample_count": self.sample_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_net_pnl_inr": round(self.avg_net_pnl_inr, 2),
            "total_net_pnl_inr": round(self.total_net_pnl_inr, 2),
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True)
class SplitMetrics:
    """Aggregated net-of-cost metrics across a chronological split (Train or OOS Test)."""

    split_name: str
    sessions_count: int
    date_start: str | None
    date_end: str | None
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_pnl_inr: float
    total_friction_inr: float
    net_pnl_inr: float
    profit_factor: float
    max_session_loss_inr: float
    saved_losses_count: int
    missed_moves_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "sessions_count": self.sessions_count,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "gross_pnl_inr": round(self.gross_pnl_inr, 2),
            "total_friction_inr": round(self.total_friction_inr, 2),
            "net_pnl_inr": round(self.net_pnl_inr, 2),
            "profit_factor": round(self.profit_factor, 4),
            "max_session_loss_inr": round(self.max_session_loss_inr, 2),
            "saved_losses_count": self.saved_losses_count,
            "missed_moves_count": self.missed_moves_count,
        }


@dataclass(frozen=True)
class WalkForward1MReport:
    """Complete chronological walk-forward report for 1-minute NSE sessions."""

    generated_at_ist: str
    symbols: tuple[str, ...]
    direction: str
    train_ratio: float
    min_calibration_samples: int
    train_metrics: SplitMetrics
    oos_test_metrics: SplitMetrics
    calibration_table: tuple[PatternCalibrationEntry, ...]
    per_symbol_oos_net_pnl_inr: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_ist": self.generated_at_ist,
            "symbols": list(self.symbols),
            "direction": self.direction,
            "train_ratio": self.train_ratio,
            "min_calibration_samples": self.min_calibration_samples,
            "train_metrics": self.train_metrics.to_dict(),
            "oos_test_metrics": self.oos_test_metrics.to_dict(),
            "calibration_table": [c.to_dict() for c in self.calibration_table],
            "per_symbol_oos_net_pnl_inr": {k: round(v, 2) for k, v in sorted(self.per_symbol_oos_net_pnl_inr.items())},
        }


def split_bundles_chronologically(
    bundles: Sequence[OneMinuteSessionBundle],
    train_ratio: float = 0.70,
) -> tuple[list[OneMinuteSessionBundle], list[OneMinuteSessionBundle]]:
    """Split 1m session bundles strictly by calendar date so max(train) < min(test)."""
    if not (0.1 <= train_ratio <= 0.9):
        raise ValueError("train_ratio must be between 0.1 and 0.9")
    unique_dates = sorted({b.trading_date for b in bundles})
    if len(unique_dates) < 2:
        raise ValueError("Need at least 2 distinct trading dates for chronological train/OOS split")
    split_idx = max(1, min(len(unique_dates) - 1, int(round(len(unique_dates) * train_ratio))))
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])
    if not (max(train_dates) < min(test_dates)):
        raise ValueError("Chronological walk-forward boundary violated: max(train_dates) must be < min(test_dates)")

    train_bundles = [b for b in bundles if b.trading_date in train_dates]
    test_bundles = [b for b in bundles if b.trading_date in test_dates]
    return train_bundles, test_bundles


def run_single_1m_session(
    bundle: OneMinuteSessionBundle,
    *,
    direction: DirectionalBias = DirectionalBias.BULLISH,
    human_permission: PermissionType = PermissionType.ALLOW_BOTH,
    initial_capital_inr: float = DEFAULT_INITIAL_CAPITAL_INR,
) -> MonitoringSessionResult:
    """Execute one genuine 1m NSE session through IntradayCandleMonitoringSession."""
    feed = InMemory1MReplayFeed(bundle)
    session = IntradayCandleMonitoringSession(
        feed=feed,
        initial_capital=initial_capital_inr,
    )
    return session.run_session(
        symbol=bundle.symbol,
        session_date=bundle.trading_date,
        timeframe="1m",
        direction=direction,
        human_permission=human_permission,
        warmup_lookback_bars=50,
    )


def _aggregate_split(split_name: str, reports: Sequence[MonitoringSessionResult]) -> SplitMetrics:
    dates = sorted({r.session_date for r in reports})
    all_trades: list[dict[str, Any]] = []
    total_friction = 0.0
    max_session_loss = 0.0
    saved_losses = 0
    missed_moves = 0

    for rep in reports:
        all_trades.extend(rep.closed_trades)
        total_friction += float(rep.total_friction_inr)
        if rep.realized_pnl_inr < max_session_loss:
            max_session_loss = float(rep.realized_pnl_inr)
        saved_losses += int(rep.outcome_report.saved_loss_count)
        missed_moves += int(rep.outcome_report.missed_opportunity_count)

    wins = sum(1 for t in all_trades if float(t.get("net_pnl_inr", 0.0)) > 0.0)
    losses = sum(1 for t in all_trades if float(t.get("net_pnl_inr", 0.0)) <= 0.0)
    total_trades = len(all_trades)
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0
    net_pnl = sum(float(t.get("net_pnl_inr", 0.0)) for t in all_trades)
    gross_pnl = net_pnl + total_friction
    gross_wins = sum(float(t.get("net_pnl_inr", 0.0)) for t in all_trades if float(t.get("net_pnl_inr", 0.0)) > 0.0)
    gross_losses = abs(sum(float(t.get("net_pnl_inr", 0.0)) for t in all_trades if float(t.get("net_pnl_inr", 0.0)) < 0.0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.99 if gross_wins > 0 else 0.0)

    return SplitMetrics(
        split_name=split_name,
        sessions_count=len(reports),
        date_start=dates[0] if dates else None,
        date_end=dates[-1] if dates else None,
        total_trades=total_trades,
        winning_trades=wins,
        losing_trades=losses,
        win_rate=win_rate,
        gross_pnl_inr=gross_pnl,
        total_friction_inr=total_friction,
        net_pnl_inr=net_pnl,
        profit_factor=profit_factor,
        max_session_loss_inr=max_session_loss,
        saved_losses_count=saved_losses,
        missed_moves_count=missed_moves,
    )


def build_empirical_calibration_table(
    train_reports: Sequence[MonitoringSessionResult],
    *,
    min_samples: int = 5,
) -> tuple[PatternCalibrationEntry, ...]:
    """Build empirical win-rate & expectancy table strictly from Train-split closed trades."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for rep in train_reports:
        for t in rep.closed_trades:
            side = str(t.get("side", "UNKNOWN"))
            reason = str(t.get("exit_reason", "UNKNOWN"))
            key = f"{rep.symbol}:{side}:{reason}"
            buckets[key].append(float(t.get("net_pnl_inr", 0.0)))
            buckets[f"ALL:{side}"].append(float(t.get("net_pnl_inr", 0.0)))

    entries: list[PatternCalibrationEntry] = []
    for key, pnls in sorted(buckets.items()):
        n = len(pnls)
        w = sum(1 for p in pnls if p > 0.0)
        l = n - w
        tot = sum(pnls)
        entries.append(
            PatternCalibrationEntry(
                key=key,
                sample_count=n,
                wins=w,
                losses=l,
                win_rate=(w / n) if n > 0 else 0.0,
                avg_net_pnl_inr=(tot / n) if n > 0 else 0.0,
                total_net_pnl_inr=tot,
                calibration_status=(
                    "EMPIRICAL_WALKFORWARD_CALIBRATED" if n >= min_samples else "UNCALIBRATED"
                ),
            )
        )
    return tuple(entries)


def run_walkforward_1m_calibration(
    bundles: Sequence[OneMinuteSessionBundle],
    *,
    direction: DirectionalBias = DirectionalBias.BULLISH,
    human_permission: PermissionType = PermissionType.ALLOW_BOTH,
    train_ratio: float = 0.70,
    min_calibration_samples: int = 5,
    initial_capital_inr: float = DEFAULT_INITIAL_CAPITAL_INR,
) -> WalkForward1MReport:
    """Execute full chronological Train -> Out-of-Sample Test evaluation on 1m NSE bundles."""
    train_bundles, test_bundles = split_bundles_chronologically(bundles, train_ratio=train_ratio)
    train_reports = [
        run_single_1m_session(
            b,
            direction=direction,
            human_permission=human_permission,
            initial_capital_inr=initial_capital_inr,
        )
        for b in train_bundles
    ]
    test_reports = [
        run_single_1m_session(
            b,
            direction=direction,
            human_permission=human_permission,
            initial_capital_inr=initial_capital_inr,
        )
        for b in test_bundles
    ]

    per_sym_oos: dict[str, float] = defaultdict(float)
    for rep in test_reports:
        per_sym_oos[rep.symbol] += float(rep.realized_pnl_inr)

    return WalkForward1MReport(
        generated_at_ist=datetime.now(IST_TZ).isoformat(),
        symbols=tuple(sorted({b.symbol for b in bundles})),
        direction=direction.value,
        train_ratio=train_ratio,
        min_calibration_samples=min_calibration_samples,
        train_metrics=_aggregate_split("TRAIN", train_reports),
        oos_test_metrics=_aggregate_split("OOS_TEST", test_reports),
        calibration_table=build_empirical_calibration_table(train_reports, min_samples=min_calibration_samples),
        per_symbol_oos_net_pnl_inr=dict(per_sym_oos),
    )
