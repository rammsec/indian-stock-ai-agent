"""Paper-Only Daily Signal Session wired to pluggable BaseMarketFeed.

Connects Research Agent (CentralDirectionalAgent), Candle Bar Agent
(CandleTimingAgent), and DeterministicRiskEngine to any BaseMarketFeed
(committed 1h archive, local CSV/Parquet, yfinance, or read-only Kite feed).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from ..config import DEFAULT_INITIAL_CAPITAL_INR, IST_TZ
from ..data.market_feed import BaseMarketFeed, TimeframeLiteral
from ..intelligence.data_contract import OHLCVEvidenceFrame, ProvenanceTier
from ..paper.cycle006c_snapshot import reconstruct_as_of_snapshot
from ..portfolio.ledger import IntradayPaperPortfolio
from ..risk.engine import DeterministicRiskEngine, RiskEvaluationResult
from .candle_timing_agent import CandleTimingAgent, TimingAssessment, TimingDecision
from .central_directional_agent import CentralDirectionalAgent, DirectionalHypothesis


@dataclass(frozen=True)
class PaperSignalCall:
    """Single symbol paper-only daily signal call."""

    symbol: str
    as_of_ist: str
    hypothesis: DirectionalHypothesis
    timing: TimingAssessment
    risk_evaluation: RiskEvaluationResult | None
    primary_provenance: str
    bars_5m_count: int
    bars_15m_count: int
    bars_1h_count: int


@dataclass(frozen=True)
class DailyPaperCallReport:
    """Aggregate paper-only daily call report across a symbol universe."""

    as_of_ist: str
    provider_name: str
    primary_timeframe: str
    paper_only: bool
    risk_state: str
    calls: tuple[PaperSignalCall, ...]


def _evidence_to_agent_df(ev: OHLCVEvidenceFrame) -> pd.DataFrame | None:
    if ev.provenance_status == ProvenanceTier.UNAVAILABLE or ev.frame.empty:
        return None
    df = ev.frame.copy()
    df["timestamp"] = df["ts_open_ist"]
    return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


class PaperFeedSignalSession:
    """Runs Research Agent + Candle Bar Agent + RiskEngine over a BaseMarketFeed."""

    def __init__(
        self,
        feed: BaseMarketFeed,
        *,
        directional_agent: CentralDirectionalAgent | None = None,
        timing_agent: CandleTimingAgent | None = None,
        risk_engine: DeterministicRiskEngine | None = None,
        portfolio: IntradayPaperPortfolio | None = None,
    ) -> None:
        self.feed = feed
        self.directional_agent = directional_agent or CentralDirectionalAgent()
        self.timing_agent = timing_agent or CandleTimingAgent()
        self.risk_engine = risk_engine or DeterministicRiskEngine()
        self.portfolio = portfolio or IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)

    def generate_daily_calls(
        self,
        *,
        symbols: Sequence[str],
        as_of_ist: datetime,
        primary_timeframe: TimeframeLiteral = "1h",
        predictions_payload: dict[str, Any] | None = None,
        intelligence_payload: dict[str, Any] | None = None,
        regime_payload: dict[str, Any] | None = None,
        snapshot_dir: Path | None = None,
        lookback_bars: int = 200,
        requested_quantity: int = 10,
    ) -> DailyPaperCallReport:
        if as_of_ist.tzinfo is None:
            raise ValueError("as_of_ist must be timezone-aware")
        cutoff_ist = as_of_ist.astimezone(IST_TZ)
        syms = [s.strip().upper() for s in symbols]

        if predictions_payload is not None and intelligence_payload is not None and regime_payload is not None:
            batch = self.directional_agent.evaluate_from_payloads(
                symbols=syms,
                as_of_ist=cutoff_ist,
                predictions_payload=predictions_payload,
                intelligence_payload=intelligence_payload,
                regime_payload=regime_payload,
            )
        else:
            out_dir = snapshot_dir or Path("data/lab/paper_snapshots")
            snap = reconstruct_as_of_snapshot(
                cutoff_ist.date(),
                out_dir,
                as_of_time_ist=cutoff_ist.time(),
                symbols=syms,
            )
            batch = self.directional_agent.evaluate_from_files(
                symbols=syms,
                as_of_ist=cutoff_ist,
                predictions_path=snap.latest_predictions_path,
                intelligence_path=snap.phase5_intelligence_path,
                regime_path=snap.regime_path,
            )

        calls: list[PaperSignalCall] = []
        for sym in syms:
            hyp = batch.hypotheses[sym]
            ev_1h = self.feed.get_bars(sym, "1h", cutoff_ist, lookback_bars=lookback_bars)
            ev_15m = self.feed.get_bars(sym, "15m", cutoff_ist, lookback_bars=lookback_bars)
            ev_5m = self.feed.get_bars(sym, "5m", cutoff_ist, lookback_bars=lookback_bars)

            ev_map = {"1h": ev_1h, "15m": ev_15m, "5m": ev_5m}
            ev_primary = ev_map[primary_timeframe]

            bars_primary = _evidence_to_agent_df(ev_primary)
            bars_15m = _evidence_to_agent_df(ev_15m)
            bars_5m = _evidence_to_agent_df(ev_5m)

            timing = self.timing_agent.evaluate(
                hypothesis=hyp,
                bars_primary=bars_primary,
                primary_timeframe=primary_timeframe,
                bars_15m=bars_15m,
                bars_5m=bars_5m,
                as_of_ist=cutoff_ist,
            )

            risk_eval: RiskEvaluationResult | None = None
            if (
                timing.decision == TimingDecision.ENTER_NOW
                and bars_primary is not None
                and not bars_primary.empty
            ):
                entry_price = float(bars_primary["close"].iloc[-1])
                stop_loss = getattr(timing, "stop_loss_price", None) or round(entry_price * 0.985, 2)
                risk_eval = self.risk_engine.evaluate_order(
                    symbol=sym,
                    side="BUY",
                    requested_quantity=requested_quantity,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    portfolio=self.portfolio,
                    current_time=cutoff_ist,
                )

            calls.append(
                PaperSignalCall(
                    symbol=sym,
                    as_of_ist=cutoff_ist.isoformat(),
                    hypothesis=hyp,
                    timing=timing,
                    risk_evaluation=risk_eval,
                    primary_provenance=ev_primary.provenance_status.value,
                    bars_5m_count=ev_5m.bar_count,
                    bars_15m_count=ev_15m.bar_count,
                    bars_1h_count=ev_1h.bar_count,
                )
            )

        return DailyPaperCallReport(
            as_of_ist=cutoff_ist.isoformat(),
            provider_name=self.feed.provider_name,
            primary_timeframe=primary_timeframe,
            paper_only=True,
            risk_state=self.risk_engine.state.value,
            calls=tuple(calls),
        )
