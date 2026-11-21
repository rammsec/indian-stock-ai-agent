"""Continuous Autonomous Paper-Trading Desk Runner (Phase 7).

Manages the real-time or simulated market session loop:
1. Hourly Central Council assessment (Layer A)
2. Every 5m candle: scan, indicator computation, autonomous order management (Layer B)
3. Enforces deterministic risk engine limits
4. Enforces 15:15 IST mandatory square-off
5. Updates cryptographic SHA-256 event ledger and generates session journal
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from .config import (
    DEFAULT_INITIAL_CAPITAL_INR,
    INTRADAY_MANDATORY_SQUAREOFF_TIME,
    IST_TZ,
)
from .data.schema import Candle5M
from .data.loader import generate_synthetic_5m_session
from .features.indicators import compute_all_features
from .portfolio.ledger import IntradayPaperPortfolio, OrderSide
from .risk.engine import DeterministicRiskEngine
from .agents.contracts import MarketPosture
from .agents.market_intelligence import MarketIntelligenceAgent
from .agents.technical_agent import TechnicalAnalysisAgent
from .agents.microstructure_agent import MicrostructureAgent
from .agents.council import CentralMarketCouncil
from .agents.autonomous_trader import AutonomousTradingAgent


class IntradayTradingDeskRunner:
    """Orchestrates continuous multi-agent intraday research and paper trading."""

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL_INR,
        symbols: list[str] | None = None,
    ) -> None:
        self.symbols = symbols or ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
        self.portfolio = IntradayPaperPortfolio(initial_capital=initial_capital)
        self.risk_engine = DeterministicRiskEngine()
        self.mi_agent = MarketIntelligenceAgent()
        self.tech_agent = TechnicalAnalysisAgent()
        self.micro_agent = MicrostructureAgent()
        self.council = CentralMarketCouncil()
        self.trader = AutonomousTradingAgent(portfolio=self.portfolio, risk_engine=self.risk_engine)

        # Wire Council posture shifts directly to Autonomous Trader
        self.council.subscribe(self.trader.on_council_posture_update)

        self.session_log: list[dict[str, Any]] = []

    def run_simulated_session(
        self,
        session_date: datetime.date | None = None,
        market_bias: str = "BULLISH",
    ) -> dict[str, Any]:
        """Run an entire 75-candle intraday trading session (09:15 to 15:30 IST) deterministically."""
        if session_date is None:
            session_date = datetime.now(tz=IST_TZ).date()

        # 1. Pre-market / Session Open Council Assessment (Layer A)
        trend_val = 0.8 if market_bias == "BULLISH" else (-0.8 if market_bias == "BEARISH" else 0.0)
        vix_val = 14.5 if market_bias != "HIGH_VOLATILITY" else 26.5

        open_ts = datetime(session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST_TZ)
        market_msg = self.mi_agent.analyze_market(
            nifty_change_pct=trend_val,
            bank_nifty_change_pct=trend_val * 1.2,
            india_vix=vix_val,
            advances=1600 if market_bias == "BULLISH" else 700,
            declines=700 if market_bias == "BULLISH" else 1600,
            fii_net_crores=1200.0 if market_bias == "BULLISH" else -1200.0,
            dii_net_crores=500.0,
            timestamp=open_ts,
        )
        council_report = self.council.evaluate_council(market_intel=market_msg)
        self.trader.on_council_posture_update(council_report)

        # 2. Generate 75 5-minute candles per symbol for the session
        stock_candles: dict[str, list[Candle5M]] = {}
        base_prices = {"RELIANCE": 2500.0, "TCS": 3500.0, "HDFCBANK": 1600.0, "INFY": 1800.0, "ITC": 430.0}

        for sym in self.symbols:
            bp = base_prices.get(sym, 2000.0)
            stock_candles[sym] = generate_synthetic_5m_session(
                symbol=sym,
                session_date=session_date,
                base_price=bp,
                daily_trend_pct=trend_val,
            )

        # 3. Step through all 75 5-minute bars chronologically (09:15 to 15:30 IST)
        num_bars = 75
        for bar_idx in range(num_bars):
            # Evaluate each stock on the closed candle
            for sym in self.symbols:
                candles_to_now = stock_candles[sym][: bar_idx + 1]
                current_candle = candles_to_now[-1]
                current_time = current_candle.timestamp.astimezone(IST_TZ).time()

                # At or past 15:15 IST -> Force square-off
                if current_time >= INTRADAY_MANDATORY_SQUAREOFF_TIME:
                    if sym in self.portfolio.positions:
                        self.portfolio.square_off_all_positions(
                            current_prices={sym: current_candle.close},
                            timestamp=current_candle.timestamp,
                        )
                    continue

                if len(candles_to_now) < 5:
                    continue

                # Run Technical and Microstructure analysis
                tech_msg = self.tech_agent.analyze_candles(candles_to_now)
                micro_msg = self.micro_agent.analyze_microstructure(
                    symbol=sym,
                    exchange=current_candle.exchange,
                    best_bid=current_candle.close - 0.20,
                    best_ask=current_candle.close + 0.20,
                    traded_volume_20m=current_candle.volume * 4,
                    timestamp=current_candle.timestamp,
                )

                # Autonomous Trader decision
                action = self.trader.process_5m_candle_setup(sym, tech_msg, micro_msg)
                if action.action in ("ENTER_LONG", "EXIT"):
                    self.session_log.append({
                        "bar_idx": bar_idx,
                        "time": current_candle.timestamp.strftime("%H:%M"),
                        "symbol": sym,
                        "action": action.action,
                        "quantity": action.quantity,
                        "price": action.entry_price,
                        "stop_loss": action.stop_loss,
                        "target": action.target,
                        "rationale": action.rationale,
                    })

        # Final reconciliation
        summary = self.portfolio.get_portfolio_summary()
        valid, msg = self.portfolio.verify_ledger_integrity()

        return {
            "session_date": session_date.isoformat(),
            "market_posture": council_report.posture.value,
            "council_confidence": council_report.confidence,
            "portfolio_summary": summary,
            "ledger_integrity": valid,
            "ledger_integrity_message": msg,
            "total_decisions_executed": len(self.session_log),
            "closed_trades_count": len(self.portfolio.closed_trades),
            "session_log": self.session_log,
        }


# --- Phase 7 Continuous Paper Trading Runner with Stale-Feed & Recovery ---

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .config import (
    IST_TZ,
    INTRADAY_ENTRY_CUTOFF_TIME,
    INTRADAY_MANDATORY_SQUAREOFF_TIME,
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    is_nse_trading_day,
)
from .data.schema import Candle5M, DataProvenance, DataSourceMode
from .data.validator import validate_candle_series
from .portfolio.ledger import IntradayPaperPortfolio
from .risk.engine import DeterministicRiskEngine
from .agents.market_intelligence import MarketIntelligenceAgent
from .agents.technical_agent import TechnicalAnalysisAgent
from .agents.microstructure_agent import MicrostructureAgent
from .agents.council import CentralMarketCouncil
from .agents.autonomous_trader import AutonomousTradingAgent


class BrokerExecutionDisabledError(RuntimeError):
    """Raised if any caller attempts to enable live broker order execution."""


class StaleFeedError(RuntimeError):
    """Raised when incoming market data feed exceeds maximum staleness threshold."""


@dataclass
class RunnerHealthStatus:
    """Live health and recovery telemetry for the continuous paper trading loop."""

    is_running: bool = False
    session_date: str = ""
    cycles_completed: int = 0
    consecutive_errors: int = 0
    total_recoveries: int = 0
    stale_feed_events: int = 0
    last_bar_timestamp: datetime | None = None
    last_council_reassessment: datetime | None = None
    broker_execution_enabled: bool = False
    status_message: str = "INITIALIZED"
    event_log: list[str] = field(default_factory=list)


class ContinuousPaperTradingRunner:
    """Orchestrates continuous 5-minute paper trading during Indian market hours.

    Enforces:
    - Broker integration and real orders strictly DISABLED (`broker_execution_enabled=False`)
    - Stale-feed detection (`max_feed_age_seconds`)
    - Automatic recovery up to `max_consecutive_errors`
    - Hourly and event-triggered Central Council reassessment
    - Mandatory 15:15 IST square-off and daily journal generation
    """

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = 1_000_000.0,
        max_feed_age_seconds: float = 360.0,
        max_consecutive_errors: int = 3,
        broker_execution_enabled: bool = False,
    ) -> None:
        if broker_execution_enabled:
            raise BrokerExecutionDisabledError(
                "Live broker integration and real orders are strictly DISABLED. Paper simulation only."
            )

        self.symbols = symbols
        self.max_feed_age_seconds = max_feed_age_seconds
        self.max_consecutive_errors = max_consecutive_errors
        self.broker_execution_enabled = False

        self.portfolio = IntradayPaperPortfolio(initial_capital=initial_capital)
        self.risk_engine = DeterministicRiskEngine()
        self.mi_agent = MarketIntelligenceAgent()
        self.tech_agent = TechnicalAnalysisAgent()
        self.micro_agent = MicrostructureAgent()
        self.council = CentralMarketCouncil()
        self.trader = AutonomousTradingAgent(
            portfolio=self.portfolio,
            risk_engine=self.risk_engine,
            router=self.council.router,
        )
        self.council.subscribe(self.trader.on_council_posture_update)

        self.health = RunnerHealthStatus()
        self._candle_buffers: dict[str, list[Candle5M]] = {s: [] for s in symbols}

    def check_feed_freshness(self, latest_bar_ts: datetime, reference_time: datetime) -> bool:
        """Return True if feed is fresh; raise StaleFeedError if stale."""
        age_sec = (reference_time - latest_bar_ts).total_seconds()
        if age_sec > self.max_feed_age_seconds:
            self.health.stale_feed_events += 1
            raise StaleFeedError(
                f"Stale market data feed: latest bar {latest_bar_ts.isoformat()} is {age_sec:.1f}s old "
                f"(limit: {self.max_feed_age_seconds}s)"
            )
        return True

    def is_within_indian_session(self, current_time: datetime) -> bool:
        """Check if current IST timestamp is an active NSE/BSE trading day & market hours."""
        ist_dt = current_time.astimezone(IST_TZ)
        if not is_nse_trading_day(ist_dt.date()):
            return False
        t = ist_dt.time()
        return MARKET_OPEN_TIME <= t <= MARKET_CLOSE_TIME

    def step_cycle(
        self,
        fetch_candles_fn: Callable[[str], list[Candle5M]],
        current_time: datetime,
        nifty_change_pct: float = 0.25,
        bank_nifty_change_pct: float = 0.30,
        india_vix: float = 14.5,
        advances: int = 1350,
        declines: int = 950,
    ) -> dict[str, object]:
        """Execute a single 5-minute ingestion, council check, and trading cycle with recovery."""
        ist_now = current_time.astimezone(IST_TZ)
        self.health.is_running = True
        self.health.session_date = ist_now.date().isoformat()

        # 1. Session & Holiday Guard
        if not self.is_within_indian_session(ist_now):
            self.health.status_message = "OUTSIDE_MARKET_HOURS_OR_HOLIDAY"
            return {"status": "SKIPPED", "reason": self.health.status_message}

        # 2. Mandatory 15:15 IST Square-Off
        if ist_now.time() >= INTRADAY_MANDATORY_SQUAREOFF_TIME and len(self.portfolio.positions) > 0:
            latest_prices = {
                sym: self._candle_buffers[sym][-1].close
                for sym in self._candle_buffers
                if self._candle_buffers[sym]
            }
            sq_orders = self.portfolio.square_off_all_positions(latest_prices, timestamp=ist_now)
            self.health.event_log.append(
                f"[{ist_now.strftime('%H:%M')}] Mandatory 15:15 IST square-off executed ({len(sq_orders)} orders)"
            )
            self.health.cycles_completed += 1
            self.health.status_message = "SQUARED_OFF_FOR_DAY"
            return {"status": "SQUARED_OFF", "orders_count": len(sq_orders)}

        # 3. Council Reassessment Check (Hourly or Material Event)
        market_msg = self.mi_agent.analyze_market(
            nifty_change_pct=nifty_change_pct,
            bank_nifty_change_pct=bank_nifty_change_pct,
            india_vix=india_vix,
            advances=advances,
            declines=declines,
            timestamp=ist_now,
        )
        should_update, reason = self.council.should_reassess(ist_now, market_msg)
        if should_update:
            posture_msg = self.council.evaluate_council(market_intel=market_msg)
            self.health.last_council_reassessment = ist_now
            self.health.event_log.append(
                f"[{ist_now.strftime('%H:%M')}] Council reassessed ({reason}) -> {posture_msg.posture.value}"
            )

        # 4. Ingest & Process each symbol with automatic error recovery
        actions_taken = []
        for sym in self.symbols:
            try:
                raw_candles = fetch_candles_fn(sym)
                valid_candles, _ = validate_candle_series(raw_candles)
                if not valid_candles:
                    raise ValueError(f"No valid candles returned for {sym}")

                latest_bar = valid_candles[-1]
                self.check_feed_freshness(latest_bar.timestamp, ist_now)
                self.health.last_bar_timestamp = latest_bar.timestamp
                self._candle_buffers[sym] = valid_candles

                if self.health.consecutive_errors > 0:
                    self.health.total_recoveries += 1
                    self.health.consecutive_errors = 0
                    self.health.event_log.append(f"[{ist_now.strftime('%H:%M')}] Feed recovered for {sym}")

                if len(valid_candles) >= 20:
                    tech_msg = self.tech_agent.analyze_candles(valid_candles)
                    micro_msg = self.micro_agent.analyze_microstructure(
                        sym,
                        latest_bar.exchange,
                        best_bid=tech_msg.current_price - 0.25,
                        best_ask=tech_msg.current_price + 0.25,
                        traded_volume_20m=sum(c.volume for c in valid_candles[-4:]),
                    )
                    action = self.trader.process_5m_candle_setup(sym, tech_msg, micro_msg)
                    actions_taken.append({"symbol": sym, "action": action.action})

            except Exception as exc:
                self.health.consecutive_errors += 1
                self.health.event_log.append(
                    f"[{ist_now.strftime('%H:%M')}] Error on {sym} ({self.health.consecutive_errors}/{self.max_consecutive_errors}): {exc}"
                )
                if self.health.consecutive_errors >= self.max_consecutive_errors:
                    self.health.is_running = False
                    self.health.status_message = f"HALTED_MAX_ERRORS: {exc}"
                    return {"status": "HALTED", "error": str(exc)}

        self.health.cycles_completed += 1
        self.health.status_message = "HEALTHY_ACTIVE"
        return {
            "status": "OK",
            "cycle": self.health.cycles_completed,
            "actions": actions_taken,
            "posture": self.council.current_posture.value,
        }

    def generate_daily_journal(self, output_path: Path | None = None) -> str:
        """Generate end-of-day markdown journal and performance report."""
        summary = self.portfolio.get_portfolio_summary()
        valid_ledger, ledger_msg = self.portfolio.verify_ledger_integrity()
        cost_summary = self.council.router.budget_manager.get_summary()

        lines = [
            f"# Daily Paper Trading Journal — {self.health.session_date or datetime.now(tz=IST_TZ).date().isoformat()}",
            "",
            "## 1. Execution & Safety Guardrails",
            f"- **Broker Integration**: DISABLED (`broker_execution_enabled={self.broker_execution_enabled}`)",
            f"- **Runner Status**: `{self.health.status_message}`",
            f"- **Cycles Completed**: {self.health.cycles_completed} | **Stale Feed Events**: {self.health.stale_feed_events} | **Auto-Recoveries**: {self.health.total_recoveries}",
            f"- **Ledger Integrity**: {'VERIFIED ✓' if valid_ledger else 'FAILED ✗'} ({ledger_msg})",
            "",
            "## 2. Portfolio Performance",
            f"- **Initial Capital**: ₹{summary['initial_capital_inr']:,.2f}",
            f"- **Ending Equity**: ₹{summary['total_equity_inr']:,.2f} ({summary['total_return_pct']:+.2f}%)",
            f"- **Realized P&L**: ₹{summary['realized_pnl_inr']:,.2f} | **Unrealized P&L**: ₹{summary['unrealized_pnl_inr']:,.2f}",
            f"- **Closed Trades**: {summary['closed_trades_count']} | **Win Rate**: {summary['win_rate_pct']:.1f}% | **Profit Factor**: {summary['profit_factor']:.2f}",
            "",
            "## 3. Token & API Cost Summary",
            f"- **Total Calls**: {cost_summary['total_calls_count']} (LLM: {cost_summary['llm_calls_count']}, Deterministic: {cost_summary['fallback_calls_count']})",
            f"- **Daily Spend**: ${cost_summary['current_daily_spend_usd']:.4f} / ${cost_summary['daily_budget_usd']:.2f}",
            "",
            "## 4. Session Event Log",
        ]
        for evt in self.health.event_log[-25:]:
            lines.append(f"- {evt}")

        report = "\n".join(lines) + "\n"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
        return report
