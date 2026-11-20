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
