"""Generate consolidated JSON data package for the Intraday Trading Desk dashboard."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import IST_TZ
from trader.data.instruments import INSTRUMENT_MASTER
from trader.data.loader import load_historical_5m_candles, generate_synthetic_5m_session
from trader.data.schema import DataSourceMode
from trader.features.indicators import compute_all_features
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide
from trader.risk.engine import DeterministicRiskEngine
from trader.agents.contracts import MarketPosture
from trader.agents.market_intelligence import MarketIntelligenceAgent
from trader.agents.technical_agent import TechnicalAnalysisAgent
from trader.agents.news_agent import NewsSentimentAgent
from trader.agents.microstructure_agent import MicrostructureAgent
from trader.agents.council import CentralMarketCouncil
from trader.agents.autonomous_trader import AutonomousTradingAgent
from trader.strategies.vwap_pullback import VWAPPullbackStrategy
from trader.strategies.orb_breakout import ORBBreakoutStrategy
from trader.strategies.ema_cross import EMACrossStrategy
from trader.strategies.supertrend import SupertrendStrategy
from trader.strategies.backtester import IntradayBacktester
from trader.arbitrage.spread_calculator import calculate_executable_arbitrage_spread

OUTPUT_DIR = Path("docs/trader/data")


def build_trader_dataset(data_mode: DataSourceMode = DataSourceMode.SYNTHETIC_TEST) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now_ist = datetime.now(tz=IST_TZ)

    # 1. Market Intelligence & Central Council
    mi_agent = MarketIntelligenceAgent()
    market_msg = mi_agent.analyze_market(
        nifty_change_pct=0.74,
        bank_nifty_change_pct=1.12,
        india_vix=14.35,
        advances=1680,
        declines=740,
        fii_net_crores=1450.0,
        dii_net_crores=620.0,
        timestamp=now_ist,
    )

    council = CentralMarketCouncil()
    council_msg = council.evaluate_council(market_intel=market_msg)

    # 2. Portfolio & Risk Engine Simulation
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    risk_engine = DeterministicRiskEngine()
    auto_trader = AutonomousTradingAgent(portfolio=portfolio, risk_engine=risk_engine, router=council.router)
    auto_trader.on_council_posture_update(council_msg)

    tech_agent = TechnicalAnalysisAgent()
    micro_agent = MicrostructureAgent()

    # 3. Stock Scanner Generation across 5 Core Watchlist Equities
    watchlist = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
    scanner_results = []
    provenance_records = []
    base_prices = {"RELIANCE": 2520.0, "TCS": 3540.0, "HDFCBANK": 1625.0, "INFY": 1820.0, "ITC": 435.0}

    for sym in watchlist:
        bp = base_prices.get(sym, 2000.0)
        if data_mode == DataSourceMode.SYNTHETIC_TEST:
            candles, prov = generate_synthetic_5m_session(
                symbol=sym, base_price=bp, daily_trend_pct=0.8, return_provenance=True
            )
        else:
            candles, prov = load_historical_5m_candles(symbol=sym, exchange="NSE", mode=data_mode)

        provenance_records.append({
            "symbol": prov.symbol,
            "exchange": prov.exchange,
            "source_mode": prov.source_mode.value,
            "source_name": prov.source_name,
            "bars_count": prov.bars_count,
            "start_timestamp": prov.start_timestamp.isoformat(),
            "end_timestamp": prov.end_timestamp.isoformat(),
            "sha256_hash": prov.sha256_hash,
        })

        df_feat = compute_all_features(candles)
        last_bar = df_feat.iloc[-1]

        tech_msg = tech_agent.analyze_candles(candles)
        micro_msg = micro_agent.analyze_microstructure(
            sym, "NSE", best_bid=tech_msg.current_price - 0.25, best_ask=tech_msg.current_price + 0.25, traded_volume_20m=35000.0
        )

        action_msg = auto_trader.process_5m_candle_setup(sym, tech_msg, micro_msg)

        # Recent 5m candle points for mini-chart
        recent_bars = [
            {
                "time": c.timestamp.strftime("%H:%M"),
                "open": round(c.open, 2),
                "high": round(c.high, 2),
                "low": round(c.low, 2),
                "close": round(c.close, 2),
                "volume": int(c.volume),
            }
            for c in candles[-24:]  # Last 2 hours of 5m bars
        ]

        scanner_results.append({
            "symbol": sym,
            "company_name": INSTRUMENT_MASTER[sym].company_name if sym in INSTRUMENT_MASTER else sym,
            "sector": INSTRUMENT_MASTER[sym].sector if sym in INSTRUMENT_MASTER else "Equity",
            "current_price": round(tech_msg.current_price, 2),
            "vwap": round(tech_msg.vwap, 2),
            "above_vwap": tech_msg.above_vwap,
            "ema_9": round(tech_msg.ema_9, 2),
            "ema_21": round(tech_msg.ema_21, 2),
            "ema_50": round(tech_msg.ema_50, 2),
            "rsi_14": round(tech_msg.rsi_14, 1),
            "volume_ratio": round(tech_msg.volume_ratio, 2),
            "technical_signal": tech_msg.signal,
            "technical_confidence": int(tech_msg.confidence * 100),
            "trading_agent_action": action_msg.action,
            "trading_rationale": action_msg.rationale,
            "stop_loss": action_msg.stop_loss,
            "target": action_msg.target,
            "recent_candles": recent_bars,
            "data_mode": prov.source_mode.value,
            "data_sha256": prov.sha256_hash[:16],
        })

    # 4. Cross-Exchange Opportunities
    arb_opportunities = [
        calculate_executable_arbitrage_spread("RELIANCE", nse_bid=2522.0, nse_ask=2522.5, bse_bid=2515.0, bse_ask=2516.0, quantity=100),
        calculate_executable_arbitrage_spread("TCS", nse_bid=3540.0, nse_ask=3540.5, bse_bid=3539.5, bse_ask=3540.0, quantity=50),
        calculate_executable_arbitrage_spread("HDFCBANK", nse_bid=1626.0, nse_ask=1626.5, bse_bid=1621.0, bse_ask=1622.0, quantity=150),
        calculate_executable_arbitrage_spread("INFY", nse_bid=1820.0, nse_ask=1820.5, bse_bid=1819.5, bse_ask=1820.0, quantity=100),
    ]

    arb_display = [
        {
            "symbol": opp.symbol,
            "isin": opp.isin,
            "buy_exchange": opp.buy_exchange,
            "sell_exchange": opp.sell_exchange,
            "buy_ask_price": opp.buy_ask_price,
            "sell_bid_price": opp.sell_bid_price,
            "quantity": opp.quantity,
            "gross_spread_inr": opp.gross_spread_inr,
            "gross_spread_pct": opp.gross_spread_pct,
            "total_friction_inr": opp.total_friction_inr,
            "net_profit_inr": opp.net_profit_inr,
            "net_return_pct": opp.net_return_pct,
            "is_actionable": opp.is_actionable,
            "status": opp.status,
            "notes": opp.notes,
        }
        for opp in arb_opportunities
    ]

    # 5. Backtest Comparison across 4 Strategies
    backtester = IntradayBacktester(initial_capital=1_000_000.0)
    rel_candles = generate_synthetic_5m_session(symbol="RELIANCE", base_price=2500.0, daily_trend_pct=1.2)

    strats = [
        VWAPPullbackStrategy(),
        ORBBreakoutStrategy(),
        EMACrossStrategy(),
        SupertrendStrategy(),
    ]
    strat_results = []
    for s in strats:
        res = backtester.run_backtest(s, rel_candles)
        strat_results.append({
            "strategy_id": res.strategy_id,
            "strategy_name": s.metadata.name,
            "origin": s.metadata.origin_source,
            "license": s.metadata.license_type,
            "validation_status": s.metadata.validation_status,
            "win_rate_pct": res.win_rate_pct,
            "profit_factor": res.profit_factor,
            "total_return_pct": res.total_return_pct,
            "max_drawdown_pct": res.max_drawdown_pct,
            "total_trades": res.total_trades,
            "friction_paid_inr": res.total_friction_paid_inr,
            "expectancy_inr": res.net_expectancy_per_trade_inr,
        })

    # 6. Portfolio Metrics
    port_summary = portfolio.get_portfolio_summary()
    open_pos = [
        {
            "symbol": pos.symbol,
            "exchange": pos.exchange,
            "quantity": pos.quantity,
            "entry_price": round(pos.entry_price, 2),
            "current_price": round(pos.current_price, 2),
            "unrealized_pnl_inr": round(pos.quantity * (pos.current_price - pos.entry_price), 2),
            "return_pct": round(((pos.current_price - pos.entry_price) / pos.entry_price) * 100.0, 2),
            "stop_loss": pos.stop_loss,
            "target": pos.target,
        }
        for pos in portfolio.positions.values()
    ]

    # 7. AI Token & Cost Statistics
    cost_summary = mi_agent.router.budget_manager.get_summary()

    # Consolidated Dashboard Payload
    payload = {
        "metadata": {
            "title": "Institutional AI Multi-Agent Intraday Trading Desk",
            "market": "NSE & BSE, India",
            "timeframe": "5-Minute Candles",
            "execution_mode": "Historical Replay & Paper Simulation Only",
            "last_updated_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            "data_mode": data_mode.value,
        },
        "data_provenance": {
            "source_mode": data_mode.value,
            "datasets_count": len(provenance_records),
            "generated_at_ist": now_ist.isoformat(),
            "datasets": provenance_records,
        },
        "market_overview": {
            "nifty_50": {"ltp": 24385.50, "change_pct": 0.74, "trend": market_msg.nifty_trend},
            "bank_nifty": {"ltp": 52140.20, "change_pct": 1.12, "trend": market_msg.bank_nifty_trend},
            "india_vix": {"value": market_msg.india_vix, "regime": market_msg.vix_regime},
            "market_breadth": {
                "advances": 1680,
                "declines": 740,
                "advance_decline_ratio": market_msg.advance_decline_ratio,
            },
            "fii_dii_flow": {
                "fii_net_crores": 1450.0,
                "dii_net_crores": 620.0,
                "bias": market_msg.fii_dii_bias,
            },
            "central_posture": {
                "posture": council_msg.posture.value,
                "confidence_pct": int(council_msg.confidence * 100),
                "summary": council_msg.summary_verdict,
                "model_used": council_msg.model_used,
                "favored_sectors": council_msg.favored_sectors,
                "avoid_sectors": council_msg.avoid_sectors,
                "restrictions": council_msg.trading_restrictions,
                "reassessment_triggers": council_msg.reassessment_triggers,
            },
            "evidence": market_msg.supporting_evidence,
        },
        "council_debate": {
            "session_id": council_msg.council_session_id,
            "model_used": council_msg.model_used,
            "bull_arguments": council_msg.bull_arguments,
            "bear_arguments": council_msg.bear_arguments,
            "consensus_verdict": council_msg.summary_verdict,
            "posture": council_msg.posture.value,
            "confidence": council_msg.confidence,
        },
        "stock_scanner": scanner_results,
        "cross_exchange_arbitrage": arb_display,
        "strategy_performance": strat_results,
        "paper_portfolio": {
            "initial_capital_inr": port_summary["initial_capital_inr"],
            "cash_inr": port_summary["cash_inr"],
            "invested_capital_inr": port_summary["invested_capital_inr"],
            "total_equity_inr": port_summary["total_equity_inr"],
            "realized_pnl_inr": port_summary["realized_pnl_inr"],
            "unrealized_pnl_inr": port_summary["unrealized_pnl_inr"],
            "total_return_pct": port_summary["total_return_pct"],
            "max_drawdown_pct": port_summary["current_drawdown_pct"],
            "win_rate_pct": port_summary["win_rate_pct"],
            "profit_factor": port_summary["profit_factor"],
            "open_positions": open_pos,
            "closed_trades": portfolio.closed_trades,
        },
        "cost_monitoring": {
            "daily_budget_usd": cost_summary["daily_budget_usd"],
            "current_daily_spend_usd": cost_summary["current_daily_spend_usd"],
            "remaining_daily_budget_usd": cost_summary["remaining_daily_budget_usd"],
            "budget_exhausted": cost_summary["budget_exhausted"],
            "total_tokens_consumed": cost_summary["total_tokens_consumed"],
            "total_calls_count": cost_summary["total_calls_count"],
            "llm_calls_count": cost_summary.get("llm_calls_count", 0),
            "fallback_calls_count": cost_summary.get("fallback_calls_count", 0),
            "model_routing_map": {
                "Agent 1 (Market Intelligence)": "Deterministic Python + Gemini 2.5 Flash",
                "Agent 2 (Technical Analysis)": "Deterministic Python + Gemini Flash",
                "Agent 3 (News Intelligence)": "Deterministic PIT + Gemini 2.5 Flash",
                "Agent 4 (Microstructure)": "Deterministic Python / Level 0 ($0.00)",
                "Agent 6 (Risk Management)": "Strict Python Gatekeeper / Level 0 ($0.00)",
                "Agent 7 (Central Council)": council_msg.model_used,
                "Agent 8 (Autonomous Trader)": "Deterministic Python Execution Engine",
            },
            "agent_costs": cost_summary["agents"],
        },
    }

    out_file = OUTPUT_DIR / "trader_state.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Trader state JSON successfully generated at: {out_file}")
    return payload


if __name__ == "__main__":
    build_trader_dataset()
