"""Automated unit and integration tests for Multi-Agent Council, Model Routing, and Autonomous Trader."""

from datetime import datetime, timedelta
import pytest

from trader.config import IST_TZ
from trader.data.schema import Candle5M
from trader.data.loader import generate_synthetic_5m_session
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide
from trader.risk.engine import DeterministicRiskEngine
from trader.agents.contracts import (
    MarketPosture,
    MarketIntelligenceMessage,
    TechnicalAnalysisMessage,
    MicrostructureMessage,
    NewsSentimentMessage,
)
from trader.agents.router import ModelRouter, TokenBudgetManager
from trader.agents.market_intelligence import MarketIntelligenceAgent
from trader.agents.technical_agent import TechnicalAnalysisAgent
from trader.agents.news_agent import NewsSentimentAgent
from trader.agents.microstructure_agent import MicrostructureAgent
from trader.agents.council import CentralMarketCouncil
from trader.agents.autonomous_trader import AutonomousTradingAgent


def test_market_intelligence_agent():
    """Verify market breadth, VIX classification, and direction output."""
    agent = MarketIntelligenceAgent()
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Bullish scenario
    msg = agent.analyze_market(
        nifty_change_pct=0.85,
        bank_nifty_change_pct=1.10,
        india_vix=14.5,
        advances=1650,
        declines=800,
        fii_net_crores=1200.0,
        dii_net_crores=450.0,
        timestamp=ts,
    )
    assert msg.market_direction == "BULLISH"
    assert msg.vix_regime == "NORMAL"
    assert msg.advance_decline_ratio == 2.06
    assert msg.fii_dii_bias == "BUYING"
    assert msg.confidence >= 0.70


def test_technical_analysis_agent():
    """Verify technical agent computes indicators and outputs structured message."""
    agent = TechnicalAnalysisAgent()
    candles = generate_synthetic_5m_session(symbol="RELIANCE", base_price=2500.0, daily_trend_pct=1.0)
    msg = agent.analyze_candles(candles, prev_session_ohlc=(2480.0, 2440.0, 2470.0))

    assert msg.symbol == "RELIANCE"
    assert msg.timeframe == "5m"
    assert msg.vwap > 0.0
    assert msg.rsi_14 >= 0.0 and msg.rsi_14 <= 100.0
    assert len(msg.evidence) > 0


def test_news_sentiment_pit_cutoff():
    """Assert news published after cutoff timestamp is strictly quarantined."""
    agent = NewsSentimentAgent()
    analysis_ts = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)
    cutoff_ts = datetime(2026, 11, 20, 10, 30, tzinfo=IST_TZ)

    headlines = [
        {"title": "RELIANCE wins major ₹5000 Cr green energy contract", "published_at": datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ), "materiality": 0.8},
        {"title": "RELIANCE acquires European solar firm", "published_at": datetime(2026, 11, 20, 10, 45, tzinfo=IST_TZ), "materiality": 0.7}, # AFTER CUTOFF
    ]

    msg = agent.analyze_news(
        symbol="RELIANCE",
        headlines=headlines,
        analysis_timestamp=analysis_ts,
        data_cutoff_timestamp=cutoff_ts,
    )

    assert len(msg.headlines) == 1
    assert "₹5000 Cr green energy" in msg.headlines[0]
    assert "European solar" not in msg.headlines[0]
    assert msg.sentiment == "BULLISH"


def test_microstructure_agent():
    """Assert spread calculation and execution feasibility."""
    agent = MicrostructureAgent()
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Tight spread on liquid stock
    msg_liquid = agent.analyze_microstructure("TCS", "NSE", best_bid=3500.0, best_ask=3500.50, traded_volume_20m=25000.0, timestamp=ts)
    assert msg_liquid.liquidity_status == "HIGH"
    assert msg_liquid.execution_feasibility == "PASS"
    assert msg_liquid.bid_ask_spread_bps < 5.0

    # Wide spread on illiquid stock
    msg_illiquid = agent.analyze_microstructure("ILLIQ", "BSE", best_bid=100.0, best_ask=105.0, traded_volume_20m=500.0, timestamp=ts)
    assert msg_illiquid.liquidity_status == "ILLIQUID"
    assert msg_illiquid.execution_feasibility == "REJECT"


def test_model_router_and_token_budget():
    """Assert token budget tracking and automatic degraded fallback when budget is exhausted."""
    budget = TokenBudgetManager(daily_budget_usd=0.01) # Low budget to test tripping
    router = ModelRouter(budget_manager=budget)

    # Level 0 is always deterministic
    assert router.select_model(task_level=0, agent_name="risk") == "deterministic_python"

    # Level 1 initially gets Flash
    assert router.select_model(task_level=1, agent_name="technical") == "gemini-2.5-flash"

    # Simulate token consumption that exceeds $0.01 budget
    # 20,000 output tokens on Gemini Pro = (20,000 / 1M) * $5.00 = $0.10 > $0.01
    cost = budget.record_usage(agent_name="central_council", model_name="gemini-2.5-pro", input_tokens=10_000, output_tokens=20_000)
    assert cost > 0.01
    assert budget.is_daily_budget_exhausted is True

    # Now router must gracefully degrade to deterministic fallback
    model_after = router.select_model(task_level=1, agent_name="technical")
    assert model_after == "deterministic_fallback"

    summary = budget.get_summary()
    assert summary["budget_exhausted"] is True
    assert summary["total_calls_count"] == 1


def test_central_council_posture_and_event_broadcast():
    """Assert council produces valid 4-posture verdict and broadcasts shifts to Autonomous Trader."""
    council = CentralMarketCouncil()
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    risk_engine = DeterministicRiskEngine()
    trader = AutonomousTradingAgent(portfolio=portfolio, risk_engine=risk_engine)

    # Subscribe Autonomous Trader to Central Council
    council.subscribe(trader.on_council_posture_update)

    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    intel_bullish = MarketIntelligenceMessage(
        timestamp=ts,
        nifty_trend="BULLISH",
        bank_nifty_trend="BULLISH",
        india_vix=14.0,
        vix_regime="NORMAL",
        advance_decline_ratio=2.2,
        fii_dii_bias="BUYING",
        leading_sectors=["Banking"],
        lagging_sectors=[],
        market_direction="BULLISH",
        confidence=0.85,
        supporting_evidence=["Strong breadth", "FII buying"],
    )

    report = council.evaluate_council(market_intel=intel_bullish)
    assert report.posture == MarketPosture.BUY_BULLISH
    assert trader.current_council_posture == MarketPosture.BUY_BULLISH

    # Now simulate sudden market shock: India VIX surges to 26.5
    intel_shock = MarketIntelligenceMessage(
        timestamp=ts + timedelta(hours=1),
        nifty_trend="BEARISH",
        bank_nifty_trend="BEARISH",
        india_vix=26.5,
        vix_regime="HIGH_VOLATILITY",
        advance_decline_ratio=0.35,
        fii_dii_bias="SELLING",
        leading_sectors=[],
        lagging_sectors=["IT", "Banking"],
        market_direction="BEARISH",
        confidence=0.90,
        supporting_evidence=["VIX spike", "Breadth collapse"],
    )

    report_shock = council.evaluate_council(market_intel=intel_shock)
    assert report_shock.posture == MarketPosture.HOLD_HIGH_RISK
    # Assert Autonomous Trader received the broadcast
    assert trader.current_council_posture == MarketPosture.HOLD_HIGH_RISK


def test_autonomous_trader_5m_order_execution():
    """Assert Autonomous Trader enters long on valid setup and manages order via Risk Engine."""
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    risk_engine = DeterministicRiskEngine()
    trader = AutonomousTradingAgent(portfolio=portfolio, risk_engine=risk_engine)

    # Set Council to BULLISH
    trader.current_council_posture = MarketPosture.BUY_BULLISH

    ts = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    tech = TechnicalAnalysisMessage(
        symbol="RELIANCE",
        exchange="NSE",
        timeframe="5m",
        timestamp=ts,
        signal="BULLISH",
        confidence=0.78,
        current_price=2500.0,
        vwap=2480.0,
        above_vwap=True,
        ema_9=2495.0,
        ema_21=2485.0,
        ema_50=2470.0,
        rsi_14=62.0,
        macd_signal="BULLISH_CROSS",
        atr_14=20.0,
        volume_ratio=1.8,
        evidence=["Price above VWAP", "EMA trend alignment", "Volume surge"],
    )
    micro = MicrostructureMessage(
        symbol="RELIANCE",
        exchange="NSE",
        timestamp=ts,
        bid_ask_spread_bps=2.5,
        liquidity_status="HIGH",
        estimated_slippage_bps=3.0,
        execution_feasibility="PASS",
    )

    action = trader.process_5m_candle_setup(symbol="RELIANCE", tech_signal=tech, microstructure=micro)
    assert action.action == "ENTER_LONG"
    assert action.quantity > 0
    assert "RELIANCE" in portfolio.positions
    assert portfolio.positions["RELIANCE"].stop_loss == 2480.0  # Entry 2500 - 1.0*ATR (20)
    assert portfolio.positions["RELIANCE"].target == 2540.0     # Entry 2500 + 2.0*ATR (40)


def test_gemini_provider_invocation_and_deterministic_fallback(monkeypatch):
    """Verify genuine Gemini REST invocation (mocked HTTP) records real tokens/cost and falls back deterministically when unavailable."""
    from trader.agents.router import GeminiProvider, ModelRouter, TokenBudgetManager
    import httpx

    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "Nifty breadth bullish; favor banking."}]}}],
                "usageMetadata": {"promptTokenCount": 420, "candidatesTokenCount": 80},
            }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            assert "gemini-2.5-pro:generateContent" in url
            return DummyResp()

    monkeypatch.setattr(httpx, "Client", DummyClient)

    provider = GeminiProvider(api_key="test-fake-key-for-mock")
    router = ModelRouter(budget_manager=TokenBudgetManager(daily_budget_usd=0.50), provider=provider)

    res = router.invoke_agent(
        task_level=2,
        agent_name="central_council",
        prompt="Analyze Nifty",
        deterministic_fn=lambda: "Deterministic fallback",
    )
    assert res.execution_mode == "LLM"
    assert res.model_used == "gemini-2.5-pro"
    assert res.input_tokens == 420
    assert res.output_tokens == 80
    assert res.cost_usd > 0.0
    assert "Nifty breadth bullish" in res.text

    # Test deterministic fallback when API key missing
    router_no_key = ModelRouter(provider=GeminiProvider(api_key=""))
    res_fb = router_no_key.invoke_agent(
        task_level=2,
        agent_name="central_council",
        prompt="Analyze Nifty",
        deterministic_fn=lambda: "Deterministic fallback",
    )
    assert res_fb.execution_mode == "DETERMINISTIC_FALLBACK"
    assert res_fb.text == "Deterministic fallback"
    assert res_fb.cost_usd == 0.0


def test_council_hourly_and_event_reassessment():
    """Verify Central Council reassesses hourly and immediately upon material VIX shock."""
    from datetime import timedelta

    mi_agent = MarketIntelligenceAgent()
    council = CentralMarketCouncil()

    t0 = datetime(2026, 11, 20, 9, 30, tzinfo=IST_TZ)
    intel_bull = mi_agent.analyze_market(1.0, 1.2, india_vix=13.5, advances=1700, declines=600, timestamp=t0)

    should, reason = council.should_reassess(t0, intel_bull)
    assert should is True
    assert "INITIAL" in reason
    council.evaluate_council(intel_bull)

    # 20 minutes later, no material change -> should NOT reassess
    t1 = t0 + timedelta(minutes=20)
    should_1, _ = council.should_reassess(t1, intel_bull)
    assert should_1 is False

    # 25 minutes later, VIX spikes to 24.5 -> MATERIAL EVENT triggers immediate reassessment
    t2 = t0 + timedelta(minutes=25)
    intel_shock = mi_agent.analyze_market(-1.5, -2.0, india_vix=24.5, advances=300, declines=1900, timestamp=t2)
    should_2, reason_2 = council.should_reassess(t2, intel_shock)
    assert should_2 is True
    assert "MATERIAL_EVENT" in reason_2

    # 65 minutes later -> SCHEDULED_HOURLY_REASSESSMENT
    t3 = council.last_assessment.timestamp + timedelta(minutes=65)
    should_3, reason_3 = council.should_reassess(t3, intel_bull)
    assert should_3 is True
    assert "SCHEDULED_HOURLY_REASSESSMENT" in reason_3

