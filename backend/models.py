"""Pydantic data models for the Indian Stock Market AI Research & Next-Day Guidance Agent."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ReportPeriod(BaseModel):
    period: str
    revenue_cr: float | None = None
    expenses_cr: float | None = None
    operating_profit_cr: float | None = None
    opm_pct: float | None = None
    other_income_cr: float | None = None
    interest_cr: float | None = None
    depreciation_cr: float | None = None
    profit_before_tax_cr: float | None = None
    tax_pct: float | None = None
    net_profit_cr: float | None = None
    eps_inr: float | None = None
    dividend_payout_pct: float | None = None
    revenue_growth_pct: float | None = None
    net_profit_growth_pct: float | None = None


class CashBalanceYear(BaseModel):
    period: str
    cash_equivalents_cr: float = 0.0
    investments_cr: float = 0.0
    total_cash_in_hand_cr: float = 0.0
    long_term_debt_cr: float = 0.0
    short_term_debt_cr: float = 0.0
    lease_liabilities_cr: float = 0.0
    total_borrowings_cr: float = 0.0
    net_cash_cr: float = 0.0
    total_equity_cr: float = 0.0
    total_assets_cr: float = 0.0
    operating_cash_flow_cr: float | None = None
    investing_cash_flow_cr: float | None = None
    free_cash_flow_cr: float | None = None


class CashAndBalanceSheet(BaseModel):
    latest_period: str = ""
    cash_equivalents_cr: float = 0.0
    liquid_investments_cr: float = 0.0
    total_cash_in_hand_cr: float = 0.0
    total_borrowings_cr: float = 0.0
    net_cash_position_cr: float = 0.0
    debt_to_equity: float = 0.0
    cash_to_debt_ratio: float | None = None
    cash_pct_of_mcap: float | None = None
    latest_ocf_cr: float = 0.0
    latest_fcf_cr: float = 0.0
    cumulative_5y_ocf_cr: float = 0.0
    cumulative_5y_fcf_cr: float = 0.0
    roce_5y_series: list[dict[str, Any]] = Field(default_factory=list)
    history_5y: list[CashBalanceYear] = Field(default_factory=list)


class NamedShareholder(BaseModel):
    name: str
    category: Literal["Promoter", "FII", "DII", "Public"]
    latest_holding_pct: float
    previous_holding_pct: float | None = None
    change_4q_pct: float = 0.0
    trend: list[dict[str, Any]] = Field(default_factory=list)


class ShareholdingQuarter(BaseModel):
    period: str
    promoters_pct: float = 0.0
    fii_pct: float = 0.0
    dii_pct: float = 0.0
    government_pct: float = 0.0
    public_pct: float = 0.0
    num_shareholders: int | None = None


class StockholderAnalysis(BaseModel):
    latest_period: str = ""
    promoters_pct: float = 0.0
    fii_pct: float = 0.0
    dii_pct: float = 0.0
    government_pct: float = 0.0
    public_pct: float = 0.0
    institutional_total_pct: float = 0.0
    promoter_change_4q_pct: float = 0.0
    fii_change_4q_pct: float = 0.0
    dii_change_4q_pct: float = 0.0
    num_shareholders: int | None = None
    quarterly_history: list[ShareholdingQuarter] = Field(default_factory=list)
    named_stockholders: list[NamedShareholder] = Field(default_factory=list)


class RDYearRecord(BaseModel):
    period: str
    revenue_cr: float = 0.0
    rd_spend_cr: float = 0.0
    rd_pct_of_sales: float = 0.0
    intangible_assets_cr: float = 0.0
    cwip_cr: float = 0.0
    employee_cost_pct: float | None = None


class AnnualReportDoc(BaseModel):
    title: str
    year: str
    url: str


class RDAndInnovation(BaseModel):
    latest_rd_spend_cr: float = 0.0
    latest_rd_pct_of_sales: float = 0.0
    cumulative_5y_rd_cr: float = 0.0
    intangible_assets_cr: float = 0.0
    cwip_cr: float = 0.0
    rd_cagr_5y_pct: float | None = None
    innovation_focus: str = ""
    rd_data_source: str = ""
    history_5y: list[RDYearRecord] = Field(default_factory=list)
    annual_reports_5y: list[AnnualReportDoc] = Field(default_factory=list)


class PricePoint(BaseModel):
    date: str
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    dma50: float | None = None
    dma200: float | None = None
    volume: float | None = None


class TechnicalSnapshot(BaseModel):
    current_price: float = 0.0
    prev_close: float = 0.0
    day_change_pct: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    return_5d_pct: float = 0.0
    return_1m_pct: float = 0.0
    return_6m_pct: float = 0.0
    return_1y_pct: float = 0.0
    rsi_14: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    ema_20: float = 0.0
    dma_50: float = 0.0
    dma_200: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    atr_14: float = 0.0
    atr_pct: float = 0.0
    volume_latest: float = 0.0
    volume_20d_avg: float = 0.0
    volume_Surge_ratio: float = 1.0
    annualized_volatility_pct: float = 0.0
    pivot_point: float = 0.0
    support_1: float = 0.0
    support_2: float = 0.0
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    trend_regime: str = "NEUTRAL"
    technical_signals: list[str] = Field(default_factory=list)
    price_series: list[PricePoint] = Field(default_factory=list)


class AgentVerdict(BaseModel):
    agent_id: str
    agent_name: str
    persona_inspiration: str
    signal: Literal["BULLISH", "NEUTRAL", "BEARISH"]
    score: float
    confidence: int
    weight_pct: int
    headline: str
    key_metrics: dict[str, str | float] = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)


class NextDayGuidance(BaseModel):
    action: Literal[
        "STRONG BUY",
        "BUY",
        "HOLD / ACCUMULATE",
        "SELL / BOOK PROFIT",
        "AVOID / SHORT",
        "NO TRADE",
    ]
    conviction_score: float
    fundamental_score: float
    technical_score: float
    current_price: float
    entry_zone_low: float
    entry_zone_high: float
    target_1_next_day: float
    target_2_swing: float
    stop_loss: float
    risk_reward_ratio: float
    expected_next_day_range_low: float
    expected_next_day_range_high: float
    position_size_suggestion_pct: float
    executive_summary: str
    rationale_points: list[str] = Field(default_factory=list)
    key_catalysts: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    llm_synthesized: bool = False
    llm_model_used: str = "Quantitative Multi-Agent Committee Engine v2"


class NewsHeadline(BaseModel):
    title: str
    source: str
    published: str
    url: str
    sentiment: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"] = "NEUTRAL"


class LeaderProfile(BaseModel):
    name: str
    role: str
    focus: str
    handle_or_source: str = ""


class PromoterStatementItem(BaseModel):
    speaker_or_source: str
    published: str
    headline: str
    url: str = ""
    share_price_impact: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "NEUTRAL"
    why_it_affects_share_value: str = ""


class CompetitorIntelItem(BaseModel):
    symbol: str
    name: str
    segment: str = ""
    current_price: float = 0.0
    day_change_pct: float = 0.0
    return_1m_pct: float = 0.0
    pe: float | None = None
    roe_pct: float | None = None
    latest_statement_or_news: str = ""
    latest_news_url: str = ""
    read_across_signal: Literal["POSITIVE READ-ACROSS", "COMPETITIVE THREAT", "NEUTRAL"] = "NEUTRAL"
    impact_on_share_value: str = ""


class PromoterAndCompetitorIntel(BaseModel):
    promoter_group: str = ""
    key_leaders: list[LeaderProfile] = Field(default_factory=list)
    key_stakeholders_watch: list[str] = Field(default_factory=list)
    promoter_statements_and_articles: list[PromoterStatementItem] = Field(default_factory=list)
    competitors: list[CompetitorIntelItem] = Field(default_factory=list)


class TopMoverItem(BaseModel):
    rank: int
    symbol: str
    name: str
    sector: str
    current_price: float
    prev_close: float
    day_change_pct: float
    return_5d_pct: float = 0.0
    volume_surge_ratio: float = 1.0
    rsi_14: float = 50.0
    reason_1: str
    reason_2: str
    headline_title: str = ""
    headline_source: str = ""
    headline_url: str = ""
    next_day_verdict: str = "HOLD / WATCH"
    next_day_target: float = 0.0
    next_day_stop_loss: float = 0.0


class CompanyOverview(BaseModel):
    symbol: str
    name: str
    exchange: str = "NSE"
    bse_code: str | None = None
    nse_code: str | None = None
    sector: str = ""
    industry: str = ""
    about: str = ""
    market_cap_cr: float = 0.0
    current_price: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    stock_pe: float | None = None
    book_value: float | None = None
    dividend_yield_pct: float | None = None
    roce_pct: float | None = None
    roe_pct: float | None = None
    face_value: float | None = None
    revenue_cagr_5y_pct: float | None = None
    profit_cagr_5y_pct: float | None = None
    piotroski_score: int = 0
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


# ============================================================================
# Phase 1 — Data Foundation, Provenance, Quality & Market Regime Schemas
# ============================================================================

ValidationStatus = Literal["VERIFIED", "ESTIMATED", "MISSING", "STALE", "CONFLICTING"]

MarketRegimeLabel = Literal[
    "Strong Bull",
    "Weak Bull",
    "Sideways",
    "Bear",
    "High Volatility",
]


class DataRecord(BaseModel):
    """Canonical provenance-tracked data record required by Section 6."""

    symbol: str
    metric: str
    value: float | str | None = None
    source: str
    observation_date: str
    publication_timestamp: str
    collection_timestamp: str
    validation_status: ValidationStatus
    data_version: str = "1.0.0"


class EquityDailyBar(BaseModel):
    """Normalized daily equity bar with NSE official delivery & VWAP fields."""

    symbol: str
    series: str = "EQ"
    observation_date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    last_price: float | None = None
    prev_close: float | None = None
    vwap: float | None = None
    adj_close: float | None = None
    volume: float | None = None
    turnover_cr: float | None = None
    trades_count: int | None = None
    deliverable_qty: float | None = None
    delivery_pct: float | None = None
    dividends: float = 0.0
    stock_splits: float = 0.0
    source: str
    publication_timestamp: str
    collection_timestamp: str
    ohlcv_validation_status: ValidationStatus = "VERIFIED"
    delivery_validation_status: ValidationStatus = "MISSING"
    data_version: str = "1.0.0"


class IndexDailyBar(BaseModel):
    """Normalized daily market or sector index bar."""

    index_code: str
    index_name: str
    category: Literal["broad", "volatility", "sector"] = "broad"
    observation_date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    change_pct: float | None = None
    volume: float | None = None
    turnover_cr: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    div_yield: float | None = None
    source: str
    publication_timestamp: str
    collection_timestamp: str
    validation_status: ValidationStatus = "VERIFIED"
    data_version: str = "1.0.0"


class MarketRegimeSnapshot(BaseModel):
    """Market regime classification & breadth snapshot."""

    observation_date: str
    regime: MarketRegimeLabel
    nifty_50_close: float | None = None
    nifty_50_day_change_pct: float | None = None
    nifty_50_return_20d_pct: float | None = None
    nifty_50_dma_50: float | None = None
    nifty_50_dma_200: float | None = None
    india_vix_close: float | None = None
    india_vix_change_pct: float | None = None
    advance_count: int = 0
    decline_count: int = 0
    unchanged_count: int = 0
    advance_decline_ratio: float | None = None
    pct_above_50dma: float | None = None
    pct_above_200dma: float | None = None
    allow_aggressive_longs: bool = True
    regime_rationale: list[str] = Field(default_factory=list)
    sector_performances: list[dict[str, Any]] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    """Per-stock data quality score & trade eligibility gate."""

    symbol: str
    evaluated_at: str
    quality_score: float
    trade_eligible: bool
    forced_action: Literal["NO TRADE"] | None = None
    verified_metrics_count: int = 0
    estimated_metrics_count: int = 0
    missing_metrics_count: int = 0
    stale_metrics_count: int = 0
    conflicting_metrics_count: int = 0
    critical_failures: list[str] = Field(default_factory=list)
    metric_statuses: dict[str, ValidationStatus] = Field(default_factory=dict)


class StockResearchBundle(BaseModel):
    fetched_at: str
    overview: CompanyOverview
    last_5_annual_reports: list[ReportPeriod]
    last_5_quarterly_reports: list[ReportPeriod]
    cash_and_balance_sheet: CashAndBalanceSheet
    stockholders: StockholderAnalysis
    rd_and_innovation: RDAndInnovation
    technicals: TechnicalSnapshot
    agent_committee: list[AgentVerdict]
    next_day_guidance: NextDayGuidance
    promoter_and_competitor_intel: PromoterAndCompetitorIntel = Field(
        default_factory=PromoterAndCompetitorIntel
    )
    news: list[NewsHeadline] = Field(default_factory=list)
    data_quality: DataQualityReport | None = None
    market_regime: MarketRegimeSnapshot | None = None

