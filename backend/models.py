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


class FeatureAttributionItem(BaseModel):
    """Per-prediction numerical feature contribution (SHAP / linear attribution)."""

    feature: str
    family: str
    value: float | None = None
    contribution: float
    direction: Literal["BULLISH", "BEARISH"]


class QuantitativePredictionRecord(BaseModel):
    """Calibrated multi-horizon quantitative prediction record (Phase 3 — Section 8)."""

    prediction_id: str
    symbol: str
    observation_date: str
    generated_at: str
    model_name: str = "ml_ensemble_v1"
    model_version: str = "3.0.0"
    market_regime: str = "Sideways"
    prob_next_day_up: float
    prob_lgbm_up: float | None = None
    prob_xgb_up: float | None = None
    prob_linear_up: float | None = None
    expected_next_day_return_pct: float
    expected_5d_return_pct: float
    downside_risk_mae_pct: float
    expected_upside_mfe_pct: float
    reward_to_risk_ratio: float
    model_agreement_score: float
    confidence_score: float
    roundtrip_delivery_cost_pct: float
    roundtrip_intraday_cost_pct: float
    net_expected_1d_return_pct: float
    net_expected_5d_return_pct: float
    recommended_action: Literal[
        "STRONG BUY",
        "BUY / ACCUMULATE",
        "HOLD / WATCH",
        "REDUCE / SELL",
        "AVOID / SHORT",
        "NO TRADE",
    ]
    abstention_reason: str | None = None
    top_bullish_drivers: list[FeatureAttributionItem] = Field(default_factory=list)
    top_bearish_drivers: list[FeatureAttributionItem] = Field(default_factory=list)
    target_session_date: str | None = None
    last_close: float | None = None
    atr_14_pct: float | None = None
    momentum_prob_up: float | None = None
    momentum_5d_return_pct: float | None = None
    feature_snapshot: dict[str, float | None] = Field(default_factory=dict)
    outcome_evaluated_at: str | None = None
    realized_next_day_return_pct: float | None = None
    realized_5d_return_pct: float | None = None


class ModelEvaluationMetrics(BaseModel):
    """Out-of-sample walk-forward evaluation metrics for a single model or benchmark."""

    model_name: str
    total_predictions: int
    active_trades_count: int
    abstention_rate_pct: float
    directional_accuracy_all_pct: float
    high_conviction_hit_rate_pct: float
    roc_auc: float
    brier_score: float
    log_loss_val: float
    rank_ic_1d: float
    rank_ic_5d: float
    gross_cumulative_return_pct: float
    transaction_cost_drag_pct: float
    net_cumulative_return_pct: float
    annualized_sharpe_gross: float
    annualized_sharpe_net: float
    sortino_ratio_net: float
    max_drawdown_net_pct: float
    profit_factor_net: float
    avg_win_loss_ratio: float
    expectancy_per_trade_bps: float
    calibration_ece: float | None = None


class BacktestComparisonSummary(BaseModel):
    """Full walk-forward backtest report comparing ML ensemble against baselines & Nifty 50."""

    generated_at: str
    train_start_date: str
    test_start_date: str
    test_end_date: str
    total_walk_forward_folds: int
    purge_gap_days: int
    embargo_sessions: int = 1
    execution_mode: str = "next_open_delivery_capped"
    symbols_evaluated: list[str]
    nifty_50_buy_hold_return_pct: float
    equal_weight_universe_return_pct: float
    models: dict[str, ModelEvaluationMetrics]
    calibration_curve: list[dict[str, float]] = Field(default_factory=list)
    by_regime_performance: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_sector_performance: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_symbol_performance: dict[str, dict[str, float]] = Field(default_factory=dict)
    top_global_features: list[dict[str, Any]] = Field(default_factory=list)
    execution_diagnostics: dict[str, Any] = Field(default_factory=dict)
    feature_stability_jaccard: float | None = None


class BulkBlockDealRecord(BaseModel):
    """Normalized NSE Bulk or Block deal disclosure record (Phase 4 — Section 5C)."""

    deal_type: Literal["BULK", "BLOCK"]
    deal_date: str
    symbol: str
    security_name: str
    client_name: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price_inr: float
    value_cr: float
    source: str = "NSE_Official_Archives"


class CorporateActionRecord(BaseModel):
    """Corporate action record (Dividend, Stock Split, Bonus Issue)."""

    symbol: str
    action_type: Literal["DIVIDEND", "SPLIT", "BONUS"]
    ex_date: str
    amount_or_ratio: float
    description: str
    source: str = "Exchange_Corporate_Actions"


class CorporateIntelligenceSummary(BaseModel):
    """Per-stock Institutional, Promoter, Bulk/Block Deal, Corporate Action & F&O Positioning summary (Phase 4)."""

    symbol: str
    evaluated_at: str
    promoter_holding_pct: float | None = None
    promoter_chg_1q: float | None = None
    promoter_chg_4q: float | None = None
    promoter_pledge_pct: float | None = None
    fii_holding_pct: float | None = None
    fii_chg_1q: float | None = None
    fii_chg_4q: float | None = None
    dii_holding_pct: float | None = None
    dii_chg_1q: float | None = None
    dii_chg_4q: float | None = None
    net_institutional_flow_4q: float | None = None
    is_fno_eligible: bool = False
    latest_delivery_pct: float | None = None
    delivery_vs_20d_avg: float | None = None
    fno_positioning_heuristic: str = "Cash / Neutral"
    net_bulk_block_value_cr: float = 0.0
    bulk_block_deals_count: int = 0
    institutional_conviction_score: float = 50.0
    smart_money_verdict: Literal["STRONG ACCUMULATION", "MODERATE ACCUMULATION", "NEUTRAL", "DISTRIBUTION"] = "NEUTRAL"
    bulk_block_deals: list[BulkBlockDealRecord] = Field(default_factory=list)
    recent_corporate_actions: list[CorporateActionRecord] = Field(default_factory=list)
    key_signals: list[str] = Field(default_factory=list)


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
    quantitative_prediction: QuantitativePredictionRecord | None = None
    corporate_intelligence: CorporateIntelligenceSummary | None = None


class StructuredNewsEvent(BaseModel):
    """Validated structured news/event record (Phase 5 — Section 3)."""

    event_id: str
    company_name: str
    ticker: str
    related_competitor_or_industry: str | None = None
    event_category: Literal[
        "company_news",
        "promoter_management",
        "industry_sector",
        "competitor",
        "regulatory_government",
        "global_macro",
    ]
    event_description: str
    original_source: str
    source_url: str = ""
    publication_timestamp: str
    event_timestamp: str | None = None
    effective_trading_date: str | None = None
    sentiment: Literal["positive", "negative", "neutral"]
    impact_magnitude: Literal["low", "medium", "high"]
    classification_confidence: float = Field(ge=0.0, le=1.0)
    expected_impact_duration: str = "1-5 trading sessions"
    relevant_horizon: Literal["1D", "5D", "10D", "LONGER"] = "5D"
    novelty_score: float = Field(ge=0.0, le=1.0)
    source_credibility: float = Field(ge=0.0, le=1.0)
    related_companies: list[str] = Field(default_factory=list)
    related_sectors: list[str] = Field(default_factory=list)
    is_duplicate_report: bool = False
    canonical_event_id: str | None = None
    extraction_method: Literal["deterministic", "gemini"] = "deterministic"
    facts_summary: str = ""
    interpretation_summary: str = ""


class CrossStockImpactItem(BaseModel):
    """Cross-stock & macro relationship impact record (Phase 5 — Section 5)."""

    affected_company: str
    originating_event_id: str
    originating_entity: str
    relationship_type: Literal[
        "competitor",
        "supplier_customer",
        "sector_peer",
        "commodity_dependency",
        "regulatory_exposure",
        "global_macro",
    ]
    expected_impact_direction: Literal["positive", "negative", "neutral"]
    impact_magnitude: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    expected_duration: str = "1-5 trading sessions"
    rationale: str = ""


class MultiHorizonSignalSummary(BaseModel):
    """Separated & combined prediction signal per horizon (1D, 5D, 10D) (Phase 5 — Section 6)."""

    horizon: Literal["1D", "5D", "10D"]
    prob_positive: float
    expected_gross_return_pct: float
    expected_net_return_pct: float
    quant_signal: str
    news_signal: str
    cross_stock_signal: str
    combined_signal: str
    confidence: float
    uncertainty: float
    decision: Literal["BUY", "HOLD", "WATCH", "NO TRADE"]


class StockIntelligenceItem(BaseModel):
    """Consolidated Phase 5 multi-dimensional intelligence view for a single pilot stock."""

    symbol: str
    company_name: str
    sector: str
    observation_date: str
    quant_score_summary: dict[str, Any] = Field(default_factory=dict)
    news_sentiment_score: float = 0.0
    news_features: dict[str, float] = Field(default_factory=dict)
    promoter_intelligence: list[str] = Field(default_factory=list)
    industry_outlook: str = "Neutral"
    competitor_developments: list[str] = Field(default_factory=list)
    regulatory_developments: list[str] = Field(default_factory=list)
    global_exposure: list[str] = Field(default_factory=list)
    events_timeline: list[StructuredNewsEvent] = Field(default_factory=list)
    cross_stock_impacts: list[CrossStockImpactItem] = Field(default_factory=list)
    horizon_outlooks: dict[str, MultiHorizonSignalSummary] = Field(default_factory=dict)
    facts_vs_interpretations_note: str = (
        "Facts (prices, exchange filings, raw headlines) are separated from extracted interpretations and model predictions."
    )


class MonitoringSnapshot(BaseModel):
    """Accuracy, drift & data quality monitoring snapshot (Phase 5 — Section 11)."""

    evaluated_at: str
    total_logged_predictions: int = 0
    evaluated_outcomes_count: int = 0
    rolling_directional_accuracy_pct: float | None = None
    brier_score_recent: float | None = None
    drift_status: Literal["OK", "WARNING", "INSUFFICIENT_DATA"] = "INSUFFICIENT_DATA"
    data_quality_warnings: list[str] = Field(default_factory=list)
    failure_alerts: list[str] = Field(default_factory=list)


class Phase5IntelligencePayload(BaseModel):
    """Top-level payload for the Phase 5 Stock Intelligence dashboard & API."""

    generated_at: str
    model_version: str = "5.0.0-pilot"
    last_news_collection_at: str
    last_quant_update_at: str
    latest_prediction_at: str
    pilot_universe: list[str] = Field(default_factory=list)
    stocks: dict[str, StockIntelligenceItem] = Field(default_factory=dict)
    consolidated_feed: list[StructuredNewsEvent] = Field(default_factory=list)
    monitoring: MonitoringSnapshot
    data_quality_warnings: list[str] = Field(default_factory=list)


