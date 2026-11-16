"""Multi-Agent Indian Stock Market Research & Next-Day Decision Engine.

Inspired by `virattt/ai-hedge-fund` (persona/specialist committee agents) and
`aswathas/pistol` (quantitative Indian market scoring), combining:
1. 5-Report Performance Agent (Last 5 Annual + Last 5 Quarterly Reports)
2. Cash-in-Hand & Balance Sheet Fortress Agent (Cash, Investments, Debt, OCF/FCF)
3. Stockholders & Institutional Flow Agent (Promoters, FIIs, DIIs, Named Holders)
4. R&D & Innovation Moat Agent (5Y R&D Spend, R&D % of Sales, Intangibles, CWIP)
5. Next-Day Technical & Catalyst Trigger Agent (RSI, MACD, DMA50/200, ATR, News)
6. Pluggable LLM CIO Synthesizer (Gemini / OpenAI / Anthropic / Ollama + Quant fallback)
"""

from __future__ import annotations

import json
from typing import Any

import requests

from .data_engine import fetch_raw_company_data
from .models import (
    AgentVerdict,
    CashAndBalanceSheet,
    CompanyOverview,
    NewsHeadline,
    NextDayGuidance,
    RDAndInnovation,
    ReportPeriod,
    StockholderAnalysis,
    StockResearchBundle,
    TechnicalSnapshot,
)
from .technical_engine import build_technical_snapshot


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return round(max(lo, min(hi, val)), 1)


def _signal_from_score(score: float) -> str:
    if score >= 62.0:
        return "BULLISH"
    if score <= 42.0:
        return "BEARISH"
    return "NEUTRAL"


def evaluate_5_reports_agent(
    overview: CompanyOverview,
    annual_5: list[ReportPeriod],
    quarterly_5: list[ReportPeriod],
) -> AgentVerdict:
    """Agent 1: Evaluates the last 5 Annual Reports and last 5 Quarterly Reports."""
    score = 50.0
    reasoning: list[str] = []

    latest_a = annual_5[-1] if annual_5 else None
    latest_q = quarterly_5[-1] if quarterly_5 else None

    # 1. 5-Year Annual Revenue & Profit CAGR
    rev_cagr = overview.revenue_cagr_5y_pct
    prof_cagr = overview.profit_cagr_5y_pct
    if rev_cagr is not None:
        if rev_cagr >= 12.0:
            score += 12.0
            reasoning.append(f"Strong 5-report annual revenue compounding at {rev_cagr:.1f}% CAGR.")
        elif rev_cagr >= 6.0:
            score += 6.0
            reasoning.append(f"Steady 5-report annual revenue growth at {rev_cagr:.1f}% CAGR.")
        elif rev_cagr < 0:
            score -= 10.0
            reasoning.append(f"Top-line contraction across last 5 annual reports ({rev_cagr:.1f}% CAGR).")

    if prof_cagr is not None:
        if prof_cagr >= 15.0:
            score += 14.0
            reasoning.append(f"Superior bottom-line expansion: Net Profit grew at {prof_cagr:.1f}% 5Y CAGR.")
        elif prof_cagr >= 7.0:
            score += 7.0
            reasoning.append(f"Consistent 5-year net profit growth ({prof_cagr:.1f}% CAGR).")
        elif prof_cagr < 0:
            score -= 12.0
            reasoning.append(f"5-year earnings drag: Net profit CAGR is negative ({prof_cagr:.1f}%).")

    # 2. Quarterly trajectory (last 5 quarterly reports)
    if len(quarterly_5) >= 2:
        q_pos_rev = sum(1 for q in quarterly_5 if (q.revenue_growth_pct or 0) > 0)
        q_pos_np = sum(1 for q in quarterly_5 if (q.net_profit_growth_pct or 0) > 0)
        if latest_q and latest_q.net_profit_growth_pct is not None:
            if latest_q.net_profit_growth_pct >= 10.0:
                score += 10.0
                reasoning.append(
                    f"Latest quarter ({latest_q.period}) net profit jumped +{latest_q.net_profit_growth_pct:.1f}% sequentially."
                )
            elif latest_q.net_profit_growth_pct < -8.0:
                score -= 8.0
                reasoning.append(
                    f"Latest quarter ({latest_q.period}) net profit contracted {latest_q.net_profit_growth_pct:.1f}%."
                )
        if q_pos_np >= 3:
            score += 6.0
            reasoning.append(f"Positive sequential profit momentum in {q_pos_np} of the last {len(quarterly_5)} quarters.")

    # 3. Profitability & Return Ratios (ROE / ROCE / OPM)
    roe = overview.roe_pct or 0.0
    roce = overview.roce_pct or 0.0
    if roe >= 18.0 or roce >= 20.0:
        score += 10.0
        reasoning.append(f"High capital efficiency: ROE {roe:.1f}% and ROCE {roce:.1f}% exceed institutional hurdle rates.")
    elif roe < 8.0 and roce < 10.0:
        score -= 8.0
        reasoning.append(f"Sub-par capital returns (ROE {roe:.1f}%, ROCE {roce:.1f}%).")

    score = _clamp(score)
    sig = _signal_from_score(score)

    return AgentVerdict(
        agent_id="five_reports_agent",
        agent_name="5-Report Earnings & Performance Agent",
        persona_inspiration="Peter Lynch / Fundamental Growth Analyst",
        signal=sig,  # type: ignore[arg-type]
        score=score,
        confidence=min(95, int(65 + abs(score - 50) * 0.7)),
        weight_pct=25,
        headline=(
            f"5Y Profit CAGR {prof_cagr if prof_cagr is not None else 'N/A'}% | "
            f"Latest FY Net Profit ₹{int(latest_a.net_profit_cr or 0):,} Cr | ROE {roe:.1f}%"
            if latest_a
            else "Analyzed last 5 annual & quarterly financial statements"
        ),
        key_metrics={
            "5Y Revenue CAGR": f"{rev_cagr:.1f}%" if rev_cagr is not None else "N/A",
            "5Y Profit CAGR": f"{prof_cagr:.1f}%" if prof_cagr is not None else "N/A",
            "Latest Qtr Net Profit": f"₹{latest_q.net_profit_cr:,.0f} Cr" if (latest_q and latest_q.net_profit_cr) else "N/A",
            "Latest OPM %": f"{latest_a.opm_pct:.1f}%" if (latest_a and latest_a.opm_pct is not None) else "N/A",
            "ROE / ROCE": f"{roe:.1f}% / {roce:.1f}%",
            "Piotroski F-Score": f"{overview.piotroski_score}/9",
        },
        reasoning=reasoning or ["Stable 5-year revenue and operating profit progression."],
    )


def evaluate_cash_balance_agent(
    overview: CompanyOverview,
    cash_bs: CashAndBalanceSheet,
) -> AgentVerdict:
    """Agent 2: Evaluates Cash in Hand, Liquid Investments, Debt, and 5Y Cash Flows."""
    score = 50.0
    reasoning: list[str] = []

    is_financial = any(
        w in (overview.sector + " " + overview.industry).lower()
        for w in ("bank", "finance", "nbfc")
    )

    total_cash = cash_bs.total_cash_in_hand_cr
    borrowings = cash_bs.total_borrowings_cr
    net_cash = cash_bs.net_cash_position_cr
    d_to_e = cash_bs.debt_to_equity

    if is_financial:
        # For banks/NBFCs, deposits/borrowings are raw material; judge on ROE/ROCE and liquid cash reserves
        if total_cash > 10000:
            score += 14.0
            reasoning.append(
                f"Strong liquidity buffer: ₹{total_cash:,.0f} Cr in cash & investment book ({overview.sector})."
            )
        if (overview.roe_pct or 0) >= 14.0:
            score += 12.0
            reasoning.append(f"Well-capitalized balance sheet generating {overview.roe_pct:.1f}% ROE.")
    else:
        if net_cash > 0:
            score += 20.0
            reasoning.append(
                f"Net Cash Fortress: Total Cash & Investments (₹{total_cash:,.0f} Cr) exceed Total Debt (₹{borrowings:,.0f} Cr) by ₹{net_cash:,.0f} Cr."
            )
        elif d_to_e <= 0.35:
            score += 12.0
            reasoning.append(
                f"Conservative leverage (Debt/Equity {d_to_e:.2f}x) backed by ₹{total_cash:,.0f} Cr cash & liquid investments in hand."
            )
        elif d_to_e <= 0.85:
            score += 3.0
            reasoning.append(
                f"Manageable Debt/Equity of {d_to_e:.2f}x with ₹{total_cash:,.0f} Cr cash in hand."
            )
        else:
            score -= 14.0
            reasoning.append(
                f"Elevated leverage: Debt/Equity at {d_to_e:.2f}x (Total Borrowings ₹{borrowings:,.0f} Cr vs Cash ₹{total_cash:,.0f} Cr)."
            )

        if cash_bs.cumulative_5y_ocf_cr > 0:
            score += 10.0
            reasoning.append(
                f"Generated ₹{cash_bs.cumulative_5y_ocf_cr:,.0f} Cr in cumulative 5-year Operating Cash Flow (latest FY OCF ₹{cash_bs.latest_ocf_cr:,.0f} Cr)."
            )
        elif cash_bs.latest_ocf_cr < 0:
            score -= 10.0
            reasoning.append(f"Negative Operating Cash Flow in latest fiscal year (₹{cash_bs.latest_ocf_cr:,.0f} Cr).")

        if (cash_bs.cash_pct_of_mcap or 0) >= 8.0:
            score += 6.0
            reasoning.append(
                f"Cash & investments represent {cash_bs.cash_pct_of_mcap:.1f}% of total Market Cap, offering downside protection."
            )

    score = _clamp(score)
    sig = _signal_from_score(score)

    return AgentVerdict(
        agent_id="cash_balance_agent",
        agent_name="Cash-in-Hand & Balance Sheet Agent",
        persona_inspiration="Benjamin Graham / Balance Sheet Specialist",
        signal=sig,  # type: ignore[arg-type]
        score=score,
        confidence=min(95, int(68 + abs(score - 50) * 0.65)),
        weight_pct=20,
        headline=f"Cash & Investments: ₹{total_cash:,.0f} Cr | Net Cash: ₹{net_cash:,.0f} Cr | D/E: {d_to_e:.2f}x",
        key_metrics={
            "Cash Equivalents": f"₹{cash_bs.cash_equivalents_cr:,.0f} Cr",
            "Liquid Investments": f"₹{cash_bs.liquid_investments_cr:,.0f} Cr",
            "Total Cash in Hand": f"₹{total_cash:,.0f} Cr",
            "Total Borrowings": f"₹{borrowings:,.0f} Cr",
            "Debt / Equity": f"{d_to_e:.2f}x",
            "5Y Cumulative OCF": f"₹{cash_bs.cumulative_5y_ocf_cr:,.0f} Cr",
        },
        reasoning=reasoning or [f"Cash in hand stands at ₹{total_cash:,.0f} Cr against borrowings of ₹{borrowings:,.0f} Cr."],
    )


def evaluate_stockholders_agent(
    shp: StockholderAnalysis,
) -> AgentVerdict:
    """Agent 3: Evaluates Promoters, FIIs, DIIs, Public float, and Named Stockholders."""
    score = 50.0
    reasoning: list[str] = []

    prom = shp.promoters_pct
    fii = shp.fii_pct
    dii = shp.dii_pct
    inst = shp.institutional_total_pct

    if prom >= 45.0:
        score += 10.0
        reasoning.append(f"High skin-in-the-game: Promoters hold {prom:.2f}% equity.")
    elif prom == 0.0 and inst >= 55.0:
        score += 10.0
        reasoning.append(
            f"Professionally managed institutional blue-chip: FIIs ({fii:.2f}%) + DIIs ({dii:.2f}%) hold {inst:.2f}%."
        )
    elif 0 < prom < 25.0 and inst < 35.0:
        score -= 8.0
        reasoning.append(f"Low promoter stake ({prom:.2f}%) with limited institutional sponsorship.")

    if shp.promoter_change_4q_pct > 0.05:
        score += 12.0
        reasoning.append(
            f"Bullish insider signal: Promoters increased stake by +{shp.promoter_change_4q_pct:.2f}% over the last 4 quarters."
        )
    elif shp.promoter_change_4q_pct < -1.0:
        score -= 10.0
        reasoning.append(
            f"Promoter stake reduced by {shp.promoter_change_4q_pct:.2f}% over the last 4 quarters."
        )

    inst_delta = round(shp.fii_change_4q_pct + shp.dii_change_4q_pct, 2)
    if inst_delta > 0.3:
        score += 10.0
        reasoning.append(
            f"Net Institutional Accumulation: Combined FII+DII holding rose +{inst_delta:.2f}% over 4 quarters (FII {shp.fii_change_4q_pct:+.2f}%, DII {shp.dii_change_4q_pct:+.2f}%)."
        )
    elif inst_delta < -1.0:
        score -= 8.0
        reasoning.append(
            f"Institutional trimming: Combined FII+DII stake fell {inst_delta:.2f}% over 4 quarters."
        )

    if shp.named_stockholders:
        top_names = [
            f"{h.name} ({h.latest_holding_pct:.2f}%)"
            for h in shp.named_stockholders[:4]
        ]
        reasoning.append(f"Marquee named stockholders: {', '.join(top_names)}.")

    score = _clamp(score)
    sig = _signal_from_score(score)

    return AgentVerdict(
        agent_id="stockholders_agent",
        agent_name="Stockholders & Institutional Flow Agent",
        persona_inspiration="Smart-Money & Insider Ownership Tracker",
        signal=sig,  # type: ignore[arg-type]
        score=score,
        confidence=min(94, int(66 + abs(score - 50) * 0.65)),
        weight_pct=20,
        headline=f"Promoters {prom:.1f}% ({shp.promoter_change_4q_pct:+.2f}% 4Q) | FII {fii:.1f}% | DII {dii:.1f}%",
        key_metrics={
            "Promoter Holding": f"{prom:.2f}% ({shp.promoter_change_4q_pct:+.2f}% 4Q)",
            "FII Holding": f"{fii:.2f}% ({shp.fii_change_4q_pct:+.2f}% 4Q)",
            "DII Holding": f"{dii:.2f}% ({shp.dii_change_4q_pct:+.2f}% 4Q)",
            "Total Institutional": f"{inst:.2f}%",
            "Public Float": f"{shp.public_pct:.2f}%",
            "Shareholder Accounts": f"{shp.num_shareholders:,}" if shp.num_shareholders else "N/A",
        },
        reasoning=reasoning,
    )


def evaluate_rd_innovation_agent(
    overview: CompanyOverview,
    rd: RDAndInnovation,
) -> AgentVerdict:
    """Agent 4: Evaluates R&D Spend, R&D % of Sales, Intangible Assets, CWIP & Innovation Moat."""
    score = 50.0
    reasoning: list[str] = []

    if rd.latest_rd_spend_cr > 0:
        score += 8.0
        reasoning.append(
            f"Latest annual R&D investment of ₹{rd.latest_rd_spend_cr:,.0f} Cr ({rd.latest_rd_pct_of_sales:.2f}% of revenue); 5-year cumulative R&D of ₹{rd.cumulative_5y_rd_cr:,.0f} Cr."
        )

    if rd.latest_rd_pct_of_sales >= 4.0:
        score += 16.0
        reasoning.append(
            f"High-intensity R&D leader ({rd.latest_rd_pct_of_sales:.2f}% of turnover), building deep IP and product pipeline moat."
        )
    elif rd.latest_rd_pct_of_sales >= 1.0:
        score += 10.0
        reasoning.append(
            f"Consistent R&D allocation ({rd.latest_rd_pct_of_sales:.2f}% of turnover) supporting digital/product differentiation."
        )

    if rd.rd_cagr_5y_pct is not None and rd.rd_cagr_5y_pct > 8.0:
        score += 8.0
        reasoning.append(
            f"R&D budget compounding at {rd.rd_cagr_5y_pct:.1f}% CAGR over the last 5 reports."
        )

    if rd.intangible_assets_cr > 0 or rd.cwip_cr > 0:
        score += 6.0
        reasoning.append(
            f"Balance-sheet innovation & capacity pipeline: Intangible Assets ₹{rd.intangible_assets_cr:,.0f} Cr and Capital Work-in-Progress (CWIP) ₹{rd.cwip_cr:,.0f} Cr."
        )

    reasoning.append(f"Strategic R&D Focus: {rd.innovation_focus}.")

    score = _clamp(score)
    sig = _signal_from_score(score)

    return AgentVerdict(
        agent_id="rd_innovation_agent",
        agent_name="R&D Intensity & Innovation Moat Agent",
        persona_inspiration="Charlie Munger / Philip Fisher Moat Analyst",
        signal=sig,  # type: ignore[arg-type]
        score=score,
        confidence=min(92, int(64 + abs(score - 50) * 0.6)),
        weight_pct=15,
        headline=f"Latest R&D: ₹{rd.latest_rd_spend_cr:,.0f} Cr ({rd.latest_rd_pct_of_sales:.2f}% of Sales) | 5Y R&D: ₹{rd.cumulative_5y_rd_cr:,.0f} Cr",
        key_metrics={
            "Latest FY R&D Spend": f"₹{rd.latest_rd_spend_cr:,.0f} Cr",
            "R&D % of Revenue": f"{rd.latest_rd_pct_of_sales:.2f}%",
            "5Y Cumulative R&D": f"₹{rd.cumulative_5y_rd_cr:,.0f} Cr",
            "5Y R&D CAGR": f"{rd.rd_cagr_5y_pct:.1f}%" if rd.rd_cagr_5y_pct is not None else "N/A",
            "Intangible Assets": f"₹{rd.intangible_assets_cr:,.0f} Cr",
            "CapEx WIP (CWIP)": f"₹{rd.cwip_cr:,.0f} Cr",
        },
        reasoning=reasoning,
    )


def evaluate_technical_agent(
    tech: TechnicalSnapshot,
    news: list[NewsHeadline],
) -> AgentVerdict:
    """Agent 5: Evaluates Next-Day Technical Setup, Volume Breakout & News Catalysts."""
    score = 50.0
    reasoning: list[str] = list(tech.technical_signals)

    p = tech.current_price
    if p > tech.dma_50 and tech.dma_50 > tech.dma_200:
        score += 15.0
    elif p > tech.dma_200 and p < tech.dma_50:
        score += 4.0
    elif p > tech.dma_50 and p < tech.dma_200:
        score += 6.0
    else:
        score -= 14.0

    # RSI scoring for next-day setup
    if 50.0 <= tech.rsi_14 <= 67.0:
        score += 10.0
    elif tech.rsi_14 < 32.0:
        # Oversold mean-reversion bounce setup
        score += 8.0
    elif tech.rsi_14 > 73.0:
        # Overbought short-term pullback risk
        score -= 8.0
    elif 35.0 <= tech.rsi_14 < 45.0:
        score -= 5.0

    # MACD histogram
    if tech.macd_hist > 0 and tech.macd_line > tech.macd_signal:
        score += 8.0
    else:
        score -= 6.0

    # Volume confirmation
    if tech.volume_Surge_ratio >= 1.25 and tech.day_change_pct > 0:
        score += 7.0
    elif tech.volume_Surge_ratio >= 1.25 and tech.day_change_pct < 0:
        score -= 6.0

    # News sentiment boost
    pos_news = sum(1 for n in news if n.sentiment == "POSITIVE")
    neg_news = sum(1 for n in news if n.sentiment == "NEGATIVE")
    if pos_news > neg_news:
        score += 5.0
        reasoning.append(f"News Catalyst: {pos_news} positive vs {neg_news} negative recent Indian market headlines.")
    elif neg_news > pos_news:
        score -= 5.0
        reasoning.append(f"News Caution: {neg_news} negative recent headlines detected.")

    score = _clamp(score)
    sig = _signal_from_score(score)

    return AgentVerdict(
        agent_id="technical_catalyst_agent",
        agent_name="Next-Day Technical & Catalyst Agent",
        persona_inspiration="Stanley Druckenmiller / Price-Action & Timing Desk",
        signal=sig,  # type: ignore[arg-type]
        score=score,
        confidence=min(94, int(65 + abs(score - 50) * 0.7)),
        weight_pct=20,
        headline=f"RSI(14) {tech.rsi_14:.1f} | 50-DMA ₹{tech.dma_50:,.1f} | 200-DMA ₹{tech.dma_200:,.1f} | ATR ₹{tech.atr_14:,.1f}",
        key_metrics={
            "RSI (14)": f"{tech.rsi_14:.1f}",
            "MACD Histogram": f"{tech.macd_hist:+.2f}",
            "50-DMA / 200-DMA": f"₹{tech.dma_50:,.1f} / ₹{tech.dma_200:,.1f}",
            "ATR (14) Daily Range": f"₹{tech.atr_14:,.2f} ({tech.atr_pct:.2f}%)",
            "Volume vs 20D Avg": f"{tech.volume_Surge_ratio:.2f}x",
            "Pivot S1 / R1": f"₹{tech.support_1:,.1f} / ₹{tech.resistance_1:,.1f}",
        },
        reasoning=reasoning,
    )


def _call_pluggable_llm(
    llm_config: dict[str, str],
    prompt: str,
) -> dict[str, Any] | None:
    """Call user-configured LLM (Gemini / OpenAI / Anthropic / Ollama) if provided."""
    provider = (llm_config.get("provider") or "").lower().strip()
    api_key = (llm_config.get("api_key") or "").strip()
    model = (llm_config.get("model") or "").strip()

    if not provider or provider == "quant":
        return None

    sys_instruction = (
        "You are the Chief Investment Officer for an Indian Stock Market (NSE/BSE) "
        "AI Research Fund. Respond ONLY with valid JSON containing keys: "
        '{"executive_summary": str, "key_catalysts": [str, str, str], "key_risks": [str, str]}'
    )

    try:
        if provider == "gemini" and api_key:
            m_name = model or "gemini-2.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": sys_instruction + "\n\n" + prompt}]}],
                "generationConfig": {"ResponseMimeType": "application/json", "temperature": 0.2},
            }
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(txt)

        elif provider == "openai" and api_key:
            m_name = model or "gpt-4o-mini"
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": m_name,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": sys_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=15,
            )
            if r.status_code == 200:
                txt = r.json()["choices"][0]["message"]["content"]
                return json.loads(txt)

        elif provider == "anthropic" and api_key:
            m_name = model or "claude-3-5-sonnet-latest"
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": m_name,
                    "max_tokens": 600,
                    "system": sys_instruction,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            if r.status_code == 200:
                txt = r.json()["content"][0]["text"]
                start = txt.find("{")
                end = txt.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(txt[start : end + 1])

        elif provider == "ollama":
            base_url = llm_config.get("base_url") or "http://localhost:11434"
            m_name = model or "llama3.1"
            r = requests.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={
                    "model": m_name,
                    "prompt": sys_instruction + "\n\n" + prompt,
                    "format": "json",
                    "stream": False,
                },
                timeout=15,
            )
            if r.status_code == 200:
                return json.loads(r.json().get("response", "{}"))
    except Exception:
        return None

    return None


def synthesize_next_day_guidance(
    overview: CompanyOverview,
    cash_bs: CashAndBalanceSheet,
    shp: StockholderAnalysis,
    rd: RDAndInnovation,
    tech: TechnicalSnapshot,
    committee: list[AgentVerdict],
    llm_config: dict[str, str] | None = None,
) -> NextDayGuidance:
    """Combine fundamental & technical agent verdicts into actionable Next-Day NSE guidance."""
    total_weight = sum(a.weight_pct for a in committee) or 100
    weighted_score = round(
        sum(a.score * a.weight_pct for a in committee) / total_weight, 1
    )

    fund_agents = [a for a in committee if a.agent_id != "technical_catalyst_agent"]
    fund_weight = sum(a.weight_pct for a in fund_agents) or 80
    fund_score = round(
        sum(a.score * a.weight_pct for a in fund_agents) / fund_weight, 1
    )
    tech_agent = next(
        (a for a in committee if a.agent_id == "technical_catalyst_agent"), committee[-1]
    )
    tech_score = round(tech_agent.score, 1)

    p = tech.current_price
    atr = tech.atr_14 if tech.atr_14 > 0 else round(p * 0.018, 2)

    if weighted_score >= 72.0 and tech_score >= 55.0:
        action = "STRONG BUY"
        entry_low = round(p - 0.25 * atr, 2)
        entry_high = round(p + 0.15 * atr, 2)
        t1 = round(max(tech.resistance_1, p + 1.35 * atr), 2)
        t2 = round(max(tech.resistance_2, p + 2.60 * atr), 2)
        sl = round(min(tech.support_1, p - 0.90 * atr), 2)
        pos_size = 8.0
    elif weighted_score >= 58.0:
        action = "BUY"
        entry_low = round(p - 0.40 * atr, 2)
        entry_high = round(p + 0.10 * atr, 2)
        t1 = round(max(tech.resistance_1, p + 1.20 * atr), 2)
        t2 = round(max(tech.resistance_2, p + 2.25 * atr), 2)
        sl = round(min(tech.support_1, p - 0.95 * atr), 2)
        pos_size = 5.0
    elif weighted_score >= 46.0:
        action = "HOLD / ACCUMULATE"
        entry_low = round(min(tech.support_1, p - 0.75 * atr), 2)
        entry_high = round(p - 0.15 * atr, 2)
        t1 = round(p + 1.10 * atr, 2)
        t2 = round(p + 2.00 * atr, 2)
        sl = round(p - 1.15 * atr, 2)
        pos_size = 3.0
    elif weighted_score >= 35.0:
        action = "SELL / BOOK PROFIT"
        entry_low = round(p, 2)
        entry_high = round(p + 0.35 * atr, 2)
        t1 = round(min(tech.support_1, p - 1.15 * atr), 2)
        t2 = round(min(tech.support_2, p - 2.10 * atr), 2)
        sl = round(max(tech.resistance_1, p + 0.95 * atr), 2)
        pos_size = 0.0
    else:
        action = "AVOID / SHORT"
        entry_low = round(p - 0.10 * atr, 2)
        entry_high = round(p + 0.25 * atr, 2)
        t1 = round(p - 1.30 * atr, 2)
        t2 = round(p - 2.40 * atr, 2)
        sl = round(p + 0.95 * atr, 2)
        pos_size = 0.0

    risk = abs(p - sl) if abs(p - sl) > 0 else atr
    reward = abs(t1 - p)
    rr = round(max(1.1, reward / risk), 2)

    rationale_pts = [
        f"5-Report Fundamental Score: {fund_score}/100 (5Y Profit CAGR: {overview.profit_cagr_5y_pct or 0:.1f}%, Piotroski: {overview.piotroski_score}/9).",
        f"Balance Sheet & Cash in Hand: ₹{cash_bs.total_cash_in_hand_cr:,.0f} Cr liquid cash/investments vs ₹{cash_bs.total_borrowings_cr:,.0f} Cr total borrowings (D/E {cash_bs.debt_to_equity:.2f}x).",
        f"Stockholder Conviction: Promoters hold {shp.promoters_pct:.2f}%, FIIs {shp.fii_pct:.2f}%, DIIs {shp.dii_pct:.2f}% (Total Institutional: {shp.institutional_total_pct:.2f}%).",
        f"R&D & Innovation Moat: ₹{rd.latest_rd_spend_cr:,.0f} Cr annual R&D ({rd.latest_rd_pct_of_sales:.2f}% of sales; 5Y cumulative ₹{rd.cumulative_5y_rd_cr:,.0f} Cr).",
        f"Next-Day Technical Trigger ({tech_score}/100): {tech.trend_regime}; RSI(14) at {tech.rsi_14:.1f} with ATR(14) daily range of ₹{atr:,.2f}.",
    ]

    catalysts = (overview.pros[:3] if overview.pros else []) + [
        f"R&D & CapEx Pipeline: {rd.innovation_focus}",
        f"Institutional Sponsorship: {shp.institutional_total_pct:.1f}% held by FIIs + DIIs",
    ]
    risks = (overview.cons[:3] if overview.cons else []) + [
        f"Invalidate next-day bullish setup if NSE price closes below strict Stop-Loss of ₹{sl:,.2f}",
    ]

    exec_summary = (
        f"{overview.name} ({overview.symbol}) earns a composite conviction score of {weighted_score}/100 "
        f"({action}) for the next NSE trading session. Fundamentally backed by ₹{cash_bs.total_cash_in_hand_cr:,.0f} Cr "
        f"in cash & investments, {overview.roe_pct or 0:.1f}% ROE, {rd.latest_rd_pct_of_sales:.2f}% R&D intensity, and "
        f"{shp.promoters_pct + shp.institutional_total_pct:.1f}% combined Promoter + Institutional ownership. "
        f"For tomorrow's session, look for entries in the ₹{entry_low:,.2f}–₹{entry_high:,.2f} zone targeting "
        f"₹{t1:,.2f} (T1) and ₹{t2:,.2f} (T2) with a strict ATR-calibrated stop-loss at ₹{sl:,.2f} (Risk:Reward 1:{rr})."
    )

    llm_used = False
    llm_model_label = "Quantitative Multi-Agent Committee Engine v2"

    if llm_config and llm_config.get("provider") not in (None, "", "quant"):
        prompt = json.dumps(
            {
                "symbol": overview.symbol,
                "name": overview.name,
                "action": action,
                "conviction_score": weighted_score,
                "current_price": p,
                "entry_zone": [entry_low, entry_high],
                "target_1": t1,
                "target_2": t2,
                "stop_loss": sl,
                "agent_scores": {a.agent_name: a.score for a in committee},
                "cash_in_hand_cr": cash_bs.total_cash_in_hand_cr,
                "borrowings_cr": cash_bs.total_borrowings_cr,
                "shareholders": {
                    "promoters_pct": shp.promoters_pct,
                    "fii_pct": shp.fii_pct,
                    "dii_pct": shp.dii_pct,
                },
                "rd_spend_cr": rd.latest_rd_spend_cr,
                "rd_pct_sales": rd.latest_rd_pct_of_sales,
                "rsi_14": tech.rsi_14,
                "trend_regime": tech.trend_regime,
            }
        )
        llm_res = _call_pluggable_llm(llm_config, prompt)
        if llm_res and isinstance(llm_res, dict):
            if llm_res.get("executive_summary"):
                exec_summary = str(llm_res["executive_summary"])
            if isinstance(llm_res.get("key_catalysts"), list) and llm_res["key_catalysts"]:
                catalysts = [str(x) for x in llm_res["key_catalysts"][:4]]
            if isinstance(llm_res.get("key_risks"), list) and llm_res["key_risks"]:
                risks = [str(x) for x in llm_res["key_risks"][:4]]
            llm_used = True
            llm_model_label = f"{llm_config.get('provider', '').upper()} ({llm_config.get('model') or 'Default'})"

    return NextDayGuidance(
        action=action,  # type: ignore[arg-type]
        conviction_score=weighted_score,
        fundamental_score=fund_score,
        technical_score=tech_score,
        current_price=p,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        target_1_next_day=t1,
        target_2_swing=t2,
        stop_loss=sl,
        risk_reward_ratio=rr,
        expected_next_day_range_low=round(p - atr, 2),
        expected_next_day_range_high=round(p + atr, 2),
        position_size_suggestion_pct=pos_size,
        executive_summary=exec_summary,
        rationale_points=rationale_pts,
        key_catalysts=catalysts[:4],
        key_risks=risks[:4],
        llm_synthesized=llm_used,
        llm_model_used=llm_model_label,
    )


def run_full_stock_research(
    symbol: str, llm_config: dict[str, str] | None = None
) -> StockResearchBundle:
    """Execute end-to-end 5-agent Indian stock research and return complete bundle."""
    from datetime import datetime

    raw = fetch_raw_company_data(symbol)
    overview = CompanyOverview(**raw["overview"])
    annual_5 = [ReportPeriod(**r) for r in raw["last_5_annual_reports"]]
    quarterly_5 = [ReportPeriod(**r) for r in raw["last_5_quarterly_reports"]]
    cash_bs = CashAndBalanceSheet(**raw["cash_and_balance_sheet"])
    shp = StockholderAnalysis(**raw["stockholders"])
    rd = RDAndInnovation(**raw["rd_and_innovation"])
    news = [NewsHeadline(**n) for n in raw.get("news", [])]

    tech = build_technical_snapshot(
        raw.get("price_rows", []),
        fallback_price=overview.current_price,
        fallback_high_52w=overview.high_52w,
        fallback_low_52w=overview.low_52w,
    )
    if tech.current_price > 0 and overview.current_price == 0:
        overview.current_price = tech.current_price

    a1 = evaluate_5_reports_agent(overview, annual_5, quarterly_5)
    a2 = evaluate_cash_balance_agent(overview, cash_bs)
    a3 = evaluate_stockholders_agent(shp)
    a4 = evaluate_rd_innovation_agent(overview, rd)
    a5 = evaluate_technical_agent(tech, news)
    committee = [a1, a2, a3, a4, a5]

    guidance = synthesize_next_day_guidance(
        overview, cash_bs, shp, rd, tech, committee, llm_config=llm_config
    )

    from .intel_engine import build_promoter_and_competitor_intel

    prom_comp_intel = build_promoter_and_competitor_intel(
        overview.symbol, overview, shp
    )

    return StockResearchBundle(
        fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        overview=overview,
        last_5_annual_reports=annual_5,
        last_5_quarterly_reports=quarterly_5,
        cash_and_balance_sheet=cash_bs,
        stockholders=shp,
        rd_and_innovation=rd,
        technicals=tech,
        agent_committee=committee,
        next_day_guidance=guidance,
        promoter_and_competitor_intel=prom_comp_intel,
        news=news,
    )
