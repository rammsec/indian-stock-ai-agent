"""Build `docs/index.html` for GitHub Pages with:
1. Direct loading of GitHub Action pre-computed `docs/data/{SYMBOL}.json` (26+ NSE leaders)
2. 100% Client-Side Browser Live NSE/BSE Scraper & 5-Agent Engine (for all 4,000+ Indian stocks)
3. Direct Browser-to-LLM (Gemini / OpenAI / Anthropic) synthesis support
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CLIENT_SIDE_ENGINE_JS = r"""
/* ============================================================================
   GitHub Pages Serverless Adapter + Live Browser NSE/BSE 5-Agent Engine
   Runs 100% from GitHub Pages (https://rammsec.github.io/indian-stock-ai-agent/)
   ============================================================================ */

const _nativeFetch = window.fetch.bind(window);

async function fetchViaCorsProxy(targetUrl) {
  const proxies = [
    (u) => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
    (u) => `https://corsproxy.io/?${encodeURIComponent(u)}`,
  ];
  for (const makeUrl of proxies) {
    try {
      const r = await _nativeFetch(makeUrl(targetUrl));
      if (r.ok) return await r.text();
    } catch (e) {}
  }
  throw new Error("CORS proxy unreachable for live scrape");
}

function parseIndianNumber(val) {
  if (val === null || val === undefined) return null;
  const s = String(val).replace(/[₹,%Cr\s,]/g, "").trim();
  if (!s || s === "-" || s === "--") return null;
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

function parseScreenerSectionTable(doc, secId) {
  const sec = doc.getElementById(secId);
  if (!sec) return { headers: [], rows: {} };
  const tbl = sec.querySelector("table.data-table");
  if (!tbl) return { headers: [], rows: {} };
  const ths = Array.from(tbl.querySelectorAll("thead th")).slice(1);
  const headers = ths.map((th) => th.textContent.trim());
  const rows = {};
  tbl.querySelectorAll("tbody tr").forEach((tr) => {
    const cells = Array.from(tr.querySelectorAll("td, th"));
    if (cells.length < 2) return;
    const label = cells[0].textContent.replace(/\+/g, "").replace(/\s+/g, " ").trim();
    if (!label) return;
    rows[label] = cells.slice(1).map((td) => parseIndianNumber(td.textContent));
  });
  return { headers, rows };
}

function extractLast5Periods(headers, rows, count = 5) {
  const validIdx = headers
    .map((h, i) => ({ h: h.trim(), i }))
    .filter((x) => x.h && x.h.toUpperCase() !== "TTM")
    .map((x) => x.i);
  const idxs = validIdx.length >= count ? validIdx : headers.map((_, i) => i);

  const getVal = (colIdx, ...names) => {
    for (const nm of names) {
      for (const [k, arr] of Object.entries(rows)) {
        if (k.toLowerCase().startsWith(nm.toLowerCase()) && colIdx < arr.length) {
          return arr[colIdx];
        }
      }
    }
    return null;
  };

  const all = idxs.map((colIdx, pos) => {
    const rev = getVal(colIdx, "Sales", "Revenue", "Interest Earned", "Total Income");
    const np = getVal(colIdx, "Net Profit");
    let revG = null;
    let npG = null;
    if (pos > 0) {
      const prevCol = idxs[pos - 1];
      const pRev = getVal(prevCol, "Sales", "Revenue", "Interest Earned");
      const pNp = getVal(prevCol, "Net Profit");
      if (rev !== null && pRev && Math.abs(pRev) > 0) {
        revG = Number((((rev - pRev) / Math.abs(pRev)) * 100).toFixed(2));
      }
      if (np !== null && pNp && Math.abs(pNp) > 0) {
        npG = Number((((np - pNp) / Math.abs(pNp)) * 100).toFixed(2));
      }
    }
    return {
      period: headers[colIdx],
      revenue_cr: rev,
      expenses_cr: getVal(colIdx, "Expenses", "Operating Expenses"),
      operating_profit_cr: getVal(colIdx, "Operating Profit", "Financing Profit"),
      opm_pct: getVal(colIdx, "OPM", "Financing Margin"),
      other_income_cr: getVal(colIdx, "Other Income"),
      interest_cr: getVal(colIdx, "Interest"),
      depreciation_cr: getVal(colIdx, "Depreciation"),
      profit_before_tax_cr: getVal(colIdx, "Profit before tax"),
      tax_pct: getVal(colIdx, "Tax %", "Tax"),
      net_profit_cr: np,
      eps_inr: getVal(colIdx, "EPS in Rs", "EPS"),
      dividend_payout_pct: getVal(colIdx, "Dividend Payout"),
      revenue_growth_pct: revG,
      net_profit_growth_pct: npG,
    };
  });
  return all.slice(-count);
}

async function liveBrowserResearch(symbol) {
  const sym = symbol.toUpperCase().replace(".NS", "").replace(".BO", "").trim();
  let html = "";
  try {
    html = await fetchViaCorsProxy(`https://www.screener.in/company/${encodeURIComponent(sym)}/consolidated/`);
  } catch (e) {
    html = await fetchViaCorsProxy(`https://www.screener.in/company/${encodeURIComponent(sym)}/`);
  }
  const doc = new DOMParser().parseFromString(html, "text/html");
  const h1 = doc.querySelector("h1");
  if (!h1) throw new Error(`Could not find NSE/BSE stock '${sym}'`);
  const companyName = h1.textContent.trim();

  const cidMatch = html.match(/data-row-company-id="(\d+)"/);
  const companyId = cidMatch ? cidMatch[1] : "";

  const topRatios = {};
  doc.querySelectorAll("#top-ratios li").forEach((li) => {
    const n = li.querySelector(".name")?.textContent.trim();
    const v = li.querySelector(".value")?.textContent.trim();
    if (n && v) topRatios[n] = v;
  });

  let pl = parseScreenerSectionTable(doc, "profit-loss");
  if (!pl.headers.length) {
    const html2 = await fetchViaCorsProxy(`https://www.screener.in/company/${encodeURIComponent(sym)}/`);
    const doc2 = new DOMParser().parseFromString(html2, "text/html");
    pl = parseScreenerSectionTable(doc2, "profit-loss");
  }
  const qtr = parseScreenerSectionTable(doc, "quarters");
  const bs = parseScreenerSectionTable(doc, "balance-sheet");
  const cf = parseScreenerSectionTable(doc, "cash-flow");
  const shp = parseScreenerSectionTable(doc, "shareholding");

  const annual5 = extractLast5Periods(pl.headers, pl.rows, 5);
  const quarterly5 = extractLast5Periods(qtr.headers, qtr.rows, 5);

  const curPrice = parseIndianNumber(topRatios["Current Price"]);
  const mcapCr = parseIndianNumber(topRatios["Market Cap"]) || 0;
  const roePct = parseIndianNumber(topRatios["ROE"]);
  const rocePct = parseIndianNumber(topRatios["ROCE"]);
  const stockPe = parseIndianNumber(topRatios["Stock P/E"]);

  // Build 5Y Cash & Debt without arbitrary multipliers
  const bsYears = bs.headers.slice(-5);
  const cashHist5y = bsYears.map((yr) => {
    const idx = bs.headers.indexOf(yr);
    const getBs = (...nms) => {
      for (const nm of nms) {
        for (const [k, arr] of Object.entries(bs.rows)) {
          if (k.toLowerCase().startsWith(nm.toLowerCase()) && idx < arr.length) return arr[idx] ?? 0;
        }
      }
      return 0;
    };
    const eq = getBs("Equity Capital") + getBs("Reserves");
    const inv = getBs("Investments");
    const cashEq = getBs("Cash & Equivalents", "Cash");
    const debt = getBs("Borrowings");
    const totCash = cashEq + inv;
    const cfIdx = cf.headers.indexOf(yr);
    let ocf = null;
    let icf = null;
    if (cfIdx !== -1) {
      for (const [k, arr] of Object.entries(cf.rows)) {
        if (k.toLowerCase().includes("operating") && cfIdx < arr.length) ocf = arr[cfIdx];
        if (k.toLowerCase().includes("investing") && cfIdx < arr.length) icf = arr[cfIdx];
      }
    }
    const fcf = ocf !== null && icf !== null ? Math.round(ocf + Math.min(0, icf)) : null;
    return {
      period: yr,
      cash_equivalents_cr: cashEq,
      investments_cr: inv,
      total_cash_in_hand_cr: totCash,
      long_term_debt_cr: 0,
      short_term_debt_cr: 0,
      lease_liabilities_cr: 0,
      total_borrowings_cr: debt,
      net_cash_cr: totCash - debt,
      total_equity_cr: eq,
      total_assets_cr: getBs("Total Assets"),
      operating_cash_flow_cr: ocf,
      free_cash_flow_cr: fcf,
    };
  });
  const latestCb = cashHist5y[cashHist5y.length - 1] || {
    period: "Latest",
    cash_equivalents_cr: 0,
    investments_cr: 0,
    total_cash_in_hand_cr: 0,
    total_borrowings_cr: 0,
    net_cash_cr: 0,
    total_equity_cr: 1,
    operating_cash_flow_cr: null,
  };

  // Shareholding
  const shpQuarters = shp.headers.slice(-6).map((q) => {
    const idx = shp.headers.indexOf(q);
    const getShp = (...nms) => {
      for (const nm of nms) {
        for (const [k, arr] of Object.entries(shp.rows)) {
          if (k.toLowerCase().startsWith(nm.toLowerCase()) && idx < arr.length) return arr[idx] || 0;
        }
      }
      return 0;
    };
    return {
      period: q,
      promoters_pct: getShp("Promoters"),
      fii_pct: getShp("FIIs", "FII"),
      dii_pct: getShp("DIIs", "DII"),
      government_pct: getShp("Government"),
      public_pct: getShp("Public"),
      num_shareholders: getShp("No. of Shareholders"),
    };
  });
  const latestShp = shpQuarters[shpQuarters.length - 1] || {
    period: "UNAVAILABLE",
    promoters_pct: null,
    fii_pct: null,
    dii_pct: null,
    government_pct: null,
    public_pct: null,
  };

  // Fetch 365-day chart from Screener
  let priceSeries = [];
  if (companyId) {
    try {
      const chartTxt = await fetchViaCorsProxy(
        `https://www.screener.in/api/company/${companyId}/chart/?q=Price-DMA50-DMA200-Volume&days=365&consolidated=true`
      );
      const chartJson = JSON.parse(chartTxt);
      const byDate = {};
      (chartJson.datasets || []).forEach((ds) => {
        (ds.values || []).forEach(([dt, val]) => {
          if (!byDate[dt]) byDate[dt] = { date: dt };
          const num = parseFloat(val);
          if (ds.metric === "Price") byDate[dt].close = num;
          if (ds.metric === "DMA50") byDate[dt].dma50 = num;
          if (ds.metric === "DMA200") byDate[dt].dma200 = num;
          if (ds.metric === "Volume") byDate[dt].volume = num;
        });
      });
      priceSeries = Object.values(byDate)
        .filter((x) => x.close)
        .sort((a, b) => a.date.localeCompare(b.date))
        .slice(-90);
    } catch (e) {}
  }

  const latestPt = priceSeries[priceSeries.length - 1] || {
    close: curPrice,
    dma50: null,
    dma200: null,
  };
  const prevPt = priceSeries[priceSeries.length - 2] || null;
  const dayChg = prevPt && prevPt.close && latestPt.close != null
    ? Number((((latestPt.close - prevPt.close) / prevPt.close) * 100).toFixed(2))
    : null;

  const rdHistory = annual5.map((r) => ({
    period: r.period,
    revenue_cr: r.revenue_cr ?? null,
    rd_spend_cr: null,
    rd_pct_of_sales: null,
    intangible_assets_cr: null,
    cwip_cr: null,
    employee_cost_pct: null,
  }));

  const fundScore = roePct != null ? (roePct >= 15 ? 76.0 : 62.0) : null;
  const techScore = curPrice != null && latestPt.dma50 != null ? (curPrice >= latestPt.dma50 ? 72.0 : 54.0) : null;
  const convScore = fundScore != null && techScore != null ? Number((fundScore * 0.8 + techScore * 0.2).toFixed(1)) : null;
  const action = convScore == null ? "NO TRADE" : convScore >= 72 ? "STRONG BUY" : convScore >= 58 ? "BUY" : "HOLD / ACCUMULATE";

  return {
    fetched_at: new Date().toISOString().slice(0, 19).replace("T", " ") + " (Live Browser Scrape)",
    overview: {
      symbol: sym,
      name: companyName,
      sector: "NSE / BSE Listed",
      industry: "Indian Equities",
      market_cap_cr: mcapCr,
      current_price: curPrice,
      stock_pe: stockPe,
      roe_pct: roePct,
      roce_pct: rocePct,
      profit_cagr_5y_pct: null,
      piotroski_score: null,
    },
    last_5_annual_reports: annual5,
    last_5_quarterly_reports: quarterly5,
    cash_and_balance_sheet: {
      latest_period: latestCb.period,
      cash_equivalents_cr: latestCb.cash_equivalents_cr,
      liquid_investments_cr: latestCb.investments_cr,
      total_cash_in_hand_cr: latestCb.total_cash_in_hand_cr,
      total_borrowings_cr: latestCb.total_borrowings_cr,
      net_cash_position_cr: latestCb.net_cash_cr,
      debt_to_equity: latestCb.total_equity_cr > 0 ? Number((latestCb.total_borrowings_cr / latestCb.total_equity_cr).toFixed(2)) : null,
      cash_pct_of_mcap: mcapCr > 0 ? Number(((latestCb.total_cash_in_hand_cr / mcapCr) * 100).toFixed(1)) : null,
      cumulative_5y_ocf_cr: cashHist5y.reduce((a, b) => a + (b.operating_cash_flow_cr || 0), 0),
      roce_5y_series: [],
      history_5y: cashHist5y,
    },
    stockholders: {
      latest_period: latestShp.period,
      promoters_pct: latestShp.promoters_pct,
      fii_pct: latestShp.fii_pct,
      dii_pct: latestShp.dii_pct,
      government_pct: latestShp.government_pct,
      public_pct: latestShp.public_pct,
      institutional_total_pct: latestShp.fii_pct != null && latestShp.dii_pct != null
        ? Number((latestShp.fii_pct + latestShp.dii_pct).toFixed(2))
        : null,
      promoter_change_4q_pct: null,
      fii_change_4q_pct: null,
      dii_change_4q_pct: null,
      quarterly_history: shpQuarters,
      named_stockholders: [],
    },
    rd_and_innovation: {
      latest_rd_spend_cr: null,
      latest_rd_pct_of_sales: null,
      cumulative_5y_rd_cr: null,
      intangible_assets_cr: null,
      cwip_cr: null,
      innovation_focus: "R&D expenditure breakdown unavailable in live browser scrape (refer to statutory Annual Report PDFs).",
      rd_data_source: "UNAVAILABLE (Live Browser Scrape)",
      history_5y: rdHistory,
      annual_reports_5y: [],
    },
    technicals: {
      current_price: curPrice,
      prev_close: prevPt ? prevPt.close : null,
      day_change_pct: dayChg,
      return_1y_pct: null,
      rsi_14: null,
      macd_hist: null,
      dma_50: latestPt.dma50 ?? null,
      dma_200: latestPt.dma200 ?? null,
      atr_14: null,
      atr_pct: null,
      volume_Surge_ratio: null,
      support_1: null,
      resistance_1: null,
      trend_regime: curPrice != null && latestPt.dma50 != null
        ? (curPrice >= latestPt.dma50 ? "BULLISH UPTREND (Above 50-DMA)" : "CONSOLIDATION")
        : "UNAVAILABLE",
      price_series: priceSeries,
    },
    agent_committee: [
      {
        agent_id: "five_reports_agent",
        agent_name: "5-Report Earnings & Performance Agent",
        persona_inspiration: "Peter Lynch / Growth Analyst",
        signal: fundScore != null && fundScore >= 70 ? "BULLISH" : "NEUTRAL",
        score: fundScore ?? 0,
        weight_pct: 25,
        key_metrics: { "ROE / ROCE": `${roePct ?? "UNAVAILABLE"}% / ${rocePct ?? "UNAVAILABLE"}%`, "Stock P/E": `${stockPe || "—"}x` },
        reasoning: [`Analyzed ${annual5.length} annual reports and ${quarterly5.length} quarterly reports live.`],
      },
      {
        agent_id: "cash_balance_agent",
        agent_name: "Cash-in-Hand & Balance Sheet Agent",
        persona_inspiration: "Benjamin Graham / Balance Sheet",
        signal: latestCb.net_cash_cr >= 0 ? "BULLISH" : "NEUTRAL",
        score: latestCb.net_cash_cr >= 0 ? 82 : 60,
        weight_pct: 20,
        key_metrics: { "Cash in Hand": `₹${latestCb.total_cash_in_hand_cr} Cr`, "Borrowings": `₹${latestCb.total_borrowings_cr} Cr` },
        reasoning: [`Total Cash & Investments: ₹${latestCb.total_cash_in_hand_cr} Cr vs Debt ₹${latestCb.total_borrowings_cr} Cr.`],
      },
      {
        agent_id: "stockholders_agent",
        agent_name: "Stockholders & Institutional Agent",
        persona_inspiration: "Smart-Money Tracker",
        signal: "NEUTRAL",
        score: 50,
        weight_pct: 20,
        key_metrics: {
          "Promoters": latestShp.promoters_pct != null ? `${latestShp.promoters_pct}%` : "UNAVAILABLE",
          "FII + DII": latestShp.fii_pct != null && latestShp.dii_pct != null ? `${(latestShp.fii_pct + latestShp.dii_pct).toFixed(1)}%` : "UNAVAILABLE",
        },
        reasoning: [`Promoters hold ${latestShp.promoters_pct ?? "UNAVAILABLE"}%.`],
      },
      {
        agent_id: "rd_innovation_agent",
        agent_name: "R&D & Innovation Moat Agent",
        persona_inspiration: "Charlie Munger / Moat Analyst",
        signal: "NEUTRAL",
        score: 50,
        weight_pct: 15,
        key_metrics: { "Latest R&D": "UNAVAILABLE", "R&D % Sales": "UNAVAILABLE" },
        reasoning: ["R&D expenditure figures are not available in live browser scrape."],
      },
      {
        agent_id: "technical_catalyst_agent",
        agent_name: "Next-Day Technical Trigger Agent",
        persona_inspiration: "Stanley Druckenmiller / Price Action",
        signal: techScore != null && techScore >= 60 ? "BULLISH" : "NEUTRAL",
        score: techScore ?? 50,
        weight_pct: 20,
        key_metrics: {
          "50-DMA": latestPt.dma50 != null ? `₹${latestPt.dma50}` : "UNAVAILABLE",
          "200-DMA": latestPt.dma200 != null ? `₹${latestPt.dma200}` : "UNAVAILABLE",
        },
        reasoning: [`Trading at ₹${curPrice ?? "UNAVAILABLE"} with 50-DMA at ${latestPt.dma50 != null ? `₹${latestPt.dma50}` : "UNAVAILABLE"}.`],
      },
    ],
    next_day_guidance: {
      action: action,
      conviction_score: convScore,
      fundamental_score: fundScore,
      technical_score: techScore,
      current_price: curPrice,
      entry_zone_low: null,
      entry_zone_high: null,
      target_1_next_day: null,
      target_2_swing: null,
      stop_loss: null,
      risk_reward_ratio: null,
      position_size_suggestion_pct: null,
      executive_summary: `${companyName} (${sym}) analyzed live in browser from Screener.in: Cash & Investments stand at ₹${latestCb.total_cash_in_hand_cr.toLocaleString("en-IN")} Cr vs Borrowings of ₹${latestCb.total_borrowings_cr.toLocaleString("en-IN")} Cr, ROE ${roePct ?? "UNAVAILABLE"}%. Quantitative ML predictions and ATR targets are unavailable in serverless browser-only fallback.`,
      key_catalysts: ["Live 5-Report earnings trajectory", `ROE ${roePct ?? "UNAVAILABLE"}% & ROCE ${rocePct ?? "UNAVAILABLE"}%`],
      key_risks: ["Serverless browser scrape omits ML predictions, ATR levels, and R&D disclosures."],
      llm_model_used: "GitHub Pages Serverless + Live Browser Scrape (No ML)",
    },
    promoter_and_competitor_intel: {
      promoter_group: `${companyName} Promoter & Insider Group (${latestShp.promoters_pct ?? "UNAVAILABLE"}% Stake)`,
      key_leaders: [
        {
          name: `${companyName} Managing Director & Board`,
          role: "Promoter & Executive Leadership",
          focus: "Capital allocation, earnings growth, and institutional stakeholder value",
          handle_or_source: "BSE/NSE Corporate Filings & Earnings Concalls",
        },
      ],
      key_stakeholders_watch: [
        `Promoters hold ${latestShp.promoters_pct ?? "UNAVAILABLE"}% | FIIs hold ${latestShp.fii_pct ?? "UNAVAILABLE"}% | DIIs hold ${latestShp.dii_pct ?? "UNAVAILABLE"}%`,
      ],
      promoter_statements_and_articles: [
        {
          speaker_or_source: `${companyName} Exchange & Earnings Filing`,
          published: "Latest Quarter",
          headline: `${companyName} (${sym}) reports ROE of ${roePct ?? "UNAVAILABLE"}% and Cash & Investments of ₹${latestCb.total_cash_in_hand_cr.toLocaleString("en-IN")} Cr`,
          url: `https://www.screener.in/company/${encodeURIComponent(sym)}/`,
          share_price_impact: roePct != null && roePct >= 15 ? "BULLISH" : "NEUTRAL",
          why_it_affects_share_value: "Return on equity and balance sheet liquidity directly drive institutional valuation multiples.",
        },
      ],
      competitors: [],
    },
    news: [],
  };
}

window.fetch = async function (url, opts) {
  const u = String(url);

  // 1. Try local FastAPI server if running locally
  if (!window.location.hostname.includes("github.io") && window.location.protocol.startsWith("http")) {
    try {
      const r = await _nativeFetch(url, opts);
      if (r.ok) return r;
    } catch (e) {}
  }

  // 2. Serve from GitHub Pages static JSON database (`./data/*.json`)
  if (u.includes("/api/universe")) {
    const r = await _nativeFetch("./data/universe.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/movers")) {
    const r = await _nativeFetch("./data/movers.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/scanner")) {
    const r = await _nativeFetch("./data/scanner.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/regime")) {
    const r = await _nativeFetch("./data/regime.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/backtest")) {
    const r = await _nativeFetch("./data/backtest_summary.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/predictions")) {
    const r = await _nativeFetch("./data/latest_predictions.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/corporate-intel")) {
    const r = await _nativeFetch("./data/corporate_intel.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/stock-intelligence")) {
    const r = await _nativeFetch("./data/phase5_intelligence.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/phase5-experiments")) {
    const r = await _nativeFetch("./data/phase5_experiments.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/forward-validation")) {
    const r = await _nativeFetch("./data/forward_validation.json");
    if (r.ok) return r;
  }
  if (u.includes("/api/search")) {
    const q = decodeURIComponent(u.split("q=")[1] || "").trim();
    try {
      const uRes = await _nativeFetch("./data/universe.json");
      const uJson = await uRes.json();
      const localMatches = (uJson.universe || []).filter(
        (x) => x.symbol.includes(q.toUpperCase()) || x.name.toUpperCase().includes(q.toUpperCase())
      );
      if (localMatches.length > 0) {
        return new Response(JSON.stringify({ results: localMatches }), { status: 200 });
      }
      // Fallback: live Screener search via CORS proxy for any of 4,000+ NSE/BSE stocks!
      const rawSearch = await fetchViaCorsProxy(
        `https://www.screener.in/api/company/search/?q=${encodeURIComponent(q)}`
      );
      const items = JSON.parse(rawSearch).map((it) => {
        const parts = (it.url || "").split("/").filter((p) => p && p !== "company" && p !== "consolidated");
        return { symbol: (parts[0] || "").toUpperCase(), name: it.name, url: it.url };
      });
      return new Response(JSON.stringify({ results: items.filter((x) => x.symbol) }), { status: 200 });
    } catch (e) {
      return new Response(JSON.stringify({ results: [] }), { status: 200 });
    }
  }
  if (u.includes("/api/research/")) {
    const sym = decodeURIComponent(u.split("/api/research/")[1].split("?")[0]).toUpperCase();
    try {
      const staticRes = await _nativeFetch(`./data/${encodeURIComponent(sym)}.json`);
      if (staticRes.ok) {
        return staticRes;
      }
    } catch (e) {}
    // Live browser scrape fallback for any NSE/BSE stock not pre-cached in ./data/
    try {
      const liveBundle = await liveBrowserResearch(sym);
      return new Response(JSON.stringify(liveBundle), { status: 200 });
    } catch (err) {
      return new Response(JSON.stringify({ detail: err.message }), { status: 404 });
    }
  }

  return _nativeFetch(url, opts);
};
"""


def main() -> None:
    css = (ROOT_DIR / "frontend" / "styles.css").read_text()
    html = (ROOT_DIR / "frontend" / "index.html").read_text()
    js = (ROOT_DIR / "frontend" / "app.js").read_text()
    report_js = (ROOT_DIR / "frontend" / "report.js").read_text()

    out_html = html.replace(
        '<link rel="stylesheet" href="/static/styles.css" />', f"<style>\n{css}\n</style>"
    )
    out_html = out_html.replace('<script src="/static/report.js"></script>', f"<script>\n{report_js}\n</script>")
    out_html = out_html.replace(
        '<script src="/static/app.js"></script>',
        f"<script>\n{CLIENT_SIDE_ENGINE_JS}\n{js}\n</script>",
    )

    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "index.html").write_text(out_html)
    print(f"Built serverless GitHub Pages app at {docs_dir / 'index.html'} ({len(out_html)} bytes)")


if __name__ == "__main__":
    main()
