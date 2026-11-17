/* BharatAlpha AI — Standardised Executive Research Report (presentation layer).
 *
 * Pure presentation: every value comes from existing backend fields (research bundle,
 * Phase 5 intelligence, backtest summary, forward validation). Nothing here trains,
 * re-predicts or invents data. Missing values render as UNAVAILABLE, never as 0.
 *
 * The functions are side-effect free string builders so they can be unit-tested in Node
 * (see tests/test_frontend_report_render.py).
 */

const RR_SECTIONS = [
  "Executive Summary",
  "Business Overview",
  "Financial Health",
  "Revenue and Profitability Trends",
  "Balance Sheet and Cash Flow",
  "Valuation",
  "Promoter and Institutional Ownership",
  "Competitive Position",
  "News and Market Intelligence",
  "Technical Analysis",
  "Quantitative Outlook \u2013 1D, 5D and 10D",
  "Key Risks",
  "What Investors Should Monitor",
  "Final Research View",
  "Sources and Data Freshness",
];

const RR_NA = "UNAVAILABLE";
const RR_ASSESSMENTS = [
  "Positive",
  "Cautiously Positive",
  "Neutral",
  "Cautiously Negative",
  "Negative",
  "Insufficient Data",
];
const RR_DISCLAIMER =
  "Research output from automated models and public data. It is not investment advice, not a guaranteed recommendation, and no model here has been shown to be profitable. Do your own research.";

/* ------------------------------------------------------------------ helpers */
function rrEsc(v) {
  return String(v === null || v === undefined ? "" : v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function rrIsNum(v) {
  return v !== null && v !== undefined && v !== "" && typeof v !== "boolean" && Number.isFinite(Number(v));
}

function rrNum(v, dec = 2) {
  if (!rrIsNum(v)) return RR_NA;
  return Number(v).toLocaleString("en-IN", { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function rrPct(v, dec = 2, signed = false) {
  if (!rrIsNum(v)) return RR_NA;
  const n = Number(v);
  return `${signed && n > 0 ? "+" : ""}${n.toFixed(dec)}%`;
}

function rrProb(v) {
  if (!rrIsNum(v)) return RR_NA;
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function rrCr(v) {
  if (!rrIsNum(v)) return RR_NA;
  return "\u20b9" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 }) + " Cr";
}

function rrTone(v) {
  if (!rrIsNum(v)) return "rr-neutral";
  return Number(v) > 0 ? "rr-pos" : Number(v) < 0 ? "rr-neg" : "rr-neutral";
}

function rrSafeUrl(u) {
  const s = String(u || "").trim();
  return /^https?:\/\//i.test(s) ? s : "";
}

function rrDate(s) {
  if (!s) return RR_NA;
  const str = String(s);
  const iso = str.match(/^(\d{4}-\d{2}-\d{2})/);
  if (iso) return iso[1];
  const d = new Date(str);
  return Number.isNaN(d.getTime()) ? str.slice(0, 16) : d.toISOString().slice(0, 10);
}

function rrDaysOld(s, ref) {
  if (!s) return null;
  const m = String(s).match(/^(\d{4}-\d{2}-\d{2})/);
  const d = m ? new Date(m[1] + "T00:00:00Z") : new Date(String(s));
  if (Number.isNaN(d.getTime())) return null;
  const r = ref ? new Date(ref) : new Date();
  return Math.floor((r.getTime() - d.getTime()) / 86400000);
}

function rrTable(headers, rows, opts = {}) {
  if (!rows.length) return `<p class="rr-muted">${rrEsc(opts.empty || "No data available.")}</p>`;
  const head = headers.map((h) => `<th>${rrEsc(h)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${r.map((c) => `<td>${c && c.html !== undefined ? c.html : rrEsc(c)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap rr-table"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function rrHtml(s) {
  return { html: s };
}

function rrBadge(text, tone) {
  return `<span class="rr-badge ${tone}">${rrEsc(text)}</span>`;
}

function rrSection(idx, interp, body, tag) {
  const title = RR_SECTIONS[idx];
  return (
    `<section class="rr-section" id="rr-sec-${idx + 1}" data-rr-section="${idx + 1}">` +
    `<h3 class="rr-h"><span class="rr-num">${String(idx + 1).padStart(2, "0")}</span>${rrEsc(title)}${tag ? ` <span class="rr-tag">${rrEsc(tag)}</span>` : ""}</h3>` +
    `<p class="rr-interp">${rrEsc(interp)}</p>${body}</section>`
  );
}

/* --------------------------------------------------------- data extraction */
function rrPhase5Stock(ctx) {
  const p5 = ctx.p5;
  if (!p5 || !p5.stocks) return null;
  return p5.stocks[String(ctx.bundle.overview.symbol || "").toUpperCase()] || null;
}

function rrLatest(arr) {
  return Array.isArray(arr) && arr.length ? arr[arr.length - 1] : null;
}

/* Quant outlook per horizon. Uses the phase-5 combined outlook when the stock is in the
 * pilot universe, otherwise the raw quantitative_prediction. */
function rrHorizons(ctx) {
  const q = ctx.bundle.quantitative_prediction || null;
  const p5s = rrPhase5Stock(ctx);
  const out = [];
  [
    ["1D", "prob_next_day_up", "expected_next_day_return_pct", "net_expected_1d_return_pct"],
    ["5D", "prob_5d_up", "expected_5d_return_pct", "net_expected_5d_return_pct"],
    ["10D", "prob_10d_up", "expected_10d_return_pct", "net_expected_10d_return_pct"],
  ].forEach(([h, pk, ek, nk]) => {
    const o = p5s && p5s.horizon_outlooks ? p5s.horizon_outlooks[h] : null;
    const prob = q && rrIsNum(q[pk]) ? Number(q[pk]) : null;
    const gross = q && rrIsNum(q[ek]) ? Number(q[ek]) : null;
    let net = q && rrIsNum(q[nk]) ? Number(q[nk]) : null;
    if (net === null && o && o.is_model_generated && rrIsNum(o.expected_net_return_pct)) net = Number(o.expected_net_return_pct);
    const available = prob !== null || (o && o.is_model_generated);
    out.push({
      h,
      prob,
      gross,
      net,
      available: !!available,
      downside: h === "1D" && q && rrIsNum(q.downside_risk_mae_pct) ? Number(q.downside_risk_mae_pct) : null,
      combinedProb: o && o.is_model_generated && rrIsNum(o.prob_positive) ? Number(o.prob_positive) : null,
      combinedSignal: o && o.is_model_generated ? o.combined_signal : null,
      decision: o ? o.decision : null,
      freshness: o && o.data_freshness ? o.data_freshness : null,
      asOf: (o && o.model_prediction_date) || (q && q.observation_date) || null,
    });
  });
  return out;
}

function rrInterpretHorizon(x) {
  if (!x.available || x.prob === null) {
    return "The model produced no forecast for this horizon, so nothing should be inferred from it.";
  }
  const p = x.prob;
  const lean =
    p >= 0.55 ? "leans towards a rise" : p <= 0.45 ? "leans towards a fall" : "is close to a coin flip";
  const netTxt =
    x.net === null
      ? "A cost-adjusted return is not available."
      : x.net > 0
      ? `After estimated trading costs the expected return stays positive (${rrPct(x.net, 2, true)}).`
      : `After estimated trading costs the expected return is not positive (${rrPct(x.net, 2, true)}).`;
  return `The model ${lean} (${rrProb(p)} chance of a positive return). ${netTxt}`;
}

/* --------------------------------------------------------------- assessment
 * Transparent rule: average of up to three evidence components already produced by the
 * backend (fundamental score, technical score, quant probabilities), each mapped to
 * [-1, +1]. Needs at least two components and acceptable data quality. */
function rrAssessment(ctx) {
  const b = ctx.bundle;
  const g = b.next_day_guidance || {};
  const comps = [];
  if (rrIsNum(g.fundamental_score)) comps.push({ name: "Fundamentals", v: (Number(g.fundamental_score) - 50) / 50 });
  if (rrIsNum(g.technical_score)) comps.push({ name: "Technicals", v: (Number(g.technical_score) - 50) / 50 });
  const probs = rrHorizons(ctx)
    .filter((x) => x.prob !== null)
    .map((x) => Math.max(-1, Math.min(1, (x.prob - 0.5) * 4)));
  if (probs.length) comps.push({ name: "Quant model", v: probs.reduce((a, c) => a + c, 0) / probs.length });
  const dq = b.data_quality || {};
  const lowQuality = rrIsNum(dq.quality_score) && Number(dq.quality_score) < 50;
  if (comps.length < 2 || lowQuality) {
    return {
      label: "Insufficient Data",
      score: null,
      comps,
      reason: lowQuality
        ? "Data quality score is below 50, so no overall assessment is given."
        : "Fewer than two independent evidence components are available.",
    };
  }
  const score = comps.reduce((a, c) => a + c.v, 0) / comps.length;
  const label =
    score >= 0.35
      ? "Positive"
      : score >= 0.12
      ? "Cautiously Positive"
      : score > -0.12
      ? "Neutral"
      : score > -0.35
      ? "Cautiously Negative"
      : "Negative";
  return { label, score, comps, reason: "" };
}

function rrAssessmentTone(label) {
  return label.includes("Positive") ? "rr-pos" : label.includes("Negative") ? "rr-neg" : "rr-neutral";
}

/* -------------------------------------------------- positives and concerns */
function rrPositivesConcerns(ctx) {
  const b = ctx.bundle;
  const ov = b.overview || {};
  const t = b.technicals || {};
  const sh = b.stockholders || {};
  const cb = b.cash_and_balance_sheet || {};
  const ci = b.corporate_intelligence || {};
  const pos = [];
  const con = [];
  if (rrIsNum(ov.profit_cagr_5y_pct) && Number(ov.profit_cagr_5y_pct) >= 10)
    pos.push(`5-year profit growth of ${rrPct(ov.profit_cagr_5y_pct, 1)} a year (source: screener.in).`);
  if (rrIsNum(ov.roe_pct) && Number(ov.roe_pct) >= 15) pos.push(`Return on equity of ${rrPct(ov.roe_pct, 1)} is healthy.`);
  if (rrIsNum(cb.net_cash_position_cr) && Number(cb.net_cash_position_cr) > 0)
    pos.push(`Cash exceeds borrowings by ${rrCr(cb.net_cash_position_cr)} (period ${cb.latest_period || RR_NA}).`);
  else if (rrIsNum(cb.debt_to_equity) && Number(cb.debt_to_equity) <= 0.5)
    pos.push(`Low leverage: debt-to-equity ${rrNum(cb.debt_to_equity)}.`);
  if (rrIsNum(ov.piotroski_score) && Number(ov.piotroski_score) >= 7) pos.push(`Strong financial-quality score (Piotroski ${ov.piotroski_score}/9).`);
  if (rrIsNum(sh.institutional_total_pct) && Number(sh.institutional_total_pct) >= 30)
    pos.push(`Institutions hold ${rrPct(sh.institutional_total_pct, 1)} of the company.`);
  if (rrIsNum(t.current_price) && rrIsNum(t.dma_200) && Number(t.current_price) > Number(t.dma_200))
    pos.push("Share price is above its 200-day average (long-term trend intact).");
  const h5 = rrHorizons(ctx).find((x) => x.h === "5D");
  if (h5 && h5.prob !== null && h5.prob >= 0.55 && h5.net !== null && h5.net > 0)
    pos.push(`5-day model probability ${rrProb(h5.prob)} with positive cost-adjusted expected return.`);

  if (rrIsNum(ov.roe_pct) && Number(ov.roe_pct) < 10) con.push(`Return on equity is modest at ${rrPct(ov.roe_pct, 1)}.`);
  if (rrIsNum(cb.debt_to_equity) && Number(cb.debt_to_equity) > 1) con.push(`High leverage: debt-to-equity ${rrNum(cb.debt_to_equity)}.`);
  if (rrIsNum(ci.promoter_pledge_pct) && Number(ci.promoter_pledge_pct) > 0) con.push(`Promoters have pledged ${rrPct(ci.promoter_pledge_pct, 1)} of their shares.`);
  if (rrIsNum(t.current_price) && rrIsNum(t.dma_200) && Number(t.current_price) < Number(t.dma_200))
    con.push("Share price is below its 200-day average (weak long-term trend).");
  if (rrIsNum(t.rsi_14) && Number(t.rsi_14) >= 70) con.push(`RSI ${rrNum(t.rsi_14, 1)} signals the stock is overbought in the short term.`);
  if (rrIsNum(t.rsi_14) && Number(t.rsi_14) <= 30) con.push(`RSI ${rrNum(t.rsi_14, 1)} signals the stock is oversold (selling pressure).`);
  if (rrIsNum(ov.stock_pe) && Number(ov.stock_pe) > 40) con.push(`Valuation is demanding at ${rrNum(ov.stock_pe, 1)}x earnings.`);
  rrHorizons(ctx).forEach((x) => {
    if (x.prob !== null && x.prob <= 0.45) con.push(`${x.h} model probability of a rise is only ${rrProb(x.prob)}.`);
    else if (x.net !== null && x.net <= 0 && x.prob !== null) con.push(`${x.h} expected return is not positive after costs (${rrPct(x.net, 2, true)}).`);
  });
  // Source-provided pros/cons (screener.in) fill remaining slots
  (Array.isArray(ov.pros) ? ov.pros : []).forEach((s) => pos.push(String(s)));
  (Array.isArray(ov.cons) ? ov.cons : []).forEach((s) => con.push(String(s)));
  const uniq = (a) => [...new Set(a)];
  return { positives: uniq(pos).slice(0, 3), concerns: uniq(con).slice(0, 3) };
}

/* ---------------------------------------------------------------- freshness */
function rrFreshness(ctx) {
  const b = ctx.bundle;
  const q = b.quantitative_prediction || {};
  const dq = b.data_quality || {};
  const annual = rrLatest(b.last_5_annual_reports);
  const quarter = rrLatest(b.last_5_quarterly_reports);
  const newsDates = (b.news || []).map((n) => rrDate(n.published)).filter((d) => d !== RR_NA).sort();
  const parts = [
    { label: "Research bundle fetched", value: b.fetched_at || RR_NA },
    { label: "Price / quant observation date", value: q.observation_date || RR_NA },
    { label: "Latest quarterly report period", value: quarter ? quarter.period : RR_NA },
    { label: "Latest annual report period", value: annual ? annual.period : RR_NA },
    { label: "Balance sheet period", value: (b.cash_and_balance_sheet || {}).latest_period || RR_NA },
    { label: "Shareholding period", value: (b.stockholders || {}).latest_period || RR_NA },
    { label: "Latest news item", value: newsDates.length ? newsDates[newsDates.length - 1] : RR_NA },
    { label: "Phase 5 intelligence generated", value: (ctx.p5 && ctx.p5.generated_at) || RR_NA },
    { label: "Forward validation generated", value: (ctx.forward && ctx.forward.generated_at) || RR_NA },
  ];
  const checks = [
    ["Fundamentals", (b.last_5_annual_reports || []).length > 0],
    ["Quarterly results", (b.last_5_quarterly_reports || []).length > 0],
    ["Balance sheet", rrIsNum((b.cash_and_balance_sheet || {}).total_borrowings_cr)],
    ["Ownership", rrIsNum((b.stockholders || {}).promoters_pct)],
    ["Technicals", rrIsNum((b.technicals || {}).rsi_14)],
    ["Quant model", !!b.quantitative_prediction],
    ["News", (b.news || []).length > 0],
    ["Phase 5 news intelligence", !!rrPhase5Stock(ctx)],
  ];
  const have = checks.filter((c) => c[1]).length;
  const missing = checks.filter((c) => !c[1]).map((c) => c[0]);
  const age = rrDaysOld(q.observation_date, ctx.now);
  const stale = age !== null && age > 4;
  return {
    parts,
    completenessPct: Math.round((have / checks.length) * 100),
    have,
    total: checks.length,
    missing,
    stale,
    ageDays: age,
    qualityScore: rrIsNum(dq.quality_score) ? Number(dq.quality_score) : null,
    warnings: Array.isArray(dq.warnings) ? dq.warnings : [],
    label: stale ? "Stale" : age === null ? "Unknown" : "Current",
  };
}

/* ------------------------------------------------------------------- news */
const RR_NEWS_GROUPS = [
  ["company_news", "Company"],
  ["promoter_management", "Promoter / Management"],
  ["industry_sector", "Industry"],
  ["competitor", "Competitor"],
  ["regulatory_government", "Regulatory"],
  ["global_macro", "Global Macro"],
];

const RR_GENERIC_PRICE_RE =
  /(share price|stock price|live (nse|bse)|price today|stock quote|\bquote\b|stock chart|live chart|price & chart|price chart|historical prices)/i;

function rrIsGenericPriceHeadline(text) {
  return RR_GENERIC_PRICE_RE.test(String(text || ""));
}

function rrClassifyNews(ctx) {
  const p5s = rrPhase5Stock(ctx);
  const groups = {};
  RR_NEWS_GROUPS.forEach(([k]) => (groups[k] = { material: [], excluded: [] }));
  let source = "none";
  if (p5s && Array.isArray(p5s.events_timeline) && p5s.events_timeline.length) {
    source = "phase5";
    p5s.events_timeline.forEach((e) => {
      const g = groups[e.event_category] || groups.company_news;
      const generic = rrIsGenericPriceHeadline(e.event_description);
      const weak =
        (rrIsNum(e.classification_confidence) && Number(e.classification_confidence) < 0.6) ||
        e.is_duplicate_report === true;
      const item = {
        headline: e.event_description,
        source: e.original_source,
        date: rrDate(e.publication_timestamp),
        company: e.company_name || e.ticker || "",
        related: e.related_competitor_or_industry || "",
        fact: e.facts_summary || "",
        interp: e.interpretation_summary || "",
        sentiment: e.sentiment || RR_NA,
        impact: e.impact_magnitude || RR_NA,
        duration: e.expected_impact_duration || RR_NA,
        url: rrSafeUrl(e.source_url),
        confidence: rrIsNum(e.classification_confidence) ? Number(e.classification_confidence) : null,
        method: e.extraction_method || "",
        weak,
        generic,
      };
      (generic ? g.excluded : g.material).push(item);
    });
  } else if (Array.isArray(ctx.bundle.news) && ctx.bundle.news.length) {
    source = "bundle";
    const sym = ctx.bundle.overview.symbol || "";
    const nm = ctx.bundle.overview.name || sym;
    ctx.bundle.news.forEach((n) => {
      const generic = rrIsGenericPriceHeadline(n.title);
      const item = {
        headline: n.title,
        source: n.source || RR_NA,
        date: rrDate(n.published),
        company: nm,
        related: "",
        fact: "",
        interp: "",
        sentiment: (n.sentiment || RR_NA).toString().toLowerCase(),
        impact: RR_NA,
        duration: RR_NA,
        url: rrSafeUrl(n.url),
        confidence: null,
        method: "keyword sentiment (no structured event extraction for this stock)",
        weak: true,
        generic,
      };
      (generic ? groups.company_news.excluded : groups.company_news.material).push(item);
    });
  }
  return { groups, source };
}

function rrNewsCard(it) {
  const tone = it.sentiment === "positive" ? "rr-pos" : it.sentiment === "negative" ? "rr-neg" : "rr-neutral";
  const link = it.url
    ? `<a href="${rrEsc(it.url)}" target="_blank" rel="noopener noreferrer">Open source</a>`
    : `<span class="rr-muted">Source link ${RR_NA}</span>`;
  const flag = it.weak
    ? `<div class="rr-flag">\u26a0 Uncertain classification${it.confidence !== null ? ` (confidence ${Math.round(it.confidence * 100)}%)` : ""}${it.method ? ` \u2013 ${rrEsc(it.method)}` : ""}. Treat as context, not a catalyst.</div>`
    : "";
  const fact = it.fact
    ? `<div><b>Verified fact:</b> ${rrEsc(it.fact)}</div>`
    : `<div><b>Verified fact:</b> <span class="rr-muted">only the headline above is verified</span></div>`;
  const interp = it.interp
    ? `<div><b>Model interpretation:</b> ${rrEsc(it.interp)}</div>`
    : `<div><b>Model interpretation:</b> <span class="rr-muted">${RR_NA}</span></div>`;
  return (
    `<div class="rr-news-card">` +
    `<div class="rr-news-head">${rrEsc(it.headline)}</div>` +
    `<div class="rr-news-meta">${rrEsc(it.source)} \u2022 ${rrEsc(it.date)} \u2022 Related: ${rrEsc(it.company || RR_NA)}${it.related ? ` / ${rrEsc(it.related)}` : ""}</div>` +
    `<div class="rr-news-body">${fact}${interp}</div>` +
    `<div class="rr-news-tags">${rrBadge("Sentiment: " + it.sentiment, tone)} ${rrBadge("Impact: " + it.impact, "rr-neutral")} ${rrBadge("Duration: " + it.duration, "rr-neutral")} ${link}</div>` +
    flag +
    `</div>`
  );
}

function rrNewsHtml(ctx) {
  const { groups, source } = rrClassifyNews(ctx);
  let html = "";
  let material = 0;
  let excluded = 0;
  RR_NEWS_GROUPS.forEach(([k, label]) => {
    const g = groups[k];
    material += g.material.length;
    excluded += g.excluded.length;
    html += `<div class="rr-news-group"><h4>${rrEsc(label)} <span class="rr-tag">${g.material.length} item${g.material.length === 1 ? "" : "s"}</span></h4>`;
    html += g.material.length
      ? g.material.map(rrNewsCard).join("")
      : `<p class="rr-muted">No material ${rrEsc(label.toLowerCase())} events collected for this stock.</p>`;
    if (g.excluded.length)
      html += `<details class="rr-details"><summary>${g.excluded.length} generic price-page headline${g.excluded.length === 1 ? "" : "s"} excluded (not meaningful catalysts)</summary>${g.excluded
        .map((i) => `<div class="rr-news-meta">${rrEsc(i.headline)} \u2022 ${rrEsc(i.source)} \u2022 ${rrEsc(i.date)}</div>`)
        .join("")}</details>`;
    html += `</div>`;
  });
  const note =
    source === "phase5"
      ? "Events come from the forward-collected Phase 5 news store with structured extraction."
      : source === "bundle"
      ? "This stock is outside the Phase 5 pilot universe: only headline-level news with keyword sentiment is available, so every item is flagged as uncertain."
      : "No news is available for this stock.";
  return { html, material, excluded, source, note };
}

/* ------------------------------------------------------------------ sections */
function rrExecutiveSummary(ctx) {
  const b = ctx.bundle;
  const ov = b.overview || {};
  const a = rrAssessment(ctx);
  const pc = rrPositivesConcerns(ctx);
  const fr = rrFreshness(ctx);
  const g = b.next_day_guidance || {};
  const t = b.technicals || {};
  const hz = rrHorizons(ctx);
  const sents = [];
  sents.push(
    `${ov.name || ov.symbol} (NSE: ${ov.symbol}) is a ${ov.sector || "listed"} company${ov.industry ? ` in ${ov.industry}` : ""}${rrIsNum(ov.market_cap_cr) ? ` with a market value of ${rrCr(ov.market_cap_cr)}` : ""}.`
  );
  sents.push(
    a.label === "Insufficient Data"
      ? `There is not enough reliable evidence for an overall research assessment (${a.reason}).`
      : `Combining fundamentals, technicals and the quantitative model gives an overall research assessment of "${a.label}".`
  );
  const avail = hz.filter((x) => x.prob !== null);
  if (avail.length) {
    const txt = avail.map((x) => `${x.h}: ${rrProb(x.prob)}`).join(", ");
    sents.push(`Model probability of a positive return \u2013 ${txt}; these are probabilities, not guarantees.`);
  } else {
    sents.push("The quantitative model has produced no usable forecast for this stock.");
  }
  if (rrIsNum(t.rsi_14) && t.trend_regime) sents.push(`Technically the trend reads "${t.trend_regime}" with RSI at ${rrNum(t.rsi_14, 1)}.`);
  sents.push(`Data completeness is ${fr.completenessPct}% and price data is ${fr.label.toLowerCase()}.`);
  const catalysts = (Array.isArray(g.key_catalysts) ? g.key_catalysts : []).slice(0, 3);
  const actions = ((b.corporate_intelligence || {}).recent_corporate_actions || []).slice(0, 3);
  return { a, pc, fr, sents: sents.slice(0, 5), catalysts, actions };
}

function rrSummaryHtml(ctx) {
  const b = ctx.bundle;
  const ov = b.overview || {};
  const s = rrExecutiveSummary(ctx);
  const reportDate = rrDate(ctx.now ? ctx.now : new Date().toISOString());
  const li = (arr, empty) =>
    arr.length ? `<ul>${arr.map((x) => `<li>${rrEsc(x)}</li>`).join("")}</ul>` : `<p class="rr-muted">${rrEsc(empty)}</p>`;
  const cat =
    s.catalysts.length || s.actions.length
      ? `<ul>${s.catalysts.map((c) => `<li>${rrEsc(c)}</li>`).join("")}${s.actions
          .map((c) => `<li>Corporate action: ${rrEsc(typeof c === "string" ? c : JSON.stringify(c))}</li>`)
          .join("")}</ul><p class="rr-muted">No dated earnings or event calendar is available in the data sources; these are themes, not scheduled dates.</p>`
      : `<p class="rr-muted">No catalysts identified in the available data. A dated event calendar is ${RR_NA}.</p>`;
  const compBar = s.a.comps.length
    ? `<div class="rr-basis">Basis: ${s.a.comps.map((c) => `${rrEsc(c.name)} ${c.v >= 0 ? "+" : ""}${c.v.toFixed(2)}`).join(" \u2022 ")} (scale \u22121 to +1, simple average).</div>`
    : "";
  return (
    `<div class="rr-exec" data-rr-block="executive-summary">` +
    `<div class="rr-exec-head"><div><div class="rr-company">${rrEsc(ov.name || ov.symbol)}</div>` +
    `<div class="rr-meta">NSE: ${rrEsc(ov.symbol)} \u2022 Report generated ${rrEsc(reportDate)}</div></div>` +
    `<div class="rr-assess ${rrAssessmentTone(s.a.label)}"><div class="rr-assess-label">Overall research assessment</div><div class="rr-assess-value">${rrEsc(s.a.label)}</div></div></div>` +
    compBar +
    `<p class="rr-summary-text">${s.sents.map(rrEsc).join(" ")}</p>` +
    `<div class="rr-cols">` +
    `<div class="rr-col rr-col-pos"><h4>Key positives</h4>${li(s.pc.positives, "No clear positives identified from available evidence.")}</div>` +
    `<div class="rr-col rr-col-neg"><h4>Key concerns</h4>${li(s.pc.concerns, "No clear concerns identified from available evidence.")}</div>` +
    `<div class="rr-col"><h4>Important upcoming catalysts</h4>${cat}</div></div>` +
    `<div class="rr-fresh"><b>Data freshness:</b> ${rrEsc(s.fr.label)}${s.fr.ageDays !== null ? ` (price data ${s.fr.ageDays} day${s.fr.ageDays === 1 ? "" : "s"} old)` : ""} \u2022 ` +
    `<b>Completeness:</b> ${s.fr.have}/${s.fr.total} data blocks (${s.fr.completenessPct}%)${s.fr.missing.length ? ` \u2013 missing: ${rrEsc(s.fr.missing.join(", "))}` : ""}` +
    `${s.fr.qualityScore !== null ? ` \u2022 <b>Pipeline quality score:</b> ${s.fr.qualityScore}/100` : ""}</div>` +
    `<p class="rr-disclaimer">${rrEsc(RR_DISCLAIMER)}</p></div>`
  );
}

function rrBusiness(ctx) {
  const ov = ctx.bundle.overview || {};
  const about = ov.about ? String(ov.about) : "";
  const rows = [
    ["Sector", ov.sector || RR_NA],
    ["Industry", ov.industry || RR_NA],
    ["Exchange / codes", `${ov.exchange || RR_NA} \u2022 NSE ${ov.nse_code || RR_NA} \u2022 BSE ${ov.bse_code || RR_NA}`],
    ["Market capitalisation", rrCr(ov.market_cap_cr)],
  ];
  const interp = about
    ? "What the company does, in the company's own description, followed by key identifiers."
    : "No business description is available from the data source.";
  return rrSection(1, interp, (about ? `<p class="rr-prose">${rrEsc(about)}</p>` : "") + rrTable(["Item", "Value"], rows));
}

function rrFinancialHealth(ctx) {
  const ov = ctx.bundle.overview || {};
  const cb = ctx.bundle.cash_and_balance_sheet || {};
  const rows = [
    ["Return on equity (ROE)", rrPct(ov.roe_pct, 1), "Profit earned per rupee of shareholder money. Above ~15% is generally strong."],
    ["Return on capital employed (ROCE)", rrPct(ov.roce_pct, 1), "Profit earned on all capital used."],
    ["Debt-to-equity", rrNum(cb.debt_to_equity), "Borrowings relative to shareholder equity. Above 1 means debt exceeds equity."],
    ["Piotroski F-score", rrIsNum(ov.piotroski_score) ? `${ov.piotroski_score}/9` : RR_NA, "Nine yes/no financial-quality tests; 7+ is strong."],
    ["5-year revenue CAGR", rrPct(ov.revenue_cagr_5y_pct, 1), "Average yearly sales growth over five years."],
    ["5-year profit CAGR", rrPct(ov.profit_cagr_5y_pct, 1), "Average yearly profit growth over five years."],
  ];
  const strong = rrIsNum(ov.roe_pct) && Number(ov.roe_pct) >= 15;
  const weak = rrIsNum(ov.roe_pct) && Number(ov.roe_pct) < 10;
  const interp = !rrIsNum(ov.roe_pct)
    ? "Key profitability ratios are not fully available, so financial strength cannot be summarised reliably."
    : strong
    ? "Profitability ratios are healthy; the table shows the supporting measures."
    : weak
    ? "Profitability ratios are on the weaker side; check whether growth or margins offset this."
    : "Profitability ratios are moderate \u2013 neither a clear strength nor a clear weakness.";
  return rrSection(2, interp, rrTable(["Measure", "Value", "What it means"], rows));
}

function rrTrends(ctx) {
  const ann = ctx.bundle.last_5_annual_reports || [];
  const qtr = ctx.bundle.last_5_quarterly_reports || [];
  const row = (r) => [
    r.period,
    rrCr(r.revenue_cr),
    rrPct(r.revenue_growth_pct, 1, true),
    rrCr(r.operating_profit_cr),
    rrPct(r.opm_pct, 1),
    rrCr(r.net_profit_cr),
    rrPct(r.net_profit_growth_pct, 1, true),
    rrNum(r.eps_inr),
  ];
  const heads = ["Period", "Revenue", "Revenue growth", "Operating profit", "Margin", "Net profit", "Profit growth", "EPS (\u20b9)"];
  let interp = "Yearly and quarterly results are not available.";
  if (ann.length >= 2) {
    const a0 = ann[0];
    const a1 = ann[ann.length - 1];
    if (rrIsNum(a0.revenue_cr) && rrIsNum(a1.revenue_cr) && Number(a0.revenue_cr) > 0) {
      const chg = (Number(a1.revenue_cr) / Number(a0.revenue_cr) - 1) * 100;
      interp = `Revenue moved from ${rrCr(a0.revenue_cr)} (${a0.period}) to ${rrCr(a1.revenue_cr)} (${a1.period}), a change of ${rrPct(chg, 1, true)}; net profit went from ${rrCr(a0.net_profit_cr)} to ${rrCr(a1.net_profit_cr)}.`;
    }
  }
  const body =
    `<h4 class="rr-sub">Annual</h4>${rrTable(heads, ann.map(row), { empty: "Annual results unavailable." })}` +
    `<h4 class="rr-sub">Quarterly</h4>${rrTable(heads, qtr.map(row), { empty: "Quarterly results unavailable." })}`;
  return rrSection(3, interp, body, "Reporting periods shown in first column");
}

function rrBalanceSheet(ctx) {
  const cb = ctx.bundle.cash_and_balance_sheet || {};
  // The source reports 0.0 for operating/free cash flow when it is not extracted: show as unavailable.
  const cfMissing = Number(cb.latest_ocf_cr) === 0 && Number(cb.latest_fcf_cr) === 0;
  const rows = [
    ["Cash and equivalents", rrCr(cb.cash_equivalents_cr)],
    ["Liquid investments", rrCr(cb.liquid_investments_cr)],
    ["Total cash in hand", rrCr(cb.total_cash_in_hand_cr)],
    ["Total borrowings", rrCr(cb.total_borrowings_cr)],
    ["Net cash position (cash minus debt)", rrCr(cb.net_cash_position_cr)],
    ["Debt-to-equity", rrNum(cb.debt_to_equity)],
    ["Cash-to-debt ratio", rrNum(cb.cash_to_debt_ratio)],
    ["Latest operating cash flow", cfMissing ? RR_NA : rrCr(cb.latest_ocf_cr)],
    ["Latest free cash flow", cfMissing ? RR_NA : rrCr(cb.latest_fcf_cr)],
  ];
  let interp = "Balance sheet data is not available.";
  if (rrIsNum(cb.net_cash_position_cr))
    interp =
      Number(cb.net_cash_position_cr) >= 0
        ? `Cash and liquid investments cover borrowings (net cash ${rrCr(cb.net_cash_position_cr)}), as of ${cb.latest_period || RR_NA}.`
        : `Borrowings exceed cash by ${rrCr(Math.abs(Number(cb.net_cash_position_cr)))} as of ${cb.latest_period || RR_NA}.`;
  const note = cfMissing ? `<p class="rr-muted">Cash-flow figures are reported as zero by the source and are shown as ${RR_NA} rather than as real zeros.</p>` : "";
  return rrSection(4, interp, rrTable(["Item", `Value (period ${cb.latest_period || RR_NA})`], rows) + note);
}

function rrValuation(ctx) {
  const ov = ctx.bundle.overview || {};
  const t = ctx.bundle.technicals || {};
  const rows = [
    ["Current price", rrIsNum(ov.current_price) ? `\u20b9${rrNum(ov.current_price)}` : RR_NA],
    ["52-week range", rrIsNum(ov.low_52w) && rrIsNum(ov.high_52w) ? `\u20b9${rrNum(ov.low_52w)} \u2013 \u20b9${rrNum(ov.high_52w)}` : RR_NA],
    ["Price-to-earnings (P/E)", rrIsNum(ov.stock_pe) ? `${rrNum(ov.stock_pe, 1)}x` : RR_NA],
    ["Book value per share", rrIsNum(ov.book_value) ? `\u20b9${rrNum(ov.book_value)}` : RR_NA],
    ["Price-to-book", rrIsNum(ov.book_value) && Number(ov.book_value) > 0 && rrIsNum(ov.current_price) ? `${(Number(ov.current_price) / Number(ov.book_value)).toFixed(2)}x` : RR_NA],
    ["Dividend yield", rrPct(ov.dividend_yield_pct, 2)],
    ["Return over 1 year", rrPct(t.return_1y_pct, 1, true)],
  ];
  const interp = !rrIsNum(ov.stock_pe)
    ? "A price-to-earnings ratio is not available, so valuation cannot be judged from earnings."
    : `The stock trades at ${rrNum(ov.stock_pe, 1)} times its earnings. No sector-peer valuation benchmark is available in the data, so this is not labelled cheap or expensive.`;
  return rrSection(5, interp, rrTable(["Metric", "Value"], rows));
}

function rrOwnership(ctx) {
  const sh = ctx.bundle.stockholders || {};
  const ci = ctx.bundle.corporate_intelligence || {};
  const rows = [
    ["Promoters", rrPct(sh.promoters_pct, 2), rrPct(sh.promoter_change_4q_pct, 2, true)],
    ["Foreign institutions (FII)", rrPct(sh.fii_pct, 2), rrPct(ci.fii_chg_4q, 2, true)],
    ["Domestic institutions (DII)", rrPct(sh.dii_pct, 2), rrPct(ci.dii_chg_4q, 2, true)],
    ["Government", rrPct(sh.government_pct, 2), RR_NA],
    ["Public", rrPct(sh.public_pct, 2), RR_NA],
  ];
  const pledge = rrIsNum(ci.promoter_pledge_pct) ? rrPct(ci.promoter_pledge_pct, 2) : RR_NA;
  let interp = "Shareholding data is not available.";
  if (rrIsNum(sh.promoters_pct))
    interp = `Promoters own ${rrPct(sh.promoters_pct, 1)} and institutions ${rrPct(sh.institutional_total_pct, 1)} (period ${sh.latest_period || RR_NA}). Promoter pledging: ${pledge}.`;
  return rrSection(6, interp, rrTable(["Holder", `Holding (${sh.latest_period || RR_NA})`, "Change over 4 quarters"], rows));
}

function rrCompetitive(ctx) {
  const comps = (ctx.bundle.promoter_and_competitor_intel || {}).competitors || [];
  const ov = ctx.bundle.overview || {};
  const rows = comps.map((c) => [
    `${c.name || c.symbol} (${c.symbol})`,
    c.segment || RR_NA,
    rrIsNum(c.current_price) ? `\u20b9${rrNum(c.current_price)}` : RR_NA,
    rrPct(c.return_1m_pct, 1, true),
    rrIsNum(c.pe) ? `${rrNum(c.pe, 1)}x` : RR_NA,
    rrIsNum(c.roe_pct) ? rrPct(c.roe_pct, 1) : RR_NA,
    c.read_across_signal || RR_NA,
  ]);
  const interp = rows.length
    ? `${ov.name || ov.symbol} competes with the peers below. Peer P/E and ROE are shown only where the source provides them.`
    : "No competitor data is available for this company.";
  return rrSection(7, interp, rrTable(["Competitor", "Segment", "Price", "1-month return", "P/E", "ROE", "Read-across to this stock"], rows));
}

function rrNewsSectionHtml(ctx) {
  const n = rrNewsHtml(ctx);
  const interp = n.material
    ? `${n.material} material item${n.material === 1 ? "" : "s"} grouped by type. ${n.note}`
    : `${n.note} Generic price-page headlines are never treated as catalysts.`;
  return rrSection(8, interp, n.html, n.source === "phase5" ? "Forward-collected, structured" : "Headline-level only");
}

function rrTechnical(ctx) {
  const t = ctx.bundle.technicals || {};
  const rows = [
    ["Trend regime", t.trend_regime || RR_NA],
    ["RSI (14)", rrNum(t.rsi_14, 1)],
    ["MACD line / signal", `${rrNum(t.macd_line)} / ${rrNum(t.macd_signal)}`],
    ["20-day EMA", rrNum(t.ema_20)],
    ["50-day / 200-day average", `${rrNum(t.dma_50)} / ${rrNum(t.dma_200)}`],
    ["Bollinger band (lower \u2013 upper)", `${rrNum(t.bb_lower)} \u2013 ${rrNum(t.bb_upper)}`],
    ["Support 1 / 2", `${rrNum(t.support_1)} / ${rrNum(t.support_2)}`],
    ["Resistance 1 / 2", `${rrNum(t.resistance_1)} / ${rrNum(t.resistance_2)}`],
    ["Daily range (ATR 14)", `${rrNum(t.atr_14)} (${rrPct(t.atr_pct, 2)})`],
    ["Annualised volatility", rrPct(t.annualized_volatility_pct, 1)],
    ["Volume vs 20-day average", rrIsNum(t.volume_Surge_ratio) ? `${rrNum(t.volume_Surge_ratio)}x` : RR_NA],
  ];
  let interp = "Technical indicators are not available.";
  if (rrIsNum(t.rsi_14)) {
    const rsi = Number(t.rsi_14);
    const rsiTxt = rsi >= 70 ? "overbought (price has risen fast and may pause)" : rsi <= 30 ? "oversold (price has fallen fast)" : "in a neutral zone";
    const above = rrIsNum(t.dma_200) && rrIsNum(t.current_price) ? (Number(t.current_price) > Number(t.dma_200) ? "above" : "below") : null;
    interp = `Momentum is ${rsiTxt} with RSI at ${rrNum(rsi, 1)}${above ? `, and the price is ${above} its 200-day average` : ""}.`;
  }
  const signals = Array.isArray(t.technical_signals) ? t.technical_signals : [];
  const sig = signals.length ? `<ul class="rr-list">${signals.slice(0, 6).map((s) => `<li>${rrEsc(typeof s === "string" ? s : s.signal || JSON.stringify(s))}</li>`).join("")}</ul>` : "";
  return rrSection(9, interp, rrTable(["Indicator", "Reading"], rows) + sig + `<p class="rr-muted">Price chart and indicator charts are in the Technical Appendix view.</p>`);
}

function rrQuant(ctx) {
  const hz = rrHorizons(ctx);
  const q = ctx.bundle.quantitative_prediction || {};
  const rows = hz.map((x) => [
    x.h,
    x.available ? rrBadge("Available", "rr-pos") : rrBadge(RR_NA, "rr-neutral"),
    rrProb(x.prob),
    rrHtml(`<span class="${rrTone(x.gross)}">${rrPct(x.gross, 2, true)}</span>`),
    rrHtml(`<span class="${rrTone(x.net)}">${rrPct(x.net, 2, true)}</span>`),
    x.downside !== null ? rrPct(x.downside, 2) : RR_NA,
    x.freshness ? `${x.freshness}${x.asOf ? ` (as of ${rrDate(x.asOf)})` : ""}` : x.asOf ? `As of ${rrDate(x.asOf)}` : RR_NA,
  ]);
  const interpRows = hz.map((x) => `<li><b>${x.h}:</b> ${rrEsc(rrInterpretHorizon(x))}</li>`).join("");
  const conf = rrIsNum(q.confidence_score) ? `${rrNum(q.confidence_score, 1)} / 100` : RR_NA;
  const explain =
    `<div class="rr-explain"><div><b>Probability</b> \u2013 the model's estimated chance that the stock ends the period higher. 50% means no edge.</div>` +
    `<div><b>Expected return</b> \u2013 the model's average expected price change; it can be positive even when probability is below 50%.</div>` +
    `<div><b>Confidence score (${conf})</b> \u2013 how much the underlying models agree and how complete the data is. It is <u>not</u> a probability and not a return forecast.</div></div>`;
  const combined = hz.some((x) => x.combinedProb !== null)
    ? `<h4 class="rr-sub">With news &amp; cross-stock signals (Phase 5 pilot)</h4>` +
      rrTable(
        ["Horizon", "Combined probability", "Combined signal", "Decision"],
        hz.map((x) => [x.h, rrProb(x.combinedProb), x.combinedSignal || RR_NA, x.decision || RR_NA])
      )
    : "";
  const body =
    rrTable(
      ["Horizon", "Model availability", "Probability of positive return", "Expected return (gross)", "Expected net return (after costs)", "Downside risk (typical adverse move, 1D basis)", "Data freshness"],
      rows
    ) +
    `<ul class="rr-list">${interpRows}</ul>` +
    explain +
    combined +
    `<p class="rr-muted">Backtest and live forward-validation evidence for these models is in the Technical Appendix. Sample sizes are small and the models have not been shown to be profitable.</p>`;
  const any = hz.some((x) => x.available);
  const interp = any
    ? "Three separate forecasts, one per horizon. Each is independent: a missing horizon is shown as unavailable, never as zero."
    : "No horizon has a model forecast for this stock.";
  return rrSection(10, interp, body);
}

function rrRisks(ctx) {
  const g = ctx.bundle.next_day_guidance || {};
  const pc = rrPositivesConcerns(ctx);
  const dq = ctx.bundle.data_quality || {};
  const items = [];
  (Array.isArray(g.key_risks) ? g.key_risks : []).forEach((r) => items.push(["Source analysis", String(r)]));
  pc.concerns.forEach((c) => items.push(["Rule-based flag", c]));
  (Array.isArray(dq.warnings) ? dq.warnings : []).forEach((w) => items.push(["Data quality", String(w)]));
  const seen = new Set();
  const rows = items.filter((r) => (seen.has(r[1]) ? false : seen.add(r[1])));
  const interp = rows.length
    ? "Risks gathered from the source analysis, rule-based checks on the numbers and data-quality warnings."
    : "No specific risks were flagged by the data, which does not mean the investment is risk-free.";
  const generic = `<p class="rr-muted">General risks always apply: market-wide falls, sector or regulatory change, earnings misses, and the limits of statistical models.</p>`;
  return rrSection(11, interp, rrTable(["Type", "Risk"], rows) + generic);
}

function rrMonitor(ctx) {
  const t = ctx.bundle.technicals || {};
  const g = ctx.bundle.next_day_guidance || {};
  const sh = ctx.bundle.stockholders || {};
  const ci = ctx.bundle.corporate_intelligence || {};
  const items = [];
  if (rrIsNum(t.support_1)) items.push(`Price holding above support near \u20b9${rrNum(t.support_1)}; a close below \u20b9${rrNum(t.support_2)} would weaken the setup.`);
  if (rrIsNum(t.resistance_1)) items.push(`Price reaction near resistance \u20b9${rrNum(t.resistance_1)}.`);
  if (rrIsNum(t.dma_200)) items.push(`Position relative to the 200-day average (\u20b9${rrNum(t.dma_200)}).`);
  if (rrIsNum(g.stop_loss)) items.push(`Source-model invalidation level: close below \u20b9${rrNum(g.stop_loss)}.`);
  if (rrIsNum(sh.promoter_change_4q_pct)) items.push(`Promoter holding trend (currently ${rrPct(sh.promoter_change_4q_pct, 2, true)} over 4 quarters) in the next shareholding disclosure.`);
  if (rrIsNum(ci.promoter_pledge_pct)) items.push(`Promoter pledge level (${rrPct(ci.promoter_pledge_pct, 2)}).`);
  items.push("Next quarterly results: revenue growth, margins and net profit versus the table in section 04.");
  items.push("Whether the 1D, 5D and 10D model probabilities move above or below 50% in later reports.");
  const interp = "A checklist of measurable items that would confirm or challenge the view in this report.";
  return rrSection(12, interp, `<ul class="rr-list">${items.map((i) => `<li>${rrEsc(i)}</li>`).join("")}</ul>`);
}

function rrFinal(ctx) {
  const s = rrExecutiveSummary(ctx);
  const interp = `Overall research assessment: ${s.a.label}. This is a summary of available evidence, not a recommendation to buy or sell.`;
  const body =
    `<div class="rr-assess ${rrAssessmentTone(s.a.label)} rr-assess-inline"><div class="rr-assess-value">${rrEsc(s.a.label)}</div></div>` +
    `<p class="rr-prose">${s.sents.map(rrEsc).join(" ")}</p>` +
    `<p class="rr-disclaimer">${rrEsc(RR_DISCLAIMER)}</p>`;
  return rrSection(13, interp, body);
}

function rrSources(ctx) {
  const fr = rrFreshness(ctx);
  const rows = fr.parts.map((p) => [p.label, p.value]);
  const dq = ctx.bundle.data_quality || {};
  const srcs = [
    ["Fundamentals, ownership, valuation", "screener.in via the research pipeline"],
    ["Prices, delivery, indices", "NSE bhavcopy / index data, Yahoo Finance (technicals)"],
    ["News", "Google News RSS headlines; Phase 5 forward-collected structured events (pilot stocks)"],
    ["Quantitative forecasts", "BharatAlpha ensemble model (LightGBM, XGBoost, linear) \u2013 see Technical Appendix"],
  ];
  const warn = fr.warnings.length ? `<ul class="rr-list">${fr.warnings.map((w) => `<li>${rrEsc(w)}</li>`).join("")}</ul>` : "";
  const interp = `Data completeness ${fr.completenessPct}%; price data ${fr.label.toLowerCase()}${fr.ageDays !== null ? ` (${fr.ageDays} days old)` : ""}.`;
  return rrSection(
    14,
    interp,
    rrTable(["Data block", "Date / period"], rows) +
      rrTable(["Data", "Source"], srcs) +
      (rrIsNum(dq.quality_score) ? `<p class="rr-muted">Pipeline data-quality score: ${dq.quality_score}/100.</p>` : "") +
      warn
  );
}

/* ---------------------------------------------------------------- appendix */
function rrAppendixHtml(ctx) {
  const q = ctx.bundle.quantitative_prediction || {};
  const parts = [];
  parts.push(`<h3 class="rr-h">Technical Appendix</h3><p class="rr-interp">Model diagnostics, validation and raw quantitative detail. Historical backtest results and live forward validation are different kinds of evidence and are shown separately.</p>`);

  // Model detail
  const modelRows = [
    ["Model", `${q.model_name || RR_NA} v${q.model_version || RR_NA}`],
    ["Observation date / target session", `${q.observation_date || RR_NA} \u2192 ${q.target_session_date || RR_NA}`],
    ["LightGBM / XGBoost / Linear P(up, 1D)", `${rrProb(q.prob_lgbm_up)} / ${rrProb(q.prob_xgb_up)} / ${rrProb(q.prob_linear_up)}`],
    ["Model agreement score", rrIsNum(q.model_agreement_score) ? rrNum(q.model_agreement_score) : RR_NA],
    ["Confidence score (not a probability)", rrIsNum(q.confidence_score) ? rrNum(q.confidence_score, 1) : RR_NA],
    ["Typical adverse move (MAE) / favourable (MFE)", `${rrPct(q.downside_risk_mae_pct)} / ${rrPct(q.expected_upside_mfe_pct)}`],
    ["Reward-to-risk", rrIsNum(q.reward_to_risk_ratio) ? rrNum(q.reward_to_risk_ratio) : RR_NA],
    ["Round-trip delivery cost assumption", rrPct(q.roundtrip_delivery_cost_pct, 3)],
    ["Market regime", q.market_regime || RR_NA],
    ["Recommended action (1D model)", q.recommended_action || RR_NA],
    ["Abstention reason", q.abstention_reason || "None"],
  ];
  parts.push(`<h4 class="rr-sub">Quantitative model detail</h4>` + rrTable(["Item", "Value"], modelRows, { empty: "No quantitative prediction for this stock." }));

  const drv = (arr) => (Array.isArray(arr) ? arr : []).map((d) => [d.feature, d.family || RR_NA, rrNum(d.value, 3), rrNum(d.contribution, 4)]);
  parts.push(`<h4 class="rr-sub">Top bullish drivers (TreeSHAP)</h4>` + rrTable(["Feature", "Family", "Value", "Contribution"], drv(q.top_bullish_drivers), { empty: "Unavailable." }));
  parts.push(`<h4 class="rr-sub">Top bearish drivers (TreeSHAP)</h4>` + rrTable(["Feature", "Family", "Value", "Contribution"], drv(q.top_bearish_drivers), { empty: "Unavailable." }));

  // Historical backtest
  const bt = ctx.backtest;
  if (bt && bt.models) {
    const names = ["ml_ensemble", "naive_momentum"].filter((k) => bt.models[k]);
    const rows = names.map((k) => {
      const m = bt.models[k];
      return [k, m.total_predictions, rrPct(m.directional_accuracy_all_pct, 1), rrNum(m.roc_auc, 3), rrNum(m.brier_score, 4), rrPct(m.net_cumulative_return_pct, 2, true), rrNum(m.annualized_sharpe_net, 2), rrPct(m.max_drawdown_net_pct, 2), m.active_trades_count];
    });
    parts.push(
      `<h4 class="rr-sub">Historical walk-forward backtest <span class="rr-tag">HISTORICAL \u2013 not live results</span></h4>` +
        `<p class="rr-muted">Test window ${rrEsc(bt.test_start_date || RR_NA)} to ${rrEsc(bt.test_end_date || RR_NA)}, ${rrEsc(bt.total_walk_forward_folds || RR_NA)} folds, ${rrEsc(bt.execution_mode || RR_NA)}. Nifty 50 buy-and-hold over the window: ${rrPct(bt.nifty_50_buy_hold_return_pct, 2, true)}. ROC AUC at or below 0.50 means no demonstrated ranking skill.</p>` +
        rrTable(["Model", "Predictions", "Directional accuracy", "ROC AUC", "Brier", "Net return", "Net Sharpe", "Max drawdown", "Trades"], rows)
    );
    const feats = (bt.top_global_features || []).slice(0, 10).map((f) => [f.feature, rrNum(f.importance, 4)]);
    parts.push(`<h4 class="rr-sub">Global feature importance (backtest)</h4>` + rrTable(["Feature", "Importance"], feats, { empty: "Unavailable." }));
  } else {
    parts.push(`<h4 class="rr-sub">Historical walk-forward backtest</h4><p class="rr-muted">Backtest summary ${RR_NA}.</p>`);
  }

  // Forward validation
  const fv = ctx.forward;
  if (fv && fv.horizon_evaluations) {
    const rows = ["1D", "5D", "10D"].map((h) => {
      const e = fv.horizon_evaluations[h] || {};
      const thin = rrIsNum(e.matured_count) && rrIsNum(e.min_sample_threshold) && Number(e.matured_count) < Number(e.min_sample_threshold);
      return [h, e.total_snapshots, e.matured_count, e.pending_count, e.min_sample_threshold, rrHtml(thin || e.status === "INSUFFICIENT_DATA" ? rrBadge("INSUFFICIENT DATA", "rr-warn") : rrBadge(e.status || RR_NA, "rr-neutral")), rrPct(e.directional_accuracy_pct, 1), rrNum(e.roc_auc, 3), rrPct(e.net_return_pct, 2, true)];
    });
    const canonTxt = rrIsNum(fv.canonical_snapshots_count) ? `${fv.canonical_snapshots_count} canonical (${fv.total_immutable_snapshots} stored)` : fv.total_immutable_snapshots;
    parts.push(
      `<h4 class="rr-sub">Live forward validation <span class="rr-tag">PROSPECTIVE \u2013 real predictions made before outcomes</span></h4>` +
        `<p class="rr-muted">Generated ${rrEsc(fv.generated_at || RR_NA)}. Snapshots ${rrEsc(canonTxt)}; matured outcomes ${rrEsc(fv.matured_outcomes_total)}; pending ${rrEsc(fv.pending_outcomes_total)}. Metrics are not meaningful until the minimum sample is reached.</p>` +
        rrTable(["Horizon", "Snapshots", "Matured (sample size)", "Pending", "Minimum sample", "Status", "Directional accuracy", "ROC AUC", "Net return"], rows)
    );
    if (fv.paper_portfolio && fv.paper_portfolio.strategies) {
      const pp = fv.paper_portfolio;
      const pRows = ["1D", "5D", "10D"].map((h) => {
        const s = pp.strategies[h] || {};
        return [
          `Strategy ${h}`,
          rrNum(s.ending_capital_inr, 0),
          rrPct(s.net_return_pct, 2, true),
          rrPct(s.max_drawdown_pct, 2),
          rrIsNum(s.entries_executed) ? s.entries_executed : RR_NA,
          rrIsNum(s.open_positions_count) ? s.open_positions_count : RR_NA,
          s.status || RR_NA,
        ];
      });
      const bms = pp.benchmarks || {};
      [
        ["Naive Momentum (1D)", bms.naive_momentum_1d || bms.naive_momentum],
        ["Naive Momentum (5D)", bms.naive_momentum_5d],
        ["Naive Momentum (10D)", bms.naive_momentum_10d],
        ["Nifty 50 Buy & Hold (80% cap)", bms.nifty_50_buy_hold],
        ["Cash (No Trade)", bms.cash_no_trade],
      ].forEach(([label, bm]) => {
        if (!bm) return;
        pRows.push([
          label,
          rrIsNum(bm.ending_capital_inr) ? rrNum(bm.ending_capital_inr, 0) : RR_NA,
          rrPct(bm.net_return_pct, 2, true),
          rrPct(bm.max_drawdown_pct, 2),
          RR_NA,
          RR_NA,
          bm.status || RR_NA,
        ]);
      });
      parts.push(
        `<h4 class="rr-sub">Paper portfolio &amp; benchmark comparisons <span class="rr-tag">HYPOTHETICAL \u2013 max ${rrEsc(pp.max_position_pct || 10)}% pos / ${rrEsc(pp.max_total_exposure_pct || 80)}% exposure</span></h4>` +
          rrTable(["Portfolio / Benchmark", "Ending capital (\u20b9)", "Net return", "Max drawdown", "Entries", "Open", "Status"], pRows)
      );
    }
  } else {
    parts.push(`<h4 class="rr-sub">Live forward validation</h4><p class="rr-muted">Forward validation data ${RR_NA}.</p>`);
  }
  parts.push(`<p class="rr-muted">The full existing dashboard sections (committee verdicts, charts, filings, backtest tables) follow below.</p>`);
  return parts.join("");
}

/* -------------------------------------------------------------- composition */
function rrDetailedHtml(ctx) {
  return [
    rrSummaryHtml(ctx).replace('class="rr-exec"', 'class="rr-exec rr-exec-in-detail"'),
    rrBusiness(ctx),
    rrFinancialHealth(ctx),
    rrTrends(ctx),
    rrBalanceSheet(ctx),
    rrValuation(ctx),
    rrOwnership(ctx),
    rrCompetitive(ctx),
    rrNewsSectionHtml(ctx),
    rrTechnical(ctx),
    rrQuant(ctx),
    rrRisks(ctx),
    rrMonitor(ctx),
    rrFinal(ctx),
    rrSources(ctx),
  ].join("");
}

function rrBuildViews(ctx) {
  return {
    quick: rrSummaryHtml(ctx),
    detailed: rrDetailedHtml(ctx),
    technical: rrAppendixHtml(ctx),
  };
}

/* ----------------------------------------------------------------- DOM glue */
const rrState = { view: "quick", extras: { p5: null, backtest: null, forward: null }, loaded: false };

function rrContext(bundle) {
  return {
    bundle,
    p5: rrState.extras.p5,
    backtest: rrState.extras.backtest,
    forward: rrState.extras.forward,
    now: new Date().toISOString(),
  };
}

function rrApplyView(view) {
  rrState.view = view;
  const page = document.getElementById("page-company");
  if (page) {
    page.classList.remove("rr-view-quick", "rr-view-detailed", "rr-view-technical");
    page.classList.add(`rr-view-${view}`);
  }
  document.querySelectorAll(".rr-view-btn").forEach((b) => {
    const on = b.getAttribute("data-rr-view") === view;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
  document.querySelectorAll(".rr-pane").forEach((p) => p.classList.toggle("active", p.getAttribute("data-rr-pane") === view));
  if (view === "technical" && typeof state !== "undefined" && state.bundle && typeof renderAllCanvases === "function") {
    requestAnimationFrame(() => renderAllCanvases(state.bundle));
  }
}

function renderResearchReport(bundle) {
  const host = document.getElementById("rr-root");
  if (!host || !bundle) return;
  const views = rrBuildViews(rrContext(bundle));
  const set = (id, html) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  };
  set("rr-pane-quick", views.quick);
  set("rr-pane-detailed", views.detailed);
  set("rr-pane-technical", views.technical);
}

async function rrLoadExtras() {
  const grab = async (url) => {
    try {
      const r = await fetch(url);
      if (!r.ok) return null;
      const txt = await r.text();
      return JSON.parse(txt.replace(/\bNaN\b/g, "null"));
    } catch (e) {
      return null;
    }
  };
  const [p5, backtest, forward] = await Promise.all([grab("/api/stock-intelligence"), grab("/api/backtest"), grab("/api/forward-validation")]);
  rrState.extras = { p5, backtest, forward };
  rrState.loaded = true;
  if (typeof state !== "undefined" && state.bundle) renderResearchReport(state.bundle);
}

function initReportViews() {
  document.querySelectorAll(".rr-view-btn").forEach((b) =>
    b.addEventListener("click", () => rrApplyView(b.getAttribute("data-rr-view")))
  );
  rrApplyView("quick");
  rrLoadExtras();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    RR_SECTIONS,
    RR_ASSESSMENTS,
    rrBuildViews,
    rrAssessment,
    rrHorizons,
    rrClassifyNews,
    rrIsGenericPriceHeadline,
    rrPositivesConcerns,
    rrFreshness,
  };
}
