/* BharatAlpha AI v3.0 — Frontend Application Logic
   Features:
   - Focused 5-to-10 Company Watchlist with Direct "+ Add Stock" & Remove on Page
   - Expand / Collapse Accordion on Every Section Heading + Collapse/Expand All
   - Deep Promoter, Stakeholder, Statement/Tweet & Competitor War-Room (Section 07)
   - Yesterday's Top 20 Gainers & Top 20 Losers Pages with 2 Reasons per Stock
*/

const DEFAULT_5_WATCHLIST = [
  { symbol: "RELIANCE", name: "Reliance Industries Ltd" },
  { symbol: "TCS", name: "Tata Consultancy Services Ltd" },
  { symbol: "TATAELXSI", name: "Tata Elxsi Ltd" },
  { symbol: "ITC", name: "ITC Ltd" },
  { symbol: "BIRLACABLE", name: "Birla Cable Ltd" },
];

const MAX_WATCHLIST_SIZE = 10;

function loadSavedWatchlist() {
  try {
    const raw = localStorage.getItem("ba_watchlist_v3");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.slice(0, MAX_WATCHLIST_SIZE);
      }
    }
  } catch (e) {}
  return [...DEFAULT_5_WATCHLIST];
}

function saveWatchlist(list) {
  try {
    localStorage.setItem("ba_watchlist_v3", JSON.stringify(list.slice(0, MAX_WATCHLIST_SIZE)));
  } catch (e) {}
}

const state = {
  watchlist: loadSavedWatchlist(),
  currentSymbol: "RELIANCE",
  bundle: null,
  moversData: null,
  allExpanded: true,
  llmConfig: {
    provider: localStorage.getItem("ba_llm_provider") || "quant",
    model: localStorage.getItem("ba_llm_model") || "",
    api_key: localStorage.getItem("ba_llm_key") || "",
  },
};

function fmtCr(val) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return "₹" + Number(val).toLocaleString("en-IN", { maximumFractionDigits: 0 }) + " Cr";
}

function fmtNum(val, dec = 2) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return Number(val).toLocaleString("en-IN", {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  });
}

function fmtPct(val, dec = 2, signed = false) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  const n = Number(val);
  const prefix = signed && n > 0 ? "+" : "";
  return `${prefix}${n.toFixed(dec)}%`;
}

function setLoading(active) {
  const bar = document.getElementById("loading-bar");
  if (!bar) return;
  if (active) {
    bar.style.width = "72%";
    bar.style.opacity = "1";
  } else {
    bar.style.width = "100%";
    setTimeout(() => {
      bar.style.opacity = "0";
      bar.style.width = "0%";
    }, 280);
  }
}

async function initApp() {
  state.currentSymbol = state.watchlist[0]?.symbol || "RELIANCE";
  updateLLMBadge();
  setupMainPageNavigation();
  setupCollapsibleSections();
  setupWatchlistAndAddControls();
  renderWatchlistBar();
  await loadStockResearch(state.currentSymbol);
}

function updateLLMBadge() {
  const badge = document.getElementById("llm-status-badge");
  if (!badge) return;
  const p = state.llmConfig.provider || "quant";
  badge.textContent = p === "quant" ? "🧠 Engine: Quant Committee" : `🧠 Engine: ${p.toUpperCase()} + Quant`;
}

function setupMainPageNavigation() {
  const navBtns = document.querySelectorAll(".page-nav-btn");
  const companyBar = document.getElementById("company-selector-bar");

  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetPage = btn.getAttribute("data-page");
      navBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".page-view").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const pane = document.getElementById(targetPage);
      if (pane) pane.classList.add("active");

      if (targetPage === "page-company") {
        companyBar.style.display = "flex";
        if (state.bundle) requestAnimationFrame(() => renderAllCanvases(state.bundle));
      } else {
        companyBar.style.display = "flex";
        loadTop20Movers();
      }
    });
  });

  document.getElementById("btn-refresh-gainers")?.addEventListener("click", () => loadTop20Movers(true));
  document.getElementById("btn-refresh-losers")?.addEventListener("click", () => loadTop20Movers(true));
}

function setupCollapsibleSections() {
  document.querySelectorAll(".collapsible-header[data-toggle]").forEach((hdr) => {
    hdr.addEventListener("click", () => {
      const secId = hdr.getAttribute("data-toggle");
      const sec = document.getElementById(secId);
      if (!sec) return;
      sec.classList.toggle("open");
      if (sec.classList.contains("open") && state.bundle) {
        requestAnimationFrame(() => renderAllCanvases(state.bundle));
      }
    });
  });

  const toggleAllBtn = document.getElementById("btn-toggle-all-sections");
  toggleAllBtn?.addEventListener("click", () => {
    state.allExpanded = !state.allExpanded;
    document.querySelectorAll("#page-company .collapsible-section").forEach((sec) => {
      sec.classList.toggle("open", state.allExpanded);
    });
    toggleAllBtn.innerHTML = state.allExpanded
      ? `<span>↕ Collapse All Sections</span>`
      : `<span>↕ Expand All Sections</span>`;
    if (state.allExpanded && state.bundle) {
      requestAnimationFrame(() => renderAllCanvases(state.bundle));
    }
  });
}

function renderWatchlistBar() {
  const container = document.getElementById("watchlist-chips-container");
  const counterLabel = document.getElementById("watchlist-count-label");
  if (!container) return;

  counterLabel.textContent = `My Portfolio (${state.watchlist.length}/${MAX_WATCHLIST_SIZE}):`;

  container.innerHTML = state.watchlist
    .map((item, idx) => {
      const isAct = item.symbol === state.currentSymbol;
      return `
      <div class="company-chip ${isAct ? "active" : ""}" data-sym="${item.symbol}" title="${item.name || item.symbol}">
        <span class="mono" style="font-size: 0.75rem; color: var(--text-muted);">${idx + 1}.</span>
        <span>${item.symbol}</span>
        ${
          state.watchlist.length > 1
            ? `<button class="chip-remove" data-remove="${item.symbol}" title="Remove ${item.symbol} from list">×</button>`
            : ""
        }
      </div>
    `;
    })
    .join("");

  container.querySelectorAll(".company-chip").forEach((chip) => {
    chip.addEventListener("click", (e) => {
      if (e.target.closest(".chip-remove")) return;
      const sym = chip.getAttribute("data-sym");
      switchToCompanyPageAndLoad(sym);
    });
  });

  container.querySelectorAll(".chip-remove").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const symToRemove = btn.getAttribute("data-remove");
      state.watchlist = state.watchlist.filter((x) => x.symbol !== symToRemove);
      saveWatchlist(state.watchlist);
      if (state.currentSymbol === symToRemove && state.watchlist.length > 0) {
        state.currentSymbol = state.watchlist[0].symbol;
        loadStockResearch(state.currentSymbol);
      }
      renderWatchlistBar();
    });
  });
}

function addStockToWatchlist(symbol, name = "") {
  const sym = (symbol || "").trim().toUpperCase().replace(".NS", "").replace(".BO", "");
  if (!sym) return;

  const existing = state.watchlist.find((x) => x.symbol === sym);
  if (existing) {
    switchToCompanyPageAndLoad(sym);
    return;
  }

  if (state.watchlist.length >= MAX_WATCHLIST_SIZE) {
    alert(`Your watchlist already has ${MAX_WATCHLIST_SIZE} companies. Click '×' on any company chip at the top to free a slot first.`);
    return;
  }

  state.watchlist.push({ symbol: sym, name: name || sym });
  saveWatchlist(state.watchlist);
  renderWatchlistBar();
  switchToCompanyPageAndLoad(sym);
}

function switchToCompanyPageAndLoad(symbol) {
  const compBtn = document.querySelector('.page-nav-btn[data-page="page-company"]');
  if (compBtn && !compBtn.classList.contains("active")) {
    compBtn.click();
  }
  loadStockResearch(symbol);
}

function setupWatchlistAndAddControls() {
  const input = document.getElementById("stock-search-input");
  const dropdown = document.getElementById("search-dropdown");
  const btnAdd = document.getElementById("btn-add-stock");
  const btnReset = document.getElementById("btn-reset-watchlist");

  let debounceTimer = null;
  input.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const q = input.value.trim();
    if (q.length < 1) {
      dropdown.classList.remove("open");
      return;
    }
    debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        const items = data.results || [];
        if (!items.length) {
          dropdown.classList.remove("open");
          return;
        }
        dropdown.innerHTML = items
          .map(
            (it) => `
          <div class="search-item" data-sym="${it.symbol}" data-name="${it.name || it.symbol}">
            <span><strong>${it.name}</strong></span>
            <span class="mono badge-neutral">+ Add ${it.symbol}</span>
          </div>
        `
          )
          .join("");
        dropdown.classList.add("open");
        dropdown.querySelectorAll(".search-item").forEach((el) => {
          el.addEventListener("click", () => {
            const sym = el.getAttribute("data-sym");
            const nm = el.getAttribute("data-name");
            input.value = "";
            dropdown.classList.remove("open");
            addStockToWatchlist(sym, nm);
          });
        });
      } catch (e) {
        console.error(e);
      }
    }, 220);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      dropdown.classList.remove("open");
      const sym = input.value.trim().toUpperCase();
      if (sym) {
        input.value = "";
        addStockToWatchlist(sym, sym);
      }
    }
  });

  btnAdd.addEventListener("click", () => {
    dropdown.classList.remove("open");
    const sym = input.value.trim().toUpperCase();
    if (sym) {
      input.value = "";
      addStockToWatchlist(sym, sym);
    } else {
      input.focus();
    }
  });

  btnReset.addEventListener("click", () => {
    state.watchlist = [...DEFAULT_5_WATCHLIST];
    saveWatchlist(state.watchlist);
    state.currentSymbol = "RELIANCE";
    renderWatchlistBar();
    switchToCompanyPageAndLoad("RELIANCE");
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".add-stock-wrap")) {
      dropdown.classList.remove("open");
    }
  });

  // LLM Modal controls
  const modal = document.getElementById("llm-modal");
  document.getElementById("btn-open-llm-modal").addEventListener("click", () => {
    document.getElementById("llm-provider").value = state.llmConfig.provider || "quant";
    document.getElementById("llm-model").value = state.llmConfig.model || "";
    document.getElementById("llm-api-key").value = state.llmConfig.api_key || "";
    modal.classList.add("open");
  });
  document.getElementById("btn-close-llm-modal").addEventListener("click", () => {
    modal.classList.remove("open");
  });
  document.getElementById("btn-save-llm-modal").addEventListener("click", () => {
    state.llmConfig.provider = document.getElementById("llm-provider").value;
    state.llmConfig.model = document.getElementById("llm-model").value.trim();
    state.llmConfig.api_key = document.getElementById("llm-api-key").value.trim();
    localStorage.setItem("ba_llm_provider", state.llmConfig.provider);
    localStorage.setItem("ba_llm_model", state.llmConfig.model);
    localStorage.setItem("ba_llm_key", state.llmConfig.api_key);
    updateLLMBadge();
    modal.classList.remove("open");
    loadStockResearch(state.currentSymbol);
  });

  window.addEventListener("resize", () => {
    if (state.bundle) renderAllCanvases(state.bundle);
  });
}

async function loadStockResearch(symbol) {
  const sym = symbol.trim().toUpperCase();
  state.currentSymbol = sym;
  renderWatchlistBar();
  setLoading(true);

  try {
    let res;
    if (state.llmConfig.provider && state.llmConfig.provider !== "quant") {
      res = await fetch(`/api/research/${encodeURIComponent(sym)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.llmConfig),
      });
    } else {
      res = await fetch(`/api/research/${encodeURIComponent(sym)}`);
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed fetching research for ${sym}`);
    }
    const bundle = await res.json();
    state.bundle = bundle;

    // Update name in watchlist if found
    const wlItem = state.watchlist.find((x) => x.symbol === bundle.overview.symbol);
    if (wlItem && bundle.overview.name) {
      wlItem.name = bundle.overview.name;
      saveWatchlist(state.watchlist);
    }

    renderBundle(bundle);
  } catch (err) {
    alert(`Could not load '${sym}': ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function renderBundle(b) {
  const ov = b.overview;
  const gd = b.next_day_guidance;
  const cb = b.cash_and_balance_sheet;
  const shp = b.stockholders;
  const rd = b.rd_and_innovation;
  const tech = b.technicals;
  const intel = b.promoter_and_competitor_intel || {};

  // Section 01: Overview & Next-Day Ticket
  document.getElementById("ov-company-name").textContent = ov.name;
  document.getElementById("ov-symbol-badge").textContent = `NSE: ${ov.symbol}`;
  document.getElementById("ov-sector-badge").textContent = `${ov.sector} • ${ov.industry}`;
  document.getElementById("ov-piotroski-badge").textContent = `Piotroski F-Score: ${ov.piotroski_score}/9`;
  document.getElementById("ov-timestamp").textContent = `Updated: ${b.fetched_at}`;

  document.getElementById("ov-current-price").textContent = `₹${fmtNum(tech.current_price)}`;
  const chgEl = document.getElementById("ov-day-change");
  const isUp = tech.day_change_pct >= 0;
  chgEl.textContent = `${isUp ? "▲" : "▼"} ${fmtPct(tech.day_change_pct, 2, true)} (1D) | 1Y: ${fmtPct(tech.return_1y_pct, 1, true)}`;
  chgEl.className = `price-change ${isUp ? "text-bull" : "text-bear"}`;

  document.getElementById("kpi-mcap").textContent = fmtCr(ov.market_cap_cr);
  document.getElementById("kpi-cash").textContent = fmtCr(cb.total_cash_in_hand_cr);
  document.getElementById("kpi-debt").textContent = fmtCr(cb.total_borrowings_cr);
  document.getElementById("kpi-rd").textContent = `${fmtCr(rd.latest_rd_spend_cr)} (${fmtPct(rd.latest_rd_pct_of_sales, 2)})`;
  document.getElementById("kpi-roe-roce").textContent = `${fmtPct(ov.roe_pct, 1)} / ${fmtPct(ov.roce_pct, 1)}`;
  document.getElementById("kpi-pe").textContent = ov.stock_pe ? `${fmtNum(ov.stock_pe, 1)}x` : "—";
  document.getElementById("kpi-ownership").textContent = `${fmtPct(shp.promoters_pct, 1)} / ${fmtPct(shp.institutional_total_pct, 1)}`;
  document.getElementById("kpi-profit-cagr").textContent = fmtPct(ov.profit_cagr_5y_pct, 1, true);

  document.getElementById("ov-model-used").textContent = gd.llm_model_used;
  document.getElementById("ov-exec-summary").textContent = gd.executive_summary;

  const banner = document.getElementById("gd-action-banner");
  banner.textContent = gd.action;
  banner.className = gd.action.includes("BUY")
    ? "signal-banner bull"
    : gd.action.includes("SELL") || gd.action.includes("AVOID")
    ? "signal-banner bear"
    : "signal-banner hold";

  document.getElementById("gd-conviction-score").textContent = `${gd.conviction_score.toFixed(1)} / 100`;
  document.getElementById("gd-entry-zone").textContent = `₹${fmtNum(gd.entry_zone_low)} – ₹${fmtNum(gd.entry_zone_high)}`;
  document.getElementById("gd-stop-loss").textContent = `₹${fmtNum(gd.stop_loss)}`;
  document.getElementById("gd-target-1").textContent = `₹${fmtNum(gd.target_1_next_day)}`;
  document.getElementById("gd-target-2").textContent = `₹${fmtNum(gd.target_2_swing)}`;

  document.getElementById("bar-fund-score").style.width = `${gd.fundamental_score}%`;
  document.getElementById("txt-fund-score").textContent = `${gd.fundamental_score}/100`;
  document.getElementById("bar-tech-score").style.width = `${gd.technical_score}%`;
  document.getElementById("txt-tech-score").textContent = `${gd.technical_score}/100`;
  document.getElementById("gd-rr").textContent = `1 : ${gd.risk_reward_ratio}`;
  document.getElementById("gd-pos-size").textContent = `${gd.position_size_suggestion_pct}% Capital`;
  document.getElementById("gd-atr-range").textContent = `±₹${fmtNum(tech.atr_14)} (${fmtPct(tech.atr_pct, 2)})`;

  renderAgentCommittee(b.agent_committee);
  renderCatalystsRisksNews(gd, b.news, tech);
  renderReportsTables(b.last_5_annual_reports, b.last_5_quarterly_reports);
  renderCashSection(cb);
  renderStockholdersSection(shp);
  renderRDSection(rd);
  renderPromotersAndCompetitorsSection(intel, ov.name);
  requestAnimationFrame(() => renderAllCanvases(b));
}

function renderAgentCommittee(committee) {
  const grid = document.getElementById("agents-committee-grid");
  grid.innerHTML = (committee || [])
    .map((ag) => {
      const cls =
        ag.signal === "BULLISH"
          ? "signal-banner bull"
          : ag.signal === "BEARISH"
          ? "signal-banner bear"
          : "signal-banner hold";
      const metricsHtml = Object.entries(ag.key_metrics || {})
        .map(
          ([k, v]) => `
        <div class="agent-metric-pair">
          <span>${k}</span>
          <strong>${v}</strong>
        </div>
      `
        )
        .join("");
      const reasonsHtml = (ag.reasoning || []).map((r) => `<li>${r}</li>`).join("");
      return `
      <article class="agent-card">
        <div class="agent-card-header">
          <div>
            <div class="agent-name">${ag.agent_name}</div>
            <div class="agent-persona">${ag.persona_inspiration} • Weight: ${ag.weight_pct}%</div>
          </div>
          <span class="agent-pill ${cls}">${ag.signal} (${ag.score})</span>
        </div>
        <div class="agent-metrics-list">${metricsHtml}</div>
        <ul class="agent-reasons">${reasonsHtml}</ul>
      </article>
    `;
    })
    .join("");
}

function renderCatalystsRisksNews(gd, news, tech) {
  document.getElementById("chart-regime-sub").textContent = tech.trend_regime;
  document.getElementById("rsi-badge").textContent = `RSI(14): ${tech.rsi_14.toFixed(1)} | Vol: ${tech.volume_Surge_ratio}x`;

  document.getElementById("list-catalysts").innerHTML = (gd.key_catalysts || [])
    .map((c) => `<li>${c}</li>`)
    .join("");
  document.getElementById("list-risks").innerHTML = (gd.key_risks || [])
    .map((r) => `<li>${r}</li>`)
    .join("");

  const newsEl = document.getElementById("news-feed-list");
  if (!news || !news.length) {
    newsEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.82rem;">No recent news headlines found.</div>`;
    return;
  }
  newsEl.innerHTML = news
    .map((n) => {
      const color =
        n.sentiment === "POSITIVE"
          ? "var(--signal-bull)"
          : n.sentiment === "NEGATIVE"
          ? "var(--signal-bear)"
          : "var(--text-muted)";
      return `
      <a href="${n.url}" target="_blank" rel="noopener" style="text-decoration: none; color: inherit; background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 0.55rem 0.75rem; display: flex; justify-content: space-between; align-items: center; gap: 0.6rem;">
        <span style="font-size: 0.84rem; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${n.title}</span>
        <span class="mono" style="font-size: 0.72rem; color: ${color}; white-space: nowrap;">${n.source} • ${n.sentiment}</span>
      </a>
    `;
    })
    .join("");
}

function renderReportsTables(annual5, quarterly5) {
  const buildTable = (elId, reports, isAnnual) => {
    const tbl = document.getElementById(elId);
    if (!tbl || !reports.length) return;
    const headers = reports.map((r) => `<th>${r.period}</th>`).join("");
    const rowSpec = [
      { label: "Revenue / Sales (₹ Cr)", key: "revenue_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "YoY / Seq Revenue Growth %", key: "revenue_growth_pct", fmt: (v) => fmtPct(v, 1, true), colored: true },
      { label: "Total Expenses (₹ Cr)", key: "expenses_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Operating Profit / EBIT (₹ Cr)", key: "operating_profit_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Operating Profit Margin (OPM %)", key: "opm_pct", fmt: (v) => fmtPct(v, 1) },
      { label: "Other Income (₹ Cr)", key: "other_income_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Interest Cost (₹ Cr)", key: "interest_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Depreciation (₹ Cr)", key: "depreciation_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Profit Before Tax (₹ Cr)", key: "profit_before_tax_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Net Profit (₹ Cr)", key: "net_profit_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Net Profit Growth %", key: "net_profit_growth_pct", fmt: (v) => fmtPct(v, 1, true), colored: true },
      { label: "EPS (₹ per Share)", key: "eps_inr", fmt: (v) => `₹${fmtNum(v, 2)}` },
    ];
    if (isAnnual) {
      rowSpec.push({ label: "Dividend Payout %", key: "dividend_payout_pct", fmt: (v) => fmtPct(v, 1) });
    }

    const bodyHtml = rowSpec
      .map((spec) => {
        const cells = reports
          .map((r) => {
            const val = r[spec.key];
            let cls = "";
            if (spec.colored && val !== null && val !== undefined) {
              cls = val > 0 ? "text-bull" : val < 0 ? "text-bear" : "";
            }
            return `<td class="${cls}">${spec.fmt(val)}</td>`;
          })
          .join("");
        return `<tr><td>${spec.label}</td>${cells}</tr>`;
      })
      .join("");

    tbl.innerHTML = `<thead><tr><th>Metric</th>${headers}</tr></thead><tbody>${bodyHtml}</tbody>`;
  };

  buildTable("table-annual-5", annual5, true);
  buildTable("table-quarterly-5", quarterly5, false);
}

function renderCashSection(cb) {
  const cardsEl = document.getElementById("cash-summary-cards");
  const netPositive = cb.net_cash_position_cr >= 0;
  cardsEl.innerHTML = `
    <div class="kpi-box">
      <span class="kpi-label">Cash Equivalents</span>
      <span class="kpi-value text-bull">${fmtCr(cb.cash_equivalents_cr)}</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">Liquid Investments</span>
      <span class="kpi-value text-bull">${fmtCr(cb.liquid_investments_cr)}</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">Total Cash in Hand</span>
      <span class="kpi-value text-bull">${fmtCr(cb.total_cash_in_hand_cr)}</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">Total Borrowings</span>
      <span class="kpi-value">${fmtCr(cb.total_borrowings_cr)}</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">Net Cash / (Debt)</span>
      <span class="kpi-value ${netPositive ? "text-bull" : "text-bear"}">${fmtCr(cb.net_cash_position_cr)}</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">Debt / Equity Ratio</span>
      <span class="kpi-value">${fmtNum(cb.debt_to_equity, 2)}x</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">5Y Cumulative OCF</span>
      <span class="kpi-value">${fmtCr(cb.cumulative_5y_ocf_cr)}</span>
    </div>
    <div class="kpi-box">
      <span class="kpi-label">Cash % of Market Cap</span>
      <span class="kpi-value">${fmtPct(cb.cash_pct_of_mcap, 1)}</span>
    </div>
  `;

  const roceStrip = document.getElementById("roce-5y-strip");
  if (cb.roce_5y_series && cb.roce_5y_series.length) {
    roceStrip.innerHTML = `
      <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.4rem; text-transform: uppercase;">5-Year Return on Capital Employed (ROCE %)</div>
      <div style="display: flex; gap: 0.55rem; flex-wrap: wrap;">
        ${cb.roce_5y_series
          .map(
            (r) => `
          <div class="badge-neutral mono" style="padding: 0.4rem 0.7rem;">
            ${r.period}: <strong style="color: var(--text-primary);">${r.roce_pct}%</strong>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  } else {
    roceStrip.innerHTML = "";
  }

  const hist = cb.history_5y || [];
  const tbl = document.getElementById("table-cash-5y");
  if (!tbl || !hist.length) return;
  const headers = hist.map((h) => `<th>${h.period}</th>`).join("");
  const rows = [
    { label: "Cash & Bank Equivalents (₹ Cr)", key: "cash_equivalents_cr" },
    { label: "Treasury & Liquid Investments (₹ Cr)", key: "investments_cr" },
    { label: "Total Cash in Hand (₹ Cr)", key: "total_cash_in_hand_cr", bull: true },
    { label: "Long-Term Borrowings (₹ Cr)", key: "long_term_debt_cr" },
    { label: "Short-Term Borrowings (₹ Cr)", key: "short_term_debt_cr" },
    { label: "Lease Liabilities (₹ Cr)", key: "lease_liabilities_cr" },
    { label: "Total Borrowings / Debt (₹ Cr)", key: "total_borrowings_cr" },
    { label: "Net Cash Position (Cash - Debt) (₹ Cr)", key: "net_cash_cr", signedColor: true },
    { label: "Total Shareholders' Equity (₹ Cr)", key: "total_equity_cr" },
    { label: "Operating Cash Flow (OCF) (₹ Cr)", key: "operating_cash_flow_cr", signedColor: true },
    { label: "Estimated Free Cash Flow (FCF) (₹ Cr)", key: "free_cash_flow_cr", signedColor: true },
  ];

  const body = rows
    .map((r) => {
      const cells = hist
        .map((h) => {
          const v = h[r.key];
          let cls = r.bull ? "text-bull" : "";
          if (r.signedColor && v !== null && v !== undefined) {
            cls = v >= 0 ? "text-bull" : "text-bear";
          }
          return `<td class="${cls}">${fmtNum(v, 0)}</td>`;
        })
        .join("");
      return `<tr><td>${r.label}</td>${cells}</tr>`;
    })
    .join("");
  tbl.innerHTML = `<thead><tr><th>Balance Sheet &amp; Cash Metric</th>${headers}</tr></thead><tbody>${body}</tbody>`;
}

function renderStockholdersSection(shp) {
  const namedTbl = document.getElementById("table-named-holders");
  const named = shp.named_stockholders || [];
  if (!named.length) {
    namedTbl.innerHTML = `<tbody><tr><td style="text-align:left; color: var(--text-muted);">Named investor breakdown unavailable for this security.</td></tr></tbody>`;
  } else {
    namedTbl.innerHTML = `
      <thead>
        <tr>
          <th>Stockholder Name</th>
          <th>Category</th>
          <th>Latest Stake %</th>
          <th>4Q Change %</th>
        </tr>
      </thead>
      <tbody>
        ${named
          .map((n) => {
            const chgCls = n.change_4q_pct > 0 ? "text-bull" : n.change_4q_pct < 0 ? "text-bear" : "";
            return `
            <tr>
              <td>${n.name}</td>
              <td><span class="badge-neutral">${n.category}</span></td>
              <td class="mono">${fmtPct(n.latest_holding_pct, 2)}</td>
              <td class="mono ${chgCls}">${fmtPct(n.change_4q_pct, 2, true)}</td>
            </tr>
          `;
          })
          .join("")}
      </tbody>
    `;
  }

  const qHist = shp.quarterly_history || [];
  const shpTbl = document.getElementById("table-shp-history");
  if (!shpTbl || !qHist.length) return;
  const headers = qHist.map((q) => `<th>${q.period}</th>`).join("");
  const spec = [
    { label: "Promoters Holding %", key: "promoters_pct", fmt: (v) => fmtPct(v, 2) },
    { label: "Foreign Institutions (FII) %", key: "fii_pct", fmt: (v) => fmtPct(v, 2) },
    { label: "Domestic Institutions (DII) %", key: "dii_pct", fmt: (v) => fmtPct(v, 2) },
    { label: "Government Holding %", key: "government_pct", fmt: (v) => fmtPct(v, 2) },
    { label: "Public / Retail Float %", key: "public_pct", fmt: (v) => fmtPct(v, 2) },
    { label: "Total Shareholder Accounts", key: "num_shareholders", fmt: (v) => (v ? Number(v).toLocaleString("en-IN") : "—") },
  ];
  const body = spec
    .map((s) => {
      const cells = qHist.map((q) => `<td>${s.fmt(q[s.key])}</td>`).join("");
      return `<tr><td>${s.label}</td>${cells}</tr>`;
    })
    .join("");
  shpTbl.innerHTML = `<thead><tr><th>Ownership Category</th>${headers}</tr></thead><tbody>${body}</tbody>`;
}

function renderRDSection(rd) {
  document.getElementById("rd-source-subtitle").textContent = `Source: ${rd.rd_data_source}`;
  document.getElementById("rd-focus-banner").innerHTML = `
    <div style="font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.3rem;">R&amp;D &amp; Innovation Pipeline Focus</div>
    <div style="color: var(--text-primary); font-weight: 600;">${rd.innovation_focus}</div>
    <div style="margin-top: 0.65rem; display: flex; gap: 1.1rem; flex-wrap: wrap; font-size: 0.82rem;" class="mono">
      <span>Latest R&amp;D: <strong class="text-bull">${fmtCr(rd.latest_rd_spend_cr)} (${fmtPct(rd.latest_rd_pct_of_sales, 2)})</strong></span>
      <span>5Y Cumulative R&amp;D: <strong>${fmtCr(rd.cumulative_5y_rd_cr)}</strong></span>
      <span>Intangibles: <strong>${fmtCr(rd.intangible_assets_cr)}</strong></span>
      <span>CapEx WIP (CWIP): <strong>${fmtCr(rd.cwip_cr)}</strong></span>
    </div>
  `;

  const docsEl = document.getElementById("annual-reports-links");
  const docs = rd.annual_reports_5y || [];
  if (!docs.length) {
    docsEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.84rem;">No direct Annual Report PDF links found.</div>`;
  } else {
    docsEl.innerHTML = docs
      .map(
        (d) => `
      <a href="${d.url}" target="_blank" rel="noopener" class="btn" style="justify-content: space-between; text-decoration: none;">
        <span>📄 ${d.title}</span>
        <span class="mono" style="font-size: 0.74rem; color: var(--accent-brand);">PDF ↗</span>
      </a>
    `
      )
      .join("");
  }

  const hist = rd.history_5y || [];
  const tbl = document.getElementById("table-rd-5y");
  if (!tbl || !hist.length) return;
  const headers = hist.map((h) => `<th>${h.period}</th>`).join("");
  const rows = [
    { label: "Annual Revenue / Turnover (₹ Cr)", key: "revenue_cr", fmt: (v) => fmtNum(v, 0) },
    { label: "R&D Expenditure (₹ Cr)", key: "rd_spend_cr", fmt: (v) => fmtNum(v, 1), bull: true },
    { label: "R&D Intensity (% of Sales)", key: "rd_pct_of_sales", fmt: (v) => fmtPct(v, 2), bull: true },
    { label: "Intangible Assets & IP (₹ Cr)", key: "intangible_assets_cr", fmt: (v) => fmtNum(v, 0) },
    { label: "Capital Work in Progress (CWIP ₹ Cr)", key: "cwip_cr", fmt: (v) => fmtNum(v, 0) },
    { label: "Employee & Knowledge Cost (% of Sales)", key: "employee_cost_pct", fmt: (v) => fmtPct(v, 2) },
  ];
  const body = rows
    .map((r) => {
      const cells = hist
        .map((h) => `<td class="${r.bull ? "text-bull" : ""}">${r.fmt(h[r.key])}</td>`)
        .join("");
      return `<tr><td>${r.label}</td>${cells}</tr>`;
    })
    .join("");
  tbl.innerHTML = `<thead><tr><th>R&amp;D &amp; Innovation Metric</th>${headers}</tr></thead><tbody>${body}</tbody>`;
}

function renderPromotersAndCompetitorsSection(intel, companyName) {
  document.getElementById("prom-group-banner").textContent =
    intel.promoter_group || `Promoters & Strategic Institutional Leadership of ${companyName}`;

  const leadersGrid = document.getElementById("prom-leaders-grid");
  const leaders = intel.key_leaders || [];
  leadersGrid.innerHTML = leaders
    .map(
      (ld) => `
    <div class="leader-card">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;">
        <strong style="font-size: 0.96rem; color: var(--text-primary);">${ld.name}</strong>
        <span class="badge-neutral" style="font-size: 0.7rem;">${ld.role}</span>
      </div>
      <p style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.45rem;">
        <strong>Strategic Focus:</strong> ${ld.focus}
      </p>
      <div class="mono" style="font-size: 0.73rem; color: var(--text-muted); margin-top: 0.45rem;">
        📡 Channel: ${ld.handle_or_source || "Corporate Filings & Concalls"}
      </div>
    </div>
  `
    )
    .join("");

  const watchEl = document.getElementById("prom-stakeholders-watch");
  watchEl.innerHTML = (intel.key_stakeholders_watch || [])
    .map((w) => `<li>${w}</li>`)
    .join("");

  // Statements & Articles
  const stmtList = document.getElementById("prom-statements-list");
  const stmts = intel.promoter_statements_and_articles || [];
  if (!stmts.length) {
    stmtList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;">No recent promoter or executive statements found.</div>`;
  } else {
    stmtList.innerHTML = stmts
      .map((st) => {
        const badgeCls =
          st.share_price_impact === "BULLISH"
            ? "signal-banner bull"
            : st.share_price_impact === "BEARISH"
            ? "signal-banner bear"
            : "signal-banner hold";
        return `
        <div class="statement-card">
          <div style="display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <span style="font-size: 0.76rem; font-weight: 700; color: var(--accent-brand);">${st.speaker_or_source}</span>
            <div style="display: flex; align-items: center; gap: 0.45rem;">
              <span class="mono" style="font-size: 0.7rem; color: var(--text-muted);">${st.published}</span>
              <span class="${badgeCls}" style="font-size: 0.7rem; padding: 0.14rem 0.5rem;">${st.share_price_impact} IMPACT</span>
            </div>
          </div>
          <a href="${st.url || '#'}" target="_blank" rel="noopener" style="font-size: 0.92rem; font-weight: 700; color: var(--text-primary); text-decoration: none;">
            “${st.headline}” ↗
          </a>
          <div style="font-size: 0.82rem; color: var(--text-secondary); background: var(--bg-surface); padding: 0.45rem 0.65rem; border-radius: 6px; border-left: 3px solid var(--accent-brand);">
            <strong>Share-Value Effect:</strong> ${st.why_it_affects_share_value}
          </div>
        </div>
      `;
      })
      .join("");
  }

  // Competitors Grid
  const compGrid = document.getElementById("competitors-cards-grid");
  const comps = intel.competitors || [];
  if (!comps.length) {
    compGrid.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;">No competitor data mapped.</div>`;
  } else {
    compGrid.innerHTML = comps
      .map((c) => {
        const chgUp = (c.day_change_pct || 0) >= 0;
        const sigCls =
          c.read_across_signal === "POSITIVE READ-ACROSS"
            ? "text-bull"
            : c.read_across_signal === "COMPETITIVE THREAT"
            ? "text-bear"
            : "text-hold";
        return `
        <div class="competitor-card">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;">
            <div>
              <strong style="font-size: 0.95rem;">${c.name} (<span class="mono">${c.symbol}</span>)</strong>
              <span style="display: block; font-size: 0.75rem; color: var(--text-muted);">${c.segment}</span>
            </div>
            <div style="text-align: right;" class="mono">
              <div style="font-weight: 700;">${c.current_price ? `₹${fmtNum(c.current_price)}` : "NSE Peer"}</div>
              <div class="${chgUp ? "text-bull" : "text-bear"}" style="font-size: 0.76rem;">
                1D: ${fmtPct(c.day_change_pct, 2, true)} | 1M: ${fmtPct(c.return_1m_pct, 1, true)}
              </div>
            </div>
          </div>

          <a href="${c.latest_news_url || '#'}" target="_blank" rel="noopener" style="font-size: 0.84rem; color: var(--text-primary); text-decoration: none; background: var(--bg-surface); padding: 0.5rem 0.7rem; border-radius: 6px; border: 1px solid var(--border-subtle);">
            📰 <strong>Latest Peer Statement/News:</strong> ${c.latest_statement_or_news} ↗
          </a>

          <div style="font-size: 0.8rem; color: var(--text-secondary);">
            <span class="mono ${sigCls}" style="font-weight: 700;">[${c.read_across_signal}]</span>
            ${c.impact_on_share_value}
          </div>

          <div style="display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 0.2rem;">
            <button class="btn btn-add-comp" data-sym="${c.symbol}" data-name="${c.name}" style="padding: 0.28rem 0.65rem; font-size: 0.74rem;">
              + Add ${c.symbol} to My 10
            </button>
          </div>
        </div>
      `;
      })
      .join("");

    compGrid.querySelectorAll(".btn-add-comp").forEach((btn) => {
      btn.addEventListener("click", () => {
        addStockToWatchlist(btn.getAttribute("data-sym"), btn.getAttribute("data-name"));
      });
    });
  }
}

/* ============================================================================
   Yesterday's Top 20 Gainers & Top 20 Losers Loader & Renderer
   ============================================================================ */

async function loadTop20Movers(force = false) {
  if (state.moversData && !force) {
    renderMoversTables(state.moversData);
    return;
  }
  setLoading(true);
  try {
    const res = await fetch("/api/movers");
    const data = await res.json();
    state.moversData = data;
    renderMoversTables(data);
  } catch (err) {
    console.error("Failed loading Top 20 Movers:", err);
  } finally {
    setLoading(false);
  }
}

function renderMoversTables(data) {
  const buildMoverRows = (items, isGainer) => {
    if (!items || !items.length) {
      return `<tr><td colspan="10" style="text-align:center; padding: 2rem; color: var(--text-muted);">No movers data available.</td></tr>`;
    }
    return items
      .map((m) => {
        const chgCls = m.day_change_pct >= 0 ? "text-bull" : "text-bear";
        const trendCls = m.return_5d_pct >= 0 ? "text-bull" : "text-bear";
        const verdictCls = m.next_day_verdict.includes("BUY") ? "text-bull" : "text-hold";
        return `
        <tr>
          <td><strong>#${m.rank}</strong></td>
          <td>
            <strong style="font-size: 0.92rem;">${m.symbol}</strong>
            <span style="display: block; font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-sans);">${m.name}</span>
            <span class="badge-neutral" style="font-size: 0.68rem; font-family: var(--font-sans);">${m.sector}</span>
          </td>
          <td>₹${fmtNum(m.current_price)}</td>
          <td class="${chgCls}" style="font-size: 0.96rem; font-weight: 800;">
            ${isGainer ? "▲" : "▼"} ${fmtPct(m.day_change_pct, 2, true)}
          </td>
          <td class="${trendCls}">${fmtPct(m.return_5d_pct, 1, true)}</td>
          <td>${fmtNum(m.volume_surge_ratio, 2)}x</td>
          <td>${fmtNum(m.rsi_14, 1)}</td>
          <td class="mover-reason-box">
            <div style="margin-bottom: 0.35rem;">
              <strong style="color: var(--text-primary);">Reason 1 (Fundamental / News):</strong> ${m.reason_1}
            </div>
            <div>
              <strong style="color: var(--text-primary);">Reason 2 (Technical / Volume):</strong> ${m.reason_2}
            </div>
            ${
              m.headline_title
                ? `<a href="${m.headline_url}" target="_blank" rel="noopener" style="display: inline-block; margin-top: 0.3rem; font-size: 0.75rem; color: var(--accent-brand); text-decoration: none;">📰 Source: ${m.headline_source} ↗</a>`
                : ""
            }
          </td>
          <td style="text-align: right;">
            <strong class="${verdictCls}" style="display: block; font-size: 0.78rem;">${m.next_day_verdict}</strong>
            <span style="font-size: 0.74rem; color: var(--text-muted); display: block; margin-top: 0.2rem;">
              T1: <span class="text-bull">₹${fmtNum(m.next_day_target)}</span> | SL: <span class="text-bear">₹${fmtNum(m.next_day_stop_loss)}</span>
            </span>
          </td>
          <td>
            <div style="display: flex; flex-direction: column; gap: 0.35rem;">
              <button class="btn btn-primary btn-mover-add" data-sym="${m.symbol}" data-name="${m.name}" style="padding: 0.3rem 0.6rem; font-size: 0.74rem;">
                + Add &amp; Inspect
              </button>
            </div>
          </td>
        </tr>
      `;
      })
      .join("");
  };

  const gBody = document.getElementById("tbody-top-gainers");
  const lBody = document.getElementById("tbody-top-losers");
  if (gBody) gBody.innerHTML = buildMoverRows(data.top_20_gainers, true);
  if (lBody) lBody.innerHTML = buildMoverRows(data.top_20_losers, false);

  document.querySelectorAll(".btn-mover-add").forEach((btn) => {
    btn.addEventListener("click", () => {
      addStockToWatchlist(btn.getAttribute("data-sym"), btn.getAttribute("data-name"));
    });
  });
}

/* ============================================================================
   High-DPI HTML5 Canvas Financial Chart Renderers
   ============================================================================ */

function prepareCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !canvas.parentElement) return null;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return null;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = (rect.width - 16) * dpr;
  canvas.height = (rect.height - 16) * dpr;
  canvas.style.width = `${rect.width - 16}px`;
  canvas.style.height = `${rect.height - 16}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width - 16, h: rect.height - 16 };
}

function renderAllCanvases(b) {
  drawPriceDmaCanvas(b.technicals, b.next_day_guidance);
  drawGroupedBarCanvas(
    "reports-bar-canvas",
    b.last_5_annual_reports.map((r) => r.period),
    b.last_5_annual_reports.map((r) => r.revenue_cr || 0),
    b.last_5_annual_reports.map((r) => r.net_profit_cr || 0),
    "Revenue (₹ Cr)",
    "Net Profit (₹ Cr)",
    "#3b82f6",
    "#22c55e"
  );
  drawGroupedBarCanvas(
    "quarterly-bar-canvas",
    b.last_5_quarterly_reports.map((r) => r.period),
    b.last_5_quarterly_reports.map((r) => r.revenue_cr || 0),
    b.last_5_quarterly_reports.map((r) => r.net_profit_cr || 0),
    "Quarterly Sales (₹ Cr)",
    "Quarterly Net Profit (₹ Cr)",
    "#60a5fa",
    "#10b981"
  );
  drawGroupedBarCanvas(
    "cash-debt-canvas",
    (b.cash_and_balance_sheet.history_5y || []).map((h) => h.period),
    (b.cash_and_balance_sheet.history_5y || []).map((h) => h.total_cash_in_hand_cr || 0),
    (b.cash_and_balance_sheet.history_5y || []).map((h) => h.total_borrowings_cr || 0),
    "Cash in Hand (₹ Cr)",
    "Total Borrowings (₹ Cr)",
    "#22c55e",
    "#f43f5e"
  );
  drawShareholdingCanvas(b.stockholders.quarterly_history || []);
  drawRDCanvas(b.rd_and_innovation.history_5y || []);
}

function drawPriceDmaCanvas(tech, gd) {
  const env = prepareCanvas("price-dma-canvas");
  if (!env) return;
  const { ctx, w, h } = env;
  const pts = tech.price_series || [];
  if (pts.length < 2) return;

  const pad = { top: 24, right: 64, bottom: 26, left: 14 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const allVals = [];
  pts.forEach((p) => {
    if (p.close) allVals.push(p.close);
    if (p.dma50) allVals.push(p.dma50);
    if (p.dma200) allVals.push(p.dma200);
  });
  if (gd.target_1_next_day) allVals.push(gd.target_1_next_day);
  if (gd.stop_loss) allVals.push(gd.stop_loss);

  const minY = Math.min(...allVals) * 0.985;
  const maxY = Math.max(...allVals) * 1.015;
  const xOf = (i) => pad.left + (i / (pts.length - 1)) * cw;
  const yOf = (v) => pad.top + ch - ((v - minY) / (maxY - minY || 1)) * ch;

  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.lineWidth = 1;
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#94a3b8";
  for (let i = 0; i <= 4; i++) {
    const val = minY + ((maxY - minY) * i) / 4;
    const y = yOf(val);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillText(`₹${Math.round(val)}`, w - pad.right + 6, y + 4);
  }

  const drawLevelLine = (val, color, label) => {
    if (!val || val < minY || val > maxY) return;
    const y = yOf(val);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.fillText(`${label}: ₹${val.toFixed(0)}`, pad.left + 6, y - 4);
    ctx.restore();
  };
  drawLevelLine(gd.target_1_next_day, "#22c55e", "T1");
  drawLevelLine(gd.stop_loss, "#f43f5e", "SL");

  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    const y = yOf(p.dma200 || p.close);
    idx === 0 ? ctx.moveTo(xOf(idx), y) : ctx.lineTo(xOf(idx), y);
  });
  ctx.stroke();

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    const y = yOf(p.dma50 || p.close);
    idx === 0 ? ctx.moveTo(xOf(idx), y) : ctx.lineTo(xOf(idx), y);
  });
  ctx.stroke();

  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
  grad.addColorStop(0, "rgba(34, 197, 94, 0.22)");
  grad.addColorStop(1, "rgba(34, 197, 94, 0.0)");

  ctx.beginPath();
  pts.forEach((p, idx) => {
    const y = yOf(p.close);
    idx === 0 ? ctx.moveTo(xOf(idx), y) : ctx.lineTo(xOf(idx), y);
  });
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2.2;
  ctx.stroke();

  ctx.lineTo(xOf(pts.length - 1), pad.top + ch);
  ctx.lineTo(xOf(0), pad.top + ch);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
  const legend = [
    { label: "Close Price", color: "#22c55e" },
    { label: "50-DMA", color: "#38bdf8" },
    { label: "200-DMA", color: "#f59e0b" },
  ];
  let lx = pad.left + 6;
  legend.forEach((it) => {
    ctx.fillStyle = it.color;
    ctx.fillRect(lx, 6, 10, 10);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(it.label, lx + 14, 15);
    lx += 88;
  });
}

function drawGroupedBarCanvas(canvasId, labels, s1, s2, label1, label2, color1, color2) {
  const env = prepareCanvas(canvasId);
  if (!env || !labels.length) return;
  const { ctx, w, h } = env;
  const pad = { top: 28, right: 16, bottom: 28, left: 58 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const maxVal = Math.max(1, ...s1, ...s2) * 1.12;
  const minVal = Math.min(0, ...s1, ...s2);
  const yOf = (v) => pad.top + ch - ((v - minVal) / (maxVal - minVal || 1)) * ch;
  const zeroY = yOf(0);

  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px 'JetBrains Mono', monospace";
  for (let i = 0; i <= 4; i++) {
    const val = minVal + ((maxVal - minVal) * i) / 4;
    const y = yOf(val);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillText(Math.round(val).toLocaleString("en-IN"), 4, y + 3);
  }

  const groupW = cw / labels.length;
  const barW = Math.min(28, groupW * 0.32);

  labels.forEach((lbl, i) => {
    const cx = pad.left + (i + 0.5) * groupW;
    const v1 = s1[i] || 0;
    const v2 = s2[i] || 0;
    const y1 = yOf(v1);
    const y2 = yOf(v2);

    ctx.fillStyle = color1;
    ctx.fillRect(cx - barW - 2, Math.min(y1, zeroY), barW, Math.abs(zeroY - y1));

    ctx.fillStyle = color2;
    ctx.fillRect(cx + 2, Math.min(y2, zeroY), barW, Math.abs(zeroY - y2));

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(lbl, cx, h - 8);
  });

  ctx.textAlign = "left";
  ctx.fillStyle = color1;
  ctx.fillRect(pad.left, 6, 10, 10);
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText(label1, pad.left + 14, 15);

  ctx.fillStyle = color2;
  ctx.fillRect(pad.left + 160, 6, 10, 10);
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText(label2, pad.left + 174, 15);
}

function drawShareholdingCanvas(qHist) {
  const env = prepareCanvas("shareholding-canvas");
  if (!env || !qHist.length) return;
  const { ctx, w, h } = env;
  const pad = { top: 28, right: 16, bottom: 28, left: 42 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);
  const groupW = cw / qHist.length;
  const barW = Math.min(34, groupW * 0.55);

  const colors = { prom: "#3b82f6", fii: "#22c55e", dii: "#f59e0b", pub: "#64748b" };

  qHist.forEach((q, i) => {
    const cx = pad.left + (i + 0.5) * groupW;
    let curPct = 0;
    const segments = [
      { val: q.promoters_pct || 0, color: colors.prom },
      { val: q.fii_pct || 0, color: colors.fii },
      { val: q.dii_pct || 0, color: colors.dii },
      { val: (q.public_pct || 0) + (q.government_pct || 0), color: colors.pub },
    ];
    segments.forEach((seg) => {
      const segH = (seg.val / 100) * ch;
      const y = pad.top + ch - ((curPct + seg.val) / 100) * ch;
      ctx.fillStyle = seg.color;
      ctx.fillRect(cx - barW / 2, y, barW, Math.max(1, segH));
      curPct += seg.val;
    });

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(q.period, cx, h - 8);
  });

  ctx.textAlign = "left";
  const leg = [
    ["Promoters", colors.prom],
    ["FIIs", colors.fii],
    ["DIIs", colors.dii],
    ["Public/Other", colors.pub],
  ];
  let lx = pad.left;
  leg.forEach(([lbl, col]) => {
    ctx.fillStyle = col;
    ctx.fillRect(lx, 6, 10, 10);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(lbl, lx + 14, 15);
    lx += 95;
  });
}

function drawRDCanvas(rdHist) {
  const env = prepareCanvas("rd-chart-canvas");
  if (!env || !rdHist.length) return;
  const { ctx, w, h } = env;
  const pad = { top: 28, right: 48, bottom: 28, left: 54 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const maxSpend = Math.max(1, ...rdHist.map((r) => r.rd_spend_cr || 0)) * 1.18;
  const maxPct = Math.max(1, ...rdHist.map((r) => r.rd_pct_of_sales || 0)) * 1.25;

  ctx.clearRect(0, 0, w, h);
  const groupW = cw / rdHist.length;
  const barW = Math.min(38, groupW * 0.45);

  rdHist.forEach((r, i) => {
    const cx = pad.left + (i + 0.5) * groupW;
    const barH = ((r.rd_spend_cr || 0) / maxSpend) * ch;
    const y = pad.top + ch - barH;
    ctx.fillStyle = "rgba(59, 130, 246, 0.78)";
    ctx.fillRect(cx - barW / 2, y, barW, barH);

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(r.period, cx, h - 8);
    ctx.fillText(`₹${Math.round(r.rd_spend_cr)}Cr`, cx, y - 5);
  });

  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rdHist.forEach((r, i) => {
    const cx = pad.left + (i + 0.5) * groupW;
    const y = pad.top + ch - ((r.rd_pct_of_sales || 0) / maxPct) * ch;
    i === 0 ? ctx.moveTo(cx, y) : ctx.lineTo(cx, y);
  });
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.fillStyle = "#3b82f6";
  ctx.fillRect(pad.left, 6, 10, 10);
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText("R&D Spend (₹ Cr)", pad.left + 14, 15);

  ctx.fillStyle = "#22c55e";
  ctx.fillRect(pad.left + 145, 6, 10, 10);
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText("R&D % of Sales", pad.left + 159, 15);
}

document.addEventListener("DOMContentLoaded", initApp);
