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
  initBeginnerHelpTooltips();
  renderWatchlistBar();
  loadMarketRegimeBanner();
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
      } else if (targetPage === "page-quant") {
        companyBar.style.display = "flex";
        loadQuantAndBulkDealsPage();
      } else if (targetPage === "page-intelligence") {
        companyBar.style.display = "flex";
        loadPhase5StockIntelligence(false);
      } else {
        companyBar.style.display = "flex";
        loadTop20Movers();
      }
    });
  });

  document.getElementById("btn-refresh-gainers")?.addEventListener("click", () => loadTop20Movers(true));
  document.getElementById("btn-refresh-losers")?.addEventListener("click", () => loadTop20Movers(true));
  document.getElementById("btn-refresh-p5")?.addEventListener("click", () => loadPhase5StockIntelligence(true));
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
  if (state.latestPredData) {
    renderFiveDayOutlookAndConditions(state.latestPredData);
  }

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

  renderQuantAndCorporateIntelSection(b);
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
   Phase 1–4 Quantitative Intelligence, Market Regime, Walk-Forward Backtest
   & Official NSE Bulk / Block Deals Renderers
   ============================================================================ */

async function loadMarketRegimeBanner() {
  try {
    const [regRes, btRes, predRes] = await Promise.allSettled([
      fetch("/api/regime"),
      fetch("/api/backtest"),
      fetch("/api/predictions"),
    ]);
    if (regRes.status === "fulfilled" && regRes.value.ok) {
      const reg = await regRes.value.json();
      updateRegimeBannerDOM(reg, null);
    }
    if (btRes.status === "fulfilled" && btRes.value.ok) {
      const bt = await btRes.value.json();
      renderWalkForwardBacktestTable(bt);
    }
    if (predRes.status === "fulfilled" && predRes.value.ok) {
      const pred = await predRes.value.json();
      renderFiveDayOutlookAndConditions(pred);
    }
  } catch (e) {}
}

function updateRegimeBannerDOM(reg, quality) {
  if (reg && reg.regime) {
    const badge = document.getElementById("mr-regime-badge");
    if (badge) {
      badge.textContent = `MARKET REGIME: ${String(reg.regime).toUpperCase()}`;
      badge.style.color = String(reg.regime).includes("Bull")
        ? "var(--signal-bull)"
        : String(reg.regime).includes("Bear")
        ? "var(--signal-bear)"
        : "var(--accent-brand)";
    }
    const niftyEl = document.getElementById("mr-nifty-metric");
    if (niftyEl && reg.nifty_50_close) {
      niftyEl.innerHTML = `Nifty 50: <strong>${fmtNum(reg.nifty_50_close, 1)}</strong> (${fmtPct(reg.nifty_50_return_1d_pct, 2, true)} 1D | ${fmtPct(reg.nifty_50_return_20d_pct, 2, true)} 20D)`;
    }
    const vixEl = document.getElementById("mr-vix-metric");
    if (vixEl && reg.india_vix !== undefined && reg.india_vix !== null) {
      vixEl.innerHTML = `India VIX: <strong>${fmtNum(reg.india_vix, 2)}</strong>`;
    }
    const brEl = document.getElementById("mr-breadth-metric");
    if (brEl && reg.advance_decline_ratio !== undefined) {
      brEl.innerHTML = `NSE Breadth A/D: <strong>${fmtNum(reg.advance_decline_ratio, 2)}x</strong> (${reg.advances_count || 0}▲ / ${reg.declines_count || 0}▼)`;
    }
  }
  if (quality) {
    const gateEl = document.getElementById("mr-gate-badge");
    if (gateEl) {
      gateEl.textContent = `Data Quality: ${fmtNum(quality.quality_score, 0)}/100 (${quality.trade_eligible ? "ELIGIBLE" : "NO TRADE"})`;
      gateEl.style.color = quality.trade_eligible ? "var(--signal-bull)" : "var(--signal-bear)";
    }
  }
}

function formatShapDriverItem(d) {
  if (!d) return "";
  if (typeof d === "string") return d;
  const feat = d.feature || "feature";
  const fam = d.family ? ` (${String(d.family).replace(/^\d+\.\s*/, "")})` : "";
  const val = d.value !== null && d.value !== undefined ? ` = ${fmtNum(d.value, 2)}` : "";
  const bps = d.contribution !== undefined ? ` [${d.contribution >= 0 ? "+" : ""}${(d.contribution * 100).toFixed(1)} bps]` : "";
  return `<strong>${feat}</strong>${val}${fam} <span style="opacity:0.85;">${bps}</span>`;
}

function renderQuantAndCorporateIntelSection(b) {
  updateRegimeBannerDOM(b.market_regime, b.data_quality);

  const rawQp = b.quantitative_prediction || null;
  const hasQp = Boolean(rawQp && rawQp.prob_next_day_up !== undefined && rawQp.prob_next_day_up !== null);
  const actionStr = hasQp ? String(rawQp.recommended_action || "N/A") : "N/A";
  const prob1d = hasQp ? rawQp.prob_next_day_up : null;
  const probLgbm = hasQp ? (rawQp.prob_lgbm_up ?? null) : null;
  const probXgb = hasQp ? (rawQp.prob_xgb_up ?? null) : null;
  const gross1d = hasQp ? (rawQp.expected_next_day_return_pct ?? null) : null;
  const net1d = hasQp ? (rawQp.net_expected_1d_return_pct ?? null) : null;
  const net5d = hasQp ? (rawQp.net_expected_5d_return_pct ?? null) : null;
  const gross5d = hasQp ? (rawQp.expected_5d_return_pct ?? null) : null;
  const cncBps = hasQp && rawQp.roundtrip_delivery_cost_pct !== undefined && rawQp.roundtrip_delivery_cost_pct !== null
    ? rawQp.roundtrip_delivery_cost_pct * 100
    : null;
  const misBps = hasQp && rawQp.roundtrip_intraday_cost_pct !== undefined && rawQp.roundtrip_intraday_cost_pct !== null
    ? rawQp.roundtrip_intraday_cost_pct * 100
    : null;
  const downsideMae = hasQp ? (rawQp.downside_risk_mae_pct ?? null) : null;
  const confScore = hasQp ? (rawQp.confidence_score ?? null) : null;
  const agreeScore = hasQp && rawQp.model_agreement_score !== undefined && rawQp.model_agreement_score !== null
    ? `${(rawQp.model_agreement_score * 100).toFixed(0)}%`
    : "N/A";
  const obsDate = hasQp ? (rawQp.observation_date || "N/A") : "N/A";
  const targetDate = hasQp ? (rawQp.target_session_date || "N/A") : "N/A";
  const genAt = hasQp && rawQp.generated_at ? String(rawQp.generated_at).slice(0, 19).replace("T", " ") + " UTC" : "N/A";
  const modelVer = hasQp && rawQp.model_version ? `${rawQp.model_name || "ml_ensemble"} v${rawQp.model_version}` : "N/A";

  const ci = b.corporate_intelligence || null;

  // Render Left Panel: Calibrated ML Ensemble
  const qpActionEl = document.getElementById("qp-action-pill");
  if (qpActionEl) {
    qpActionEl.textContent = hasQp ? `ML: ${actionStr}` : "ML: N/A";
    qpActionEl.className = actionStr.includes("BUY")
      ? "signal-banner bull"
      : actionStr.includes("SELL") || actionStr.includes("AVOID")
      ? "signal-banner bear"
      : "signal-banner hold";
  }
  const qpSub = document.getElementById("qp-model-sub");
  if (qpSub) {
    qpSub.textContent = hasQp
      ? `Obs: ${obsDate} → Target: ${targetDate} • Generated: ${genAt} • Model: ${modelVer} • Agreement: ${agreeScore}`
      : "No quantitative prediction record available for this symbol.";
  }

  const qpKpi = document.getElementById("qp-kpi-strip");
  if (qpKpi) {
    const p1d = prob1d !== null ? `${(prob1d * 100).toFixed(1)}%` : "N/A";
    const pLgbm = probLgbm !== null ? `${(probLgbm * 100).toFixed(1)}%` : "N/A";
    const pXgb = probXgb !== null ? `${(probXgb * 100).toFixed(1)}%` : "N/A";
    qpKpi.innerHTML = `
      <div class="kpi-box">
        <span class="kpi-label">P(Up 1D) Calibrated</span>
        <span class="kpi-value ${prob1d !== null && prob1d >= 0.53 ? "text-bull" : ""}">${p1d}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">LGBM / XGB P(Up)</span>
        <span class="kpi-value">${pLgbm} / ${pXgb}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Gross E[R] (1D)</span>
        <span class="kpi-value ${gross1d !== null ? (gross1d >= 0 ? "text-bull" : "text-bear") : ""}">${gross1d !== null ? fmtPct(gross1d, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Net E[R] 1D (Post Cost)</span>
        <span class="kpi-value ${net1d !== null ? (net1d >= 0 ? "text-bull" : "text-bear") : ""}">${net1d !== null ? fmtPct(net1d, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Gross / Net E[R] 5D</span>
        <span class="kpi-value ${net5d !== null ? (net5d >= 0 ? "text-bull" : "text-bear") : ""}">${gross5d !== null && net5d !== null ? `${fmtPct(gross5d, 2, true)} / ${fmtPct(net5d, 2, true)}` : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Indian CNC / MIS Cost</span>
        <span class="kpi-value">${cncBps !== null && misBps !== null ? `${fmtNum(cncBps, 1)} / ${fmtNum(misBps, 1)} bps` : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">1D Downside MAE Risk</span>
        <span class="kpi-value ${downsideMae !== null ? "text-bear" : ""}">${downsideMae !== null ? fmtPct(downsideMae, 2) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Calibrated Confidence</span>
        <span class="kpi-value">${confScore !== null ? `${fmtNum(confScore, 1)} / 100` : "N/A"}</span>
      </div>
    `;
  }

  const bullDrv = document.getElementById("qp-bull-drivers");
  if (bullDrv) {
    const items = (rawQp && Array.isArray(rawQp.top_bullish_drivers)) ? rawQp.top_bullish_drivers : [];
    bullDrv.innerHTML = items.length
      ? items.map((d) => `<li>${formatShapDriverItem(d)}</li>`).join("")
      : `<li style="color: var(--text-muted);">No TreeSHAP bullish drivers available</li>`;
  }
  const bearDrv = document.getElementById("qp-bear-drivers");
  if (bearDrv) {
    const items = (rawQp && Array.isArray(rawQp.top_bearish_drivers)) ? rawQp.top_bearish_drivers : [];
    bearDrv.innerHTML = items.length
      ? items.map((d) => `<li>${formatShapDriverItem(d)}</li>`).join("")
      : `<li style="color: var(--text-muted);">No TreeSHAP bearish drivers available</li>`;
  }

  // Render Right Panel: Phase 4 Corporate & Institutional Intelligence
  const verdictStr = ci && ci.smart_money_verdict ? String(ci.smart_money_verdict) : "N/A";
  const ciPill = document.getElementById("ci-conviction-pill");
  if (ciPill) {
    const ciScoreTxt = ci && ci.institutional_conviction_score != null ? ` (${fmtNum(ci.institutional_conviction_score, 1)}/100)` : "";
    ciPill.textContent = `${verdictStr}${ciScoreTxt}`;
    ciPill.style.color = verdictStr.includes("ACCUMULATION")
      ? "var(--signal-bull)"
      : verdictStr.includes("DISTRIBUTION")
      ? "var(--signal-bear)"
      : "var(--accent-brand)";
  }

  const ciKpi = document.getElementById("ci-kpi-strip");
  if (ciKpi) {
    const f4q = ci && ci.net_institutional_flow_4q != null ? ci.net_institutional_flow_4q : null;
    const f1q = ci && ci.fii_chg_1q != null && ci.dii_chg_1q != null ? Number((ci.fii_chg_1q + ci.dii_chg_1q).toFixed(2)) : null;
    const delivDiff = ci && ci.delivery_vs_20d_avg != null ? ci.delivery_vs_20d_avg : null;
    ciKpi.innerHTML = `
      <div class="kpi-box">
        <span class="kpi-label">NSE Delivery %</span>
        <span class="kpi-value ${ci && (ci.latest_delivery_pct || 0) >= 55 ? "text-bull" : ""}">${ci && ci.latest_delivery_pct != null ? fmtPct(ci.latest_delivery_pct, 1) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Delivery vs 20D Avg</span>
        <span class="kpi-value ${delivDiff !== null ? (delivDiff >= 0 ? "text-bull" : "text-bear") : ""}">${delivDiff !== null ? fmtPct(delivDiff, 1, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">1Q Net FII+DII Flow</span>
        <span class="kpi-value ${f1q !== null ? (f1q > 0 ? "text-bull" : f1q < 0 ? "text-bear" : "") : ""}">${f1q !== null ? fmtPct(f1q, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">4Q Net FII+DII Flow</span>
        <span class="kpi-value ${f4q !== null ? (f4q > 0 ? "text-bull" : f4q < 0 ? "text-bear" : "") : ""}">${f4q !== null ? fmtPct(f4q, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box" style="grid-column: span 2;">
        <span class="kpi-label">F&amp;O Positioning Heuristic</span>
        <span class="kpi-value" style="font-size: 0.78rem;">${ci && ci.fno_positioning_heuristic ? ci.fno_positioning_heuristic : "N/A"}</span>
      </div>
      <div class="kpi-box" style="grid-column: span 2;">
        <span class="kpi-label">Promoter Δ (1Q / 4Q)</span>
        <span class="kpi-value">${ci && ci.promoter_chg_1q != null && ci.promoter_chg_4q != null ? `${fmtPct(ci.promoter_chg_1q, 2, true)} / ${fmtPct(ci.promoter_chg_4q, 2, true)}` : "N/A"}</span>
      </div>
    `;
  }

  const sigList = document.getElementById("ci-signals-list");
  if (sigList) {
    const signals = ci && Array.isArray(ci.key_signals) ? ci.key_signals : [];
    sigList.innerHTML = signals.length
      ? signals.map((s) => `<li>${s}</li>`).join("")
      : `<li style="color: var(--text-muted);">No smart-money delivery signals available</li>`;
  }

  const actList = document.getElementById("ci-actions-deals-list");
  if (actList) {
    const items = [];
    const deals = ci ? (ci.bulk_block_deals || ci.recent_bulk_block_deals || []) : [];
    deals.slice(0, 3).forEach((d) => {
      items.push(
        `<li><strong class="${d.side === "BUY" ? "text-bull" : "text-bear"}">[${d.deal_type} ${d.side}]</strong> ${d.deal_date}: ₹${fmtNum(d.value_cr, 2)} Cr @ ₹${fmtNum(d.price_inr, 1)} (${d.client_name})</li>`
      );
    });
    const actions = ci ? (ci.recent_corporate_actions || []) : [];
    actions.slice(0, 4).forEach((a) => {
      items.push(`<li><strong>[${a.action_type}]</strong> Ex-Date ${a.ex_date}: ${a.description}</li>`);
    });
    actList.innerHTML = items.length
      ? items.join("")
      : `<li style="color: var(--text-muted);">No recent exchange bulk deals or corporate actions recorded</li>`;
  }
}

async function loadQuantAndBulkDealsPage() {
  setLoading(true);
  try {
    const [btRes, predRes, ciRes] = await Promise.all([
      fetch("/api/backtest"),
      fetch("/api/predictions"),
      fetch("/api/corporate-intel"),
    ]);

    if (btRes.ok) {
      const bt = await btRes.json();
      renderWalkForwardBacktestTable(bt);
    }
    if (predRes.ok) {
      const pred = await predRes.json();
      renderAllStockPredictionsMatrix(pred);
    }
    if (ciRes.ok) {
      const ci = await ciRes.json();
      renderNseBulkBlockDealsTable(ci);
    }
  } catch (err) {
    console.error("Failed loading Quant & Bulk Deals page:", err);
  } finally {
    setLoading(false);
  }
}

function renderWalkForwardBacktestTable(bt) {
  const kpiEl = document.getElementById("quant-bt-kpis");
  const ens = bt.models?.ml_ensemble || null;
  const niftyRet = bt.nifty_50_buy_hold_return_pct ?? null;

  // Determine best model objectively by Net Sharpe (do not hardcode ml_ensemble as best)
  let bestKey = null;
  let bestSharpe = -Infinity;
  if (bt.models) {
    Object.entries(bt.models).forEach(([k, m]) => {
      const sh = m?.annualized_sharpe_net;
      if (sh !== undefined && sh !== null && sh > bestSharpe) {
        bestSharpe = sh;
        bestKey = k;
      }
    });
  }

  const pillEl = document.getElementById("mr-walkforward-pill");
  if (pillEl && bestKey && bt.models[bestKey]) {
    const bm = bt.models[bestKey];
    pillEl.textContent = `Walk-Forward OOS (${bt.execution_mode || "next_open"}): ${bestKey} Sharpe ${fmtNum(bm.annualized_sharpe_net, 3)} (${fmtPct(bm.net_cumulative_return_pct, 2, true)} Net)`;
  } else if (pillEl) {
    pillEl.textContent = "Walk-Forward OOS: N/A";
  }

  if (kpiEl && ens) {
    const netRet = ens.net_cumulative_return_pct ?? null;
    const alpha = (netRet !== null && niftyRet !== null) ? Number((netRet - niftyRet).toFixed(2)) : null;
    const genStr = bt.generated_at ? String(bt.generated_at).slice(0, 19).replace("T", " ") + " UTC" : "N/A";
    kpiEl.innerHTML = `
      <div class="kpi-box">
        <span class="kpi-label">OOS Test Window</span>
        <span class="kpi-value" style="font-size: 0.82rem;">${bt.test_start_date || "N/A"} → ${bt.test_end_date || "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Folds / Purge / Embargo</span>
        <span class="kpi-value">${bt.total_walk_forward_folds ?? "N/A"} Folds (${bt.purge_gap_days ?? "N/A"}s / ${bt.embargo_sessions ?? "N/A"}s)</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">ML Ensemble Net Return</span>
        <span class="kpi-value ${netRet !== null ? (netRet >= 0 ? "text-bull" : "text-bear") : ""}">${netRet !== null ? fmtPct(netRet, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Nifty 50 Benchmark</span>
        <span class="kpi-value ${niftyRet !== null ? (niftyRet >= 0 ? "text-bull" : "text-bear") : ""}">${niftyRet !== null ? fmtPct(niftyRet, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">ML vs Nifty 50 (Alpha)</span>
        <span class="kpi-value ${alpha !== null ? (alpha >= 0 ? "text-bull" : "text-bear") : ""}">${alpha !== null ? fmtPct(alpha, 2, true) : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">ML Net Sharpe / Expectancy</span>
        <span class="kpi-value ${ens.annualized_sharpe_net >= 0 ? "text-bull" : "text-bear"}">${ens.annualized_sharpe_net != null ? fmtNum(ens.annualized_sharpe_net, 3) : "N/A"} / ${ens.expectancy_per_trade_bps != null ? `${fmtNum(ens.expectancy_per_trade_bps, 1)} bps` : "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Execution Mode</span>
        <span class="kpi-value" style="font-size: 0.78rem;">${bt.execution_mode || "N/A"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Backtest Freshness</span>
        <span class="kpi-value" style="font-size: 0.74rem;">${genStr}</span>
      </div>
    `;
  }

  const tbody = document.getElementById("tbody-quant-models");
  if (!tbody || !bt.models) return;
  const order = ["ml_ensemble", "naive_momentum", "linear_ridge_logistic", "technical_rule"];
  const labels = {
    ml_ensemble: "BharatAlpha Calibrated ML Ensemble (LightGBM + XGBoost + Ridge)",
    naive_momentum: "Baseline 1: Naive Momentum & Trend Rule",
    linear_ridge_logistic: "Baseline 3: Linear Ridge + Logistic L2",
    technical_rule: "Baseline 2: RSI(14) + MACD + 50/200-DMA Rule",
  };
  tbody.innerHTML = order
    .filter((k) => bt.models[k])
    .map((k) => {
      const m = bt.models[k];
      const isBest = k === bestKey;
      const grossRet = m.gross_cumulative_return_pct ?? null;
      const costDrag = m.transaction_cost_drag_pct ?? null;
      const netRet = m.net_cumulative_return_pct ?? null;
      const vsNifty = (netRet !== null && niftyRet !== null) ? Number((netRet - niftyRet).toFixed(2)) : null;
      const tradeFreq = m.abstention_rate_pct != null ? Number((100 - m.abstention_rate_pct).toFixed(1)) : null;
      const labelTxt = `${isBest ? "★ " : ""}${labels[k] || k}`;
      return `
        <tr style="${isBest ? "background: rgba(34, 197, 94, 0.07);" : ""}">
          <td style="text-align: left; font-weight: ${isBest ? "800" : "600"}; color: ${isBest ? "var(--signal-bull)" : "var(--text-primary)"};">${labelTxt}</td>
          <td>${m.active_trades_count ?? "N/A"} (${tradeFreq !== null ? fmtPct(tradeFreq, 1) : "N/A"})</td>
          <td>${m.directional_accuracy_all_pct != null ? fmtPct(m.directional_accuracy_all_pct, 2) : "N/A"}</td>
          <td>${m.high_conviction_hit_rate_pct != null ? fmtPct(m.high_conviction_hit_rate_pct, 2) : "N/A"}</td>
          <td class="${grossRet !== null ? (grossRet >= 0 ? "text-bull" : "text-bear") : ""}">${grossRet !== null ? fmtPct(grossRet, 2, true) : "N/A"}</td>
          <td class="text-bear">${costDrag !== null ? `-${fmtPct(costDrag, 2)}` : "N/A"}</td>
          <td class="${netRet !== null ? (netRet >= 0 ? "text-bull" : "text-bear") : ""}" style="font-weight: 800;">${netRet !== null ? fmtPct(netRet, 2, true) : "N/A"}</td>
          <td class="${m.annualized_sharpe_net != null ? (m.annualized_sharpe_net >= 0 ? "text-bull" : "text-bear") : ""}" style="font-weight: 700;">${m.annualized_sharpe_net != null ? fmtNum(m.annualized_sharpe_net, 3) : "N/A"}</td>
          <td>${m.sortino_ratio_net != null ? fmtNum(m.sortino_ratio_net, 3) : "N/A"}</td>
          <td class="text-bear">${m.max_drawdown_net_pct != null ? fmtPct(m.max_drawdown_net_pct, 2) : "N/A"}</td>
          <td class="${vsNifty !== null ? (vsNifty >= 0 ? "text-bull" : "text-bear") : ""}">${vsNifty !== null ? fmtPct(vsNifty, 2, true) : "N/A"}</td>
          <td>${m.brier_score != null ? fmtNum(m.brier_score, 4) : "N/A"} / ${m.roc_auc != null ? fmtNum(m.roc_auc, 3) : "N/A"}</td>
        </tr>
      `;
    })
    .join("");
}

function renderAllStockPredictionsMatrix(predData) {
  const freshEl = document.getElementById("quant-pred-freshness");
  const items = predData.predictions || [];
  if (freshEl) {
    const genStr = predData.generated_at ? String(predData.generated_at).slice(0, 19).replace("T", " ") + " UTC" : "N/A";
    const targetStr = items[0]?.target_session_date || "N/A";
    freshEl.textContent = `Generated: ${genStr} • Target Session: ${targetStr} • ${items.length} NSE Equities Ranked by Calibrated P(Up 1D)`;
  }

  const tbody = document.getElementById("tbody-quant-predictions");
  if (!tbody) return;
  tbody.innerHTML = items
    .map((p) => {
      const actStr = String(p.recommended_action || "N/A");
      const actCls = actStr.includes("BUY")
        ? "signal-banner bull"
        : actStr.includes("SELL") || actStr.includes("AVOID")
        ? "signal-banner bear"
        : "signal-banner hold";
      const p1d = p.prob_next_day_up ?? null;
      const pLgbm = p.prob_lgbm_up ?? null;
      const gross1d = p.expected_next_day_return_pct ?? null;
      const net1d = p.net_expected_1d_return_pct ?? null;
      const cncBps = p.roundtrip_delivery_cost_pct != null ? p.roundtrip_delivery_cost_pct * 100 : null;
      const conf = p.confidence_score ?? null;
      const bullTxt = (p.top_bullish_drivers || []).slice(0, 2).map(formatShapDriverItem).join(" • ");
      const bearTxt = (p.top_bearish_drivers || []).slice(0, 2).map(formatShapDriverItem).join(" • ");
      return `
        <tr>
          <td style="text-align: left;">
            <strong>${p.symbol || "N/A"}</strong>
            <span style="display: block; font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-sans);">Regime: ${p.market_regime || "N/A"}</span>
          </td>
          <td>${p.observation_date || "N/A"}</td>
          <td><span class="${actCls}" style="font-size: 0.72rem; padding: 0.18rem 0.5rem;">${actStr}</span></td>
          <td class="${p1d !== null && p1d >= 0.53 ? "text-bull" : ""}" style="font-weight: 700;">${p1d !== null ? fmtPct(p1d * 100, 1) : "N/A"}</td>
          <td>${pLgbm !== null ? fmtPct(pLgbm * 100, 1) : "N/A"}</td>
          <td class="${gross1d !== null ? (gross1d >= 0 ? "text-bull" : "text-bear") : ""}">${gross1d !== null ? fmtPct(gross1d, 2, true) : "N/A"}</td>
          <td class="text-bear">${cncBps !== null ? `-${fmtNum(cncBps, 1)} bps` : "N/A"}</td>
          <td class="${net1d !== null ? (net1d >= 0 ? "text-bull" : "text-bear") : ""}" style="font-weight: 800;">${net1d !== null ? fmtPct(net1d, 2, true) : "N/A"}</td>
          <td>${conf !== null ? fmtNum(conf, 1) : "N/A"}</td>
          <td style="text-align: left; font-size: 0.74rem; max-width: 360px;">
            ${bullTxt ? `<div class="text-bull">▲ ${bullTxt}</div>` : ""}
            ${bearTxt ? `<div class="text-bear">▼ ${bearTxt}</div>` : ""}
          </td>
          <td>
            <button class="btn btn-primary btn-quant-inspect" data-sym="${p.symbol}" style="padding: 0.28rem 0.6rem; font-size: 0.73rem;">
              Inspect
            </button>
          </td>
        </tr>
      `;
    })
    .join("");

  tbody.querySelectorAll(".btn-quant-inspect").forEach((btn) => {
    btn.addEventListener("click", () => {
      addStockToWatchlist(btn.getAttribute("data-sym"), btn.getAttribute("data-sym"));
    });
  });
}

function renderFiveDayOutlookAndConditions(predData) {
  if (!predData) return;
  state.latestPredData = predData;
  const tbody = document.getElementById("tbody-5d-outlook");
  const subEl = document.getElementById("outlook-5d-sub");
  const rawItems = (predData && Array.isArray(predData.predictions)) ? predData.predictions : [];
  const watchlistSyms = new Set((state.watchlist || []).map((w) => String(w.symbol).toUpperCase()));
  const items = rawItems.filter((p) => watchlistSyms.has(String(p.symbol || "").toUpperCase()));

  if (subEl) {
    const genStr = predData?.generated_at ? String(predData.generated_at).slice(0, 19).replace("T", " ") + " UTC" : "N/A";
    const regStr = predData?.market_regime || items[0]?.market_regime || "N/A";
    subEl.textContent = `Showing ${items.length} Dashboard Stocks • Regime: ${regStr} • Generated: ${genStr} • Anchored on Validated Naive Momentum Benchmark + ATR(14) Bands (Not Investment Advice)`;
  }

  if (tbody) {
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 1.5rem; color: var(--text-muted);">No prediction records available.</td></tr>`;
    } else {
      const sorted = [...items].sort((a, b) => (b.momentum_5d_return_pct ?? -999) - (a.momentum_5d_return_pct ?? -999));
      tbody.innerHTML = sorted
        .map((p) => {
          const close = p.last_close ?? null;
          const atr = p.atr_14_pct ?? null;
          const mom5 = p.momentum_5d_return_pct ?? null;
          const ml5 = p.expected_5d_return_pct ?? null;
          const actStr = String(p.recommended_action || "N/A");

          let momTxt = mom5 !== null ? fmtPct(mom5, 2, true) : "N/A";
          let mlTxt = ml5 !== null ? fmtPct(ml5, 2, true) : "N/A";
          let rangeTxt = "N/A";
          let buyTxt = "N/A";
          let sellTxt = "N/A";

          if (close !== null && atr !== null) {
            const sigma5d = close * (atr / 100.0) * Math.sqrt(5) * 0.65;
            if (mom5 !== null) {
              const midMom = close * (1.0 + mom5 / 100.0);
              momTxt = `₹${fmtNum(midMom, 2)} (${fmtPct(mom5, 2, true)})`;
              rangeTxt = `₹${fmtNum(midMom - sigma5d, 2)} – ₹${fmtNum(midMom + sigma5d, 2)}`;
            }
            if (ml5 !== null) {
              const midMl = close * (1.0 + ml5 / 100.0);
              mlTxt = `₹${fmtNum(midMl, 2)} (${fmtPct(ml5, 2, true)})`;
            }
            const buyLo = close * (1.0 - 0.5 * atr / 100.0);
            const buyHi = close * (1.0 + 0.25 * atr / 100.0);
            const sl = close * (1.0 - 1.5 * atr / 100.0);
            const sellLo = close * (1.0 + 1.5 * atr / 100.0);
            const sellHi = close * (1.0 + 2.5 * atr / 100.0);
            buyTxt = mom5 !== null && mom5 < 0
              ? `Avoid new long [SL ₹${fmtNum(sl, 2)}]`
              : `₹${fmtNum(buyLo, 2)} – ₹${fmtNum(buyHi, 2)} [SL ₹${fmtNum(sl, 2)}]`;
            sellTxt = `₹${fmtNum(sellLo, 2)} – ₹${fmtNum(sellHi, 2)}`;
          }

          return `
            <tr>
              <td style="text-align: left;"><strong>${p.symbol || "N/A"}</strong></td>
              <td>${close !== null ? `₹${fmtNum(close, 2)}` : "N/A"}</td>
              <td>${atr !== null ? fmtPct(atr, 2) : "N/A"}</td>
              <td class="${mom5 !== null ? (mom5 >= 0 ? "text-bull" : "text-bear") : ""}" style="font-weight: 700;">${momTxt}</td>
              <td class="${ml5 !== null ? (ml5 >= 0 ? "text-bull" : "text-bear") : ""}">${mlTxt}</td>
              <td>${rangeTxt}</td>
              <td>${buyTxt}</td>
              <td class="text-bull">${sellTxt}</td>
              <td>${actStr}</td>
            </tr>
          `;
        })
        .join("");
    }
  }

  // Populate live 5D upside & downside drivers from TreeSHAP attributions
  const upEl = document.getElementById("cond-5d-up");
  const downEl = document.getElementById("cond-5d-down");
  const bullCounts = {};
  const bearCounts = {};
  items.forEach((p) => {
    (p.top_bullish_drivers || []).slice(0, 2).forEach((d) => {
      if (d?.feature) bullCounts[d.feature] = (bullCounts[d.feature] || 0) + 1;
    });
    (p.top_bearish_drivers || []).slice(0, 2).forEach((d) => {
      if (d?.feature) bearCounts[d.feature] = (bearCounts[d.feature] || 0) + 1;
    });
  });
  const topBull = Object.entries(bullCounts).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([f, c]) => `<strong>${f}</strong> (${c}/${items.length} stocks positive)`).join(", ");
  const topBear = Object.entries(bearCounts).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([f, c]) => `<strong>${f}</strong> (${c}/${items.length} stocks negative)`).join(", ");

  if (upEl) {
    upEl.innerHTML = `
      <li>Live positive drivers: ${topBull || "N/A"}.</li>
      <li>NSE Delivery % rising above 20D average on up-sessions, with clean 09:15 IST opens (gap &le; +2.5%).</li>
      <li>NSE Advance/Decline breadth improving above 1.0x with India VIX &lt; 18.</li>
    `;
  }
  if (downEl) {
    downEl.innerHTML = `
      <li>Live negative drivers: ${topBear || "N/A"}.</li>
      <li>Overnight gap-downs &lt; -3.0% or India VIX spiking &ge; 21.0 (triggers High-Volatility regime block).</li>
      <li>Indian CNC round-trip cost drag (~35 bps) eroding low-conviction short-term bounces.</li>
    `;
  }
}

function renderNseBulkBlockDealsTable(ciData) {
  const sub = document.getElementById("quant-deals-subtitle");
  if (sub && ciData.total_exchange_bulk_block_deals) {
    sub.textContent = `${ciData.total_exchange_bulk_block_deals} Verified NSE Bulk & Block Deals Captured from Official Exchange Archives — Showing Top 25 by Value (₹ Cr)`;
  }
  const tbody = document.getElementById("tbody-nse-bulk-deals");
  if (!tbody) return;
  const deals = ciData.top_exchange_bulk_block_deals || [];
  tbody.innerHTML = deals
    .map((d) => {
      const isBuy = d.side === "BUY";
      return `
        <tr>
          <td>${d.deal_date}</td>
          <td><span class="badge-neutral">${d.deal_type}</span></td>
          <td style="text-align: left;">
            <strong>${d.symbol}</strong>
            <span style="display: block; font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-sans);">${d.security_name}</span>
          </td>
          <td style="text-align: left; font-family: var(--font-sans);">${d.client_name}</td>
          <td><span class="signal-banner ${isBuy ? "bull" : "bear"}" style="font-size: 0.72rem; padding: 0.15rem 0.5rem;">${d.side}</span></td>
          <td>${Number(d.quantity).toLocaleString("en-IN")}</td>
          <td>₹${fmtNum(d.price_inr, 2)}</td>
          <td class="${isBuy ? "text-bull" : "text-bear"}" style="font-weight: 800;">₹${fmtNum(d.value_cr, 2)} Cr</td>
        </tr>
      `;
    })
    .join("");
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

/* ============================================================================
   Phase 5 Multi-Dimensional Stock Intelligence & Controlled Experiments
   ============================================================================ */

async function loadPhase5StockIntelligence(refresh = false) {
  setLoading(true);
  try {
    const [intelRes, expRes, fvRes] = await Promise.allSettled([
      fetch(`/api/stock-intelligence?refresh=${refresh ? "true" : "false"}`),
      fetch("/api/phase5-experiments"),
      fetch(`/api/forward-validation?refresh=${refresh ? "true" : "false"}`),
    ]);
    if (intelRes.status === "fulfilled" && intelRes.value.ok) {
      const data = await intelRes.value.json();
      renderPhase5Intelligence(data);
    }
    if (expRes.status === "fulfilled" && expRes.value.ok) {
      const exp = await expRes.value.json();
      renderPhase5ExperimentsTable(exp);
    }
    if (fvRes.status === "fulfilled" && fvRes.value.ok) {
      const fv = await fvRes.value.json();
      renderForwardValidation(fv);
    }
  } catch (err) {
    console.error("Failed loading Phase 5 Stock Intelligence:", err);
  } finally {
    setLoading(false);
  }
}

function renderPhase5Intelligence(data) {
  const sub = document.getElementById("p5-freshness-sub");
  if (sub) {
    sub.textContent = `Model: ${data.model_version || "5.0.0-pilot"} • News Collected: ${String(data.last_news_collection_at || "").slice(0, 19).replace("T", " ")} IST • Quant Updated: ${String(data.last_quant_update_at || "").slice(0, 19).replace("T", " ")} • Pilot Universe: ${(data.pilot_universe || []).join(", ")}`;
  }

  const warnEl = document.getElementById("p5-warnings-banner");
  const allWarnings = [...(data.data_quality_warnings || []), ...(data.monitoring?.failure_alerts || [])];
  if (warnEl) {
    warnEl.innerHTML = allWarnings.length
      ? `<strong>Data Quality &amp; Monitoring Notices:</strong> ${allWarnings.join(" | ")}`
      : `<span class="text-bull">All 5 pilot stock intelligence feeds active. No failure alerts.</span>`;
  }

  const monKpi = document.getElementById("p5-monitoring-kpis");
  const mon = data.monitoring || {};
  if (monKpi) {
    const hm = mon.horizon_metrics || {};
    const hmBadges = ["1D", "5D", "10D"]
      .filter((h) => hm[h])
      .map((h) => {
        const m = hm[h];
        const accTxt = m.directional_accuracy_pct != null ? fmtPct(m.directional_accuracy_pct, 1) : "Pending";
        const aucTxt = m.roc_auc != null ? fmtNum(m.roc_auc, 3) : "—";
        const icTxt = m.rank_ic != null ? fmtNum(m.rank_ic, 3) : "—";
        const covTxt = m.prediction_coverage_pct != null ? fmtPct(m.prediction_coverage_pct, 0) : "—";
        const benchTxt = m.model_vs_benchmark_accuracy_diff_pct != null ? `${m.model_vs_benchmark_accuracy_diff_pct >= 0 ? "+" : ""}${m.model_vs_benchmark_accuracy_diff_pct}%` : "—";
        return `<span class="badge-neutral mono" style="font-size: 0.7rem;"><strong>${h}</strong>: ${m.status || "INSUFFICIENT_DATA"} • Cov ${covTxt} • Acc ${accTxt} • AUC ${aucTxt} • IC ${icTxt} • vsBench ${benchTxt} • ${m.data_freshness || "—"}</span>`;
      })
      .join(" ");
    monKpi.innerHTML = `
      <div class="kpi-box">
        <span class="kpi-label">Logged Predictions</span>
        <span class="kpi-value">${mon.total_logged_predictions ?? 0}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Evaluated Outcomes</span>
        <span class="kpi-value">${mon.evaluated_outcomes_count ?? 0}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Rolling 1D Accuracy</span>
        <span class="kpi-value">${mon.rolling_directional_accuracy_pct != null ? fmtPct(mon.rolling_directional_accuracy_pct, 1) : "Pending"}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Recent Brier / Drift</span>
        <span class="kpi-value">${mon.brier_score_recent != null ? fmtNum(mon.brier_score_recent, 4) : "—"} (${mon.drift_status || "INSUFFICIENT_DATA"})</span>
      </div>
      ${hmBadges ? `<div style="grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.25rem;">${hmBadges}</div>` : ""}
    `;
  }

  const grid = document.getElementById("p5-stocks-grid");
  const stocks = data.stocks || {};
  if (grid) {
    grid.innerHTML = Object.values(stocks)
      .map((s) => {
        const outlooks = s.horizon_outlooks || {};
        const hRows = ["1D", "5D", "10D"]
          .filter((h) => outlooks[h])
          .map((h) => {
            const o = outlooks[h];
            const isGen = Boolean(o.is_model_generated);
            const decCls = o.decision === "BUY" ? "text-bull" : (o.decision === "NO TRADE" || o.decision === "UNAVAILABLE") ? "text-bear" : "text-hold";
            const qProbTxt = o.quant_prob_positive != null ? fmtPct(o.quant_prob_positive * 100, 1) : "Unavailable";
            const retTxt = (o.expected_gross_return_pct != null && o.expected_net_return_pct != null)
              ? `${fmtPct(o.expected_gross_return_pct, 2, true)} / ${fmtPct(o.expected_net_return_pct, 2, true)}`
              : "Unavailable";
            const decTxt = o.prob_positive != null ? `${o.decision} (${fmtPct(o.prob_positive * 100, 1)})` : o.decision;
            const srcTag = isGen ? `<span class="text-bull">Model</span>` : `<span class="text-bear">Unavailable</span>`;
            const metaTxt = `${o.model_prediction_date || s.observation_date || "—"} • ${o.data_freshness || "—"}`;
            return `
              <tr title="${o.status_reason || ""}">
                <td><strong>${h}</strong><div style="font-size:0.65rem;">${srcTag}</div></td>
                <td>${qProbTxt}<div style="font-size:0.65rem;color:var(--text-muted);">${metaTxt}</div></td>
                <td>${o.news_signal}</td>
                <td>${o.cross_stock_signal}</td>
                <td>${isGen ? fmtPct(o.confidence, 0) : "—"}</td>
                <td class="${decCls}" style="font-weight: 700;">${decTxt}</td>
                <td class="${(o.expected_net_return_pct ?? 0) >= 0 ? "text-bull" : "text-bear"}">${retTxt}</td>
              </tr>
            `;
          })
          .join("");
        const crossTxt = (s.cross_stock_impacts || [])
          .slice(0, 2)
          .map((c) => `<li><strong>[${c.relationship_type}]</strong> ${c.originating_entity}: ${c.expected_impact_direction.toUpperCase()} — ${c.rationale}</li>`)
          .join("");
        return `
          <div class="sub-panel">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.45rem;">
              <div>
                <h4 style="font-size: 0.98rem; font-weight: 800;">${s.symbol} <span style="font-weight: 400; font-size: 0.78rem; color: var(--text-muted);">(${s.sector})</span></h4>
                <div style="font-size: 0.75rem; color: var(--text-muted);">Industry: <strong>${s.industry_outlook}</strong> • News Sent: <strong>${fmtNum(s.news_sentiment_score, 3)}</strong></div>
              </div>
              <span class="badge-neutral mono">${s.quant_score_summary?.recommended_action || "WATCH"}</span>
            </div>
            <div class="table-wrap" style="margin-bottom: 0.55rem;">
              <table class="fin-table mono" style="font-size: 0.71rem;">
                <thead><tr><th>Hor</th><th>Quant P(Up) / Date</th><th>News</th><th>Cross</th><th>Conf</th><th>Decision</th><th>Gross / Net E[R]</th></tr></thead>
                <tbody>${hRows}</tbody>
              </table>
            </div>
            <div style="font-size: 0.75rem; line-height: 1.45;">
              <div><strong>Promoter/Insider:</strong> ${(s.promoter_intelligence || [])[0] || "—"}</div>
              <div><strong>Competitors:</strong> ${(s.competitor_developments || [])[0] || "—"}</div>
              <div><strong>Regulatory:</strong> ${(s.regulatory_developments || [])[0] || "—"}</div>
              <div><strong>Global Macro:</strong> ${(s.global_exposure || [])[0] || "—"}</div>
              ${crossTxt ? `<ul class="agent-reasons" style="margin-top: 0.35rem;">${crossTxt}</ul>` : ""}
            </div>
          </div>
        `;
      })
      .join("");
  }

  const feedBody = document.getElementById("tbody-p5-feed");
  if (feedBody) {
    const feed = data.consolidated_feed || [];
    feedBody.innerHTML = feed.length
      ? feed
          .map((e) => {
            const sCls = e.sentiment === "positive" ? "text-bull" : e.sentiment === "negative" ? "text-bear" : "";
            const linkHtml = e.source_url
              ? `<a href="${e.source_url}" target="_blank" rel="noopener" style="color: var(--accent-brand); text-decoration: none;">${e.event_description}</a>`
              : e.event_description;
            return `
              <tr>
                <td><strong>${e.ticker}</strong></td>
                <td><span class="badge-neutral">${e.event_category}</span></td>
                <td style="text-align: left;">${linkHtml} <span style="font-size: 0.72rem; color: var(--text-muted);">(${e.original_source})</span></td>
                <td class="mono">${e.effective_trading_date || "—"}</td>
                <td class="${sCls}" style="font-weight: 700;">${e.sentiment.toUpperCase()}</td>
                <td class="mono">${e.impact_magnitude.toUpperCase()}</td>
              </tr>
            `;
          })
          .join("")
      : `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No live news events collected.</td></tr>`;
  }
}

function renderPhase5ExperimentsTable(expData) {
  const tbody = document.getElementById("tbody-p5-experiments");
  if (!tbody || !expData?.experiments) return;
  const rows = Object.entries(expData.experiments).map(([name, r]) => {
    const d1 = r["1D"] || {};
    const d5 = r["5D"] || {};
    const d10 = r["10D"] || {};
    return `
      <tr>
        <td style="text-align: left;"><strong>${name}</strong></td>
        <td>${r.feature_count ?? "—"}</td>
        <td>${d1.roc_auc != null ? fmtNum(d1.roc_auc, 4) : "—"}</td>
        <td class="${(d1.net_return_pct || 0) >= 0 ? "text-bull" : "text-bear"}">${d1.net_return_pct != null ? fmtPct(d1.net_return_pct, 2, true) : "—"}</td>
        <td>${d1.net_sharpe != null ? fmtNum(d1.net_sharpe, 3) : "—"}</td>
        <td>${d5.roc_auc != null ? fmtNum(d5.roc_auc, 4) : "—"}</td>
        <td>${d5.rank_ic != null ? fmtNum(d5.rank_ic, 4) : "—"}</td>
        <td>${d10.roc_auc != null ? fmtNum(d10.roc_auc, 4) : "—"}</td>
        <td>${d10.rank_ic != null ? fmtNum(d10.rank_ic, 4) : "—"}</td>
      </tr>
    `;
  });
  const nm = expData.naive_momentum_benchmark || {};
  if (nm["1D"]) {
    rows.push(`
      <tr style="background: rgba(56, 189, 248, 0.08);">
        <td style="text-align: left;"><strong>Benchmark: naive_momentum</strong></td>
        <td>Rule</td>
        <td>${fmtNum(nm["1D"].roc_auc, 4)}</td>
        <td class="${(nm["1D"].net_return_pct || 0) >= 0 ? "text-bull" : "text-bear"}">${fmtPct(nm["1D"].net_return_pct, 2, true)}</td>
        <td>${fmtNum(nm["1D"].net_sharpe, 3)}</td>
        <td>${nm["5D"] ? fmtNum(nm["5D"].roc_auc, 4) : "—"}</td>
        <td>${nm["5D"] ? fmtNum(nm["5D"].rank_ic, 4) : "—"}</td>
        <td>${nm["10D"] ? fmtNum(nm["10D"].roc_auc, 4) : "—"}</td>
        <td>${nm["10D"] ? fmtNum(nm["10D"].rank_ic, 4) : "—"}</td>
      </tr>
    `);
  }
  tbody.innerHTML = rows.join("");
}

function renderForwardValidation(fv) {
  if (!fv) return;
  const sub = document.getElementById("fv-subtitle");
  if (sub) {
    const ts = fv.last_news_collection_at ? String(fv.last_news_collection_at).slice(0, 19).replace("T", " ") + " IST" : "Not collected yet";
    sub.textContent = `Model ${fv.model_version || "5.3.0-pilot"} (${fv.feature_set_version || "237+17"}) • Last Forward News: ${ts}`;
  }
  const distEl = document.getElementById("fv-distinction-note");
  if (distEl && fv.distinction_note) {
    distEl.innerHTML = `<strong>Evidence Separation:</strong> ${fv.distinction_note}`;
  }
  const warnEl = document.getElementById("fv-warnings-banner");
  const notices = [...(fv.data_quality_warnings || []), ...(fv.monitoring_alerts || [])];
  if (warnEl) {
    warnEl.innerHTML = notices.length
      ? `<strong>Forward Validation Notices:</strong> ${notices.join(" | ")}`
      : `<span class="text-bull">Forward news collection, snapshot logging, and outcome tracking healthy.</span>`;
  }
  const kpiEl = document.getElementById("fv-kpis");
  if (kpiEl) {
    kpiEl.innerHTML = `
      <div class="kpi-box">
        <span class="kpi-label">Real News (Valid / Quarantined)</span>
        <span class="kpi-value">${fv.total_real_news_events ?? 0} (${fv.valid_news_events ?? 0} / ${fv.quarantined_news_events ?? 0})</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Immutable Snapshots</span>
        <span class="kpi-value">${fv.total_immutable_snapshots ?? 0}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Matured / Pending Outcomes</span>
        <span class="kpi-value">${fv.matured_outcomes_total ?? 0} / ${fv.pending_outcomes_total ?? 0}</span>
      </div>
      <div class="kpi-box">
        <span class="kpi-label">Last News Collection</span>
        <span class="kpi-value" style="font-size: 0.8rem;">${fv.last_news_collection_at ? String(fv.last_news_collection_at).slice(0, 16).replace("T", " ") : "Pending"}</span>
      </div>
    `;
  }

  const hBody = document.getElementById("tbody-fv-horizons");
  if (hBody && fv.horizon_evaluations) {
    hBody.innerHTML = ["1D", "5D", "10D"]
      .filter((h) => fv.horizon_evaluations[h])
      .map((h) => {
        const m = fv.horizon_evaluations[h];
        const stCls = m.status === "OK" ? "text-bull" : m.status === "WARNING" ? "text-bear" : "text-hold";
        return `
          <tr>
            <td style="text-align: left;"><strong>${h}</strong></td>
            <td>${m.model_version || "5.3.0-pilot"}</td>
            <td>${fmtPct(m.prediction_coverage_pct ?? 0, 1)}</td>
            <td>${m.matured_count ?? 0} / ${m.pending_count ?? 0}</td>
            <td class="${stCls}" style="font-weight: 700;">${m.status}</td>
            <td>${m.directional_accuracy_pct != null ? fmtPct(m.directional_accuracy_pct, 1) : "—"}</td>
            <td>${m.roc_auc != null ? fmtNum(m.roc_auc, 4) : "—"}</td>
            <td>${m.brier_score != null ? fmtNum(m.brier_score, 4) : "—"}</td>
            <td>${m.rank_ic != null ? fmtNum(m.rank_ic, 4) : "—"}</td>
            <td class="${(m.net_return_pct ?? 0) >= 0 ? "text-bull" : "text-bear"}">${m.net_return_pct != null ? fmtPct(m.net_return_pct, 2, true) : "—"}</td>
            <td>${m.vs_naive_momentum_net_diff_pct != null ? fmtPct(m.vs_naive_momentum_net_diff_pct, 2, true) : "—"}</td>
          </tr>
        `;
      })
      .join("");
  }

  const pBody = document.getElementById("tbody-fv-paper");
  if (pBody && fv.paper_portfolio) {
    const pp = fv.paper_portfolio;
    const rows = ["1D", "5D", "10D"]
      .filter((h) => pp.strategies && pp.strategies[h])
      .map((h) => {
        const s = pp.strategies[h];
        return `
          <tr>
            <td style="text-align: left;"><strong>Paper ML ${h}</strong></td>
            <td>${s.status}</td>
            <td>₹${Number(s.ending_capital_inr || 1000000).toLocaleString("en-IN")}</td>
            <td class="${(s.net_return_pct || 0) >= 0 ? "text-bull" : "text-bear"}">${fmtPct(s.net_return_pct ?? 0, 2, true)}</td>
            <td class="text-bear">${fmtPct(s.max_drawdown_pct ?? 0, 2)}</td>
            <td>${fmtPct(s.peak_exposure_pct ?? 0, 1)}</td>
            <td>${s.executed_trades ?? 0} / ${s.rejected_signals ?? 0}</td>
            <td>${s.win_rate_pct != null ? fmtPct(s.win_rate_pct, 1) : "—"}</td>
          </tr>
        `;
      });
    const bm = pp.benchmarks || {};
    for (const [bName, bVal] of Object.entries(bm)) {
      rows.push(`
        <tr style="background: rgba(56, 189, 248, 0.06);">
          <td style="text-align: left;"><strong>Benchmark: ${bName}</strong></td>
          <td>${bVal.status || "ACTIVE"}</td>
          <td>—</td>
          <td class="${(bVal.net_return_pct || 0) >= 0 ? "text-bull" : "text-bear"}">${fmtPct(bVal.net_return_pct ?? 0, 2, true)}</td>
          <td>—</td><td>—</td><td>—</td><td>—</td>
        </tr>
      `);
    }
    pBody.innerHTML = rows.join("");
  }
}

/* ============================================================================
   Beginner-Friendly Top-Right '?' Hover Tooltips (All Charts & Start/Sections)
   ============================================================================ */

const BEGINNER_HELP_GUIDES = {
  "market-regime-strip": {
    title: "Market Regime & Breadth Ribbon (Start Here)",
    what: "A live weather report for the overall Indian stock market (Nifty 50, India VIX fear gauge, and how many NSE stocks are rising vs falling).",
    read: "Strong/Weak Bull = market trend is healthy; Sideways = choppy; Bear or High Volatility (VIX >= 21) = risky environment.",
    use: "Even good stocks struggle when the market is in Bear or High Volatility mode. Trade smaller or wait when the regime warns of high risk.",
  },
  "sec-overview": {
    title: "01. Company Overview & Next-Day Execution Ticket",
    what: "A quick snapshot of the company's size, valuation (P/E), financial health (Piotroski F-Score out of 9), and tomorrow's suggested price levels.",
    read: "Entry Zone = ideal buy price range; Target 1/2 = profit-taking levels; Stop-Loss = price where you exit to limit losses; R:R = potential reward vs risk.",
    use: "Beginners should look for Piotroski >= 6, Risk:Reward >= 1.5x, and never enter a trade without knowing the Stop-Loss first.",
  },
  "sec-quant-intel": {
    title: "02. Quantitative ML Prediction & Smart-Money Tape",
    what: "Left side shows our walk-forward ML probability for tomorrow after Indian brokerage/STT costs. Right side shows institutional delivery % and bulk/block deals.",
    read: "P(Up 1D) > 53% with positive Net E[R] = mild bullish edge; Delivery % > 50% means buyers are holding shares rather than day-trading.",
    use: "Use this as a reality check: if Net E[R] after Indian taxes/fees is negative, avoid short-term trades.",
  },
  "sec-committee": {
    title: "03. 5-Agent Investment Committee & Technical Chart",
    what: "Five specialized analysts (Value, Growth, Quality, Momentum, Risk) vote Bullish, Neutral, or Bearish on the stock.",
    read: "Check if both fundamental agents (Value/Quality) and technical agents (Momentum) agree, or if there is a conflict.",
    use: "Strongest setups happen when 4+ agents agree; if Risk warns of overvaluation or high volatility, be patient.",
  },
  "price-dma-canvas": {
    title: "Chart: 1-Year Price vs. 50-DMA & 200-DMA",
    what: "Shows the daily closing price (green), 50-day moving average (blue), and 200-day moving average (amber), plus T1 Target and Stop-Loss lines.",
    read: "Price above both blue (50-DMA) and amber (200-DMA) lines = healthy uptrend. Price below both = downtrend.",
    use: "Beginners often avoid buying when price is below a falling 200-DMA, and use the Stop-Loss (SL) line as a safety floor.",
  },
  "sec-reports": {
    title: "04. Financial Results (Annual & Quarterly)",
    what: "Tracks whether the company is growing its sales (Revenue) and keeping more profit (Net Profit & Operating Margin %) over 5 years and 5 quarters.",
    read: "Look for steady upward Revenue & Net Profit growth and stable or rising OPM%.",
    use: "Avoid companies with shrinking sales or sudden profit drops unless there is a clear turnaround.",
  },
  "reports-bar-canvas": {
    title: "Chart: 5-Year Annual Revenue vs. Net Profit",
    what: "Compares total yearly sales (blue bars) against yearly net profit (green bars) over the last 5 financial years.",
    read: "Taller bars moving left-to-right mean the business is expanding. Green bars growing alongside blue bars mean profitable growth.",
    use: "Prefer companies where both Revenue and Net Profit trend upward over 5 years.",
  },
  "quarterly-bar-canvas": {
    title: "Chart: Last 5 Quarterly Revenue vs. Net Profit",
    what: "Shows short-term business momentum across the most recent 5 quarters.",
    read: "Compare the latest quarter to the same quarter last year (YoY) and the previous quarter (QoQ) to spot acceleration or slowdown.",
    use: "Rising quarterly profit often acts as a catalyst for stock prices over the next 1–3 months.",
  },
  "sec-cash": {
    title: "05. Cash in Hand, Debt & Cash Flow",
    what: "Shows how much cash the company holds vs how much money it owes (borrowings), plus whether profits turn into real cash (Operating Cash Flow).",
    read: "Cash > Debt (Net Cash positive) is very safe. OCF/Net Profit >= 0.8x means profits are real cash, not just accounting numbers.",
    use: "Companies with low debt and strong cash flow survive market downturns much better.",
  },
  "cash-debt-canvas": {
    title: "Chart: 5-Year Cash in Hand vs. Total Borrowings",
    what: "Compares liquid cash reserves (green) against total debt (red) over 5 years.",
    read: "Green bars taller than red bars = conservative, low-debt balance sheet. Rising red bars with shrinking green bars = rising debt risk.",
    use: "For long-term investing, favor stocks where cash is growing or debt is steadily shrinking.",
  },
  "sec-stockholders": {
    title: "06. Shareholding Pattern (Promoters, FIIs, DIIs)",
    what: "Shows who owns the company: Promoters (founders), FIIs (foreign funds), DIIs (Indian mutual funds/banks), and Public.",
    read: "High or rising Promoter & Institutional (FII+DII) ownership generally signals confidence; promoter share pledging is a caution flag.",
    use: "Watch for steady FII/DII accumulation and zero or decreasing promoter pledges.",
  },
  "shareholding-canvas": {
    title: "Chart: Quarterly Shareholding Trend",
    what: "Tracks how ownership % among Promoters, FIIs, DIIs, and Public has shifted over recent quarters.",
    read: "Rising FII/DII lines with falling Public % often means 'smart money' is accumulating shares from retail sellers.",
    use: "Be cautious if Promoters are rapidly selling their stake quarter after quarter.",
  },
  "sec-rd": {
    title: "07. R&D & Innovation Spend",
    what: "Measures how much the company invests in Research & Development to build future products and stay ahead of competitors.",
    read: "R&D % of Sales shows innovation intensity compared to industry peers.",
    use: "In tech, pharma, and engineering (like Tata Elxsi or TCS), consistent R&D protects profit margins long-term.",
  },
  "rd-chart-canvas": {
    title: "Chart: 5-Year R&D Spend & % of Sales",
    what: "Blue bars show total ₹ Crore spent on R&D; the green line shows R&D as a percentage of total revenue.",
    read: "A rising or steady green line alongside growing revenue means the company is reinvesting for future growth.",
    use: "Use this to compare innovation commitment across tech/manufacturing peers.",
  },
  "sec-promoters-competitors": {
    title: "08. Promoter, Competitor & News Intelligence",
    what: "Summarizes key leadership, recent news headlines, and how direct competitors are performing.",
    read: "Bullish/Bearish tags classify whether news helps or hurts the stock; competitor read-across shows industry tailwinds or threats.",
    use: "Always check recent headlines before buying so you don't get surprised by earnings, lawsuits, or regulatory news.",
  },
  "sec-p5-overview": {
    title: "Phase 5 Multi-Dimensional Stock Intelligence",
    what: "Combines quantitative ML signals, 6 categories of live news (company, promoter, industry, competitor, regulatory, global macro), and cross-stock relationships across 1D, 5D, and 10D.",
    read: "BUY/HOLD/WATCH/NO TRADE decisions show whether Quant, News, and Cross-Stock signals agree. Facts are separated from model predictions.",
    use: "Look for alignment across Quant + News + Cross-Stock before acting, and treat this strictly as research/paper-trading guidance.",
  },
  "sec-5d-outlook": {
    title: "5-Day Price Forecast, Buy/Sell Ranges & Conditions",
    what: "Shows estimated 5-day price ranges based on recent volatility (ATR) and the validated Naive Momentum benchmark, plus what could push prices up or down.",
    read: "Buy Call Range = pullback zone with Stop-Loss; Sell Call Range = profit-taking zone; Upside/Downside conditions explain what to watch over 5 days, 1 month, and 1 year.",
    use: "Never chase prices far above the Buy Range; respect the Stop-Loss if the downside conditions trigger.",
  },
};

function createHelpTooltipEl(guide, cornerMode = true) {
  const wrap = document.createElement("div");
  wrap.className = cornerMode ? "help-tip-wrap help-tip-corner" : "help-tip-wrap";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "help-tip-btn";
  btn.setAttribute("aria-label", `Help: ${guide.title}`);
  btn.textContent = "?";

  const pop = document.createElement("div");
  pop.className = "help-tip-popover";
  pop.setAttribute("role", "tooltip");
  pop.innerHTML = `
    <div class="help-tip-title">${guide.title}</div>
    <div class="help-tip-row"><strong>What it is:</strong> ${guide.what}</div>
    <div class="help-tip-row"><strong>How to read it:</strong> ${guide.read}</div>
    <div class="help-tip-row"><strong>How to use it:</strong> ${guide.use}</div>
  `;

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = wrap.classList.contains("open");
    document.querySelectorAll(".help-tip-wrap.open").forEach((el) => el.classList.remove("open"));
    if (!isOpen) wrap.classList.add("open");
  });

  wrap.appendChild(btn);
  wrap.appendChild(pop);
  return wrap;
}

function initBeginnerHelpTooltips() {
  Object.entries(BEGINNER_HELP_GUIDES).forEach(([targetId, guide]) => {
    const el = document.getElementById(targetId);
    if (!el) return;

    if (el.tagName === "CANVAS") {
      const card = el.closest(".sub-panel") || el.closest(".canvas-container") || el.parentElement;
      if (!card || card.querySelector(".help-tip-wrap")) return;
      if (targetId === "price-dma-canvas") {
        const rsiBadge = document.getElementById("rsi-badge");
        if (rsiBadge && rsiBadge.parentElement) {
          const rightBox = document.createElement("div");
          rightBox.style.cssText = "display:flex;align-items:center;gap:0.5rem;";
          rsiBadge.parentElement.insertBefore(rightBox, rsiBadge);
          rightBox.appendChild(rsiBadge);
          rightBox.appendChild(createHelpTooltipEl(guide, false));
          return;
        }
      }
      card.classList.add("has-help-corner");
      const h3 = card.querySelector("h3");
      if (h3) h3.style.paddingRight = "2rem";
      card.appendChild(createHelpTooltipEl(guide, true));
      return;
    }

    if (el.classList.contains("collapsible-section")) {
      const hdr = el.querySelector(":scope > .collapsible-header");
      if (!hdr || hdr.querySelector(".help-tip-wrap")) return;
      const chevron = hdr.querySelector(".chevron-btn");
      const tip = createHelpTooltipEl(guide, false);
      tip.style.marginLeft = "auto";
      tip.style.marginRight = "0.6rem";
      if (chevron) {
        hdr.insertBefore(tip, chevron);
      } else {
        hdr.appendChild(tip);
      }
      return;
    }

    if (targetId === "market-regime-strip") {
      if (!el.querySelector(".help-tip-wrap")) {
        const rightGroup = el.children[1];
        if (rightGroup) {
          rightGroup.appendChild(createHelpTooltipEl(guide, false));
        } else {
          el.appendChild(createHelpTooltipEl(guide, false));
        }
      }
      return;
    }

    if (!el.querySelector(".help-tip-wrap")) {
      el.classList.add("has-help-corner");
      el.appendChild(createHelpTooltipEl(guide, true));
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".help-tip-wrap")) {
      document.querySelectorAll(".help-tip-wrap.open").forEach((el) => el.classList.remove("open"));
    }
  });
}

document.addEventListener("DOMContentLoaded", initApp);
