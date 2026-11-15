/* BharatAlpha AI — Frontend Application Logic & Custom High-DPI Financial Canvas Charts */

const state = {
  currentSymbol: "SUNPHARMA",
  bundle: null,
  universe: [],
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
  updateLLMBadge();
  setupTabs();
  setupSearchAndControls();
  await loadUniverseRibbon();
  await loadStockResearch(state.currentSymbol);
}

function updateLLMBadge() {
  const badge = document.getElementById("llm-status-badge");
  if (!badge) return;
  const p = state.llmConfig.provider || "quant";
  if (p === "quant") {
    badge.textContent = "🧠 Engine: Quant Committee";
  } else {
    badge.textContent = `🧠 Engine: ${p.toUpperCase()} + Quant`;
  }
}

function setupTabs() {
  const btns = document.querySelectorAll(".tab-btn");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      btns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const pane = document.getElementById(target);
      if (pane) pane.classList.add("active");
      if (state.bundle) {
        requestAnimationFrame(() => renderAllCanvases(state.bundle));
      }
      if (target === "tab-scanner") {
        const tbody = document.getElementById("scanner-tbody");
        if (tbody && tbody.children.length <= 1) {
          runMarketScanner();
        }
      }
    });
  });
}

async function loadUniverseRibbon() {
  try {
    const res = await fetch("/api/universe");
    const data = await res.json();
    state.universe = data.universe || [];
    const ribbon = document.getElementById("ticker-ribbon");
    if (!ribbon) return;
    ribbon.innerHTML = `<span class="ticker-label">NSE Watchlist:</span>`;
    state.universe.forEach((item) => {
      const chip = document.createElement("button");
      chip.className = `ticker-chip ${item.symbol === state.currentSymbol ? "active" : ""}`;
      chip.dataset.symbol = item.symbol;
      chip.textContent = item.symbol;
      chip.title = `${item.name} (${item.sector})`;
      chip.addEventListener("click", () => {
        loadStockResearch(item.symbol);
      });
      ribbon.appendChild(chip);
    });
  } catch (err) {
    console.error("Failed loading universe:", err);
  }
}

function setupSearchAndControls() {
  const input = document.getElementById("stock-search-input");
  const dropdown = document.getElementById("search-dropdown");
  const btnRun = document.getElementById("btn-run-research");

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
          <div class="search-item" data-sym="${it.symbol}">
            <span><strong>${it.name}</strong></span>
            <span class="mono badge-neutral">NSE: ${it.symbol}</span>
          </div>
        `
          )
          .join("");
        dropdown.classList.add("open");
        dropdown.querySelectorAll(".search-item").forEach((el) => {
          el.addEventListener("click", () => {
            const sym = el.getAttribute("data-sym");
            input.value = sym;
            dropdown.classList.remove("open");
            loadStockResearch(sym);
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
      if (sym) loadStockResearch(sym);
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-bar-wrap")) {
      dropdown.classList.remove("open");
    }
  });

  btnRun.addEventListener("click", () => {
    const sym = input.value.trim().toUpperCase() || state.currentSymbol;
    loadStockResearch(sym);
  });

  document.getElementById("btn-open-scanner").addEventListener("click", () => {
    const scanTabBtn = document.querySelector('[data-tab="tab-scanner"]');
    if (scanTabBtn) scanTabBtn.click();
  });

  document.getElementById("btn-refresh-scanner").addEventListener("click", () => {
    runMarketScanner();
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
  setLoading(true);

  document.querySelectorAll(".ticker-chip").forEach((ch) => {
    ch.classList.toggle("active", ch.dataset.symbol === sym);
  });

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
    renderBundle(bundle);
  } catch (err) {
    alert(`Error analyzing ${sym}: ${err.message}`);
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

  // Header & Overview
  document.getElementById("ov-company-name").textContent = ov.name;
  document.getElementById("ov-symbol-badge").textContent = `NSE: ${ov.symbol}`;
  document.getElementById("ov-sector-badge").textContent = `${ov.sector} • ${ov.industry}`;
  document.getElementById("ov-piotroski-badge").textContent = `Piotroski F-Score: ${ov.piotroski_score}/9`;
  document.getElementById("ov-timestamp").textContent = `As of ${b.fetched_at}`;

  document.getElementById("ov-current-price").textContent = `₹${fmtNum(tech.current_price)}`;
  const chgEl = document.getElementById("ov-day-change");
  const isUp = tech.day_change_pct >= 0;
  chgEl.textContent = `${isUp ? "▲" : "▼"} ${fmtPct(tech.day_change_pct, 2, true)} (1D) | 1Y: ${fmtPct(tech.return_1y_pct, 1, true)}`;
  chgEl.className = `price-change ${isUp ? "text-bull" : "text-bear"}`;

  // KPI strip
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

  // Next-Day Guidance Card
  const banner = document.getElementById("gd-action-banner");
  banner.textContent = gd.action;
  if (gd.action.includes("BUY")) {
    banner.className = "signal-banner bull";
  } else if (gd.action.includes("SELL") || gd.action.includes("AVOID")) {
    banner.className = "signal-banner bear";
  } else {
    banner.className = "signal-banner hold";
  }

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
  document.getElementById("gd-pos-size").textContent = `${gd.position_size_suggestion_pct}% of Capital`;
  document.getElementById("gd-atr-range").textContent = `±₹${fmtNum(tech.atr_14)} (${fmtPct(tech.atr_pct, 2)})`;

  renderAgentCommittee(b.agent_committee);
  renderCatalystsRisksNews(gd, b.news, tech);
  renderReportsTables(b.last_5_annual_reports, b.last_5_quarterly_reports);
  renderCashSection(cb);
  renderStockholdersSection(shp);
  renderRDSection(rd);
  renderAllCanvases(b);
}

function renderAgentCommittee(committee) {
  const grid = document.getElementById("agents-committee-grid");
  grid.innerHTML = committee
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
    newsEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.78rem;">No recent news headlines found.</div>`;
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
      <a href="${n.url}" target="_blank" rel="noopener" style="text-decoration: none; color: inherit; background: var(--bg-canvas); border: 1px solid var(--border-subtle); border-radius: 4px; padding: 0.45rem 0.65rem; display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;">
        <span style="font-size: 0.78rem; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${n.title}</span>
        <span class="mono" style="font-size: 0.68rem; color: ${color}; white-space: nowrap;">${n.source} • ${n.sentiment}</span>
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
      {
        label: "YoY / Seq Revenue Growth %",
        key: "revenue_growth_pct",
        fmt: (v) => fmtPct(v, 1, true),
        colored: true,
      },
      { label: "Total Expenses (₹ Cr)", key: "expenses_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Operating Profit / EBIT (₹ Cr)", key: "operating_profit_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Operating Profit Margin (OPM %)", key: "opm_pct", fmt: (v) => fmtPct(v, 1) },
      { label: "Other Income (₹ Cr)", key: "other_income_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Interest Cost (₹ Cr)", key: "interest_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Depreciation (₹ Cr)", key: "depreciation_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Profit Before Tax (₹ Cr)", key: "profit_before_tax_cr", fmt: (v) => fmtNum(v, 0) },
      { label: "Net Profit (₹ Cr)", key: "net_profit_cr", fmt: (v) => fmtNum(v, 0) },
      {
        label: "Net Profit Growth %",
        key: "net_profit_growth_pct",
        fmt: (v) => fmtPct(v, 1, true),
        colored: true,
      },
      { label: "EPS (₹ per Share)", key: "eps_inr", fmt: (v) => `₹${fmtNum(v, 2)}` },
    ];
    if (isAnnual) {
      rowSpec.push({
        label: "Dividend Payout %",
        key: "dividend_payout_pct",
        fmt: (v) => fmtPct(v, 1),
      });
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
      <div style="font-size: 0.76rem; color: var(--text-muted); margin-bottom: 0.35rem; text-transform: uppercase;">5-Year Return on Capital Employed (ROCE %)</div>
      <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
        ${cb.roce_5y_series
          .map(
            (r) => `
          <div class="badge-neutral mono" style="padding: 0.35rem 0.65rem;">
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
    {
      label: "Total Shareholder Accounts",
      key: "num_shareholders",
      fmt: (v) => (v ? Number(v).toLocaleString("en-IN") : "—"),
    },
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
    <div style="font-size: 0.76rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem;">R&amp;D &amp; Innovation Pipeline Focus</div>
    <div style="color: var(--text-primary); font-weight: 500;">${rd.innovation_focus}</div>
    <div style="margin-top: 0.55rem; display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.78rem;" class="mono">
      <span>Latest R&amp;D: <strong class="text-bull">${fmtCr(rd.latest_rd_spend_cr)} (${fmtPct(rd.latest_rd_pct_of_sales, 2)})</strong></span>
      <span>5Y Cumulative R&amp;D: <strong>${fmtCr(rd.cumulative_5y_rd_cr)}</strong></span>
      <span>Intangibles: <strong>${fmtCr(rd.intangible_assets_cr)}</strong></span>
      <span>CapEx WIP (CWIP): <strong>${fmtCr(rd.cwip_cr)}</strong></span>
    </div>
  `;

  const docsEl = document.getElementById("annual-reports-links");
  const docs = rd.annual_reports_5y || [];
  if (!docs.length) {
    docsEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem;">No direct Annual Report PDF links found.</div>`;
  } else {
    docsEl.innerHTML = docs
      .map(
        (d) => `
      <a href="${d.url}" target="_blank" rel="noopener" class="btn" style="justify-content: space-between; text-decoration: none;">
        <span>📄 ${d.title}</span>
        <span class="mono" style="font-size: 0.7rem; color: var(--accent-brand);">PDF ↗</span>
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

async function runMarketScanner() {
  const tbody = document.getElementById("scanner-tbody");
  const btn = document.getElementById("btn-refresh-scanner");
  if (!tbody) return;
  btn.disabled = true;
  btn.textContent = "Scanning NSE Universe...";
  tbody.innerHTML = `<tr><td colspan="14" style="text-align: center; padding: 2rem; color: var(--text-secondary);">Running 5-Agent Research Committee across top NSE stocks...</td></tr>`;

  try {
    const res = await fetch("/api/scanner");
    const data = await res.json();
    const items = data.items || [];
    tbody.innerHTML = items
      .map((it) => {
        const sigCls = it.action.includes("BUY")
          ? "text-bull"
          : it.action.includes("SELL") || it.action.includes("AVOID")
          ? "text-bear"
          : "text-hold";
        const chgCls = it.day_change_pct >= 0 ? "text-bull" : "text-bear";
        return `
        <tr>
          <td>
            <strong>${it.symbol}</strong>
            <span style="display:block; font-size:0.7rem; color:var(--text-muted); font-family:var(--font-sans);">${it.name}</span>
          </td>
          <td>₹${fmtNum(it.current_price)}</td>
          <td class="${chgCls}">${fmtPct(it.day_change_pct, 2, true)}</td>
          <td class="${sigCls}"><strong>${it.action}</strong></td>
          <td><strong>${it.conviction_score.toFixed(1)}</strong>/100</td>
          <td>${it.fundamental_score.toFixed(1)}</td>
          <td>${it.technical_score.toFixed(1)}</td>
          <td>${it.entry_zone}</td>
          <td class="text-bull">₹${fmtNum(it.target_1)}</td>
          <td class="text-bear">₹${fmtNum(it.stop_loss)}</td>
          <td>1:${it.risk_reward}</td>
          <td>${fmtCr(it.cash_in_hand_cr)}</td>
          <td>${fmtPct(it.rd_pct, 2)}</td>
          <td>
            <button class="btn btn-primary btn-inspect-scan" data-sym="${it.symbol}" style="padding:0.25rem 0.6rem; font-size:0.72rem;">
              Deep Dive
            </button>
          </td>
        </tr>
      `;
      })
      .join("");

    tbody.querySelectorAll(".btn-inspect-scan").forEach((b) => {
      b.addEventListener("click", () => {
        const sym = b.getAttribute("data-sym");
        document.querySelector('[data-tab="tab-committee"]').click();
        loadStockResearch(sym);
      });
    });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="14" class="text-bear" style="text-align:center; padding: 1.5rem;">Scanner error: ${e.message}</td></tr>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Live NSE Scan";
  }
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

  // Horizontal Grid
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

  // Draw T1 and SL dashed lines
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

  // 200-DMA
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    const y = yOf(p.dma200 || p.close);
    idx === 0 ? ctx.moveTo(xOf(idx), y) : ctx.lineTo(xOf(idx), y);
  });
  ctx.stroke();

  // 50-DMA
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    const y = yOf(p.dma50 || p.close);
    idx === 0 ? ctx.moveTo(xOf(idx), y) : ctx.lineTo(xOf(idx), y);
  });
  ctx.stroke();

  // Price line + subtle area gradient
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

  // Legend
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

  const colors = {
    prom: "#3b82f6",
    fii: "#22c55e",
    dii: "#f59e0b",
    pub: "#64748b",
  };

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

  // R&D % line
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
