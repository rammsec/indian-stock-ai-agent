// Historical AI Trading Lab — Interactive Replay & Model Evaluation App

const LAB_STATE = {
  index: null,
  currentDateData: null,
  currentDate: null,
  currentStrategy: null,
  currentStock: "RELIANCE",
  replayBarIdx: 74,
  maxBars: 75,
  isPlaying: false,
  timerId: null,
  speedMs: 800,
  candleChart: null,
  candleSeries: null,
  ema9Series: null,
  ema21Series: null,
  vwapSeries: null,
  volSeries: null,
  equityChart: null,
  drawdownChart: null,
};

document.addEventListener("DOMContentLoaded", async () => {
  initNavigation();
  initControls();
  await loadLabIndex();
});

function initNavigation() {
  document.querySelectorAll(".nav-tab[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab[data-view]").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".section-view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.view).classList.add("active");
    });
  });
}

function initControls() {
  document.getElementById("btn-run-backtest").addEventListener("click", async () => {
    const dt = document.getElementById("date-select").value;
    const strat = document.getElementById("strategy-select").value;
    await selectDateAndStrategy(dt, strat);
  });

  document.getElementById("stock-select").addEventListener("change", (e) => {
    LAB_STATE.currentStock = e.target.value;
    renderCandlestickChart();
    renderFiveStockCards();
  });

  document.getElementById("btn-play-pause").addEventListener("click", togglePlayPause);
  document.getElementById("btn-step-fwd").addEventListener("click", () => stepReplay(1));
  document.getElementById("btn-step-back").addEventListener("click", () => stepReplay(-1));
  document.getElementById("btn-restart").addEventListener("click", () => {
    stopReplay();
    LAB_STATE.replayBarIdx = 0;
    syncReplayUI();
  });
  document.getElementById("btn-reset-replay").addEventListener("click", () => {
    stopReplay();
    LAB_STATE.replayBarIdx = LAB_STATE.maxBars - 1;
    syncReplayUI();
  });
  document.getElementById("speed-select").addEventListener("change", (e) => {
    LAB_STATE.speedMs = parseInt(e.target.value, 10);
    if (LAB_STATE.isPlaying) {
      stopReplay();
      startReplay();
    }
  });
  document.getElementById("replay-slider").addEventListener("input", (e) => {
    stopReplay();
    LAB_STATE.replayBarIdx = parseInt(e.target.value, 10);
    syncReplayUI();
  });
}

async function loadLabIndex() {
  try {
    const res = await fetch("data/lab_index.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    LAB_STATE.index = await res.json();

    const dateSel = document.getElementById("date-select");
    dateSel.innerHTML = LAB_STATE.index.available_dates
      .map((d) => `<option value="${d.date}">${d.date} — ${d.date_status || "PARTIAL"} (${d.ready_count}/${d.total_stocks} Ready)</option>`)
      .join("");

    const stratSel = document.getElementById("strategy-select");
    stratSel.innerHTML = LAB_STATE.index.strategies
      .map((s) => `<option value="${s.id}">${s.name}</option>`)
      .join("");

    renderPerformanceHistory(LAB_STATE.index.history_runs || []);

    const defaultDate = LAB_STATE.index.available_dates[0].date;
    const defaultStrat = LAB_STATE.index.strategies[0].id;
    await selectDateAndStrategy(defaultDate, defaultStrat);
  } catch (err) {
    console.error("Failed to load lab_index.json:", err);
  }
}

async function selectDateAndStrategy(dateStr, strategyId) {
  stopReplay();
  LAB_STATE.currentDate = dateStr;
  LAB_STATE.currentStrategy = strategyId;

  const res = await fetch(`data/runs_${dateStr}.json`);
  LAB_STATE.currentDateData = await res.json();

  const runBundle = getActiveRunBundle();
  const symBars = runBundle.candles[LAB_STATE.currentStock] || [];
  LAB_STATE.maxBars = Math.max(1, symBars.length);
  LAB_STATE.replayBarIdx = LAB_STATE.maxBars - 1;

  const slider = document.getElementById("replay-slider");
  slider.max = LAB_STATE.maxBars - 1;
  slider.value = LAB_STATE.replayBarIdx;

  renderDataQuality(dateStr);
  renderLeaderboard();
  renderStageBC();
  syncReplayUI();
}

function getActiveRunBundle() {
  return LAB_STATE.currentDateData.runs[LAB_STATE.currentStrategy];
}

function renderDataQuality(dateStr) {
  const dateInfo = LAB_STATE.index.available_dates.find((d) => d.date === dateStr);
  document.getElementById("lbl-selected-date").textContent =
    dateInfo ? `${dateStr} • ${dateInfo.date_status || "PARTIAL"} (${dateInfo.ready_count}/${dateInfo.total_stocks} Ready)` : dateStr;
  const tbody = document.getElementById("data-quality-tbody");
  if (!dateInfo || !tbody) return;

  const runBundle = getActiveRunBundle();
  const provMap = runBundle.run.data_provenance || {};

  tbody.innerHTML = dateInfo.stocks
    .map((s) => {
      const p = provMap[s.symbol] || {};
      const badgeCls = s.status === "Ready" ? "badge-green" : (s.status === "Incomplete" ? "badge-amber" : "badge-red");
      return `
        <tr>
          <td style="font-weight: 700; color: #fff;">${s.symbol}</td>
          <td>${p.availability || s.data_availability}</td>
          <td>${p.candles ?? s.candles_count} / ${s.expected_candles || 75}</td>
          <td>${p.warmup ?? 0}</td>
          <td>${p.missing ?? s.missing_intervals_count}</td>
          <td style="font-size: 11px; color: var(--text-muted);">${p.provider || s.provider}</td>
          <td><span class="badge ${badgeCls}">${s.status}</span></td>
          <td style="font-size: 11px; color: var(--text-muted);">${s.quality_reason || ""}</td>
        </tr>
      `;
    })
    .join("");
}

function togglePlayPause() {
  if (LAB_STATE.isPlaying) {
    stopReplay();
  } else {
    if (LAB_STATE.replayBarIdx >= LAB_STATE.maxBars - 1) {
      LAB_STATE.replayBarIdx = 0;
    }
    startReplay();
  }
}

function startReplay() {
  LAB_STATE.isPlaying = true;
  document.getElementById("btn-play-pause").textContent = "⏸ Pause";
  LAB_STATE.timerId = setInterval(() => {
    if (LAB_STATE.replayBarIdx < LAB_STATE.maxBars - 1) {
      LAB_STATE.replayBarIdx += 1;
      syncReplayUI();
    } else {
      stopReplay();
    }
  }, LAB_STATE.speedMs);
}

function stopReplay() {
  LAB_STATE.isPlaying = false;
  document.getElementById("btn-play-pause").textContent = "▶ Play";
  if (LAB_STATE.timerId) {
    clearInterval(LAB_STATE.timerId);
    LAB_STATE.timerId = null;
  }
}

function stepReplay(delta) {
  stopReplay();
  const next = Math.max(0, Math.min(LAB_STATE.maxBars - 1, LAB_STATE.replayBarIdx + delta));
  LAB_STATE.replayBarIdx = next;
  syncReplayUI();
}

function syncReplayUI() {
  const bundle = getActiveRunBundle();
  const bars = bundle.candles[LAB_STATE.currentStock] || [];
  const idx = Math.min(LAB_STATE.replayBarIdx, Math.max(0, bars.length - 1));
  const curBar = bars[idx];

  document.getElementById("replay-slider").value = idx;
  document.getElementById("lbl-bar-counter").textContent = `Bar ${idx + 1} / ${bars.length}`;
  document.getElementById("lbl-replay-clock").textContent = curBar ? `${curBar.time_str} IST` : "--:-- IST";

  renderPortfolioOverview(idx);
  renderCandlestickChart();
  renderFiveStockCards();
  renderDecisionTimeline();
}

function renderPortfolioOverview(barIdx) {
  const bundle = getActiveRunBundle();
  const snaps = bundle.snapshots || [];
  const snap = snaps[Math.min(barIdx, snaps.length - 1)] || {
    cash: 50000,
    total_equity: 50000,
    realised_pnl: 0,
    unrealised_pnl: 0,
    drawdown_pct: 0,
  };
  const run = bundle.run;
  const retPct = (((snap.total_equity - 50000) / 50000) * 100).toFixed(2);
  const bm = bundle.benchmark;

  document.getElementById("lbl-benchmark-summary").textContent =
    `5-Stock Equal-Weight Buy & Hold: ₹${bm.ending_capital_inr.toLocaleString("en-IN")} (${bm.total_return_pct >= 0 ? "+" : ""}${bm.total_return_pct}%)`;

  const grid = document.getElementById("portfolio-metrics-grid");
  grid.innerHTML = `
    <div class="metric-box"><div class="metric-label">Starting Capital</div><div class="metric-val">₹50,000</div></div>
    <div class="metric-box"><div class="metric-label">Portfolio Value</div><div class="metric-val ${snap.total_equity >= 50000 ? "text-green" : "text-red"}">₹${snap.total_equity.toLocaleString("en-IN")}</div></div>
    <div class="metric-box"><div class="metric-label">Realised P&L</div><div class="metric-val ${snap.realised_pnl >= 0 ? "text-green" : "text-red"}">₹${snap.realised_pnl.toFixed(2)}</div></div>
    <div class="metric-box"><div class="metric-label">Unrealised P&L</div><div class="metric-val ${snap.unrealised_pnl >= 0 ? "text-green" : "text-red"}">₹${snap.unrealised_pnl.toFixed(2)}</div></div>
    <div class="metric-box"><div class="metric-label">Total Return</div><div class="metric-val ${retPct >= 0 ? "text-green" : "text-red"}">${retPct >= 0 ? "+" : ""}${retPct}%</div></div>
    <div class="metric-box"><div class="metric-label">Available Cash</div><div class="metric-val">₹${snap.cash.toLocaleString("en-IN")}</div></div>
    <div class="metric-box"><div class="metric-label">Trades / Win Rate</div><div class="metric-val">${run.total_trades} / ${run.win_rate_pct}%</div></div>
    <div class="metric-box"><div class="metric-label">Max Drawdown</div><div class="metric-val text-amber">${run.max_drawdown_pct.toFixed(2)}%</div></div>
  `;

  renderEquityAndDrawdownCharts(snaps.slice(0, barIdx + 1), bm.equity_curve.slice(0, barIdx + 1));
}

function renderEquityAndDrawdownCharts(snaps, bmCurve) {
  const eqEl = document.getElementById("equity-chart-container");
  const ddEl = document.getElementById("drawdown-chart-container");
  if (!window.LightweightCharts || !eqEl || !ddEl) return;

  eqEl.innerHTML = "";
  ddEl.innerHTML = "";

  const eqChart = LightweightCharts.createChart(eqEl, {
    layout: { background: { color: "#131a26" }, textColor: "#94a3b8" },
    grid: { vertLines: { color: "#1e293b" }, horzLines: { color: "#1e293b" } },
    height: 200,
    timeScale: { timeVisible: true, secondsVisible: false },
  });
  const aiLine = eqChart.addLineSeries({ color: "#38bdf8", lineWidth: 2, title: "Strategy" });
  const bmLine = eqChart.addLineSeries({ color: "#f59e0b", lineWidth: 1, lineStyle: 2, title: "Benchmark" });

  aiLine.setData(snaps.map((s) => ({ time: Math.floor(new Date(s.timestamp).getTime() / 1000), value: s.total_equity })));
  if (bmCurve.length > 0) {
    bmLine.setData(bmCurve.map((b) => ({ time: Math.floor(new Date(b.timestamp).getTime() / 1000), value: b.equity })));
  }
  eqChart.timeScale().fitContent();

  const ddChart = LightweightCharts.createChart(ddEl, {
    layout: { background: { color: "#131a26" }, textColor: "#94a3b8" },
    grid: { vertLines: { color: "#1e293b" }, horzLines: { color: "#1e293b" } },
    height: 200,
    timeScale: { timeVisible: true, secondsVisible: false },
  });
  const ddArea = ddChart.addAreaSeries({ lineColor: "#ef4444", topColor: "rgba(239,68,68,0.35)", bottomColor: "rgba(239,68,68,0.02)" });
  ddArea.setData(snaps.map((s) => ({ time: Math.floor(new Date(s.timestamp).getTime() / 1000), value: s.drawdown_pct })));
  ddChart.timeScale().fitContent();
}

function renderCandlestickChart() {
  const container = document.getElementById("candlestick-chart-container");
  const sym = LAB_STATE.currentStock;
  document.getElementById("lbl-chart-symbol").textContent = sym;
  if (!container || !window.LightweightCharts) return;

  container.innerHTML = "";
  const bundle = getActiveRunBundle();
  const allBars = bundle.candles[sym] || [];
  const visibleBars = allBars.slice(0, LAB_STATE.replayBarIdx + 1);
  if (visibleBars.length === 0) return;

  const chart = LightweightCharts.createChart(container, {
    layout: { background: { color: "#0b0f17" }, textColor: "#94a3b8" },
    grid: { vertLines: { color: "#1a2333" }, horzLines: { color: "#1a2333" } },
    height: 420,
    timeScale: { timeVisible: true, secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: "#22c55e",
    downColor: "#ef4444",
    borderVisible: false,
    wickUpColor: "#22c55e",
    wickDownColor: "#ef4444",
  });
  candleSeries.setData(visibleBars.map((b) => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close })));

  const ema9 = chart.addLineSeries({ color: "#38bdf8", lineWidth: 1, title: "EMA 9" });
  ema9.setData(visibleBars.map((b) => ({ time: b.time, value: b.ema_9 })));

  const ema21 = chart.addLineSeries({ color: "#a855f7", lineWidth: 1, title: "EMA 21" });
  ema21.setData(visibleBars.map((b) => ({ time: b.time, value: b.ema_21 })));

  const vwap = chart.addLineSeries({ color: "#f59e0b", lineWidth: 1, lineStyle: 2, title: "VWAP" });
  vwap.setData(visibleBars.map((b) => ({ time: b.time, value: b.vwap })));

  const lastVisibleTime = visibleBars[visibleBars.length - 1].time;
  const rawMarkers = (bundle.markers[sym] || []).filter((m) => m.time <= lastVisibleTime);

  const chartMarkers = rawMarkers.map((m) => {
    if (m.type === "BUY") {
      return { time: m.time, position: "belowBar", color: "#22c55e", shape: "arrowUp", text: `BUY ₹${m.price}` };
    } else if (m.type === "SELL") {
      return { time: m.time, position: "aboveBar", color: "#ef4444", shape: "arrowDown", text: `SELL ₹${m.price}` };
    } else {
      return { time: m.time, position: "aboveBar", color: "#f59e0b", shape: "circle", text: "REJ" };
    }
  });
  candleSeries.setMarkers(chartMarkers);

  chart.subscribeClick((param) => {
    if (!param || !param.time) return;
    const match = rawMarkers.find((m) => m.time === param.time);
    if (match) showDecisionDetail(match);
  });

  chart.timeScale().fitContent();
}

function showDecisionDetail(markerOrDecision) {
  const panel = document.getElementById("decision-detail-panel");
  const d = markerOrDecision.decision || markerOrDecision;
  const exec = markerOrDecision.execution || null;
  const snap = d.indicator_snapshot || {};

  panel.innerHTML = `
    <h4 style="color: #fff; margin-bottom: 8px;">${d.symbol} — ${d.recommendation || markerOrDecision.type} Decision</h4>
    <div style="font-size: 12px; display: flex; flex-direction: column; gap: 6px;">
      <div><span style="color: var(--text-muted);">Decision Time:</span> <strong>${d.time_str || d.decision_timestamp}</strong></div>
      <div><span style="color: var(--text-muted);">Agent / Model:</span> <strong>${d.agent_name}</strong> (<code>${d.model_name} ${d.model_version}</code>)</div>
      <div><span style="color: var(--text-muted);">Execution Mode:</span> <span class="badge badge-blue">${d.execution_mode}</span></div>
      <div><span style="color: var(--text-muted);">Confidence:</span> <strong>${Math.round((d.confidence || 0) * 100)}%</strong></div>
      <div><span style="color: var(--text-muted);">Council Posture:</span> <strong>${d.council_decision}</strong></div>
      ${exec ? `
        <hr style="border-color: var(--border-subtle); margin: 4px 0;">
        <div><span style="color: var(--text-muted);">Execution Time:</span> <strong>${exec.execution_time}</strong></div>
        <div><span style="color: var(--text-muted);">Simulated Fill:</span> <strong class="text-green">₹${exec.execution_price}</strong> × ${exec.quantity} qty</div>
        <div><span style="color: var(--text-muted);">Fees & Taxes:</span> ₹${exec.fees_inr} (${exec.status})</div>
      ` : ""}
      ${markerOrDecision.stop_loss ? `<div><span style="color: var(--text-muted);">Stop Loss / Target:</span> ₹${markerOrDecision.stop_loss} / ₹${markerOrDecision.target}</div>` : ""}
      <hr style="border-color: var(--border-subtle); margin: 4px 0;">
      <div style="color: var(--text-muted);">Reasoning:</div>
      <div style="background: var(--bg-card); padding: 8px; border-radius: 4px; color: #fff;">${d.reasoning_summary || markerOrDecision.reason}</div>
      <div style="color: var(--text-muted); margin-top: 4px;">Indicator Snapshot at Decision:</div>
      <div style="font-family: var(--font-mono); font-size: 11px; color: var(--accent-blue);">
        Close: ₹${snap.close || "-"} | VWAP: ₹${snap.vwap || "-"} | EMA9: ${snap.ema_9 || "-"} | EMA21: ${snap.ema_21 || "-"} | RSI: ${snap.rsi_14 || "-"} | ATR: ${snap.atr_14 || "-"}
      </div>
    </div>
  `;
}

function renderFiveStockCards() {
  const grid = document.getElementById("five-stock-grid");
  const bundle = getActiveRunBundle();
  const syms = ["RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI"];
  const idx = LAB_STATE.replayBarIdx;

  grid.innerHTML = syms
    .map((sym) => {
      const bars = bundle.candles[sym] || [];
      const curBar = bars[Math.min(idx, Math.max(0, bars.length - 1))];
      const openDay = bars[0] ? bars[0].open : 1;
      const curClose = curBar ? curBar.close : openDay;
      const chgPct = (((curClose - openDay) / openDay) * 100).toFixed(2);
      const lastTime = curBar ? curBar.time : 0;
      const markers = (bundle.markers[sym] || []).filter((m) => m.time <= lastTime);
      const buys = markers.filter((m) => m.type === "BUY").length;
      const sells = markers.filter((m) => m.type === "SELL").length;
      const symTrades = (bundle.trades || []).filter((t) => t.symbol === sym);
      const realised = symTrades.reduce((acc, t) => acc + t.net_pnl, 0);

      return `
        <div class="stock-card ${LAB_STATE.currentStock === sym ? "selected" : ""}" onclick="focusStock('${sym}')">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: #fff; font-size: 14px;">${sym}</strong>
            <span class="${chgPct >= 0 ? "text-green" : "text-red"}" style="font-weight: 700;">${chgPct >= 0 ? "+" : ""}${chgPct}%</span>
          </div>
          <div style="font-family: var(--font-mono); font-size: 16px; color: #fff; margin: 4px 0;">₹${curClose.toFixed(2)}</div>
          <div style="font-size: 11px; color: var(--text-muted);">Signals: ${buys} BUY / ${sells} SELL</div>
          <div style="font-size: 11px; color: ${realised >= 0 ? "var(--accent-green)" : "var(--accent-red)"}; margin-top: 2px;">
            Realised P&L: ₹${realised.toFixed(2)} (${symTrades.length} trades)
          </div>
        </div>
      `;
    })
    .join("");
}

window.focusStock = function (sym) {
  LAB_STATE.currentStock = sym;
  document.getElementById("stock-select").value = sym;
  renderCandlestickChart();
  renderFiveStockCards();
};

function renderDecisionTimeline() {
  const tbody = document.getElementById("decision-timeline-tbody");
  const bundle = getActiveRunBundle();
  const bars = bundle.candles[LAB_STATE.currentStock] || [];
  const curBar = bars[Math.min(LAB_STATE.replayBarIdx, Math.max(0, bars.length - 1))];
  const cutoffIso = curBar ? new Date(curBar.time * 1000).toISOString() : "9999";

  const decs = (bundle.decisions || []).filter((d) => new Date(d.decision_timestamp).toISOString() <= cutoffIso);
  tbody.innerHTML = decs
    .slice(-30)
    .reverse()
    .map((d) => {
      const badgeCls =
        d.recommendation === "BUY" ? "badge-green" : d.recommendation === "SELL" ? "badge-red" : "badge-gray";
      return `
        <tr style="cursor: pointer;" onclick='showDecisionDetail(${JSON.stringify(d).replace(/'/g, "&#39;")})'>
          <td style="font-family: var(--font-mono);">${d.time_str}</td>
          <td style="font-weight: 700; color: #fff;">${d.symbol}</td>
          <td>${d.agent_name}</td>
          <td><span class="badge ${badgeCls}">${d.recommendation}</span></td>
          <td>${Math.round(d.confidence * 100)}%</td>
          <td>${d.council_decision}</td>
          <td style="color: var(--text-muted);">${d.reasoning_summary}</td>
        </tr>
      `;
    })
    .join("");
}

function renderLeaderboard() {
  document.getElementById("lbl-leaderboard-date").textContent = LAB_STATE.currentDate;
  const tbody = document.getElementById("leaderboard-tbody");
  const runs = Object.values(LAB_STATE.currentDateData.runs).map((b) => b.run);
  runs.sort((a, b) => (b.total_return_pct - a.total_return_pct) || (b.sharpe_ratio - a.sharpe_ratio));

  tbody.innerHTML = runs
    .map((r, i) => {
      return `
        <tr>
          <td><strong>#${i + 1}</strong></td>
          <td style="font-weight: 700; color: #fff;">${r.strategy_name}</td>
          <td><span class="badge badge-blue">${r.mode}</span></td>
          <td>₹50,000</td>
          <td>₹${r.final_portfolio_value.toLocaleString("en-IN")}</td>
          <td class="${r.net_pnl >= 0 ? "text-green" : "text-red"}">₹${r.net_pnl.toFixed(2)}</td>
          <td class="${r.total_return_pct >= 0 ? "text-green" : "text-red"}">${r.total_return_pct >= 0 ? "+" : ""}${r.total_return_pct}%</td>
          <td>${r.win_rate_pct}%</td>
          <td>${r.profit_factor}</td>
          <td>${r.max_drawdown_pct}%</td>
          <td>${r.sharpe_ratio}</td>
          <td>${r.total_trades}</td>
          <td>${(r.avg_holding_minutes || 0).toFixed(1)}m</td>
          <td style="color: var(--accent-red);">₹${r.total_fees.toFixed(2)}</td>
          <td>$${(r.ai_api_cost_usd || 0).toFixed(4)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderStageBC() {
  const container = document.getElementById("stage-bc-container");
  const bundle = getActiveRunBundle();
  const bc = bundle.stage_bc_analysis || { stage_b_observations: [], stage_c_suggestions: [] };

  container.innerHTML = `
    <div class="card" style="background: var(--bg-elevated);">
      <h4 style="color: var(--accent-blue); margin-bottom: 8px;">Stage B — Performance Analysis (${bundle.run.strategy_name})</h4>
      <ul style="padding-left: 18px; color: var(--text-main);">
        ${bc.stage_b_observations.map((o) => `<li style="margin-bottom: 6px;">${o}</li>`).join("")}
      </ul>
    </div>
    <div class="card" style="background: var(--bg-elevated);">
      <h4 style="color: var(--accent-amber); margin-bottom: 8px;">Stage C — Experiment Suggestions (Non-Mutating)</h4>
      <ul style="padding-left: 18px; color: var(--text-main);">
        ${bc.stage_c_suggestions.map((s) => `<li style="margin-bottom: 6px;">${s}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderPerformanceHistory(runs) {
  const tbody = document.getElementById("history-tbody");
  if (!tbody) return;
  tbody.innerHTML = runs
    .slice(0, 60)
    .map(
      (r) => `
      <tr>
        <td>${r.selected_date}</td>
        <td style="font-weight: 700; color: #fff;">${r.strategy_id}</td>
        <td>₹${Number(r.final_portfolio_value).toLocaleString("en-IN")}</td>
        <td class="${r.net_pnl >= 0 ? "text-green" : "text-red"}">₹${Number(r.net_pnl).toFixed(2)}</td>
        <td class="${r.total_return_pct >= 0 ? "text-green" : "text-red"}">${r.total_return_pct >= 0 ? "+" : ""}${r.total_return_pct}%</td>
        <td>${r.win_rate_pct}%</td>
        <td>${r.max_drawdown_pct}%</td>
        <td>${r.total_trades}</td>
        <td style="color: var(--accent-red);">₹${Number(r.total_fees).toFixed(2)}</td>
      </tr>
    `
    )
    .join("");
}

function renderDataQualityAnalytics(dq) {
  if (!dq) return;
  const grid = document.getElementById("dq-summary-grid");
  if (grid) {
    grid.innerHTML = `
      <div class="metric-box"><div class="metric-label">Total Cached Days</div><div class="metric-val">${dq.total_available_days}</div></div>
      <div class="metric-box"><div class="metric-label">Fully Ready (75/75 All 5)</div><div class="metric-val text-amber">${dq.fully_ready_sessions}</div></div>
      <div class="metric-box"><div class="metric-label">Partial Sessions</div><div class="metric-val text-blue">${dq.partial_sessions}</div></div>
      <div class="metric-box"><div class="metric-label">Research Eligible (&ge;${dq.research_min_bars_threshold} bars)</div><div class="metric-val text-green">${dq.research_eligible_sessions}</div></div>
      <div class="metric-box"><div class="metric-label">Unavailable Sessions</div><div class="metric-val">${dq.unavailable_sessions}</div></div>
    `;
  }

  const tbody = document.getElementById("dq-matrix-tbody");
  if (tbody && dq.coverage_by_stock) {
    tbody.innerHTML = Object.values(dq.coverage_by_stock)
      .map(
        (s) => `
        <tr>
          <td style="font-weight: 700; color: #fff;">${s.symbol}</td>
          <td>${s.total_bars}</td>
          <td>${s.sessions_present}</td>
          <td class="text-green">${s.ready_sessions}</td>
          <td class="text-amber">${s.incomplete_sessions}</td>
          <td>${s.total_missing_intervals}</td>
          <td>${s.zero_volume_bars}</td>
          <td>${s.outlier_bars_gt_10pct}</td>
          <td>${s.avg_bars_per_session}</td>
          <td style="font-size: 11px; color: var(--text-muted);">${s.provider}</td>
        </tr>
      `
      )
      .join("");
  }

  const winEl = document.getElementById("dq-missing-windows");
  if (winEl && dq.missing_windows_histogram) {
    const entries = Object.entries(dq.missing_windows_histogram);
    winEl.innerHTML = entries.length
      ? entries.map(([w, cnt]) => `<span class="badge badge-amber" style="margin: 3px;">${w} IST: ${cnt} occurrences</span>`).join("")
      : "No missing intervals detected.";
  }
}

function renderMultiDayLeaderboard(rows, cov) {
  if (cov) {
    const rngEl = document.getElementById("lbl-multiday-range");
    const sesEl = document.getElementById("lbl-multiday-sessions");
    if (rngEl) rngEl.textContent = `${cov.start_date || "--"} to ${cov.end_date || "--"}`;
    if (sesEl) sesEl.textContent = `${cov.evaluated_days || 0} Sessions (PARTIAL Disclosed)`;
  }

  const tbody = document.getElementById("multiday-leaderboard-tbody");
  if (!tbody) return;
  tbody.innerHTML = rows
    .map((r, i) => {
      const catBadge = r.performance_category === "GENUINE_AI" ? "badge-green" : (r.performance_category === "DETERMINISTIC_FALLBACK" ? "badge-amber" : "badge-blue");
      const sufBadge = r.statistical_sufficiency === "VALIDATED_SAMPLE" ? "badge-green" : "badge-gray";
      return `
        <tr>
          <td><strong>#${i + 1}</strong></td>
          <td style="font-weight: 700; color: #fff;">${r.strategy_name}</td>
          <td><span class="badge ${catBadge}">${r.performance_category}</span></td>
          <td><span class="badge ${sufBadge}">${r.statistical_sufficiency}</span></td>
          <td>${r.sessions_evaluated}</td>
          <td class="${r.gross_return_pct >= 0 ? "text-green" : "text-red"}">${r.gross_return_pct >= 0 ? "+" : ""}${r.gross_return_pct}%</td>
          <td class="${r.total_return_pct >= 0 ? "text-green" : "text-red"}">${r.total_return_pct >= 0 ? "+" : ""}${r.total_return_pct}%</td>
          <td class="${r.alpha_vs_buy_and_hold_pct >= 0 ? "text-green" : "text-red"}">${r.alpha_vs_buy_and_hold_pct >= 0 ? "+" : ""}${r.alpha_vs_buy_and_hold_pct}%</td>
          <td>${r.sharpe_ratio}</td>
          <td>${r.sortino_ratio}</td>
          <td>${r.profit_factor}</td>
          <td>${r.max_drawdown_pct}%</td>
          <td>${r.total_trades}</td>
          <td style="color: var(--accent-red);">₹${r.total_fees_inr.toFixed(2)}</td>
          <td>$${(r.total_ai_cost_usd || 0).toFixed(4)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderSlippageAndWalkForward(slipRows, wf) {
  const slipBody = document.getElementById("slippage-tbody");
  if (slipBody) {
    slipBody.innerHTML = slipRows
      .map(
        (s) => `
        <tr>
          <td><strong>${s.slippage_bps} bps</strong></td>
          <td>${s.gross_return_pct >= 0 ? "+" : ""}${s.gross_return_pct}%</td>
          <td class="${s.net_return_pct >= 0 ? "text-green" : "text-red"}">${s.net_return_pct >= 0 ? "+" : ""}${s.net_return_pct}%</td>
          <td>₹${s.net_pnl_inr.toFixed(2)}</td>
          <td style="color: var(--accent-red);">₹${s.total_fees_inr.toFixed(2)}</td>
          <td>${s.total_trades}</td>
          <td>${s.win_rate_pct}%</td>
        </tr>
      `
      )
      .join("");
  }

  const wfBody = document.getElementById("walkforward-tbody");
  if (wfBody && wf.train) {
    const splits = [
      ["Train (60%)", wf.train],
      ["Validation (20%)", wf.validation],
      ["Unseen Test (20%)", wf.unseen_test],
    ];
    wfBody.innerHTML = splits
      .map(
        ([label, d]) => `
        <tr>
          <td style="font-weight: 700; color: #fff;">${label}</td>
          <td style="font-size: 11px;">${d.start_date} &rarr; ${d.end_date}</td>
          <td>${d.sessions_evaluated}</td>
          <td class="${d.total_return_pct >= 0 ? "text-green" : "text-red"}">${d.total_return_pct >= 0 ? "+" : ""}${d.total_return_pct}%</td>
          <td>${d.benchmark_return_pct >= 0 ? "+" : ""}${d.benchmark_return_pct}%</td>
          <td class="${d.alpha_vs_buy_and_hold_pct >= 0 ? "text-green" : "text-red"}">${d.alpha_vs_buy_and_hold_pct >= 0 ? "+" : ""}${d.alpha_vs_buy_and_hold_pct}%</td>
          <td>${d.total_trades}</td>
        </tr>
      `
      )
      .join("");
  }
}

function renderExperiments(exps) {
  const tbody = document.getElementById("experiments-tbody");
  if (!tbody) return;
  tbody.innerHTML = exps
    .map((e) => {
      const cls = e.approval_status === "APPROVED_FOR_PAPER" ? "badge-green" : "badge-amber";
      return `
        <tr>
          <td style="font-weight: 700; color: #fff;">${e.experiment_id}</td>
          <td><code>${e.parent_strategy_version}</code></td>
          <td style="font-size: 11px;">${e.validation_period}</td>
          <td style="color: var(--text-muted);">${e.suggestion_summary}</td>
          <td><span class="badge ${cls}">${e.approval_status}</span></td>
        </tr>
      `;
    })
    .join("");
}


