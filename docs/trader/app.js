// Institutional Trading Desk Dashboard Application

document.addEventListener("DOMContentLoaded", async () => {
  initClock();
  await loadTraderState();
});

function initClock() {
  const clockEl = document.getElementById("live-ist-clock");
  function update() {
    if (!clockEl) return;
    const now = new Date();
    // Format in IST
    const options = {
      timeZone: "Asia/Kolkata",
      hour12: false,
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    };
    clockEl.textContent = new Intl.DateTimeFormat("en-IN", options).format(now) + " IST";
  }
  update();
  setInterval(update, 1000);
}

async function loadTraderState() {
  try {
    const res = await fetch("data/trader_state.json");
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    renderDashboard(data);
  } catch (err) {
    console.error("Failed to load trader_state.json:", err);
    document.getElementById("terminal-main").innerHTML = `
      <div class="card" style="margin: 40px auto; max-width: 600px; text-align: center; border-color: #ef4444;">
        <h3 class="text-red">Dashboard Data Loading Error</h3>
        <p style="margin-top: 10px; color: #94a3b8;">Unable to fetch <code>data/trader_state.json</code>.</p>
        <p style="margin-top: 5px; font-size: 12px; color: #64748b;">Run <code>scripts/generate_trader_data.py</code> to regenerate state.</p>
      </div>
    `;
  }
}

function renderDashboard(data) {
  renderDataProvenance(data.metadata || data.meta, data.data_provenance, data.cost_monitoring);
  renderMarketOverview(data.market_overview);
  renderCouncilDebate(data.council_debate, data.market_overview.central_posture);
  renderStockScanner(data.stock_scanner);
  renderCrossExchange(data.cross_exchange_arbitrage);
  renderPaperPortfolio(data.paper_portfolio);
  renderStrategyPerformance(data.strategy_performance);
  renderCostMonitoring(data.cost_monitoring);
}

function renderDataProvenance(meta, prov, cost) {
  const el = document.getElementById("data-provenance-banner");
  if (!el || !prov) return;

  const isSynthetic = prov.source_mode === "SYNTHETIC_TEST";
  const badgeClass = isSynthetic ? "badge-amber" : "badge-green";
  const firstHash = prov.datasets && prov.datasets.length > 0 ? prov.datasets[0].sha256_hash.slice(0, 12) : "N/A";
  const llmCalls = cost ? (cost.llm_calls_count || 0) : 0;
  const fallbackCalls = cost ? (cost.fallback_calls_count || 0) : 0;

  el.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
      <span class="badge ${badgeClass}">DATA MODE: ${prov.source_mode}</span>
      <span style="color: var(--text-muted);">Timestamp: <strong>${meta ? meta.last_updated_ist : prov.generated_at_ist}</strong></span>
      <span style="color: var(--text-muted);">SHA-256: <code>${firstHash}...</code> (${prov.datasets_count} datasets)</span>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <span class="badge badge-blue">LLM Calls: ${llmCalls} | Deterministic: ${fallbackCalls}</span>
      <span style="color: var(--text-dim);">Broker Orders: DISABLED</span>
    </div>
  `;

  const provStatusEl = document.getElementById("badge-provenance-status");
  if (provStatusEl) {
    if (isSynthetic) {
      provStatusEl.textContent = "SYNTHETIC TEST DATA";
      provStatusEl.style.color = "var(--accent-amber, #f59e0b)";
    } else {
      provStatusEl.textContent = "SHA-256 Provenance Verified ✓";
      provStatusEl.style.color = "var(--accent-green)";
    }
  }

  const wfStatusEl = document.getElementById("badge-wf-status");
  if (wfStatusEl) {
    wfStatusEl.textContent = isSynthetic ? "SYNTHETIC TEST DATA" : "WALK-FORWARD VERIFIED";
  }
}

function renderMarketOverview(mo) {
  // Nifty 50 Card
  const n50El = document.getElementById("nifty-card");
  if (n50El) {
    const isUp = mo.nifty_50.change_pct >= 0;
    n50El.innerHTML = `
      <div class="metric-label">Nifty 50</div>
      <div class="metric-value">₹${mo.nifty_50.ltp.toLocaleString("en-IN")}</div>
      <div class="metric-sub ${isUp ? 'text-green' : 'text-red'}">
        ${isUp ? '▲' : '▼'} ${Math.abs(mo.nifty_50.change_pct)}% (${mo.nifty_50.trend})
      </div>
    `;
  }

  // Bank Nifty Card
  const bnEl = document.getElementById("banknifty-card");
  if (bnEl) {
    const isUp = mo.bank_nifty.change_pct >= 0;
    bnEl.innerHTML = `
      <div class="metric-label">Nifty Bank</div>
      <div class="metric-value">₹${mo.bank_nifty.ltp.toLocaleString("en-IN")}</div>
      <div class="metric-sub ${isUp ? 'text-green' : 'text-red'}">
        ${isUp ? '▲' : '▼'} ${Math.abs(mo.bank_nifty.change_pct)}% (${mo.bank_nifty.trend})
      </div>
    `;
  }

  // India VIX Card
  const vixEl = document.getElementById("vix-card");
  if (vixEl) {
    const isHigh = mo.india_vix.value > 18.0;
    vixEl.innerHTML = `
      <div class="metric-label">India VIX</div>
      <div class="metric-value ${isHigh ? 'text-amber' : 'text-green'}">${mo.india_vix.value.toFixed(2)}</div>
      <div class="metric-sub ${isHigh ? 'text-amber' : 'text-green'}">
        ${mo.india_vix.regime}
      </div>
    `;
  }

  // Breadth & Flow Card
  const brEl = document.getElementById("breadth-card");
  if (brEl) {
    const b = mo.market_breadth;
    const f = mo.fii_dii_flow;
    brEl.innerHTML = `
      <div class="metric-label">Breadth & FII Flow</div>
      <div class="metric-value text-blue">A/D: ${b.advance_decline_ratio.toFixed(2)}</div>
      <div class="metric-sub" style="color: var(--text-muted)">
        ${b.advances} Adv / ${b.declines} Dec | FII Net: ₹${f.fii_net_crores > 0 ? '+' : ''}${f.fii_net_crores} Cr
      </div>
    `;
  }

  // Central Posture Hero Banner
  const post = mo.central_posture;
  const postEl = document.getElementById("posture-banner-container");
  if (postEl) {
    postEl.innerHTML = `
      <div class="posture-banner">
        <div class="posture-badge-box">
          <div class="posture-badge ${post.posture}">${post.posture.replace('_', ' ')}</div>
          <div>
            <div style="font-size: 13px; color: var(--text-muted);">Central Intelligence Council Posture</div>
            <div style="font-weight: 600; color: #fff; margin-top: 2px;">${post.summary}</div>
          </div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 12px; color: var(--text-muted);">Council Conviction</div>
          <div style="font-size: 20px; font-weight: 700; color: var(--accent-green); font-family: var(--font-mono);">
            ${post.confidence_pct}%
          </div>
          <div style="font-size: 11px; color: var(--text-dim);">
            Favored: ${post.favored_sectors.join(', ') || 'Broad Market'}
          </div>
        </div>
      </div>
    `;
  }
}

function renderCouncilDebate(debate, posture) {
  const container = document.getElementById("council-debate-container");
  if (!container) return;

  const bullHtml = debate.bull_arguments.map(a => `<li class="argument-item">${a}</li>`).join('');
  const bearHtml = debate.bear_arguments.map(a => `<li class="argument-item">${a}</li>`).join('');
  const triggersHtml = posture.reassessment_triggers.map(t => `<li class="argument-item">${t}</li>`).join('');

  container.innerHTML = `
    <div class="debate-container">
      <div class="debate-column bull">
        <div style="font-weight: 700; color: var(--accent-green); font-size: 13px; display: flex; justify-content: space-between;">
          <span>BULL RESEARCHER THESIS</span>
          <span class="badge badge-green">UPSIDE CATALYSTS</span>
        </div>
        <ul class="argument-list">
          ${bullHtml || '<li class="argument-item">No strong bullish catalyst registered</li>'}
        </ul>
      </div>
      <div class="debate-column bear">
        <div style="font-weight: 700; color: var(--accent-red); font-size: 13px; display: flex; justify-content: space-between;">
          <span>BEAR RESEARCHER THESIS</span>
          <span class="badge badge-red">RISK FACTORS</span>
        </div>
        <ul class="argument-list">
          ${bearHtml || '<li class="argument-item">No major macro risks registered</li>'}
        </ul>
      </div>
    </div>
    <div style="margin-top: 12px; background: #121926; border-radius: 6px; padding: 12px; border: 1px solid var(--border-subtle);">
      <div style="font-size: 11px; text-transform: uppercase; color: var(--text-dim); font-weight: 600;">Reassessment Triggers (Asynchronous Broadcasts)</div>
      <ul class="argument-list" style="margin-top: 6px;">
        ${triggersHtml}
      </ul>
    </div>
  `;
}

function renderStockScanner(stocks) {
  const container = document.getElementById("scanner-cards-container");
  if (!container) return;

  container.innerHTML = stocks.map(stk => {
    const isBull = stk.technical_signal === 'BULLISH';
    const isBear = stk.technical_signal === 'BEARISH';
    const badgeClass = isBull ? 'badge-green' : isBear ? 'badge-red' : 'badge-blue';
    const actionClass = stk.trading_agent_action === 'ENTER_LONG' ? 'badge-green' : 'badge-amber';

    const sparkSvg = generateSparklineSvg(stk.recent_candles);

    return `
      <div class="stock-card">
        <div class="stock-card-header">
          <div>
            <div style="font-weight: 700; font-size: 16px; color: #fff;">${stk.symbol}</div>
            <div style="font-size: 11px; color: var(--text-muted);">${stk.company_name}</div>
          </div>
          <div style="text-align: right;">
            <div class="metric-value" style="font-size: 18px;">₹${stk.current_price.toFixed(2)}</div>
            <span class="badge ${badgeClass}">${stk.technical_signal} (${stk.technical_confidence}%)</span>
          </div>
        </div>

        <div style="background: #0f1523; border-radius: 4px; padding: 6px; margin: 4px 0;">
          ${sparkSvg}
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); background: #121926; padding: 8px; border-radius: 4px;">
          <div>VWAP: <span style="color: #fff;">${stk.vwap}</span></div>
          <div>EMA 9: <span style="color: #fff;">${stk.ema_9}</span></div>
          <div>EMA 21: <span style="color: #fff;">${stk.ema_21}</span></div>
          <div>RSI: <span style="color: #fff;">${stk.rsi_14}</span></div>
          <div>Vol: <span style="color: #fff;">${stk.volume_ratio}x</span></div>
          <div>Above VWAP: <span style="color: ${stk.above_vwap ? 'var(--accent-green)' : 'var(--accent-red)'}">${stk.above_vwap ? 'YES' : 'NO'}</span></div>
        </div>

        <div style="border-top: 1px solid var(--border-subtle); padding-top: 8px; display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 11px; color: var(--text-dim);">Trader Action:</span>
          <span class="badge ${actionClass}" style="font-size: 12px; font-weight: 700;">${stk.trading_agent_action}</span>
        </div>
      </div>
    `;
  }).join('');
}

function generateSparklineSvg(candles) {
  if (!candles || candles.length < 2) return '';
  const closes = candles.map(c => c.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = (max - min) || 1;

  const width = 300;
  const height = 50;
  const points = closes.map((val, idx) => {
    const x = (idx / (closes.length - 1)) * width;
    const y = height - ((val - min) / range) * (height - 8) - 4;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const strokeColor = closes[closes.length - 1] >= closes[0] ? '#10b981' : '#ef4444';

  return `
    <svg viewBox="0 0 ${width} ${height}" class="sparkline-svg">
      <polyline fill="none" stroke="${strokeColor}" stroke-width="2" points="${points}" />
    </svg>
  `;
}

function renderCrossExchange(arbs) {
  const tbody = document.getElementById("cross-exchange-tbody");
  if (!tbody) return;

  tbody.innerHTML = arbs.map(a => {
    const statusClass = a.is_actionable ? 'badge-green' : 'badge-amber';
    return `
      <tr>
        <td style="font-weight: 700; color: #fff;">${a.symbol}</td>
        <td style="color: var(--text-dim); font-size: 11px;">${a.isin}</td>
        <td>Buy ${a.buy_exchange} @ ₹${a.buy_ask_price.toFixed(2)}</td>
        <td>Sell ${a.sell_exchange} @ ₹${a.sell_bid_price.toFixed(2)}</td>
        <td style="color: #fff;">₹${a.gross_spread_inr.toFixed(2)} (${a.gross_spread_pct.toFixed(2)}%)</td>
        <td style="color: var(--accent-red);">₹${a.total_friction_inr.toFixed(2)}</td>
        <td style="font-weight: 700; color: ${a.net_profit_inr > 0 ? 'var(--accent-green)' : 'var(--text-dim)'};">
          ₹${a.net_profit_inr.toFixed(2)} (${a.net_return_pct.toFixed(2)}%)
        </td>
        <td><span class="badge ${statusClass}">${a.status.replace('_', ' ')}</span></td>
      </tr>
    `;
  }).join('');
}

function renderPaperPortfolio(port) {
  // Metrics
  document.getElementById("port-cash").textContent = `₹${port.cash_inr.toLocaleString('en-IN')}`;
  document.getElementById("port-equity").textContent = `₹${port.total_equity_inr.toLocaleString('en-IN')}`;
  document.getElementById("port-realized").textContent = `₹${port.realized_pnl_inr.toLocaleString('en-IN')}`;
  document.getElementById("port-unrealized").textContent = `₹${port.unrealized_pnl_inr.toLocaleString('en-IN')}`;
  document.getElementById("port-drawdown").textContent = `${port.max_drawdown_pct.toFixed(2)}%`;
  document.getElementById("port-winrate").textContent = `${port.win_rate_pct.toFixed(1)}%`;

  // Positions Table
  const posTbody = document.getElementById("positions-tbody");
  if (posTbody) {
    if (port.open_positions.length === 0) {
      posTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 16px;">No open positions (100% Cash Allocation)</td></tr>`;
    } else {
      posTbody.innerHTML = port.open_positions.map(p => `
        <tr>
          <td style="font-weight: 700; color: #fff;">${p.symbol}</td>
          <td>${p.exchange}</td>
          <td>${p.quantity}</td>
          <td>₹${p.entry_price.toFixed(2)}</td>
          <td>₹${p.current_price.toFixed(2)}</td>
          <td class="${p.unrealized_pnl_inr >= 0 ? 'text-green' : 'text-red'}">
            ₹${p.unrealized_pnl_inr.toFixed(2)} (${p.return_pct > 0 ? '+' : ''}${p.return_pct.toFixed(2)}%)
          </td>
          <td style="font-size: 11px; color: var(--text-muted)">SL: ₹${p.stop_loss || '-'} | TP: ₹${p.target || '-'}</td>
        </tr>
      `).join('');
    }
  }

  // Trade History Table
  const histTbody = document.getElementById("trades-tbody");
  if (histTbody) {
    if (port.closed_trades.length === 0) {
      histTbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-dim); padding: 16px;">No closed trades completed in current session</td></tr>`;
    } else {
      histTbody.innerHTML = port.closed_trades.slice(-5).map(t => `
        <tr>
          <td style="font-weight: 700; color: #fff;">${t.symbol}</td>
          <td>${t.quantity}</td>
          <td>₹${t.entry_price.toFixed(2)}</td>
          <td>₹${t.exit_price.toFixed(2)}</td>
          <td style="color: var(--accent-red);">₹${t.total_friction_inr.toFixed(2)}</td>
          <td class="${t.net_pnl_inr >= 0 ? 'text-green' : 'text-red'}" style="font-weight: 700;">
            ₹${t.net_pnl_inr.toFixed(2)} (${t.return_pct > 0 ? '+' : ''}${t.return_pct.toFixed(2)}%)
          </td>
        </tr>
      `).join('');
    }
  }
}

function renderStrategyPerformance(strats) {
  const tbody = document.getElementById("strategy-tbody");
  if (!tbody) return;

  tbody.innerHTML = strats.map(s => `
    <tr>
      <td style="font-weight: 700; color: #fff;">${s.strategy_name}</td>
      <td style="color: var(--text-dim); font-size: 11px;">${s.origin}</td>
      <td><span class="badge ${s.validation_status === 'APPROVED_PAPER' ? 'badge-green' : 'badge-blue'}">${s.validation_status}</span></td>
      <td>${s.win_rate_pct.toFixed(1)}%</td>
      <td>${s.profit_factor.toFixed(2)}</td>
      <td class="${s.total_return_pct >= 0 ? 'text-green' : 'text-red'}">${s.total_return_pct > 0 ? '+' : ''}${s.total_return_pct.toFixed(2)}%</td>
      <td>${s.max_drawdown_pct.toFixed(2)}%</td>
      <td style="color: var(--accent-red);">₹${s.friction_paid_inr.toFixed(2)}</td>
    </tr>
  `).join('');
}

function renderCostMonitoring(cm) {
  document.getElementById("cost-daily-spend").textContent = `$${cm.current_daily_spend_usd.toFixed(4)}`;
  document.getElementById("cost-daily-budget").textContent = `$${cm.daily_budget_usd.toFixed(2)}`;
  document.getElementById("cost-calls-count").textContent = cm.total_calls_count;
  document.getElementById("cost-tokens-count").textContent = cm.total_tokens_consumed.toLocaleString();

  const tbody = document.getElementById("agent-costs-tbody");
  if (!tbody) return;

  tbody.innerHTML = Object.entries(cm.model_routing_map).map(([agent, model]) => {
    return `
      <tr>
        <td style="font-weight: 600; color: #fff;">${agent}</td>
        <td><code>${model}</code></td>
        <td><span class="badge badge-green">ACTIVE</span></td>
      </tr>
    `;
  }).join('');
}
