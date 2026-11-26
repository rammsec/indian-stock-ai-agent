/**
 * BharatAlpha AI — Unified Causal Intraday Candle Monitoring Engine (CYCLE-010 Phase 4)
 *
 * Pure, deterministic, zero-lookahead JS engine mirroring Python Phases 1-3:
 *  - ₹0.05 NSE tick rounding (roundToNseTick)
 *  - Statutory Indian equity intraday transaction costs (calculateIndianIntradayCosts)
 *  - Causal indicators over warmup + revealed bars (EMA9, EMA21, Session VWAP, ATR14, RSI14, RVOL14)
 *  - 10 candlestick patterns + structural events (False Breakout, Failed Breakdown, VWAP Reclaim/Reject, ORB, Pullback)
 *  - Stateful 9-call CandleTimingAgent (OPEN_LONG, OPEN_SHORT, WAIT, HOLD, EXIT_LONG, COVER_SHORT, NO_TRADE, CANCEL_SETUP, REJECT)
 *  - Monotonic trailing stop ratcheting on HOLD
 *  - 7-Point Mandatory Execution Gate (protective exits always permitted on mid-session Human permission flip)
 *  - 3-State Deterministic Risk Engine (ACTIVE -> REDUCING @ 15:05 IST -> HALTED @ 2.5% daily / 3.0% MTM DD; 15% position cap; 14:45 cutoff)
 *  - Next-bar-open causal execution (PendingOrder queued at bar t close, filled at bar t+1 open with gap-stop/gap-target cancel)
 *  - Intrabar OHLC stop-first evaluation & 15:15 IST mandatory square-off
 *  - Strictly post-session counterfactual outcome review (MFE/MAE, saved loss ₹/%, missed move %)
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.BharatAlphaEngine = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const NSE_TICK_SIZE_INR = 0.05;

  function roundToNseTick(price, tickSize = NSE_TICK_SIZE_INR) {
    return Number((Math.round(Number(price) / tickSize) * tickSize).toFixed(2));
  }

  /**
   * Exact parity with trader/portfolio/cost_calculator.py::calculate_intraday_costs
   */
  function calculateIndianIntradayCosts(price, quantity, side, exchange = "NSE", slippageBps = 5.0) {
    const turnover = Number((price * quantity).toFixed(2));
    const brokerage = Math.min(turnover * 0.0003, 20.0);
    const stt = side === "SELL" ? turnover * 0.00025 : 0.0;
    const exchangeCharges = turnover * (exchange === "BSE" ? 0.0000375 : 0.0000297);
    const sebiFee = turnover * 0.000001;
    const stampDuty = side === "BUY" ? turnover * 0.00003 : 0.0;
    const gst = 0.18 * (brokerage + exchangeCharges + sebiFee);
    const slippage = turnover * (slippageBps / 10000.0);
    const totalFriction = Number((brokerage + stt + exchangeCharges + sebiFee + stampDuty + gst + slippage).toFixed(4));
    return {
      turnover,
      brokerage: Number(brokerage.toFixed(4)),
      stt: Number(stt.toFixed(4)),
      exchangeCharges: Number(exchangeCharges.toFixed(4)),
      sebiFee: Number(sebiFee.toFixed(4)),
      stampDuty: Number(stampDuty.toFixed(4)),
      gst: Number(gst.toFixed(4)),
      slippage: Number(slippage.toFixed(4)),
      totalFrictionInr: totalFriction,
    };
  }

  function computeEmaSeries(values, period) {
    const out = new Array(values.length).fill(null);
    if (values.length === 0) return out;
    const alpha = 2.0 / (period + 1.0);
    let ema = values[0];
    out[0] = ema;
    for (let i = 1; i < values.length; i++) {
      ema = alpha * values[i] + (1.0 - alpha) * ema;
      out[i] = ema;
    }
    return out;
  }

  function computeAtrSeries(bars, period = 14) {
    const n = bars.length;
    const tr = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const h = bars[i].h;
      const l = bars[i].l;
      if (i === 0) {
        tr[i] = h - l;
      } else {
        const pc = bars[i - 1].c;
        tr[i] = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
      }
    }
    const atr = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
      const start = Math.max(0, i - period + 1);
      const count = i - start + 1;
      if (count < Math.min(period, 5)) continue;
      let sum = 0;
      for (let j = start; j <= i; j++) sum += tr[j];
      atr[i] = sum / count;
    }
    return atr;
  }

  function computeRsiSeries(closes, period = 14) {
    const n = closes.length;
    const out = new Array(n).fill(null);
    if (n <= period) return out;
    let avgGain = 0;
    let avgLoss = 0;
    for (let i = 1; i <= period; i++) {
      const d = closes[i] - closes[i - 1];
      if (d > 0) avgGain += d;
      else avgLoss += -d;
    }
    avgGain /= period;
    avgLoss /= period;
    out[period] = avgLoss === 0 ? 100.0 : 100.0 - 100.0 / (1.0 + avgGain / avgLoss);
    for (let i = period + 1; i < n; i++) {
      const d = closes[i] - closes[i - 1];
      const gain = d > 0 ? d : 0;
      const loss = d < 0 ? -d : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
      out[i] = avgLoss === 0 ? 100.0 : 100.0 - 100.0 / (1.0 + avgGain / avgLoss);
    }
    return out;
  }

  function computeSessionVwapSeries(bars) {
    const out = new Array(bars.length).fill(null);
    let curDate = null;
    let cumTpVol = 0;
    let cumVol = 0;
    for (let i = 0; i < bars.length; i++) {
      const d = bars[i].t.slice(0, 10);
      if (d !== curDate) {
        curDate = d;
        cumTpVol = 0;
        cumVol = 0;
      }
      const tp = (bars[i].h + bars[i].l + bars[i].c) / 3.0;
      const v = Number(bars[i].v || 0);
      cumTpVol += tp * v;
      cumVol += v;
      out[i] = cumVol > 0 ? cumTpVol / cumVol : null;
    }
    return out;
  }

  function computeRvol(bars, window = 14) {
    const n = bars.length;
    if (n === 0) return { rvol: null, status: "UNAVAILABLE" };
    const curr = bars[n - 1];
    if (curr.v === 0 && curr.h > curr.l) {
      return { rvol: null, status: "VOLUME_UNREPORTED_BY_PROVIDER" };
    }
    const priorNonZero = [];
    for (let i = Math.max(0, n - 1 - window); i < n - 1; i++) {
      if (bars[i].v > 0) priorNonZero.push(bars[i].v);
    }
    if (priorNonZero.length === 0) return { rvol: null, status: "NORMAL_VOLUME" };
    const meanV = priorNonZero.reduce((a, b) => a + b, 0) / priorNonZero.length;
    const rvol = meanV > 0 ? curr.v / meanV : null;
    const status = rvol >= 1.25 ? "HIGH_VOLUME_CONFIRMATION" : rvol < 0.5 ? "LOW_VOLUME_WEAK" : "NORMAL_VOLUME";
    return { rvol, status };
  }

  function detectCandlePatterns(curr, prev) {
    const range = Math.max(0, curr.h - curr.l);
    const body = Math.abs(curr.c - curr.o);
    const bodyRatio = range > 0 ? body / range : 0;
    const upperWickRatio = range > 0 ? (curr.h - Math.max(curr.o, curr.c)) / range : 0;
    const lowerWickRatio = range > 0 ? (Math.min(curr.o, curr.c) - curr.l) / range : 0;
    const clv = range > 0 ? (curr.c - curr.l) / range : 0.5;
    const isBull = curr.c > curr.o;
    const isBear = curr.c < curr.o;

    const patterns = [];
    if (range === 0 || bodyRatio <= 0.12) patterns.push("DOJI");
    if (bodyRatio >= 0.8 && upperWickRatio <= 0.1 && lowerWickRatio <= 0.1) {
      if (isBull) patterns.push("BULLISH_MARUBOZU");
      else if (isBear) patterns.push("BEARISH_MARUBOZU");
    }
    if (lowerWickRatio >= 0.55 && upperWickRatio <= 0.18 && bodyRatio <= 0.35 && clv >= 0.6) {
      patterns.push("HAMMER");
    }
    if (lowerWickRatio >= 0.6 && bodyRatio <= 0.3 && clv >= 0.65) {
      patterns.push("BULLISH_PIN_BAR");
    }
    if (upperWickRatio >= 0.55 && lowerWickRatio <= 0.18 && bodyRatio <= 0.35 && clv <= 0.4) {
      patterns.push("SHOOTING_STAR");
    }
    if (upperWickRatio >= 0.6 && bodyRatio <= 0.3 && clv <= 0.35) {
      patterns.push("BEARISH_PIN_BAR");
    }
    if (prev) {
      const prevBody = Math.abs(prev.c - prev.o);
      if (prev.c < prev.o && isBull && prevBody > 0 && curr.o <= prev.c && curr.c >= prev.o && body > prevBody) {
        patterns.push("BULLISH_ENGULFING");
      }
      if (prev.c > prev.o && isBear && prevBody > 0 && curr.o >= prev.c && curr.c <= prev.o && body > prevBody) {
        patterns.push("BEARISH_ENGULFING");
      }
      if (curr.h <= prev.h && curr.l >= prev.l && (curr.h < prev.h || curr.l > prev.l)) {
        patterns.push("INSIDE_BAR");
      }
    }
    return {
      patterns,
      bodyRatio,
      upperWickRatio,
      lowerWickRatio,
      clv,
      isBull,
      isBear,
    };
  }

  function analyzeCausalBars(visibleBars, sessionDateStr) {
    const n = visibleBars.length;
    const closes = visibleBars.map((b) => b.c);
    const ema9 = computeEmaSeries(closes, 9);
    const ema21 = computeEmaSeries(closes, 21);
    const atr = computeAtrSeries(visibleBars, 14);
    const rsi = computeRsiSeries(closes, 14);
    const vwap = computeSessionVwapSeries(visibleBars);
    const { rvol, status: volumeRegime } = computeRvol(visibleBars, 14);

    const curr = visibleBars[n - 1];
    const prev = n >= 2 ? visibleBars[n - 2] : null;
    const anatomy = detectCandlePatterns(curr, prev);

    const e9 = ema9[n - 1];
    const e21 = ema21[n - 1];
    const atrVal = atr[n - 1] || Math.max(0.05, curr.c * 0.01);
    const vwapVal = vwap[n - 1];
    const vwapPrev = n >= 2 ? vwap[n - 2] : null;

    // 12-bar half-window swing structure (exact parity with CandleIntelligenceEngine)
    const swingWin = visibleBars.slice(Math.max(0, n - 12));
    const half = Math.floor(swingWin.length / 2);
    let swingStructure = "RANGE";
    if (half >= 2) {
      const h1 = Math.max(...swingWin.slice(0, half).map((b) => b.h));
      const l1 = Math.min(...swingWin.slice(0, half).map((b) => b.l));
      const h2 = Math.max(...swingWin.slice(half).map((b) => b.h));
      const l2 = Math.min(...swingWin.slice(half).map((b) => b.l));
      if (h2 > h1 && l2 > l1) swingStructure = "UPTREND";
      else if (h2 < h1 && l2 < l1) swingStructure = "DOWNTREND";
    }

    // 10-bar breakout / rejection / 2-bar trap
    const priorWin = visibleBars.slice(Math.max(0, n - 11), n - 1);
    const priorHigh = priorWin.length ? Math.max(...priorWin.map((b) => b.h)) : curr.h;
    const priorLow = priorWin.length ? Math.min(...priorWin.map((b) => b.l)) : curr.l;
    const priorPrevWin = visibleBars.slice(Math.max(0, n - 12), Math.max(0, n - 2));
    const priorHPrev = priorPrevWin.length ? Math.max(...priorPrevWin.map((b) => b.h)) : priorHigh;
    const priorLPrev = priorPrevWin.length ? Math.min(...priorPrevWin.map((b) => b.l)) : priorLow;

    let breakoutState = "INSIDE_RANGE";
    if ((curr.h >= priorHigh && curr.c < priorHigh && anatomy.clv <= 0.45) || (prev && prev.c > priorHPrev && curr.c <= priorHPrev)) {
      breakoutState = "FALSE_BREAKOUT";
    } else if ((curr.l <= priorLow && curr.c > priorLow && anatomy.clv >= 0.55) || (prev && prev.c < priorLPrev && curr.c >= priorLPrev)) {
      breakoutState = "FAILED_BREAKDOWN";
    } else if (curr.c > priorHigh) {
      breakoutState = "BREAKOUT_HIGH";
    } else if (curr.c < priorLow) {
      breakoutState = "BREAKDOWN_LOW";
    }

    // VWAP event
    let vwapEvent = "UNAVAILABLE";
    if (vwapVal !== null && vwapVal > 0) {
      if ((vwapPrev !== null && prev && prev.c < vwapPrev && curr.c > vwapVal) || (curr.l <= vwapVal && curr.c > vwapVal && anatomy.clv >= 0.65)) {
        vwapEvent = "VWAP_RECLAIM";
      } else if ((vwapPrev !== null && prev && prev.c > vwapPrev && curr.c < vwapVal) || (curr.h >= vwapVal && curr.c < vwapVal && anatomy.upperWickRatio >= 0.35 && anatomy.clv <= 0.4)) {
        vwapEvent = "VWAP_REJECTION";
      } else {
        vwapEvent = curr.c >= vwapVal ? "ABOVE_VWAP" : "BELOW_VWAP";
      }
    }

    const emaAlignment = curr.c > e9 && e9 > e21 ? "BULLISH" : curr.c < e9 && e9 < e21 ? "BEARISH" : "NEUTRAL";

    // Composite price action score [-1, +1] matching CandleIntelligenceEngine
    let pa = 0;
    if (anatomy.isBull) pa += 0.25 * anatomy.bodyRatio;
    else if (anatomy.isBear) pa -= 0.25 * anatomy.bodyRatio;
    pa += (anatomy.clv - 0.5) * 0.25;

    const bullPats = ["BULLISH_MARUBOZU", "HAMMER", "BULLISH_PIN_BAR", "BULLISH_ENGULFING"];
    const bearPats = ["BEARISH_MARUBOZU", "SHOOTING_STAR", "BEARISH_PIN_BAR", "BEARISH_ENGULFING"];
    if (anatomy.patterns.some((p) => bullPats.includes(p))) pa += 0.22;
    if (anatomy.patterns.some((p) => bearPats.includes(p))) pa -= 0.22;
    if (swingStructure === "UPTREND") pa += 0.2;
    else if (swingStructure === "DOWNTREND") pa -= 0.2;
    if (emaAlignment === "BULLISH") pa += 0.2;
    else if (emaAlignment === "BEARISH") pa -= 0.2;
    if (vwapEvent === "ABOVE_VWAP" || vwapEvent === "VWAP_RECLAIM") pa += 0.1;
    else if (vwapEvent === "BELOW_VWAP" || vwapEvent === "VWAP_REJECTION") pa -= 0.1;
    if (breakoutState === "BREAKOUT_HIGH" || breakoutState === "FAILED_BREAKDOWN") pa += 0.15;
    else if (breakoutState === "BREAKDOWN_LOW" || breakoutState === "FALSE_BREAKOUT") pa -= 0.15;
    pa = Number(Math.max(-1, Math.min(1, pa)).toFixed(4));

    const candleBias = pa >= 0.2 ? "BULLISH" : pa <= -0.2 ? "BEARISH" : "NEUTRAL";
    const recentSwingLow = Math.min(...swingWin.map((b) => b.l));
    const recentSwingHigh = Math.max(...swingWin.map((b) => b.h));

    return {
      ema9Series: ema9,
      ema21Series: ema21,
      vwapSeries: vwap,
      ema9: e9,
      ema21: e21,
      vwap: vwapVal,
      atr: atrVal,
      atrPct: curr.c > 0 ? (atrVal / curr.c) * 100.0 : 0,
      rsi14: rsi[n - 1],
      rvol,
      volumeRegime,
      patterns: anatomy.patterns,
      anatomy,
      swingStructure,
      emaAlignment,
      vwapEvent,
      breakoutState,
      priceActionScore: pa,
      candleBias,
      recentSwingLow,
      recentSwingHigh,
    };
  }

  function humanPermissionAllows(permission, actionType) {
    const isExit = actionType === "EXIT_LONG" || actionType === "COVER_SHORT" || actionType === "RISK_EXIT";
    if (isExit) return { allowed: true, reason: `${permission} always permits protective exit (${actionType})` };
    if (permission === "STOP_ALL" || permission === "PAUSE" || permission === "NO_NEW_ENTRIES" || permission === "EXIT_ONLY") {
      return { allowed: false, reason: `HUMAN permission ${permission} blocks new entries` };
    }
    if (permission === "ALLOW_LONG" && actionType === "OPEN_SHORT") {
      return { allowed: false, reason: "HUMAN permission ALLOW_LONG prohibits short entries" };
    }
    if (permission === "ALLOW_SHORT" && actionType === "OPEN_LONG") {
      return { allowed: false, reason: "HUMAN permission ALLOW_SHORT prohibits long entries" };
    }
    return { allowed: true, reason: `${permission} permits ${actionType}` };
  }

  function evaluateCausal5mMtfTrend(bars) {
    if (!bars || bars.length < 5) return "NEUTRAL";
    const buckets = new Map();
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i];
      if (!b || !b.t) continue;
      const dayPart = b.t.slice(0, 10);
      const hh = Number(b.t.slice(11, 13));
      const mm = Number(b.t.slice(14, 16));
      const minsSinceOpen = hh * 60 + mm - (9 * 60 + 15);
      if (minsSinceOpen < 0) continue;
      const bIdx = Math.floor(minsSinceOpen / 5);
      buckets.set(`${dayPart}#${bIdx}`, b.c);
    }
    const closes = Array.from(buckets.values());
    if (closes.length < 5) return "NEUTRAL";
    const ema = (vals, span) => {
      const k = 2 / (span + 1);
      let e = vals[0];
      for (let i = 1; i < vals.length; i++) e = vals[i] * k + e * (1 - k);
      return e;
    };
    const lastClose = closes[closes.length - 1];
    const ema9 = ema(closes, 9);
    if (closes.length >= 21) {
      const ema21 = ema(closes, 21);
      if (lastClose >= ema9 && ema9 >= ema21) return "BULLISH";
      if (lastClose <= ema9 && ema9 <= ema21) return "BEARISH";
      return "NEUTRAL";
    }
    if (lastClose > ema9) return "BULLISH";
    if (lastClose < ema9) return "BEARISH";
    return "NEUTRAL";
  }

  function computeChandelierTrailingStop({
    side,
    entryPrice,
    currentClose,
    currentStopLoss,
    highestPrice,
    lowestPrice,
    atr14,
    initialRiskDistance,
    roundTripFeePerShare = 0.0,
    chandelierAtrMult = 2.2,
    breakevenTriggerR = 1.2,
  }) {
    if (atr14 <= 0 && initialRiskDistance <= 0) {
      return { stopLoss: currentStopLoss, breakevenLocked: false };
    }
    let breakevenLocked = false;
    const candidates = [];
    if (currentStopLoss !== null && currentStopLoss !== undefined) {
      candidates.push(Number(currentStopLoss));
    }
    if (side === "LONG") {
      if (initialRiskDistance > 0 && highestPrice >= entryPrice + breakevenTriggerR * initialRiskDistance) {
        candidates.push(roundToNseTick(entryPrice + Math.max(0, roundTripFeePerShare) + 0.05));
        breakevenLocked = true;
      }
      if (atr14 > 0) {
        const ch = roundToNseTick(highestPrice - chandelierAtrMult * atr14);
        if (ch > 0) candidates.push(ch);
      }
      if (candidates.length === 0) return { stopLoss: null, breakevenLocked };
      let best = Math.max(...candidates);
      const maxValid = roundToNseTick(Math.max(0.05, currentClose - 0.05));
      best = currentStopLoss !== null && currentStopLoss !== undefined ? Math.max(Number(currentStopLoss), Math.min(best, maxValid)) : Math.min(best, maxValid);
      return { stopLoss: roundToNseTick(best), breakevenLocked };
    }
    if (initialRiskDistance > 0 && lowestPrice <= entryPrice - breakevenTriggerR * initialRiskDistance) {
      candidates.push(roundToNseTick(Math.max(0.05, entryPrice - Math.max(0, roundTripFeePerShare) - 0.05)));
      breakevenLocked = true;
    }
    if (atr14 > 0) {
      candidates.push(roundToNseTick(lowestPrice + chandelierAtrMult * atr14));
    }
    if (candidates.length === 0) return { stopLoss: null, breakevenLocked };
    let best = Math.min(...candidates);
    const minValid = roundToNseTick(currentClose + 0.05);
    best = currentStopLoss !== null && currentStopLoss !== undefined ? Math.min(Number(currentStopLoss), Math.max(best, minValid)) : Math.max(best, minValid);
    return { stopLoss: roundToNseTick(best), breakevenLocked };
  }

  function computeChartOverlays(bars, sessionDate) {
    const empty = {
      adx14: null, plusDi14: null, minusDi14: null, marketRegime: "WARMING_UP",
      supertrendValue: null, supertrendDirection: "NEUTRAL", supertrendSeries: [], supertrendDirSeries: [],
      pdh: null, pdl: null, orbHigh: null, orbLow: null, orbReady: false, orbState: "ORB_FORMING",
      vwapUpper1Series: [], vwapLower1Series: [], vwapUpper2Series: [], vwapLower2Series: [],
      vwapStretch: "UNAVAILABLE", nearKeyLevel: false, nearestKeyLevelName: null,
      liquiditySweep: "NONE", recentFvg: null,
    };
    if (!bars || bars.length === 0) return empty;
    const n = bars.length;
    const highs = bars.map((b) => b.h);
    const lows = bars.map((b) => b.l);
    const closes = bars.map((b) => b.c);
    const vols = bars.map((b) => b.v || 0);

    // 1. Causal Wilder ADX(14)
    let adx14 = null, plusDi14 = null, minusDi14 = null, marketRegime = "WARMING_UP";
    const adxP = 14;
    if (n >= adxP + 2) {
      const tr = new Array(n).fill(0), pdm = new Array(n).fill(0), mdm = new Array(n).fill(0);
      for (let i = 1; i < n; i++) {
        const up = highs[i] - highs[i - 1], dn = lows[i - 1] - lows[i];
        pdm[i] = up > dn && up > 0 ? up : 0;
        mdm[i] = dn > up && dn > 0 ? dn : 0;
        tr[i] = Math.max(highs[i] - lows[i], Math.abs(highs[i] - closes[i - 1]), Math.abs(lows[i] - closes[i - 1]));
      }
      let smTr = 0, smP = 0, smM = 0;
      for (let i = 1; i <= adxP; i++) { smTr += tr[i]; smP += pdm[i]; smM += mdm[i]; }
      const dxs = [];
      for (let i = adxP; i < n; i++) {
        if (i > adxP) {
          smTr = smTr - smTr / adxP + tr[i];
          smP = smP - smP / adxP + pdm[i];
          smM = smM - smM / adxP + mdm[i];
        }
        const pdi = smTr > 0 ? (100 * smP) / smTr : 0;
        const mdi = smTr > 0 ? (100 * smM) / smTr : 0;
        plusDi14 = pdi; minusDi14 = mdi;
        const denom = pdi + mdi;
        dxs.push(denom > 0 ? (100 * Math.abs(pdi - mdi)) / denom : 0);
      }
      if (dxs.length > 0) {
        if (dxs.length < adxP) {
          adx14 = dxs.reduce((a, b) => a + b, 0) / dxs.length;
        } else {
          adx14 = dxs.slice(0, adxP).reduce((a, b) => a + b, 0) / adxP;
          for (let i = adxP; i < dxs.length; i++) adx14 = (adx14 * (adxP - 1) + dxs[i]) / adxP;
        }
        marketRegime = adx14 < 20 ? "CHOPPY_SIDEWAYS" : adx14 <= 40 ? "TRENDING" : "STRONG_EXHAUSTION_WATCH";
      }
    }

    // 2. Causal Supertrend(10, 3.0)
    const stP = 10, stMult = 3.0;
    const stSeries = new Array(n).fill(null), stDirSeries = new Array(n).fill(0);
    let supertrendValue = null, supertrendDirection = "NEUTRAL";
    if (n >= stP + 2) {
      const tr = new Array(n).fill(0);
      tr[0] = highs[0] - lows[0];
      for (let i = 1; i < n; i++) {
        tr[i] = Math.max(highs[i] - lows[i], Math.abs(highs[i] - closes[i - 1]), Math.abs(lows[i] - closes[i - 1]));
      }
      const atr = new Array(n).fill(0);
      let s0 = 0;
      for (let i = 0; i < stP; i++) s0 += tr[i];
      atr[stP - 1] = s0 / stP;
      for (let i = stP; i < n; i++) atr[i] = (atr[i - 1] * (stP - 1) + tr[i]) / stP;

      const fUp = new Array(n).fill(0), fLo = new Array(n).fill(0), dir = new Array(n).fill(1);
      for (let i = stP - 1; i < n; i++) {
        const hl2 = (highs[i] + lows[i]) * 0.5;
        const bUp = hl2 + stMult * atr[i], bLo = hl2 - stMult * atr[i];
        if (i === stP - 1) {
          fUp[i] = bUp; fLo[i] = bLo; dir[i] = closes[i] >= hl2 ? 1 : -1;
        } else {
          fUp[i] = bUp < fUp[i - 1] || closes[i - 1] > fUp[i - 1] ? bUp : fUp[i - 1];
          fLo[i] = bLo > fLo[i - 1] || closes[i - 1] < fLo[i - 1] ? bLo : fLo[i - 1];
          dir[i] = dir[i - 1] === 1 ? (closes[i] < fLo[i] ? -1 : 1) : (closes[i] > fUp[i] ? 1 : -1);
        }
        stDirSeries[i] = dir[i];
        stSeries[i] = dir[i] === 1 ? fLo[i] : fUp[i];
      }
      supertrendDirection = dir[n - 1] === 1 ? "BULLISH" : "BEARISH";
      supertrendValue = stSeries[n - 1];
    }

    // 3. PDH/PDL, ORB 15m (09:15-09:30 IST), and Session VWAP ±1σ/±2σ bands
    let pdh = null, pdl = null, orbHigh = null, orbLow = null, orbReady = false, orbState = "ORB_FORMING";
    const u1S = new Array(n).fill(null), l1S = new Array(n).fill(null);
    const u2S = new Array(n).fill(null), l2S = new Array(n).fill(null);
    let vwapStretch = "UNAVAILABLE", lastVwap = null;

    let prevDate = null;
    for (let i = 0; i < n; i++) {
      const d = bars[i].t.slice(0, 10);
      if (d < sessionDate && (prevDate === null || d > prevDate)) prevDate = d;
    }
    if (prevDate !== null) {
      for (let i = 0; i < n; i++) {
        if (bars[i].t.slice(0, 10) === prevDate) {
          pdh = pdh === null ? highs[i] : Math.max(pdh, highs[i]);
          pdl = pdl === null ? lows[i] : Math.min(pdl, lows[i]);
        }
      }
    }

    let cumVol = 0, cumPv = 0, cumPv2 = 0;
    let lastSessHHMM = "";
    for (let i = 0; i < n; i++) {
      if (bars[i].t.slice(0, 10) !== sessionDate) continue;
      const hhmm = bars[i].t.slice(11, 16);
      lastSessHHMM = hhmm;
      if (hhmm >= "09:15" && hhmm < "09:30") {
        orbHigh = orbHigh === null ? highs[i] : Math.max(orbHigh, highs[i]);
        orbLow = orbLow === null ? lows[i] : Math.min(orbLow, lows[i]);
      }
      const tp = (highs[i] + lows[i] + closes[i]) / 3.0;
      const v = vols[i];
      if (v > 0) {
        cumVol += v;
        cumPv += tp * v;
        cumPv2 += tp * tp * v;
      }
      if (cumVol > 0) {
        const vw = cumPv / cumVol;
        const sd = Math.sqrt(Math.max(0, cumPv2 / cumVol - vw * vw));
        lastVwap = vw;
        u1S[i] = vw + sd; l1S[i] = vw - sd;
        u2S[i] = vw + 2 * sd; l2S[i] = vw - 2 * sd;
        if (i === n - 1) {
          vwapStretch = sd > 0 && closes[i] > u2S[i] ? "OVEREXTENDED_ABOVE_2SD" : sd > 0 && closes[i] < l2S[i] ? "OVEREXTENDED_BELOW_2SD" : "NORMAL";
        }
      }
    }
    if (orbHigh !== null && orbLow !== null && lastSessHHMM >= "09:29") {
      orbReady = true;
      const lc = closes[n - 1];
      orbState = lc > orbHigh ? "ORB_BREAKOUT_UP" : lc < orbLow ? "ORB_BREAKOUT_DOWN" : "INSIDE_ORB";
    }

    // 4. Liquidity Sweep & 3-Candle Fair Value Gap (FVG)
    let liquiditySweep = "NONE";
    if (n >= 12) {
      let priorLow = Infinity, priorHigh = -Infinity;
      for (let i = n - 11; i < n - 1; i++) {
        if (lows[i] < priorLow) priorLow = lows[i];
        if (highs[i] > priorHigh) priorHigh = highs[i];
      }
      if (lows[n - 1] < priorLow && closes[n - 1] > priorLow) liquiditySweep = "BULLISH_SWEEP";
      else if (highs[n - 1] > priorHigh && closes[n - 1] < priorHigh) liquiditySweep = "BEARISH_SWEEP";
    }
    let recentFvg = null;
    for (let i = n - 1; i >= Math.max(2, n - 12); i--) {
      if (lows[i] > highs[i - 2]) {
        recentFvg = { direction: "BULLISH", low: highs[i - 2], high: lows[i], barIndex: i };
        break;
      }
      if (highs[i] < lows[i - 2]) {
        recentFvg = { direction: "BEARISH", low: highs[i], high: lows[i - 2], barIndex: i };
        break;
      }
    }

    // 5. Key-Level Location Proximity (within 0.35%)
    const lc = closes[n - 1];
    const lvls = [];
    if (pdh !== null) lvls.push(["PDH", pdh]);
    if (pdl !== null) lvls.push(["PDL", pdl]);
    if (orbReady && orbHigh !== null) lvls.push(["ORB_HIGH", orbHigh]);
    if (orbReady && orbLow !== null) lvls.push(["ORB_LOW", orbLow]);
    if (lastVwap !== null) lvls.push(["VWAP", lastVwap]);
    let nearKeyLevel = false, nearestKeyLevelName = null, bestD = Infinity;
    for (const [nm, lv] of lvls) {
      const dist = Math.abs(lc - lv) / Math.max(0.05, lc);
      if (dist < bestD) { bestD = dist; nearestKeyLevelName = nm; }
    }
    if (bestD <= 0.0035) nearKeyLevel = true; else nearestKeyLevelName = null;

    return {
      adx14, plusDi14, minusDi14, marketRegime,
      supertrendValue, supertrendDirection, supertrendSeries: stSeries, supertrendDirSeries: stDirSeries,
      pdh, pdl, orbHigh, orbLow, orbReady, orbState,
      vwapUpper1Series: u1S, vwapLower1Series: l1S, vwapUpper2Series: u2S, vwapLower2Series: l2S,
      vwapStretch, nearKeyLevel, nearestKeyLevelName,
      liquiditySweep, recentFvg,
    };
  }

  function inferAutoMarketDirection(mtf5mTrend, ov, currentClose, lastVwap = null) {
    let score = 0;
    if (mtf5mTrend === "BULLISH") score += 1;
    else if (mtf5mTrend === "BEARISH") score -= 1;

    if (ov && ov.supertrendDirection === "BULLISH") score += 1;
    else if (ov && ov.supertrendDirection === "BEARISH") score -= 1;

    const vw = lastVwap !== null && lastVwap !== undefined ? lastVwap : null;
    if (vw !== null) {
      if (currentClose > vw) score += 1;
      else if (currentClose < vw) score -= 1;
    }

    if (ov && ov.orbReady) {
      if (ov.orbState === "ORB_BREAKOUT_UP") score += 1;
      else if (ov.orbState === "ORB_BREAKOUT_DOWN") score -= 1;
    }

    if (score >= 2) return "BULLISH";
    if (score <= -2) return "BEARISH";
    return "NEUTRAL";
  }

  class IntradaySessionSimulator {
    constructor({
      symbol,
      sessionDate,
      timeframe,
      warmupBars,
      sessionBars,
      centralDirection,
      humanPermission,
      setupFilter = "ALL_QUALIFYING",
      initialCapital = 50000,
      minRewardToFrictionRatio = null,
      cooldownBarsAfterExit = null,
      maxEntriesPerSession = null,
      requireMtfTrendAlignment = null,
      minWinnerRrMultiple = null,
      enableChandelierTrailing = null,
      chandelierAtrMult = 2.2,
      breakevenTriggerR = 1.2,
      minHoldBarsBeforeAgentExit = null,
      strategyProfile = "BALANCED_ALL",
      minAdxForEntry = null,
      blockVwap2sdChase = null,
    }) {
      this.symbol = symbol;
      this.sessionDate = sessionDate;
      this.timeframe = timeframe;
      this.warmupBars = warmupBars || [];
      this.sessionBars = sessionBars || [];
      this.centralDirection = centralDirection || "BULLISH";
      this.humanPermission = humanPermission || "ALLOW_LONG";
      this.setupFilter = setupFilter;
      this.initialCapital = Number(initialCapital) || 50000;
      this.minRewardToFrictionRatio =
        minRewardToFrictionRatio !== null ? Number(minRewardToFrictionRatio) : timeframe === "1m" ? 3.0 : 0.0;
      this.cooldownBarsAfterExit =
        cooldownBarsAfterExit !== null ? Number(cooldownBarsAfterExit) : timeframe === "1m" ? 5 : 0;
      this.maxEntriesPerSession =
        maxEntriesPerSession !== null ? Number(maxEntriesPerSession) : timeframe === "1m" ? 4 : null;
      this.requireMtfTrendAlignment =
        requireMtfTrendAlignment !== null ? Boolean(requireMtfTrendAlignment) : timeframe === "1m";
      this.minWinnerRrMultiple =
        minWinnerRrMultiple !== null ? Number(minWinnerRrMultiple) : timeframe === "1m" ? 2.5 : 0.0;
      this.enableChandelierTrailing =
        enableChandelierTrailing !== null ? Boolean(enableChandelierTrailing) : timeframe === "1m";
      this.chandelierAtrMult = Number(chandelierAtrMult) || 2.2;
      this.breakevenTriggerR = Number(breakevenTriggerR) || 1.2;
      this.minHoldBarsBeforeAgentExit =
        minHoldBarsBeforeAgentExit !== null ? Number(minHoldBarsBeforeAgentExit) : timeframe === "1m" ? 6 : 0;
      this.strategyProfile = strategyProfile || "BALANCED_ALL";
      this.minAdxForEntry = minAdxForEntry !== null ? Number(minAdxForEntry) : timeframe === "1m" ? 18.0 : 0.0;
      this.blockVwap2sdChase = blockVwap2sdChase !== null ? Boolean(blockVwap2sdChase) : timeframe === "1m";

      this.reset();
    }

    reset() {
      this.barIdx = 0;
      this.cash = this.initialCapital;
      this.dayStartEquity = this.initialCapital;
      this.peakEquity = this.initialCapital;
      this.riskState = "ACTIVE"; // ACTIVE, REDUCING, HALTED
      this.position = null; // { side: "LONG"|"SHORT", qty, entryPrice, currentPrice, stopLoss, target, entryCost, highestPrice, lowestPrice, barsHeld }
      this.pendingOrder = null; // { actionType, qty, stopLoss, target, signalCloseTs }
      this.activeSetup = null; // { direction, invalidationPrice, barsWatching }
      this.entriesOpenedThisSession = 0;
      this.lastExitBarIndex = null;
      this.activeInitialRisk = 0.0;
      this.breakevenLocked = false;
      this.chandelierStopHistory = [];
      this.lastMtf5mTrend = "NEUTRAL";
      this.lastOverlays = null;
      this.decisions = [];
      this.closedTrades = [];
      this.orders = [];
      this.lastAnalysis = null;
      this.outcomeReport = null;
    }

    setHumanPermission(perm) {
      this.humanPermission = perm;
      if (perm === "STOP_ALL") {
        this.pendingOrder = null;
        if (this.position && this.barIdx > 0) {
          const curBar = this.sessionBars[this.barIdx - 1];
          this._closePosition(roundToNseTick(curBar.c), curBar.tc || curBar.t, "EMERGENCY_HUMAN_STOP_ALL");
        }
      }
    }

    setCentralDirection(dir) {
      this.centralDirection = dir;
    }

    getInvestedCapital() {
      if (!this.position) return 0;
      const p = this.position;
      if (p.side === "SHORT") {
        return p.qty * p.entryPrice + p.qty * (p.entryPrice - p.currentPrice);
      }
      return p.qty * p.currentPrice;
    }

    getTotalEquity() {
      return this.cash + this.getInvestedCapital();
    }

    getRealizedPnl() {
      return this.closedTrades.reduce((s, t) => s + t.netPnlInr, 0);
    }

    getUnrealizedPnl() {
      if (!this.position) return 0;
      const p = this.position;
      return p.side === "SHORT" ? p.qty * (p.entryPrice - p.currentPrice) : p.qty * (p.currentPrice - p.entryPrice);
    }

    getTotalFriction() {
      const closedF = this.closedTrades.reduce((s, t) => s + t.totalFrictionInr, 0);
      const openF = this.position ? this.position.entryCost : 0;
      return Number((closedF + openF).toFixed(2));
    }

    _checkCircuitBreakers() {
      const eq = this.getTotalEquity();
      if (eq > this.peakEquity) this.peakEquity = eq;
      const dailyPnl = eq - this.dayStartEquity;
      const dailyLossPct = dailyPnl < 0 ? Math.abs(dailyPnl) / this.dayStartEquity : 0;
      const mtmDdPct = this.peakEquity > 0 ? (this.peakEquity - eq) / this.peakEquity : 0;
      if (dailyLossPct >= 0.025 || mtmDdPct >= 0.03) {
        this.riskState = "HALTED";
        return true;
      }
      return this.riskState === "HALTED";
    }

    _sizeOrder(side, entryPrice, stopLoss) {
      const eq = this.getTotalEquity();
      const riskPerShare = Math.abs(entryPrice - stopLoss);
      if (riskPerShare <= 0 || entryPrice <= 0) return 0;
      const maxByRisk = Math.floor((eq * 0.02) / riskPerShare);
      const maxByCap = Math.floor((eq * 0.15) / entryPrice); // 15% max position weight parity!
      const maxByExp = Math.floor(Math.max(0, eq * 0.8 - this.getInvestedCapital()) / entryPrice);
      let maxByCash = Math.floor(this.cash / entryPrice);
      while (maxByCash > 0) {
        const c = calculateIndianIntradayCosts(entryPrice, maxByCash, side).totalFrictionInr;
        if (entryPrice * maxByCash + c <= this.cash) break;
        maxByCash--;
      }
      return Math.max(0, Math.min(maxByRisk, maxByCap, maxByExp, maxByCash));
    }

    _openPosition(side, qty, fillPx, stopLoss, target, ts) {
      const orderSide = side === "SHORT" ? "SELL" : "BUY";
      const cost = calculateIndianIntradayCosts(fillPx, qty, orderSide);
      const req = fillPx * qty + cost.totalFrictionInr;
      if (req > this.cash || qty <= 0) return false;
      this.cash -= req;
      this.position = {
        side,
        qty,
        entryPrice: fillPx,
        currentPrice: fillPx,
        stopLoss,
        target,
        entryCost: cost.totalFrictionInr,
        highestPrice: fillPx,
        lowestPrice: fillPx,
        barsHeld: 0,
        entryTime: ts,
      };
      this.orders.push({
        time: ts,
        action: side === "SHORT" ? "OPEN_SHORT" : "OPEN_LONG",
        qty,
        price: fillPx,
        fees: cost.totalFrictionInr,
        pnl: 0,
        reason: `Filled at next-bar open (${side})`,
      });
      this.entriesOpenedThisSession += 1;
      this.activeInitialRisk = stopLoss !== null && stopLoss !== undefined ? Math.abs(fillPx - stopLoss) : Math.max(0.05, fillPx * 0.005);
      this.breakevenLocked = false;
      return true;
    }

    _closePosition(exitPx, ts, reason) {
      if (!this.position) return false;
      const p = this.position;
      const orderSide = p.side === "SHORT" ? "BUY" : "SELL";
      const cost = calculateIndianIntradayCosts(exitPx, p.qty, orderSide);
      const grossPnl = p.side === "SHORT" ? (p.entryPrice - exitPx) * p.qty : (exitPx - p.entryPrice) * p.qty;
      const netPnl = grossPnl - p.entryCost - cost.totalFrictionInr;
      if (p.side === "SHORT") {
        this.cash += p.entryPrice * p.qty + grossPnl - cost.totalFrictionInr;
      } else {
        this.cash += exitPx * p.qty - cost.totalFrictionInr;
      }
      const totalFriction = Number((p.entryCost + cost.totalFrictionInr).toFixed(2));
      this.closedTrades.push({
        symbol: this.symbol,
        side: p.side,
        qty: p.qty,
        entryPrice: p.entryPrice,
        exitPrice: exitPx,
        entryTime: p.entryTime,
        exitTime: ts,
        grossPnlInr: Number(grossPnl.toFixed(2)),
        totalFrictionInr: totalFriction,
        netPnlInr: Number(netPnl.toFixed(2)),
        reason,
      });
      this.orders.push({
        time: ts,
        action: p.side === "SHORT" ? "COVER_SHORT" : "EXIT_LONG",
        qty: p.qty,
        price: exitPx,
        fees: cost.totalFrictionInr,
        pnl: Number(netPnl.toFixed(2)),
        reason,
      });
      this.position = null;
      this.activeInitialRisk = 0.0;
      this.breakevenLocked = false;
      this.lastExitBarIndex = Math.max(0, this.barIdx - 1);
      this._checkCircuitBreakers();
      return true;
    }

    stepNextBar() {
      if (this.barIdx >= this.sessionBars.length) return null;
      const bar = this.sessionBars[this.barIdx];
      const idx = this.barIdx;
      this.barIdx += 1;

      const tsOpen = bar.t;
      const tsClose = bar.tc || bar.t;
      const closeHHMM = tsClose.slice(11, 16);
      const barNotes = [];

      // STEP A: Execute PendingOrder from bar t-1 close at bar t open (next-bar-open causality!)
      if (this.pendingOrder) {
        const po = this.pendingOrder;
        this.pendingOrder = null;
        const fillPx = roundToNseTick(bar.o);
        const gapStop =
          po.actionType === "OPEN_LONG"
            ? po.stopLoss !== null && fillPx <= po.stopLoss
            : po.actionType === "OPEN_SHORT"
            ? po.stopLoss !== null && fillPx >= po.stopLoss
            : false;
        const gapTarget =
          po.actionType === "OPEN_LONG"
            ? po.target !== null && fillPx >= po.target
            : po.actionType === "OPEN_SHORT"
            ? po.target !== null && fillPx <= po.target
            : false;

        if (gapStop || gapTarget) {
          barNotes.push(gapStop ? `GAP_CANCEL_STOP@${fillPx}` : `GAP_CANCEL_TARGET@${fillPx}`);
        } else {
          const isPendingExit = po.actionType === "EXIT_LONG" || po.actionType === "COVER_SHORT";
          const permCheck = humanPermissionAllows(this.humanPermission, po.actionType);
          const haltedBlock = !isPendingExit && this._checkCircuitBreakers();
          if (permCheck.allowed && !haltedBlock) {
            if (po.actionType === "OPEN_LONG" && !this.position) {
              const qty = this._sizeOrder("BUY", fillPx, po.stopLoss);
              if (qty > 0 && this._openPosition("LONG", qty, fillPx, po.stopLoss, po.target, tsOpen)) {
                barNotes.push(`FILLED OPEN_LONG ${qty}@₹${fillPx.toFixed(2)}`);
              }
            } else if (po.actionType === "OPEN_SHORT" && !this.position) {
              const qty = this._sizeOrder("SELL", fillPx, po.stopLoss);
              if (qty > 0 && this._openPosition("SHORT", qty, fillPx, po.stopLoss, po.target, tsOpen)) {
                barNotes.push(`FILLED OPEN_SHORT ${qty}@₹${fillPx.toFixed(2)}`);
              }
            } else if ((po.actionType === "EXIT_LONG" || po.actionType === "COVER_SHORT") && this.position) {
              this._closePosition(fillPx, tsOpen, `AGENT_${po.actionType}`);
              barNotes.push(`FILLED ${po.actionType}@₹${fillPx.toFixed(2)}`);
            }
          } else {
            barNotes.push(`OPEN_GATE_BLOCKED:${permCheck.reason}`);
          }
        }
      }

      // STEP B: Intrabar OHLC stop-first check on open position
      let closedIntrabarSide = null;
      if (this.position) {
        const p = this.position;
        p.highestPrice = Math.max(p.highestPrice, bar.h);
        p.lowestPrice = Math.min(p.lowestPrice, bar.l);
        if (p.side === "LONG") {
          if (p.stopLoss !== null && bar.o <= p.stopLoss) {
            closedIntrabarSide = "LONG";
            this._closePosition(roundToNseTick(bar.o), tsClose, "GAP_STOP_LOSS");
          } else if (p.target !== null && bar.o >= p.target) {
            closedIntrabarSide = "LONG";
            this._closePosition(roundToNseTick(bar.o), tsClose, "GAP_TARGET");
          } else if (p.stopLoss !== null && bar.l <= p.stopLoss) {
            closedIntrabarSide = "LONG";
            this._closePosition(roundToNseTick(p.stopLoss), tsClose, "INTRABAR_STOP_LOSS");
          } else if (p.target !== null && bar.h >= p.target) {
            closedIntrabarSide = "LONG";
            this._closePosition(roundToNseTick(p.target), tsClose, "INTRABAR_TARGET");
          } else {
            p.currentPrice = bar.c;
            p.barsHeld += 1;
          }
        } else {
          if (p.stopLoss !== null && bar.o >= p.stopLoss) {
            closedIntrabarSide = "SHORT";
            this._closePosition(roundToNseTick(bar.o), tsClose, "GAP_STOP_LOSS");
          } else if (p.target !== null && bar.o <= p.target) {
            closedIntrabarSide = "SHORT";
            this._closePosition(roundToNseTick(bar.o), tsClose, "GAP_TARGET");
          } else if (p.stopLoss !== null && bar.h >= p.stopLoss) {
            closedIntrabarSide = "SHORT";
            this._closePosition(roundToNseTick(p.stopLoss), tsClose, "INTRABAR_STOP_LOSS");
          } else if (p.target !== null && bar.l <= p.target) {
            closedIntrabarSide = "SHORT";
            this._closePosition(roundToNseTick(p.target), tsClose, "INTRABAR_TARGET");
          } else {
            p.currentPrice = bar.c;
            p.barsHeld += 1;
          }
        }
        this._checkCircuitBreakers();
      }

      // STEP C: NSE Time Boundaries (15:05 REDUCING, 15:15 Mandatory Square-Off evaluated on tsClose)
      if (this.riskState === "ACTIVE" && closeHHMM >= "15:05") {
        this.riskState = "REDUCING";
      }
      let squaredOffSide = null;
      if (closeHHMM >= "15:15" && this.position) {
        squaredOffSide = this.position.side;
        this._closePosition(roundToNseTick(bar.c), tsClose, "MANDATORY_1515_SQUAREOFF");
        barNotes.push(`MANDATORY_1515_SQUAREOFF (${squaredOffSide})`);
      }

      // STEP D: Causal Analysis over warmup + sessionBars[0..idx]
      const causalSlice = this.warmupBars.concat(this.sessionBars.slice(0, idx + 1));
      const analysis = analyzeCausalBars(causalSlice, this.sessionDate);
      this.lastAnalysis = analysis;
      this.lastMtf5mTrend = evaluateCausal5mMtfTrend(causalSlice);
      const ov = computeChartOverlays(causalSlice, this.sessionDate);
      this.lastOverlays = ov;
      const lastVwap = analysis.vwapSeries && analysis.vwapSeries.length > 0 ? analysis.vwapSeries[analysis.vwapSeries.length - 1] : null;
      const effectiveDirection =
        this.centralDirection === "AUTO"
          ? inferAutoMarketDirection(this.lastMtf5mTrend, ov, bar.c, lastVwap)
          : this.centralDirection;
      this.lastEffectiveDirection = effectiveDirection;

      // Compute structural candidate stop/target (NSE tick rounded)
      const isBearCentral = effectiveDirection === "BEARISH";
      const rawSl = isBearCentral
        ? Math.max(bar.h, analysis.recentSwingHigh) + 0.25 * analysis.atr
        : Math.min(bar.l, analysis.recentSwingLow) - 0.25 * analysis.atr;
      const distPct = Math.abs(bar.c - rawSl) / bar.c;
      const clampedDist = Math.max(0.002, Math.min(0.05, distPct));
      const candSl = roundToNseTick(isBearCentral ? bar.c * (1 + clampedDist) : bar.c * (1 - clampedDist));
      const rrMult = this.minWinnerRrMultiple > 2.0 ? this.minWinnerRrMultiple : 2.0;
      const candTp = roundToNseTick(isBearCentral ? bar.c - rrMult * (candSl - bar.c) : bar.c + rrMult * (bar.c - candSl));

      let call = "WAIT";
      let gateStatus = "N/A";
      const rejectionReasons = [];

      if (squaredOffSide) {
        call = squaredOffSide === "SHORT" ? "COVER_SHORT" : "EXIT_LONG";
        gateStatus = "MANDATORY_EXIT";
      } else if (closedIntrabarSide) {
        call = closedIntrabarSide === "SHORT" ? "COVER_SHORT" : "EXIT_LONG";
        gateStatus = "INTRABAR_EXIT";
      } else if (this.position) {
        // Stateful open position management -> HOLD (with Chandelier + trailing stop ratchet) or EXIT_LONG / COVER_SHORT
        const p = this.position;
        if (this.enableChandelierTrailing) {
          const eSide = p.side === "SHORT" ? "SELL" : "BUY";
          const xSide = p.side === "SHORT" ? "BUY" : "SELL";
          const estRtFeePerSh =
            (calculateIndianIntradayCosts(roundToNseTick(p.entryPrice), p.qty, eSide).totalFrictionInr +
              calculateIndianIntradayCosts(roundToNseTick(bar.c), p.qty, xSide).totalFrictionInr) /
            Math.max(1, p.qty);
          const chRes = computeChandelierTrailingStop({
            side: p.side,
            entryPrice: p.entryPrice,
            currentClose: bar.c,
            currentStopLoss: p.stopLoss,
            highestPrice: p.highestPrice,
            lowestPrice: p.lowestPrice,
            atr14: analysis.atr,
            initialRiskDistance: this.activeInitialRisk,
            roundTripFeePerShare: estRtFeePerSh,
            chandelierAtrMult: this.chandelierAtrMult,
            breakevenTriggerR: this.breakevenTriggerR,
          });
          if (chRes.breakevenLocked) this.breakevenLocked = true;
          if (chRes.stopLoss !== null && chRes.stopLoss !== p.stopLoss) {
            p.stopLoss = chRes.stopLoss;
            barNotes.push(`CHANDELIER_TRAIL_SL -> ₹${chRes.stopLoss.toFixed(2)}${chRes.breakevenLocked ? " (BREAKEVEN_LOCKED)" : ""}`);
          }
        }

        let shouldExitLong =
          p.side === "LONG" &&
          (analysis.breakoutState === "BREAKDOWN_LOW" ||
            analysis.breakoutState === "FALSE_BREAKOUT" ||
            (analysis.candleBias === "BEARISH" && analysis.priceActionScore <= -0.35));
        let shouldCoverShort =
          p.side === "SHORT" &&
          (analysis.breakoutState === "BREAKOUT_HIGH" ||
            analysis.breakoutState === "FAILED_BREAKDOWN" ||
            (analysis.candleBias === "BULLISH" && analysis.priceActionScore >= 0.35));

        if (
          this.enableChandelierTrailing &&
          this.minHoldBarsBeforeAgentExit > 0 &&
          p.barsHeld < this.minHoldBarsBeforeAgentExit &&
          this.riskState === "ACTIVE" &&
          this.humanPermission !== "EXIT_ONLY" &&
          this.humanPermission !== "STOP_ALL"
        ) {
          if (shouldExitLong || shouldCoverShort) {
            barNotes.push(`CHANDELIER_HOLD_SUPPRESSED_EARLY_EXIT:barsHeld=${p.barsHeld}<${this.minHoldBarsBeforeAgentExit}`);
          }
          shouldExitLong = false;
          shouldCoverShort = false;
        }

        if (shouldExitLong) {
          call = "EXIT_LONG";
          gateStatus = "APPROVED";
          this.pendingOrder = { actionType: "EXIT_LONG", qty: p.qty, stopLoss: null, target: null, signalCloseTs: tsClose };
          barNotes.push("Queued EXIT_LONG for next bar open (structural reversal)");
        } else if (shouldCoverShort) {
          call = "COVER_SHORT";
          gateStatus = "APPROVED";
          this.pendingOrder = { actionType: "COVER_SHORT", qty: p.qty, stopLoss: null, target: null, signalCloseTs: tsClose };
          barNotes.push("Queued COVER_SHORT for next bar open (structural reversal)");
        } else {
          call = "HOLD";
          // Monotonic trailing stop ratchet
          if (p.side === "LONG" && bar.c >= p.entryPrice + 0.5 * analysis.atr) {
            const trail = Math.min(roundToNseTick(bar.c - 1.5 * analysis.atr), roundToNseTick(bar.c - 0.25 * analysis.atr));
            if (p.stopLoss === null || trail > p.stopLoss) {
              p.stopLoss = trail;
              barNotes.push(`Trailing SL ratcheted -> ₹${trail.toFixed(2)}`);
            }
          } else if (p.side === "SHORT" && bar.c <= p.entryPrice - 0.5 * analysis.atr) {
            const trail = Math.max(roundToNseTick(bar.c + 1.5 * analysis.atr), roundToNseTick(bar.c + 0.25 * analysis.atr));
            if (p.stopLoss === null || trail < p.stopLoss) {
              p.stopLoss = trail;
              barNotes.push(`Short Trailing SL ratcheted -> ₹${trail.toFixed(2)}`);
            }
          }
        }
      } else {
        // FLAT: Evaluate entry / setup cancellation / abstention
        if (analysis.atrPct > 6.0) {
          call = "REJECT";
          gateStatus = "REJECTED";
          rejectionReasons.push(`VOLATILITY_BREACH:atrPct=${analysis.atrPct.toFixed(2)}%`);
        } else if (effectiveDirection === "NEUTRAL") {
          call = this.activeSetup ? "CANCEL_SETUP" : "NO_TRADE";
          this.activeSetup = null;
        } else {
          const wantLong = effectiveDirection === "BULLISH";
          const aligned = wantLong ? analysis.candleBias === "BULLISH" : analysis.candleBias === "BEARISH";
          const contradicted = wantLong ? analysis.candleBias === "BEARISH" : analysis.candleBias === "BULLISH";
          const trap = wantLong
            ? analysis.breakoutState === "FALSE_BREAKOUT" || analysis.vwapEvent === "VWAP_REJECTION"
            : analysis.breakoutState === "FAILED_BREAKDOWN" || analysis.vwapEvent === "VWAP_RECLAIM";

          const quality = Math.min(
            1,
            Math.max(
              0,
              Math.abs(analysis.priceActionScore) * 0.75 +
                (analysis.volumeRegime === "HIGH_VOLUME_CONFIRMATION" ? 0.12 : 0) -
                (analysis.volumeRegime === "LOW_VOLUME_WEAK" ? 0.18 : 0) -
                (analysis.patterns.includes("DOJI") || analysis.patterns.includes("INSIDE_BAR") ? 0.2 : 0)
            )
          );

          if (this.activeSetup && (contradicted || trap || this.activeSetup.barsWatching >= 6)) {
            call = "CANCEL_SETUP";
            this.activeSetup = null;
            barNotes.push("Watched setup cancelled (invalidation/contradiction/trap)");
          } else if (aligned && !trap && quality >= 0.45 && analysis.volumeRegime !== "LOW_VOLUME_WEAK") {
            const candidateCall = wantLong ? "OPEN_LONG" : "OPEN_SHORT";
            const permCheck = humanPermissionAllows(this.humanPermission, candidateCall);
            if (!permCheck.allowed) {
              call = "REJECT";
              gateStatus = "REJECTED";
              rejectionReasons.push(permCheck.reason);
            } else if (this.riskState !== "ACTIVE" || closeHHMM >= "14:45") {
              call = "REJECT";
              gateStatus = "REJECTED";
              rejectionReasons.push(`RISK_CUTOFF:${this.riskState} @ ${closeHHMM} IST`);
            } else {
              const qty = this._sizeOrder(wantLong ? "BUY" : "SELL", bar.c, candSl);
              if (qty <= 0) {
                call = "REJECT";
                gateStatus = "REJECTED";
                rejectionReasons.push("RISK_SIZED_DOWN_TO_ZERO");
              } else {
                const disciplineRejects = [];
                if (this.requireMtfTrendAlignment) {
                  const expectedTrend = wantLong ? "BULLISH" : "BEARISH";
                  if (this.lastMtf5mTrend !== expectedTrend) {
                    disciplineRejects.push(`MTF_5M_TREND_MISALIGNED:5m=${this.lastMtf5mTrend},req=${expectedTrend}`);
                  }
                }
                if (this.minAdxForEntry > 0 && ov.adx14 !== null && ov.adx14 < this.minAdxForEntry) {
                  disciplineRejects.push(`ADX_CHOPPY_MARKET:adx=${ov.adx14.toFixed(1)}<${this.minAdxForEntry.toFixed(1)}`);
                }
                if (this.blockVwap2sdChase) {
                  if (wantLong && ov.vwapStretch === "OVEREXTENDED_ABOVE_2SD") {
                    disciplineRejects.push("VWAP_2SD_OVEREXTENDED_CHASE:LONG_ABOVE_+2SD");
                  } else if (!wantLong && ov.vwapStretch === "OVEREXTENDED_BELOW_2SD") {
                    disciplineRejects.push("VWAP_2SD_OVEREXTENDED_CHASE:SHORT_BELOW_-2SD");
                  }
                }
                const wantDir = wantLong ? "BULLISH" : "BEARISH";
                if (this.strategyProfile === "ORB_BREAKOUT") {
                  const reqOrb = wantLong ? "ORB_BREAKOUT_UP" : "ORB_BREAKOUT_DOWN";
                  if (!ov.orbReady || ov.orbState !== reqOrb) {
                    disciplineRejects.push(`PROFILE_ORB_REQUIRED:${ov.orbState}!=${reqOrb}`);
                  }
                } else if (this.strategyProfile === "SUPERTREND_ADX") {
                  if (ov.supertrendDirection !== wantDir || (ov.adx14 !== null && ov.adx14 < 20.0)) {
                    disciplineRejects.push(`PROFILE_SUPERTREND_ADX_MISMATCH:st=${ov.supertrendDirection},adx=${ov.adx14}`);
                  }
                } else if (this.strategyProfile === "LIQUIDITY_REVERSAL") {
                  const reqSw = wantLong ? "BULLISH_SWEEP" : "BEARISH_SWEEP";
                  const fvgOk = ov.recentFvg !== null && ov.recentFvg.direction === wantDir;
                  if (ov.liquiditySweep !== reqSw && !ov.nearKeyLevel && !fvgOk) {
                    disciplineRejects.push("PROFILE_LIQUIDITY_REVERSAL_NO_LEVEL_OR_SWEEP");
                  }
                }
                if (this.maxEntriesPerSession !== null && this.entriesOpenedThisSession >= this.maxEntriesPerSession) {
                  disciplineRejects.push(`DAILY_ENTRY_CAP_REACHED:${this.entriesOpenedThisSession}/${this.maxEntriesPerSession}`);
                }
                if (
                  this.cooldownBarsAfterExit > 0 &&
                  this.lastExitBarIndex !== null &&
                  idx - this.lastExitBarIndex < this.cooldownBarsAfterExit
                ) {
                  disciplineRejects.push(`POST_EXIT_COOLDOWN_ACTIVE:${idx - this.lastExitBarIndex}<${this.cooldownBarsAfterExit}`);
                }
                if (this.minRewardToFrictionRatio > 0 && candTp !== null) {
                  const eSide = wantLong ? "BUY" : "SELL";
                  const xSide = wantLong ? "SELL" : "BUY";
                  const estRtCost =
                    calculateIndianIntradayCosts(roundToNseTick(bar.c), qty, eSide).totalFrictionInr +
                    calculateIndianIntradayCosts(roundToNseTick(candTp), qty, xSide).totalFrictionInr;
                  const expGross = Math.abs(candTp - bar.c) * qty;
                  if (expGross < this.minRewardToFrictionRatio * estRtCost) {
                    disciplineRejects.push(
                      `FRICTION_EDGE_TOO_LOW:exp=${expGross.toFixed(2)}<${this.minRewardToFrictionRatio.toFixed(1)}x${estRtCost.toFixed(2)}`
                    );
                  }
                }
                if (disciplineRejects.length > 0) {
                  call = "REJECT";
                  gateStatus = "REJECTED";
                  disciplineRejects.forEach((r) => rejectionReasons.push(r));
                } else {
                  call = candidateCall;
                  gateStatus = "APPROVED";
                  this.activeSetup = null;
                  this.pendingOrder = {
                    actionType: candidateCall,
                    qty,
                    stopLoss: candSl,
                    target: candTp,
                    signalCloseTs: tsClose,
                  };
                  barNotes.push(`Queued ${candidateCall} (${qty} sh) for next bar open`);
                }
              }
            }
          } else {
            call = "WAIT";
            if (!contradicted) {
              this.activeSetup = {
                direction: wantLong ? "LONG" : "SHORT",
                invalidationPrice: candSl,
                barsWatching: this.activeSetup ? this.activeSetup.barsWatching + 1 : 1,
              };
            }
          }
        }
      }

      const rec = {
        barIndex: idx,
        tsOpen,
        tsClose,
        o: bar.o,
        h: bar.h,
        l: bar.l,
        c: bar.c,
        v: bar.v,
        call,
        mtf5mTrend: this.lastMtf5mTrend,
        intendedDirection: effectiveDirection === "BULLISH" ? "LONG" : effectiveDirection === "BEARISH" ? "SHORT" : "NEUTRAL",
        patterns: analysis.patterns,
        breakoutState: analysis.breakoutState,
        vwapEvent: analysis.vwapEvent,
        swingStructure: analysis.swingStructure,
        volumeRegime: analysis.volumeRegime,
        priceActionScore: analysis.priceActionScore,
        candleBias: analysis.candleBias,
        gateStatus,
        rejectionReasons,
        riskState: this.riskState,
        positionSide: this.position ? this.position.side : null,
        positionQty: this.position ? this.position.qty : 0,
        stopLoss: this.position ? this.position.stopLoss : candSl,
        target: this.position ? this.position.target : candTp,
        cashInr: Number(this.cash.toFixed(2)),
        equityInr: Number(this.getTotalEquity().toFixed(2)),
        realizedPnlInr: Number(this.getRealizedPnl().toFixed(2)),
        breakevenLocked: Boolean(this.position && this.breakevenLocked),
        notes: barNotes,
      };
      this.chandelierStopHistory.push(this.position ? this.position.stopLoss : null);
      this.decisions.push(rec);

      if (this.barIdx >= this.sessionBars.length) {
        this.outcomeReport = this.evaluatePostSessionOutcomes();
      }
      return rec;
    }

    evaluatePostSessionOutcomes() {
      const outcomes = [];
      let savedCount = 0;
      let missedCount = 0;
      let totalSavedInr = 0;

      for (const d of this.decisions) {
        const future = this.sessionBars.slice(d.barIndex + 1);
        if (future.length === 0) {
          outcomes.push({ barIndex: d.barIndex, tsClose: d.tsClose, call: d.call, mfePct: 0, maePct: 0, verdict: "END_OF_SESSION", savedLossInr: 0 });
          continue;
        }
        const maxH = Math.max(...future.map((b) => b.h));
        const minL = Math.min(...future.map((b) => b.l));
        const eodC = future[future.length - 1].c;
        const dir = d.call === "OPEN_SHORT" || d.call === "COVER_SHORT" ? "SHORT" : d.call === "OPEN_LONG" || d.call === "EXIT_LONG" ? "LONG" : d.intendedDirection;

        const mfePct = dir === "SHORT" ? Math.max(0, ((d.c - minL) / d.c) * 100) : Math.max(0, ((maxH - d.c) / d.c) * 100);
        const maePct = dir === "SHORT" ? Math.min(0, ((d.c - maxH) / d.c) * 100) : Math.min(0, ((minL - d.c) / d.c) * 100);
        const eodRet = dir === "SHORT" ? ((d.c - eodC) / d.c) * 100 : ((eodC - d.c) / d.c) * 100;

        let verdict = "NEUTRAL";
        let savedInr = 0;
        if (d.call === "OPEN_LONG" || d.call === "OPEN_SHORT") {
          verdict = eodRet > 0 ? "OPTIMAL_ENTRY" : "ADVERSE_ENTRY";
        } else if (d.call === "EXIT_LONG" || d.call === "COVER_SHORT") {
          if (eodRet < -0.1 || Math.abs(maePct) > mfePct) {
            verdict = "PROTECTIVE_EXIT_SAVED_LOSS";
            savedInr = Number(((Math.abs(maePct) / 100) * d.c * Math.max(1, d.positionQty || 3)).toFixed(2));
            savedCount++;
            totalSavedInr += savedInr;
          } else {
            verdict = "NEUTRAL_EXIT";
          }
        } else if (d.call === "WAIT" || d.call === "REJECT" || d.call === "CANCEL_SETUP") {
          if (eodRet < -0.15) {
            verdict = d.call === "REJECT" ? "RISK_REJECT_SAVED_LOSS" : "CORRECT_WAIT_SAVED_LOSS";
            savedInr = Number(((Math.abs(eodRet) / 100) * d.c * 3).toFixed(2));
            savedCount++;
            totalSavedInr += savedInr;
          } else if (mfePct >= 0.6 && eodRet > 0.3) {
            verdict = "MISSED_OPPORTUNITY";
            missedCount++;
          }
        } else if (d.call === "HOLD") {
          verdict = eodRet >= 0 ? "HOLD_FAVORABLE" : "HOLD_ADVERSE";
        }

        outcomes.push({
          barIndex: d.barIndex,
          tsClose: d.tsClose,
          call: d.call,
          mfePct: Number(mfePct.toFixed(2)),
          maePct: Number(maePct.toFixed(2)),
          eodRetPct: Number(eodRet.toFixed(2)),
          savedLossInr: savedInr,
          verdict,
        });
      }

      return {
        totalDecisions: outcomes.length,
        savedCount,
        missedCount,
        totalSavedInr: Number(totalSavedInr.toFixed(2)),
        outcomes,
      };
    }
  }

  return {
    NSE_TICK_SIZE_INR,
    roundToNseTick,
    calculateIndianIntradayCosts,
    analyzeCausalBars,
    humanPermissionAllows,
    evaluateCausal5mMtfTrend,
    computeChandelierTrailingStop,
    computeChartOverlays,
    inferAutoMarketDirection,
    IntradaySessionSimulator,
  };
});
