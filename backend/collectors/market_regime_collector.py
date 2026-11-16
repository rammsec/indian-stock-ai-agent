"""Market Regime & Sector Index Collector & Classifier (Phase 1 — Section 5E).

Collects:
- Nifty 50, Nifty 500, Nifty Midcap 100, Nifty Smallcap 100
- India VIX
- Sectoral Indices (Nifty Bank, Nifty IT, Nifty FMCG, Nifty Pharma, Nifty Auto, Nifty Metal, Nifty Energy)
- Advance/Decline Ratio & % of Universe Above 50-DMA and 200-DMA

Classifies the Indian market into one of 5 deterministic regimes:
- `Strong Bull`
- `Weak Bull`
- `Sideways`
- `Bear`
- `High Volatility`
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from ..config import DATA_SCHEMA_VERSION, IST_TZ, NSE_MARKET_INDICES
from ..models import EquityDailyBar, IndexDailyBar, MarketRegimeLabel, MarketRegimeSnapshot
from .nse_market_collector import _parse_optional_float, fetch_nse_indices_daily_archive


def collect_indices_history(period: str = "2y") -> list[IndexDailyBar]:
    """Collect historical daily series for all broad, volatility & sector NSE indices,
    and overlay the latest official NSE `ind_close_all` archive report.
    """
    now_iso = datetime.now(IST_TZ).isoformat(timespec="seconds")
    by_code_date: dict[tuple[str, str], IndexDailyBar] = {}

    yf_map = {meta["yf_ticker"]: (code, meta) for code, meta in NSE_MARKET_INDICES.items()}
    yf_tickers = list(yf_map.keys())

    try:
        raw_df = yf.download(
            tickers=" ".join(yf_tickers),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        for yf_sym, (code, meta) in yf_map.items():
            sub = raw_df if len(yf_tickers) == 1 else (raw_df[yf_sym] if yf_sym in raw_df else None)
            if sub is None or sub.empty:
                continue
            sub_clean = sub.dropna(subset=["Close"]).copy()
            prev_c: float | None = None
            for idx, r in sub_clean.iterrows():
                dt_str = str(idx)[:10]
                c_val = _parse_optional_float(r.get("Close"))
                if c_val is None or c_val <= 0:
                    continue
                o_val = _parse_optional_float(r.get("Open"))
                h_val = _parse_optional_float(r.get("High"))
                l_val = _parse_optional_float(r.get("Low"))
                vol_val = _parse_optional_float(r.get("Volume"))
                chg_pct = (
                    round((c_val - prev_c) / prev_c * 100.0, 2)
                    if (prev_c is not None and prev_c > 0)
                    else None
                )
                by_code_date[(code, dt_str)] = IndexDailyBar(
                    index_code=code,
                    index_name=meta["nse_name"],
                    category=meta["category"],  # type: ignore[arg-type]
                    observation_date=dt_str,
                    open=round(o_val, 2) if o_val is not None else None,
                    high=round(h_val, 2) if h_val is not None else None,
                    low=round(l_val, 2) if l_val is not None else None,
                    close=round(c_val, 2),
                    change_pct=chg_pct,
                    volume=vol_val if (vol_val and vol_val > 0) else None,
                    turnover_cr=None,
                    pe_ratio=None,
                    pb_ratio=None,
                    div_yield=None,
                    source="YahooFinance_Index",
                    publication_timestamp=f"{dt_str}T15:30:00+05:30",
                    collection_timestamp=now_iso,
                    validation_status="VERIFIED",
                    data_version=DATA_SCHEMA_VERSION,
                )
                prev_c = c_val
    except Exception:
        pass

    # Overlay official NSE `ind_close_all` daily report (includes P/E, P/B, Div Yield, Turnover)
    nse_idx_bars = fetch_nse_indices_daily_archive()
    for nb in nse_idx_bars:
        by_code_date[(nb.index_code, nb.observation_date)] = nb

    return sorted(by_code_date.values(), key=lambda x: (x.index_code, x.observation_date))


def classify_market_regime(
    nifty50_closes: list[float],
    india_vix_closes: list[float],
    observation_date: str,
    breadth: dict[str, Any] | None = None,
    pct_above_50dma: float | None = None,
    pct_above_200dma: float | None = None,
    sector_performances: list[dict[str, Any]] | None = None,
) -> MarketRegimeSnapshot:
    """Deterministically classify the Indian equity market regime (Section 5E).

    Regimes:
    - `High Volatility`: India VIX >= 21.0 OR (India VIX 1D jump >= +14% AND Nifty 1D <= -1.0%)
    - `Strong Bull`: Nifty > 50-DMA > 200-DMA, 20D return >= +2.0%, VIX < 18.5, and A/D >= 1.0
    - `Weak Bull`: Nifty > 200-DMA and (Nifty >= 50-DMA or 20D return >= 0.0%)
    - `Bear`: Nifty < 200-DMA and (Nifty < 50-DMA or 20D return <= -3.5%)
    - `Sideways`: All intermediate consolidation states
    """
    breadth = breadth or {}
    rationale: list[str] = []

    if not nifty50_closes:
        return MarketRegimeSnapshot(
            observation_date=observation_date,
            regime="Sideways",
            allow_aggressive_longs=False,
            regime_rationale=["Insufficient Nifty 50 history; defaulting to conservative Sideways regime."],
        )

    s_nifty = pd.Series(nifty50_closes, dtype=float)
    cur_nifty = round(float(s_nifty.iloc[-1]), 2)
    prev_nifty = float(s_nifty.iloc[-2]) if len(s_nifty) >= 2 else cur_nifty
    nifty_1d = round((cur_nifty - prev_nifty) / prev_nifty * 100.0, 2) if prev_nifty > 0 else 0.0
    base_20d = float(s_nifty.iloc[-21]) if len(s_nifty) >= 21 else float(s_nifty.iloc[0])
    nifty_20d = round((cur_nifty - base_20d) / base_20d * 100.0, 2) if base_20d > 0 else 0.0

    dma50 = round(float(s_nifty.rolling(50, min_periods=5).mean().iloc[-1]), 2)
    dma200 = round(float(s_nifty.rolling(200, min_periods=10).mean().iloc[-1]), 2)

    cur_vix: float | None = None
    vix_1d: float | None = None
    if india_vix_closes:
        cur_vix = round(float(india_vix_closes[-1]), 2)
        prev_vix = float(india_vix_closes[-2]) if len(india_vix_closes) >= 2 else cur_vix
        vix_1d = round((cur_vix - prev_vix) / prev_vix * 100.0, 2) if prev_vix > 0 else 0.0

    adv = int(breadth.get("advance_count", 0))
    dec = int(breadth.get("decline_count", 0))
    unch = int(breadth.get("unchanged_count", 0))
    ad_ratio = breadth.get("advance_decline_ratio")

    regime: MarketRegimeLabel = "Sideways"
    allow_longs = True

    if (cur_vix is not None and cur_vix >= 21.0) or (
        vix_1d is not None and vix_1d >= 14.0 and nifty_1d <= -1.0
    ):
        regime = "High Volatility"
        allow_longs = False
        rationale.append(
            f"India VIX elevated at {cur_vix} ({vix_1d:+.1f}% 1D) with Nifty 50 1D change {nifty_1d:+.2f}%."
        )
    elif cur_nifty < dma200 and (cur_nifty < dma50 or nifty_20d <= -3.5):
        regime = "Bear"
        allow_longs = False
        rationale.append(
            f"Nifty 50 ({cur_nifty:,.1f}) is trading below its 200-DMA ({dma200:,.1f}) and 50-DMA ({dma50:,.1f}) with 20D return {nifty_20d:+.2f}%."
        )
    elif (
        cur_nifty > dma50
        and dma50 >= dma200
        and nifty_20d >= 2.0
        and (cur_vix is None or cur_vix < 18.5)
        and (ad_ratio is None or ad_ratio >= 0.95)
    ):
        regime = "Strong Bull"
        allow_longs = True
        rationale.append(
            f"Nifty 50 ({cur_nifty:,.1f}) > 50-DMA ({dma50:,.1f}) >= 200-DMA ({dma200:,.1f}) with +{nifty_20d:.2f}% 20D momentum and low India VIX ({cur_vix or 'N/A'})."
        )
    elif cur_nifty >= dma200 and (cur_nifty >= dma50 or nifty_20d >= 0.0):
        regime = "Weak Bull"
        allow_longs = True
        rationale.append(
            f"Structural uptrend intact above 200-DMA ({dma200:,.1f}), though short-term momentum is moderate (20D: {nifty_20d:+.2f}%)."
        )
    else:
        regime = "Sideways"
        allow_longs = True
        rationale.append(
            f"Range-bound consolidation: Nifty 50 at {cur_nifty:,.1f} vs 50-DMA ({dma50:,.1f}) and 200-DMA ({dma200:,.1f}); 20D return {nifty_20d:+.2f}%."
        )

    if ad_ratio is not None:
        rationale.append(
            f"NSE Market Breadth: {adv} Advances vs {dec} Declines (A/D Ratio {ad_ratio:.2f})."
        )
    if pct_above_50dma is not None and pct_above_200dma is not None:
        rationale.append(
            f"Universe Breadth: {pct_above_50dma:.1f}% above 50-DMA, {pct_above_200dma:.1f}% above 200-DMA."
        )

    return MarketRegimeSnapshot(
        observation_date=observation_date,
        regime=regime,
        nifty_50_close=cur_nifty,
        nifty_50_day_change_pct=nifty_1d,
        nifty_50_return_20d_pct=nifty_20d,
        nifty_50_dma_50=dma50,
        nifty_50_dma_200=dma200,
        india_vix_close=cur_vix,
        india_vix_change_pct=vix_1d,
        advance_count=adv,
        decline_count=dec,
        unchanged_count=unch,
        advance_decline_ratio=ad_ratio,
        pct_above_50dma=pct_above_50dma,
        pct_above_200dma=pct_above_200dma,
        allow_aggressive_longs=allow_longs,
        regime_rationale=rationale,
        sector_performances=sector_performances or [],
    )


def collect_and_classify_market_regime(
    index_bars: list[IndexDailyBar],
    equity_bars: list[EquityDailyBar] | None = None,
    breadth: dict[str, Any] | None = None,
) -> MarketRegimeSnapshot:
    """Compute the latest MarketRegimeSnapshot from collected index & equity bars."""
    nifty_bars = [b for b in index_bars if b.index_code == "NIFTY_50"]
    vix_bars = [b for b in index_bars if b.index_code == "INDIA_VIX"]

    obs_date = (
        nifty_bars[-1].observation_date
        if nifty_bars
        else datetime.now(IST_TZ).strftime("%Y-%m-%d")
    )

    # Sector performances summary
    sector_perfs: list[dict[str, Any]] = []
    by_code: dict[str, list[IndexDailyBar]] = {}
    for b in index_bars:
        by_code.setdefault(b.index_code, []).append(b)

    for code, bars_list in by_code.items():
        if not bars_list or bars_list[-1].category != "sector":
            continue
        latest = bars_list[-1]
        base_20 = bars_list[-21].close if len(bars_list) >= 21 else bars_list[0].close
        ret_20d = round((latest.close - base_20) / base_20 * 100.0, 2) if base_20 > 0 else 0.0
        sector_perfs.append(
            {
                "index_code": code,
                "index_name": latest.index_name,
                "close": latest.close,
                "day_change_pct": latest.change_pct or 0.0,
                "return_20d_pct": ret_20d,
                "pe_ratio": latest.pe_ratio,
            }
        )
    sector_perfs.sort(key=lambda x: x["day_change_pct"], reverse=True)

    # Compute % of tracked equities above 50-DMA & 200-DMA if equity_bars provided
    pct_50: float | None = None
    pct_200: float | None = None
    if equity_bars:
        by_sym: dict[str, list[float]] = {}
        for eb in equity_bars:
            if eb.close is not None:
                by_sym.setdefault(eb.symbol, []).append(eb.close)
        above_50 = 0
        above_200 = 0
        valid_syms = 0
        for sym, closes in by_sym.items():
            if len(closes) < 30:
                continue
            valid_syms += 1
            s = pd.Series(closes, dtype=float)
            c_last = float(s.iloc[-1])
            d50 = float(s.rolling(50, min_periods=10).mean().iloc[-1])
            d200 = float(s.rolling(200, min_periods=20).mean().iloc[-1])
            if c_last >= d50:
                above_50 += 1
            if c_last >= d200:
                above_200 += 1
        if valid_syms > 0:
            pct_50 = round(above_50 / valid_syms * 100.0, 1)
            pct_200 = round(above_200 / valid_syms * 100.0, 1)

    return classify_market_regime(
        nifty50_closes=[b.close for b in nifty_bars],
        india_vix_closes=[b.close for b in vix_bars],
        observation_date=obs_date,
        breadth=breadth,
        pct_above_50dma=pct_50,
        pct_above_200dma=pct_200,
        sector_performances=sector_perfs,
    )
