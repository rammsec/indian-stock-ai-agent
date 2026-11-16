"""FastAPI Backend Server for BharatAlpha — Indian Market AI Research & Next-Day Agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import run_full_stock_research
from .config import DEFAULT_WATCHLIST_5
from .data_engine import search_indian_stocks
from .intel_engine import fetch_top_20_gainers_and_losers

app = FastAPI(
    title="BharatAlpha AI — Indian Market Research & Next-Day Trading Agent",
    description=(
        "Multi-Agent Indian Stock Market (NSE/BSE) Research Engine analyzing "
        "Last 5 Annual & Quarterly Reports, Cash in Hand, Stockholders, R&D Intensity, "
        "Promoter/Competitor Intelligence, and Yesterday's Top 20 Gainers & Losers."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class LLMResearchRequest(BaseModel):
    provider: str = "quant"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "market": "NSE/BSE India"}


@app.get("/api/universe")
def get_universe() -> dict[str, Any]:
    return {"universe": DEFAULT_WATCHLIST_5}


@app.get("/api/search")
def api_search_stocks(q: str = Query("", description="Stock symbol or company name")) -> dict[str, Any]:
    results = search_indian_stocks(q)
    return {"query": q, "results": results[:15]}


@app.get("/api/movers")
def api_get_top_20_movers() -> dict[str, Any]:
    try:
        return fetch_top_20_gainers_and_losers()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Movers error: {exc}") from exc


@app.get("/api/regime")
def api_get_market_regime() -> dict[str, Any]:
    """Return latest NSE Market Regime & Breadth snapshot."""
    try:
        import json

        regime_file = Path(__file__).resolve().parent.parent / "docs" / "data" / "regime.json"
        if regime_file.exists():
            return json.loads(regime_file.read_text())
        from .collectors import collect_and_classify_market_regime, collect_indices_history

        idx_bars = collect_indices_history(period="1y")
        snap = collect_and_classify_market_regime(index_bars=idx_bars)
        return snap.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Market regime error: {exc}") from exc


@app.get("/api/backtest")
def api_get_backtest_summary() -> dict[str, Any]:
    """Return Phase 3 Walk-Forward Out-of-Sample Backtest summary & model comparison."""
    import json

    bt_file = Path(__file__).resolve().parent.parent / "docs" / "data" / "backtest_summary.json"
    if bt_file.exists():
        return json.loads(bt_file.read_text())
    raise HTTPException(status_code=404, detail="backtest_summary.json not found")


@app.get("/api/predictions")
def api_get_latest_predictions() -> dict[str, Any]:
    """Return Phase 3 Calibrated Next-Day Predictions & TreeSHAP drivers for all stocks."""
    import json

    pred_file = Path(__file__).resolve().parent.parent / "docs" / "data" / "latest_predictions.json"
    if pred_file.exists():
        return json.loads(pred_file.read_text())
    raise HTTPException(status_code=404, detail="latest_predictions.json not found")


@app.get("/api/corporate-intel")
def api_get_corporate_intel() -> dict[str, Any]:
    """Return Phase 4 NSE Bulk/Block deals, institutional ownership flows & delivery intelligence."""
    import json

    ci_file = Path(__file__).resolve().parent.parent / "docs" / "data" / "corporate_intel.json"
    if ci_file.exists():
        return json.loads(ci_file.read_text())
    raise HTTPException(status_code=404, detail="corporate_intel.json not found")


@app.get("/api/stock-intelligence")
def api_get_stock_intelligence(refresh: bool = Query(False)) -> dict[str, Any]:
    """Return Phase 5 Multi-Dimensional Stock Intelligence for the 5 pilot stocks."""
    import json
    from .config import BACKTESTS_DATA_DIR, PREDICTIONS_DATA_DIR
    from .news_intel.pipeline import run_phase5_intelligence_pipeline

    p5_file = PREDICTIONS_DATA_DIR / "phase5_intelligence.json"
    if p5_file.exists() and not refresh:
        try:
            return json.loads(p5_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        payload = run_phase5_intelligence_pipeline(use_network=refresh, persist=True)
        return payload.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stock intelligence error: {exc}") from exc


@app.get("/api/phase5-experiments")
def api_get_phase5_experiments() -> dict[str, Any]:
    """Return Phase 5 Controlled Walk-Forward Experiments (A–E across 1D, 5D, 10D)."""
    import json
    from .config import BACKTESTS_DATA_DIR

    exp_file = BACKTESTS_DATA_DIR / "phase5_experiments.json"
    if exp_file.exists():
        return json.loads(exp_file.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="phase5_experiments.json not found")


@app.get("/api/research/{symbol}")
def api_get_research(
    symbol: str,
    provider: str = Query("quant"),
    model: str = Query(""),
) -> dict[str, Any]:
    try:
        llm_cfg = {"provider": provider, "model": model} if provider != "quant" else None
        bundle = run_full_stock_research(symbol, llm_config=llm_cfg)
        return bundle.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {exc}") from exc


@app.post("/api/research/{symbol}")
def api_post_research(symbol: str, req: LLMResearchRequest) -> dict[str, Any]:
    try:
        llm_cfg = req.model_dump()
        bundle = run_full_stock_research(symbol, llm_config=llm_cfg)
        return bundle.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {exc}") from exc


@app.get("/api/scanner")
def api_market_scanner(
    symbols: str = Query(
        "RELIANCE,TCS,TATAELXSI,ITC,BIRLACABLE",
        description="Comma-separated NSE symbols to scan for next-day signals",
    )
) -> dict[str, Any]:
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]

    def _scan_one(sym: str) -> dict[str, Any] | None:
        try:
            b = run_full_stock_research(sym)
            return {
                "symbol": b.overview.symbol,
                "name": b.overview.name,
                "sector": b.overview.sector,
                "current_price": b.technicals.current_price,
                "day_change_pct": b.technicals.day_change_pct,
                "action": b.next_day_guidance.action,
                "conviction_score": b.next_day_guidance.conviction_score,
                "fundamental_score": b.next_day_guidance.fundamental_score,
                "technical_score": b.next_day_guidance.technical_score,
                "entry_zone": f"₹{b.next_day_guidance.entry_zone_low:,.1f} – ₹{b.next_day_guidance.entry_zone_high:,.1f}",
                "target_1": b.next_day_guidance.target_1_next_day,
                "stop_loss": b.next_day_guidance.stop_loss,
                "risk_reward": b.next_day_guidance.risk_reward_ratio,
                "cash_in_hand_cr": b.cash_and_balance_sheet.total_cash_in_hand_cr,
                "rd_pct": b.rd_and_innovation.latest_rd_pct_of_sales,
                "promoters_pct": b.stockholders.promoters_pct,
                "inst_pct": b.stockholders.institutional_total_pct,
                "rsi_14": b.technicals.rsi_14,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [r for r in pool.map(_scan_one, sym_list) if r is not None]

    results.sort(key=lambda x: x["conviction_score"], reverse=True)
    return {"count": len(results), "items": results}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
