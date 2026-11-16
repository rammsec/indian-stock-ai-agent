"""Generate static JSON data files (`docs/data/*.json`) and `docs/index.html` for GitHub Pages.

Used by `.github/workflows/daily_nse_research.yml` so the entire Indian Stock Market
AI Research & Next-Day Buy/Sell Agent runs 100% from GitHub (GitHub Actions + GitHub Pages)
without needing any local machine.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.agents import run_full_stock_research  # noqa: E402
from backend.config import DEFAULT_NSE_UNIVERSE  # noqa: E402

EXTENDED_NSE_SYMBOLS = [
    item["symbol"] for item in DEFAULT_NSE_UNIVERSE
] + [
    "SBIN",
    "AXISBANK",
    "TITAN",
    "TATAPOWER",
    "TRENT",
    "COALINDIA",
]


def fetch_one(symbol: str) -> tuple[str, dict | None]:
    sym = symbol.strip().upper()
    try:
        bundle = run_full_stock_research(sym)
        data = bundle.model_dump()
        # Keep 90 trading days of price series so each JSON file stays fast (~15KB)
        if "technicals" in data and "price_series" in data["technicals"]:
            data["technicals"]["price_series"] = data["technicals"]["price_series"][-90:]
        print(f"[OK] {sym}: {data['next_day_guidance']['action']} ({data['next_day_guidance']['conviction_score']}/100)")
        return sym, data
    except Exception as exc:
        print(f"[WARN] Failed {sym}: {exc}")
        return sym, None


def main() -> None:
    extra_env = os.environ.get("EXTRA_SYMBOLS", "")
    extra_syms = [s.strip().upper() for s in extra_env.split(",") if s.strip()]
    all_symbols = list(dict.fromkeys(EXTENDED_NSE_SYMBOLS + extra_syms))

    data_dir = ROOT_DIR / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing symbols in docs/data so custom user-added stocks persist
    bundles: dict[str, dict] = {}
    for existing_file in data_dir.glob("*.json"):
        if existing_file.stem.upper() not in ("UNIVERSE", "SCANNER", "META"):
            try:
                bundles[existing_file.stem.upper()] = json.loads(existing_file.read_text())
            except Exception:
                pass

    to_fetch = [s for s in all_symbols if s not in bundles or os.environ.get("FORCE_REFRESH") == "1"]
    print(f"Loaded {len(bundles)} existing bundles; fetching {len(to_fetch)} NSE equities...")
    for sym in to_fetch:
        _, payload = fetch_one(sym)
        if payload is not None:
            bundles[sym] = payload
            (data_dir / f"{sym}.json").write_text(
                json.dumps(payload, separators=(",", ":"))
            )

    universe_out = []
    scanner_items = []
    for sym, b in bundles.items():
        ov = b["overview"]
        g = b["next_day_guidance"]
        t = b["technicals"]
        cb = b["cash_and_balance_sheet"]
        rd = b["rd_and_innovation"]
        shp = b["stockholders"]

        universe_out.append(
            {
                "symbol": ov["symbol"],
                "name": ov["name"],
                "sector": ov["sector"],
            }
        )
        scanner_items.append(
            {
                "symbol": ov["symbol"],
                "name": ov["name"],
                "sector": ov["sector"],
                "current_price": t["current_price"],
                "day_change_pct": t["day_change_pct"],
                "action": g["action"],
                "conviction_score": g["conviction_score"],
                "fundamental_score": g["fundamental_score"],
                "technical_score": g["technical_score"],
                "entry_zone": f"₹{g['entry_zone_low']:,.1f} – ₹{g['entry_zone_high']:,.1f}",
                "target_1": g["target_1_next_day"],
                "stop_loss": g["stop_loss"],
                "risk_reward": g["risk_reward_ratio"],
                "cash_in_hand_cr": cb["total_cash_in_hand_cr"],
                "rd_pct": rd["latest_rd_pct_of_sales"],
                "promoters_pct": shp["promoters_pct"],
                "inst_pct": shp["institutional_total_pct"],
                "rsi_14": t["rsi_14"],
            }
        )

    scanner_items.sort(key=lambda x: x["conviction_score"], reverse=True)
    meta = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "stock_count": len(bundles),
        "symbols": list(bundles.keys()),
    }

    (data_dir / "universe.json").write_text(json.dumps({"universe": universe_out}, indent=2))
    (data_dir / "scanner.json").write_text(
        json.dumps({"count": len(scanner_items), "items": scanner_items}, indent=2)
    )
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(bundles)} stock bundles to {data_dir}")


if __name__ == "__main__":
    main()
