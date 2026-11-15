"""Configuration, NSE universe, and Indian corporate R&D disclosure benchmarks."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DB_PATH = BASE_DIR / ".cache" / "market_cache.sqlite3"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 12

# Curated high-liquidity NSE universe across key Indian sectors
DEFAULT_NSE_UNIVERSE: list[dict[str, str]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Energy & Conglomerate"},
    {"symbol": "TCS", "name": "Tata Consultancy Services Ltd", "sector": "IT Services"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "sector": "Banking & Financials"},
    {"symbol": "INFY", "name": "Infosys Ltd", "sector": "IT Services"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "sector": "Banking & Financials"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
    {"symbol": "LT", "name": "Larsen & Toubro Ltd", "sector": "Capital Goods & EPC"},
    {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "BEL", "name": "Bharat Electronics Ltd", "sector": "Defence & Aerospace"},
    {"symbol": "HAL", "name": "Hindustan Aeronautics Ltd", "sector": "Defence & Aerospace"},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "sector": "Automobiles"},
    {"symbol": "M&M", "name": "Mahindra & Mahindra Ltd", "sector": "Automobiles"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "sector": "NBFC & Financials"},
    {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG & Hospitality"},
    {"symbol": "DIVISLAB", "name": "Divi's Laboratories Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "CIPLA", "name": "Cipla Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "PERSISTENT", "name": "Persistent Systems Ltd", "sector": "IT Services"},
    {"symbol": "TATASTEEL", "name": "Tata Steel Ltd", "sector": "Metals & Mining"},
    {"symbol": "NTPC", "name": "NTPC Ltd", "sector": "Power & Utilities"},
]

# Reported Annual Report R&D intensity (% of turnover) from statutory BRSR / Board Report
# Annexures for major NSE companies where Ind-AS groups R&D inside Other Expenses.
# Used when direct IFRS R&D line is not split out on Yahoo Finance.
REPORTED_RD_INTENSITY_PCT: dict[str, dict[str, float | str]] = {
    "SUNPHARMA": {
        "pct_of_sales": 6.75,
        "focus": "Specialty pipeline (Ilumya, Winlevi, deuruxolitinib), complex generics & ANDA filings",
        "source": "Annual Report Board Annexure (R&D Expenditure)",
    },
    "DRREDDY": {
        "pct_of_sales": 8.10,
        "focus": "Biosimilars, complex injectables, novel oncology & GLP-1 peptides",
        "source": "Annual Report Form 20-F / Ind-AS R&D Schedule",
    },
    "CIPLA": {
        "pct_of_sales": 5.65,
        "focus": "Respiratory inhalation assets, peptide generics & complex depot injectables",
        "source": "Annual Report R&D Note",
    },
    "DIVISLAB": {
        "pct_of_sales": 1.45,
        "focus": "Green chemistry process synthesis, custom pharma synthesis (CS) & peptides",
        "source": "Annual Report Statutory Annexure",
    },
    "BEL": {
        "pct_of_sales": 7.20,
        "focus": "Indigenous radar systems, electronic warfare, C4I & DRDO co-development",
        "source": "Annual Report R&D & Indigenisation Disclosure",
    },
    "HAL": {
        "pct_of_sales": 8.45,
        "focus": "LCA Tejas Mk1A/Mk2, indigenous rotary platforms (LCH Prachand, LUH) & aero-engines",
        "source": "Annual Report R&D Reserve & Expenditure Disclosure",
    },
    "RELIANCE": {
        "pct_of_sales": 0.44,
        "focus": "New Energy giga-factories (solar, sodium-ion battery, green hydrogen), 6G & JioBrain AI",
        "source": "Integrated Annual Report Innovation & R&D Section",
    },
    "TCS": {
        "pct_of_sales": 1.15,
        "focus": "TCS Pace Ports, enterprise GenAI platforms (WisdomNext), ignio & cloud IP",
        "source": "Annual Report Research & Innovation Annexure",
    },
    "INFY": {
        "pct_of_sales": 1.35,
        "focus": "Infosys Topaz AI-first suite, Cobalt cloud assets & Finacle next-gen core banking",
        "source": "Annual Report Form 20-F / BRSR R&D Disclosure",
    },
    "PERSISTENT": {
        "pct_of_sales": 2.65,
        "focus": "SASVA AI software engineering platform, hyperscaler accelerators & healthcare IP",
        "source": "Annual Report R&D & IP Disclosure",
    },
    "MARUTI": {
        "pct_of_sales": 1.65,
        "focus": "Rohtak R&D centre, EV powertrainlocalization, hybrid & flex-fuel platforms",
        "source": "Annual Report Technology Absorption Annexure",
    },
    "M&M": {
        "pct_of_sales": 3.40,
        "focus": "INGLO Electric Origin SUV architecture (BE 6e, XEV 9e) & autonomous farm machinery",
        "source": "Annual Report R&D Expenditure Annexure",
    },
    "LT": {
        "pct_of_sales": 0.95,
        "focus": "Electrolyser manufacturing, green hydrogen EPC, modular construction & defence systems",
        "source": "Integrated Annual Report Technology & R&D Note",
    },
    "ITC": {
        "pct_of_sales": 0.38,
        "focus": "ITC Life Sciences & Technology Centre (LSTC), crop genetics & sustainable packaging",
        "source": "Annual Report LSTC R&D Annexure",
    },
    "BHARTIARTL": {
        "pct_of_sales": 0.55,
        "focus": "Airtel Xstream cloud, AI spam detection network intelligence & 5G standalone orchestration",
        "source": "Integrated Annual Report Digital Innovation",
    },
    "TATASTEEL": {
        "pct_of_sales": 0.42,
        "focus": "Low-carbon hydrogen blast furnace steelmaking, graphene & advanced high-strength automotive steel",
        "source": "Annual Report R&D & Technology Absorption",
    },
    "NTPC": {
        "pct_of_sales": 0.28,
        "focus": "NETRA (NTPC Energy Technology Research Alliance) carbon capture, green methanol & nuclear",
        "source": "Annual Report R&D Disclosure",
    },
    "HDFCBANK": {
        "pct_of_sales": 0.85,
        "focus": "Core banking cloud modernization, UPI high-throughput rails & AI credit underwriting",
        "source": "Technology & Digital CapEx Estimate",
    },
    "ICICIBANK": {
        "pct_of_sales": 0.90,
        "focus": "iMobile Pay architecture, trade stack automation & real-time fraud risk engines",
        "source": "Technology & Digital Innovation Estimate",
    },
    "BAJFINANCE": {
        "pct_of_sales": 1.10,
        "focus": "Omnichannel consumer app platform, GenAI credit risk decisioning & collections automation",
        "source": "Technology & Platform Investment Estimate",
    },
}

SECTOR_DEFAULT_RD_PCT: dict[str, float] = {
    "Pharmaceuticals": 5.8,
    "Healthcare": 5.2,
    "IT Services": 1.5,
    "Technology": 2.1,
    "Defence & Aerospace": 6.5,
    "Automobiles": 2.4,
    "Capital Goods & EPC": 1.2,
    "Chemicals": 1.8,
    "FMCG & Hospitality": 0.5,
    "Energy & Conglomerate": 0.45,
    "Metals & Mining": 0.4,
    "Banking & Financials": 0.8,
    "NBFC & Financials": 0.9,
    "Telecom": 0.6,
    "Power & Utilities": 0.3,
}
