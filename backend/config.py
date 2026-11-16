"""Configuration, 5-stock default watchlist, deep promoter/competitor profiles, and R&D benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DB_PATH = BASE_DIR / ".cache" / "market_cache.sqlite3"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 12

# User's curated 5 starter companies (expandable up to 10 directly on the page)
DEFAULT_WATCHLIST_5: list[dict[str, str]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Energy, Telecom & Retail"},
    {"symbol": "TCS", "name": "Tata Consultancy Services Ltd", "sector": "IT Services & Consulting"},
    {"symbol": "TATAELXSI", "name": "Tata Elxsi Ltd", "sector": "ER&D & Embedded AI/Auto Design"},
    {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG, Agri & Paperboards"},
    {"symbol": "BIRLACABLE", "name": "Birla Cable Ltd", "sector": "Optical Fiber & Telecom Cables"},
]

# Retained alias for compatibility
DEFAULT_NSE_UNIVERSE = DEFAULT_WATCHLIST_5

# Broad liquid NSE universe used to identify Yesterday's Top 20 Gainers & Top 20 Losers
BROAD_NSE_MOVERS_POOL: list[dict[str, str]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Energy & Telecom"},
    {"symbol": "TCS", "name": "Tata Consultancy Services Ltd", "sector": "IT Services"},
    {"symbol": "TATAELXSI", "name": "Tata Elxsi Ltd", "sector": "Auto & Media ER&D"},
    {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG & Cigarettes"},
    {"symbol": "BIRLACABLE", "name": "Birla Cable Ltd", "sector": "Telecom Cables & OFC"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "sector": "Banking"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "sector": "Banking"},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "PSU Banking"},
    {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "sector": "Banking"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "sector": "NBFC"},
    {"symbol": "INFY", "name": "Infosys Ltd", "sector": "IT Services"},
    {"symbol": "HCLTECH", "name": "HCL Technologies Ltd", "sector": "IT Services"},
    {"symbol": "WIPRO", "name": "Wipro Ltd", "sector": "IT Services"},
    {"symbol": "TECHM", "name": "Tech Mahindra Ltd", "sector": "IT Services"},
    {"symbol": "PERSISTENT", "name": "Persistent Systems Ltd", "sector": "IT Services"},
    {"symbol": "KPITTECH", "name": "KPIT Technologies Ltd", "sector": "Auto Software ER&D"},
    {"symbol": "LTTS", "name": "L&T Technology Services Ltd", "sector": "Engineering R&D"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "CIPLA", "name": "Cipla Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "DIVISLAB", "name": "Divi's Laboratories Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "LUPIN", "name": "Lupin Ltd", "sector": "Pharmaceuticals"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
    {"symbol": "LT", "name": "Larsen & Toubro Ltd", "sector": "EPC & Capital Goods"},
    {"symbol": "BEL", "name": "Bharat Electronics Ltd", "sector": "Defence Electronics"},
    {"symbol": "HAL", "name": "Hindustan Aeronautics Ltd", "sector": "Aerospace & Defence"},
    {"symbol": "MAZDOCK", "name": "Mazagon Dock Shipbuilders Ltd", "sector": "Defence Shipbuilding"},
    {"symbol": "BHEL", "name": "Bharat Heavy Electricals Ltd", "sector": "Power Equipment"},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "sector": "Automobiles"},
    {"symbol": "M&M", "name": "Mahindra & Mahindra Ltd", "sector": "Automobiles & Farm"},
    {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd", "sector": "Two-Wheelers & EV"},
    {"symbol": "EICHERMOT", "name": "Eicher Motors Ltd", "sector": "Automobiles"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp Ltd", "sector": "Two-Wheelers"},
    {"symbol": "TITAN", "name": "Titan Company Ltd", "sector": "Consumer Discretionary"},
    {"symbol": "TRENT", "name": "Trent Ltd", "sector": "Retail & Apparel"},
    {"symbol": "DMART", "name": "Avenue Supermarts Ltd", "sector": "Supermarket Retail"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "sector": "FMCG"},
    {"symbol": "NESTLEIND", "name": "Nestle India Ltd", "sector": "FMCG Food"},
    {"symbol": "VBL", "name": "Varun Beverages Ltd", "sector": "Beverages"},
    {"symbol": "TATASTEEL", "name": "Tata Steel Ltd", "sector": "Steel & Mining"},
    {"symbol": "JSWSTEEL", "name": "JSW Steel Ltd", "sector": "Steel"},
    {"symbol": "HINDALCO", "name": "Hindalco Industries Ltd", "sector": "Aluminium & Copper"},
    {"symbol": "COALINDIA", "name": "Coal India Ltd", "sector": "Mining & Energy"},
    {"symbol": "NTPC", "name": "NTPC Ltd", "sector": "Power Generation"},
    {"symbol": "TATAPOWER", "name": "Tata Power Company Ltd", "sector": "Power & Renewables"},
    {"symbol": "POWERGRID", "name": "Power Grid Corporation of India Ltd", "sector": "Power Transmission"},
    {"symbol": "ONGC", "name": "Oil & Natural Gas Corporation Ltd", "sector": "Oil & Gas Exploration"},
    {"symbol": "BPCL", "name": "Bharat Petroleum Corporation Ltd", "sector": "Oil Refining & Marketing"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd", "sector": "Cement"},
    {"symbol": "DIXON", "name": "Dixon Technologies (India) Ltd", "sector": "EMS & Electronics"},
    {"symbol": "POLYCAB", "name": "Polycab India Ltd", "sector": "Cables & Wires"},
    {"symbol": "HFCL", "name": "HFCL Ltd", "sector": "Optical Fiber & Telecom Equipment"},
    {"symbol": "STLTECH", "name": "Sterlite Technologies Ltd", "sector": "Optical Fiber & Digital Networks"},
    {"symbol": "FINCABLES", "name": "Finolex Cables Ltd", "sector": "Electrical & Telecom Cables"},
    {"symbol": "IRCTC", "name": "Indian Railway Catering & Tourism Corp", "sector": "Railways & Travel Tech"},
    {"symbol": "RVNL", "name": "Rail Vikas Nigam Ltd", "sector": "Railway Infrastructure"},
    {"symbol": "SUZLON", "name": "Suzlon Energy Ltd", "sector": "Wind Energy & Turbines"},
]

# Deep Promoter, Key Leaders, Social/Media Query & Direct Competitor Intelligence
DEEP_COMPANY_PROFILES: dict[str, dict[str, Any]] = {
    "RELIANCE": {
        "promoter_group": "Ambani Family & Promoter Holding Trusts (Srichakra, Devarshi, Karuna Commercials LLP)",
        "key_leaders": [
            {
                "name": "Mukesh D. Ambani",
                "role": "Chairman & Managing Director",
                "focus": "Capital allocation, New Energy Giga-Complex at Jamnagar, AI infrastructure & value unlocking",
                "handle_or_source": "RIL AGM / Chairman Statements / @RelianceJio / @RIL_Updates",
            },
            {
                "name": "Isha M. Ambani",
                "role": "Executive Director, Reliance Retail Ventures",
                "focus": "Omni-channel retail footprint expansion, FMCG (Campa, Independence) & quick-commerce scale",
                "handle_or_source": "Reliance Retail Concalls & Brand Launches",
            },
            {
                "name": "Akash M. Ambani",
                "role": "Chairman, Reliance Jio Infocomm",
                "focus": "5G Standalone monetization, AirFiber home broadband, JioBrain AI & ARPU expansion",
                "handle_or_source": "IMC Keynotes / Jio Investor Briefings",
            },
            {
                "name": "Anant M. Ambani",
                "role": "Executive Director, New Energy & Materials",
                "focus": "Solar PV, Sodium-ion battery storage, Green Hydrogen & Bio-energy giga-factories",
                "handle_or_source": "New Energy Project Disclosures",
            },
        ],
        "key_stakeholders_watch": [
            "Life Insurance Corporation of India (LIC — ~6.49% stake, steady domestic anchor)",
            "SBI Mutual Fund & ICICI Prudential MF (increasing DII weight across Nifty funds)",
            "Government of Singapore (GIC) & Abu Dhabi Investment Authority / Saudi PIF (Jio/Retail strategic backers)",
        ],
        "promoter_search_queries": [
            "Mukesh Ambani Reliance statement OR Jio OR New Energy",
            "Reliance Retail Isha Ambani OR Akash Ambani Jio ARPU tariff",
        ],
        "competitors": [
            {
                "symbol": "BHARTIARTL",
                "name": "Bharti Airtel Ltd",
                "segment": "Telecom & 5G Home Broadband (vs. Jio)",
                "impact_driver": "Airtel tariff hike commentary & postpaid ARPU gains directly re-rate Jio's valuation multiple.",
            },
            {
                "symbol": "DMART",
                "name": "Avenue Supermarts (DMart)",
                "segment": "Organized Grocery & Value Retail (vs. Reliance Retail)",
                "impact_driver": "Same-store sales growth (SSSG) & quick-commerce margin pressure read-across for Reliance Retail.",
            },
            {
                "symbol": "TRENT",
                "name": "Trent Ltd (Zudio / Westside)",
                "segment": "Fast Fashion & Lifestyle Retail (vs. Reliance Trends / Yousta)",
                "impact_driver": "Apparel discretionary footfalls & store addition velocity benchmark.",
            },
            {
                "symbol": "BPCL",
                "name": "Bharat Petroleum Corp Ltd",
                "segment": "Oil-to-Chemicals & Refining Margins (O2C)",
                "impact_driver": "Singapore Gross Refining Margin (GRM) & diesel/ATF crack spreads affect RIL's O2C cash flows.",
            },
        ],
    },
    "TCS": {
        "promoter_group": "Tata Sons Private Limited (71.74% Promoter Stake)",
        "key_leaders": [
            {
                "name": "N. Chandrasekaran",
                "role": "Chairman, Tata Sons & TCS",
                "focus": "Group-wide AI transformation, BSNL/sovereign cloud execution & semiconductor/digital synergy",
                "handle_or_source": "Tata Sons Chairman Letters / Davos & Nasscom Statements",
            },
            {
                "name": "K. Krithivasan",
                "role": "Chief Executive Officer & Managing Director",
                "focus": "US/UK BFSI recovery, mega-deal TCV conversions ($8B–$10B/quarter), GenAI production deployment",
                "handle_or_source": "TCS Quarterly Earnings Concall / @TCS / CNBC-TV18 Interviews",
            },
            {
                "name": "Samir Seksaria",
                "role": "Chief Financial Officer",
                "focus": "26%–28% operating margin band discipline, utilization, pyramid optimization & 95%+ FCF payout",
                "handle_or_source": "TCS Analyst Briefings & Margin Commentary",
            },
        ],
        "key_stakeholders_watch": [
            "Tata Sons Pvt Ltd (71.74% holding — occasional buyback participation funds group capex)",
            "LIC of India (~4.86% stake — largest domestic institutional supporter)",
            "SBI Mutual Fund & NPS Trust (core passive and active domestic accumulators)",
        ],
        "promoter_search_queries": [
            "Krithivasan TCS deal wins OR BFSI OR GenAI statement",
            "Tata Consultancy Services Chandrasekaran OR TCS margin guidance",
        ],
        "competitors": [
            {
                "symbol": "INFY",
                "name": "Infosys Ltd",
                "segment": "Tier-1 Global IT Services & Consulting",
                "impact_driver": "Infosys constant-currency revenue guidance upgrades/cuts immediately impact TCS next-day sentiment.",
            },
            {
                "symbol": "HCLTECH",
                "name": "HCL Technologies Ltd",
                "segment": "IT Infrastructure, ER&D & Enterprise Apps",
                "impact_driver": "Vendor consolidation mega-deals & software margins compete directly with TCS.",
            },
            {
                "symbol": "WIPRO",
                "name": "Wipro Ltd",
                "segment": "Global IT & Capco BFSI Consulting",
                "impact_driver": "US banking discretionary consulting trends give early signal for TCS's core BFSI vertical.",
            },
            {
                "symbol": "TECHM",
                "name": "Tech Mahindra Ltd",
                "segment": "Telecom & Communications IT",
                "impact_driver": "Global telco IT spend and 5G network integration budget read-across.",
            },
        ],
    },
    "TATAELXSI": {
        "promoter_group": "Tata Sons Private Limited (43.91% Promoter Stake)",
        "key_leaders": [
            {
                "name": "Manoj Raghavan",
                "role": "Managing Director & CEO",
                "focus": "Software-Defined Vehicles (SDV), OEM R&D deals in Europe/US/Japan, Healthcare & Media OTT recovery",
                "handle_or_source": "Tata Elxsi Earnings Concalls / @tataelxsi / Auto Tech Keynotes",
            },
            {
                "name": "N. Ganapathy Subramaniam",
                "role": "Chairman (Non-Executive)",
                "focus": "Tata Group automotive/Jaguar Land Rover (JLR) & Agratas battery ecosystem synergy",
                "handle_or_source": "Board & Annual General Meeting Statements",
            },
            {
                "name": "Gaurav Bajaj",
                "role": "Chief Financial Officer",
                "focus": "Maintaining industry-leading ~26-28% EBITDA margins, offshore mix & R&D center utilization",
                "handle_or_source": "Investor Presentations & Quarterly Concalls",
            },
        ],
        "key_stakeholders_watch": [
            "Tata Sons Private Limited (43.91% rock-solid promoter anchor)",
            "LIC of India & Axis/Mirae Domestic Mutual Funds",
            "High Retail & HNI Shareholder Base (~39% public float — creates sharp momentum moves on earnings days)",
        ],
        "promoter_search_queries": [
            "Manoj Raghavan Tata Elxsi automotive OR SDV OR JLR",
            "Tata Elxsi deal win OR EV software OR quarterly results",
        ],
        "competitors": [
            {
                "symbol": "KPITTECH",
                "name": "KPIT Technologies Ltd",
                "segment": "Pure-Play Automotive Embedded Software & SDV",
                "impact_driver": "KPIT's European/US OEM deal pipeline & CC growth guidance is the #1 read-across catalyst for Tata Elxsi.",
            },
            {
                "symbol": "LTTS",
                "name": "L&T Technology Services Ltd",
                "segment": "Multi-Vertical Engineering R&D (Mobility, MedTech, Hi-Tech)",
                "impact_driver": "Competing directly for global ER&D outsourcing budgets across Mobility and Medical Devices.",
            },
            {
                "symbol": "PERSISTENT",
                "name": "Persistent Systems Ltd",
                "segment": "Digital Product Engineering & AI Platforms",
                "impact_driver": "Benchmark for high-multiple mid-cap product engineering valuation and US enterprise R&D spend.",
            },
            {
                "symbol": "HCLTECH",
                "name": "HCLTech ER&D Division",
                "segment": "Semiconductor, Telecom & Auto Engineering Services",
                "impact_driver": "Large-scale chip-to-cloud and broadcast media engineering competition.",
            },
        ],
    },
    "ITC": {
        "promoter_group": "100% Institutionally Owned Professional Blue-Chip (Zero Individual Promoter; BAT, LIC & SUUTI Anchors)",
        "key_leaders": [
            {
                "name": "Sanjiv Puri",
                "role": "Chairman & Managing Director",
                "focus": "'ITC Next' strategy: scaling FMCG-Others (Aashirvaad, Sunfeast, Bingo!), ITC MAARS Agri-tech, Value-Added Paperboards & post-demerger ITC Hotels value unlocking",
                "handle_or_source": "ITC AGM / CII Presidency Statements / @ITCCorpCom",
            },
            {
                "name": "Supratim Dutta",
                "role": "Executive Director & CFO",
                "focus": "Cigarette volume vs tax stability, FMCG EBITDA margin expansion, leaf tobacco export realizations & high dividend payout",
                "handle_or_source": "ITC Institutional Investor Day & Financial Filings",
            },
            {
                "name": "Tadeu Marroco (BAT CEO)",
                "role": "CEO of British American Tobacco (Largest Shareholder ~25.5%)",
                "focus": "BAT's capital allocation statements regarding its remaining stake in ITC directly drive ITC's supply overhang or relief rallies",
                "handle_or_source": "BAT London LSE Filings & Earnings Calls",
            },
        ],
        "key_stakeholders_watch": [
            "British American Tobacco (BAT — Tobacco Manufacturers India Ltd ~25.5%; trimmed ~3.5% via block deal earlier, watched closely)",
            "Life Insurance Corporation of India (LIC — ~15.2% massive domestic anchor)",
            "Specified Undertaking of the Unit Trust of India (SUUTI — ~7.8% Govt-linked stake)",
            "SBI Mutual Fund, ICICI Prudential MF & GIC Singapore (absorbed foreign block supply)",
        ],
        "promoter_search_queries": [
            "Sanjiv Puri ITC FMCG OR agri OR cigarette tax statement",
            "ITC BAT stake sale OR ITC Hotels OR dividend",
        ],
        "competitors": [
            {
                "symbol": "HINDUNILVR",
                "name": "Hindustan Unilever Ltd (HUL)",
                "segment": "FMCG Personal Care & Packaged Foods",
                "impact_driver": "HUL's rural vs urban volume growth & input cost commentary sets the tone for ITC's FMCG business.",
            },
            {
                "symbol": "NESTLEIND",
                "name": "Nestle India Ltd",
                "segment": "Packaged Foods, Instant Noodles (Maggi vs YiPPee!) & Dairy",
                "impact_driver": "Wheat, edible oil, and packaging inflation + urban food pricing power read-across.",
            },
            {
                "symbol": "VBL",
                "name": "Varun Beverages Ltd",
                "segment": "Beverages & Snacks Distribution",
                "impact_driver": "Out-of-home consumption & summer beverage/snack volume momentum across Indian retail.",
            },
            {
                "symbol": "TRENT",
                "name": "Indian Consumption Benchmark",
                "segment": "Domestic Consumption & Retail Allocation",
                "impact_driver": "Institutional rotation between defensive staple compounders (ITC) and high-beta discretionary.",
            },
        ],
    },
    "BIRLACABLE": {
        "promoter_group": "M.P. Birla Group (Vindhya Telelinks Ltd, Universal Cables Ltd & Birla Group Entities — 66.35% Promoter Stake)",
        "key_leaders": [
            {
                "name": "Harsh V. Lodha",
                "role": "Chairman, Birla Cable & M.P. Birla Group",
                "focus": "Strategic capacity expansion in Optical Fiber Cables (OFC), export market diversification (Europe/Americas) & MP Birla Group governance",
                "handle_or_source": "Birla Cable AGM / Board Releases / MP Birla Group Filings",
            },
            {
                "name": "R. Sridharan",
                "role": "Chief Executive Officer & Manager",
                "focus": "Rewa plant capacity utilization, BharatNet Phase-III tender execution, specialty telecom & structured copper/data cables",
                "handle_or_source": "BSE/NSE Earnings Releases & Management Discussion (MD&A)",
            },
            {
                "name": "Somesh Laddha",
                "role": "Chief Financial Officer",
                "focus": "Working capital cycle (receivables from telcos/EPCs), optical fiber preform cost management & debt reduction",
                "handle_or_source": "Quarterly Financial Results & BSE Filings",
            },
        ],
        "key_stakeholders_watch": [
            "Vindhya Telelinks Ltd (19.34% Promoter Group EPC & Telecom anchor)",
            "Universal Cables Ltd (13.00% Promoter Group power cable sister concern)",
            "August Agents Ltd & Insilco Agents Ltd (MP Birla Group holding entities — total Promoter stake tightly held at 66.35% with 0% pledge)",
        ],
        "promoter_search_queries": [
            "Birla Cable optical fiber OR BharatNet OR Harsh Lodha",
            "Vindhya Telelinks OR Birla Cable order win OR results",
        ],
        "competitors": [
            {
                "symbol": "HFCL",
                "name": "HFCL Ltd",
                "segment": "Optical Fiber Cables (OFC) & Telecom Turnkey EPC",
                "impact_driver": "HFCL's BharatNet Phase-III order wins, global OFC realization prices & anti-dumping duty updates directly move Birla Cable.",
            },
            {
                "symbol": "STLTECH",
                "name": "Sterlite Technologies Ltd (STL)",
                "segment": "Optical Fiber Preform, OFC & US/EU Export Markets",
                "impact_driver": "STL's commentary on global optical fiber inventory destocking and per-fkm fiber pricing is the primary margin indicator for Birla Cable.",
            },
            {
                "symbol": "FINCABLES",
                "name": "Finolex Cables Ltd",
                "segment": "Communication Cables, Optic Fiber & Electrical Wires",
                "impact_driver": "Copper rod & polymer insulation raw material pass-through and domestic telecom cable demand.",
            },
            {
                "symbol": "POLYCAB",
                "name": "Polycab India Ltd",
                "segment": "Cables, Wires & BharatNet EPC Participation",
                "impact_driver": "Sector valuation multiple leader and indicator of domestic infrastructure cable demand.",
            },
        ],
    },
}

# Reported Annual Report R&D intensity (% of turnover) from statutory BRSR / Board Report
# Annexures for major NSE companies where Ind-AS groups R&D inside Other Expenses.
REPORTED_RD_INTENSITY_PCT: dict[str, dict[str, float | str]] = {
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
    "TATAELXSI": {
        "pct_of_sales": 2.85,
        "focus": "AVENUE autonomous vehicle & SDV software stack, TETHER connected vehicle platform, iCX media AI & medical device regulatory IP",
        "source": "Annual Report Technology Absorption & R&D Annexure",
    },
    "ITC": {
        "pct_of_sales": 0.38,
        "focus": "ITC Life Sciences & Technology Centre (LSTC Bengaluru — 350+ scientists, 900+ patent filings in agro-forestry, nutrition & sustainable packaging)",
        "source": "Annual Report LSTC R&D Annexure",
    },
    "BIRLACABLE": {
        "pct_of_sales": 0.65,
        "focus": "High-fiber-count ribbon optical fiber cables, bend-insensitive FTTH micro-cables, flame-retardant specialty telecom & structured LAN cables at Rewa R&D facility",
        "source": "Annual Report Form B (R&D & Technology Absorption with NKT/Ericsson heritage)",
    },
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
    "INFY": {
        "pct_of_sales": 1.35,
        "focus": "Infosys Topaz AI-first suite, Cobalt cloud assets & Finacle next-gen core banking",
        "source": "Annual Report Form 20-F / BRSR R&D Disclosure",
    },
    "KPITTECH": {
        "pct_of_sales": 4.20,
        "focus": "Software-Defined Vehicle (SDV) middleware, autonomous driving, electric powertrain & AUTOSAR stacks",
        "source": "Annual Report R&D Expenditure Note",
    },
    "PERSISTENT": {
        "pct_of_sales": 2.65,
        "focus": "SASVA AI software engineering platform, hyperscaler accelerators & healthcare IP",
        "source": "Annual Report R&D & IP Disclosure",
    },
    "MARUTI": {
        "pct_of_sales": 1.65,
        "focus": "Rohtak R&D centre, EV powertrain localization, hybrid & flex-fuel platforms",
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
