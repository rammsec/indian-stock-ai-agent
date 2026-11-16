"""5-Stock Pilot Universe & Cross-Stock Relationship Map (Phase 5 — Section 5)."""

from __future__ import annotations

from typing import Any

PILOT_UNIVERSE_5: tuple[str, ...] = ("RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE")

RELATIONSHIP_MAP: dict[str, dict[str, Any]] = {
    "RELIANCE": {
        "company_name": "Reliance Industries Ltd",
        "sector": "Energy, Telecom & Retail",
        "nse_sector_index": "NIFTY_ENERGY",
        "competitors": [
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "segment": "Telecom (Jio vs Airtel)", "asymmetry": "competitive_zero_sum"},
            {"symbol": "ONGC", "name": "ONGC", "segment": "Upstream Oil & Gas", "asymmetry": "commodity_peer"},
            {"symbol": "DMART", "name": "Avenue Supermarts (DMart)", "segment": "Organized Retail", "asymmetry": "competitive_zero_sum"},
            {"symbol": "IOC", "name": "Indian Oil Corp", "segment": "Refining & Marketing", "asymmetry": "commodity_peer"},
        ],
        "suppliers_customers": [
            "BIRLACABLE (Optical fiber & telecom cable supplier ecosystem for Jio FTTH/5G)",
            "HFCL (Telecom equipment & OFC vendor for Jio)",
            "STERLITE / STLTECH (Optical fiber partner/vendor)",
        ],
        "sector_peers": ["ONGC", "IOC", "BPCL", "BHARTIARTL", "DMART"],
        "commodity_dependencies": [
            {"commodity": "Brent Crude Oil", "keywords": ["crude", "brent", "opec", "oil price", "refining margin", "grm"], "direction": "mixed_refining_margin"},
            {"commodity": "Natural Gas & Petrochemicals", "keywords": ["petchem", "polymer", "natural gas", "lng"], "direction": "positive_on_spread_expansion"},
        ],
        "regulatory_exposures": [
            "TRAI / DoT telecom tariff & spectrum regulations",
            "Government windfall tax on crude & diesel exports",
            "SEBI & CCI retail/media M&A approvals",
        ],
        "global_macro_links": [
            "Global refining margins (Singapore GRM) & OPEC+ supply cuts",
            "USD/INR exchange rate (export refining & USD debt)",
            "US Fed rates & global EM FII flows",
        ],
    },
    "TCS": {
        "company_name": "Tata Consultancy Services Ltd",
        "sector": "IT Services & Consulting",
        "nse_sector_index": "NIFTY_IT",
        "competitors": [
            {"symbol": "INFY", "name": "Infosys", "segment": "Tier-1 Indian IT", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "HCLTECH", "name": "HCL Technologies", "segment": "Tier-1 Indian IT", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "WIPRO", "name": "Wipro", "segment": "Tier-1 Indian IT", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "ACN", "name": "Accenture", "segment": "Global IT Consulting", "asymmetry": "demand_bellwether"},
        ],
        "suppliers_customers": [
            "US & European BFSI, Retail & Telecom enterprise IT budgets",
            "Hyperscalers (AWS, Microsoft Azure, Google Cloud, NVIDIA AI partnerships)",
        ],
        "sector_peers": ["INFY", "HCLTECH", "WIPRO", "TECHM", "TATAELXSI"],
        "commodity_dependencies": [
            {"commodity": "USD/INR Exchange Rate", "keywords": ["rupee", "usd/inr", "inr", "dollar", "forex"], "direction": "weak_inr_positive"},
        ],
        "regulatory_exposures": [
            "US H-1B visa rules & immigration fees",
            "US/EU data privacy & AI governance regulations",
            "SEBI buyback & dividend taxation rules",
        ],
        "global_macro_links": [
            "US Fed interest rates & US recession/BFSI IT spending outlook",
            "Nasdaq / Accenture quarterly guidance",
            "USD/INR currency movement",
        ],
    },
    "TATAELXSI": {
        "company_name": "Tata Elxsi Ltd",
        "sector": "ER&D & Embedded AI/Auto Design",
        "nse_sector_index": "NIFTY_IT",
        "competitors": [
            {"symbol": "KPITTECH", "name": "KPIT Technologies", "segment": "Automotive ER&D / SDV", "asymmetry": "competitive_zero_sum"},
            {"symbol": "LTTS", "name": "L&T Technology Services", "segment": "Engineering R&D", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "TATATECH", "name": "Tata Technologies", "segment": "Automotive & Aerospace ER&D", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "PERSISTENT", "name": "Persistent Systems", "segment": "Digital Engineering", "asymmetry": "demand_readacross_and_rival"},
        ],
        "suppliers_customers": [
            "Jaguar Land Rover (JLR) & European/US OEMs (SDV, ADAS, EV platforms)",
            "Global Media & Broadcast / Healthcare medical device OEMs",
        ],
        "sector_peers": ["KPITTECH", "LTTS", "TATATECH", "CYIENT", "TCS"],
        "commodity_dependencies": [
            {"commodity": "USD/INR & EUR/INR Exchange Rate", "keywords": ["rupee", "eur", "usd/inr", "dollar", "forex"], "direction": "weak_inr_positive"},
        ],
        "regulatory_exposures": [
            "EU & US automotive safety / ADAS / EV emission mandates",
            "Global healthcare FDA / MDR medical device compliance",
        ],
        "global_macro_links": [
            "European & US automotive OEM R&D capex cycles",
            "EV / Software-Defined Vehicle (SDV) adoption pace",
            "USD/INR & EUR/INR currency movements",
        ],
    },
    "ITC": {
        "company_name": "ITC Ltd",
        "sector": "FMCG, Agri & Paperboards",
        "nse_sector_index": "NIFTY_FMCG",
        "competitors": [
            {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "segment": "FMCG Staples & Personal Care", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "NESTLEIND", "name": "Nestle India", "segment": "Packaged Foods", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "GODFRYPHLP", "name": "Godfrey Phillips", "segment": "Cigarettes", "asymmetry": "competitive_zero_sum"},
            {"symbol": "DABUR", "name": "Dabur India", "segment": "FMCG & Personal Care", "asymmetry": "demand_readacross_and_rival"},
        ],
        "suppliers_customers": [
            "Indian agricultural farmers (e-Choupal wheat, leaf tobacco, spices)",
            "Rural & urban FMCG distributors, modern trade & quick-commerce",
        ],
        "sector_peers": ["HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
        "commodity_dependencies": [
            {"commodity": "Agri Commodities (Wheat, Leaf Tobacco, Pulp, Palm Oil)", "keywords": ["wheat", "tobacco", "palm oil", "pulp", "monsoon", "agri"], "direction": "high_input_cost_negative"},
        ],
        "regulatory_exposures": [
            "Union Budget & GST Council cigarette / NCCD excise duty changes",
            "FSSAI food labeling & packaging regulations",
            "ITC Hotels demerger & SEBI/NCLT compliance",
        ],
        "global_macro_links": [
            "Indian monsoon & rural consumption demand",
            "Global soft commodity / palm oil & wood pulp inflation",
            "RBI interest rate & domestic inflation trajectory",
        ],
    },
    "BIRLACABLE": {
        "company_name": "Birla Cable Ltd",
        "sector": "Optical Fiber & Telecom Cables",
        "nse_sector_index": "NIFTY_ENERGY",
        "competitors": [
            {"symbol": "HFCL", "name": "HFCL Ltd", "segment": "Optical Fiber Cables & Telecom", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "STLTECH", "name": "Sterlite Technologies", "segment": "Optical Fiber Cables", "asymmetry": "demand_readacross_and_rival"},
            {"symbol": "VINDHYATEL", "name": "Vindhya Telelinks", "segment": "MP Birla Group Telecom Cables", "asymmetry": "group_peer_positive"},
            {"symbol": "FINCABLES", "name": "Finolex Cables", "segment": "Communication & Electrical Cables", "asymmetry": "demand_readacross_and_rival"},
        ],
        "suppliers_customers": [
            "RELIANCE (Reliance Jio FTTH/5G fiber rollout demand)",
            "BHARTIARTL (Airtel 5G/broadband fiber rollout demand)",
            "BSNL / BharatNet government rural broadband tenders",
        ],
        "sector_peers": ["HFCL", "STLTECH", "VINDHYATEL", "FINCABLES", "POLYCab"],
        "commodity_dependencies": [
            {"commodity": "Optical Fiber Preform, Copper & Aluminium", "keywords": ["copper", "aluminium", "optical fiber", "polymer", "raw material"], "direction": "high_input_cost_negative"},
        ],
        "regulatory_exposures": [
            "Government BharatNet / DoT / BSNL fiber tenders & PLI telecom incentives",
            "Anti-dumping duties on imported optical fiber",
            "US/EU broadband infrastructure (BEAD) export tariffs",
        ],
        "global_macro_links": [
            "Global 5G, FTTH & hyperscale AI data-center fiber capex",
            "US/European telecom capex & export demand",
            "Copper & industrial metal price cycles",
        ],
    },
}


def get_pilot_relationship(symbol: str) -> dict[str, Any]:
    """Return relationship metadata for a pilot stock (or safe default)."""
    sym = symbol.strip().upper()
    if sym in RELATIONSHIP_MAP:
        return RELATIONSHIP_MAP[sym]
    return {
        "company_name": sym,
        "sector": "Indian Equities",
        "nse_sector_index": "NIFTY_50",
        "competitors": [],
        "suppliers_customers": [],
        "sector_peers": [],
        "commodity_dependencies": [],
        "regulatory_exposures": [],
        "global_macro_links": [],
    }
