"""Indian Stock Market Instrument Master & ISIN Directory.

Provides deterministic mapping across NSE ticker symbols, BSE scrip codes,
and unique International Securities Identification Numbers (ISINs).
"""

from __future__ import annotations

from .schema import InstrumentInfo

INSTRUMENT_MASTER: dict[str, InstrumentInfo] = {
    "RELIANCE": InstrumentInfo(
        symbol="RELIANCE",
        nse_symbol="RELIANCE",
        bse_code="500325",
        isin="INE002A01018",
        company_name="Reliance Industries Ltd",
        sector="Energy & Petrochemicals",
        is_fno_eligible=True,
    ),
    "TCS": InstrumentInfo(
        symbol="TCS",
        nse_symbol="TCS",
        bse_code="532540",
        isin="INE467B01029",
        company_name="Tata Consultancy Services Ltd",
        sector="Information Technology",
        is_fno_eligible=True,
    ),
    "HDFCBANK": InstrumentInfo(
        symbol="HDFCBANK",
        nse_symbol="HDFCBANK",
        bse_code="500180",
        isin="INE040A01034",
        company_name="HDFC Bank Ltd",
        sector="Financial Services / Banking",
        is_fno_eligible=True,
    ),
    "INFY": InstrumentInfo(
        symbol="INFY",
        nse_symbol="INFY",
        bse_code="500209",
        isin="INE009A01021",
        company_name="Infosys Ltd",
        sector="Information Technology",
        is_fno_eligible=True,
    ),
    "ICICIBANK": InstrumentInfo(
        symbol="ICICIBANK",
        nse_symbol="ICICIBANK",
        bse_code="532174",
        isin="INE090A01021",
        company_name="ICICI Bank Ltd",
        sector="Financial Services / Banking",
        is_fno_eligible=True,
    ),
    "ITC": InstrumentInfo(
        symbol="ITC",
        nse_symbol="ITC",
        bse_code="500875",
        isin="INE154A01025",
        company_name="ITC Ltd",
        sector="Fast Moving Consumer Goods",
        is_fno_eligible=True,
    ),
    "TATAELXSI": InstrumentInfo(
        symbol="TATAELXSI",
        nse_symbol="TATAELXSI",
        bse_code="500408",
        isin="INE670A01012",
        company_name="Tata Elxsi Ltd",
        sector="Information Technology / Design",
        is_fno_eligible=True,
    ),
    "SBIN": InstrumentInfo(
        symbol="SBIN",
        nse_symbol="SBIN",
        bse_code="500112",
        isin="INE062A01020",
        company_name="State Bank of India",
        sector="Financial Services / Banking",
        is_fno_eligible=True,
    ),
    "BHARTIARTL": InstrumentInfo(
        symbol="BHARTIARTL",
        nse_symbol="BHARTIARTL",
        bse_code="532454",
        isin="INE397D01024",
        company_name="Bharti Airtel Ltd",
        sector="Telecommunications",
        is_fno_eligible=True,
    ),
    "LT": InstrumentInfo(
        symbol="LT",
        nse_symbol="LT",
        bse_code="500510",
        isin="INE018A01030",
        company_name="Larsen & Toubro Ltd",
        sector="Capital Goods / Infrastructure",
        is_fno_eligible=True,
    ),
}

# Lookup indices
_ISIN_MAP: dict[str, InstrumentInfo] = {inst.isin: inst for inst in INSTRUMENT_MASTER.values()}
_BSE_MAP: dict[str, InstrumentInfo] = {inst.bse_code: inst for inst in INSTRUMENT_MASTER.values()}


def get_instrument_by_symbol(symbol: str) -> InstrumentInfo | None:
    """Retrieve instrument metadata by canonical or exchange symbol."""
    sym = symbol.upper().replace(".NS", "").replace(".BO", "")
    return INSTRUMENT_MASTER.get(sym)


def get_instrument_by_isin(isin: str) -> InstrumentInfo | None:
    """Retrieve instrument metadata by ISIN."""
    return _ISIN_MAP.get(isin.upper())


def get_instrument_by_bse_code(bse_code: str) -> InstrumentInfo | None:
    """Retrieve instrument metadata by BSE scrip code."""
    return _BSE_MAP.get(bse_code)
