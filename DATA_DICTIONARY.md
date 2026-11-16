# DATA_DICTIONARY.md — Canonical Schemas, Units & Provenance Specification

All datasets persisted in `/data/processed/*.parquet` and exposed via DuckDB views (`v_equities_daily`, `v_indices_daily`, `v_provenance_records`, `v_market_regimes`, `v_data_quality`) conform to schema version `1.0.0` defined in [`backend/models.py`](file:///home/antigravity/indian-stock-ai-agent/backend/models.py).

---

## 1. Validation Status Taxonomy (`ValidationStatus`)

Every record in BharatAlpha AI explicitly distinguishes data reliability:
- **`VERIFIED`**: Directly observed from an official exchange report (`NSE_SecBhavdata_Full`, `NSE_IndCloseAll_Archive`) or audited financial statement (`Screener_Consolidated_PL`) and passes range/consistency checks.
- **`ESTIMATED`**: Derived from a documented statutory disclosure (e.g., Annual Report BRSR R&D % applied to reported turnover) where line-item P&L granularity is grouped under Ind-AS `Other Expenses`.
- **`MISSING`**: Value was not reported or unavailable for that date/security (stored as `NULL` / `None`, **never silently replaced with `0.0`**).
- **`STALE`**: Observation date is older than the maximum permissible freshness threshold (e.g., >10 calendar days for daily prices).
- **`CONFLICTING`**: Cross-check failed (e.g., shareholding percentages sum to `<98%` or `>102%`, or `high < low`).

---

## 2. Canonical Provenance Record (`provenance_records.parquet` / `v_provenance_records`)

| Column | Type | Unit | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `symbol` | `VARCHAR` | Ticker | No | NSE/BSE symbol (e.g., `RELIANCE`, `TCS`, `TATAELXSI`) |
| `metric` | `VARCHAR` | Name | No | Standardized metric identifier (e.g., `current_price`, `delivery_pct`, `annual_revenue_cr`) |
| `value` | `VARCHAR` | Metric-specific | Yes (`NULL` if missing) | Observed value (stored as nullable string/float; never silent zero) |
| `source` | `VARCHAR` | Provider ID | No | Exact upstream source (`NSE_SecBhavdata_Full`, `Screener_Consolidated_PL`, etc.) |
| `observation_date` | `VARCHAR` | ISO Date / Period | No | Market session date (`YYYY-MM-DD`) or fiscal period (`Mar 2025`, `Dec 2025`) |
| `publication_timestamp` | `VARCHAR` | ISO-8601 (`+05:30`) | No | Point-in-time timestamp when the data became publicly available (prevents look-ahead bias) |
| `collection_timestamp` | `VARCHAR` | ISO-8601 (`+05:30`) | No | Timestamp when BharatAlpha AI ingested the record |
| `validation_status` | `VARCHAR` | Enum | No | `VERIFIED`, `ESTIMATED`, `MISSING`, `STALE`, or `CONFLICTING` |
| `data_version` | `VARCHAR` | SemVer | No | Schema version (`1.0.0`) |

---

## 3. Daily Equity Price, Volume & Delivery Table (`equities_daily.parquet` / `v_equities_daily`)

| Column | Type | Unit | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `symbol` | `VARCHAR` | NSE Symbol | No | Equity symbol (`RELIANCE`, `TCS`, `TATAELXSI`, `ITC`, `BIRLACABLE`, etc.) |
| `series` | `VARCHAR` | NSE Series | No | `EQ` (rolling settlement) or `BE` (trade-for-trade) |
| `observation_date` | `VARCHAR` | `YYYY-MM-DD` | No | NSE trading session date |
| `open` | `DOUBLE` | INR (`₹`) | Yes | Session opening price |
| `high` | `DOUBLE` | INR (`₹`) | Yes | Session intraday high (never synthetically fabricated) |
| `low` | `DOUBLE` | INR (`₹`) | Yes | Session intraday low (never synthetically fabricated) |
| `close` | `DOUBLE` | INR (`₹`) | Yes | Official NSE closing price |
| `last_price` | `DOUBLE` | INR (`₹`) | Yes | Last traded price (`LAST_PRICE` from NSE Bhavcopy) |
| `prev_close` | `DOUBLE` | INR (`₹`) | Yes | Previous session closing price |
| `vwap` | `DOUBLE` | INR (`₹`) | Yes | Volume-Weighted Average Price (`AVG_PRICE` from NSE Bhavcopy; `NULL` if unavailable) |
| `adj_close` | `DOUBLE` | INR (`₹`) | Yes | Split- and dividend-adjusted closing price |
| `volume` | `DOUBLE` | Shares | Yes | Total traded quantity (`TTL_TRD_QNTY`) |
| `turnover_cr` | `DOUBLE` | INR Crores (`₹ Cr`) | Yes | Total traded value in ₹ Crores (`TURNOVER_LACS / 100`) |
| `trades_count` | `BIGINT` | Trades | Yes | Total number of executed trades (`NO_OF_TRADES`) |
| `deliverable_qty` | `DOUBLE` | Shares | Yes | Shares taken for demat delivery (`DELIV_QTY`; `NULL` when unreported) |
| `delivery_pct` | `DOUBLE` | Percentage (`0–100%`) | Yes | Deliverable quantity as % of traded volume (`DELIV_PER`; `NULL` when unreported) |
| `dividends` | `DOUBLE` | INR per share | No | Cash dividend ex-date amount (`0.0` on non-ex dates) |
| `stock_splits` | `DOUBLE` | Ratio | No | Split/bonus multiplier on ex-date (`0.0` on non-ex dates) |
| `source` | `VARCHAR` | Source ID | No | `NSE_SecBhavdata_Full` or `YahooFinance_Supplementary` |
| `publication_timestamp` | `VARCHAR` | ISO-8601 (`+05:30`) | No | `T18:00:00+05:30` for NSE Bhavcopy; `T15:30:00+05:30` for close |
| `collection_timestamp` | `VARCHAR` | ISO-8601 (`+05:30`) | No | Ingestion timestamp |
| `ohlcv_validation_status` | `VARCHAR` | Enum | No | `VERIFIED` or `MISSING` |
| `delivery_validation_status` | `VARCHAR` | Enum | No | `VERIFIED` (from NSE Bhavcopy) or `MISSING` (supplementary historical bar) |
| `data_version` | `VARCHAR` | SemVer | No | `1.0.0` |

---

## 4. Daily Market & Sector Index Table (`indices_daily.parquet` / `v_indices_daily`)

| Column | Type | Unit | Nullable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `index_code` | `VARCHAR` | Code | No | Canonical code (`NIFTY_50`, `NIFTY_500`, `NIFTY_MIDCAP_100`, `INDIA_VIX`, `NIFTY_IT`, etc.) |
| `index_name` | `VARCHAR` | Name | No | Official NSE index name (`Nifty 50`, `India VIX`, `Nifty Bank`, etc.) |
| `category` | `VARCHAR` | Enum | No | `broad`, `volatility`, or `sector` |
| `observation_date` | `VARCHAR` | `YYYY-MM-DD` | No | Trading session date |
| `open`, `high`, `low`, `close` | `DOUBLE` | Index Points | Yes (`close` No) | Daily index OHLC |
| `change_pct` | `DOUBLE` | `%` | Yes | 1-day percentage change |
| `volume` | `DOUBLE` | Shares | Yes | Index constituent volume |
| `turnover_cr` | `DOUBLE` | `₹ Cr` | Yes | Index constituent turnover in ₹ Crores |
| `pe_ratio`, `pb_ratio`, `div_yield` | `DOUBLE` | Ratio / `%` | Yes | Official NSE Index P/E, P/B, and Dividend Yield % from `ind_close_all` |
| `source` | `VARCHAR` | Source ID | No | `NSE_IndCloseAll_Archive` or `YahooFinance_Index` |

---

## 5. Market Regime Snapshot (`market_regimes.parquet` / `v_market_regimes`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `observation_date` | `VARCHAR` | Session date (`YYYY-MM-DD`) |
| `regime` | `VARCHAR` | `Strong Bull`, `Weak Bull`, `Sideways`, `Bear`, or `High Volatility` |
| `nifty_50_close`, `nifty_50_dma_50`, `nifty_50_dma_200` | `DOUBLE` | Nifty 50 close and moving averages |
| `india_vix_close`, `india_vix_change_pct` | `DOUBLE` | India VIX level and 1-day % change |
| `advance_count`, `decline_count`, `advance_decline_ratio` | `BIGINT` / `DOUBLE` | Exchange-wide NSE `EQ` series advances, declines, and A/D ratio |
| `pct_above_50dma`, `pct_above_200dma` | `DOUBLE` | % of tracked universe trading above 50-DMA and 200-DMA |
| `allow_aggressive_longs` | `BOOLEAN` | `False` during `Bear` and `High Volatility` regimes; `True` otherwise |
