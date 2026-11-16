# BharatAlpha AI — Phase 5 Architecture Assessment, 237-Feature Inventory & Pilot Relationship Map

## 1. Architecture Assessment (v3.2 Quantitative Core → Phase 5 Multi-Dimensional Intelligence)

### Preserved v3.2 Quantitative & Integrity Guarantees
- **Walk-Forward Validation**: Expanding-window chronological splits (`backend/backtesting/walk_forward.py`) with **5-session purge** + **1-session embargo** for 1D/5D horizons and **10-session purge** + **1-session embargo** for 10D horizons.
- **In-Fold Feature Selection**: All correlation pruning and top-K (`K=40`) selection occur strictly inside each training fold (`BharatAlphaQuantEnsemble.fit`).
- **Execution & Cost Realism**: Next-session `09:15 IST` open entry (`next_open_delivery_capped`), explicit gap handling, Indian NSE delivery/intraday costs (STT, Stamp Duty, Exchange/SEBI fees, 18% GST, DP charges, ATR/Amihud slippage), and `naive_momentum` benchmark retained.
- **Data Storage**: ZSTD-compressed Parquet + in-memory DuckDB views (`backend/storage/duckdb_store.py`), extended with immutable `structured_events.parquet` (`v_structured_events`) and `cross_stock_impacts.parquet` (`v_cross_stock_impacts`).

### New Phase 5 Modules (`backend/news_intel/`)
1. `relationships.py`: Defines `PILOT_UNIVERSE_5` (`RELIANCE`, `TCS`, `TATAELXSI`, `ITC`, `BIRLACABLE`) and `RELATIONSHIP_MAP` (competitors, suppliers/customers, sector peers, commodity dependencies, regulatory exposures, global macro links).
2. `collector.py`: Six separate category collectors (`company_news`, `promoter_management`, `industry_sector`, `competitor`, `regulatory_government`, `global_macro`) reusing `backend/intel_engine.py` and `data/processed/nse_bulk_block_deals.parquet`.
3. `extractor.py`: Headline normalization, token Jaccard deduplication (`is_duplicate_report`, `canonical_event_id`), SQLite caching, cost-guarded optional Gemini (`ENABLE_GEMINI_EXTRACTOR=false` by default, `MAX_GEMINI_CALLS_PER_DAY=10`), deterministic fallback, strict `StructuredNewsEvent` validation, and `>15:30 IST` / holiday -> next NSE trading session `effective_trading_date`.
4. `features.py`: 17 numerical news features (`NEWS_FEATURE_NAMES`) with strict `publication_timestamp <= as_of_ts` PIT cutoff, plus `reconstruct_pit_historical_news_features` for honest historical walk-forward testing without fabricating historical news.
5. `cross_stock.py`: Asymmetric `CrossStockImpactItem` mapping (e.g., rival contract win = `negative` threat; rival disruption = `positive` opportunity; `RELIANCE` Jio FTTH capex -> `BIRLACABLE` supplier spillover).
6. `combiner.py`: Separated (`quant_signal`, `news_signal`, `cross_stock_signal`) and combined (`combined_signal`, `decision`, `confidence`, `uncertainty`) outlooks across **1D, 5D, and 10D** horizons.
7. `monitoring.py`: `MonitoringSnapshot` tracking logged predictions, evaluated outcomes, rolling 1D accuracy, recent Brier score, drift status, and data quality warnings.

---

## 2. Complete 237-Feature Inventory & Quality / Redundancy Audit

Audited from `data/features/feature_catalog.json` (7,410 rows evaluated across the stored universe):

| Feature Family | Candidate Count | Selected (Post-Audit) | Dropped (Redundant / Low-Coverage) | Audit Notes |
| --- | ---: | ---: | ---: | --- |
| 1. Price & Technical Momentum (`pm_*`) | 113 | 80 | 33 | Retained all 113 candidates; 33 dropped inside-fold due to `|corr| > 0.92` redundancy across overlapping lookbacks. |
| 2. Volume & Delivery (`vd_*`) | 26 | 24 | 2 | High-quality NSE Bhavcopy delivery & Amihud illiquidity features. |
| 3. Market & Sector Strength (`ms_*`, `rel_*`) | 36 | 33 | 3 | Nifty 50, India VIX, breadth, and relative sector strength. |
| 4. Institutional & Promoter Activity (`inst_*`) | 15 | 14 | 1 | Quarterly promoter/FII/DII changes & NSE bulk/block deal flags. |
| 5. Derivatives Positioning (`dp_*`) | 9 | 5 | 4 | 4 dropped for non-F&O coverage (e.g., `BIRLACABLE` is cash-segment only). |
| 6. Financial Quality (`fq_*`) | 20 | 19 | 1 | PIT quarterly/annual ratios (ROE, ROCE, OPM, OCF/PAT). |
| 7. Earnings & Management Guidance (`eg_*`) | 10 | 10 | 0 | 100% retained; low cross-correlation. |
| 8. News & Corporate Events (`ne_*`) | 8 | 2 | 6 | 6 dropped in historical matrix due to sparse coverage prior to Phase 5 live event logging. |
| **Total** | **237** | **187** | **50** | **All 237 candidate features preserved; zero deleted.** |

### Phase 5 Additions: 17 Numerical News & Cross-Stock Features (`NEWS_FEATURE_NAMES`)
1. `news_company_sentiment_1d`, `news_company_sentiment_5d`, `news_company_sentiment_20d`
2. `news_volume_1d`, `news_volume_5d`, `news_surprise_intensity`
3. `news_promoter_sentiment_5d`, `news_promoter_activity_flag`
4. `news_industry_sentiment_5d`
5. `news_competitor_threat_score`, `news_competitor_positive_readacross`
6. `news_regulatory_risk_score`, `news_regulatory_positive_score`
7. `news_global_macro_risk_score`
8. `news_novelty_weighted_sentiment`, `news_high_impact_event_count_5d`
9. `news_cross_stock_net_impact`

---

## 3. Pilot 5-Stock Relationship Map

| Symbol | Sector | Key Competitors | Suppliers / Customers / Peers | Commodity & Macro Links | Regulatory Exposures |
| --- | --- | --- | --- | --- | --- |
| **RELIANCE** | Energy, Telecom & Retail | `BHARTIARTL`, `ONGC`, `DMART`, `IOC` | Supplier ecosystem: `BIRLACABLE`, `HFCL`, `STLTECH` | Brent Crude, Singapore GRM, Petchem spreads, USD/INR | TRAI/DoT telecom tariffs, crude windfall tax, SEBI/CCI |
| **TCS** | IT Services & Consulting | `INFY`, `HCLTECH`, `WIPRO`, `ACN` | US/EU BFSI enterprise budgets, AWS/Azure/NVIDIA | USD/INR, US Fed rates, US/EU IT capex | US H-1B visa rules, US/EU AI & privacy, SEBI buyback tax |
| **TATAELXSI** | ER&D & Embedded AI/Auto | `KPITTECH`, `LTTS`, `TATATECH`, `PERSISTENT` | JLR, EU/US auto OEMs, broadcast/medtech OEMs | EUR/INR, USD/INR, global EV/SDV R&D cycle | EU/US ADAS/safety mandates, FDA/MDR medtech |
| **ITC** | FMCG, Agri & Paperboards | `HINDUNILVR`, `NESTLEIND`, `GODFRYPHLP`, `DABUR` | e-Choupal farmers, rural/urban FMCG channels | Wheat, leaf tobacco, palm oil, wood pulp, monsoon | Union Budget / GST cigarette excise, FSSAI, SEBI/NCLT |
| **BIRLACABLE** | Optical Fiber & Telecom Cables | `HFCL`, `STLTECH`, `VINDHYATEL`, `FINCABLES` | Customers: `RELIANCE` (Jio), `BHARTIARTL`, BSNL/BharatNet | Optical fiber preform, copper, aluminium, global 5G/FTTH | BharatNet/DoT tenders, fiber anti-dumping, US BEAD tariffs |

---

## 4. Limitations & Next-Phase Recommendations

1. **Historical News Archive Gap**: Free Google News RSS covers only ~30 days of recent articles. Historical Experiments A–E reconstruct PIT event/promoter/sector/macro proxies (`ne_*`, `inst_*`, `rel_*`, `ms_*`) rather than fabricating historical news. True news-alpha validation requires accumulating live daily snapshots in `data/predictions/structured_events.parquet` over 6–12 months.
2. **Sample Size**: The 5-stock pilot (`PILOT_UNIVERSE_5`) provides clean controlled experimentation but limited cross-sectional breadth (`N=5`). Do not expand the universe until live structured event accumulation confirms positive net Sharpe / Rank IC uplift over `naive_momentum`.
3. **Intraday Execution**: Daily OHLCV bars cannot resolve intra-bar stop-vs-target sequencing; `next_open` execution with conservative ambiguity handling remains active.
4. **Paper-Trading Only**: All Phase 5 signals and monitoring are strictly for research and paper-trading evaluation; no live broker order execution is connected.
