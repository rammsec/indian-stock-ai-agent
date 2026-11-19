# News Entity & Signal Quality Report (Before vs After Prototype `quality_audit.py`)

**Generated From:** `backend/news_intel/quality_audit.py` (`audit_news_intelligence_datasets()`) against live `data/predictions/{structured_events,quarantined_events,cross_stock_impacts}.parquet`  
**Branch / Worktree:** `research/news-quality` (`/home/antigravity/worktrees/research-news-quality`)  
**Freeze Status:** Production modules and the 17 numerical news features (`backend/news_intel/features.py`) were **not modified**.

---

## 1. Before / After Headline Quality Metrics (Live Parquet Replay)

| Metric | Production Baseline (`structured_events.parquet` / `cross_stock_impacts.parquet`) | Isolated Prototype (`quality_audit.py` v2) | Delta / Improvement |
|---|---|---|---|
| Total structured events evaluated | **254** | **196** retained (**156** unique non-dup) | **-58** noisy/contaminated rows quarantined |
| Additional quarantined events | 0 (beyond 32 already in `quarantined_events.parquet`) | **+58** (`31` generic price, `22` irrelevant entity/acronym, `5` SEO/podcast spam) | Removes **22.8%** noise from retained corpus |
| Unstripped `" - <Source>"` suffixes in descriptions | **254 / 254 (100.0%)** | **0 / 196 (0.0%)** | Eliminates publisher-token contamination (`PRESS Insider`) |
| `effective_trading_date` shifted by `[:22]` GMT->IST bug | **31 / 254 (12.2%)** | **0** (when full RFC-2822 or UTC recovery is applied) | Restores strict PIT cutoff accuracy |
| Unflagged same-ticker duplicate pairs (Jaccard `>= 0.55`) | **43 pairs** | **0 unflagged pairs** (**40** cleanly flagged `is_duplicate_report=True`) | Eliminates cross-category & cross-source double-counting |
| Cross-category identical-headline groups | **27 groups** | **0** unlinked groups (unified via category-independent `canonical_event_id`) | Prevents 1 headline inflating multiple category features |
| Self-referential duplicates (`canonical_event_id == event_id`) | **2** in `docs/data/phase5_intelligence.json` | **0** | Guarantees valid canonical pointers |
| Reclassified events (category fixed) | — | **24** non-duplicate events reclassified | Fixes `Cognizant` (`promoter_management` -> `competitor`), `Britannia CEO` (`promoter_management` -> `company_news`), etc. |
| Total cross-stock impacts | **154** | **89** (-42.2% noise) | Removes false-positive spillover clutter |
| `RELIANCE -> BIRLACABLE` (`supplier_customer`) spillovers | **16** (all Jio IPO/listing/speech items) | **0** (0 genuine FTTH/5G capex items in current batch) | **100% of Jio IPO false positives eliminated** |
| `TCS -> TATAELXSI` (`sector_peer`) spillovers | **25** (incl. Yahoo quote pages, `"Mumbai"`/`"Limited"` matches) | **10** (strictly IT spending, GenAI/AI, BFSI, IT rally/deal items) | **-60.0%** noisy peer spillovers removed |

---

## 2. Category Distribution: Before vs After (Non-Duplicate Events)

### Before (`structured_events.parquet`, 247 non-duplicate rows)
| Ticker | `company_news` | `competitor` | `global_macro` | `industry_sector` | `promoter_management` | `regulatory_government` | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BIRLACABLE` | 8 | 16 | 1 | 4 | 0 | 0 | 29 |
| `ITC` | 20 | 10 | 6 | 7 | 4 | 10 | 57 |
| `RELIANCE` | 28 | 14 | 5 | 4 | 6 | 5 | 62 |
| `TATAELXSI` | 6 | 12 | 4 | 0 | 1 | 5 | 28 |
| `TCS` | 23 | 11 | 10 | 12 | 10 | 4 | 70 |
| **Total** | **85** | **63** | **26** | **27** | **21** | **24** | **246*** |

*\*Plus 1 `BIRLACABLE` `promoter_management` row marked `is_duplicate_report=True` (7 duplicates total).*

### After (`quality_audit.py` v2, 156 unique non-duplicate rows)
| Ticker | `company_news` | `competitor` | `global_macro` | `industry_sector` | `promoter_management` | `regulatory_government` | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BIRLACABLE` | 5 | 13 | 0 | 2 | 0 | 0 | 20 |
| `ITC` | 15 | 8 | 6 | 2 | 0 | 5 | 36 |
| `RELIANCE` | 25 | 11 | 4 | 0 | 0 | 4 | 44 |
| `TATAELXSI` | 3 | 5 | 1 | 4 | 0 | 0 | 13 |
| `TCS` | 14 | 10 | 8 | 7 | 0 | 4 | 43 |
| **Total** | **62** | **47** | **19** | **15** | **0** | **13** | **156** |

> [!NOTE]
> `promoter_management` drops to `0` on the 254-row RSS batch because all 24 legacy `promoter_management` rows were either generic/earnings articles (`"RIL posts record Q2 revenue..."`), third-party exec quotes (`"Britannia CEO..."`), Tata Sons governance disputes (`"Jimmy Tata..."`), `ITC Holdings` US utility contamination, or `Cognizant` (`"- PRESS Insider"`). Verified NSE bulk/block promoter deals (`metadata["is_verified_promoter"]`) still route cleanly to `promoter_management`. Meanwhile genuine `TCS` H-1B/visa items (`4`) and `ITC` RBI/GST items (`5`) remain under `regulatory_government`, while `"Indian Economic Policy Is Adrift"`, `"Podcast"`, and `"ease ITC claims"` (Input Tax Credit) are quarantined.

---

## 3. Concrete Before/After Defect Resolutions

| Sample Headline | Baseline Behavior | `quality_audit.py` v2 Behavior |
|---|---|---|
| `"Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech - PRESS Insider"` | `TCS` / `promoter_management` / **`positive` (0.78)** | `TCS` / `competitor` (`HCLTECH`) / **`negative`** |
| `"Draft GST return changes aim to cut tax notices, ease ITC claims for firms - Business Standard"` | `ITC` / `regulatory_government` / `neutral` | **Quarantined (`irrelevant_entity_contamination`)** (GST Input Tax Credit) |
| `"ITC Holdings CEO says data center growth is helping cut Midwest transmission rates 13% - CNBC"` | `ITC` / `promoter_management` / `negative` | **Quarantined (`irrelevant_entity_contamination`)** (US utility ITC Holdings) |
| `"Some relief as crude slips below USD 90 for the first time since August - The New Indian Express"` | `RELIANCE` / `competitor` (`IOC` via `"indian"`) + `global_macro` | Unified to single `RELIANCE` / `global_macro` (no `IOC` false match) |
| `"Tata Group shares rise as RBI reiterates Tata Sons listing requirement; Tata Chemicals gains up to 6%"` | `TATAELXSI` / `competitor` (`TATATECH` via `"tata"`) | **Quarantined (`irrelevant_entity_contamination`)** |
| `"Intellivate Cap Nine Monthly Results... - The Economic Times"` / `"Upcoming Honda Compact SUV... - CarLelo"` | `TATAELXSI` / `competitor` (fallback line 311 leak) | **Quarantined (`irrelevant_entity_contamination`)** |
| `"EV Charging Station Market Size, Share, Trends 2036 - Future Market Insights"` | `TATAELXSI` / `global_macro` | **Quarantined (`generic_seo_noise`)** |
| `"ITC Share Price - Upstox"` / `"RELIANCE Share Price — NSE:RELIANCE - TradingView"` / `"Stock Forecast & Analyst Predictions - Simply Wall St"` | Admitted to `company_news` / `promoter_management` | **Quarantined (`generic_price_headline`)** |
| `"Birla Cable Ltd Downgraded to Buy Amid Mixed Technical Signals and Strong Financials"` | `positive (0.70)` | `negative` / `neutral` conflict handled cleanly (`downgraded` recognized) |
| `"HFCL Secures ₹834.36 Crore Export Orders For Optical Fiber Cables - Sahi"` | `neutral (0.54)` | **`positive`** (`secures` + `orders` recognized) |
| `"Tata Consultancy Services Limited (TCS.NS) stock price... - Yahoo Finance UK"` (`"limited"` -> `"it"`) | `TCS -> TATAELXSI` `sector_peer` spillover | **Quarantined** + blocked by `\bit\b` word boundary |
| `"Billionaire Ambani cries tears of joy as Jio Platforms becomes..."` | `RELIANCE -> BIRLACABLE` `supplier_customer` (`negative`) spillover | **Blocked** (requires capex/5G/FTTH/fiber context, excludes IPO/speech) |
| `'Sun, 15 Nov 2026 06:51'` (`[:22]` truncated GMT) | Parsed as `06:51+05:30` IST (`-5h30m` error) | Recovered as `12:21+05:30` IST (and full RFC-2822 GMT parsed natively) |

---

## 4. Recommended Post-Freeze Rollout Plan (After 8 Dec Prospective Window)

1. Remove `[:22]` truncation in `backend/intel_engine.py:69` and ensure `entry.published` passes full RFC-2822 timezone metadata into `extractor.py`.
2. Port `strip_source_suffix`, `normalize_headline_v2`, `_GENERIC_HEADLINE_PATTERNS_V2`, `_SEO_SPAM_PATTERNS_V2`, `_FALSE_POSITIVE_ENTITY_PATTERNS_V2`, `_COMPETITOR_EXACT_TOKENS_V2`, and `resolve_category_and_entity_v2` into `backend/news_intel/extractor.py` (without altering the 17 feature definitions in `features.py`).
3. Port `_SPILLOVER_RULES_V2` regex word-boundary matching into `backend/news_intel/cross_stock.py`.
4. Filter `not e.is_duplicate_report` in `backend/news_intel/pipeline.py:162` (`events_timeline`) and run a one-time clean rebuild of `structured_events.parquet` & `cross_stock_impacts.parquet` after the prospective evaluation window closes.
