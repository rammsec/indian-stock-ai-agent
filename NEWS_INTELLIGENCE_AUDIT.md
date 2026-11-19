# News Intelligence Quality Audit (`backend/news_intel`)

**Audit Date:** 2026-11-19  
**Branch / Worktree:** `research/news-quality` (`/home/antigravity/worktrees/research-news-quality`)  
**Freeze Compliance:** Read-only audit of frozen production modules (`collector.py`, `extractor.py`, `cross_stock.py`, `features.py`, `combiner.py`, `pipeline.py`). All prototype fixes and regression tests are isolated in `backend/news_intel/quality_audit.py` and `tests/test_news_quality_regression.py`.

---

## 1. Executive Summary

Auditing the live Phase 5 datasets (`structured_events.parquet` = **254 rows**, `quarantined_events.parquet` = **32 rows**, `cross_stock_impacts.parquet` = **154 rows**, `docs/data/phase5_intelligence.json`) and code paths surfaced **7 concrete quality defects**:

1. **RSS Timestamp Truncation & `-5h 30m` Timezone Shift (`backend/intel_engine.py:69`, `backend/news_intel/extractor.py:176-185`)** — Google News RSS `entry.published` is sliced to `[:22]`, stripping `":00 GMT"`. `try_parse_publication_timestamp_ist` treats naive datetimes as `IST_TZ`, shifting every RSS timestamp backward by **5h 30m** and misaligning `effective_trading_date` on **31 / 254 (12.2%)** stored events.
2. **Unstripped `" - <Source>"` Suffixes (`backend/intel_engine.py:59-64`, `backend/news_intel/extractor.py:147-151`)** — **254 / 254 (100%)** stored descriptions retain `" - <Source>"`, causing `"Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech - PRESS Insider"` to match `\binsider\b` and enter `TCS` as `promoter_management` (`positive` / `high`).
3. **Cross-Category & Cross-Run Duplicate Leakage (`backend/news_intel/extractor.py:390-392, 574-610`, `backend/news_intel/pipeline.py:162`)** — Hashing `resolved_cat` inside `event_id` creates **27** `(ticker, headline)` groups duplicated across categories, **43** unflagged same-ticker pairs with Jaccard `>= 0.55`, and **2** self-referential duplicates (`canonical_event_id == event_id`) inside `docs/data/phase5_intelligence.json`'s `events_timeline`.
4. **Generic Price, Quote & SEO Spam Leakage (`backend/news_intel/extractor.py:56-62`)** — **36 / 254** stored events are generic price wrappers (`"... Share Price - Upstox"`, `"Stock Forecast & Analyst Predictions - Simply Wall St"`, `"Price Band Hitters"`, `"Top 6 Semiconductor Stocks"`, `"Four Stocks To Buy Today"`, `"Weekly market commentary"`) or SEO market-size spam (`Future Market Insights`, `IndexBox`, podcasts).
5. **Entity & Acronym False Positives (`backend/news_intel/extractor.py:51-54, 269, 311`)** — `ITC` ingests GST **Input Tax Credit** articles (`"ease ITC claims"`), `IOC` (`"Indian Oil Corp"`) expands to `"indian"`, `TATATECH` (`"Tata Technologies"`) expands to `"tata"`, and the unvalidated fallback at `extractor.py:311` admits irrelevant articles (`Intellivate Cap`, `CarLelo`, `Motoring Trends`, `RBI Bulletin`) under `competitor`, `industry_sector`, `regulatory_government`, and `global_macro`.
6. **Substring Cross-Stock Spillover Contamination (`backend/news_intel/cross_stock.py:109-120`)** — Substring matching `"it"` (`"limited"`) and `"ai"` (`"mumbai"`) generated **25** `TCS -> TATAELXSI` spillovers (15 false positives including Yahoo Finance quote pages and analyst meeting notices); bare `"jio"` generated **16** `RELIANCE -> BIRLACABLE` spillovers entirely from Jio IPO/listing/speech coverage rather than FTTH/5G capex.
7. **Competitor-Win Sentiment Polarity Inversion (`backend/news_intel/extractor.py:118-132, 332-377`)** — `"Cognizant wins ..., beating TCS"` scored `positive (0.78)` for `TCS`; `"Downgraded to Buy"` scored `positive`; `"HFCL Secures ₹834.36 Crore Export Orders"` scored `neutral` due to missing `"secures"` / `"orders"` tokens.

---

## 2. Dataset Baseline Inventory

| Artifact | Total Rows | Key Breakdown |
|---|---|---|
| `data/predictions/structured_events.parquet` | **254** (254 unique `event_id`) | `TCS`: 70, `RELIANCE`: 65, `ITC`: 57, `BIRLACABLE`: 31, `TATAELXSI`: 31 |
| `structured_events` by `event_category` | **254** | `company_news`: 88, `competitor`: 64, `industry_sector`: 28, `global_macro`: 26, `regulatory_government`: 24, `promoter_management`: 24 |
| `structured_events` duplicates & nulls | **254** | `is_duplicate_report=True`: 7; `collection_timestamp` & `validation_status` NULL: **156** (legacy pre-5.4 rows); `promoter_classification_confidence` NULL: **254** (0 pilot bulk deals) |
| `data/predictions/quarantined_events.parquet` | **32** (26 unique tuples) | `generic_price_headline`: 18, `irrelevant_entity_contamination`: 14, `invalid_or_missing_timestamp`: 0, `unverified_promoter_association`: 0 |
| `data/predictions/cross_stock_impacts.parquet` | **154** | `neutral`: 114, `positive`: 24, `negative`: 16 (`TATAELXSI`: 47, `BIRLACABLE`: 32, `ITC`: 26, `TCS`: 25, `RELIANCE`: 24) |
| `docs/data/phase5_intelligence.json` | **5** stocks, **30** `consolidated_feed` | 12 `events_timeline` + 8 `cross_stock_impacts` per stock (`generated_at`: `2026-11-19T17:28:47+05:30`) |

---

## 3. Code-Level Root Cause Inventory

| Defect | Exact File & Lines | Mechanism | Concrete Data Example |
|---|---|---|---|
| Timestamp `[:22]` truncation | `backend/intel_engine.py:69` + `extractor.py:176-185` | `entry.get("published", "")[:22]` drops `":00 GMT"`; naive datetime defaults to `IST_TZ` (`-5h30m` shift) | `quarantined_events`: `'Sun, 15 Nov 2026 06:51'`; `structured_events`: 107 rows at `07:00+05:30`, 31 wrong `effective_trading_date` |
| Source suffix not stripped | `backend/intel_engine.py:59-64` + `extractor.py:147-151` | When `entry.source.title` exists, `" - Source"` stays in `title`; `normalize_headline` keeps source tokens | Row 249 (`TCS`): `"Cognizant wins ... beating TCS and HCLTech - PRESS Insider"` -> `promoter_management` via `\binsider\b` |
| Cross-category & timeline duplicates | `extractor.py:390-392, 574-610` + `pipeline.py:162` | `event_id` hashes `resolved_cat`; `events_timeline` does not filter `is_duplicate_report`; same-cat hash collides | `phase5_intelligence.json`: `TATAELXSI` (`6fdca47cea367f9c`) & `BIRLACABLE` (`78e9d415b3b22dad`) have `is_duplicate_report=True` with `canonical_event_id == event_id` |
| Unvalidated non-company fallback | `extractor.py:311` | Only `company_news`/`promoter_management` require `mentions_target`; all other fetches pass unconditionally | `TATAELXSI` `competitor`: `"Intellivate Cap Nine Monthly Results"`, `"Upcoming Honda Compact SUV Launch Timeline"` |
| Single-word competitor tokens | `extractor.py:269` | `c_name.split()[0]` generates `"indian"` (`IOC`), `"tata"` (`TATATECH`), `"hindustan"` (`HINDUNILVR`) | Row 185 (`RELIANCE`): `"Some relief as crude slips below USD 90... - The New Indian Express"` -> `competitor` (`IOC`) |
| Acronym & namesake leaks | `extractor.py:51-54` | Missing GST `ITC` (Input Tax Credit) exclusion; legacy rows not purged by `keep="first"` upsert | Row 240 (`ITC`): `"Draft GST return changes aim to cut tax notices, ease ITC claims..."`; Row 183 (`ITC`): `"ITC Holdings CEO..."` |
| Substring spillover matching | `cross_stock.py:109-120` | `kw in desc` matches `"it"` inside `"limited"` and `"ai"` inside `"mumbai"`; `"jio"` matches Jio IPO | `TCS -> TATAELXSI`: `"TCS schedules analyst meetings in Mumbai..."`, `"TCS ... (TCS.NS) ... - Yahoo Finance UK"`; `RELIANCE -> BIRLACABLE`: 16 Jio IPO/speech items |
| Sentiment inversions / misses | `extractor.py:118-132, 332-377` | No rival-beat inversion; missing `"secures"`, `"orders"`, `"downgraded"` | `"Cognizant wins ..., beating TCS"` -> `positive (0.78)`; `"Downgraded to Buy"` -> `positive (0.70)`; `"HFCL Secures ... Export Orders"` -> `neutral` |

---

## 4. Test Coverage Gaps Identified in Existing Suite

Inspected `tests/test_phase5_news_intel.py` (19 tests) and `tests/test_phase5_4_integrity.py` (4 tests):
- Zero tests for Google News RFC-2822 GMT timestamp parsing or `[:22]` truncation detection.
- Zero tests with `" - <Source>"` suffixes in titles (e.g., `- PRESS Insider`, `- Reuters` vs `- ET Telecom`).
- Zero tests for cross-category duplicate suppression or `canonical_event_id != event_id` invariants on duplicates.
- Zero tests for `competitor`/`industry_sector`/`regulatory_government`/`global_macro` irrelevance gating (`Intellivate Cap`, `CarLelo`, `Future Market Insights`), `ITC` Input Tax Credit acronym collision, or `"indian"`/`"tata"` token collisions.
- Zero negative tests for `cross_stock.py` substring collisions (`"limited"`/`"mumbai"`/`"jio ipo"`).
- Zero tests for rival-win negative polarity (`"beating TCS"`).

All gaps are now covered by the isolated regression suite in `tests/test_news_quality_regression.py`.
