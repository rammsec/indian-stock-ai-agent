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

---

## 6. Point-in-Time Daily Feature Matrix (`/data/features/daily_features.parquet` / `v_daily_features`)

Generated by [`backend/features/feature_engine.py`](backend/features/feature_engine.py) and evaluated by [`backend/features/selector.py`](backend/features/selector.py). Contains **237 candidate numerical features** (and **187 selected non-collinear features** stored in `/data/features/feature_catalog.json`) plus 6 metadata columns and 7 forward return/excursion target columns:

| Family Prefix | Feature Family Name | Candidate Count | Selected Count | Key Features & Point-in-Time Lag Rules |
| :--- | :--- | :---: | :---: | :--- |
| `pm_` | **1. Price & Technical Momentum** | 113 | 81 | Multi-horizon returns (`1D`–`252D`), log returns, acceleration, SMA/EMA distances & slopes (`5`–`200`), Wilder RSI (`3`, `5`, `7`, `14`, `21`), MACD (`12,26,9`), Stochastic, Williams %R, CCI(20), Bollinger %B & width (`20`, `50`), Donchian channel positions (`20`, `55`, `252`), ATR (`7`, `14`, `21`), ATR-normalized moves, overnight gaps, candle shadows, Parkinson volatility, skewness & kurtosis. Computed strictly up to session `t` close. |
| `vd_` | **2. Volume & Delivery** | 26 | 24 | Relative volume (`5D`–`100D`), log-volume z-score, OBV z-score, CMF(20), MFI(14), VPT, Amihud illiquidity, up/down volume ratio, verified NSE `vd_delivery_pct`, `vd_delivery_vs_5d_avg`, `vd_delivery_vs_20d_avg`, `vd_vwap_dist_pct`, `vd_trades_per_cr_turnover`, and pocket-pivot flags. Unverified historical delivery remains `NaN` (never `0.0`). |
| `ms_` | **3. Market & Sector Strength** | 36 | 29 | Nifty 50 returns & stock alpha (`1D`–`60D`), NSE Sector Index returns & sector alpha (`1D`–`60D`), rolling `20D`/`60D` Beta & correlation vs Nifty 50, India VIX level/changes/z-score, and market regime score. |
| `inst_` | **4. Institutional & Promoter Activity** | 15 | 13 | Point-in-time lagged Promoter %, FII %, DII %, Public %, `1Q`/`4Q` ownership deltas, net institutional flow (`4Q`), retail dilution (`4Q`), and named holders count. Gated by statutory `45-day` / `60-day` SEBI LODR publication timestamps via backward `merge_asof`. |
| `dv_` | **5. Derivatives Positioning** | 9 | 5 | `dv_is_fno_eligible`, nullable exchange `dv_futures_oi` / `dv_options_pcr` / `dv_atm_iv` (preserved as `NaN` when unreported), and explicit price-volume positioning heuristics (`dv_long_buildup_heuristic`, `dv_short_buildup_heuristic`, `dv_short_covering_heuristic`, `dv_long_unwinding_heuristic`). |
| `fq_` | **6. Financial Quality** | 20 | 20 | Point-in-time lagged ROE, ROCE, ROE-ROCE spread, annual OPM & YoY change, NPM & YoY change, Debt/Equity, Cash/Debt, Net Cash/MCap, OCF/Net Profit, FCF/Sales, Accruals/Assets, Interest Coverage, Asset Turnover, Piotroski F-score, R&D % of Sales, CWIP/Assets, `fq_dynamic_pe`, and `fq_earnings_yield_pct`. |
| `eg_` | **7. Earnings & Management Guidance** | 10 | 7 | Point-in-time lagged quarterly revenue & net profit QoQ/YoY growth, sequential OPM expansion, revenue/profit growth acceleration, and 5Y CAGRs. |
| `ne_` | **8. News & Corporate Events** | 8 | 8 | `ne_days_since_qtr_filing`, `ne_is_earnings_window`, 7-day news count, positive/negative/net sentiment ratios, promoter bull/bear balance, and competitor read-across score. |
| `target_` | **Forward Targets (Separated)** | 7 | — | `target_next_day_ret`, `target_next_day_up`, `target_3d_ret`, `target_5d_ret`, `target_10d_ret`, `target_next_day_mae`, `target_next_day_mfe`. Strictly excluded from candidate and selected feature sets by [`backend/features/leakage_guard.py`](backend/features/leakage_guard.py). |

---

## 7. Quantitative Prediction Log & Walk-Forward Backtest Tables (`/data/predictions/` & `/data/backtests/`)

### A. Permanent Prediction Log (`/data/predictions/prediction_log.parquet` / `v_predictions`)

| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `prediction_id` | `VARCHAR` | ID | `{symbol}_{observation_date}_{model_name}_{model_version}` |
| `symbol`, `observation_date`, `generated_at` | `VARCHAR` | ISO-8601 | NSE ticker, session date, and UTC generation timestamp |
| `model_name`, `model_version` | `VARCHAR` | SemVer | `ml_ensemble` (`3.0.0`) |
| `market_regime` | `VARCHAR` | Enum | Active market regime (`Strong Bull`, `Weak Bull`, `Sideways`, `Bear`, `High Volatility`) |
| `prob_next_day_up` | `DOUBLE` | `[0, 1]` | Calibrated ensemble probability of positive next-day return |
| `prob_lgbm_up`, `prob_xgb_up`, `prob_linear_up` | `DOUBLE` | `[0, 1]` | Constituent sub-model probabilities (`LightGBM`, `XGBoost`, `Linear`) |
| `expected_next_day_return_pct`, `expected_5d_return_pct` | `DOUBLE` | `%` | Gross expected 1-day and 5-day return % |
| `downside_risk_mae_pct`, `expected_upside_mfe_pct` | `DOUBLE` | `%` | Expected next-day intraday Maximum Adverse Excursion (`MAE`) and Maximum Favorable Excursion (`MFE`) |
| `reward_to_risk_ratio` | `DOUBLE` | Ratio | `expected_upside_mfe_pct / abs(downside_risk_mae_pct)` |
| `model_agreement_score`, `confidence_score` | `DOUBLE` | `[0, 1]` / `[0, 100]` | Sub-model consensus score and calibrated confidence score |
| `roundtrip_delivery_cost_pct`, `roundtrip_intraday_cost_pct` | `DOUBLE` | `%` | Realistic Indian NSE round-trip transaction cost (`STT + Stamp + Exchange + SEBI + GST + DP + Slippage`) |
| `net_expected_1d_return_pct`, `net_expected_5d_return_pct` | `DOUBLE` | `%` | Expected return net of Indian round-trip transaction costs |
| `recommended_action`, `abstention_reason` | `VARCHAR` | Enum / Text | `STRONG BUY`, `BUY / ACCUMULATE`, `HOLD / WATCH`, `REDUCE / SELL`, `AVOID / SHORT`, or `NO TRADE` with explicit abstention rationale |
| `top_bullish_drivers`, `top_bearish_drivers` | `VARCHAR` (JSON) | SHAP | Top 5 bullish and top 5 bearish exact TreeSHAP feature contributions (`pred_contrib=True`) |

### B. Walk-Forward Out-of-Sample Predictions (`/data/backtests/walk_forward_predictions.parquet` / `v_backtest_predictions`)

Stores all out-of-sample walk-forward predictions across `ml_ensemble`, `linear_ridge_logistic`, `technical_rule`, and `naive_momentum`, with realized forward returns (`target_next_day_ret`, `target_5d_ret`), active position flag (`position`), gross return (`gross_trade_ret_pct`), Indian transaction cost drag (`cost_drag_pct`), and net return (`net_trade_ret_pct`).


