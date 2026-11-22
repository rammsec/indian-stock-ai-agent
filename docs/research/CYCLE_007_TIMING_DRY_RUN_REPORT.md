# CYCLE-007 — Directional Prediction & Intelligent Entry Timing (Numerical Phase) Dry-Run Report

- **Cycle ID**: `CYCLE-007`
- **Phase**: `DIRECTIONAL_PREDICTION_AND_INTELLIGENT_ENTRY_TIMING_NUMERICAL`
- **Run Status**: **`COMPLETED`**
- **Session Date**: `2026-11-19` (as-of `2026-11-19T09:15:00+05:30`)
- **Market Regime**: `Weak Bull`
- **Code Commit SHA**: `c8aeb62ff6d3750723ea04d1592365a14fc7a6e2`
- **Completed At (IST)**: `2026-11-22T14:31:36.910138+05:30`
- **Output Report SHA-256**: `8ce5a7e31d8e87cced6e3cfbd6b9b0aaa918ed7f9c0a8339770ede1746789a1f`

> **Governance & Scope Notice**: Numerical-only historical paper-trading dry run.
> `qualification_allowed=false`, `unseen_evaluation_status=NOT_EVALUATED`,
> `live_trading_enabled=false`, `visual_ai_enabled=false`, `synthetic_candles_used=0`,
> `external_api_calls=0`, `paid_api_cost=0.0`.

## 1. Per-Symbol Stage 1 (Central Directional) + Stage 2 (Candle Timing) + Risk Decisions

| Symbol | Stage 1 Direction | Prob Up | Exp Ret % | Stage 2 Candle Bias | PA Score | Timing Quality | Patterns | Stage 2 Decision | Alignment | Risk Gate |
|---|---|---:|---:|---|---:|---:|---|---|---|---|
| `RELIANCE` | `NEUTRAL` | 0.5049 | +0.0226% | `NEUTRAL` | +0.0800 | 0.0400 | `BEARISH_MARUBOZU, INSIDE_BAR` | **`NO_TRADE`** | `NEUTRAL_CENTRAL` | `TIMING_NO_TRADE` |
| `TCS` | `NEUTRAL` | 0.4618 | -0.1679% | `BEARISH` | -0.6500 | 0.3250 | `DOJI` | **`NO_TRADE`** | `NEUTRAL_CENTRAL` | `TIMING_NO_TRADE` |
| `BIRLACABLE` | `NEUTRAL` | 0.4681 | -0.4444% | `BULLISH` | +0.4000 | 0.2000 | `INSIDE_BAR` | **`NO_TRADE`** | `NEUTRAL_CENTRAL` | `TIMING_NO_TRADE` |
| `ITC` | `NEUTRAL` | 0.4996 | +0.0183% | `NEUTRAL` | -0.0200 | 0.0100 | `BEARISH_MARUBOZU, INSIDE_BAR` | **`NO_TRADE`** | `NEUTRAL_CENTRAL` | `TIMING_NO_TRADE` |
| `TATAELXSI` | `NEUTRAL` | 0.4681 | -0.1595% | `NEUTRAL` | +0.0200 | 0.0100 | `BULLISH_MARUBOZU, INSIDE_BAR` | **`NO_TRADE`** | `NEUTRAL_CENTRAL` | `TIMING_NO_TRADE` |

## 2. Timeframe Provenance (Honest Classification)

| Timeframe | Classification | Notes |
|---|---|---|
| **1h** | `PARTIALLY_VERIFIED` | Committed `data/intraday/*_1h.parquet` (Yahoo Finance; never upgraded to VERIFIED) |
| **15m** | `UNAVAILABLE` | Not committed in repo; never fabricated |
| **5m** | `UNAVAILABLE` | Not committed in repo; never fabricated |

## 3. Paper Portfolio & Ledger Reconciliation

- **Initial Capital**: `₹50,000.00`
- **Ending Cash / Equity**: `₹50,000.00` / `₹50,000.00`
- **Net P&L**: `₹0.00` (`0.0000%`)
- **Transaction Friction**: `₹0.00` (brokerage `₹0.00`, statutory `₹0.00`, slippage `₹0.00`)
- **Orders Created / Filled**: `0` / `0`
- **Open / Closed Positions**: `0` / `0`
- **Ledger Events**: `2` | **SHA-256 Chain Valid**: `True` | **Equity Reconciles**: `True` (`diff=0.0`)

## 4. Safety, Isolation & Holdout Integrity Attestation

- `qualification_allowed`: `false`
- `unseen_evaluation_status`: `NOT_EVALUATED`
- `live_trading_enabled`: `false`
- `visual_ai_enabled`: `false`
- `synthetic_candles_used`: `0`
- `external_api_calls`: `0` | `broker_connections`: `0` | `paid_api_cost`: `0.0`
- `holdout_registry_unchanged`: `true` (`before=5fc9013346b5e4209576b98f7f035f8179f0f7c10173acf289badd3faa60e352`, `after=5fc9013346b5e4209576b98f7f035f8179f0f7c10173acf289badd3faa60e352`)
