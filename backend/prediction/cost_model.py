"""Indian Equity Market Transaction Cost & Slippage Model (Phase 3 — Section 8 & 10).

Computes realistic round-trip execution friction for NSE/BSE equities under two
standard Indian execution regimes:
1. **Cash Delivery (`CNC` — Multi-Day / Swing Hold)**:
   - STT: `0.10%` on Buy + `0.10%` on Sell (`20.0 bps` round-trip)
   - Stamp Duty: `0.015%` on Buy (`1.5 bps`)
   - NSE Exchange Transaction Charge: `0.00297%` per leg (`0.594 bps` round-trip)
   - SEBI Turnover Fee: `0.0001%` (`₹10 / Cr`) per leg (`0.02 bps` round-trip)
   - DP (Depository Participant) Demat Debit Charge: `0.015%` (`1.5 bps` amortized)
   - GST: `18%` on (Brokerage + NSE Exchange Charge + SEBI Fee)
   - Bid-Ask Spread & Market-Impact Slippage: `5.0 bps` base per leg (`10.0 bps` round-trip)
     plus dynamic volatility (`pm_atr_14_pct`) and Amihud illiquidity (`vd_amihud_illiquidity_20d`)
     penalty (`~34–48 bps` total round-trip).
2. **Intraday (`MIS` — Next-Day Open-to-Close / Single-Session)**:
   - Brokerage: `0.03%` per leg (`6.0 bps` round-trip)
   - STT: `0.025%` on Sell side only (`2.5 bps`)
   - Stamp Duty: `0.003%` on Buy side (`0.3 bps`)
   - NSE Exchange Transaction Charge: `0.00297%` per leg (`0.594 bps` round-trip)
   - SEBI Turnover Fee: `0.0001%` per leg (`0.02 bps` round-trip)
   - GST: `18%` on (Brokerage + NSE Exchange Charge + SEBI Fee) (`~1.19 bps`)
   - Bid-Ask Spread & Market-Impact Slippage: `4.0 bps` base per leg (`8.0 bps` round-trip)
     plus dynamic volatility/illiquidity penalty (`~19–28 bps` total round-trip).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndianCostBreakdown:
    """Detailed statutory & market-impact cost breakdown in percentage points (`1.0` = `1%`)."""

    mode: str
    brokerage_pct: float
    stt_pct: float
    exchange_txn_pct: float
    sebi_fee_pct: float
    stamp_duty_pct: float
    gst_pct: float
    dp_charge_pct: float
    slippage_pct: float
    total_roundtrip_pct: float

    @property
    def total_roundtrip_bps(self) -> float:
        return round(self.total_roundtrip_pct * 100.0, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "brokerage_pct": round(self.brokerage_pct, 5),
            "stt_pct": round(self.stt_pct, 5),
            "exchange_txn_pct": round(self.exchange_txn_pct, 5),
            "sebi_fee_pct": round(self.sebi_fee_pct, 5),
            "stamp_duty_pct": round(self.stamp_duty_pct, 5),
            "gst_pct": round(self.gst_pct, 5),
            "dp_charge_pct": round(self.dp_charge_pct, 5),
            "slippage_pct": round(self.slippage_pct, 5),
            "total_roundtrip_pct": round(self.total_roundtrip_pct, 5),
            "total_roundtrip_bps": self.total_roundtrip_bps,
        }


def compute_dynamic_slippage_pct(
    atr_14_pct: float | None = None,
    amihud_illiquidity: float | None = None,
    base_per_leg_bps: float = 5.0,
) -> float:
    """Compute round-trip bid-ask + market-impact slippage in percentage points.

    Liquid Nifty 50 large caps (`atr_14_pct ~ 1.4%`, `amihud ~ 0.0005`) incur ~`10.7 bps` (`0.107%`).
    Less liquid small/mid caps (`atr_14_pct ~ 3.2%`, `amihud ~ 0.08`) incur ~`16–22 bps` (`0.16–0.22%`).
    """
    atr_val = float(atr_14_pct) if (atr_14_pct is not None and not np.isnan(atr_14_pct) and atr_14_pct > 0) else 1.8
    amihud_val = (
        float(amihud_illiquidity)
        if (amihud_illiquidity is not None and not np.isnan(amihud_illiquidity) and amihud_illiquidity >= 0)
        else 0.005
    )

    vol_addon_bps = min(8.0, max(0.0, (atr_val - 1.0) * 1.5))
    illiq_addon_bps = min(10.0, amihud_val * 40.0)
    total_bps = (2.0 * base_per_leg_bps) + vol_addon_bps + illiq_addon_bps
    return round(total_bps / 100.0, 5)


def estimate_indian_roundtrip_cost(
    mode: str = "delivery",
    atr_14_pct: float | None = None,
    amihud_illiquidity: float | None = None,
    delivery_brokerage_per_leg_pct: float = 0.0,
) -> IndianCostBreakdown:
    """Calculate exact statutory + slippage round-trip cost in percentage points (`1.0` = `1%`)."""
    exchange_txn_pct = 2.0 * 0.00297  # 0.00594% (NSE Cash EQ)
    sebi_fee_pct = 2.0 * 0.0001       # 0.00020% (₹10 per Crore per leg)

    if mode.lower() in ("intraday", "mis"):
        brokerage_pct = 2.0 * 0.03    # 0.06000% (0.03% per leg)
        stt_pct = 0.025               # 0.02500% (Sell leg only)
        stamp_duty_pct = 0.003        # 0.00300% (Buy leg only)
        dp_charge_pct = 0.0
        gst_pct = 0.18 * (brokerage_pct + exchange_txn_pct + sebi_fee_pct)
        slippage_pct = compute_dynamic_slippage_pct(
            atr_14_pct=atr_14_pct,
            amihud_illiquidity=amihud_illiquidity,
            base_per_leg_bps=4.0,
        )
    else:
        brokerage_pct = 2.0 * delivery_brokerage_per_leg_pct
        stt_pct = 2.0 * 0.10          # 0.20000% (0.10% Buy + 0.10% Sell)
        stamp_duty_pct = 0.015        # 0.01500% (Buy leg)
        dp_charge_pct = 0.015         # ~1.5 bps amortized DP debit charge
        gst_pct = 0.18 * (brokerage_pct + exchange_txn_pct + sebi_fee_pct)
        slippage_pct = compute_dynamic_slippage_pct(
            atr_14_pct=atr_14_pct,
            amihud_illiquidity=amihud_illiquidity,
            base_per_leg_bps=5.0,
        )

    total_pct = (
        brokerage_pct
        + stt_pct
        + exchange_txn_pct
        + sebi_fee_pct
        + stamp_duty_pct
        + gst_pct
        + dp_charge_pct
        + slippage_pct
    )

    return IndianCostBreakdown(
        mode=mode.lower(),
        brokerage_pct=brokerage_pct,
        stt_pct=stt_pct,
        exchange_txn_pct=exchange_txn_pct,
        sebi_fee_pct=sebi_fee_pct,
        stamp_duty_pct=stamp_duty_pct,
        gst_pct=gst_pct,
        dp_charge_pct=dp_charge_pct,
        slippage_pct=slippage_pct,
        total_roundtrip_pct=total_pct,
    )


def compute_row_cost_series(
    df: pd.DataFrame,
    mode: str = "delivery",
) -> pd.Series:
    """Vectorized calculation of round-trip transaction cost (%) for every row in `df`."""
    atr_s = (
        pd.to_numeric(df["pm_atr_14_pct"], errors="coerce").fillna(1.8).clip(lower=0.5, upper=10.0)
        if "pm_atr_14_pct" in df.columns
        else pd.Series(1.8, index=df.index)
    )
    amihud_s = (
        pd.to_numeric(df["vd_amihud_illiquidity_20d"], errors="coerce").fillna(0.005).clip(lower=0.0, upper=1.0)
        if "vd_amihud_illiquidity_20d" in df.columns
        else pd.Series(0.005, index=df.index)
    )

    vol_addon_bps = ((atr_s - 1.0) * 1.5).clip(lower=0.0, upper=8.0)
    illiq_addon_bps = (amihud_s * 40.0).clip(lower=0.0, upper=10.0)

    if mode.lower() in ("intraday", "mis"):
        statutory_pct = 0.06 + 0.025 + 0.00594 + 0.0002 + 0.003 + 0.18 * (0.06 + 0.00594 + 0.0002)
        slippage_pct = (8.0 + vol_addon_bps + illiq_addon_bps) / 100.0
    else:
        statutory_pct = 0.20 + 0.015 + 0.00594 + 0.0002 + 0.015 + 0.18 * (0.00594 + 0.0002)
        slippage_pct = (10.0 + vol_addon_bps + illiq_addon_bps) / 100.0

    return (statutory_pct + slippage_pct).round(5)
