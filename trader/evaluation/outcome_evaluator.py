"""CYCLE-010 Phase 3 — Strictly Post-Session Counterfactual Outcome Evaluator.

Runs ONLY after an intraday monitoring session has completed (Phase 2 review).
Evaluates 100% of bar-by-bar decisions (OPEN_LONG, OPEN_SHORT, WAIT, HOLD,
EXIT_LONG, COVER_SHORT, NO_TRADE, CANCEL_SETUP, REJECT) against the realized
subsequent price path within the session — without ever leaking future bars
into the agent's causal decision loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from trader.portfolio.cost_calculator import calculate_intraday_costs


@dataclass(frozen=True)
class DecisionOutcomeRecord:
    """Post-session realized & counterfactual outcome for a single bar-by-bar decision."""

    bar_index: int
    timestamp_ist: str
    symbol: str
    call: str
    intended_direction: str  # LONG, SHORT, NEUTRAL
    reference_price: float
    forward_bars_count: int
    forward_mfe_pct: float
    forward_mae_pct: float
    forward_close_return_pct: float
    missed_move_pct: float
    saved_loss_pct: float
    saved_loss_inr: float
    left_on_table_pct: float
    counterfactual_net_pnl_inr: float | None
    counterfactual_exit_reason: str | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar_index": self.bar_index,
            "timestamp_ist": self.timestamp_ist,
            "symbol": self.symbol,
            "call": self.call,
            "intended_direction": self.intended_direction,
            "reference_price": round(self.reference_price, 2),
            "forward_bars_count": self.forward_bars_count,
            "forward_mfe_pct": round(self.forward_mfe_pct, 3),
            "forward_mae_pct": round(self.forward_mae_pct, 3),
            "forward_close_return_pct": round(self.forward_close_return_pct, 3),
            "missed_move_pct": round(self.missed_move_pct, 3),
            "saved_loss_pct": round(self.saved_loss_pct, 3),
            "saved_loss_inr": round(self.saved_loss_inr, 2),
            "left_on_table_pct": round(self.left_on_table_pct, 3),
            "counterfactual_net_pnl_inr": (
                round(self.counterfactual_net_pnl_inr, 2)
                if self.counterfactual_net_pnl_inr is not None
                else None
            ),
            "counterfactual_exit_reason": self.counterfactual_exit_reason,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class SessionOutcomeReport:
    """Aggregate post-session outcome summary across all bar-by-bar decisions."""

    symbol: str
    total_decisions: int
    entries_evaluated: int
    exits_evaluated: int
    abstentions_and_rejects_evaluated: int
    saved_loss_count: int
    missed_opportunity_count: int
    total_saved_loss_inr: float
    avg_entry_mfe_pct: float
    avg_entry_mae_pct: float
    per_decision_outcomes: tuple[DecisionOutcomeRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_decisions": self.total_decisions,
            "entries_evaluated": self.entries_evaluated,
            "exits_evaluated": self.exits_evaluated,
            "abstentions_and_rejects_evaluated": self.abstentions_and_rejects_evaluated,
            "saved_loss_count": self.saved_loss_count,
            "missed_opportunity_count": self.missed_opportunity_count,
            "total_saved_loss_inr": round(self.total_saved_loss_inr, 2),
            "avg_entry_mfe_pct": round(self.avg_entry_mfe_pct, 3),
            "avg_entry_mae_pct": round(self.avg_entry_mae_pct, 3),
            "per_decision_outcomes": [r.to_dict() for r in self.per_decision_outcomes],
        }


def _simulate_counterfactual_trade(
    *,
    direction: str,
    entry_price: float,
    stop_loss: float | None,
    target: float | None,
    quantity: int,
    future_bars: pd.DataFrame,
) -> tuple[float, str]:
    """Deterministically simulate a counterfactual intraday trade over subsequent session bars."""
    if future_bars.empty or entry_price <= 0 or quantity <= 0:
        return 0.0, "NO_FUTURE_BARS"

    exit_px = float(future_bars["close"].iloc[-1])
    exit_reason = "SESSION_CLOSE"

    for _, row in future_bars.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        if direction == "SHORT":
            if stop_loss is not None and o >= stop_loss:
                exit_px, exit_reason = o, "GAP_STOP_LOSS"
                break
            if target is not None and o <= target:
                exit_px, exit_reason = o, "GAP_TARGET"
                break
            if stop_loss is not None and h >= stop_loss:
                exit_px, exit_reason = stop_loss, "STOP_LOSS_HIT"
                break
            if target is not None and l <= target:
                exit_px, exit_reason = target, "TARGET_HIT"
                break
        else:
            if stop_loss is not None and o <= stop_loss:
                exit_px, exit_reason = o, "GAP_STOP_LOSS"
                break
            if target is not None and o >= target:
                exit_px, exit_reason = o, "GAP_TARGET"
                break
            if stop_loss is not None and l <= stop_loss:
                exit_px, exit_reason = stop_loss, "STOP_LOSS_HIT"
                break
            if target is not None and h >= target:
                exit_px, exit_reason = target, "TARGET_HIT"
                break

    if direction == "SHORT":
        gross_pnl = (entry_price - exit_px) * quantity
        entry_cost = calculate_intraday_costs(price=entry_price, quantity=quantity, side="SELL").total_friction_inr
        exit_cost = calculate_intraday_costs(price=exit_px, quantity=quantity, side="BUY").total_friction_inr
    else:
        gross_pnl = (exit_px - entry_price) * quantity
        entry_cost = calculate_intraday_costs(price=entry_price, quantity=quantity, side="BUY").total_friction_inr
        exit_cost = calculate_intraday_costs(price=exit_px, quantity=quantity, side="SELL").total_friction_inr

    return round(gross_pnl - entry_cost - exit_cost, 2), exit_reason


def evaluate_session_outcomes(
    *,
    symbol: str,
    decision_records: Sequence[dict[str, Any]],
    session_bars: pd.DataFrame,
    default_eval_quantity: int = 10,
    missed_move_threshold_pct: float = 0.40,
) -> SessionOutcomeReport:
    """Evaluate every bar-by-bar decision against subsequent realized session bars (Phase 2 Post-Session Review).

    Each item in `decision_records` must provide:
    - `bar_index` (0-based index into `session_bars`)
    - `timestamp_ist` (ISO string)
    - `call` (one of the 9 canonical AgentCall strings)
    - `intended_direction` ("LONG", "SHORT", or "NEUTRAL")
    - `reference_price` (float)
    - Optional: `stop_loss`, `target`, `quantity`
    """
    outcomes: list[DecisionOutcomeRecord] = []
    entry_mfes: list[float] = []
    entry_maes: list[float] = []
    entries_count = 0
    exits_count = 0
    abstentions_count = 0
    saved_loss_count = 0
    missed_opp_count = 0
    total_saved_loss_inr = 0.0

    n_bars = len(session_bars)

    for rec in decision_records:
        idx = int(rec["bar_index"])
        call = str(rec["call"]).upper()
        intended_dir = str(rec.get("intended_direction", "NEUTRAL")).upper()
        ref_px = float(rec["reference_price"])
        sl = rec.get("stop_loss")
        tp = rec.get("target")
        qty = max(1, int(rec.get("quantity") or default_eval_quantity))

        future = session_bars.iloc[idx + 1 :] if idx + 1 < n_bars else session_bars.iloc[0:0]
        fwd_count = len(future)

        if fwd_count == 0 or ref_px <= 0:
            outcomes.append(
                DecisionOutcomeRecord(
                    bar_index=idx,
                    timestamp_ist=str(rec["timestamp_ist"]),
                    symbol=symbol,
                    call=call,
                    intended_direction=intended_dir,
                    reference_price=ref_px,
                    forward_bars_count=0,
                    forward_mfe_pct=0.0,
                    forward_mae_pct=0.0,
                    forward_close_return_pct=0.0,
                    missed_move_pct=0.0,
                    saved_loss_pct=0.0,
                    saved_loss_inr=0.0,
                    left_on_table_pct=0.0,
                    counterfactual_net_pnl_inr=None,
                    counterfactual_exit_reason="END_OF_SESSION",
                    verdict="END_OF_SESSION",
                )
            )
            continue

        max_h = float(future["high"].max())
        min_l = float(future["low"].min())
        eod_c = float(future["close"].iloc[-1])

        # Directional MFE / MAE / EOD return
        eval_dir = intended_dir
        if call in ("OPEN_LONG", "EXIT_LONG"):
            eval_dir = "LONG"
        elif call in ("OPEN_SHORT", "COVER_SHORT"):
            eval_dir = "SHORT"

        if eval_dir == "SHORT":
            mfe_pct = max(0.0, ((ref_px - min_l) / ref_px) * 100.0)
            mae_pct = min(0.0, ((ref_px - max_h) / ref_px) * 100.0)
            eod_ret_pct = ((ref_px - eod_c) / ref_px) * 100.0
        else:
            mfe_pct = max(0.0, ((max_h - ref_px) / ref_px) * 100.0)
            mae_pct = min(0.0, ((min_l - ref_px) / ref_px) * 100.0)
            eod_ret_pct = ((eod_c - ref_px) / ref_px) * 100.0

        missed_move_pct = 0.0
        saved_loss_pct = 0.0
        saved_loss_inr = 0.0
        left_on_table_pct = 0.0
        cf_pnl: float | None = None
        cf_reason: str | None = None
        verdict = "NEUTRAL"

        if call in ("OPEN_LONG", "OPEN_SHORT"):
            entries_count += 1
            entry_mfes.append(mfe_pct)
            entry_maes.append(mae_pct)
            cf_pnl, cf_reason = _simulate_counterfactual_trade(
                direction=eval_dir,
                entry_price=float(future["open"].iloc[0]),
                stop_loss=sl,
                target=tp,
                quantity=qty,
                future_bars=future,
            )
            verdict = "OPTIMAL_ENTRY" if cf_pnl > 0 else "ADVERSE_ENTRY"

        elif call in ("EXIT_LONG", "COVER_SHORT"):
            exits_count += 1
            # For an exit, adverse continuation after exit is SAVED LOSS; favorable continuation is LEFT ON TABLE
            saved_loss_pct = abs(mae_pct)
            saved_loss_inr = round((saved_loss_pct / 100.0) * ref_px * qty, 2)
            left_on_table_pct = mfe_pct
            if eod_ret_pct < -0.10 or saved_loss_pct > left_on_table_pct:
                verdict = "PROTECTIVE_EXIT_SAVED_LOSS"
                saved_loss_count += 1
                total_saved_loss_inr += saved_loss_inr
            elif left_on_table_pct >= missed_move_threshold_pct:
                verdict = "PREMATURE_EXIT"
            else:
                verdict = "NEUTRAL_EXIT"

        elif call in ("WAIT", "NO_TRADE", "CANCEL_SETUP", "REJECT"):
            abstentions_count += 1
            if eval_dir in ("LONG", "SHORT"):
                cf_pnl, cf_reason = _simulate_counterfactual_trade(
                    direction=eval_dir,
                    entry_price=float(future["open"].iloc[0]),
                    stop_loss=sl,
                    target=tp,
                    quantity=qty,
                    future_bars=future,
                )
                if cf_pnl < 0:
                    saved_loss_pct = abs(mae_pct)
                    saved_loss_inr = round(abs(cf_pnl), 2)
                    saved_loss_count += 1
                    total_saved_loss_inr += saved_loss_inr
                    verdict = "RISK_REJECT_SAVED_LOSS" if call == "REJECT" else "CORRECT_WAIT_SAVED_LOSS"
                elif mfe_pct >= missed_move_threshold_pct and cf_pnl > 0:
                    missed_move_pct = mfe_pct
                    missed_opp_count += 1
                    verdict = "RISK_REJECT_FOREGONE_GAIN" if call == "REJECT" else "MISSED_OPPORTUNITY"
                else:
                    verdict = "NEUTRAL_ABSTENTION"
            else:
                verdict = "NEUTRAL_ABSTENTION"

        elif call == "HOLD":
            verdict = "HOLD_FAVORABLE" if eod_ret_pct >= 0 else "HOLD_ADVERSE"

        outcomes.append(
            DecisionOutcomeRecord(
                bar_index=idx,
                timestamp_ist=str(rec["timestamp_ist"]),
                symbol=symbol,
                call=call,
                intended_direction=eval_dir,
                reference_price=ref_px,
                forward_bars_count=fwd_count,
                forward_mfe_pct=mfe_pct,
                forward_mae_pct=mae_pct,
                forward_close_return_pct=eod_ret_pct,
                missed_move_pct=missed_move_pct,
                saved_loss_pct=saved_loss_pct,
                saved_loss_inr=saved_loss_inr,
                left_on_table_pct=left_on_table_pct,
                counterfactual_net_pnl_inr=cf_pnl,
                counterfactual_exit_reason=cf_reason,
                verdict=verdict,
            )
        )

    avg_mfe = (sum(entry_mfes) / len(entry_mfes)) if entry_mfes else 0.0
    avg_mae = (sum(entry_maes) / len(entry_maes)) if entry_maes else 0.0

    return SessionOutcomeReport(
        symbol=symbol,
        total_decisions=len(outcomes),
        entries_evaluated=entries_count,
        exits_evaluated=exits_count,
        abstentions_and_rejects_evaluated=abstentions_count,
        saved_loss_count=saved_loss_count,
        missed_opportunity_count=missed_opp_count,
        total_saved_loss_inr=round(total_saved_loss_inr, 2),
        avg_entry_mfe_pct=avg_mfe,
        avg_entry_mae_pct=avg_mae,
        per_decision_outcomes=tuple(outcomes),
    )
