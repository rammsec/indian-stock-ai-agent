"""Simulate Live Candle-by-Candle Streaming for TCS.

Demonstrates strictly causal, lookahead-free live execution:
- At session open (09:15 IST), the agent only has data up to the prior trading day.
- During the session, candles arrive ONE BY ONE as they close.
- The Candle Timing Agent has ZERO access to future bars.
- Starting capital is 50,000 INR, managed by DeterministicRiskEngine.
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd

from trader.config import IST_TZ
from trader.data.market_feed import ParquetCSVMarketFeed
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine
from trader.timing.candle_timing_agent import CandleTimingAgent, TimingDecision
from trader.timing.central_directional_agent import DirectionalBias, DirectionalHypothesis


def run_live_simulation(session_date_str: str = "2026-11-09", initial_capital: float = 50_000.0) -> None:
    print("=" * 80)
    print(f"  LIVE BAR-BY-BAR SIMULATION FOR TCS ON {session_date_str}")
    print(f"  Starting Capital: Rs {initial_capital:,.2f} | Target: Increase Money Intraday")
    print("  Zero lookahead: Candle Agent only sees bars closed up to that exact timestamp.")
    print("=" * 80)

    # 1. Initialize Portfolio, Risk Engine, and Timing Agent
    portfolio = IntradayPaperPortfolio(initial_capital=initial_capital)
    risk_engine = DeterministicRiskEngine()
    timing_agent = CandleTimingAgent()
    feed = ParquetCSVMarketFeed()

    # 2. Prior Day Intelligence / Hypothesis (as of prior trading day close)
    # The agent enters the morning with a hypothesis formed strictly before 09:15 market open
    hypothesis = DirectionalHypothesis(
        symbol="TCS",
        direction=DirectionalBias.BULLISH,
        confidence=0.70,
        prob_next_day_up=0.62,
        expected_next_day_return_pct=1.2,
        news_sentiment_score=0.45,
        market_regime="BULLISH_TREND",
        observation_date=pd.to_datetime("2026-11-06").date(),
        as_of_ist=datetime(2026, 11, 6, 15, 30, tzinfo=IST_TZ),
        rationale=("Strong technical momentum", "Favorable regime alignment"),
    )
    print(f"\n[Prior Day Context] Baseline Hypothesis: {hypothesis.direction.value} "
          f"(Confidence: {hypothesis.confidence:.0%}, Regime: {hypothesis.market_regime})")

    # 3. Load full historical parquet to extract the day's bar timestamps
    df_raw = pd.read_parquet("data/intraday/TCS_1h.parquet")
    df_raw["dt"] = pd.to_datetime(df_raw["timestamp"]).dt.tz_convert(IST_TZ)
    target_date = pd.to_datetime(session_date_str).date()
    day_bars = df_raw[df_raw["dt"].dt.date == target_date].sort_values("dt").reset_index(drop=True)

    if day_bars.empty:
        print(f"No trading bars found for {session_date_str} (market may have been closed).")
        return

    print(f"\n[Intraday Stream] Total scheduled 1h candles today: {len(day_bars)}")
    print("-" * 80)

    # 4. Stream candles one by one (simulating live market clock)
    active_position_qty = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    target_price = 0.0

    for idx, row in day_bars.iterrows():
        bar_open_ts = row["dt"]
        bar_close_ts = bar_open_ts + pd.Timedelta(hours=1)

        # Causal fetch: Feed strictly filters out any bar closing AFTER bar_close_ts
        evidence = feed.get_bars("TCS", "1h", as_of_ist=bar_close_ts, lookback_bars=50)
        bars_df = evidence.frame.copy()
        bars_df["timestamp"] = bars_df["ts_open_ist"]
        agent_bars = bars_df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

        current_close = float(row["close"])
        current_time_str = bar_close_ts.strftime("%H:%M IST")

        # Candle Timing Agent evaluates independently
        assessment = timing_agent.evaluate(
            hypothesis=hypothesis,
            bars_primary=agent_bars,
            primary_timeframe="1h",
            as_of_ist=bar_close_ts,
        )

        decision = assessment.decision.value
        bias = assessment.candle_bias.value
        quality = assessment.timing_quality_score

        print(f"[{current_time_str}] Bar {idx + 1}: O={row['open']:.2f} H={row['high']:.2f} "
              f"L={row['low']:.2f} C={current_close:.2f} | Vol={int(row['volume']):,}")
        print(f"   Candle Agent: Bias={bias} | Quality={quality:.2f} | Decision={decision} "
              f"({assessment.alignment_status.value})")

        # Position management
        if active_position_qty > 0:
            portfolio.update_market_price("TCS", current_close, bar_close_ts)
            unrealized_pnl = (current_close - entry_price) * active_position_qty
            print(f"   Holding {active_position_qty} TCS @ avg {entry_price:.2f} | "
                  f"Cur={current_close:.2f} | MTM PnL=Rs {unrealized_pnl:+,.2f}")

            # Check Stop Loss / Target or 15:05 wind-down
            if current_close <= stop_loss_price:
                print(f"   >>> STOP-LOSS TRIGGERED at Rs {current_close:.2f}! Exiting position.")
                portfolio.submit_order(symbol="TCS", side="SELL", quantity=active_position_qty,
                                       price=current_close, timestamp=bar_close_ts)
                active_position_qty = 0
            elif current_close >= target_price:
                print(f"   >>> PROFIT TARGET REACHED at Rs {current_close:.2f}! Banking profit.")
                portfolio.submit_order(symbol="TCS", side="SELL", quantity=active_position_qty,
                                       price=current_close, timestamp=bar_close_ts)
                active_position_qty = 0
            elif bar_close_ts.time() >= pd.to_datetime("15:05").time():
                print(f"   >>> 15:05 WIND-DOWN TRIGGERED. Closing intraday position at market.")
                portfolio.submit_order(symbol="TCS", side="SELL", quantity=active_position_qty,
                                       price=current_close, timestamp=bar_close_ts)
                active_position_qty = 0

        # New entry evaluation
        elif decision == TimingDecision.ENTER_NOW.value:
            # Calculate position sizing with 50,000 INR capital
            max_risk_inr = initial_capital * 0.02  # 2% max risk per trade
            sl = assessment.invalidation_price or round(current_close * 0.985, 2)
            tp = assessment.target_price or round(current_close * 1.03, 2)
            risk_per_share = max(current_close - sl, current_close * 0.01)

            # Max shares we can buy with available cash
            affordable_qty = int(portfolio.cash // current_close)
            risk_qty = int(max_risk_inr // risk_per_share)
            target_qty = max(1, min(affordable_qty, risk_qty if risk_qty > 0 else affordable_qty))

            risk_eval = risk_engine.evaluate_order(
                symbol="TCS",
                side="BUY",
                requested_quantity=target_qty,
                entry_price=current_close,
                stop_loss_price=sl,
                portfolio=portfolio,
                current_time=bar_close_ts,
            )

            if risk_eval.approved and risk_eval.approved_quantity > 0:
                active_position_qty = risk_eval.approved_quantity
                entry_price = current_close
                stop_loss_price = sl
                target_price = tp
                portfolio.submit_order(symbol="TCS", side="BUY", quantity=active_position_qty,
                                       price=entry_price, timestamp=bar_close_ts)
                print(f"   >>> BUY ORDER EXECUTED: {active_position_qty} shares @ Rs {entry_price:.2f} "
                      f"(Total: Rs {active_position_qty * entry_price:,.2f})")
                print(f"       Stop Loss: Rs {stop_loss_price:.2f} | Target: Rs {target_price:.2f}")
            else:
                print(f"   >>> Risk Engine rejected entry: {risk_eval.rejection_reasons}")

    print("\n" + "=" * 80)
    print("  SESSION SUMMARY REPORT")
    print(f"  Ending Cash:        Rs {portfolio.cash:,.2f}")
    print(f"  Ending Portfolio:   Rs {portfolio.total_equity:,.2f}")
    print(f"  Net Realized PnL:   Rs {portfolio.realized_pnl:+,.2f}")
    print(f"  Orders Submitted:   {len(portfolio.orders)}")
    print("=" * 80)


if __name__ == "__main__":
    run_live_simulation()
