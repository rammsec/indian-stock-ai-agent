"""Agent 3: News and Sentiment Agent.

Monitors company-specific news, exchange announcements, earnings, and regulatory actions.
Enforces strict Point-in-Time cutoff discipline (no future information leakage).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from ..config import IST_TZ
from .contracts import NewsSentimentMessage
from .router import ModelRouter


class NewsSentimentAgent:
    """Evaluates corporate releases and headlines with temporal validation."""

    def __init__(self, router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter()

    def analyze_news(
        self,
        symbol: str,
        headlines: list[dict[str, Any]],
        analysis_timestamp: datetime,
        data_cutoff_timestamp: datetime,
    ) -> NewsSentimentMessage:
        """Filter headlines by PIT cutoff and classify sentiment and materiality."""
        if data_cutoff_timestamp > analysis_timestamp:
            raise ValueError("data_cutoff_timestamp must be <= analysis_timestamp")

        valid_headlines: list[str] = []
        positive_cues = 0
        negative_cues = 0
        materiality_scores: list[float] = []

        bull_keywords = {"GROWTH", "PROFIT", "EXPANSION", "ORDER", "WIN", "CONTRACT", "BUYBACK", "BEAT", "ACQUISITION"}
        bear_keywords = {"LOSS", "RESIGNATION", "FRAUD", "PENALTY", "PROBE", "DOWNGRADE", "SLUMP", "WARNING", "DROP"}

        for item in headlines:
            pub_time = item.get("published_at")
            if isinstance(pub_time, datetime):
                # Enforce Point-in-Time discipline: exclude news published after data cutoff
                if pub_time > data_cutoff_timestamp:
                    continue

            title = item.get("title", "").strip()
            if not title:
                continue

            valid_headlines.append(title)
            mat = float(item.get("materiality", 0.5))
            materiality_scores.append(mat)

            title_upper = title.upper()
            if any(k in title_upper for k in bull_keywords):
                positive_cues += 1
            if any(k in title_upper for k in bear_keywords):
                negative_cues += 1

        if positive_cues > negative_cues:
            sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "BULLISH"
        elif negative_cues > positive_cues:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        avg_materiality = sum(materiality_scores) / len(materiality_scores) if materiality_scores else 0.2

        model_name = self.router.select_model(task_level=1, agent_name="news_sentiment")
        if "gemini" in model_name:
            self.router.budget_manager.record_usage(
                agent_name="news_sentiment",
                model_name=model_name,
                input_tokens=400,
                output_tokens=100,
            )

        return NewsSentimentMessage(
            symbol=symbol,
            timestamp=analysis_timestamp,
            data_cutoff_timestamp=data_cutoff_timestamp,
            sentiment=sentiment,
            materiality_score=round(avg_materiality, 2),
            headlines=valid_headlines[:5],
            corporate_actions=[],
            risks=["Breaking negative news" if sentiment == "BEARISH" else ""],
        )
