"""Media source reputation scoring for MosahAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class SourceReputationEngine:
    trusted_sources: set[str] = field(
        default_factory=lambda: {
            "reuters",
            "bloomberg",
            "cnbc",
            "bbc",
            "associated press",
            "ap",
            "the wall street journal",
            "wsj",
            "financial times",
            "ft",
            "the economist",
            "new york times",
            "nytimes",
            "washington post",
            "the guardian",
            "cnn",
            "fox news",
            "sky news",
            "al jazeera",
            "nhk",
        }
    )
    spam_indicators: set[str] = field(
        default_factory=lambda: {
            "giveaway",
            "free money",
            "crypto pump",
            "airdrops",
            "earn fast",
            "get rich",
            "loan",
            "forex",
            "bet",
            "casino",
            "slots",
            "promo",
            "subscribe",
            "discount",
        }
    )
    official_indicators: set[str] = field(
        default_factory=lambda: {
            "official",
            "gov",
            "government",
            "ministry",
            "department",
            "press office",
            "press bureau",
        }
    )

    def get_source_score(self, channel_name: str | None) -> float:
        name = (channel_name or "").strip().lower()
        if not name:
            return 0.5

        if _contains_any(name, self.spam_indicators):
            return 0.2

        if _contains_any(name, self.official_indicators):
            return 1.0

        if _contains_any(name, self.trusted_sources):
            return 1.0

        return 0.5


def _contains_any(text: str, items: Iterable[str]) -> bool:
    for item in items:
        if item and item in text:
            return True
    return False
