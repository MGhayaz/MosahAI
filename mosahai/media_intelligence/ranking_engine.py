"""Ranking engine for MosahAI media candidates."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from mosahai.media_intelligence.source_reputation import SourceReputationEngine


@dataclass(slots=True)
class MediaRankingEngine:
    weight_relevance: float = 0.45
    weight_recency: float = 0.25
    weight_quality: float = 0.20
    weight_source: float = 0.10
    max_results: int = 3
    model_name: str = "all-MiniLM-L6-v2"
    source_reputation_engine: SourceReputationEngine | None = None

    def __post_init__(self) -> None:
        total = (
            self.weight_relevance
            + self.weight_recency
            + self.weight_quality
            + self.weight_source
        )
        if total <= 0:
            self.weight_relevance = 0.45
            self.weight_recency = 0.25
            self.weight_quality = 0.20
            self.weight_source = 0.10
            total = 1.0
        self.weight_relevance /= total
        self.weight_recency /= total
        self.weight_quality /= total
        self.weight_source /= total
        self.max_results = max(1, int(self.max_results))
        self._model = None
        if self.source_reputation_engine is None:
            self.source_reputation_engine = SourceReputationEngine()

    def calculate_similarity(self, news_title: str, media_title: str) -> float:
        news_title = (news_title or "").strip()
        media_title = (media_title or "").strip()
        if not news_title or not media_title:
            return 0.0

        model = self._get_model()
        if model:
            try:
                embeddings = model.encode([news_title, media_title], normalize_embeddings=True)
                return float(embeddings[0] @ embeddings[1])
            except Exception:
                pass

        return _token_overlap_similarity(news_title, media_title)

    def rank_candidates(
        self,
        candidates: Sequence[Any],
        *,
        news_title: str | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        resolved_news_title = (news_title or "").strip()
        if not resolved_news_title:
            resolved_news_title = str(getattr(candidates[0], "news_title", "") or "").strip()
            if not resolved_news_title and isinstance(candidates[0], Mapping):
                resolved_news_title = str(candidates[0].get("news_title", "") or "").strip()

        if not resolved_news_title:
            resolved_news_title = ""

        now = now or datetime.now(timezone.utc)

        scored: list[dict[str, Any]] = []
        seen: set[str] = set()

        for candidate in candidates:
            normalized = _normalize_candidate(candidate)
            url = normalized.get("url") or ""
            canonical = _canonicalize_url(url)
            if canonical and canonical in seen:
                continue
            if canonical:
                seen.add(canonical)

            title = str(normalized.get("title") or "").strip()
            published_at = _coerce_datetime(normalized.get("published_at"))
            duration_seconds = _safe_int(normalized.get("duration_seconds"))
            width = _safe_int(normalized.get("width"))
            height = _safe_int(normalized.get("height"))
            channel_name = _extract_source_name(normalized)

            relevance_score = self.calculate_similarity(resolved_news_title, title)
            recency_score = _recency_score(published_at, now)
            quality_score = _quality_score(
                duration_seconds=duration_seconds,
                width=width,
                height=height,
            )
            source_score = 0.5
            if self.source_reputation_engine:
                source_score = self.source_reputation_engine.get_source_score(channel_name)

            final_score = (
                self.weight_relevance * relevance_score
                + self.weight_recency * recency_score
                + self.weight_quality * quality_score
                + self.weight_source * source_score
            )

            enriched = {
                **normalized,
                "relevance_score": round(float(relevance_score), 4),
                "recency_score": round(float(recency_score), 4),
                "quality_score": round(float(quality_score), 4),
                "source_score": round(float(source_score), 4),
                "final_score": round(float(final_score), 4),
            }
            scored.append(enriched)

        scored.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        return scored[: self.max_results]

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None
        return self._model


def _normalize_candidate(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        payload = dict(candidate)
    else:
        payload = {
            "source": getattr(candidate, "source", None),
            "url": getattr(candidate, "url", None),
            "title": getattr(candidate, "title", None),
            "published_at": getattr(candidate, "published_at", None),
            "duration_seconds": getattr(candidate, "duration_seconds", None),
            "raw": getattr(candidate, "raw", None),
        }

    raw = payload.get("raw") or {}
    if isinstance(raw, Mapping):
        if payload.get("width") is None:
            payload["width"] = raw.get("width") or raw.get("resolution", {}).get("width")
        if payload.get("height") is None:
            payload["height"] = raw.get("height") or raw.get("resolution", {}).get("height")

    if not payload.get("url"):
        payload["url"] = payload.get("media_url") or payload.get("tweet_url")

    if not payload.get("title"):
        payload["title"] = payload.get("text")

    return payload


def _extract_source_name(candidate: Mapping[str, Any]) -> str | None:
    for key in (
        "channel",
        "channel_name",
        "uploader",
        "uploader_id",
        "publisher",
        "author",
        "source_name",
    ):
        value = candidate.get(key)
        if value:
            return str(value)

    raw = candidate.get("raw")
    if isinstance(raw, Mapping):
        for key in ("uploader", "channel", "publisher", "author", "creator"):
            value = raw.get(key)
            if value:
                return str(value)

        tweet = raw.get("tweet")
        if tweet is not None:
            user = getattr(tweet, "user", None)
            if user is not None:
                username = getattr(user, "username", None)
                display_name = getattr(user, "displayname", None)
                if username:
                    return str(username)
                if display_name:
                    return str(display_name)

    return None


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens.intersection(right_tokens)
    return len(overlap) / max(len(left_tokens), len(right_tokens))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None
    return None


def _recency_score(published_at: datetime | None, now: datetime) -> float:
    if not published_at:
        return 0.2
    age_hours = (now - published_at).total_seconds() / 3600.0
    if age_hours <= 0:
        return 1.0
    if age_hours <= 48.0:
        return 1.0
    if age_hours <= 168.0:
        return max(0.0, 1.0 - (age_hours - 48.0) / 120.0)
    return 0.0


def _quality_score(
    *,
    duration_seconds: int | None,
    width: int | None,
    height: int | None,
) -> float:
    if width and height:
        pixels = width * height
        target = 1280 * 720
        return min(1.0, pixels / target)

    if duration_seconds is None:
        return 0.0

    duration_seconds = max(0, int(duration_seconds))
    if duration_seconds <= 0:
        return 0.0

    return min(1.0, duration_seconds / 240.0)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower() or parsed.path.split("/")[0].lower()
    path = parsed.path
    query = parse_qs(parsed.query)

    if "youtu" in netloc:
        if "v" in query:
            return f"youtube.com/watch?v={query['v'][0]}"
        if parsed.path.startswith("/embed/"):
            return f"youtube.com/watch?v={parsed.path.split('/embed/')[-1]}"
        if parsed.path and parsed.path != "/":
            return f"youtube.com/watch?v={parsed.path.strip('/')}"

    if "twitter.com" in netloc or "x.com" in netloc:
        match = re.search(r"/status/(\d+)", url)
        if match:
            return f"twitter.com/i/status/{match.group(1)}"

    return f"{netloc}{path}"
