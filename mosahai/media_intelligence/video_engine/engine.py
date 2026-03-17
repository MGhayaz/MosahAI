from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from mosahai.logger import setup_logger
from mosahai.media_intelligence.logger import MediaEngineLogger
from mosahai.media_intelligence.media_quality import MediaQualityAnalyzer
from mosahai.media_intelligence.query_builder import QueryBuilder
from mosahai.media_intelligence.dedup_engine import VideoDeduplicationEngine
from mosahai.media_intelligence.ranking_engine import MediaRankingEngine
from mosahai.media_intelligence.relevance_filter import NewsMediaRelevanceFilter


LOGGER = setup_logger("mosahai.video_engine", level=logging.INFO)


@dataclass(slots=True)
class VideoQueryContext:
    batch_id: str
    news_id: str
    news_title: str
    keywords: list[str]
    entities: list[str]
    summary: str | None
    now: datetime
    article_urls: list[str] | None = None


@dataclass(slots=True)
class VideoCandidate:
    source: str
    url: str
    score: float
    title: str | None = None
    published_at: datetime | None = None
    duration_seconds: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, include_details: bool = False) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "url": self.url,
            "score": round(float(self.score), 4),
        }
        if include_details:
            payload.update(
                {
                    "title": self.title,
                    "published_at": self.published_at.isoformat() if self.published_at else None,
                    "duration_seconds": self.duration_seconds,
                }
            )
        return payload


class VideoSourceAgent(ABC):
    source_name = "base"

    def __init__(
        self,
        *,
        max_age_hours: float = 72.0,
        prefer_age_hours: float = 48.0,
        request_timeout_seconds: int = 12,
    ) -> None:
        self.max_age_hours = max(1.0, float(max_age_hours))
        self.prefer_age_hours = max(1.0, float(prefer_age_hours))
        self.request_timeout_seconds = max(1, int(request_timeout_seconds))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MosahAI/4.0)"})

    @abstractmethod
    def search(self, query: str, *, max_results: int | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def filter_results(
        self, results: Sequence[Mapping[str, Any]], context: VideoQueryContext
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def extract_media_url(self, result: Mapping[str, Any]) -> str | None:
        raise NotImplementedError

    def rank_results(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        context: VideoQueryContext,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        return [dict(item) for item in results]

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        try:
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None

    def _filter_by_recency(
        self, results: Sequence[Mapping[str, Any]], *, now: datetime
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for result in results:
            published_at = result.get("published_at")
            if isinstance(published_at, str):
                parsed = self._parse_datetime(published_at)
                if parsed:
                    published_at = parsed
                    result = dict(result)
                    result["published_at"] = parsed
            if isinstance(published_at, datetime):
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                    result = dict(result)
                    result["published_at"] = published_at
                age_hours = (now - published_at).total_seconds() / 3600.0
                if age_hours > self.max_age_hours:
                    continue
            filtered.append(dict(result))
        return filtered

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme:
            return "https://" + url.lstrip("/")
        return url

    def _dedupe_urls(self, results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for result in results:
            url = str(result.get("url") or "").strip()
            if not url:
                deduped.append(dict(result))
                continue
            normalized = self._normalize_url(url)
            if normalized in seen:
                continue
            seen.add(normalized)
            item = dict(result)
            item["url"] = normalized
            deduped.append(item)
        return deduped


class YouTubeVideoAgent(VideoSourceAgent):
    source_name = "youtube"

    def __init__(
        self,
        *,
        max_age_hours: float = 96.0,
        prefer_age_hours: float = 48.0,
        request_timeout_seconds: int = 12,
        min_duration_seconds: int = 10,
        max_duration_seconds: int = 240,
        top_k_per_query: int = 3,
    ) -> None:
        super().__init__(
            max_age_hours=max_age_hours,
            prefer_age_hours=prefer_age_hours,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.min_duration_seconds = max(1, int(min_duration_seconds))
        self.max_duration_seconds = max(self.min_duration_seconds, int(max_duration_seconds))
        self.top_k_per_query = max(1, int(top_k_per_query))
        self._batch_state: dict[str, Any] = {
            "id": None,
            "failures": 0,
            "warned": False,
            "disabled": False,
        }

    def search(self, query: str, *, max_results: int | None = None) -> list[dict[str, Any]]:
        safe_query = str(query or "").strip()
        if not safe_query:
            return []

        if self._batch_state.get("disabled"):
            return []

        max_results = int(max_results or 3)
        max_results = max(1, min(3, max_results))

        command = [
            "yt-dlp",
            "--dump-json",
            "--skip-download",
            f"ytsearch{max_results}:{safe_query}",
        ]

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.request_timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            self._register_failure("yt-dlp is not installed or not on PATH.")
            return []
        except Exception as exc:
            self._register_failure(f"yt-dlp search failed. query={safe_query} error={exc}")
            return []

        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            self._register_failure(
                f"yt-dlp search returned non-zero exit code. query={safe_query} stderr={stderr}"
            )
            return []

        results: list[dict[str, Any]] = []
        for line in (process.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue

            published_at = _parse_yt_dlp_published_at(payload)
            results.append(
                {
                    "title": payload.get("title"),
                    "description": payload.get("description"),
                    "duration_seconds": _safe_int(payload.get("duration")),
                    "view_count": _safe_int(payload.get("view_count")),
                    "published_at": published_at,
                    "url": payload.get("webpage_url") or payload.get("url"),
                    "raw": payload,
                }
            )

        self._reset_failures()
        return results

    def prepare_batch(self, batch_id: str) -> None:
        batch_id = str(batch_id or "")
        if self._batch_state.get("id") != batch_id:
            self._batch_state = {
                "id": batch_id,
                "failures": 0,
                "warned": False,
                "disabled": False,
            }

    def _register_failure(self, message: str) -> None:
        failures = int(self._batch_state.get("failures", 0)) + 1
        self._batch_state["failures"] = failures
        if failures >= 3:
            self._batch_state["disabled"] = True

        if not self._batch_state.get("warned"):
            LOGGER.warning(message)
            self._batch_state["warned"] = True

    def _reset_failures(self) -> None:
        self._batch_state["failures"] = 0

    def filter_results(
        self, results: Sequence[Mapping[str, Any]], context: VideoQueryContext
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for result in results:
            duration = _safe_int(result.get("duration_seconds")) or 0
            if duration < self.min_duration_seconds or duration > self.max_duration_seconds:
                continue
            filtered.append(dict(result))

        filtered = self._filter_by_recency(filtered, now=context.now)
        return self._dedupe_urls(filtered)

    def extract_media_url(self, result: Mapping[str, Any]) -> str | None:
        url = result.get("url")
        if url:
            return str(url)
        return None

    def rank_results(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        context: VideoQueryContext,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, dict[str, Any]]] = []
        title_terms = _collect_title_terms(context)
        for result in results:
            title = str(result.get("title") or "")
            similarity = _keyword_overlap_score(title, title_terms)
            recency = _recency_score(_coerce_datetime(result.get("published_at")), context.now)
            views = _safe_int(result.get("view_count")) or 0
            view_score = min(1.0, math.log10(views + 1) / 6.0)
            score = 100.0 * (0.45 * similarity + 0.35 * recency + 0.20 * view_score)
            ranked.append((score, dict(result)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked]


class TwitterVideoAgent(VideoSourceAgent):
    source_name = "twitter"

    def __init__(
        self,
        *,
        max_age_hours: float = 24.0,
        prefer_age_hours: float = 12.0,
        request_timeout_seconds: int = 12,
        top_k_per_query: int = 3,
    ) -> None:
        super().__init__(
            max_age_hours=max_age_hours,
            prefer_age_hours=prefer_age_hours,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.top_k_per_query = max(1, int(top_k_per_query))

    def search(self, query: str, *, max_results: int | None = None) -> list[dict[str, Any]]:
        safe_query = str(query or "").strip()
        if not safe_query:
            return []

        max_results = int(max_results or 15)
        max_results = max(5, min(50, max_results))

        try:
            from snscrape.modules.twitter import TwitterSearchScraper
        except Exception as exc:
            LOGGER.warning("snscrape not available. Install snscrape. error=%s", exc)
            return []

        since_date = (datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)).date()
        query_with_since = f"{safe_query} since:{since_date.isoformat()}"

        results: list[dict[str, Any]] = []
        try:
            scraper = TwitterSearchScraper(query_with_since)
            for index, tweet in enumerate(scraper.get_items()):
                if index >= max_results:
                    break

                media_entries = _extract_snscrape_media(tweet)
                if not media_entries:
                    continue

                results.append(
                    {
                        "tweet_id": getattr(tweet, "id", None),
                        "text": getattr(tweet, "rawContent", None) or getattr(tweet, "content", None),
                        "published_at": getattr(tweet, "date", None),
                        "media_entries": media_entries,
                        "tweet_url": getattr(tweet, "url", None),
                        "author": _snscrape_author_name(tweet),
                        "public_metrics": {
                            "like_count": getattr(tweet, "likeCount", None),
                            "retweet_count": getattr(tweet, "retweetCount", None),
                        },
                        "raw": {"tweet": tweet},
                    }
                )
        except Exception as exc:
            LOGGER.warning("snscrape search failed. query=%s error=%s", safe_query, exc)
            return []

        return results

    def filter_results(
        self, results: Sequence[Mapping[str, Any]], context: VideoQueryContext
    ) -> list[dict[str, Any]]:
        filtered = self._filter_by_recency(results, now=context.now)

        media_filtered: list[dict[str, Any]] = []
        for result in filtered:
            media_entries = result.get("media_entries") or []
            if not media_entries:
                continue
            for entry in media_entries:
                media_type = str(entry.get("media_type") or "").lower()
                if media_type == "image":
                    continue
                if entry.get("media_url"):
                    media_filtered.append(
                        {
                            **result,
                            "media_url": entry["media_url"],
                            "url": entry["media_url"],
                        }
                    )
        return self._dedupe_urls(media_filtered)

    def extract_media_url(self, result: Mapping[str, Any]) -> str | None:
        media_url = result.get("media_url")
        if media_url:
            return _standardize_media_url(str(media_url))
        tweet_url = result.get("tweet_url")
        if tweet_url:
            return str(tweet_url)
        return None

    def rank_results(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        context: VideoQueryContext,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, dict[str, Any]]] = []
        title_terms = _collect_title_terms(context)
        for result in results:
            text = str(result.get("text") or "")
            similarity = _keyword_overlap_score(text, title_terms)
            recency = _recency_score(_coerce_datetime(result.get("published_at")), context.now)
            engagement = _engagement_score(result)
            score = 100.0 * (0.45 * similarity + 0.35 * recency + 0.20 * engagement)
            ranked.append((score, dict(result)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked]


class ArticleVideoAgent(VideoSourceAgent):
    source_name = "article"

    def __init__(
        self,
        *,
        serpapi_key: str | None = None,
        search_endpoint: str | None = None,
        seed_article_urls: Sequence[str] | None = None,
        max_age_hours: float = 168.0,
        prefer_age_hours: float = 72.0,
        request_timeout_seconds: int = 12,
    ) -> None:
        super().__init__(
            max_age_hours=max_age_hours,
            prefer_age_hours=prefer_age_hours,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        self.search_endpoint = search_endpoint or os.getenv(
            "SERPAPI_SEARCH_URL", "https://serpapi.com/search.json"
        )
        self.seed_article_urls = list(seed_article_urls or [])

    def search(self, query: str, *, max_results: int | None = None) -> list[dict[str, Any]]:
        if self.seed_article_urls:
            return [{"article_url": url} for url in self.seed_article_urls]

        if not self.serpapi_key:
            LOGGER.warning("SerpAPI key missing. Set SERPAPI_KEY to enable article search.")
            return []

        safe_query = str(query or "").strip()
        if not safe_query:
            return []

        max_results = int(max_results or 5)
        max_results = max(1, min(10, max_results))

        params = {
            "engine": "google",
            "q": safe_query,
            "tbm": "nws",
            "num": max_results,
            "api_key": self.serpapi_key,
        }

        try:
            response = self.session.get(
                self.search_endpoint,
                params=params,
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Article search failed. query=%s error=%s", safe_query, exc)
            return []

        results: list[dict[str, Any]] = []
        for section in ("news_results", "organic_results", "top_stories"):
            for item in payload.get(section, []) or []:
                link = item.get("link") or item.get("url")
                if not link:
                    continue
                results.append(
                    {
                        "article_url": link,
                        "title": item.get("title"),
                        "source": item.get("source"),
                    }
                )

        return results

    def filter_results(
        self, results: Sequence[Mapping[str, Any]], context: VideoQueryContext
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for result in results:
            article_url = result.get("article_url") or result.get("url")
            if not article_url:
                continue
            extraction = self._extract_from_article(str(article_url))
            if not extraction:
                continue
            extraction.update({"article_url": article_url})
            filtered.append(extraction)
        filtered = self._filter_by_recency(filtered, now=context.now)
        return self._dedupe_urls(filtered)

    def extract_media_url(self, result: Mapping[str, Any]) -> str | None:
        media_url = result.get("video_url") or result.get("embedded_url")
        if media_url:
            return str(media_url)
        return None

    def _extract_from_article(self, article_url: str) -> dict[str, Any] | None:
        article_url = str(article_url or "").strip()
        if not article_url:
            return None

        publish_date: datetime | None = None
        title: str | None = None
        description: str | None = None
        video_urls: list[str] = []
        raw_payload: dict[str, Any] = {}

        try:
            from newspaper import Article

            article = Article(article_url)
            article.download()
            article.parse()
            title = article.title
            description = article.summary
            publish_date = article.publish_date
            video_urls.extend(list(article.movies or []))
        except Exception as exc:
            LOGGER.debug("newspaper3k parse failed. url=%s error=%s", article_url, exc)

        if not video_urls:
            try:
                response = self.session.get(
                    article_url,
                    timeout=self.request_timeout_seconds,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; MosahAI/4.0)"},
                )
                response.raise_for_status()
                html = response.text
                raw_payload["html_sample"] = html[:2000]
            except Exception as exc:
                LOGGER.debug("Article fetch failed. url=%s error=%s", article_url, exc)
                return None

            video_urls.extend(_extract_video_urls_from_html(html))
            if not publish_date:
                publish_date = _extract_published_time_from_html(html)

        if not video_urls:
            return None

        video_url = _normalize_embedded_url(video_urls[0])

        return {
            "video_url": video_url,
            "title": title,
            "description": description,
            "published_at": publish_date,
            "raw": raw_payload,
        }


class VideoIntelligenceEngine:
    def __init__(
        self,
        *,
        agents: Sequence[VideoSourceAgent] | None = None,
        max_candidates: int = 5,
        max_results_per_query: int = 10,
        early_exit_min_results: int = 3,
        min_score: float = 30.0,
        include_details: bool = False,
        query_builder: QueryBuilder | None = None,
        relevance_filter: NewsMediaRelevanceFilter | None = None,
        media_logger: MediaEngineLogger | None = None,
        quality_analyzer: MediaQualityAnalyzer | None = None,
        dedup_engine: VideoDeduplicationEngine | None = None,
        ranking_engine: MediaRankingEngine | None = None,
        image_fallback_provider: Callable[[VideoQueryContext], str | None] | None = None,
    ) -> None:
        self.agents = list(agents or [
            TwitterVideoAgent(),
            ArticleVideoAgent(),
            YouTubeVideoAgent(),
        ])
        self.max_candidates = max(1, int(max_candidates))
        self.max_results_per_query = max(1, int(max_results_per_query))
        self.early_exit_min_results = max(1, int(early_exit_min_results))
        self.min_score = float(min_score)
        self.include_details = bool(include_details)
        self.query_builder = query_builder or QueryBuilder()
        self.relevance_filter = relevance_filter or NewsMediaRelevanceFilter()
        self.media_logger = media_logger or MediaEngineLogger()
        self.quality_analyzer = quality_analyzer or MediaQualityAnalyzer()
        self.dedup_engine = dedup_engine or VideoDeduplicationEngine()
        self.ranking_engine = ranking_engine or MediaRankingEngine()
        self.image_fallback_provider = image_fallback_provider

    def run(
        self,
        *,
        batch_id: str,
        news_id: str | None,
        news_title: str,
        keywords: Sequence[str] | None = None,
        entities: Sequence[str] | None = None,
        summary: str | None = None,
        article_urls: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        context = VideoQueryContext(
            batch_id=str(batch_id),
            news_id=str(news_id) if news_id is not None else "",
            news_title=str(news_title or ""),
            keywords=_normalize_terms(keywords),
            entities=_normalize_terms(entities),
            summary=str(summary) if summary else None,
            now=datetime.now(timezone.utc),
            article_urls=list(article_urls or []),
        )

        for agent in self.agents:
            if isinstance(agent, YouTubeVideoAgent):
                agent.prepare_batch(context.batch_id)

        queries = self.build_queries(context)[:5]
        if self.media_logger:
            self.media_logger.search_started(
                batch_id=context.batch_id,
                news_title=context.news_title,
                queries=queries,
            )

        candidates = self._collect_from_agents_parallel(queries, context)

        source_counts = _count_by_source(candidates)
        candidates = self._apply_relevance_filter(candidates, context.news_title)
        candidates = self._apply_quality_filter(candidates)
        if self.dedup_engine:
            candidates = self.dedup_engine.remove_duplicates(candidates)

        ranked = self._rank_candidates(candidates, context)
        ranked = [c for c in ranked if c.score >= self.min_score]

        selected = self._select_by_priority(ranked, context)[:2]

        if not selected:
            fallback_ranked = self._run_fallback_strategy(context)
            fallback_ranked = [c for c in fallback_ranked if c.score >= self.min_score]
            selected = self._select_by_priority(fallback_ranked, context)[:2]

        image_candidate = None
        if not selected:
            image_candidate = self._build_image_candidate(context, ranked)
            if image_candidate:
                selected = [image_candidate]

        response: dict[str, Any] = {
            "batch_id": context.batch_id,
            "news_id": context.news_id,
            "video_candidates": [c.to_dict(include_details=self.include_details) for c in selected],
            "debug": {
                "source_counts": source_counts,
                "after_filter": len(candidates),
                "selected": len(selected),
            },
        }

        if not selected:
            response["fallback"] = "image_required"
        elif image_candidate:
            response["image_fallback"] = {"url": image_candidate.url, "source": "image"}

        if self.media_logger:
            self.media_logger.ranking_results(
                batch_id=context.batch_id,
                news_title=context.news_title,
                total_candidates=len(selected),
                top_urls=[candidate.url for candidate in selected],
            )

        return response

    def build_queries(self, context: VideoQueryContext) -> list[str]:
        title = context.news_title.strip()
        keywords = context.keywords
        entities = context.entities

        candidates: list[str] = []
        try:
            candidates = self.query_builder.generate_queries(title, keywords, entities)
        except Exception as exc:
            LOGGER.warning("Query builder failed. error=%s", exc)

        if candidates:
            return candidates

        fallback: list[str] = []
        if title:
            fallback.append(title)
        if title and keywords:
            fallback.append(f"{title} {' '.join(keywords[:4])}")
        if keywords:
            fallback.append(" ".join(keywords[:6]))
        if entities:
            fallback.append(" ".join(entities[:4]))
        if title and entities:
            fallback.append(f"{title} {' '.join(entities[:3])}")

        return _dedupe_strings(fallback)

    def _rank_candidates(
        self, candidates: Sequence[VideoCandidate], context: VideoQueryContext
    ) -> list[VideoCandidate]:
        if not candidates:
            return []

        if self.ranking_engine:
            try:
                ranked_payloads = self.ranking_engine.rank_candidates(
                    candidates,
                    news_title=context.news_title,
                    now=context.now,
                )
                ranked_candidates: list[VideoCandidate] = []
                for payload in ranked_payloads:
                    score = payload.get("final_score", payload.get("score", 0.0))
                    try:
                        score_value = float(score)
                    except Exception:
                        score_value = 0.0
                    if score_value <= 1.0:
                        score_value *= 100.0

                    ranked_candidates.append(
                        VideoCandidate(
                            source=payload.get("source") or "unknown",
                            url=str(payload.get("url") or ""),
                            score=score_value,
                            title=payload.get("title"),
                            published_at=_coerce_datetime(payload.get("published_at")),
                            duration_seconds=_safe_int(payload.get("duration_seconds")),
                            raw=payload.get("raw") or payload,
                        )
                    )
                return ranked_candidates
            except Exception as exc:
                LOGGER.warning("Ranking engine failed. error=%s", exc)

        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def rank_candidates(self, candidates: Sequence[VideoCandidate]) -> list[VideoCandidate]:
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _select_by_priority(
        self,
        candidates: Sequence[VideoCandidate],
        context: VideoQueryContext,
    ) -> list[VideoCandidate]:
        if not candidates:
            return []

        buckets: dict[str, list[VideoCandidate]] = {
            "twitter": [],
            "article": [],
            "youtube": [],
            "image": [],
        }

        for candidate in candidates:
            source_key = _normalize_source(candidate.source)
            if source_key not in buckets:
                source_key = "article"
            buckets[source_key].append(candidate)

        if buckets["twitter"]:
            return buckets["twitter"][:2]

        if buckets["article"]:
            return buckets["article"][:2]

        youtube_threshold = 0.7
        if self.relevance_filter and hasattr(self.relevance_filter, "youtube_similarity_threshold"):
            try:
                youtube_threshold = float(self.relevance_filter.youtube_similarity_threshold)
            except Exception:
                youtube_threshold = 0.7

        youtube_allowed = [
            candidate
            for candidate in buckets["youtube"]
            if _candidate_similarity(candidate) >= youtube_threshold
        ]
        if youtube_allowed:
            return youtube_allowed[:2]

        if buckets["image"]:
            return buckets["image"][:2]

        return []

    def _build_image_candidate(
        self,
        context: VideoQueryContext,
        seen_candidates: Sequence[VideoCandidate],
    ) -> VideoCandidate | None:
        if not self.image_fallback_provider:
            return None

        try:
            fallback_url = self.image_fallback_provider(context)
        except Exception as exc:
            LOGGER.warning("Image fallback failed. error=%s", exc)
            return None

        fallback_url = str(fallback_url or "").strip()
        if not fallback_url:
            return None

        seen_urls = {_canonicalize_url(candidate.url) for candidate in seen_candidates if candidate.url}
        if _canonicalize_url(fallback_url) in seen_urls:
            return None

        candidate = VideoCandidate(
            source="image",
            url=fallback_url,
            score=40.0,
            title=context.news_title,
            raw={"media_type": "image"},
        )

        if self.quality_analyzer:
            try:
                result = self.quality_analyzer.evaluate_quality(candidate)
                if result.get("reject"):
                    return None
                if isinstance(candidate.raw, dict):
                    candidate.raw = {
                        **candidate.raw,
                        "quality_score": result.get("quality_score"),
                    }
            except Exception:
                return None

        return candidate

    def _collect_from_agent(
        self,
        agent: VideoSourceAgent,
        queries: Sequence[str],
        context: VideoQueryContext,
    ) -> list[VideoCandidate]:
        aggregated_results: list[dict[str, Any]] = []
        for query in queries:
            try:
                results = agent.search(query, max_results=self.max_results_per_query)
            except Exception as exc:
                LOGGER.warning("Agent search failed. agent=%s error=%s", agent.source_name, exc)
                continue

            filtered_results = agent.filter_results(results, context)
            ranked_results = agent.rank_results(filtered_results, context=context, query=query)

            if hasattr(agent, "top_k_per_query"):
                ranked_results = ranked_results[: getattr(agent, "top_k_per_query")]

            aggregated_results.extend(ranked_results)

        candidates: list[VideoCandidate] = []
        for result in aggregated_results:
            media_url = agent.extract_media_url(result)
            if not media_url:
                continue
            score = self.score_result(result, agent.source_name, context)
            candidates.append(
                VideoCandidate(
                    source=agent.source_name,
                    url=str(media_url),
                    score=score,
                    title=result.get("title") or result.get("text"),
                    published_at=_coerce_datetime(result.get("published_at")),
                    duration_seconds=_safe_int(result.get("duration_seconds")),
                    raw=result.get("raw", {}),
                )
            )
        return self._dedupe_candidates(candidates)

    def _collect_from_agents_parallel(
        self,
        queries: Sequence[str],
        context: VideoQueryContext,
    ) -> list[VideoCandidate]:
        candidates: list[VideoCandidate] = []
        if not queries or not self.agents:
            return candidates

        for agent in self.agents:
            if isinstance(agent, ArticleVideoAgent) and context.article_urls:
                agent.seed_article_urls = list(context.article_urls)

        max_workers = max(1, len(self.agents))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for query in queries:
                future_map = {
                    executor.submit(self._search_agent_for_query, agent, query, context): agent
                    for agent in self.agents
                }

                for future in as_completed(future_map):
                    agent = future_map[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        LOGGER.warning(
                            "Agent parallel search failed. agent=%s error=%s",
                            agent.source_name,
                            exc,
                        )
                        if self.media_logger:
                            self.media_logger.agent_result(
                                batch_id=context.batch_id,
                                news_title=context.news_title,
                                agent=agent.source_name,
                                query=query,
                                status="failure",
                                result_count=0,
                            )
                        continue
                    if self.media_logger:
                        self.media_logger.agent_result(
                            batch_id=context.batch_id,
                            news_title=context.news_title,
                            agent=agent.source_name,
                            query=query,
                            status="success",
                            result_count=len(results),
                        )
                    candidates.extend(results)

                candidates = self._dedupe_candidates(candidates)
                if self._should_stop(candidates):
                    break

        return candidates

    def _run_fallback_strategy(self, context: VideoQueryContext) -> list[VideoCandidate]:
        fallback_queries = self._build_fallback_queries(context)
        if not fallback_queries:
            return []

        ordered_agents = [
            agent for agent in self.agents if isinstance(agent, TwitterVideoAgent)
        ] + [
            agent for agent in self.agents if isinstance(agent, ArticleVideoAgent)
        ] + [
            agent for agent in self.agents if isinstance(agent, YouTubeVideoAgent)
        ]

        for agent in ordered_agents:
            if isinstance(agent, ArticleVideoAgent) and context.article_urls:
                agent.seed_article_urls = list(context.article_urls)
            candidates = self._collect_from_agent(agent, fallback_queries, context)
            candidates = self._apply_relevance_filter(candidates, context.news_title)
            candidates = self._apply_quality_filter(candidates)
            if self.dedup_engine:
                candidates = self.dedup_engine.remove_duplicates(candidates)
            ranked = self._rank_candidates(candidates, context)
            ranked = [c for c in ranked if c.score >= self.min_score][: self.max_candidates]
            if ranked:
                return ranked

        return []

    def _build_fallback_queries(self, context: VideoQueryContext) -> list[str]:
        title = context.news_title.strip()
        keywords = context.keywords
        entities = context.entities

        primary_entity = entities[0] if entities else (keywords[0] if keywords else "")
        candidates: list[str] = []

        if title:
            candidates.append(title)
            candidates.append(f"{title} news")
            candidates.append(f"{title} video")

        if primary_entity:
            candidates.append(f"{primary_entity} news")
            candidates.append(f"{primary_entity} update")
            candidates.append(f"{primary_entity} video")

        if keywords:
            candidates.append(" ".join(keywords[:4]))

        if entities and keywords:
            candidates.append(f"{entities[0]} {keywords[0]}")

        return _dedupe_strings(candidates)[:8]

    def _search_agent_for_query(
        self,
        agent: VideoSourceAgent,
        query: str,
        context: VideoQueryContext,
    ) -> list[VideoCandidate]:
        try:
            results = agent.search(query, max_results=self.max_results_per_query)
        except Exception as exc:
            LOGGER.warning("Agent search failed. agent=%s error=%s", agent.source_name, exc)
            return []

        filtered_results = agent.filter_results(results, context)
        ranked_results = agent.rank_results(filtered_results, context=context, query=query)

        if hasattr(agent, "top_k_per_query"):
            ranked_results = ranked_results[: getattr(agent, "top_k_per_query")]

        candidates: list[VideoCandidate] = []
        for result in ranked_results:
            media_url = agent.extract_media_url(result)
            if not media_url:
                continue
            score = self.score_result(result, agent.source_name, context)
            candidates.append(
                VideoCandidate(
                    source=agent.source_name,
                    url=str(media_url),
                    score=score,
                    title=result.get("title") or result.get("text"),
                    published_at=_coerce_datetime(result.get("published_at")),
                    duration_seconds=_safe_int(result.get("duration_seconds")),
                    raw=result.get("raw", {}),
                )
            )

        if self.media_logger:
            self.media_logger.candidate_found(
                batch_id=context.batch_id,
                news_title=context.news_title,
                agent=agent.source_name,
                count=len(candidates),
            )

        return candidates

    def score_result(
        self, result: Mapping[str, Any], source: str, context: VideoQueryContext
    ) -> float:
        source_weight = {
            "youtube": 1.0,
            "twitter": 0.85,
            "article": 0.7,
            "image": 0.4,
        }.get(source, 0.6)

        published_at = _coerce_datetime(result.get("published_at"))
        recency_score = _recency_score(published_at, context.now)

        text = " ".join(
            str(value)
            for value in [result.get("title"), result.get("description"), result.get("text")]
            if value
        )
        keyword_score = _keyword_overlap_score(text, context.keywords + context.entities)

        engagement_score = _engagement_score(result)

        score = 100.0 * (
            0.35 * source_weight
            + 0.35 * recency_score
            + 0.20 * keyword_score
            + 0.10 * engagement_score
        )
        return score

    def _should_stop(self, candidates: Sequence[VideoCandidate]) -> bool:
        if len(candidates) >= self.early_exit_min_results:
            return True
        return False

    def _dedupe_candidates(self, candidates: Sequence[VideoCandidate]) -> list[VideoCandidate]:
        seen: set[str] = set()
        deduped: list[VideoCandidate] = []
        for candidate in candidates:
            key = _canonicalize_url(candidate.url)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _apply_relevance_filter(
        self, candidates: Sequence[VideoCandidate], news_title: str
    ) -> list[VideoCandidate]:
        if not self.relevance_filter:
            return list(candidates)

        try:
            filtered = self.relevance_filter.filter_candidates(candidates, news_title)
        except Exception as exc:
            LOGGER.warning("Relevance filter failed. error=%s", exc)
            return list(candidates)

        if not filtered:
            return []

        similarity_by_url = {
            _canonicalize_url(item.get("url", "")): item.get("score")
            for item in filtered
            if item.get("url")
        }
        allowed = set(similarity_by_url.keys())
        retained: list[VideoCandidate] = []
        for candidate in candidates:
            key = _canonicalize_url(candidate.url)
            if key not in allowed:
                continue
            if isinstance(candidate.raw, dict):
                candidate.raw = {
                    **candidate.raw,
                    "relevance_similarity": similarity_by_url.get(key),
                }
            retained.append(candidate)

        return retained

    def _apply_quality_filter(self, candidates: Sequence[VideoCandidate]) -> list[VideoCandidate]:
        if not self.quality_analyzer:
            return list(candidates)

        filtered: list[VideoCandidate] = []
        for candidate in candidates:
            try:
                result = self.quality_analyzer.evaluate_quality(candidate)
            except Exception as exc:
                LOGGER.warning("Quality analyzer failed. error=%s", exc)
                filtered.append(candidate)
                continue

            if result.get("reject"):
                continue

            if isinstance(candidate.raw, dict):
                candidate.raw = {
                    **candidate.raw,
                    "quality_score": result.get("quality_score"),
                }
            filtered.append(candidate)

        return filtered


def _normalize_terms(terms: Sequence[str] | None) -> list[str]:
    if not terms:
        return []
    cleaned: list[str] = []
    for term in terms:
        value = str(term or "").strip()
        if not value:
            continue
        cleaned.append(value)
    return _dedupe_strings(cleaned)


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _canonicalize_url(url: str) -> str:
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
        tweet_id = _extract_tweet_id(url)
        if tweet_id:
            return f"twitter.com/i/status/{tweet_id}"

    return f"{netloc}{path}"


def _normalize_source(source: str | None) -> str:
    value = str(source or "").strip().lower()
    if value in {"x", "twitter"}:
        return "twitter"
    if value in {"article", "news"}:
        return "article"
    if value in {"youtube", "yt"}:
        return "youtube"
    if value in {"image", "news_image"}:
        return "image"
    return value or "article"


def _candidate_similarity(candidate: VideoCandidate) -> float:
    raw = candidate.raw if isinstance(candidate.raw, dict) else {}
    for key in ("relevance_similarity", "relevance_score"):
        value = raw.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                return 0.0
    return 0.0


def _count_by_source(candidates: Sequence[VideoCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {"twitter": 0, "article": 0, "youtube": 0, "image": 0}
    for candidate in candidates:
        key = _normalize_source(candidate.source)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _parse_yt_dlp_published_at(payload: Mapping[str, Any]) -> datetime | None:
    timestamp = payload.get("timestamp") or payload.get("release_timestamp")
    if timestamp is not None:
        try:
            return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        except Exception:
            pass

    upload_date = payload.get("upload_date")
    if upload_date:
        value = str(upload_date).strip()
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _collect_title_terms(context: VideoQueryContext) -> list[str]:
    terms: list[str] = []
    if context.news_title:
        terms.extend(re.findall(r"[A-Za-z0-9]+", context.news_title))
    terms.extend(context.keywords or [])
    terms.extend(context.entities or [])
    return _dedupe_strings([term for term in terms if term])


def _extract_snscrape_media(tweet: Any) -> list[dict[str, Any]]:
    media_entries: list[dict[str, Any]] = []
    media_items = getattr(tweet, "media", None) or []

    for item in media_items:
        item_type = type(item).__name__.lower()

        full_url = getattr(item, "fullUrl", None) or getattr(item, "thumbnailUrl", None)
        if full_url and item_type in {"photo", "image"}:
            media_entries.append(
                {
                    "media_type": "image",
                    "media_url": full_url,
                }
            )
            continue

        variants = getattr(item, "variants", None) or []
        best_variant = _select_best_snscrape_variant(variants)
        if best_variant:
            media_entries.append(
                {
                    "media_type": "video" if item_type in {"video", "gif"} else item_type,
                    "media_url": best_variant,
                }
            )

    return media_entries


def _select_best_snscrape_variant(variants: Sequence[Any]) -> str | None:
    best_url = None
    best_bitrate = -1
    for variant in variants or []:
        url = None
        bitrate = None
        if isinstance(variant, dict):
            url = variant.get("url")
            bitrate = variant.get("bitrate") or variant.get("bit_rate")
        else:
            url = getattr(variant, "url", None)
            bitrate = getattr(variant, "bitrate", None)

        if not url:
            continue
        bitrate_value = _safe_int(bitrate) or 0
        if bitrate_value > best_bitrate:
            best_bitrate = bitrate_value
            best_url = url

    return best_url


def _snscrape_author_name(tweet: Any) -> str | None:
    user = getattr(tweet, "user", None)
    if user is None:
        return None
    return getattr(user, "username", None) or getattr(user, "displayname", None)


def _standardize_media_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return value
    if value.startswith("//"):
        value = "https:" + value
    if value.startswith("http://"):
        value = "https://" + value[len("http://") :]

    parsed = urlparse(value)
    if "pbs.twimg.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "name" not in query:
            query["name"] = ["orig"]
        if "format" not in query:
            ext = os.path.splitext(parsed.path)[1].lstrip(".")
            if ext:
                query["format"] = [ext]
        normalized_query = urlencode({key: values[0] for key, values in query.items()})
        return urlunparse(parsed._replace(query=normalized_query))

    return value


def _extract_tweet_id(url: str) -> str | None:
    match = re.search(r"/status/(\d+)", url)
    if match:
        return match.group(1)
    return None


def _parse_iso8601_duration(value: str | None) -> int | None:
    if not value:
        return None
    pattern = r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?"
    match = re.match(pattern, value)
    if not match:
        return None
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _safe_int(value: Any, *, divisor: int = 1) -> int | None:
    if value is None:
        return None
    try:
        return int(int(value) / divisor)
    except Exception:
        return None


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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _keyword_overlap_score(text: str, terms: Sequence[str]) -> float:
    if not text or not terms:
        return 0.0
    text_lower = text.lower()
    tokens = _tokenize(text)
    hits = 0
    for term in terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            hits += 1
        elif term_lower in tokens:
            hits += 1
    return hits / max(1, len(terms))


def _engagement_score(result: Mapping[str, Any]) -> float:
    view_count = _safe_int(result.get("view_count")) or 0
    metrics = result.get("public_metrics") or {}
    like_count = _safe_int(metrics.get("like_count")) or 0
    retweet_count = _safe_int(metrics.get("retweet_count")) or 0
    total = view_count + like_count + retweet_count
    if total <= 0:
        return 0.0
    return min(1.0, math.log10(total + 1) / 6.0)


def _select_best_variant(variants: Sequence[Mapping[str, Any]]) -> str | None:
    best_url = None
    best_bitrate = -1
    for variant in variants or []:
        url = variant.get("url")
        if not url:
            continue
        bitrate = _safe_int(variant.get("bit_rate")) or 0
        if bitrate > best_bitrate:
            best_bitrate = bitrate
            best_url = url
    return best_url


def _extract_video_urls_from_html(html: str) -> list[str]:
    urls: list[str] = []
    if not html:
        return urls

    patterns = [
        r"<video[^>]+src=\"([^\"]+)\"",
        r"<source[^>]+src=\"([^\"]+)\"",
        r"<iframe[^>]+src=\"([^\"]+)\"",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html, flags=re.IGNORECASE)
        urls.extend(matches)

    return urls


def _extract_published_time_from_html(html: str) -> datetime | None:
    if not html:
        return None
    patterns = [
        r"property=\"article:published_time\" content=\"([^\"]+)\"",
        r"name=\"pubdate\" content=\"([^\"]+)\"",
        r"name=\"publish-date\" content=\"([^\"]+)\"",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except Exception:
                continue
    return None


def _normalize_embedded_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if url.startswith("//"):
        url = "https:" + url

    if "youtube.com/embed/" in url:
        video_id = url.split("/embed/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"

    if "player.vimeo.com/video/" in url:
        video_id = url.split("/video/")[-1].split("?")[0]
        return f"https://vimeo.com/{video_id}"

    return url
