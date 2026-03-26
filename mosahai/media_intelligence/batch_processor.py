"""Batch media processor for MosahAI media intelligence."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence
from urllib import response

from mosahai.media_intelligence.article_discovery import (
    discover_articles,
    filter_allowed_article_urls,
    select_best_articles,
)
from mosahai.media_intelligence.batch_registry import BatchMediaRegistry
from mosahai.media_intelligence.image_downloader import ImageDownloader
from mosahai.media_intelligence.image_pipeline import fetch_primary_image
from mosahai.media_intelligence.video_downloader import VideoDownloader
from mosahai.media_intelligence.video_engine.engine import VideoIntelligenceEngine


@dataclass(slots=True)
class NewsMediaRequest:
    batch_id: str
    news_id: str
    headline: str
    keywords: list[str]
    entities: list[str]
    summary: str | None
    article_urls: list[str] | None
    media_dir: str
    metadata_path: str


class MediaBatchProcessor:
    def __init__(
        self,
        *,
        engine: VideoIntelligenceEngine | None = None,
        video_downloader: VideoDownloader | None = None,
        image_downloader: ImageDownloader | None = None,
        batch_registry: BatchMediaRegistry | None = None,
        image_pipeline: Any | None = None,
    ) -> None:
        self.batch_registry = batch_registry or BatchMediaRegistry()
        self.engine = engine or VideoIntelligenceEngine()
        self.video_downloader = video_downloader or VideoDownloader(
            batch_registry=self.batch_registry
        )
        self.image_downloader = image_downloader or ImageDownloader(
            batch_registry=self.batch_registry
        )
        self.image_pipeline = image_pipeline

    def process_news_items(self, news_items: Sequence[NewsMediaRequest]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, item in enumerate(news_items):
            print(f"[MEDIA] Processing {item.news_id}")
            results.append(self.process_media_for_news(item))
        return results

    def process_media_for_news(self, item: NewsMediaRequest) -> dict[str, Any]:
        article_urls = self._prepare_article_urls(item)
        print(f"[PIPELINE] Selected article URLs: {list(article_urls)}")

        try:
            response = self.engine.run(
                batch_id=item.batch_id,
                news_id=item.news_id,
                news_title=(getattr(item, "title", None) or getattr(item, "headline", "")),
                segment_headline=(getattr(item, "headline", None) or None),
                keywords=item.keywords,
                entities=item.entities,
                summary=item.summary,
                article_urls=article_urls,
            )
        except Exception:
            response = {
                "batch_id": item.batch_id,
                "news_id": item.news_id,
                "video_candidates": [],
                "debug": {
                    "source_counts": {"twitter": 0, "article": 0, "youtube": 0, "image": 0},
                    "after_filter": 0,
                    "selected": 0,
                },
            }
        print(f"[DEBUG] item headline: {getattr(item, 'headline', None)}")
        print(f"[DEBUG] item title: {getattr(item, 'title', None)}")

        

        debug = response.get("debug", {}) or {}
        source_counts = debug.get("source_counts", {}) or {}
        print(f"[MEDIA] X results: {int(source_counts.get('twitter', 0))}")
        print(f"[MEDIA] Article results: {int(source_counts.get('article', 0))}")
        print(f"[MEDIA] YouTube results: {int(source_counts.get('youtube', 0))}")
        print(f"[MEDIA] After filter: {int(debug.get('after_filter', 0))}")

        selected_candidates = response.get("video_candidates", []) or []
        print(f"[MEDIA] Selected: {len(selected_candidates)}")

        selected_media = self._download_video_candidates(
            selected_candidates=selected_candidates,
            batch_id=item.batch_id,
            news_id=item.news_id,
            media_dir=item.media_dir,
        )

        image_candidates = self._collect_image_candidates(item)
        image_media = self._download_image_candidates(
            image_candidates=image_candidates,
            batch_id=item.batch_id,
            news_id=item.news_id,
            media_dir=item.media_dir,
        )

        selected_media.extend(image_media)
        print(
            f"[FINAL OUTPUT] Images attached to news: "
            f"{[str(media.get('local_path') or media.get('url') or '') for media in image_media]}"
        )

        self._update_metadata(
            metadata_path=item.metadata_path,
            headline=item.headline,
            selected_media=selected_media,
        )

        response["article_urls"] = list(article_urls)
        response["image_candidates"] = [_image_candidate_payload(candidate) for candidate in image_candidates]
        response["selected_media"] = selected_media
        return response

    def _download_video_candidates(
        self,
        *,
        selected_candidates: Sequence[dict[str, Any]],
        batch_id: str,
        news_id: str,
        media_dir: str,
    ) -> list[dict[str, Any]]:
        if not selected_candidates:
            return []

        os.makedirs(media_dir, exist_ok=True)
        seen_urls: set[str] = set()
        selected_media: list[dict[str, Any]] = []
        video_index = 1
        image_index = 1

        for candidate in selected_candidates[:2]:
            url = str(candidate.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            source = str(candidate.get("source") or "").strip().lower()
            media_type = "image" if source == "image" else "video"
            score = candidate.get("score", 0)

            local_path = None
            try:
                if media_type == "video":
                    filename = f"video_{video_index}.mp4"
                    local_path = self.video_downloader.download_video(
                        url=url,
                        batch_id=batch_id,
                        news_id=news_id,
                        output_dir=media_dir,
                        filename=filename,
                    )
                    if local_path:
                        video_index += 1
                else:
                    safe_source = _safe_filename_part(source or "image")
                    filename = f"image_{image_index}_{safe_source}.jpg"
                    local_path = self.image_downloader.download_image(
                        url=url,
                        batch_id=batch_id,
                        news_id=news_id,
                        output_dir=media_dir,
                        filename=filename,
                    )
                    if local_path:
                        image_index += 1
            except Exception:
                continue

            if not local_path:
                continue

            selected_media.append(
                {
                    "source": _normalize_source_label(source),
                    "url": url,
                    "local_path": local_path,
                    "score": float(score) if score is not None else 0.0,
                    "type": media_type,
                }
            )

        return selected_media

    def _collect_image_candidates(self, item: NewsMediaRequest) -> list[Any]:
        news_title = str(getattr(item, "headline", "") or getattr(item, "title", "") or "").strip()
        if not news_title:
            return []

        try:
            image_candidate = fetch_primary_image(news_title)
        except Exception:
            image_candidate = None

        if image_candidate is None:
            return []
        return [image_candidate]

    def _prepare_article_urls(self, item: NewsMediaRequest) -> list[str]:
        headline = str(getattr(item, "headline", "") or getattr(item, "title", "") or "").strip()
        provided_urls = filter_allowed_article_urls(getattr(item, "article_urls", None) or [])

        discovered_urls: list[str] = []
        if headline:
            try:
                discovered_urls = discover_articles(headline)
            except Exception:
                discovered_urls = []

        candidate_urls = _dedupe_article_urls(list(discovered_urls) + list(provided_urls))
        if not headline or not candidate_urls:
            return []

        try:
            selected_urls = select_best_articles(headline, candidate_urls)
        except Exception:
            selected_urls = []

        return list(selected_urls)

    def _download_image_candidates(
        self,
        *,
        image_candidates: Sequence[Any],
        batch_id: str,
        news_id: str,
        media_dir: str,
    ) -> list[dict[str, Any]]:
        if not image_candidates:
            return []

        try:
            os.makedirs(media_dir, exist_ok=True)
        except OSError:
            return []
        candidate = image_candidates[0]
        source = str(getattr(candidate, "source", "") or "article").strip().lower()
        safe_source = _safe_filename_part(source or "article")
        filename = f"image_1_{safe_source}.jpg"

        try:
            local_path = self.image_downloader.download_image(
                url=str(getattr(candidate, "url", "") or ""),
                batch_id=batch_id,
                news_id=news_id,
                output_dir=media_dir,
                filename=filename,
            )
        except Exception:
            return []

        if not local_path:
            return []

        return [
            {
                "source": source or "article",
                "url": str(getattr(candidate, "url", "") or ""),
                "local_path": local_path,
                "score": float(getattr(candidate, "quality_score", 0.0) or 0.0),
                "type": "image",
                "domain": str(getattr(candidate, "domain", "") or ""),
            }
        ]

    def _update_metadata(
        self,
        *,
        metadata_path: str,
        headline: str,
        selected_media: Sequence[dict[str, Any]],
    ) -> None:
        payload = _load_metadata(metadata_path)
        payload.setdefault("headline", headline)
        existing_media = payload.get("selected_media") or []
        if selected_media:
            payload["selected_media"] = list(selected_media)
            payload["generation_status"] = "success"
            if "notes" not in payload:
                payload["notes"] = ""
        else:
            payload["selected_media"] = list(existing_media)
            if not existing_media:
                payload["generation_status"] = "partial"
                payload["notes"] = "media not found, fallback needed"

        _write_metadata(metadata_path, payload)


def _normalize_source_label(source: str) -> str:
    if source in {"twitter", "x"}:
        return "X"
    if source == "youtube":
        return "youtube"
    if source == "article":
        return "article"
    if source == "image":
        return "image"
    return source or "unknown"


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    return cleaned or "image"


def _dedupe_article_urls(urls: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_url in urls or []:
        url = str(raw_url or "").strip()
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(url)
    return deduped[:8]


def _image_candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "url": str(getattr(candidate, "url", "") or ""),
        "source": str(getattr(candidate, "source", "") or ""),
        "domain": str(getattr(candidate, "domain", "") or ""),
        "quality_score": float(getattr(candidate, "quality_score", 0.0) or 0.0),
    }


def _build_image_pipeline(image_downloader: ImageDownloader) -> Any | None:
    try:
        from mosahai.media_intelligence.image_pipeline.image_pipeline import ImagePipeline

        return ImagePipeline(downloader=image_downloader)
    except Exception:
        return None


def _load_metadata(path: str) -> dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _write_metadata(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception:
        return
