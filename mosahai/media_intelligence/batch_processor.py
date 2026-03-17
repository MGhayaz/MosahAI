"""Batch media processor for MosahAI media intelligence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from mosahai.media_intelligence.batch_registry import BatchMediaRegistry
from mosahai.media_intelligence.image_downloader import ImageDownloader
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
    ) -> None:
        self.batch_registry = batch_registry or BatchMediaRegistry()
        self.engine = engine or VideoIntelligenceEngine()
        self.video_downloader = video_downloader or VideoDownloader(
            batch_registry=self.batch_registry
        )
        self.image_downloader = image_downloader or ImageDownloader(
            batch_registry=self.batch_registry
        )

    def process_news_items(self, news_items: Sequence[NewsMediaRequest]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, item in enumerate(news_items):
            print(f"[MEDIA] Processing {item.news_id}")
            results.append(self.process_media_for_news(item))
        return results

    def process_media_for_news(self, item: NewsMediaRequest) -> dict[str, Any]:
        response = self.engine.run(
            batch_id=item.batch_id,
            news_id=item.news_id,
            news_title=item.headline,
            keywords=item.keywords,
            entities=item.entities,
            summary=item.summary,
            article_urls=item.article_urls,
        )

        debug = response.get("debug", {}) or {}
        source_counts = debug.get("source_counts", {}) or {}
        print(f"[MEDIA] X results: {int(source_counts.get('twitter', 0))}")
        print(f"[MEDIA] Article results: {int(source_counts.get('article', 0))}")
        print(f"[MEDIA] YouTube results: {int(source_counts.get('youtube', 0))}")
        print(f"[MEDIA] After filter: {int(debug.get('after_filter', 0))}")

        selected_candidates = response.get("video_candidates", []) or []
        print(f"[MEDIA] Selected: {len(selected_candidates)}")

        selected_media = self._download_candidates(
            selected_candidates=selected_candidates,
            batch_id=item.batch_id,
            news_id=item.news_id,
            media_dir=item.media_dir,
        )

        self._update_metadata(
            metadata_path=item.metadata_path,
            headline=item.headline,
            selected_media=selected_media,
        )

        response["selected_media"] = selected_media
        return response

    def _download_candidates(
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
                filename = f"image_{image_index}.jpg"
                local_path = self.image_downloader.download_image(
                    url=url,
                    batch_id=batch_id,
                    news_id=news_id,
                    output_dir=media_dir,
                    filename=filename,
                )
                if local_path:
                    image_index += 1

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

    def _update_metadata(
        self,
        *,
        metadata_path: str,
        headline: str,
        selected_media: Sequence[dict[str, Any]],
    ) -> None:
        payload = _load_metadata(metadata_path)
        payload.setdefault("headline", headline)
        payload["selected_media"] = list(selected_media)

        if not selected_media:
            payload["generation_status"] = "partial"
            payload["notes"] = "media not found, fallback needed"
        else:
            payload["generation_status"] = "success"
            if "notes" not in payload:
                payload["notes"] = ""

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
