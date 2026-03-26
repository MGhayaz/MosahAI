"""Image intelligence pipeline orchestration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from mosahai.media_intelligence.image_downloader import ImageDownloader

from .article_image_extractor import ArticleImageExtractor
from .article_url_resolver import ArticleURLResolver
from .google_image_fetcher import GoogleImageFetcher
from .image_validator import ImageValidator
from .twitter_image_fetcher import TwitterImageFetcher


LOGGER = logging.getLogger("mosahai.image_pipeline.image_pipeline")
MINIMUM_IMAGE_COUNT = 5
ARTICLE_PRIORITY_BOOST = 5.0


@dataclass(slots=True)
class ImageCandidate:
    url: str
    source: str  # "article", "google", "twitter"
    domain: str
    quality_score: float = 0.0


class ImagePipeline:
    """Composable pipeline for sourcing, validating, and downloading images."""

    def __init__(self, downloader: ImageDownloader) -> None:
        self.downloader = downloader
        self.url_resolver = ArticleURLResolver()
        self.article_extractor = ArticleImageExtractor()
        self.google_fetcher = GoogleImageFetcher()
        self.twitter_fetcher = TwitterImageFetcher()
        self.validator = ImageValidator()

    def process_news_images(
        self,
        news_id: str,
        batch_id: str,
        title: str,
        brief: str,
        source_urls: list[str],
        output_dir: str,
    ) -> list[str]:
        """
        Main entry point.
        Should:
        1. Resolve article URLs
        2. Extract article images
        3. Fetch fallback images (google/twitter)
        4. Validate images
        5. Download images using existing ImageDownloader
        """

        cleaned_candidates = self.collect_news_images(
            title=title,
            brief=brief,
            source_urls=source_urls,
        )
        downloaded = [
            str(item.get("local_path") or "").strip()
            for item in self.download_candidates(
                news_id=news_id,
                batch_id=batch_id,
                candidates=cleaned_candidates,
                output_dir=output_dir,
            )
            if str(item.get("local_path") or "").strip()
        ]

        LOGGER.info("Image pipeline: total downloaded=%s", len(downloaded))
        return downloaded

    def collect_news_images(
        self,
        *,
        title: str,
        brief: str,
        source_urls: Sequence[str] | None,
    ) -> list[ImageCandidate]:
        del brief

        print(f"[PIPELINE] Incoming article URLs: {list(source_urls or [])}")
        resolved_urls = self._resolve_article_urls(source_urls)
        print(f"[PIPELINE] Selected article URLs: {list(resolved_urls or [])}")
        print("[PIPELINE] Calling collect_images_from_articles...")
        try:
            article_images = self.article_extractor.collect_images_from_articles(resolved_urls)
        except Exception as exc:
            LOGGER.warning("Image pipeline: multi-article extraction failed. error=%s", exc)
            article_images = []
        print(f"[PIPELINE] Total images collected: {len(article_images)}")

        original_query = "" if title is None else str(title)
        enhanced_queries = _build_enhanced_queries(original_query)
        google_images = self._collect_multi_query_results(
            fetcher=self.google_fetcher,
            source_name="google",
            queries=enhanced_queries,
        )
        twitter_images = self._collect_multi_query_results(
            fetcher=self.twitter_fetcher,
            source_name="twitter",
            queries=enhanced_queries,
        )

        all_candidates = article_images + google_images + twitter_images
        cleaned_candidates = self._validate_candidates(
            all_candidates,
            article_priority=bool(article_images),
        )

        if len(cleaned_candidates) < MINIMUM_IMAGE_COUNT and original_query.strip():
            retry_queries = _build_retry_queries(original_query)
            extra_google = self._collect_multi_query_results(
                fetcher=self.google_fetcher,
                source_name="google",
                queries=retry_queries,
            )
            extra_twitter = self._collect_multi_query_results(
                fetcher=self.twitter_fetcher,
                source_name="twitter",
                queries=retry_queries,
            )
            google_images.extend(extra_google)
            twitter_images.extend(extra_twitter)
            all_candidates = article_images + google_images + twitter_images
            cleaned_candidates = self._validate_candidates(
                all_candidates,
                article_priority=bool(article_images),
            )

        LOGGER.info(
            "[IMAGE PIPELINE] Images found per source: article=%s google=%s twitter=%s",
            len(article_images),
            len(google_images),
            len(twitter_images),
        )
        LOGGER.info("[IMAGE PIPELINE] Final selected: %s", len(cleaned_candidates))
        return cleaned_candidates

    def download_candidates(
        self,
        *,
        news_id: str,
        batch_id: str,
        candidates: Sequence[ImageCandidate],
        output_dir: str,
    ) -> list[dict[str, Any]]:
        downloaded: list[dict[str, Any]] = []

        for index, candidate in enumerate(candidates, start=1):
            source = str(getattr(candidate, "source", "") or "image").strip().lower()
            safe_source = _safe_filename_part(source)
            filename = f"image_{index}_{safe_source}.jpg"
            try:
                local_path = self.downloader.download_image(
                    url=candidate.url,
                    batch_id=batch_id,
                    news_id=news_id,
                    output_dir=output_dir,
                    filename=filename,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Image pipeline: download failed. url=%s error=%s",
                    getattr(candidate, "url", ""),
                    exc,
                )
                continue

            if not local_path:
                continue

            downloaded.append(
                {
                    "source": source or "image",
                    "url": str(getattr(candidate, "url", "") or ""),
                    "local_path": local_path,
                    "score": float(getattr(candidate, "quality_score", 0.0) or 0.0),
                    "type": "image",
                    "domain": str(getattr(candidate, "domain", "") or ""),
                }
            )

        return downloaded

    def _resolve_article_urls(self, source_urls: Sequence[str] | None) -> list[str]:
        try:
            return self.url_resolver.resolve(list(source_urls or []))
        except Exception as exc:
            LOGGER.warning("Image pipeline: URL resolve failed. error=%s", exc)
            return []

    def _collect_multi_query_results(
        self,
        *,
        fetcher: Any,
        source_name: str,
        queries: Sequence[str],
    ) -> list[ImageCandidate]:
        results: list[ImageCandidate] = []
        for query in queries:
            if not str(query or "").strip():
                continue
            LOGGER.info("[IMAGE PIPELINE] Query used: %s", query)
            try:
                results.extend(fetcher.search(query, max_results=None))
            except Exception as exc:
                LOGGER.warning("Image pipeline: %s fetch failed. query=%s error=%s", source_name, query, exc)
        return results

    def _validate_candidates(
        self,
        candidates: Sequence[ImageCandidate],
        *,
        article_priority: bool,
    ) -> list[ImageCandidate]:
        LOGGER.info("Image pipeline: total candidates=%s", len(candidates))
        try:
            cleaned_candidates = self.validator.validate(candidates)
        except Exception as exc:
            LOGGER.warning("Image pipeline: validation failed. error=%s", exc)
            cleaned_candidates = list(candidates)

        if article_priority:
            for candidate in cleaned_candidates:
                if str(getattr(candidate, "source", "") or "").strip().lower() != "article":
                    continue
                try:
                    candidate.quality_score = float(getattr(candidate, "quality_score", 0.0) or 0.0) + ARTICLE_PRIORITY_BOOST
                except Exception:
                    continue

        LOGGER.info("Image pipeline: after validation=%s", len(cleaned_candidates))
        return cleaned_candidates


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    return cleaned or "image"


def extract_keywords(text: str) -> str:
    words = str(text or "").split()
    return " ".join(words[:6])


def _build_enhanced_queries(original_query: str) -> list[str]:
    base = "" if original_query is None else str(original_query)
    if not base.strip():
        return []
    queries = [
        base,
        f"{base} news",
        f"{base} event",
        extract_keywords(base),
    ]
    return _dedupe_queries(queries)


def _build_retry_queries(original_query: str) -> list[str]:
    base = "" if original_query is None else str(original_query)
    if not base.strip():
        return []
    return _dedupe_queries(
        [
            f"{base} breaking news",
            f"{base} live",
        ]
    )


def _dedupe_queries(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = str(value or "")
        key = query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped
