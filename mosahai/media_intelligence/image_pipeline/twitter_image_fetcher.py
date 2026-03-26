"""Fetch fallback image URLs from Twitter/X."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote_plus, urljoin, urlparse, urlunparse

import requests

if TYPE_CHECKING:
    from .image_pipeline import ImageCandidate


LOGGER = logging.getLogger("mosahai.image_pipeline.twitter_image_fetcher")
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass(slots=True)
class TwitterImageFetcher:
    """Fetch image URLs from Twitter/X via snscrape or HTML fallback."""

    max_tweets: int = 250
    timeout_seconds: int = 10

    def search(self, query: str, max_results: int | None = None) -> list["ImageCandidate"]:
        """Search for image URLs using a query string."""
        query_text = "" if query is None else str(query)
        if not query_text.strip():
            return []

        candidates = self._search_with_snscrape(query_text, max_results=max_results)
        if not candidates:
            candidates = self._search_with_html_fallback(query_text, max_results=max_results)
        if max_results is None:
            return candidates
        return candidates[: max(1, int(max_results))]

    def _search_with_snscrape(self, query: str, max_results: int | None) -> list["ImageCandidate"]:
        try:
            from snscrape.modules.twitter import Photo, TwitterSearchScraper
        except Exception as exc:
            LOGGER.warning("snscrape unavailable. error=%s", exc)
            return []

        try:
            scraper = TwitterSearchScraper(query)
        except Exception as exc:
            LOGGER.warning("snscrape search init failed. query=%s error=%s", query, exc)
            return []

        seen: set[str] = set()
        candidates: list["ImageCandidate"] = []
        processed = 0

        try:
            for tweet in scraper.get_items():
                processed += 1
                media_items = list(getattr(tweet, "media", []) or [])
                if media_items:
                    for media in media_items:
                        if not isinstance(media, Photo):
                            continue
                        _add_candidate(
                            raw_url=_select_photo_url(media),
                            seen=seen,
                            candidates=candidates,
                        )
                        if _has_reached_limit(candidates, max_results):
                            return candidates

                if processed >= self.max_tweets:
                    break
        except Exception as exc:
            LOGGER.warning("snscrape search failed. query=%s error=%s", query, exc)
            return []

        return candidates

    def _search_with_html_fallback(self, query: str, max_results: int | None) -> list["ImageCandidate"]:
        try:
            from bs4 import BeautifulSoup
        except Exception as exc:
            LOGGER.warning("BeautifulSoup unavailable for Twitter fallback. error=%s", exc)
            return []

        search_url = f"https://twitter.com/search?q={quote_plus(query)}&src=typed_query&f=media"
        try:
            response = requests.get(
                search_url,
                headers=REQUEST_HEADERS,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Twitter fallback fetch failed. query=%s error=%s", query, exc)
            return []

        soup = BeautifulSoup(response.text or "", "html.parser")
        seen: set[str] = set()
        candidates: list["ImageCandidate"] = []

        for tag in soup.select('img[src*="twimg.com/media"], img[data-src*="twimg.com/media"]'):
            raw_url = tag.get("src") or tag.get("data-src")
            _add_candidate(
                raw_url=_normalize_image_url(raw_url, response.url or search_url),
                seen=seen,
                candidates=candidates,
            )
            if _has_reached_limit(candidates, max_results):
                break

        return candidates


def _select_photo_url(media) -> str:
    for key in ("fullUrl", "full_url", "url"):
        value = getattr(media, key, None)
        if value:
            return str(value)
    return ""


def _add_candidate(
    *,
    raw_url: str | None,
    seen: set[str],
    candidates: list["ImageCandidate"],
) -> None:
    normalized = _normalize_image_url(raw_url, "https://twitter.com")
    if not normalized:
        return
    if _is_blocked_image_url(normalized):
        return
    if _looks_like_logo_filename(normalized):
        return
    key = normalized.lower()
    if key in seen:
        return
    domain = _extract_domain(normalized)
    if not domain:
        return
    from .image_pipeline import ImageCandidate

    candidates.append(
        ImageCandidate(
            url=normalized,
            source="twitter",
            domain=domain,
        )
    )
    seen.add(key)


def _has_reached_limit(candidates: list["ImageCandidate"], max_results: int | None) -> bool:
    if max_results is None:
        return False
    return len(candidates) >= max(1, int(max_results))


def _normalize_image_url(url: str | None, base_url: str) -> str:
    raw = str(url or "").strip()
    if not raw or raw.startswith("data:"):
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    absolute = urljoin(base_url, raw)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunparse(parsed._replace(fragment=""))


def _is_blocked_image_url(url: str) -> bool:
    lower = (url or "").lower()
    return lower.startswith("data:image") or ".svg" in urlparse(lower).path


def _looks_like_logo_filename(url: str) -> bool:
    filename = urlparse((url or "").lower()).path.split("/")[-1]
    return "logo" in filename


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if not host:
        return ""
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) <= 1:
        return host
    if len(parts) >= 3 and parts[-1] in {"uk", "au", "in"} and parts[-2] in {"co", "com", "org", "net"}:
        return parts[-3]
    return parts[-2]
