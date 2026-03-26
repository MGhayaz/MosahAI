"""Resolve and normalize article URLs for image extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from mosahai.media_intelligence.article_discovery import normalize_allowed_article_url


LOGGER = logging.getLogger("mosahai.image_pipeline.article_url_resolver")


@dataclass(slots=True)
class ArticleURLResolver:
    """Resolve and normalize article URLs with retries and redirect support."""

    timeout_seconds: int = 10
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    user_agent: str = "Mozilla/5.0 (compatible; MosahAI/4.0)"

    def resolve(self, source_urls: Sequence[str]) -> list[str]:
        """Resolve a list of source URLs into canonical article URLs."""
        if not source_urls:
            return []

        resolved: list[str] = []
        seen: set[str] = set()

        session = requests.Session()
        headers = {"User-Agent": self.user_agent}

        for raw_url in source_urls:
            url = str(raw_url or "").strip()
            if not url:
                continue
            candidate_url = normalize_allowed_article_url(url)
            if not candidate_url:
                continue
            if not _is_http_url(candidate_url):
                continue

            final_url = self._resolve_single(session, candidate_url, headers)
            if not final_url:
                continue

            normalized = _normalize_url(final_url)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(normalized)

        return resolved

    def _resolve_single(self, session: requests.Session, url: str, headers: dict[str, str]) -> str:
        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            try:
                response = _resolve_redirect_url(
                    session=session,
                    url=url,
                    headers=headers,
                    timeout_seconds=self.timeout_seconds,
                )
                final_url = str(response.url or url).strip()
                if not final_url:
                    return ""
                if response.status_code >= 400:
                    LOGGER.warning(
                        "Resolver returned HTTP %s for url=%s",
                        response.status_code,
                        url,
                    )
                return final_url
            except requests.RequestException as exc:
                LOGGER.warning("Resolver request failed. url=%s error=%s", url, exc)
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                return ""
        return ""


def _resolve_redirect_url(
    *,
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    timeout_seconds: int,
) -> requests.Response:
    response = session.get(
        url,
        allow_redirects=True,
        timeout=timeout_seconds,
        headers=headers,
    )
    response.raise_for_status()
    return response


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_url(url: str) -> str | None:
    cleaned = normalize_allowed_article_url(url)
    if not cleaned:
        return None
    parsed = urlparse(cleaned)
    netloc = parsed.netloc.lower()
    if any(token in netloc for token in ("googleusercontent", "gstatic")):
        return None
    sanitized = parsed._replace(fragment="")
    normalized = urlunparse(sanitized).strip()
    return _strip_tracking_params(normalized)


def _strip_tracking_params(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url
    params = parse_qs(parsed.query, keep_blank_values=True)
    filtered: dict[str, list[str]] = {}
    for key, values in params.items():
        lower_key = key.lower()
        if lower_key.startswith("utm_"):
            continue
        filtered[key] = values
    if not filtered:
        return urlunparse(parsed._replace(query=""))
    return urlunparse(parsed._replace(query=urlencode(filtered, doseq=True)))
