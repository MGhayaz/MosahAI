"""Fetch fallback image URLs from Google sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests

if TYPE_CHECKING:
    from .image_pipeline import ImageCandidate


LOGGER = logging.getLogger("mosahai.image_pipeline.google_image_fetcher")


@dataclass(slots=True)
class GoogleImageFetcher:
    """Fetch image URLs from Google Images with basic HTML parsing."""

    timeout_seconds: int = 10
    user_agent: str = "Mozilla/5.0 (compatible; MosahAI/4.0)"
    min_width: int = 200
    min_height: int = 200

    def search(self, query: str, max_results: int | None = None) -> list["ImageCandidate"]:
        """Search for image URLs using a query string."""
        query_text = "" if query is None else str(query)
        if not query_text.strip():
            return []
        url = "https://www.google.com/search"
        params = {"tbm": "isch", "q": query_text}
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Google image fetch failed. query=%s error=%s", query_text, exc)
            return []

        try:
            from bs4 import BeautifulSoup
        except Exception as exc:
            LOGGER.warning("BeautifulSoup unavailable. error=%s", exc)
            return []

        soup = BeautifulSoup(response.text or "", "html.parser")
        candidates: list["ImageCandidate"] = []
        seen: set[str] = set()

        def add_candidate(raw_url: str | None, width: int | None = None, height: int | None = None) -> None:
            del width
            del height
            normalized = _normalize_image_url(raw_url, response.url or url)
            if not normalized:
                return
            if _is_blocked_image_url(normalized):
                return
            if _looks_like_logo_filename(normalized):
                return
            if _is_svg(normalized):
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
                    source="google",
                    domain=domain,
                )
            )
            seen.add(key)

        for img in soup.find_all("img"):
            width = _safe_int(img.get("data-ow") or img.get("width"))
            height = _safe_int(img.get("data-oh") or img.get("height"))
            add_candidate(_select_img_url(img), width=width, height=height)
            if _has_reached_limit(candidates, max_results):
                break

        if max_results is None:
            return candidates
        return candidates[: max(1, int(max_results))]


def _has_reached_limit(candidates: list["ImageCandidate"], max_results: int | None) -> bool:
    if max_results is None:
        return False
    return len(candidates) >= max(1, int(max_results))


def _select_img_url(tag) -> str | None:
    if tag is None:
        return None
    for key in ("data-iurl", "data-src", "data-lazy-src", "data-original", "src"):
        value = tag.get(key)
        if value:
            return str(value)
    srcset = tag.get("srcset") or tag.get("data-srcset")
    if srcset:
        return _select_srcset_url(str(srcset))
    return None


def _select_srcset_url(srcset: str) -> str | None:
    entries = [item.strip() for item in str(srcset or "").split(",") if item.strip()]
    if not entries:
        return None
    parsed: list[tuple[int, str]] = []
    for entry in entries:
        parts = entry.split()
        if not parts:
            continue
        url = parts[0]
        width = 0
        if len(parts) > 1 and parts[1].endswith("w"):
            try:
                width = int(parts[1][:-1])
            except Exception:
                width = 0
        parsed.append((width, url))
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[0], reverse=True)
    return parsed[0][1]


def _normalize_image_url(url: str | None, base_url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.startswith("data:"):
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    absolute = urljoin(base_url, raw)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    sanitized = parsed._replace(fragment="")
    cleaned = urlunparse(sanitized)
    return _strip_tracking_params(cleaned)


def _strip_tracking_params(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url
    params = parse_qs(parsed.query, keep_blank_values=True)
    tracking_keys = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "cmpid"}
    filtered: dict[str, list[str]] = {}
    for key, values in params.items():
        lower_key = key.lower()
        if lower_key in tracking_keys or lower_key.startswith("utm_"):
            continue
        filtered[key] = values
    if not filtered:
        return urlunparse(parsed._replace(query=""))
    return urlunparse(parsed._replace(query=urlencode(filtered, doseq=True)))


def _is_blocked_image_url(url: str) -> bool:
    lower = (url or "").lower()
    return lower.startswith("data:image")


def _looks_like_logo_filename(url: str) -> bool:
    filename = urlparse((url or "").lower()).path.split("/")[-1]
    return "logo" in filename


def _is_svg(url: str) -> bool:
    lower = (url or "").lower()
    parsed = urlparse(lower)
    if parsed.path.endswith(".svg"):
        return True
    if "image/svg" in lower:
        return True
    return False


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        try:
            return int(value)
        except Exception:
            return None
    try:
        return int(float(value))
    except Exception:
        return None


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
