"""Extract image URLs from a resolved article."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html import unescape
from typing import TYPE_CHECKING, Sequence
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests

from mosahai.media_intelligence.article_discovery import normalize_allowed_article_url
def _print_skip_url_debug(url: str, reason: str):
    print(f"[SKIP URL] {url} :: {reason}")

if TYPE_CHECKING:
    from .image_pipeline import ImageCandidate


LOGGER = logging.getLogger("mosahai.image_pipeline.article_image_   extractor")
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}
FETCH_ATTEMPTS = 3
MAX_IMAGES_PER_ARTICLE = 5
MAX_MULTI_ARTICLE_IMAGES = 30
MIN_ARTICLE_IMAGE_WIDTH = 300
HIGH_PRIORITY_DOMAIN_TOKENS = ("toiimg", "timesofindia")
LOW_QUALITY_IMAGE_TOKENS = ("thumbnail", "small", "logo","icon","sprite","banner","ads","placeholder",)


@dataclass(slots=True)
class _RawImageOption:
    raw_url: str
    source_kind: str
    width_hint: int | None = None
    resolution_hint: str = ""


@dataclass(slots=True)
class _RankedImageOption:
    url: str
    score: float
    resolution_hint: str
    order: int


@dataclass(slots=True)
class ArticleImageExtractor:
    """Extract image URLs from article HTML with filtering and normalization."""

    timeout_seconds: int = 10
    user_agent: str = "Mozilla/5.0 (compatible; MosahAI/4.0)"
    min_width: int = 200
    min_height: int = 200

    def extract(self, article_url: str) -> list["ImageCandidate"]:
        """Return candidate image URLs for a given article."""
        url = str(article_url or "").strip()
        if not url:
            return []

        html, resolved_url = _fetch_article_html(
            url,
            timeout_seconds=self.timeout_seconds,
            user_agent=self.user_agent,
        )
        if not html or not resolved_url:
            _print_article_debug(resolved_url or url, 0)
            return []

        candidates: list["ImageCandidate"] = []
        seen: set[str] = set()

        for normalized in extract_best_images(html, resolved_url):
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            domain = _extract_domain(normalized)
            if not domain:
                continue

            from .image_pipeline import ImageCandidate

            candidates.append(
                ImageCandidate(
                    url=normalized,
                    source="article",
                    domain=domain,
                )
            )

        _print_article_debug(resolved_url or url, len(candidates))
        return candidates

    def collect_images_from_articles(self, article_urls: Sequence[str]) -> list["ImageCandidate"]:
        """Return merged article image candidates from multiple article URLs."""
        merged_candidates: list["ImageCandidate"] = []
        seen: set[str] = set()

        for article_url in article_urls or []:
            url = str(article_url or "").strip()
            if not url:
                continue

            html, resolved_url = _fetch_article_html(
                url,
                timeout_seconds=self.timeout_seconds,
                user_agent=self.user_agent,
            )
            if not html or not resolved_url:
                _print_image_extraction_debug(url=resolved_url or url, images_found=0)
                continue

            extracted_urls = extract_multi_article_images(html, resolved_url)
            _print_image_extraction_debug(url=resolved_url or url, images_found=len(extracted_urls))

            for image_url in extracted_urls:
                key = image_url.lower()
                if key in seen:
                    continue
                seen.add(key)
                domain = _extract_domain(image_url)
                if not domain:
                    continue

                from .image_pipeline import ImageCandidate

                merged_candidates.append(
                    ImageCandidate(
                        url=image_url,
                        source="article",
                        domain=domain,
                        quality_score=0.0,
                    )
                )
                if len(merged_candidates) >= MAX_MULTI_ARTICLE_IMAGES:
                    return merged_candidates

        return merged_candidates


def collect_images_from_articles(article_urls: list[str]) -> list["ImageCandidate"]:
    """Collect merged image candidates from multiple article URLs."""
    extractor = ArticleImageExtractor()
    return extractor.collect_images_from_articles(article_urls)


def extract_best_images(html: str, base_url: str) -> list[str]:
    """Return clean image URLs from article HTML in priority order."""
    markup = str(html or "")
    root_url = str(base_url or "").strip()
    if not markup:
        return []

    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable. error=%s", exc)
        return []

    try:
        soup = BeautifulSoup(markup, "html.parser")
    except Exception as exc:
        LOGGER.warning("Image extraction parse failed. base_url=%s error=%s", root_url, exc)
        return []

    raw_candidates: list[_RawImageOption] = []
    raw_candidates.extend(_find_meta_image_candidates(soup, attribute="property", value="og:image"))
    raw_candidates.extend(_find_meta_image_candidates(soup, attribute="property", value="og:image:url"))
    raw_candidates.extend(_find_meta_image_candidates(soup, attribute="property", value="og:image:secure_url"))
    raw_candidates.extend(_find_meta_image_candidates(soup, attribute="name", value="twitter:image"))
    raw_candidates.extend(_find_meta_image_candidates(soup, attribute="name", value="twitter:image:src"))

    for img in soup.find_all("img"):
        candidate = _select_img_candidate(img)
        if candidate is not None:
            raw_candidates.append(candidate)

    best_by_url: dict[str, _RankedImageOption] = {}
    for order, candidate in enumerate(raw_candidates):
        ranked = _rank_image_option(candidate, root_url=root_url, order=order)
        if ranked is None:
            continue
        key = ranked.url.lower()
        existing = best_by_url.get(key)
        if existing is None or ranked.score > existing.score:
            best_by_url[key] = ranked

    ranked_images = sorted(
        best_by_url.values(),
        key=lambda item: (-item.score, item.order),
    )[:MAX_IMAGES_PER_ARTICLE]

    for item in ranked_images:
        _print_hq_filter_debug(item.url, item.resolution_hint)

    return [item.url for item in ranked_images]


def extract_multi_article_images(html: str, base_url: str) -> list[str]:
    """Return lightly filtered image URLs from article HTML."""
    markup = str(html or "")
    root_url = str(base_url or "").strip()
    if not markup:
        return []

    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable. error=%s", exc)
        return []

    try:
        soup = BeautifulSoup(markup, "html.parser")
    except Exception as exc:
        LOGGER.warning("Multi-article image extraction parse failed. base_url=%s error=%s", root_url, exc)
        return []

    best_by_url: dict[str, _RankedImageOption] = {}
    for order, img in enumerate(soup.find_all("img")):
        candidate = _select_img_candidate_light(img)
        if candidate is None:
            continue

        ranked = _rank_light_image_option(candidate, root_url=root_url, order=order)
        if ranked is None:
            continue

        key = ranked.url.lower()
        existing = best_by_url.get(key)
        if existing is None or ranked.score > existing.score:
            best_by_url[key] = ranked

    ranked_images = sorted(
        best_by_url.values(),
        key=lambda item: (-item.score, item.order),
    )
    return [item.url for item in ranked_images[:MAX_MULTI_ARTICLE_IMAGES]]


def _fetch_article_html(url: str, *, timeout_seconds: int, user_agent: str) -> tuple[str, str]:
    del user_agent
    candidate_url = str(url or "").strip()
    if not candidate_url:
        _print_skip_url_debug(url=url, reason="invalid article URL")
        return []

    headers = dict(REQUEST_HEADERS)
    last_resolved_url = candidate_url

    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            response = requests.get(
                candidate_url,
                allow_redirects=True,
                timeout=timeout_seconds,
                headers=headers,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning(
                "Image extraction fetch failed. url=%s attempt=%s error=%s",
                candidate_url,
                attempt,
                exc,
            )
            continue

        resolved_candidate_url = normalize_allowed_article_url(str(response.url or candidate_url).strip())
        if not resolved_candidate_url:
            return "", ""
        last_resolved_url = resolved_candidate_url
        html = _extract_response_text(response)
        if html.strip():
            return html, last_resolved_url

        LOGGER.warning(
            "Image extraction fetch returned empty HTML. url=%s resolved_url=%s attempt=%s",
            candidate_url,
            last_resolved_url,
            attempt,
        )

    return "", last_resolved_url


def _is_google_news_url(url: str) -> bool:
    lower = str(url or "").strip().lower()
    return "news.google.com" in lower


def _find_meta_image_candidates(soup, *, attribute: str, value: str) -> list[_RawImageOption]:
    urls: list[_RawImageOption] = []
    for tag in soup.find_all("meta", attrs={attribute: value}):
        content = str(tag.get("content") or "").strip()
        if content:
            urls.append(
                _RawImageOption(
                    raw_url=content,
                    source_kind=f"meta:{value}",
                    resolution_hint=value,
                )
            )
    return urls


def _select_img_candidate(tag) -> _RawImageOption | None:
    if tag is None:
        return None

    width_hint = _extract_width_hint(tag)

    for key in ("data-src", "data-lazy-src", "data-original"):
        value = tag.get(key)
        if value:
            candidate = _RawImageOption(
                raw_url=str(value),
                source_kind="img:data-src",
                width_hint=width_hint,
                resolution_hint=_format_resolution_hint(width_hint, fallback="data-src"),
            )
            if not _looks_low_quality_hint(candidate.raw_url, candidate.width_hint):
                return candidate

    srcset = tag.get("srcset") or tag.get("data-srcset")
    if srcset:
        srcset_url, srcset_width = _select_srcset_url(str(srcset))
        if srcset_url:
            width = srcset_width or width_hint
            candidate = _RawImageOption(
                raw_url=srcset_url,
                source_kind="img:srcset",
                width_hint=width,
                resolution_hint=_format_resolution_hint(width, fallback="srcset"),
            )
            if not _looks_low_quality_hint(candidate.raw_url, candidate.width_hint):
                return candidate

    src = tag.get("src")
    if src:
        return _RawImageOption(
            raw_url=str(src),
            source_kind="img:src",
            width_hint=width_hint,
            resolution_hint=_format_resolution_hint(width_hint, fallback="src"),
        )

    return None


def _select_img_candidate_light(tag) -> _RawImageOption | None:
    if tag is None:
        return None

    width_hint = _extract_width_hint(tag)

    for key in ("data-src", "data-lazy-src", "data-original"):
        value = tag.get(key)
        if value:
            return _RawImageOption(
                raw_url=str(value),
                source_kind="img:data-src",
                width_hint=width_hint,
                resolution_hint=_format_resolution_hint(width_hint, fallback="data-src"),
            )

    srcset = tag.get("srcset") or tag.get("data-srcset")
    if srcset:
        srcset_url, srcset_width = _select_srcset_url(str(srcset))
        if srcset_url:
            width = srcset_width or width_hint
            return _RawImageOption(
                raw_url=srcset_url,
                source_kind="img:srcset",
                width_hint=width,
                resolution_hint=_format_resolution_hint(width, fallback="srcset"),
            )

    src = tag.get("src")
    if src:
        return _RawImageOption(
            raw_url=str(src),
            source_kind="img:src",
            width_hint=width_hint,
            resolution_hint=_format_resolution_hint(width_hint, fallback="src"),
        )

    return None


def _select_srcset_url(srcset: str) -> tuple[str | None, int | None]:
    entries = [item.strip() for item in str(srcset or "").split(",") if item.strip()]
    if not entries:
        return None, None
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
        return None, None
    parsed.sort(key=lambda item: item[0], reverse=True)
    return parsed[0][1], parsed[0][0] or None


def _extract_width_hint(tag) -> int | None:
    for key in ("data-width", "data-ow", "width", "data-image-width", "data-original-width"):
        value = _safe_int(tag.get(key))
        if value is not None:
            return value
    return None


def _format_resolution_hint(width_hint: int | None, *, fallback: str) -> str:
    if width_hint is not None:
        return f"{width_hint}w"
    return fallback


def _looks_low_quality_hint(url: str, width_hint: int | None) -> bool:
    lower = str(url or "").strip().lower()
    if any(token in lower for token in LOW_QUALITY_IMAGE_TOKENS):
        return True
    if width_hint is not None and width_hint < MIN_ARTICLE_IMAGE_WIDTH:
        return True
    return False


def _rank_image_option(
    candidate: _RawImageOption,
    *,
    root_url: str,
    order: int,
) -> _RankedImageOption | None:
    normalized = _normalize_image_url(candidate.raw_url, root_url)
    if not normalized:
        return None
    if _should_reject_image_url(normalized, width_hint=candidate.width_hint):
        return None

    score = _source_priority_score(candidate.source_kind)
    score += _width_priority_score(candidate.width_hint)
    if _has_priority_domain(normalized):
        score += 3.0

    return _RankedImageOption(
        url=normalized,
        score=score,
        resolution_hint=candidate.resolution_hint or _format_resolution_hint(candidate.width_hint, fallback="unknown"),
        order=order,
    )


def _rank_light_image_option(
    candidate: _RawImageOption,
    *,
    root_url: str,
    order: int,
) -> _RankedImageOption | None:
    normalized = _normalize_image_url(candidate.raw_url, root_url)
    if not normalized:
        return None
    if _should_light_reject_image_url(normalized):
        return None

    score = _source_priority_score(candidate.source_kind)
    score += _light_width_priority_score(candidate.width_hint)
    if _has_priority_domain(normalized):
        score += 3.0

    return _RankedImageOption(
        url=normalized,
        score=score,
        resolution_hint=candidate.resolution_hint or _format_resolution_hint(candidate.width_hint, fallback="unknown"),
        order=order,
    )


def _source_priority_score(source_kind: str) -> float:
    kind = str(source_kind or "").strip().lower()
    if kind.startswith("meta:og:image"):
        return 30.0
    if kind.startswith("meta:twitter:image"):
        return 28.0
    if kind == "img:data-src":
        return 22.0
    if kind == "img:srcset":
        return 20.0
    if kind == "img:src":
        return 14.0
    return 10.0


def _width_priority_score(width_hint: int | None) -> float:
    width = _safe_int(width_hint)
    if width is None:
        return 0.0
    if width >= 1600:
        return 7.0
    if width >= 1200:
        return 6.0
    if width >= 900:
        return 5.0
    if width >= 720:
        return 4.0
    if width >= 480:
        return 2.0
    if width >= MIN_ARTICLE_IMAGE_WIDTH:
        return 1.0
    return -100.0


def _light_width_priority_score(width_hint: int | None) -> float:
    width = _safe_int(width_hint)
    if width is None:
        return 0.0
    if width >= 1600:
        return 7.0
    if width >= 1200:
        return 6.0
    if width >= 900:
        return 5.0
    if width >= 720:
        return 4.0
    if width >= 480:
        return 2.0
    if width >= MIN_ARTICLE_IMAGE_WIDTH:
        return 1.0
    return 0.0


def _has_priority_domain(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(token in host for token in HIGH_PRIORITY_DOMAIN_TOKENS)


def _extract_response_text(response: requests.Response) -> str:
    text = str(response.text or "")
    if text.strip():
        return text

    content = response.content or b""
    if not content:
        return ""

    encoding = getattr(response, "apparent_encoding", None) or response.encoding or "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except Exception:
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return ""


def _normalize_image_url(url: str | None, base_url: str) -> str:
    raw = unescape(str(url or "").strip())
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


def _should_reject_image_url(url: str, *, width_hint: int | None = None) -> bool:
    lower = (url or "").lower()
    if lower.startswith("data:image"):
        return True
    if _is_svg(url):
        return True
    if _looks_like_logo_filename(url):
        return True
    if any(token in lower for token in LOW_QUALITY_IMAGE_TOKENS):
        return True
    if width_hint is not None and width_hint < MIN_ARTICLE_IMAGE_WIDTH:
        return True
    return False


def _should_light_reject_image_url(url: str) -> bool:
    lower = (url or "").lower()
    if lower.startswith("data:image"):
        return True
    if _is_svg(url):
        return True
    if _looks_like_logo_filename(url):
        return True
    return False


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


def _print_article_debug(url: str, images_found: int) -> None:
    print(f"[ARTICLE DEBUG] URL: {str(url or '').strip()}")
    print(f"[ARTICLE DEBUG] Images found: {int(images_found or 0)}")


def _print_hq_filter_debug(url: str, resolution_hint: str) -> None:
    print(f"[HQ FILTER] Selected image: {str(url or '').strip()}")
    print(f"[HQ FILTER] Resolution hint: {str(resolution_hint or 'unknown').strip()}")


def _print_image_extraction_debug(*, url: str, images_found: int) -> None:
    print(f"[IMAGE EXTRACTION] Article: {str(url or '').strip()}")
    print(f"[IMAGE EXTRACTION] Images found: {int(images_found or 0)}")
