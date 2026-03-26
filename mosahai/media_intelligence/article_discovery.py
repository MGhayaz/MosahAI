"""Title-based discovery of trusted article URLs."""

from __future__ import annotations

import base64
import logging
import math
import re
from html import unescape
from typing import Sequence
from urllib.parse import parse_qs, quote_plus, urlencode, unquote, urljoin, urlparse, urlunparse

import requests


LOGGER = logging.getLogger("mosahai.article_discovery")
SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
}
ALLOWED_DOMAINS = (
    "bbc.com",
    "bbc.co.uk",
    "ndtv.com",
    "timesofindia.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "indianexpress.com",
    "reuters.com",
    "cnbc.com",
)
SEARCH_TIMEOUT_SECONDS = 10
MAX_DISCOVERED_URLS = 8
MAX_SELECTED_ARTICLES = 3
MIN_GOOGLE_RESULT_COUNT = 3
MIN_WEAK_RESULT_COUNT = 2
TARGET_DISCOVERED_URLS = 5
SEMANTIC_SCORE_THRESHOLD = 0.3
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_SEMANTIC_MODEL: object | bool | None = None
_SITE_LABELS = {
    "bbc.com": ("bbc", "bbc news"),
    "bbc.co.uk": ("bbc", "bbc news"),
    "ndtv.com": ("ndtv", "ndtv news"),
    "timesofindia.com": ("times of india", "the times of india", "toi"),
    "timesofindia.indiatimes.com": ("times of india", "the times of india", "toi"),
    "hindustantimes.com": ("hindustan times", "ht"),
    "indianexpress.com": ("indian express", "the indian express"),
    "reuters.com": ("reuters",),
    "cnbc.com": ("cnbc",),
}
TRACKING_QUERY_PARAM_NAMES = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "ocid",
    "cvid",
    "form",
    "ved",
    "usg",
    "ei",
    "sei",
    "ref_src",
}


def discover_articles(news_title: str) -> list[str]:
    safe_title = str(news_title or "").strip()
    if not safe_title:
        _print_discovery_clean_debug(final_urls=[])
        _print_discovery_weak_debug(final_urls=[])
        return []

    query_attempts = _build_query_attempts(safe_title)
    google_urls: list[str] = []
    bing_urls: list[str] = []

    for query in query_attempts:
        try:
            attempt_urls = _search_google(query)
        except Exception as exc:
            LOGGER.warning("Article discovery Google search failed. query=%s error=%s", query, exc)
            attempt_urls = []
        _print_discovery_attempt_debug(query=query, urls_found=attempt_urls)
        google_urls.extend(attempt_urls)
        if len(filter_allowed_article_urls(google_urls, log_cleaned=False)) >= TARGET_DISCOVERED_URLS:
            break

    filtered_google_urls = filter_allowed_article_urls(google_urls, log_cleaned=False)

    if len(filtered_google_urls) < MIN_GOOGLE_RESULT_COUNT:
        for query in query_attempts:
            try:
                attempt_urls = _search_bing(query)
            except Exception as exc:
                LOGGER.warning("Article discovery Bing search failed. query=%s error=%s", query, exc)
                attempt_urls = []
            _print_discovery_attempt_debug(query=query, urls_found=attempt_urls)
            bing_urls.extend(attempt_urls)
            combined_urls = filter_allowed_article_urls([*google_urls, *bing_urls], log_cleaned=False)
            if len(combined_urls) >= TARGET_DISCOVERED_URLS:
                break

    filtered_urls = filter_allowed_article_urls([*google_urls, *bing_urls])
    final_urls = filtered_urls[:MAX_DISCOVERED_URLS]
    _print_discovery_clean_debug(final_urls=final_urls)
    if len(final_urls) < MIN_WEAK_RESULT_COUNT:
        _print_discovery_weak_debug(final_urls=final_urls)
    return final_urls


def filter_allowed_article_urls(urls: Sequence[str], *, log_cleaned: bool = True) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()

    for raw_url in urls or []:
        cleaned = normalize_allowed_article_url(raw_url)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        if log_cleaned:
            _print_clean_article_url_debug(cleaned)
        filtered.append(cleaned)

    return filtered[:MAX_DISCOVERED_URLS]


def select_best_articles(news_title: str, urls: list[str]) -> list[str]:
    resolved_title = str(news_title or "").strip()
    candidate_urls = filter_allowed_article_urls(urls or [])
    if not resolved_title or not candidate_urls:
        _print_semantic_fail_debug()
        return []

    article_titles: list[tuple[str, str]] = []
    for url in candidate_urls:
        article_title = _fetch_article_title(url)
        if not article_title:
            continue
        article_titles.append((url, article_title))

    if not article_titles:
        _print_semantic_fail_debug()
        return []

    scores = _semantic_similarity_scores(
        news_title=resolved_title,
        article_titles=[title for _, title in article_titles],
    )

    ranked: list[tuple[float, str]] = []
    for (url, article_title), score in zip(article_titles, scores):
        similarity = float(score)
        _print_semantic_match_debug(url=url, article_title=article_title, score=similarity)
        if similarity < SEMANTIC_SCORE_THRESHOLD:
            continue
        ranked.append((similarity, url))

    if not ranked:
        _print_semantic_fail_debug()
        return []

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [url for _, url in ranked[:MAX_SELECTED_ARTICLES]]


def _build_query(news_title: str) -> str:
    return str(news_title or "").strip()


def _build_query_attempts(news_title: str) -> list[str]:
    title = _build_query(news_title)
    if not title:
        return []
    return _dedupe_strings(
        [
            title,
            _first_n_words(title, 8),
            _remove_numbers_and_special_characters(title),
            title.lower(),
        ]
    )


def _first_n_words(text: str, count: int) -> str:
    words = [word for word in str(text or "").split() if word.strip()]
    return " ".join(words[: max(1, int(count))]).strip()


def _remove_numbers_and_special_characters(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z\s]+", " ", str(text or ""))
    return " ".join(cleaned.split()).strip()


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _search_google(query: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable for Google discovery. error=%s", exc)
        return []

    url = f"https://www.google.com/search?q={quote_plus(query)}&num=10"
    response = requests.get(
        url,
        headers=SEARCH_HEADERS,
        timeout=SEARCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text or "", "html.parser")
    selectors = (
        "div.tF2Cxc a[href]",
        "div.yuRUbf a[href]",
        'a[href^="/url?q="]',
    )
    urls: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        for anchor in soup.select(selector):
            href = anchor.get("href")
            cleaned = _clean_result_url(str(href or ""))
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append(cleaned)
            if len(urls) >= MAX_DISCOVERED_URLS:
                return urls
    return urls


def _search_bing(query: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable for Bing discovery. error=%s", exc)
        return []

    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    response = requests.get(
        url,
        headers=SEARCH_HEADERS,
        timeout=SEARCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text or "", "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("li.b_algo h2 a"):
        href = anchor.get("href")
        cleaned = _clean_bing_result_url(str(href or ""))
        if cleaned:
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append(cleaned)
            if len(urls) >= MAX_DISCOVERED_URLS:
                break
    return urls


def _fetch_article_title(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable for article title parsing. url=%s error=%s", url, exc)
        return ""

    try:
        response = requests.get(
            url,
            headers=SEARCH_HEADERS,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Article title fetch failed. url=%s error=%s", url, exc)
        return ""

    try:
        soup = BeautifulSoup(response.text or "", "html.parser")
    except Exception as exc:
        LOGGER.warning("Article title parse failed. url=%s error=%s", url, exc)
        return ""

    raw_title = _extract_title_from_soup(soup)
    if not raw_title:
        return ""

    return _clean_article_title(raw_title, url)


def _clean_result_url(url: str) -> str:
    return str(unwrap_article_candidate_url(url) or "").strip()


def _clean_bing_result_url(url: str) -> str:
    cleaned = str(unwrap_article_candidate_url(url) or "").strip()
    if not cleaned:
        return ""
    if "bing.com" in (urlparse(cleaned).netloc or "").lower():
        return ""
    return cleaned


def unwrap_article_candidate_url(url: str) -> str | None:
    raw = unescape(str(url or "").strip())
    if not raw:
        return None

    raw = _expand_search_result_url(raw)
    if not raw:
        return None

    raw = _extract_bing_destination(raw)
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        return None

    cleaned = _strip_tracking_params(urlunparse(parsed._replace(fragment="")))
    if _is_blocked_rss_url(cleaned):
        _print_blocked_rss_debug(cleaned)
        return None
    if _is_google_result_artifact(cleaned):
        return None
    if _is_bing_redirect_url(cleaned):
        _print_blocked_bing_redirect_debug(cleaned)
        return None
    return cleaned


def normalize_allowed_article_url(url: str) -> str | None:
    cleaned = unwrap_article_candidate_url(url)
    if not cleaned:
        return None
    if not _is_allowed_domain(cleaned):
        return None
    return cleaned


def is_allowed_article_url(url: str) -> bool:
    return normalize_allowed_article_url(url) is not None


def _expand_search_result_url(url: str) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    if raw.startswith("/url?"):
        wrapped = urljoin("https://www.google.com", raw)
        query_params = parse_qs(urlparse(wrapped).query)
        for key in ("q", "url"):
            candidate = str((query_params.get(key) or [""])[0] or "").strip()
            if candidate:
                return candidate
        return None

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if host.startswith("www."):
        host = host[4:]
    if host.endswith("google.com") and path == "/url":
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        for key in ("q", "url"):
            candidate = str((query_params.get(key) or [""])[0] or "").strip()
            if candidate:
                return candidate
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("/"):
        return urljoin("https://www.google.com", raw)
    return raw


def _extract_bing_destination(url: str) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    if not _is_bing_redirect_url(raw):
        return raw

    parsed = urlparse(raw)
    params = parse_qs(parsed.query, keep_blank_values=True)
    keys = ("u", "url", "r", "redir", "target", "destination", "dest", "ru")

    for key in keys:
        for value in params.get(key, []):
            candidate = _decode_embedded_destination(value)
            if candidate:
                return candidate

    for values in params.values():
        for value in values:
            candidate = _decode_embedded_destination(value)
            if candidate:
                return candidate

    candidate = _decode_embedded_destination(parsed.query)
    if candidate:
        return candidate

    _print_blocked_bing_redirect_debug(raw)
    return None


def _decode_embedded_destination(value: str) -> str | None:
    text = unescape(str(value or "").strip())
    if not text:
        return None

    candidates: list[str] = []
    current = text
    for _ in range(4):
        current = str(current or "").strip()
        if not current:
            break
        if current not in candidates:
            candidates.append(current)
        decoded = unquote(current)
        if decoded == current:
            break
        current = decoded

    for candidate in list(candidates):
        lower_candidate = candidate.lower()
        if lower_candidate.startswith("a1"):
            decoded_base64 = _decode_base64_destination(candidate[2:])
            if decoded_base64:
                candidates.append(decoded_base64)

    for candidate in candidates:
        match = re.search(r"https?://[^\s\"'<>]+", candidate)
        if not match:
            continue
        cleaned = match.group(0).rstrip(")]},.;")
        if cleaned:
            return cleaned
    return None


def _decode_base64_destination(value: str) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    padded = token + "=" * (-len(token) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            decoded = decoder(padded.encode("utf-8"))
        except Exception:
            continue
        text = decoded.decode("utf-8", errors="ignore").strip()
        if text:
            return text
    return None


def _strip_tracking_params(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url

    params = parse_qs(parsed.query, keep_blank_values=True)
    filtered: dict[str, list[str]] = {}
    for key, values in params.items():
        lower_key = key.lower()
        if lower_key.startswith("utm_") or lower_key in TRACKING_QUERY_PARAM_NAMES:
            continue
        filtered[key] = values

    if not filtered:
        return urlunparse(parsed._replace(query=""))

    return urlunparse(parsed._replace(query=urlencode(filtered, doseq=True)))


def _extract_title_from_soup(soup) -> str:
    for attribute, value in (
        ("property", "og:title"),
        ("name", "twitter:title"),
    ):
        tag = soup.find("meta", attrs={attribute: value})
        if tag is None:
            continue
        content = " ".join(str(tag.get("content") or "").split())
        if content:
            return content

    title_tag = soup.find("title")
    if title_tag is None:
        return ""
    return " ".join(str(title_tag.get_text(" ", strip=True) or "").split())


def _clean_article_title(title: str, url: str) -> str:
    cleaned = " ".join(str(title or "").strip().split())
    if not cleaned:
        return ""

    cleaned = (
        cleaned.replace("—", " - ")
        .replace("–", " - ")
        .replace("â€”", " - ")
        .replace("â€“", " - ")
    )
    labels = _site_labels_for_url(url)
    separators = (" | ", " - ")

    for _ in range(3):
        updated = cleaned
        for separator in separators:
            if separator not in updated:
                continue
            parts = [part.strip() for part in updated.split(separator) if part.strip()]
            if len(parts) <= 1:
                continue
            if _looks_like_site_label(parts[-1], labels):
                updated = separator.join(parts[:-1]).strip()
                break
            if _looks_like_site_label(parts[0], labels):
                updated = separator.join(parts[1:]).strip()
                break
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned.strip()


def _site_labels_for_url(url: str) -> tuple[str, ...]:
    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for domain, labels in _SITE_LABELS.items():
        if host == domain or host.endswith("." + domain):
            return labels
    return tuple()


def _looks_like_site_label(text: str, labels: Sequence[str]) -> bool:
    value = re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()
    if not value:
        return False
    return any(label in value for label in labels)


def _semantic_similarity_scores(news_title: str, article_titles: Sequence[str]) -> list[float]:
    titles = [str(title or "").strip() for title in article_titles if str(title or "").strip()]
    if not titles:
        return []

    model = _load_semantic_model()
    if model:
        try:
            embeddings = model.encode(
                [news_title, *titles],
                normalize_embeddings=True,
            )
            news_embedding = embeddings[0]
            return [
                _cosine_similarity(news_embedding, article_embedding)
                for article_embedding in embeddings[1:]
            ]
        except Exception as exc:
            LOGGER.warning("Semantic similarity scoring failed. error=%s", exc)

    return [_token_overlap_similarity(news_title, title) for title in titles]


def _load_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is not None:
        return _SEMANTIC_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
    except Exception:
        _SEMANTIC_MODEL = False
    return _SEMANTIC_MODEL


def _cosine_similarity(left_embedding, right_embedding) -> float:
    try:
        return float(left_embedding @ right_embedding)
    except Exception:
        left = [float(value) for value in left_embedding]
        right = [float(value) for value in right_embedding]
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        if denominator <= 0:
            return 0.0
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        return float(numerator / denominator)


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens.intersection(right_tokens)
    return len(overlap) / max(len(left_tokens), len(right_tokens))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", str(text or "").lower()))


def _is_allowed_domain(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return any(host == domain or host.endswith("." + domain) for domain in ALLOWED_DOMAINS)


def _is_google_result_artifact(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith("google.com") and path == "/url":
        return True
    if host.endswith("googleusercontent.com"):
        return True
    if host.endswith("gstatic.com"):
        return True
    if host.endswith("google.com"):
        return True
    return False


def _is_blocked_rss_url(url: str) -> bool:
    lower = str(url or "").strip().lower()
    if not lower:
        return False
    if "news.google.com" in lower:
        return True

    parsed = urlparse(lower)
    path = parsed.path or ""
    if path.startswith("/rss") or "/rss/" in path or path.endswith("/rss") or path.endswith(".rss"):
        return True

    params = parse_qs(parsed.query, keep_blank_values=True)
    for key, values in params.items():
        lower_key = key.lower()
        if lower_key not in {"feed", "format", "output", "type"}:
            continue
        if any(str(value or "").strip().lower() == "rss" for value in values):
            return True
    return False


def _is_bing_redirect_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host.endswith("bing.com") and "/ck/" in path


def _print_discovery_attempt_debug(*, query: str, urls_found: Sequence[str]) -> None:
    print("[DISCOVERY]")
    print(f"Query: {str(query or '').strip()}")
    print(f"URLs found: {list(urls_found or [])}")


def _print_discovery_clean_debug(*, final_urls: Sequence[str]) -> None:
    print("[DISCOVERY CLEAN]")
    print(f"Final URLs: {list(final_urls or [])}")


def _print_discovery_weak_debug(*, final_urls: Sequence[str]) -> None:
    print(f"[DISCOVERY WEAK] {list(final_urls or [])}")


def _print_blocked_rss_debug(url: str) -> None:
    print(f"[BLOCKED RSS] {str(url or '').strip()}")


def _print_blocked_bing_redirect_debug(url: str) -> None:
    print(f"[BLOCKED BING REDIRECT] {str(url or '').strip()}")


def _print_clean_article_url_debug(url: str) -> None:
    print(f"[CLEAN ARTICLE URL] {str(url or '').strip()}")


def _print_semantic_match_debug(*, url: str, article_title: str, score: float) -> None:
    print(f"[SEMANTIC MATCH] URL: {str(url or '').strip()}")
    print(f"[SEMANTIC MATCH] Article Title: {str(article_title or '').strip()}")
    print(f"[SEMANTIC MATCH] Score: {round(float(score), 4)}")


def _print_semantic_fail_debug() -> None:
    print("[SEMANTIC FAIL] No high-quality article found")
