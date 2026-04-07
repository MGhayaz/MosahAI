"""Title-based article image fetching for a single primary image."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Sequence, final
from urllib.parse import parse_qs, quote_plus, urlencode, urljoin, urlparse, urlunparse
from wsgiref import headers
from bs4 import BeautifulSoup
from feedparser import urls
import requests

from mosahai.media_intelligence.article_discovery import (
    filter_allowed_article_urls,
    normalize_allowed_article_url,
    unwrap_article_candidate_url,
)
from mosahai.media_intelligence.image_pipeline.twitter_image_fetcher import _normalize_image_url




LOGGER = logging.getLogger("mosahai.image_pipeline.title_image_fetcher")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT_SECONDS = 8
REQUEST_ATTEMPTS = 2
MAX_RESULT_URLS = 5
MAX_DISCOVERY_URLS = 10
MAX_ARTICLE_FETCHES = 15
TRUSTED_ARTICLE_DOMAIN_TOKENS = (

    # 🔹 Global Tech / AI / Startup
    "techcrunch", "theverge", "wired", "venturebeat",
    "technologyreview",
    "inc42", "yourstory", "analyticsindiamag",
    "medianama", "gadgets360",

    # 🔹 Global News / Geopolitics
    "bbc", "reuters", "aljazeera",
    "bloomberg", "foreignpolicy",
    "cfr",
    "carnegieindia",
    "orfonline",
    "thediplomat",

    # 🔹 Indian Major Media
    "ndtv", "indianexpress", "theprint",
    "livemint", "hindustantimes",
    "timesofindia", "toi",

    # 🔹 Hyderabad / Regional
    "siasat", "telanganatoday",
    "thehindu", "deccanchronicle",
    "thehansindia", "greatandhra",
    "newsmeter", "hyderabadmail",

    # 🔹 Extra High Quality
    "economictimes", "businessinsider",
    "forbes", "washingtonpost",
    "nytimes", "guardian",

    # 🔹 Indian Media Groups / Channels
    "news18", "firstpost", "cnnnews18",
    "dainikbhaskar",
    "ptcnews",
    "thequint",
    "punjabkesari",
    "lokmat",
    "financialexpress",
    "jansatta",
    "loksatta",
    "jagran",
    "inquilab",
    "timesnow",
    "cnbctv18",
    "cricbuzz",
    "kalaignar",
    "frontline",
    "sportstar",
    "rajasthanpatrika",
    "catchnews",
    "mathrubhumi",
    "manoramaonline",
    "amarujala",
    "sunnews",
    "dinakaran",
    "odishatv",
    "sakal",
    "indiatoday",
    "aajtak",
    "businesstoday",
    "eenadu",
    "sakshi",
    "abplive",
    "telegraphindia",
    "northeastlive",
    "indiatvnews",
    "nationalherald",
    "outlookindia",
    "dinamalar",
    "dinathanthi",
    "thanthitv",
    "zeenews",
    "wionews",
    "thepioneer",
    "republicworld",

    # 🔹 Sub-brands / Sister Publications (IMPORTANT 🔥)
    "hindubusinessline",     # The Hindu group
    "thehindubusinessline",
    "business-standard",
    "businessstandard",
    "moneycontrol",          # Network18
    "cnbc",                  # CNBC India/global
    "ndtvprofit",
    "ndtvprime",
    "indiatodaygroup",
    "timesinternet",         # Times group digital arm
    "economictimes.indiatimes",
    "navbharattimes",
    "maharashtratimes",
    "vijaykarnataka",
    "bangaloremirror",
    "ahmedabadmirror",

    # 🔹 Indian News Agencies
    "pti",
    "uniindia",
    "hindustansamachar",
    "aninews",
    "ians",
    "ptibhasha",
    "news18wire",
    "newsvoir",

    # 🔹 Global News Agencies
    "apnews",
    "afp",
    "xinhuanet",
    "tass",
    "dpa",
    "kyodonews",
    "efe",
    "unnews"
)
PREFERRED_IMAGE_URL_TOKENS = ("getty", "apnews", "image", "media")
LOW_QUALITY_IMAGE_TOKENS = ("thumbnail", "small")
BLOCKED_IMAGE_DOMAIN_TOKENS = ("googleusercontent", "gstatic", "news.google")
BLOCKED_FILENAME_TOKENS = ("logo", "icon", "sprite")
LOCAL_TRUSTED_DOMAINS = [
    "siasat",
    "telangana",
    "thehindu",
    "ndtv",
]

def _is_trusted_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(token in domain for token in TRUSTED_ARTICLE_DOMAIN_TOKENS)


@dataclass(slots=True)
class _ImageMetadataCandidate:
    url: str
    width: int | None = None
    article_domain: str = ""
    query: str = ""   # ✅ ADD THIS

@dataclass(slots=True)
class _ScoredImageCandidate:
    url: str
    article_domain: str
    width: int | None
    score: float
    order: int


def fetch_primary_image(news_title: str) -> ImageCandidate | None:
    """Find the best available article image for a news title."""
    from .image_pipeline import ImageCandidate
    try:
        title = _clean_title(news_title)
        if not title:
            print("[FINAL SELECTED IMAGE] ")
            return None

        seen_article_urls: set[str] = set()
        total_article_fetches = 0

        for attempt_index, query in enumerate(_build_queries(title), start=1):
            print(f"[ATTEMPT {attempt_index}] query: {query}")
            print(f"[FETCH IMAGE] Query: {query}")
            try:
                result_urls = _discover_result_urls(query)
            except Exception as exc:
                LOGGER.warning("Title image discovery failed. query=%s error=%s", query, exc)
                _print_fetch_fail_debug(context="search", target=query, error=exc)
                result_urls = []

            if not result_urls:
                print("[ATTEMPT RESULT] fail")
                continue

            attempt_candidates: dict[str, _ImageMetadataCandidate] = {}
            for raw_url in result_urls[:MAX_RESULT_URLS]:
                if total_article_fetches >= MAX_ARTICLE_FETCHES:
                    _print_skip_url_debug(url=str(raw_url or ""), reason="article fetch limit reached")
                    break

                cleaned_url = _clean_result_url(raw_url)
                if not cleaned_url:
                    _print_skip_url_debug(url=str(raw_url or ""), reason="invalid result URL")
                    continue

                key = cleaned_url.lower()
                if key in seen_article_urls:
                    _print_skip_url_debug(url=cleaned_url, reason="duplicate result URL")
                    continue
                seen_article_urls.add(key)

                print(f"[FETCH IMAGE] URL: {cleaned_url}")
                total_article_fetches += 1
                try:
                    extracted_candidates = _fetch_image_candidates_from_article(cleaned_url, query)
                except Exception as exc:
                    LOGGER.warning("Title image article processing failed. url=%s error=%s", cleaned_url, exc)
                    _print_fetch_fail_debug(context="article", target=cleaned_url, error=exc)
                    extracted_candidates = []

                if not extracted_candidates:
                    _print_skip_url_debug(url=cleaned_url, reason="no valid image candidates")
                    continue

                for candidate in extracted_candidates:
                    image_key = candidate.url.lower()
                    existing = attempt_candidates.get(image_key)
                    if existing is None or _is_better_candidate(candidate, existing):
                        attempt_candidates[image_key] = candidate

            ranked_candidates = _rank_candidates(list(attempt_candidates.values()))

            valid_candidates = [c for c in ranked_candidates if c.score >= 0]

            if valid_candidates:
                best = valid_candidates[0]

                return ImageCandidate(
                    url=best.url,
                    source="article",
                    domain=best.article_domain,
                    quality_score=best.score,
                )

            print("[ATTEMPT RESULT] fail")
            if total_article_fetches >= MAX_ARTICLE_FETCHES:
                break

        print("[FETCH IMAGE] Selected Image: ")
        print("[FINAL SELECTED IMAGE] ")
        return None
    except Exception as exc:
        LOGGER.warning("Title image fetcher failed unexpectedly. error=%s", exc)
        _print_fetch_fail_debug(context="fetch_primary_image", target=str(news_title or ""), error=exc)
        print("[FINAL SELECTED IMAGE] ")
        return None


def _rank_candidates(candidates: Sequence[_ImageMetadataCandidate]) -> list[_ScoredImageCandidate]:
    ranked: list[_ScoredImageCandidate] = []
    for order, candidate in enumerate(candidates):
        score = _score_image_candidate(candidate)
        print(f"[IMAGE RANKING] URL: {candidate.url}")
        print(f"[IMAGE RANKING] Score: {score}")
        ranked.append(
            _ScoredImageCandidate(
                url=candidate.url,
                article_domain=candidate.article_domain,
                width=candidate.width,
                score=score,
                order=order,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score,
            -(item.width or 0),
            item.order,
        )
    )
    ranked = [item for item in ranked if item.score > 0]
    return ranked


def _select_best_candidate(candidates: Sequence[_ImageMetadataCandidate]) -> _ScoredImageCandidate | None:
    ranked = _rank_candidates(candidates)
    if not ranked and candidates:
        print("[IMAGE FALLBACK] Using first available candidate")
        first = candidates[0]
        return _ScoredImageCandidate(
            url=first.url,
            article_domain=first.article_domain,
            width=first.width,
            score=0.1,
            order=0,
    )
    print(f"[CANDIDATES COUNT] {len(candidates)}")
    return ranked[0]


def _score_image_candidate(candidate: _ImageMetadataCandidate) -> float:
    score = 0.0

    lower_url = (candidate.url or "").lower()
    article_domain = (candidate.article_domain or "").strip().lower()
    image_host = (urlparse(lower_url).netloc or "").lower()

    # 🚫 HARD REJECT (LOGO / ICON) — FIRST
    if any(token in lower_url for token in ["logo", "icon", "sprite"]):
        return -999

    # ✅ trusted domain boost
    if any(token in article_domain or token in image_host for token in TRUSTED_ARTICLE_DOMAIN_TOKENS):
        score += 2.0
    else:
        score += 0.5

    # ✅ preferred image patterns
    if any(token in lower_url for token in PREFERRED_IMAGE_URL_TOKENS):
        score += 2.0

    # ✅ local trusted boost
    if any(domain in article_domain for domain in LOCAL_TRUSTED_DOMAINS):
        score += 2.5

    # ✅ size handling
    if candidate.width is None:
        score += 0.5
    elif candidate.width >= 800:
        score += 1.0

    # ❌ low quality penalty
    if any(token in lower_url for token in LOW_QUALITY_IMAGE_TOKENS):
        score -= 2.0

    # 🌍 irrelevant context penalty
    if any(token in lower_url for token in ["tourism", "trip"]):
        score -= 2.0

    # 🔥 domain relevance boost
    if article_domain and article_domain in lower_url:
        score += 1.5

    # 🔥 TITLE MATCH BOOST (FIXED)
    title_words = set(re.split(r"[\s\-_]+", article_domain))
    url_words = set(re.split(r"[\s\-_]+", lower_url))

    match_score = len(title_words & url_words)

    if match_score >= 2:
        score += 2.0
    elif match_score == 1:
        score += 0.5

    return score


def _is_better_candidate(left: _ImageMetadataCandidate, right: _ImageMetadataCandidate) -> bool:
    left_score = _score_image_candidate(left)
    right_score = _score_image_candidate(right)
    if left_score != right_score:
        return left_score > right_score
    return (left.width or 0) > (right.width or 0)

def _clean_title(title: str) -> str:
    import re

    text = title or ""

    # remove everything after dash
    text = re.sub(r"\s[-–]\s.*$", "", text)

    # remove common news suffix words
    text = re.sub(r"\b(news|live|updates)\b", "", text, flags=re.I)

    # remove domains
    text = re.sub(r"\b\w+\.com\b", "", text)

    # remove ALL CAPS words (usually brands)
    text = re.sub(r"\b[A-Z]{3,}\b", "", text)

    return " ".join(text.split()).strip()
def _build_queries(news_title: str) -> list[str]:
    clean = _clean_title(news_title)
    return [
    clean,
    clean + " news",
    clean + " india news"
] if clean else []

def _first_n_words(text: str, count: int) -> str:
    words = [word for word in str(text or "").split() if word.strip()]
    return " ".join(words[: max(1, int(count))]).strip()


def _remove_numbers_and_special_characters(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z\s]+", " ", str(text or ""))
    return " ".join(cleaned.split()).strip()


def _dedupe_queries(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _is_trusted_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(token in domain for token in TRUSTED_ARTICLE_DOMAIN_TOKENS)


def _discover_result_urls(query: str) -> list[str]:
    try:
        google_urls = _search_google(query)
    except Exception as exc:
        LOGGER.warning("Google discovery execution failed. query=%s error=%s", query, exc)
        _print_fetch_fail_debug(context="google", target=query, error=exc)
        google_urls = []

    bing_urls: list[str] = []
    if not google_urls:
        try:
            bing_urls = _search_bing(query)
        except Exception as exc:
            LOGGER.warning("Bing discovery execution failed. query=%s error=%s", query, exc)
            _print_fetch_fail_debug(context="bing", target=query, error=exc)
            bing_urls = []

    # STEP 1: dedupe
    urls_found = _dedupe_urls([*google_urls, *bing_urls])

    # STEP 2: allowed filter (basic cleanup)
    filtered_urls = filter_allowed_article_urls(urls_found)

    # STEP 3: trusted filter
    trusted_urls = [
        url for url in filtered_urls
        if _is_trusted_domain(url)
    ]

    # STEP 4: fallback logic
    if trusted_urls:
        clean_urls = trusted_urls[:MAX_DISCOVERY_URLS]
    else:
    # ❌ instead of blindly fallback
        clean_urls = [
        url for url in filtered_urls
        if any(token in url.lower() for token in ["news", "india", "report", "live"])
    ]   [:MAX_DISCOVERY_URLS]

    _print_discovery_debug(
        query=query,
        urls_found=urls_found,
        clean_urls=clean_urls
    )

    print(f"[DEBUG RAW URLS] {urls_found}")
    return clean_urls


def _search_google(query: str) -> list[str]:
    
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable for title image Google search. error=%s", exc)
        _print_parse_fail_debug(context="google-import", target=query, error=exc)
        return []

    url = f"https://www.google.com/search?q={quote_plus(query)}&num={MAX_RESULT_URLS}"
    response = _request_with_retry(url, allow_redirects=False, context="google-search")
    if response is None:
        return []

    try:
        soup = BeautifulSoup(response.text or "", "html.parser")
    except Exception as exc:
        LOGGER.warning("Title image Google result parse failed. query=%s error=%s", query, exc)
        _print_parse_fail_debug(context="google-parse", target=query, error=exc)
        return []

    urls: list[str] = []
    try:
        for a_tag in soup.find_all("a"):
            href = a_tag.get("href", "")

            if "/url?q=" in href:
                try:
                    url = parse_qs(urlparse(href).query).get("q", [""])[0]
                except:
                    continue

                if url and "http" in url:
                    if any(bad in url.lower() for bad in [
                        # Google junk
                        "google.com", "google.co.in", "webhp", "search?", "tbm=", "source=", "ved=", "ei=", "uact=", "oq=", "gs_lcp=",

                        # Google services
                        "accounts.google", "support.google", "images.google", "maps.google", "policies.google", "news.google",

                        # Social / useless
                        # "youtube.com", "youtu.be", "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com", "pinterest.com",

                        # Auth pages
                        "login", "signup", "register", "signin", "logout", "auth",

                        # Tracking / redirects
                        "redirect", "utm_", "ref=", "fbclid", "gclid",

                        # Files (SEO me bekaar)
                        ".pdf", ".jpg", ".jpeg", ".png", ".webp", ".svg", ".mp4", ".mp3", ".zip",

                        # Misc junk
                        "mailto:", "tel:", "/privacy", "/terms", "/policy", "/cookies", "/about", "/contact",
                        "stackoverflow.com",
                        "stackexchange.com",
                        "superuser.com",
                        "serverfault.com",
                        "zhihu.com",
                        "zhidao.baidu.com",
                        "baidu.com",
                        "quora.com",
                        "reddit.com",
                        "geeksforgeeks.org",  # optional (agar avoid karna hai)
                        "tutorialspoint.com", # optional low-quality content
                        "medium.com",  # optional (depends on use case)
                        "brainly",
                        "chegg.com"
                        
                    ]):
                        continue
                    if any(spam in url.lower() for spam in [
                        "filmweb", "douban", "imdb", "rottentomatoes",
                          "boxoffice", "film", "movie", "cinema"
                    ]):
                        continue
                    
                    urls.append(url)

                if len(urls) >= MAX_RESULT_URLS:
                    break   
    except Exception as exc:
        LOGGER.warning("Title image Google result traversal failed. query=%s error=%s", query, exc)
        _print_parse_fail_debug(context="google-results", target=query, error=exc)
        return []
    return urls


def _search_bing(query: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable for title image Bing search. error=%s", exc)
        _print_parse_fail_debug(context="bing-import", target=query, error=exc)
        return []

    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    response = _request_with_retry(url, allow_redirects=False, context="bing-search")
    if response is None:
        return []

    try:
        soup = BeautifulSoup(response.text or "", "html.parser")
    except Exception as exc:
        LOGGER.warning("Title image Bing result parse failed. query=%s error=%s", query, exc)
        _print_parse_fail_debug(context="bing-parse", target=query, error=exc)
        return []

    urls: list[str] = []
    try:
        for anchor in soup.select("li.b_algo h2 a"):
            href = str(anchor.get("href") or "").strip()
            cleaned = _clean_result_url(href)
            if not cleaned:
                continue

            lower_url = cleaned.lower()

            if any(bad in lower_url for bad in [
                "zhihu.com",
                "baidu.com",
                "stackoverflow.com",
                "stackexchange.com",
                "quora.com",
                "reddit.com"
            ]):
                continue

            urls.append(cleaned)
            if len(urls) >= MAX_RESULT_URLS:
                break
    except Exception as exc:
        LOGGER.warning("Title image Bing result traversal failed. query=%s error=%s", query, exc)
        _print_parse_fail_debug(context="bing-results", target=query, error=exc)
        return []
    return urls


def _fetch_image_candidates_from_article(url: str) -> list[_ImageMetadataCandidate]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        LOGGER.warning("BeautifulSoup unavailable for title image article parse. url=%s error=%s", url, exc)
        _print_parse_fail_debug(context="article-import", target=url, error=exc)
        return []

    candidate_url = str(url or "").strip()
    if not candidate_url:
        _print_skip_url_debug(url=url, reason="invalid article URL")
        return []

    response = _request_with_retry(candidate_url, allow_redirects=True, context="article-fetch")
    if response is None:
        _print_skip_url_debug(url=candidate_url, reason="fetch failed")
        return []

    resolved_url = str(response.url or candidate_url).strip() or candidate_url
    clean_resolved_url = resolved_url
    try:
        soup = BeautifulSoup(response.text or "", "html.parser")
    except Exception as exc:
        LOGGER.warning("Title image article HTML parse failed. url=%s error=%s", clean_resolved_url, exc)
        _print_parse_fail_debug(context="article-parse", target=clean_resolved_url, error=exc)
        return []

    article_domain = _extract_domain(clean_resolved_url)
    accepted: list[_ImageMetadataCandidate] = []
    try:
        for candidate in _extract_metadata_images(soup, clean_resolved_url):
            rejection_reason = _rejected_image_reason(candidate)
            if rejection_reason:
                _print_rejected_image_debug(url=candidate.url, reason=rejection_reason)
                continue
            candidate.article_domain = article_domain
            candidate.query = query
            if not candidate.article_domain:
                continue
            accepted.append(candidate)
    except Exception as exc:
        LOGGER.warning("Title image candidate extraction failed. url=%s error=%s", resolved_url, exc)
        _print_parse_fail_debug(context="article-candidates", target=resolved_url, error=exc)
        return []
    return accepted


def _extract_metadata_images(soup, base_url: str) -> list[_ImageMetadataCandidate]:
    candidates: list[_ImageMetadataCandidate] = []

    candidates.extend(
        _find_meta_candidates(
            soup,
            base_url=base_url,
            attribute="property",
            value="og:image",
            width_attribute="property",
            width_value="og:image:width",
        )
    )
    candidates.extend(
        _find_meta_candidates(
            soup,
            base_url=base_url,
            attribute="name",
            value="twitter:image",
            width_attribute="name",
            width_value="twitter:image:width",
        )
    )
    candidates.extend(_find_jsonld_candidates(soup, base_url=base_url))

    best_by_url: dict[str, _ImageMetadataCandidate] = {}
    for candidate in candidates:
        key = candidate.url.lower()
        existing = best_by_url.get(key)
        if existing is None:
            best_by_url[key] = candidate
            continue
        existing_width = existing.width or 0
        candidate_width = candidate.width or 0
        if candidate_width > existing_width:
            best_by_url[key] = candidate
    # 🔥 FALLBACK: first <img> tag
    try:
        first_img = soup.find("img")
        if first_img:
            src = _normalize_image_url(first_img.get("src"), base_url)
            if src:
                candidate = _ImageMetadataCandidate(url=src, width=None)
                candidates.append(candidate)

                # 🔥 also inject into best_by_url
                key = candidate.url.lower()
                if key not in best_by_url:
                    best_by_url[key] = candidate
    except Exception:
        pass        
        final: list[_ImageMetadataCandidate] = []

    try:
        final = list(best_by_url.values())

        if not final and candidates:
            print("[IMAGE FALLBACK] using raw img candidates")
            final = candidates

    except Exception as e:
        print(f"[FINAL BUILD ERROR] {e}")
        final = candidates if candidates else []

    return final


def _find_meta_candidates(
    soup,
    *,
    base_url: str,
    attribute: str,
    value: str,
    width_attribute: str,
    width_value: str,
) -> list[_ImageMetadataCandidate]:
    width = _extract_meta_width(soup, attribute=width_attribute, value=width_value)
    results: list[_ImageMetadataCandidate] = []
    for tag in soup.find_all("meta", attrs={attribute: value}):
        content = _normalize_image_url(tag.get("content"), base_url)
        if not content:
            continue
        results.append(_ImageMetadataCandidate(url=content, width=width))
    return results


def _extract_meta_width(soup, *, attribute: str, value: str) -> int | None:
    tag = soup.find("meta", attrs={attribute: value})
    if tag is None:
        return None
    return _safe_int(tag.get("content"))


def _find_jsonld_candidates(soup, *, base_url: str) -> list[_ImageMetadataCandidate]:
    candidates: list[_ImageMetadataCandidate] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_text = str(script.string or script.get_text() or "").strip()
        if not raw_text:
            continue

        for payload in _parse_jsonld_payloads(raw_text):
            for image_value, width in _collect_jsonld_images(payload):
                normalized = _normalize_image_url(image_value, base_url)
                if not normalized:
                    continue
                candidates.append(_ImageMetadataCandidate(url=normalized, width=width))
    return candidates


def _parse_jsonld_payloads(raw_text: str) -> list[object]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    payloads: list[object] = []
    try:
        payloads.append(json.loads(text))
        return payloads
    except Exception:
        pass

    decoder = json.JSONDecoder()
    remaining = text
    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        try:
            parsed, end_index = decoder.raw_decode(remaining)
        except Exception:
            break
        payloads.append(parsed)
        remaining = remaining[end_index:]
    if not payloads and text:
        _print_parse_fail_debug(context="jsonld", target=text[:120], error="unable to decode JSON-LD")
    return payloads


def _request_with_retry(url: str, *, allow_redirects: bool, context: str) -> requests.Response | None:
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            headers = HEADERS.copy()
            headers["Referer"] = "https://www.google.com/"
            headers["Accept"] = "text/html,application/xhtml+xml"

            response = requests.get(
                url,
                headers=headers,
                timeout=TIMEOUT_SECONDS,
                allow_redirects=allow_redirects,
            )

            response.raise_for_status()
            return response

        except requests.Timeout as exc:
            LOGGER.warning(
                "Title image request timed out. context=%s url=%s attempt=%s error=%s",
                context,
                url,
                attempt,
                exc,
            )
            _print_fetch_fail_debug(context=context, target=url, error=exc)

        except requests.RequestException as exc:
            LOGGER.warning(
                "Title image request failed. context=%s url=%s attempt=%s error=%s",
                context,
                url,
                attempt,
                exc,
            )
            _print_fetch_fail_debug(context=context, target=url, error=exc)

            # 🔥 SECOND FALLBACK (CRITICAL FIX)
            try:
                fallback_headers = HEADERS.copy()
                fallback_headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"
                fallback_headers["Referer"] = ""
                fallback_headers["Accept"] = "*/*"

                response = requests.get(
                    url,
                    headers=fallback_headers,
                    timeout=TIMEOUT_SECONDS,
                    allow_redirects=True
                )

                response.raise_for_status()
                print("[FALLBACK SUCCESS]", url)
                return response

            except Exception as fallback_exc:
                print(f"[FALLBACK FAILED] {url} :: {fallback_exc}")

        except Exception as exc:
            LOGGER.warning(
                "Title image request crashed. context=%s url=%s attempt=%s error=%s",
                context,
                url,
                attempt,
                exc,
            )
            _print_fetch_fail_debug(context=context, target=url, error=exc)
            break

    return None


def _collect_jsonld_images(payload: object) -> list[tuple[str, int | None]]:
    collected: list[tuple[str, int | None]] = []
    if isinstance(payload, dict):
        if "image" in payload:
            collected.extend(_coerce_image_values(payload.get("image"), parent_width=_safe_int(payload.get("width"))))
        if "@graph" in payload:
            collected.extend(_collect_jsonld_images(payload.get("@graph")))
        for value in payload.values():
            if isinstance(value, (dict, list)):
                collected.extend(_collect_jsonld_images(value))
    elif isinstance(payload, list):
        for item in payload:
            collected.extend(_collect_jsonld_images(item))
    return collected


def _coerce_image_values(value: object, *, parent_width: int | None = None) -> list[tuple[str, int | None]]:
    if isinstance(value, str):
        return [(value, parent_width)]
    if isinstance(value, dict):
        url = value.get("url") or value.get("contentUrl")
        width = _safe_int(value.get("width")) or parent_width
        results: list[tuple[str, int | None]] = []
        if isinstance(url, str) and url.strip():
            results.append((url, width))
        if "image" in value:
            results.extend(_coerce_image_values(value.get("image"), parent_width=width))
        return results
    if isinstance(value, list):
        results: list[tuple[str, int | None]] = []
        for item in value:
            results.extend(_coerce_image_values(item, parent_width=parent_width))
        return results
    return []


def _clean_result_url(url: str) -> str:
    return str(unwrap_article_candidate_url(url) or "").strip()


def _normalize_image_url(url: object, base_url: str) -> str:
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

    return _strip_tracking_params(urlunparse(parsed._replace(fragment="")))


def _strip_tracking_params(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url

    params = parse_qs(parsed.query, keep_blank_values=True)
    kept: list[tuple[str, str]] = []
    for key, values in params.items():
        lower_key = key.lower()
        if lower_key.startswith("utm_"):
            continue
        for value in values:
            kept.append((key, value))

    if not kept:
        return urlunparse(parsed._replace(query=""))

    query = urlencode(kept, doseq=True)
    return urlunparse(parsed._replace(query=query))


def _rejected_image_reason(candidate: _ImageMetadataCandidate) -> str:
    lower = candidate.url.lower()
    if lower.startswith("data:image"):
        return "data:image URL"
    if _is_svg(candidate.url):
        return "SVG image"
    parsed = urlparse(lower)
    netloc = parsed.netloc or ""
    if any(token in netloc for token in BLOCKED_IMAGE_DOMAIN_TOKENS):
        return "blocked Google-hosted domain"
    filename = parsed.path.split("/")[-1]
    if any(token in filename for token in ["logo","icon","sprite","avatar","placeholder"]):
        return "logo/icon/sprite filename"



def _is_svg(url: str) -> bool:
    lower = str(url or "").lower()
    parsed = urlparse(lower)
    return parsed.path.endswith(".svg") or "image/svg" in lower


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except Exception:
        return None


def _extract_domain(url: str) -> str:
    host = (urlparse(url).netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 1:
        return host
    if len(parts) >= 3 and parts[-1] in {"uk", "au", "in"} and parts[-2] in {"co", "com", "org", "net"}:
        return parts[-3]
    return parts[-2]


def _dedupe_urls(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        url = _clean_result_url(value)
        key = url.lower()
        if not url or key in seen:
            continue
        seen.add(key)
        deduped.append(url)
    return deduped


def _print_discovery_debug(*, query: str, urls_found: Sequence[str], clean_urls: Sequence[str]) -> None:
    print("[DISCOVERY CLEAN]")
    print(f"Query: {str(query or '').strip()}")
    print(f"URLs found: {list(urls_found or [])}")
    print(f"Clean URLs: {list(clean_urls or [])}")


def _print_fetch_fail_debug(*, context: str, target: str, error: object) -> None:
    print(f"[FETCH FAIL] {str(context or '').strip()}: {str(target or '').strip()} :: {str(error or '').strip()}")


def _print_parse_fail_debug(*, context: str, target: str, error: object) -> None:
    print(f"[PARSE FAIL] {str(context or '').strip()}: {str(target or '').strip()} :: {str(error or '').strip()}")


def _print_skip_url_debug(*, url: str, reason: str) -> None:
    print(f"[SKIP URL] {str(url or '').strip()} :: {str(reason or '').strip()}")


def _print_rejected_image_debug(*, url: str, reason: str) -> None:
    print(f"[REJECTED IMAGE] URL: {str(url or '').strip()}")
    print(f"[REJECTED IMAGE] Reason: {str(reason or '').strip()}")
