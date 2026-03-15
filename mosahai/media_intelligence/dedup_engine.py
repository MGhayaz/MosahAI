"""Video deduplication engine for MosahAI media candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


@dataclass(slots=True)
class VideoDeduplicationEngine:
    """Remove duplicate video candidates across multiple sources."""

    def normalize_url(self, url: str) -> str:
        value = str(url or "").strip()
        if not value:
            return ""
        if value.startswith("//"):
            value = "https:" + value
        if not value.startswith("http"):
            value = "https://" + value.lstrip("/")

        parsed = urlparse(value)
        netloc = parsed.netloc.lower()
        path = parsed.path
        query = parse_qs(parsed.query)

        if "youtu.be" in netloc:
            video_id = path.strip("/").split("/")[0]
            return f"youtube.com/watch?v={video_id}" if video_id else "youtube.com"

        if "youtube.com" in netloc:
            if "v" in query:
                return f"youtube.com/watch?v={query['v'][0]}"
            if path.startswith("/embed/"):
                return f"youtube.com/watch?v={path.split('/embed/')[-1].split('?')[0]}"
            if path.startswith("/shorts/"):
                return f"youtube.com/watch?v={path.split('/shorts/')[-1].split('?')[0]}"
            if path and path != "/":
                return f"youtube.com/watch?v={path.strip('/')}"
            return "youtube.com"

        if "twitter.com" in netloc or "x.com" in netloc:
            match = re.search(r"/status/(\d+)", value)
            if match:
                return f"twitter.com/i/status/{match.group(1)}"

        cleaned_query = _strip_tracking_params(query)
        normalized_query = urlencode({key: values[0] for key, values in cleaned_query.items()})
        normalized = urlunparse((parsed.scheme, netloc, path, "", normalized_query, ""))
        return normalized.strip("/")

    def detect_duplicate_videos(self, candidates: Iterable[Any]) -> dict[str, list[Any]]:
        grouped: dict[str, list[Any]] = {}
        for candidate in candidates:
            url = _extract_url(candidate)
            key = self.normalize_url(url)
            if not key:
                key = "__empty__"
            grouped.setdefault(key, []).append(candidate)
        return grouped

    def remove_duplicates(self, candidates: Iterable[Any]) -> list[Any]:
        grouped = self.detect_duplicate_videos(candidates)
        unique: list[Any] = []

        for key, items in grouped.items():
            if key == "__empty__":
                unique.extend(items)
                continue

            if len(items) == 1:
                unique.append(items[0])
                continue

            best = max(items, key=_extract_score)
            unique.append(best)

        return unique


def _extract_url(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return str(
            candidate.get("url")
            or candidate.get("media_url")
            or candidate.get("tweet_url")
            or ""
        )
    return str(getattr(candidate, "url", "") or "")


def _extract_score(candidate: Any) -> float:
    if isinstance(candidate, Mapping):
        score = candidate.get("score")
        if score is None:
            score = candidate.get("final_score")
    else:
        score = getattr(candidate, "score", None)

    try:
        return float(score)
    except Exception:
        return 0.0


def _strip_tracking_params(query: Mapping[str, list[str]]) -> dict[str, list[str]]:
    cleaned: dict[str, list[str]] = {}
    for key, values in query.items():
        key_lower = key.lower()
        if key_lower.startswith("utm_"):
            continue
        if key_lower in {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid"}:
            continue
        cleaned[key] = values
    return cleaned
