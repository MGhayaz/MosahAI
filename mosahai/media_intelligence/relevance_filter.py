"""News-to-media relevance filter for MosahAI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from torch import threshold

from mosahai.media_intelligence.dedup_engine import _extract_url


@dataclass(slots=True)
class NewsMediaRelevanceFilter:
    similarity_threshold: float = 0.3
    youtube_similarity_threshold: float = 0.25
    model_name: str = "all-MiniLM-L6-v2"
    _model: object | bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.similarity_threshold = float(self.similarity_threshold)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = False

    def compute_similarity(self, news_title: str, media_title: str) -> float:
        news_title = (news_title or "").strip()
        media_title = (media_title or "").strip()
        if not news_title or not media_title:
            return 0.0

        self._load_model()
        if self._model:
            try:
                embeddings = self._model.encode(
                    [news_title, media_title],
                    normalize_embeddings=True,
                )
                return float(embeddings[0] @ embeddings[1])
            except Exception:
                pass

        return _token_overlap_similarity(news_title, media_title)

    def filter_candidates(
        self,
        candidates: Sequence[Any],
        news_title: str,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        resolved_title = (news_title or "").strip()
        if not resolved_title:
            return [
                {
                    "title": _extract_title(candidate),
                    "url": _extract_url(candidate),
                    "score": 1.0,
                }
                for candidate in candidates
            ]

        filtered: list[dict[str, Any]] = []
        seen: set[str] = set()

        for candidate in candidates:
            title = _extract_title(candidate)
            url = _extract_url(candidate)
            if not title:
                continue

            canonical = _canonicalize_url(url)
            if canonical and canonical in seen:
                continue
            if canonical:
                seen.add(canonical)

            source = _extract_source(candidate)
            threshold = self.similarity_threshold
            if source == "youtube":
                threshold = self.youtube_similarity_threshold

            similarity = self.compute_similarity(resolved_title, title)
            # allow slightly weak matches (soft filter)
            if similarity < threshold:
                # if it's YouTube, allow borderline matches
                if source == "youtube" and similarity >= (threshold - 0.15):
                    pass
                else:
                    continue
            
            filtered.append(
                {
                    "title": title,
                    "url": url,
                    "score": round(float(similarity), 4),
                }
            )
        # fallback: agar kuch bhi pass nahi hua, best candidate allow karo
            if not filtered and candidates:
                best = None
                try:
                    best = max(
                        candidates,
                        key=lambda c: self.compute_similarity(resolved_title, _extract_title(c)),
                    )
                except Exception:
                    pass
                if best:
                    filtered.append({
                    "title": _extract_title(best),
                    "url": _extract_url(best),
                    "score": round(self.compute_similarity(resolved_title, _extract_title(best)), 4),
                    })
            print("[DEBUG][FILTER] Fallback activated — best candidate selected")

            print(f"[DEBUG][FILTER] Final selected: {len(filtered)}")
            return filtered


def _extract_source(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return str(candidate.get("source") or "").strip().lower()
    return str(getattr(candidate, "source", "") or "").strip().lower()

def _extract_title(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return str(candidate.get("title") or candidate.get("text") or "").strip()
    return str(getattr(candidate, "title", "") or "").strip()


def _extract_url(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return str(
            candidate.get("url")
            or candidate.get("media_url")
            or candidate.get("tweet_url")
            or ""
        ).strip()
    return str(getattr(candidate, "url", "") or "").strip()


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens.intersection(right_tokens)
    return len(overlap) / max(len(left_tokens), len(right_tokens))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower() or parsed.path.split("/")[0].lower()
    path = parsed.path
    query = parse_qs(parsed.query)

    if "youtu" in netloc:
        if "v" in query:
            return f"youtube.com/watch?v={query['v'][0]}"
        if parsed.path.startswith("/embed/"):
            return f"youtube.com/watch?v={parsed.path.split('/embed/')[-1]}"
        if parsed.path and parsed.path != "/":
            return f"youtube.com/watch?v={parsed.path.strip('/')}"

    if "twitter.com" in netloc or "x.com" in netloc:
        match = re.search(r"/status/(\d+)", url)
        if match:
            return f"twitter.com/i/status/{match.group(1)}"

    return f"{netloc}{path}"
