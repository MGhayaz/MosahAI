"""Validate and score image candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

if TYPE_CHECKING:
    from .image_pipeline import ImageCandidate


LOGGER = logging.getLogger("mosahai.image_pipeline.image_validator")

_POSITIVE_TOKENS = ("news", "image", "photo")
_NEGATIVE_TOKENS = ("logo", "thumbnail", "small", "lowres")
_TRACKING_KEYS = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "cmpid"}


@dataclass(slots=True)
class ImageValidator:
    """Validator for cleaning and lightly scoring image candidates."""

    def validate(self, candidates: Sequence["ImageCandidate"]) -> list["ImageCandidate"]:
        """Return cleaned image candidates with updated quality scores."""
        total = len(candidates or [])
        if not candidates:
            LOGGER.info("Image validation: total candidates=0 cleaned=0")
            _print_validator_debug(accepted=0, rejected=0)
            return []

        validated: list["ImageCandidate"] = []
        seen: set[str] = set()
        accepted_count = 0
        rejected_count = 0

        for candidate in candidates:
            raw_url = str(getattr(candidate, "url", "") or "").strip()
            if not raw_url:
                rejected_count += 1
                continue

            cleaned_url = clean_url(raw_url)
            if not cleaned_url:
                rejected_count += 1
                continue

            lower_url = cleaned_url.lower()
            if lower_url in seen:
                rejected_count += 1
                continue
            if _looks_like_logo_filename(cleaned_url):
                rejected_count += 1
                continue
            if _is_base64_image(lower_url):
                rejected_count += 1
                continue
            if _is_svg_image(lower_url):
                rejected_count += 1
                continue

            seen.add(lower_url)

            domain = _extract_domain(cleaned_url)
            if domain:
                try:
                    candidate.domain = domain
                except Exception:
                    pass

            score = _score_candidate(cleaned_url, domain or "")
            try:
                candidate.quality_score = float(score)
            except Exception:
                pass

            try:
                candidate.url = cleaned_url
            except Exception:
                pass

            validated.append(candidate)
            accepted_count += 1

        _print_validator_debug(
            accepted=accepted_count,
            rejected=rejected_count,
        )

        LOGGER.info(
            "Image validation: total candidates=%s cleaned=%s",
            total,
            len(validated),
        )
        return validated


def clean_url(url: str) -> str:
    """Normalize URLs and strip common tracking parameters."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw

    parsed = urlparse(raw)
    if not parsed.scheme:
        return raw
    if parsed.scheme not in {"http", "https", "data"}:
        return raw
    if parsed.scheme == "data":
        return raw

    filtered_query = _strip_tracking_params(parsed.query)
    sanitized = parsed._replace(query=filtered_query, fragment="")
    return urlunparse(sanitized).strip()


def _strip_tracking_params(query: str) -> str:
    if not query:
        return ""
    params = parse_qs(query, keep_blank_values=True)
    filtered: dict[str, list[str]] = {}
    for key, values in params.items():
        lower_key = key.lower()
        if lower_key in _TRACKING_KEYS or lower_key.startswith("utm_"):
            continue
        filtered[key] = values
    return urlencode(filtered, doseq=True) if filtered else ""


def _score_candidate(url: str, domain: str) -> float:
    del domain
    score = 1.0
    url_key = url.lower()
    if _contains_any(url_key, _POSITIVE_TOKENS):
        score += 2.0
    if _contains_any(url_key, _NEGATIVE_TOKENS):
        score -= 1.0
    return score


def _contains_any(value: str, tokens: Iterable[str]) -> bool:
    return any(token in value for token in tokens)


def _is_base64_image(url: str) -> bool:
    return url.startswith("data:image")


def _is_svg_image(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".svg"):
        return True
    return "image/svg" in url.lower()


def _looks_like_logo_filename(url: str) -> bool:
    filename = urlparse(url).path.lower().split("/")[-1]
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


def _print_validator_debug(*, accepted: int, rejected: int) -> None:
    print(f"[VALIDATOR] Accepted: {int(accepted)}")
    print(f"[VALIDATOR] Rejected: {int(rejected)}")
