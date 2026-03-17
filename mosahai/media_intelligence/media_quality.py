"""Media quality analyzer for MosahAI."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class MediaQualityAnalyzer:
    min_resolution: int = 720
    min_duration_seconds: int = 8
    max_duration_seconds: int = 240
    preferred_aspect_ratios: tuple[float, ...] = (16 / 9, 9 / 16, 1.0)
    aspect_ratio_tolerance: float = 0.25
    extreme_aspect_ratio_bounds: tuple[float, float] = (0.5, 2.5)

    def evaluate_quality(self, candidate: Any) -> dict[str, Any]:
        width, height = _extract_dimensions(candidate)
        duration = _extract_duration(candidate)
        fps = _extract_fps(candidate)

        reject = False

        url = _extract_media_url(candidate)
        if _is_image_candidate(candidate, url) and _looks_like_logo(url):
            reject = True

        if duration is not None:
            if duration < self.min_duration_seconds or duration > self.max_duration_seconds:
                reject = True

        if width and height:
            if min(width, height) < self.min_resolution:
                reject = True

        aspect_score = 0.5
        if width and height:
            ratio = width / height
            if ratio < self.extreme_aspect_ratio_bounds[0] or ratio > self.extreme_aspect_ratio_bounds[1]:
                reject = True
            aspect_score = _aspect_score(ratio, self.preferred_aspect_ratios, self.aspect_ratio_tolerance)

        resolution_score = _resolution_score(width, height, self.min_resolution)
        duration_score = _duration_score(duration, self.max_duration_seconds)
        fps_score = _fps_score(fps)

        quality_score = (
            0.4 * resolution_score + 0.25 * duration_score + 0.25 * aspect_score + 0.1 * fps_score
        )

        quality_score = max(0.0, min(1.0, quality_score))

        return {
            "quality_score": round(quality_score, 4),
            "reject": bool(reject),
        }


def _extract_dimensions(candidate: Any) -> tuple[int | None, int | None]:
    width = _extract_int(candidate, "width")
    height = _extract_int(candidate, "height")

    if width and height:
        return width, height

    resolution = _extract_value(candidate, "resolution")
    if resolution:
        parsed = _parse_resolution(str(resolution))
        if parsed:
            return parsed

    raw = _extract_value(candidate, "raw")
    if isinstance(raw, Mapping):
        width = width or _safe_int(raw.get("width"))
        height = height or _safe_int(raw.get("height"))
        if width and height:
            return width, height

        fmt = raw.get("resolution")
        if fmt:
            parsed = _parse_resolution(str(fmt))
            if parsed:
                return parsed

    return width, height


def _extract_media_url(candidate: Any) -> str:
    for key in ("url", "media_url", "image_url", "thumbnail_url"):
        value = _extract_value(candidate, key)
        if value:
            return str(value)

    raw = _extract_value(candidate, "raw")
    if isinstance(raw, Mapping):
        for key in ("url", "media_url", "image_url", "thumbnail_url"):
            value = raw.get(key)
            if value:
                return str(value)
    return ""


def _is_image_candidate(candidate: Any, url: str) -> bool:
    source = _extract_value(candidate, "source")
    if source and str(source).strip().lower() == "image":
        return True

    raw = _extract_value(candidate, "raw")
    if isinstance(raw, Mapping):
        media_type = raw.get("media_type")
        if isinstance(media_type, str) and media_type.lower() == "image":
            return True

    lower_url = (url or "").lower()
    if lower_url.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
        return True
    return False


def _looks_like_logo(url: str) -> bool:
    if not url:
        return False
    lower_url = url.lower()
    filename = os.path.basename(lower_url)
    for token in ("logo", "icon", "favicon"):
        if token in lower_url or token in filename:
            return True
    return False


def _extract_duration(candidate: Any) -> int | None:
    value = _extract_value(candidate, "duration_seconds")
    if value is None:
        value = _extract_value(candidate, "duration")
    return _safe_int(value)


def _extract_fps(candidate: Any) -> float | None:
    value = _extract_value(candidate, "fps")
    if value is None:
        value = _extract_value(candidate, "frame_rate")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _extract_value(candidate: Any, key: str) -> Any:
    if isinstance(candidate, Mapping):
        if key in candidate:
            return candidate.get(key)
    else:
        if hasattr(candidate, key):
            return getattr(candidate, key)
    return None


def _extract_int(candidate: Any, key: str) -> int | None:
    value = _extract_value(candidate, key)
    return _safe_int(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_resolution(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.search(r"(\d{3,4})\s*[xX]\s*(\d{3,4})", value)
    if not match:
        return None
    try:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    except Exception:
        return None


def _resolution_score(width: int | None, height: int | None, min_resolution: int) -> float:
    if not width or not height:
        return 0.5
    min_dim = min(width, height)
    return min(1.0, min_dim / float(min_resolution))


def _duration_score(duration: int | None, max_duration: int) -> float:
    if duration is None:
        return 0.5
    if duration <= 0:
        return 0.0
    return min(1.0, duration / float(max_duration))


def _fps_score(fps: float | None) -> float:
    if fps is None:
        return 0.5
    if fps <= 0:
        return 0.0
    return min(1.0, fps / 30.0)


def _aspect_score(
    ratio: float,
    preferred: tuple[float, ...],
    tolerance: float,
) -> float:
    if ratio <= 0:
        return 0.0
    diffs = []
    for target in preferred:
        if target <= 0:
            continue
        diffs.append(abs(ratio - target) / target)
    if not diffs:
        return 0.5
    best = min(diffs)
    if best >= tolerance:
        return max(0.0, 1.0 - best)
    return 1.0 - (best / tolerance)
