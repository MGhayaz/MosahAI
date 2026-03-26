"""Image download utility for MosahAI media intelligence."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from mosahai.logger import setup_logger
from mosahai.media_intelligence.batch_registry import BatchMediaRegistry


LOGGER = setup_logger("mosahai.image_downloader")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(slots=True)
class ImageDownloader:
    request_timeout_seconds: int = 10
    retries: int = 1
    retry_backoff_seconds: float = 1.0
    batch_registry: BatchMediaRegistry | None = field(default=None)
    seen_urls: set[str] = field(default_factory=set)

    def download_image(
        self,
        url: str,
        batch_id: str,
        news_id: str,
        output_dir: str,
        filename: str,
    ) -> Optional[str]:
        safe_url = str(url or "").strip()
        if not safe_url:
            return None

        seen_key = _normalize_seen_url(safe_url)
        if seen_key in self.seen_urls:
            return None

        if self.batch_registry is None:
            self.batch_registry = BatchMediaRegistry()

        if self.batch_registry.prevent_duplicate_usage(safe_url):
            return None

        target_dir = str(output_dir or "").strip()
        if not target_dir:
            return None
        os.makedirs(target_dir, exist_ok=True)

        attempt = 0
        while attempt <= self.retries:
            attempt += 1
            try:
                response = requests.get(
                    safe_url,
                    timeout=self.request_timeout_seconds,
                    headers=HEADERS,
                )
                response.raise_for_status()
                if not _is_valid_image_response(response):
                    return None

                content = response.content or b""
                if not content:
                    return None

                extension = get_extension(response.headers.get("Content-Type", ""))
                output_path = _build_output_path(target_dir, filename, extension)

                with open(output_path, "wb") as handle:
                    handle.write(content)

                self.seen_urls.add(seen_key)
                if self.batch_registry:
                    self.batch_registry.register_media(
                        batch_id=batch_id,
                        news_id=news_id,
                        source="image",
                        url=safe_url,
                        local_file_path=output_path,
                    )
                return output_path
            except requests.Timeout:
                if attempt <= self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                continue
            except requests.RequestException:
                if attempt <= self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                continue
            except OSError:
                if attempt <= self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                continue

        LOGGER.warning("Image download failed after retries. url=%s", safe_url)
        return None


def get_extension(content_type: str) -> str:
    value = str(content_type or "").strip().lower()
    if "png" in value:
        return ".png"
    if "webp" in value:
        return ".webp"
    if "jpeg" in value or "jpg" in value:
        return ".jpg"
    return ".jpg"


def _is_valid_image_response(response: requests.Response) -> bool:
    content_type = str(response.headers.get("Content-Type", "") or "").strip().lower()
    if not content_type:
        return True
    if content_type.startswith("image/"):
        return True
    if "octet-stream" in content_type:
        return True
    return False


def _normalize_seen_url(url: str) -> str:
    return str(url or "").strip().lower()


def _build_output_path(target_dir: str, filename: str, extension: str) -> str:
    safe_name = os.path.basename(str(filename or "image_1.jpg"))
    desired_stem, _ = os.path.splitext(safe_name)
    if not desired_stem:
        desired_stem = "image_1"

    output_name = f"{desired_stem}{extension}"
    output_path = os.path.join(target_dir, output_name)
    if not os.path.exists(output_path):
        return output_path
    return os.path.join(target_dir, _next_image_filename(target_dir, desired_stem, extension))


def _next_image_filename(target_dir: str, desired_stem: str, extension: str) -> str:
    stem = str(desired_stem or "").strip() or "image_1"
    match = re.match(r"^(image_)(\d+)(.*)$", stem, flags=re.IGNORECASE)

    if match:
        prefix = match.group(1)
        index = int(match.group(2))
        suffix = match.group(3)
    else:
        prefix = ""
        index = 0
        suffix = stem

    while True:
        index += 1
        if prefix:
            filename = f"{prefix}{index}{suffix}{extension}"
        else:
            filename = f"{suffix}_{index}{extension}"
        if not os.path.exists(os.path.join(target_dir, filename)):
            return filename
