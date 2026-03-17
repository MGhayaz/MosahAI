"""Image download utility for MosahAI media intelligence."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from mosahai.logger import setup_logger
from mosahai.media_intelligence.batch_registry import BatchMediaRegistry


LOGGER = setup_logger("mosahai.image_downloader")


@dataclass(slots=True)
class ImageDownloader:
    request_timeout_seconds: int = 12
    retries: int = 1
    retry_backoff_seconds: float = 1.0
    batch_registry: BatchMediaRegistry | None = field(default=None)

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
            LOGGER.warning("Image download skipped: empty url.")
            return None

        if self.batch_registry is None:
            self.batch_registry = BatchMediaRegistry()

        if self.batch_registry.prevent_duplicate_usage(safe_url):
            LOGGER.info("Image download skipped: duplicate media. url=%s", safe_url)
            return None

        target_dir = str(output_dir or "").strip()
        if not target_dir:
            return None
        os.makedirs(target_dir, exist_ok=True)

        safe_name = os.path.basename(str(filename or "image_1.jpg"))
        output_path = os.path.join(target_dir, safe_name)
        if os.path.exists(output_path):
            output_path = os.path.join(target_dir, _next_image_filename(target_dir))

        attempt = 0
        while attempt <= self.retries:
            attempt += 1
            try:
                response = requests.get(
                    safe_url,
                    timeout=self.request_timeout_seconds,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; MosahAI/4.0)"},
                )
                response.raise_for_status()
                with open(output_path, "wb") as handle:
                    handle.write(response.content)
                if self.batch_registry:
                    self.batch_registry.register_media(
                        batch_id=batch_id,
                        news_id=news_id,
                        source="image",
                        url=safe_url,
                        local_file_path=output_path,
                    )
                return output_path
            except Exception:
                if attempt <= self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                continue

        LOGGER.warning("Image download failed after retries. url=%s", safe_url)
        return None


def _next_image_filename(target_dir: str) -> str:
    existing = [
        name
        for name in os.listdir(target_dir)
        if name.lower().startswith("image_") and name.lower().endswith(".jpg")
    ]
    index = len(existing) + 1
    while True:
        filename = f"image_{index}.jpg"
        if not os.path.exists(os.path.join(target_dir, filename)):
            return filename
        index += 1
