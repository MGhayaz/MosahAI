"""Video download utility for MosahAI media intelligence."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from mosahai.logger import setup_logger
from mosahai.media_intelligence.logger import MediaEngineLogger
from mosahai.media_intelligence.batch_registry import BatchMediaRegistry


LOGGER = setup_logger("mosahai.video_downloader", level=logging.INFO)


@dataclass(slots=True)
class VideoDownloader:
    base_dir: str = "batches"
    max_duration_seconds: int = 240
    min_height: int = 720
    preferred_ext: str = "mp4"
    retries: int = 0
    retry_backoff_seconds: float = 1.5
    request_timeout_seconds: int = 180
    media_logger: MediaEngineLogger | None = None
    batch_registry: BatchMediaRegistry | None = field(default=None)

    def download_video(
        self,
        url: str,
        batch_id: str,
        news_id: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> Optional[str]:
        safe_url = str(url or "").strip()
        if not safe_url:
            LOGGER.warning("Download skipped: empty url.")
            return None

        if self.batch_registry is None:
            self.batch_registry = BatchMediaRegistry()

        if self.batch_registry.prevent_duplicate_usage(safe_url):
            LOGGER.info("Download skipped: duplicate media. url=%s", safe_url)
            return None

        safe_batch = _safe_segment(batch_id, fallback="BATCH_UNKNOWN")
        safe_news = _normalize_news_id(news_id)

        if output_dir:
            target_dir = str(output_dir).strip()
        else:
            target_dir = os.path.join(self.base_dir, safe_batch, "videos", safe_news)
        os.makedirs(target_dir, exist_ok=True)

        if filename:
            safe_name = os.path.basename(str(filename))
            output_path = os.path.join(target_dir, safe_name)
            if os.path.exists(output_path):
                output_path = os.path.join(target_dir, self._next_filename(target_dir))
        else:
            output_path = os.path.join(target_dir, self._next_filename(target_dir))
        metadata = self._fetch_metadata(safe_url)
        command = self._build_command(safe_url, output_path)

        LOGGER.info("Downloading video (single attempt). url=%s output=%s", safe_url, output_path)
        try:
            print(f"[DEBUG][YT] Running command: {command}")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
            print(f"[DEBUG][YT] Return code: {completed.returncode}")
            print(f"[DEBUG][YT] STDERR: {(completed.stderr or '')[:300]}")
        except FileNotFoundError:
            LOGGER.error("yt-dlp not found. Install yt-dlp and retry.")
            return None
        except subprocess.TimeoutExpired:
            LOGGER.warning("yt-dlp download timed out. url=%s", safe_url)
            return None
        except Exception as exc:
            LOGGER.warning("yt-dlp download failed. url=%s error=%s", safe_url, exc)
            return None

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            LOGGER.warning(
                "yt-dlp download skipped after failure. url=%s returncode=%s stderr=%s",
                safe_url,
                completed.returncode,
                stderr[:500],
            )
            return None

        LOGGER.info("Download completed. file=%s", output_path)
        print(f"[DEBUG][YT] Return code: {completed.returncode}")
        print(f"[DEBUG][YT] Output exists: {os.path.exists(output_path)}")
        if self.media_logger:
            self.media_logger.video_downloaded(
                batch_id=safe_batch,
                news_id=safe_news,
                url=safe_url,
                output_path=output_path,
            )
        if self.batch_registry:
            self.batch_registry.register_media(
                batch_id=safe_batch,
                news_id=safe_news,
                source=_infer_source(safe_url),
                url=safe_url,
                local_file_path=output_path,
            )
        self._write_metadata(
            output_path=output_path,
            url=safe_url,
            source=_infer_source(safe_url),
            metadata=metadata,
        )
        return output_path

    def _build_command(self, url: str, output_path: str) -> list[str]:
        return [
            "yt-dlp",
            "--no-playlist",
            "--quiet",
            "--extractor-args",
            "youtube:player_client=web",
            "-f",
            "best[height<=720]",
            "-o",
            output_path,
            url,
        ]

    def _fetch_metadata(self, url: str) -> dict[str, Optional[int | str]]:
        command = [
            "yt-dlp",
            "--dump-json",
            "--skip-download",
            "--quiet",
            "--extractor-args",
            "youtube:player_client=web",
            url,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=35,
                check=False
            )
        except Exception:
            return {}

        if completed.returncode != 0:
            return {}

        payload = {}
        for line in (completed.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue

        if not payload:
            return {}

        width = _safe_int(payload.get("width"))
        height = _safe_int(payload.get("height"))
        resolution = None
        if width and height:
            resolution = f"{width}x{height}"

        return {
            "duration": _safe_int(payload.get("duration")),
            "resolution": resolution,
            "width": width,
            "height": height,
        }

    def _write_metadata(
        self,
        *,
        output_path: str,
        url: str,
        source: str,
        metadata: dict[str, Optional[int | str]],
    ) -> None:
        if not output_path:
            return

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = {
            "source": source,
            "url": url,
            "duration": metadata.get("duration"),
            "resolution": metadata.get("resolution"),
            "downloaded_at": timestamp,
        }

        metadata_path = os.path.splitext(output_path)[0] + ".json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception as exc:
            LOGGER.warning("Failed to write metadata. file=%s error=%s", metadata_path, exc)

    def _next_filename(self, target_dir: str) -> str:
        existing = [
            name
            for name in os.listdir(target_dir)
            if name.lower().startswith("video_") and name.lower().endswith(".mp4")
        ]
        index = len(existing) + 1
        while True:
            filename = f"video_{index}.mp4"
            if not os.path.exists(os.path.join(target_dir, filename)):
                return filename
            index += 1

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt <= self.retries:
            time.sleep(self.retry_backoff_seconds * attempt)


def _safe_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned or fallback


def _normalize_news_id(news_id: str) -> str:
    cleaned = _safe_segment(news_id, fallback="news_1")
    if cleaned.isdigit():
        return f"news_{cleaned}"
    if not cleaned.lower().startswith("news_"):
        return f"news_{cleaned}"
    return cleaned


def _infer_source(url: str) -> str:
    value = (url or "").lower()
    if "youtu" in value:
        return "youtube"
    if "twitter.com" in value or "x.com" in value:
        return "twitter"
    return "article"


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
