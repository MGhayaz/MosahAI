"""Batch media registry for MosahAI."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass(slots=True)
class BatchMediaRegistry:
    db_path: str = "batch_registry.db"

    def register_media(
        self,
        *,
        batch_id: str,
        news_id: str,
        source: str,
        url: str,
        local_file_path: str,
    ) -> bool:
        safe_url = (url or "").strip()
        if not safe_url:
            return False

        normalized = self._normalize_url(safe_url)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            self._ensure_table(conn)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO batch_media "
                    "(batch_id, news_id, source, url, normalized_url, local_file, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(batch_id),
                        str(news_id),
                        str(source),
                        safe_url,
                        normalized,
                        str(local_file_path),
                        created_at,
                    ),
                )
                conn.commit()
            except Exception:
                return False

        return True

    def get_batch_media(self, batch_id: str) -> list[dict[str, Any]]:
        batch_id = str(batch_id or "").strip()
        if not batch_id:
            return []

        with self._connect() as conn:
            self._ensure_table(conn)
            rows = conn.execute(
                "SELECT batch_id, news_id, source, url, local_file FROM batch_media WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()

        return [
            {
                "batch_id": row[0],
                "news_id": row[1],
                "source": row[2],
                "url": row[3],
                "local_file": row[4],
            }
            for row in rows
        ]

    def prevent_duplicate_usage(self, url: str) -> bool:
        safe_url = (url or "").strip()
        if not safe_url:
            return False

        normalized = self._normalize_url(safe_url)
        with self._connect() as conn:
            self._ensure_table(conn)
            row = conn.execute(
                "SELECT 1 FROM batch_media WHERE normalized_url = ? LIMIT 1",
                (normalized,),
            ).fetchone()

        return row is not None

    def _connect(self) -> sqlite3.Connection:
        db_path = self.db_path
        if db_path and not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        return sqlite3.connect(db_path)

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS batch_media ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "batch_id TEXT NOT NULL,"
            "news_id TEXT NOT NULL,"
            "source TEXT NOT NULL,"
            "url TEXT NOT NULL,"
            "normalized_url TEXT NOT NULL,"
            "local_file TEXT NOT NULL,"
            "created_at TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_batch_media_url ON batch_media (normalized_url)"
        )

    def _normalize_url(self, url: str) -> str:
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

        return f"{netloc}{path}"
