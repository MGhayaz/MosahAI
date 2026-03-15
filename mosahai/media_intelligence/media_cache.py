"""Media search cache for MosahAI."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence


@dataclass(slots=True)
class MediaSearchCache:
    db_path: str = "media_cache.db"
    ttl_hours: float = 6.0

    def check_cache(self, news_title: str) -> list[dict[str, Any]] | None:
        title = (news_title or "").strip()
        if not title:
            return None

        cache_key = self._hash_title(title)
        with self._connect() as conn:
            self._ensure_table(conn)
            row = conn.execute(
                "SELECT payload, created_at FROM media_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

            if not row:
                return None

            payload_text, created_at_text = row
            created_at = _parse_datetime(created_at_text)
            if not created_at:
                self._delete_key(conn, cache_key)
                return None

            if self._is_expired(created_at):
                self._delete_key(conn, cache_key)
                return None

            try:
                payload = json.loads(payload_text)
            except Exception:
                self._delete_key(conn, cache_key)
                return None

            if isinstance(payload, list):
                return payload
            return None

    def store_results(self, news_title: str, results: Sequence[dict[str, Any]]) -> None:
        title = (news_title or "").strip()
        if not title:
            return

        cache_key = self._hash_title(title)
        payload_text = json.dumps(list(results), ensure_ascii=True)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            self._ensure_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO media_cache (cache_key, payload, created_at) VALUES (?, ?, ?)",
                (cache_key, payload_text, created_at),
            )
            conn.commit()

    def _hash_title(self, title: str) -> str:
        normalized = title.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _is_expired(self, created_at: datetime) -> bool:
        ttl = timedelta(hours=float(self.ttl_hours))
        return datetime.now(timezone.utc) - created_at > ttl

    def _connect(self) -> sqlite3.Connection:
        db_path = self.db_path
        if db_path and not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        return sqlite3.connect(db_path)

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS media_cache ("
            "cache_key TEXT PRIMARY KEY,"
            "payload TEXT NOT NULL,"
            "created_at TEXT NOT NULL"
            ")"
        )

    def _delete_key(self, conn: sqlite3.Connection, cache_key: str) -> None:
        conn.execute("DELETE FROM media_cache WHERE cache_key = ?", (cache_key,))
        conn.commit()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
