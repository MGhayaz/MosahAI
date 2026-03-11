import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone

import feedparser

from config import DEFAULT_DB_PATH, RSS_BASE_URL_TEMPLATE


class MosahAIBrain:
    """RSS intake, ranking, and JSON storage for Phase 1.5."""

    def __init__(self, db_path=DEFAULT_DB_PATH, rss_template=RSS_BASE_URL_TEMPLATE):
        self.db_path = db_path
        self.rss_template = rss_template
        self.encoding = "utf-8"
        self._ensure_db_file()

    def _ensure_db_file(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding=self.encoding) as handle:
                json.dump([], handle, ensure_ascii=False, indent=2)
            return

        try:
            with open(self.db_path, "r", encoding=self.encoding) as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("JSON root must be a list.")
        except Exception:
            backup_path = f"{self.db_path}.broken_{int(time.time())}"
            try:
                os.replace(self.db_path, backup_path)
            except Exception:
                pass
            with open(self.db_path, "w", encoding=self.encoding) as handle:
                json.dump([], handle, ensure_ascii=False, indent=2)

    def _load_db(self):
        with open(self.db_path, "r", encoding=self.encoding) as handle:
            return json.load(handle)

    def _save_db_atomic(self, data):
        temp_path = f"{self.db_path}.tmp"
        with open(temp_path, "w", encoding=self.encoding) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.db_path)

    def _is_within_last_24h(self, parsed_time):
        if not parsed_time:
            return False
        published_dt = datetime(
            parsed_time.tm_year,
            parsed_time.tm_mon,
            parsed_time.tm_mday,
            parsed_time.tm_hour,
            parsed_time.tm_min,
            parsed_time.tm_sec,
            tzinfo=timezone.utc,
        )
        age = datetime.now(timezone.utc) - published_dt
        return timedelta(0) <= age <= timedelta(hours=24)

    def fetch_news(self, topic, language_code, limit=6):
        topic_query = str(topic).strip().replace(" ", "+")
        lang = str(language_code).strip()
        url = self.rss_template.format(topic=topic_query, lang=lang)
        feed = feedparser.parse(url)

        items = []
        for entry in feed.entries:
            parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
            if not self._is_within_last_24h(parsed_time):
                continue

            source_name = ""
            source_obj = entry.get("source")
            if isinstance(source_obj, dict):
                source_name = source_obj.get("title", "") or ""
            elif source_obj is not None:
                source_name = getattr(source_obj, "title", "") or ""

            items.append(
                {
                    "title": str(entry.get("title", "")).strip(),
                    "summary": str(entry.get("summary", "") or entry.get("description", "")).strip(),
                    "link": str(entry.get("link", "")).strip(),
                    "published": str(entry.get("published", "")).strip(),
                    "source": source_name.strip(),
                }
            )

            if len(items) >= int(limit):
                break

        return items

    def score_news(self, title):
        text = " ".join(str(title).strip().lower().split())
        score = 50

        high_impact = ["alert", "massive", "update", "crime", "political", "tech breakout", "breaking"]
        low_impact = ["company announces", "quarterly earnings", "ceo says", "corporate partnership"]

        for keyword in high_impact:
            if keyword in text:
                score += 18
        for keyword in low_impact:
            if keyword in text:
                score -= 20

        if len(text) < 40:
            score += 4
        if len(text) > 140:
            score -= 8

        return max(0, score)

    def select_top_news(self, items, count=3):
        ranked = sorted(items, key=lambda item: self.score_news(item.get("title", "")), reverse=True)
        return ranked[: int(count)]

    def generate_short_id(self, topic, titles):
        seed = "|".join([str(topic)] + [str(title) for title in titles]).encode(self.encoding)
        digest = hashlib.md5(seed).hexdigest()[:8]
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        topic_tag = "".join([ch for ch in str(topic).upper() if ch.isalnum()])[:4] or "NEWS"
        return f"{topic_tag}_BATCH_{date_tag}_{digest}"

    def batch_exists(self, short_id):
        short_id = str(short_id).strip()
        if not short_id:
            return False
        for item in self._load_db():
            if str(item.get("short_id", "")).strip() == short_id:
                return True
        return False

    def save_batch(self, record):
        record["status"] = "pending_review"
        db_data = self._load_db()

        if self.batch_exists(record.get("short_id", "")):
            return False

        db_data.append(record)
        self._save_db_atomic(db_data)
        return True
