"""Inspect downloaded media for a MosahAI batch."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from mosahai.media_intelligence.batch_registry import BatchMediaRegistry


@dataclass(slots=True)
class MediaEntry:
    file_path: str
    duration: int | None = None
    resolution: str | None = None
    source: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="View downloaded media for a batch.")
    parser.add_argument("batch_id", help="Batch identifier (e.g. BATCH_20260314_AI_001)")
    parser.add_argument(
        "--base-dir",
        default="batches",
        help="Base batches directory (default: batches)",
    )
    parser.add_argument(
        "--registry-db",
        default="batch_registry.db",
        help="Batch registry database path (default: batch_registry.db)",
    )

    args = parser.parse_args()
    batch_id = str(args.batch_id)

    print(f"Batch: {batch_id}\n")

    registry = BatchMediaRegistry(db_path=args.registry_db)
    registry_entries = registry.get_batch_media(batch_id)
    registry_by_file = {
        os.path.normpath(entry.get("local_file", "")): entry for entry in registry_entries
    }

    batch_dir = os.path.join(args.base_dir, batch_id)
    videos_dir = os.path.join(batch_dir, "videos")

    news_dirs = _discover_news_dirs(videos_dir)
    registry_news = _group_registry_by_news(registry_entries)
    news_ids = _merge_news_ids(news_dirs, registry_news)

    if not news_ids:
        print("No media found.")
        return 0

    for news_id in news_ids:
        print(f"News {news_id.split('_')[-1] if news_id.startswith('news_') else news_id}:")
        entries = _collect_news_entries(
            videos_dir, news_id, registry_by_file, registry_news.get(news_id, [])
        )
        if not entries:
            print("fallback: image_required\n")
            continue

        for entry in entries:
            print(f"* {os.path.basename(entry.file_path)}")
            if entry.duration is not None:
                print(f"* duration: {entry.duration}s")
            if entry.resolution:
                print(f"* resolution: {entry.resolution}")
            if entry.source:
                print(f"* source: {entry.source}")
            print("")

    return 0


def _discover_news_dirs(videos_dir: str) -> list[str]:
    if not os.path.isdir(videos_dir):
        return []
    entries = []
    for name in os.listdir(videos_dir):
        path = os.path.join(videos_dir, name)
        if os.path.isdir(path) and name.lower().startswith("news"):
            entries.append(name)
    return _sort_news_ids(entries)


def _merge_news_ids(news_dirs: list[str], registry_news: dict[str, list[dict[str, Any]]]) -> list[str]:
    merged = set(news_dirs)
    merged.update(registry_news.keys())
    return _sort_news_ids(list(merged))


def _sort_news_ids(news_ids: list[str]) -> list[str]:
    def _key(value: str) -> tuple[int, str]:
        if value.lower().startswith("news_"):
            suffix = value.split("_", 1)[-1]
            if suffix.isdigit():
                return (int(suffix), value)
        return (9999, value)

    return sorted(news_ids, key=_key)


def _group_registry_by_news(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        news_id = entry.get("news_id") or ""
        if news_id:
            grouped.setdefault(news_id, []).append(entry)
    return grouped


def _collect_news_entries(
    videos_dir: str,
    news_id: str,
    registry_by_file: dict[str, dict[str, Any]],
    registry_entries: list[dict[str, Any]],
) -> list[MediaEntry]:
    entries: list[MediaEntry] = []
    news_dir = os.path.join(videos_dir, news_id)

    if os.path.isdir(news_dir):
        for name in sorted(os.listdir(news_dir)):
            if not name.lower().endswith(".mp4"):
                continue
            file_path = os.path.join(news_dir, name)
            metadata = _load_metadata(file_path)
            source = metadata.get("source")
            duration = metadata.get("duration")
            resolution = metadata.get("resolution")

            if not source:
                registry_entry = registry_by_file.get(os.path.normpath(file_path))
                if registry_entry:
                    source = registry_entry.get("source")

            entries.append(
                MediaEntry(
                    file_path=file_path,
                    duration=_safe_int(duration),
                    resolution=resolution,
                    source=source,
                )
            )

    if not entries and registry_entries:
        for entry in registry_entries:
            local_file = entry.get("local_file")
            if not local_file:
                continue
            metadata = _load_metadata(local_file)
            entries.append(
                MediaEntry(
                    file_path=local_file,
                    duration=_safe_int(metadata.get("duration")),
                    resolution=metadata.get("resolution"),
                    source=entry.get("source"),
                )
            )

    return entries


def _load_metadata(video_path: str) -> dict[str, Any]:
    metadata_path = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
