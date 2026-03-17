import json
import os
import shutil
from datetime import datetime, timedelta


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_ROOT = os.path.join(BASE_DIR, "assets")
BATCHES_DIR = os.path.join(ASSETS_ROOT, "batches")
ARCHIVE_DIR = os.path.join(ASSETS_ROOT, "archive")


def ensure_assets_structure():
    os.makedirs(BATCHES_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)


def create_batch_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    return f"BATCH_{timestamp}"


def create_unique_batch_id() -> str:
    ensure_assets_structure()
    base = create_batch_id()
    batch_id = base
    counter = 1
    while os.path.exists(os.path.join(BATCHES_DIR, batch_id)):
        batch_id = f"{base}_{counter:02d}"
        counter += 1
    return batch_id


def _parse_batch_timestamp(batch_id: str) -> datetime | None:
    parts = str(batch_id or "").split("_")
    if len(parts) < 3 or parts[0] != "BATCH":
        return None
    date_part = parts[1]
    time_part = parts[2]
    try:
        return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M")
    except Exception:
        return None


def archive_old_batches(days_to_archive: int = 7, days_to_delete: int = 15):
    ensure_assets_structure()
    now = datetime.now()
    archive_cutoff = now - timedelta(days=int(days_to_archive))
    delete_cutoff = now - timedelta(days=int(days_to_delete))

    def _delete_path(path: str):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _safe_move(src: str, dest: str):
        if os.path.exists(dest):
            counter = 1
            base = dest
            while os.path.exists(dest):
                dest = f"{base}_{counter:02d}"
                counter += 1
        shutil.move(src, dest)

    for entry in os.scandir(BATCHES_DIR):
        if not entry.is_dir():
            continue
        batch_time = _parse_batch_timestamp(entry.name)
        if not batch_time:
            continue
        if batch_time <= delete_cutoff:
            _delete_path(entry.path)
        elif batch_time <= archive_cutoff:
            dest = os.path.join(ARCHIVE_DIR, entry.name)
            try:
                _safe_move(entry.path, dest)
            except Exception:
                pass

    for entry in os.scandir(ARCHIVE_DIR):
        if not entry.is_dir():
            continue
        batch_time = _parse_batch_timestamp(entry.name)
        if not batch_time:
            continue
        if batch_time <= delete_cutoff:
            _delete_path(entry.path)


def build_batch_layout(batch_id: str, news_count: int = 3) -> list[dict[str, str]]:
    ensure_assets_structure()
    batch_root = os.path.join(BATCHES_DIR, batch_id)
    os.makedirs(batch_root, exist_ok=True)

    layout = []
    for index in range(1, int(news_count) + 1):
        news_id = f"NEWS_{index}"
        news_root = os.path.join(batch_root, news_id)
        script_dir = os.path.join(news_root, "script")
        media_dir = os.path.join(news_root, "media")
        metadata_dir = os.path.join(news_root, "metadata")
        os.makedirs(script_dir, exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        layout.append(
            {
                "news_id": news_id,
                "news_root": news_root,
                "script_dir": script_dir,
                "media_dir": media_dir,
                "metadata_dir": metadata_dir,
                "script_path": os.path.join(script_dir, "script.json"),
                "metadata_path": os.path.join(metadata_dir, "metadata.json"),
            }
        )
    return layout


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
