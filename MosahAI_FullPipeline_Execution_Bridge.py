import os
from turtle import title
import warnings
import logging

from mosahai.media_intelligence.image_downloader import ImageDownloader
from mosahai.media_intelligence.image_pipeline.title_image_fetcher import fetch_primary_image

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

import time
from datetime import datetime, timezone
import torch
from sentence_transformers import SentenceTransformer

from MultiKey_APIHealth_SQLite_ResilienceTracker import APIHealthTracker, CircuitBreakerOpenError
from ThreeSegment_DynamicScript_Synthesis_Processor import ThreeSegmentDynamicScriptSynthesisProcessor
from config import (
    DEFAULT_DB_PATH,
    ENABLE_VAULT_STORAGE,
    GEMINI_KEYS,
    GEMINI_MODEL,
    PHASE15_NICHES,
    SCRIPT_VAULT_DB_PATH,
    USAGE_TRACKER_DB_PATH,
)
from key_manager import KeyManager
from mosah_ai_engine import MosahAIBrain
from SemanticSearch_NicheHistory_EvolutionaryVault_Manager import EvolutionaryVaultManager
from VisualResource_Intelligent_Asset_Acquisition_Engine import (
    VisualResourceIntelligentAssetAcquisitionEngine,
)
from mosahai.media_intelligence import MediaBatchProcessor, NewsMediaRequest
from mosahai.storage_manager import (
    archive_old_batches,
    build_batch_layout,
    create_unique_batch_id,
    ensure_assets_structure,
    save_json,
)


def language_name_to_code(language_name):
    mapping = {
        "english": "en",
        "telugu": "te",
        "tamil": "ta",
        "kannada": "kn",
    }
    return mapping.get(str(language_name).strip().lower(), "en")


def build_terminal_table(rows):
    headers = ["Topic", "Segment 1 Headline", "Segment 2 Headline", "Segment 3 Headline"]
    normalized = []
    for row in rows:
        normalized.append(
            [
                str(row.get("topic", "")),
                str(row.get("segment_1", "")),
                str(row.get("segment_2", "")),
                str(row.get("segment_3", "")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in normalized:
        for idx in range(4):
            widths[idx] = min(60, max(widths[idx], len(row[idx])))

    def _clip(text, width):
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    line = "+" + "+".join(["-" * (width + 2) for width in widths]) + "+"
    print(line)
    print("| " + " | ".join([headers[i].ljust(widths[i]) for i in range(4)]) + " |")
    print(line)
    for row in normalized:
        print("| " + " | ".join([_clip(row[i], widths[i]).ljust(widths[i]) for i in range(4)]) + " |")
    print(line)


class MosahAIFullPipelineExecutionBridge:
    """Run a one-shot Phase 1.5 pipeline for all configured niches."""

    def __init__(self):
        if not GEMINI_KEYS:
            raise ValueError("Missing GEMINI_KEYS in .env. Add comma-separated free-tier API keys.")

        self.tracker = APIHealthTracker(db_path=USAGE_TRACKER_DB_PATH, keys=GEMINI_KEYS)
        key_manager = KeyManager(tracker=self.tracker)
        self.enable_vault_storage = bool(ENABLE_VAULT_STORAGE)

        try:
            self.vault_manager = EvolutionaryVaultManager(db_path=SCRIPT_VAULT_DB_PATH)
        except Exception as exc:
            self.vault_manager = None
            print(f"[VAULT WARN] Vault manager disabled in run bridge. error={exc}")

        self.brain = MosahAIBrain(db_path=DEFAULT_DB_PATH)
        self.batch_engine = ThreeSegmentDynamicScriptSynthesisProcessor(
            key_manager=key_manager,
            model_name=GEMINI_MODEL,
            vault_manager=self.vault_manager,
            enable_vault_storage=self.enable_vault_storage,
        )
        # self.visual_engine = VisualResourceIntelligentAssetAcquisitionEngine()
        self.media_processor = MediaBatchProcessor()
        self.niches = dict(PHASE15_NICHES)

    def _pause_if_circuit_open(self):
        wait_seconds = self.tracker.get_circuit_wait_seconds()
        if wait_seconds > 0:
            print(f"[CIRCUIT BREAKER] Open. Pausing {wait_seconds}s before new request.")
            time.sleep(wait_seconds)

    def run(self):
        ensure_assets_structure()
        archive_old_batches()
        summary_rows = []
        niche_items = list(self.niches.items())

        for index, (topic, language_name) in enumerate(niche_items):
            try:
                self._pause_if_circuit_open()
                language_code = language_name_to_code(language_name)
                print(f"\n[TOPIC] {topic} | [LANG] {language_name}")

                news_items = self.brain.fetch_news(topic=topic, language_code=language_code, limit=6)
                if not news_items:
                    print("[INFO] No valid news in last 24h.")
                    summary_rows.append(
                        {
                            "topic": topic,
                            "segment_1": "No fresh news",
                            "segment_2": "-",
                            "segment_3": "-",
                        }
                    )
                    continue

                top_three = self.brain.select_top_news(news_items, count=3)
                if len(top_three) < 3:
                    print("[INFO] Less than 3 news found; duplicating latest to complete segment set.")
                    while len(top_three) < 3:
                        top_three.append(top_three[-1])

                titles = [item.get("title", "").strip() for item in top_three]
                news_contexts = [
                    {
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                    }
                    for item in top_three
                ]
                short_id = self.brain.generate_short_id(topic=topic, titles=titles)

                if self.brain.batch_exists(short_id):
                    print(f"[SKIP DUPLICATE] short_id={short_id}")
                    summary_rows.append(
                        {
                            "topic": topic,
                            "segment_1": "Skipped duplicate",
                            "segment_2": "-",
                            "segment_3": "-",
                        }
                    )
                    continue

                batch_id = create_unique_batch_id()
                batch_layout = build_batch_layout(batch_id=batch_id, news_count=3)

                try:
                    payload = self.batch_engine.generate_segments(
                        topic=topic,
                        language=language_name,
                        titles=titles,
                        short_id=short_id,
                        news_contexts=news_contexts,
                    )
                except Exception as exc:
                    if isinstance(exc, CircuitBreakerOpenError):
                        wait_seconds = max(
                            1, int(getattr(exc, "wait_seconds", self.tracker.get_circuit_wait_seconds()))
                        )
                        print(
                            "[CIRCUIT BREAKER] Tracker signaled pause after consecutive failures. "
                            f"Sleeping {wait_seconds}s."
                        )
                        time.sleep(wait_seconds)
                    print(f"[ERROR] Batch generation failed for topic={topic}: {exc}")
                    summary_rows.append(
                        {
                            "topic": topic,
                            "segment_1": "Generation failed",
                            "segment_2": "-",
                            "segment_3": "-",
                        }
                    )
                    continue

                segments = payload.get("segments", [])[:3]
                if len(segments) < 3:
                    while len(segments) < 3:
                        idx = len(segments)
                        title = titles[idx]
                        segments.append(
                            {
                                "id": idx + 1,
                                "title": title,
                                "script": title,
                                "visual_keywords": [
                                    f"{title} headline",
                                    f"{topic} news footage",
                                    "breaking news graphics",
                                ],
                                "duration_target": "15-18s",
                                "transition_type": "cut" if idx < 2 else "end",
                                "transition_trigger": idx < 2,
                            }
                        )

                fallback_used = all(key in payload for key in ("segment_1", "segment_2", "segment_3"))
                primary_visual_asset_path = None
                media_requests: list[NewsMediaRequest] = []
                for idx, layout in enumerate(batch_layout):
                    segment = segments[idx] if idx < len(segments) else {}
                    news_item = top_three[idx] if idx < len(top_three) else {}
                    title = str(news_item.get("title", "")).strip()
                    visual_asset_path = None
                    candidate = None

                    if title:
                        candidate = fetch_primary_image(title)

                        if candidate:
                            downloader = ImageDownloader()

                            media_dir = layout["media_dir"]

                            print("DEBUG media_dir:", media_dir)

                            visual_asset_path = downloader.download_image(
                                url=candidate.url,
                                batch_id=batch_id,
                                news_id=layout["news_id"],
                                output_dir=media_dir,
                                filename="image_1.jpg"
                            )

                            print(f"[TITLE IMAGE] {title}")
                            print(f"[IMAGE URL] {candidate.url}")
                    if visual_asset_path:
                        print(f"[VISUAL] Asset saved at {visual_asset_path}")
                        if idx == 0:
                            primary_visual_asset_path = visual_asset_path
                    else:
                        print(f"[VISUAL] No asset extracted for {layout['news_id']}.")

                    script_payload = {
                        "short_id": payload.get("short_id", short_id),
                        "batch_id": batch_id,
                        "news_id": layout["news_id"],
                        "segment": segment,
                    }
                    save_json(layout["script_path"], script_payload)

                    selected_media = []
                    media_found = False
                    if visual_asset_path:
                        media_found = True
                        selected_media.append(
                            {
                                "source": "article",
                                "url": candidate.url,
                                "local_path": visual_asset_path,
                                "duration": "",
                                "resolution": "",
                                "score": "",
                            }
                        )

                    generation_status = "fallback" if fallback_used else "success"
                    notes = ""
                    if not media_found:
                        generation_status = "partial"
                        notes = "media not found, fallback needed"

                    metadata_payload = {
                        "headline": str(segment.get("title", "")),
                        "topic": str(topic),
                        "language": str(language_name),
                        "batch_id": str(batch_id),
                        "news_id": str(layout["news_id"]),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "queries_used": [],
                        "selected_media": selected_media,
                        "fallback_used": bool(fallback_used),
                        "generation_status": generation_status,
                        "notes": notes,
                    }
                    save_json(layout["metadata_path"], metadata_payload)

                    headline = str(news_item.get("title") or segment.get("title") or "").strip()
                    summary = str(news_item.get("summary") or "").strip()
                    keywords = [
                        str(keyword).strip()
                        for keyword in (segment.get("visual_keywords") or [])
                        if str(keyword).strip()
                    ]
                    media_requests.append(
                        NewsMediaRequest(
                            batch_id=batch_id,
                            news_id=str(layout["news_id"]),
                            headline=headline,
                            keywords=keywords,
                            entities=[],
                            summary=summary if summary else None,
                            article_urls=[candidate.url] if candidate else None,
                            media_dir=layout["media_dir"],
                            metadata_path=layout["metadata_path"],
                        )
                    )

                if media_requests:
                    self.media_processor.process_news_items(media_requests)

                record = {
                    "short_id": payload.get("short_id", short_id),
                    "topic": topic,
                    "language": language_name,
                    "segments": segments,
                    "source_urls": [item.get("link", "") for item in top_three],
                    "visual_asset_path": primary_visual_asset_path,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "pending_review",
                }

                if self.brain.save_batch(record):
                    print(f"[SAVED] short_id={record['short_id']}")
                else:
                    print(f"[SKIP] Duplicate detected while saving short_id={record['short_id']}")

                summary_rows.append(
                    {
                        "topic": topic,
                        "segment_1": segments[0].get("title", "-"),
                        "segment_2": segments[1].get("title", "-"),
                        "segment_3": segments[2].get("title", "-"),
                    }
                )
            finally:
                if index < len(niche_items) - 1:
                    print("[THROTTLE] sleeping 5s before next topic")
                    time.sleep(15)

        print("\nPhase 1.5 Final Summary")
        build_terminal_table(summary_rows)


def main():
    MosahAIFullPipelineExecutionBridge().run()


if __name__ == "__main__":
    main()


# Backward-compatible alias.
ExecutionBridge = MosahAIFullPipelineExecutionBridge
