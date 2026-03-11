import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from datetime import datetime, timezone

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
        self.visual_engine = VisualResourceIntelligentAssetAcquisitionEngine()
        self.niches = dict(PHASE15_NICHES)

    def _pause_if_circuit_open(self):
        wait_seconds = self.tracker.get_circuit_wait_seconds()
        if wait_seconds > 0:
            print(f"[CIRCUIT BREAKER] Open. Pausing {wait_seconds}s before new request.")
            time.sleep(wait_seconds)

    def run(self):
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

                primary_news_url = str(top_three[0].get("link", "")).strip() if top_three else ""
                visual_asset_path = None
                if primary_news_url:
                    visual_asset_path = self.visual_engine.extract_visual_asset(
                        news_url=primary_news_url,
                        short_id=payload.get("short_id", short_id),
                    )
                if visual_asset_path:
                    print(f"[VISUAL] Asset saved at {visual_asset_path}")
                else:
                    print("[VISUAL] No asset extracted for this short.")

                record = {
                    "short_id": payload.get("short_id", short_id),
                    "topic": topic,
                    "language": language_name,
                    "segments": segments,
                    "source_urls": [item.get("link", "") for item in top_three],
                    "visual_asset_path": visual_asset_path,
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
                    time.sleep(5)

        print("\nPhase 1.5 Final Summary")
        build_terminal_table(summary_rows)


def main():
    MosahAIFullPipelineExecutionBridge().run()


if __name__ == "__main__":
    main()


# Backward-compatible alias.
ExecutionBridge = MosahAIFullPipelineExecutionBridge
