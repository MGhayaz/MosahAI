import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")

class EvolutionaryVaultManager:
    """SQLite-backed storage and semantic retrieval for script style references."""

    VALID_HOOK_STRENGTH = {"low", "medium", "high"}
    VALID_ENTITY_DOMAINS = {
        "technology",
        "politics",
        "crime",
        "finance",
        "business",
        "geopolitics",
        "sports",
        "entertainment",
        "science",
        "health",
        "local",
        "world",
        "general",
    }

    def __init__(self, db_path=None, embedding_model_name="all-MiniLM-L6-v2"):
        base_dir = os.path.dirname(__file__)
        self.db_path = db_path or os.path.join(base_dir, "NicheSpecific_ScriptEvolution_History_Bank")
        self.embedding_model_name = str(embedding_model_name).strip() or "all-MiniLM-L6-v2"

        self._embedding_model = None
        self._embedding_model_unavailable = False
        self._embedding_lock = threading.Lock()
        self._ensure_vault()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_vault(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS script_vault (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding_vector TEXT NOT NULL,
                    hook_strength TEXT,
                    hook_type TEXT,
                    topic_entity TEXT,
                    entity_domain TEXT,
                    engagement_score REAL DEFAULT 0.0
                )
                """
            )

            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(script_vault)").fetchall()
            }
            required_columns = {
                "hook_strength": "TEXT",
                "hook_type": "TEXT",
                "topic_entity": "TEXT",
                "entity_domain": "TEXT",
                "engagement_score": "REAL DEFAULT 0.0",
            }
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE script_vault ADD COLUMN {column_name} {column_type}")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_script_vault_id ON script_vault(id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_script_vault_entity ON script_vault(topic_entity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_script_vault_hook ON script_vault(hook_strength)")
            conn.commit()

    def _word_count(self, text):
        return len([part for part in str(text).replace("\n", " ").split() if part.strip()])

    def _validate_script(self, script_text):
        script = str(script_text or "").strip()
        if not script:
            return False, {"reason": "empty_script", "word_count": 0}

        word_count = self._word_count(script)
        if word_count < 35 or word_count > 45:
            return False, {"reason": "word_count_out_of_range", "word_count": word_count}

        # Approximate narration speed window for short-form delivery.
        duration_fast = word_count / 2.8
        duration_slow = word_count / 2.2
        overlaps_target = duration_fast <= 18 and duration_slow >= 15
        if not overlaps_target:
            return False, {
                "reason": "duration_out_of_range",
                "word_count": word_count,
                "estimated_duration_range": [round(duration_fast, 2), round(duration_slow, 2)],
            }

        return True, {
            "word_count": word_count,
            "estimated_duration_range": [round(duration_fast, 2), round(duration_slow, 2)],
        }

    def _normalize_metadata(self, metadata):
        source = metadata if isinstance(metadata, dict) else {}
        normalized = {str(key): value for key, value in source.items()}

        defaults = {
            "niche": "general",
            "tone": "neutral",
            "pace": "medium",
            "style": "news short",
            "timestamp": datetime.now(timezone.utc).date().isoformat(),
            "hook_strength": None,
            "hook_type": None,
            "topic_entity": None,
            "entity_domain": None,
            "engagement_score": 0.0,
        }
        for key, default_value in defaults.items():
            value = normalized.get(key, default_value)
            normalized[key] = default_value if value is None else value

        errors = []
        for key in {"niche", "tone", "pace", "style", "timestamp"}:
            normalized[key] = str(normalized.get(key, defaults[key])).strip() or defaults[key]

        hook_strength = normalized.get("hook_strength")
        if hook_strength not in (None, ""):
            normalized["hook_strength"] = str(hook_strength).strip().lower()
            if normalized["hook_strength"] not in self.VALID_HOOK_STRENGTH:
                errors.append("hook_strength must be one of low/medium/high")
        else:
            normalized["hook_strength"] = None

        hook_type = normalized.get("hook_type")
        if hook_type not in (None, ""):
            if not isinstance(hook_type, str):
                errors.append("hook_type must be a string")
            else:
                hook_type_normalized = hook_type.strip().lower()
                if not hook_type_normalized:
                    errors.append("hook_type cannot be empty when provided")
                else:
                    normalized["hook_type"] = hook_type_normalized
        else:
            normalized["hook_type"] = None

        topic_entity = normalized.get("topic_entity")
        if topic_entity not in (None, ""):
            if not isinstance(topic_entity, str):
                errors.append("topic_entity must be a string")
            else:
                topic_entity_normalized = topic_entity.strip()
                if not topic_entity_normalized:
                    errors.append("topic_entity length must be > 0")
                else:
                    normalized["topic_entity"] = topic_entity_normalized
        else:
            normalized["topic_entity"] = None

        entity_domain = normalized.get("entity_domain")
        if entity_domain not in (None, ""):
            if not isinstance(entity_domain, str):
                errors.append("entity_domain must be a string")
            else:
                entity_domain_normalized = entity_domain.strip().lower()
                if not entity_domain_normalized:
                    errors.append("entity_domain cannot be empty when provided")
                elif entity_domain_normalized not in self.VALID_ENTITY_DOMAINS:
                    errors.append(
                        f"entity_domain must be one of: {', '.join(sorted(self.VALID_ENTITY_DOMAINS))}"
                    )
                else:
                    normalized["entity_domain"] = entity_domain_normalized
        else:
            normalized["entity_domain"] = None

        engagement_score = normalized.get("engagement_score", 0.0)
        if engagement_score in (None, ""):
            normalized["engagement_score"] = 0.0
        else:
            try:
                engagement_value = float(engagement_score)
                if not (0.0 <= engagement_value <= 10.0):
                    errors.append("engagement_score must be between 0 and 10")
                else:
                    normalized["engagement_score"] = round(engagement_value, 3)
            except (TypeError, ValueError):
                errors.append("engagement_score must be a float between 0 and 10")

        return normalized, errors

    def _load_embedding_model(self):
        if self._embedding_model is not None or self._embedding_model_unavailable:
            return self._embedding_model

        with self._embedding_lock:
            if self._embedding_model is not None or self._embedding_model_unavailable:
                return self._embedding_model

            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as exc:
                self._embedding_model_unavailable = True
                print(
                    "[VAULT WARN] sentence-transformers model unavailable; "
                    f"falling back to deterministic embeddings. error={exc}"
                )

        return self._embedding_model

    def _fallback_embedding(self, text, dimension=256):
        vector = [0.0] * int(dimension)
        tokens = re.findall(r"[a-z0-9']+", str(text).lower())
        if not tokens:
            tokens = ["empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % dimension
            sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0
            weight = 1.0 + (int(digest[16:24], 16) % 100) / 500.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def _cosine_similarity(self, vector_a, vector_b):
        a = [float(item) for item in vector_a]
        b = [float(item) for item in vector_b]

        size = min(len(a), len(b))
        if size <= 0:
            return 0.0

        dot = sum(a[index] * b[index] for index in range(size))
        norm_a = math.sqrt(sum(a[index] * a[index] for index in range(size)))
        norm_b = math.sqrt(sum(b[index] * b[index] for index in range(size)))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _next_script_id(self, conn):
        row = conn.execute("SELECT id FROM script_vault ORDER BY rowid DESC LIMIT 1").fetchone()
        if row and row["id"]:
            match = re.search(r"(\d+)$", str(row["id"]))
            if match:
                return f"script_{int(match.group(1)) + 1:04d}"

        count_row = conn.execute("SELECT COUNT(1) AS total FROM script_vault").fetchone()
        next_index = int(count_row["total"]) + 1 if count_row else 1
        return f"script_{next_index:04d}"

    def _increment_script_id(self, script_id):
        match = re.search(r"(\d+)$", str(script_id))
        if not match:
            return f"script_{int(datetime.now(timezone.utc).timestamp())}"
        return f"script_{int(match.group(1)) + 1:04d}"

    def _extract_query_entity(self, query_text):
        text = str(query_text or "").strip()
        if not text:
            return ""

        stop_words = {"the", "a", "an", "and", "or", "in", "on", "for", "with", "to", "of"}
        pattern = r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
        matches = [match.group(0).strip() for match in re.finditer(pattern, text)]
        candidates = [item for item in matches if item.lower() not in stop_words]
        if candidates:
            candidates.sort(key=len, reverse=True)
            return candidates[0]

        token_candidates = [token for token in re.findall(r"[A-Za-z0-9']+", text) if len(token) > 3]
        if token_candidates:
            return token_candidates[0]
        return ""

    def _safe_parse_datetime(self, value):
        raw_value = str(value or "").strip()
        if not raw_value:
            return None

        raw_value = raw_value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass

        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(raw_value, fmt).replace(tzinfo=timezone.utc)
                return parsed
            except Exception:
                continue
        return None

    def _metadata_bonus(self, candidate, query_text, query_entity):
        query_lower = str(query_text or "").lower()
        candidate_entity = str(candidate.get("topic_entity") or "").strip()
        candidate_entity_lower = candidate_entity.lower()

        entity_bonus = 0.0
        if query_entity and candidate_entity_lower:
            query_entity_lower = query_entity.lower()
            if candidate_entity_lower == query_entity_lower:
                entity_bonus = 0.18
            elif query_entity_lower in candidate_entity_lower or candidate_entity_lower in query_entity_lower:
                entity_bonus = 0.12
        elif candidate_entity_lower and candidate_entity_lower in query_lower:
            entity_bonus = 0.1

        hook_strength = str(candidate.get("hook_strength") or "").lower()
        hook_bonus = {"high": 0.08, "medium": 0.04, "low": 0.01}.get(hook_strength, 0.0)

        recent_bonus = 0.0
        timestamp_value = candidate.get("timestamp")
        parsed_timestamp = self._safe_parse_datetime(timestamp_value)
        if parsed_timestamp is not None:
            age_days = (datetime.now(timezone.utc) - parsed_timestamp).total_seconds() / 86400.0
            if age_days <= 2:
                recent_bonus = 0.08
            elif age_days <= 7:
                recent_bonus = 0.05
            elif age_days <= 30:
                recent_bonus = 0.02

        return entity_bonus, hook_bonus, recent_bonus

    def generate_embedding(self, text):
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("Cannot generate embedding for empty text.")

        model = self._load_embedding_model()
        if model is not None:
            vector = model.encode(normalized_text, normalize_embeddings=True)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            return [float(value) for value in vector]

        return self._fallback_embedding(normalized_text)

    def add_script_to_vault(self, script_text, metadata):
        is_valid, details = self._validate_script(script_text)
        if not is_valid:
            print(f"[VAULT SKIP] Script rejected. details={details}")
            return None

        normalized_text = str(script_text).strip()
        normalized_metadata, metadata_errors = self._normalize_metadata(metadata)
        if metadata_errors:
            print(f"[VAULT SKIP] Metadata validation failed. errors={metadata_errors}")
            return None

        embedding_vector = self.generate_embedding(normalized_text)

        with self._connect() as conn:
            script_id = self._next_script_id(conn)
            for _ in range(20):
                try:
                    conn.execute(
                        """
                        INSERT INTO script_vault (
                            id,
                            text,
                            metadata,
                            embedding_vector,
                            hook_strength,
                            hook_type,
                            topic_entity,
                            entity_domain,
                            engagement_score
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            script_id,
                            normalized_text,
                            json.dumps(normalized_metadata, ensure_ascii=False),
                            json.dumps(embedding_vector),
                            normalized_metadata.get("hook_strength"),
                            normalized_metadata.get("hook_type"),
                            normalized_metadata.get("topic_entity"),
                            normalized_metadata.get("entity_domain"),
                            float(normalized_metadata.get("engagement_score", 0.0)),
                        ),
                    )
                    conn.commit()
                    return {
                        "id": script_id,
                        "text": normalized_text,
                        "metadata": normalized_metadata,
                        "embedding_vector": embedding_vector,
                    }
                except sqlite3.IntegrityError:
                    script_id = self._increment_script_id(script_id)

        raise RuntimeError("Unable to insert script into vault after retries.")

    def retrieve_similar_scripts(self, query_text, top_k=3):
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return []

        query_embedding = self.generate_embedding(normalized_query)
        requested_top_k = max(1, int(top_k))
        candidate_pool_size = max(10, requested_top_k)
        query_entity = self._extract_query_entity(normalized_query)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    text,
                    metadata,
                    embedding_vector,
                    hook_strength,
                    hook_type,
                    topic_entity,
                    entity_domain,
                    engagement_score
                FROM script_vault
                """
            ).fetchall()

        scored = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}

            try:
                stored_embedding = json.loads(row["embedding_vector"]) if row["embedding_vector"] else []
            except Exception:
                stored_embedding = []

            if not stored_embedding:
                continue

            metadata.setdefault("hook_strength", row["hook_strength"])
            metadata.setdefault("hook_type", row["hook_type"])
            metadata.setdefault("topic_entity", row["topic_entity"])
            metadata.setdefault("entity_domain", row["entity_domain"])
            metadata.setdefault("engagement_score", row["engagement_score"])

            embedding_similarity = self._cosine_similarity(query_embedding, stored_embedding)
            scored.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": metadata,
                    "embedding_vector": stored_embedding,
                    "embedding_similarity": embedding_similarity,
                    "hook_strength": metadata.get("hook_strength"),
                    "topic_entity": metadata.get("topic_entity"),
                    "timestamp": metadata.get("timestamp"),
                }
            )

        scored.sort(key=lambda item: item.get("embedding_similarity", 0.0), reverse=True)
        candidates = scored[:candidate_pool_size]

        reranked = []
        for candidate in candidates:
            entity_bonus, hook_bonus, recent_bonus = self._metadata_bonus(
                candidate=candidate,
                query_text=normalized_query,
                query_entity=query_entity,
            )
            final_score = candidate.get("embedding_similarity", 0.0) + entity_bonus + hook_bonus + recent_bonus
            candidate["entity_bonus"] = entity_bonus
            candidate["hook_bonus"] = hook_bonus
            candidate["recent_bonus"] = recent_bonus
            candidate["similarity"] = final_score
            reranked.append(candidate)

        reranked.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        return reranked[:requested_top_k]


_DEFAULT_MANAGER = None
_DEFAULT_MANAGER_LOCK = threading.Lock()


def _get_default_manager():
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is not None:
        return _DEFAULT_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        if _DEFAULT_MANAGER is None:
            _DEFAULT_MANAGER = EvolutionaryVaultManager()
    return _DEFAULT_MANAGER


def generate_embedding(text):
    return _get_default_manager().generate_embedding(text)


def add_script_to_vault(script_text, metadata):
    return _get_default_manager().add_script_to_vault(script_text, metadata)


def retrieve_similar_scripts(query_text, top_k=3):
    return _get_default_manager().retrieve_similar_scripts(query_text, top_k=top_k)


# Backward-compatible alias.
ScriptVaultManager = EvolutionaryVaultManager
