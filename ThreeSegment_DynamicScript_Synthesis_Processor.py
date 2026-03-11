import json
import time
from datetime import datetime, timezone

from google import genai

from SemanticSearch_NicheHistory_EvolutionaryVault_Manager import EvolutionaryVaultManager


class ThreeSegmentDynamicScriptSynthesisProcessor:
    """Generate 3-segment short scripts with strict output normalization."""

    def __init__(
        self,
        key_manager,
        model_name="gemini-1.5-flash",
        max_attempts=12,
        retry_sleep_seconds=2,
        vault_manager=None,
        enable_vault_storage=True,
    ):
        self.key_manager = key_manager
        self.model_name = model_name
        self.max_attempts = max(1, int(max_attempts))
        self.retry_sleep_seconds = max(1, int(retry_sleep_seconds))
        self.enable_vault_storage = bool(enable_vault_storage)

        if vault_manager is not None:
            self.vault_manager = vault_manager
        else:
            try:
                self.vault_manager = EvolutionaryVaultManager()
            except Exception as exc:
                self.vault_manager = None
                print(f"[VAULT WARN] Vault manager unavailable in synthesis processor. error={exc}")

    def _extract_status_code(self, exc):
        status = getattr(exc, "status", None)
        if isinstance(status, int):
            return status
        if isinstance(status, str) and status.isdigit():
            return int(status)

        for token in str(exc).replace(":", " ").split():
            if token.isdigit() and token in {"400", "401", "403", "404", "429", "500", "503"}:
                return int(token)
        return None

    def _build_prompt(self, topic, language, news_contexts, short_id, reference_scripts):
        references = []
        for item in reference_scripts or []:
            if not isinstance(item, dict):
                continue
            script_text = str(item.get("text", "")).strip()
            if script_text:
                references.append(script_text)
            if len(references) >= 3:
                break
        while len(references) < 3:
            references.append("No reference script available in Script Vault yet.")

        context_lines = []
        for index, item in enumerate(news_contexts):
            title = str(item.get("title", "")).strip()
            summary = str(item.get("summary", "")).strip()
            if not summary:
                summary = "No summary available."
            context_lines.append(f"{index + 1}. Headline: {title}\n   Summary: {summary}")

        context_block = "\n".join(context_lines)
        return (
            "You are a professional news editor creating high-retention shorts.\n\n"
            "Reference the following scripts for tone and pacing only.\n\n"
            f"Script A:\n{references[0]}\n\n"
            f"Script B:\n{references[1]}\n\n"
            f"Script C:\n{references[2]}\n\n"
            "Use BOTH the Headline and Summary of the news article.\n"
            "Reference Vault scripts ONLY for:\n"
            "- writing style\n"
            "- pacing\n"
            "- narrative tone\n\n"
            "DO NOT copy facts from reference scripts.\n"
            "DO NOT hallucinate information.\n"
            "All facts must originate from the current news headline and summary.\n\n"
            "Now rewrite the following news into fast-paced short scripts with completely original content.\n"
            "Generate 3 distinct news scripts for a vertical 9:16 short.\n"
            "Each script must be roughly 35-45 words for a 15-18 second window.\n"
            "Each news item must be independent but high-retention and fast-paced.\n"
            "After generating each script, extract metadata for:\n"
            "- primary entity mentioned\n"
            "- domain of the news\n"
            "- hook type used in script opening\n"
            "- estimated hook strength\n"
            "For each segment, provide 3 visual keywords describing what imagery or footage should appear on screen.\n"
            "Keywords must be simple descriptive phrases.\n"
            "No emojis. No hashtags. No intro. No outro.\n"
            "Return strict JSON only.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "short_id": "STRING",\n'
            '  "segments": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "title": "Headline",\n'
            '      "script": "35-45 words",\n'
            '      "visual_keywords": [\n'
            '        "keyword one",\n'
            '        "keyword two",\n'
            '        "keyword three"\n'
            "      ],\n"
            '      "metadata": {\n'
            '        "topic_entity": "Primary entity name",\n'
            '        "entity_domain": "technology|politics|crime|finance|general",\n'
            '        "hook_type": "shock|question|conflict|reveal",\n'
            '        "hook_strength": "low|medium|high"\n'
            "      },\n"
            '      "duration_target": "15-18s",\n'
            '      "transition_type": "cut",\n'
            '      "transition_trigger": true\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Current news context input:\n"
            f"Topic: {topic}\n"
            f"Language: {language}\n"
            f"short_id: {short_id}\n"
            f"Items:\n{context_block}\n"
        )

    def _fit_word_window(self, text, minimum=35, maximum=45):
        words = [part for part in str(text).replace("\n", " ").split() if part.strip()]
        if len(words) > maximum:
            return " ".join(words[:maximum]).strip()
        if len(words) < minimum:
            filler = "The impact is immediate, and more updates are expected very soon."
            filler_words = [part for part in filler.split() if part.strip()]
            while len(words) < minimum:
                words.extend(filler_words)
            words = words[:maximum]
        return " ".join(words).strip()

    def _fallback_script(self, title):
        base = (
            f"{title}. This development moved quickly and triggered immediate reactions. "
            "Officials and key observers are now watching outcomes closely as fresh details "
            "continue to emerge through the day."
        )
        return self._fit_word_window(base)

    def _safe_parse_json(self, raw_text):
        if not raw_text:
            raise ValueError("Gemini returned empty response text.")

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        payload_text = raw_text[start : end + 1] if start != -1 and end != -1 and end > start else raw_text
        try:
            return json.loads(payload_text)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Malformed JSON from Gemini, using fallback segments. error={exc}")
            return {}

    def _normalize_hook_strength(self, hook_strength):
        normalized = str(hook_strength or "").strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return None

    def _estimate_engagement_score(self, hook_strength):
        strength = self._normalize_hook_strength(hook_strength)
        if strength == "high":
            return 8.0
        if strength == "medium":
            return 6.0
        if strength == "low":
            return 4.0
        return 0.0

    def _normalize_segment_metadata(self, source_metadata, title, topic):
        source = source_metadata if isinstance(source_metadata, dict) else {}
        topic_entity = str(source.get("topic_entity", "")).strip() or str(title).strip()
        entity_domain = str(source.get("entity_domain", "")).strip().lower() or "general"
        hook_type = str(source.get("hook_type", "")).strip().lower() or None
        hook_strength = self._normalize_hook_strength(source.get("hook_strength"))

        return {
            "niche": str(topic).strip() or "general",
            "tone": "dramatic" if hook_strength == "high" else "neutral",
            "pace": "fast",
            "style": hook_type or "news short",
            "timestamp": datetime.now(timezone.utc).date().isoformat(),
            "topic_entity": topic_entity if topic_entity else None,
            "entity_domain": entity_domain,
            "hook_type": hook_type,
            "hook_strength": hook_strength,
            "engagement_score": self._estimate_engagement_score(hook_strength),
        }

    def _normalize_visual_keywords(self, source_keywords, title, topic_entity, topic):
        normalized = []
        if isinstance(source_keywords, list):
            candidates = source_keywords
        elif isinstance(source_keywords, str):
            candidates = [part.strip() for part in source_keywords.replace(";", ",").split(",")]
        else:
            candidates = []

        for candidate in candidates:
            keyword = str(candidate).strip()
            if not keyword:
                continue
            if keyword.lower() in {item.lower() for item in normalized}:
                continue
            normalized.append(keyword)
            if len(normalized) >= 3:
                break

        fallback_keywords = [
            f"{str(topic_entity or title).strip()} headline",
            f"{str(topic).strip()} news footage",
            "breaking news graphics",
        ]
        for keyword in fallback_keywords:
            if len(normalized) >= 3:
                break
            if keyword.lower() in {item.lower() for item in normalized}:
                continue
            normalized.append(keyword)

        return normalized[:3]

    def _store_generated_segments_in_vault(self, segments, extracted_metadata):
        if not self.enable_vault_storage or self.vault_manager is None:
            return

        for index, segment in enumerate(segments):
            script_text = str(segment.get("script", "")).strip()
            if not script_text:
                continue

            metadata = extracted_metadata[index] if index < len(extracted_metadata) else {}
            try:
                self.vault_manager.add_script_to_vault(script_text=script_text, metadata=metadata)
            except Exception as exc:
                print(f"[VAULT WARN] Failed to store generated script. error={exc}")

    def _normalize_segments(self, payload, short_id, titles, topic):
        if not isinstance(payload, dict):
            payload = {}

        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            raw_segments = []

        normalized_segments = []
        extracted_metadata = []
        for index in range(3):
            source = raw_segments[index] if index < len(raw_segments) and isinstance(raw_segments[index], dict) else {}
            title = str(source.get("title", "")).strip() or titles[index]
            script = str(source.get("script", "")).strip()
            script = self._fit_word_window(script) if script else self._fallback_script(title)
            metadata = self._normalize_segment_metadata(
                source_metadata=source.get("metadata", {}),
                title=title,
                topic=topic,
            )
            visual_keywords = self._normalize_visual_keywords(
                source_keywords=source.get("visual_keywords", []),
                title=title,
                topic_entity=metadata.get("topic_entity"),
                topic=topic,
            )
            metadata["visual_keywords"] = list(visual_keywords)

            normalized_segments.append(
                {
                    "id": index + 1,
                    "title": title,
                    "script": script,
                    "visual_keywords": visual_keywords,
                    "duration_target": "15-18s",
                    "transition_type": "cut" if index < 2 else "end",
                    "transition_trigger": index < 2,
                }
            )
            extracted_metadata.append(metadata)

        normalized_payload = {
            "short_id": str(payload.get("short_id", "")).strip() or short_id,
            "segments": normalized_segments,
        }
        return normalized_payload, extracted_metadata

    def _normalize_news_contexts(self, titles, news_contexts):
        if not isinstance(news_contexts, list):
            news_contexts = []

        contexts = []
        for index in range(3):
            source = news_contexts[index] if index < len(news_contexts) and isinstance(news_contexts[index], dict) else {}
            title = str(source.get("title", "")).strip() or titles[index]
            summary = str(source.get("summary", "")).strip()
            contexts.append({"title": title, "summary": summary})
        return contexts

    def generate_segments(self, topic, language, titles, short_id, news_contexts=None):
        if len(titles) != 3:
            raise ValueError("ThreeSegmentDynamicScriptSynthesisProcessor expects exactly 3 titles.")

        normalized_contexts = self._normalize_news_contexts(titles=titles, news_contexts=news_contexts)

        reference_scripts = []
        if self.vault_manager is not None:
            query_text = "\n".join(
                [f"{item['title']}. {item['summary']}" for item in normalized_contexts]
            )
            try:
                reference_scripts = self.vault_manager.retrieve_similar_scripts(query_text=query_text, top_k=3)
            except Exception as exc:
                print(f"[VAULT WARN] Similar-script retrieval failed. error={exc}")

        prompt = self._build_prompt(
            topic=topic,
            language=language,
            news_contexts=normalized_contexts,
            short_id=short_id,
            reference_scripts=reference_scripts,
        )

        for attempt in range(1, self.max_attempts + 1):
            api_key = self.key_manager.get_next_key()
            if not api_key:
                raise RuntimeError("No active API keys available.")

            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )

                payload = self._safe_parse_json(
                    (getattr(response, "text", "") or getattr(response, "output_text", "") or "").strip()
                )
                normalized_payload, extracted_metadata = self._normalize_segments(
                    payload=payload,
                    short_id=short_id,
                    titles=titles,
                    topic=topic,
                )
                self._store_generated_segments_in_vault(
                    segments=normalized_payload.get("segments", []),
                    extracted_metadata=extracted_metadata,
                )
                self.key_manager.mark_success(api_key)
                return normalized_payload
            except Exception as exc:
                status_code = self._extract_status_code(exc)
                message = str(exc).lower()
                print(
                    f"[GEMINI ERROR] attempt={attempt} status={status_code} "
                    f"key=...{api_key[-4:]} error={exc}"
                )

                if status_code == 429 or "quota" in message or "rate" in message:
                    self.key_manager.mark_error(api_key, 429)
                elif status_code == 404 or "not found" in message:
                    self.key_manager.mark_error(api_key, 404)
                elif status_code is not None:
                    self.key_manager.mark_error(api_key, status_code)

                if attempt >= self.max_attempts:
                    raise
                time.sleep(self.retry_sleep_seconds)

        raise RuntimeError("Batch segment generation failed after all retries.")


# Backward-compatible alias.
BatchSegmentEngine = ThreeSegmentDynamicScriptSynthesisProcessor
