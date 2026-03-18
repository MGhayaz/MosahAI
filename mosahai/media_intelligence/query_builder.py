"""Query builder for MosahAI video intelligence searches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


@dataclass(slots=True)
class QueryBuilder:
    min_queries: int = 5
    max_queries: int = 6
    max_keywords: int = 6
    max_entities: int = 4
    max_title_words: int = 10
    min_meaningful_tokens: int = 2

    def generate_queries(
        self, title: str | None, keywords: Sequence[str] | None, entities: Sequence[str] | None
    ) -> list[str]:
        clean_title = _normalize_text(title)
        title_phrase = _trim_words(clean_title, self.max_title_words)

        keyword_terms = _normalize_terms(keywords)
        entity_terms = _normalize_terms(entities)

        min_tokens = max(self.min_meaningful_tokens, 2)
        max_queries = min(self.max_queries, 6)
        min_queries = min(self.min_queries, max_queries)

        primary_entity = _choose_primary_entity(entity_terms, keyword_terms, title_phrase)
        if primary_entity:
            entity_terms = [primary_entity] + [e for e in entity_terms if e != primary_entity]

        keyword_terms = keyword_terms[: self.max_keywords]
        if primary_entity:
            keyword_terms = [
                keyword for keyword in keyword_terms if not _equivalent_phrase(keyword, primary_entity)
            ]
        modifiers = ["announcement", "launch", "delay", "conference", "press conference", "update", "model"]
        modifier_set = {modifier.lower() for modifier in modifiers}
        primary_keyword = _choose_primary_keyword(keyword_terms, primary_entity, modifier_set)
        primary_keyword_key = primary_keyword.lower() if primary_keyword else ""

        topic_hints = _collect_topic_hints(title_phrase, keyword_terms, entity_terms)

        candidates: list[tuple[float, int, str]] = []
        order = 0

        def add_candidate(text: str, score: float, force_entity: bool = False) -> None:
            nonlocal order
            normalized = _normalize_query(text)
            if not normalized:
                return
            if force_entity and primary_entity and not _contains_entity(normalized, primary_entity):
                normalized = _normalize_query(f"{primary_entity} {normalized}")
            if not _has_min_meaningful_tokens(normalized, min_tokens):
                return
            for variant in _expand_with_intent_suffixes(normalized):
                candidates.append((score, order, variant))
                order += 1

        if title_phrase:
            base = title_phrase
            if primary_entity and not _contains_entity(base, primary_entity):
                base = f"{primary_entity} {base}"
            add_candidate(base, 70.0)
            add_candidate(_append_tail(base, "news"), 68.0)
            add_candidate(_append_tail(base, "video"), 67.0)

        if title_phrase and (keyword_terms or entity_terms):
            combo_terms = _dedupe_strings(keyword_terms[:2] + entity_terms[:2])
            if combo_terms:
                add_candidate(f"{title_phrase} {' '.join(combo_terms)}", 78.0, force_entity=True)

        core_pair = ""
        if primary_entity and primary_keyword:
            core_pair = f"{primary_entity} {primary_keyword}"
            add_candidate(core_pair, 96.0)
            add_candidate(f"{core_pair} announcement", 95.0)
            add_candidate(f"{core_pair} launch", 94.0)
            add_candidate(f"{core_pair} update", 93.0)

            extra_terms: list[tuple[str, float]] = []
            seen_extras: set[str] = set()
            for keyword in keyword_terms:
                if _equivalent_phrase(keyword, primary_keyword):
                    continue
                if _equivalent_phrase(keyword, primary_entity):
                    continue
                key = keyword.lower()
                if key in seen_extras:
                    continue
                seen_extras.add(key)
                extra_terms.append((keyword, 95.0))

            for modifier in modifiers:
                key = modifier.lower()
                if key in seen_extras:
                    continue
                seen_extras.add(key)
                extra_terms.append((modifier, 90.0))

            for term, score in extra_terms:
                add_candidate(f"{core_pair} {term}", score)

        if primary_entity and topic_hints:
            for hint in topic_hints:
                add_candidate(f"{primary_entity} {hint}", 92.0, force_entity=True)

        if not primary_entity:
            for left, right in combinations(keyword_terms[:4], 2):
                add_candidate(f"{left} {right}", 70.0)

        if primary_entity and not primary_keyword:
            for keyword in keyword_terms:
                if _equivalent_phrase(primary_entity, keyword):
                    continue
                if keyword.lower() in modifier_set:
                    continue
                add_candidate(f"{primary_entity} {keyword}", 80.0)

        if primary_entity and keyword_terms:
            add_candidate(f"{primary_entity} {' '.join(keyword_terms[:2])}", 86.0, force_entity=True)
        if primary_entity and len(entity_terms) > 1:
            add_candidate(f"{primary_entity} {entity_terms[1]}", 82.0, force_entity=True)
        if keyword_terms and entity_terms:
            add_candidate(f"{entity_terms[0]} {keyword_terms[0]}", 76.0)

        if primary_entity and not keyword_terms:
            for modifier in modifiers:
                add_candidate(f"{primary_entity} {modifier}", 55.0)

        candidates.sort(key=lambda item: (-item[0], item[1]))

        queries: list[str] = []
        seen: set[str] = set()
        for _, _, query in candidates:
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
            if len(queries) >= max_queries:
                break

        if len(queries) < min_queries:
            fallback_base = core_pair or primary_entity or title_phrase
            queries = _add_fallback_queries(
                queries,
                primary_entity,
                fallback_base,
                min_queries,
                min_tokens,
            )

        return queries[: max_queries]


def _normalize_terms(terms: Sequence[str] | None) -> list[str]:
    if not terms:
        return []
    cleaned: list[str] = []
    for term in terms:
        value = _normalize_query(term)
        if value:
            cleaned.append(value)
    return _dedupe_strings(cleaned)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\-\s]+", " ", text)
    return _normalize_whitespace(text)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _contains_entity(text: str, entity: str) -> bool:
    if not text or not entity:
        return False
    return entity.lower() in text.lower()


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _dedupe_tokens(tokens: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


def _normalize_query(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    tokens = _dedupe_tokens(cleaned.split())
    return " ".join(tokens)


_STOPWORDS = {"a", "an", "the"}
_INTENT_TOKENS = ("news", "video", "footage", "breaking")
_INTENT_SUFFIXES = ("news", "video", "footage", "breaking news")


def _is_meaningful_token(token: str) -> bool:
    if not token:
        return False
    if token.lower() in _STOPWORDS:
        return False
    alnum = re.sub(r"[^A-Za-z0-9]+", "", token)
    if not alnum:
        return False
    if len(alnum) >= 2:
        return True
    return token.isupper()


def _has_min_meaningful_tokens(text: str, min_tokens: int) -> bool:
    tokens = text.split()
    meaningful = [token for token in tokens if _is_meaningful_token(token)]
    return len(meaningful) >= min_tokens


def _contains_intent_token(text: str) -> bool:
    if not text:
        return False
    pattern = r"\b(" + "|".join(re.escape(token) for token in _INTENT_TOKENS) + r")\b"
    return re.search(pattern, text.lower()) is not None


def _expand_with_intent_suffixes(base: str) -> list[str]:
    normalized = _normalize_query(base)
    if not normalized:
        return []
    if _contains_intent_token(normalized):
        return [normalized]
    variants: list[str] = []
    for suffix in _INTENT_SUFFIXES:
        candidate = _normalize_query(f"{normalized} {suffix}")
        if not candidate:
            continue
        if candidate.lower() in {item.lower() for item in variants}:
            continue
        variants.append(candidate)
    return variants


def _choose_primary_entity(
    entities: Sequence[str], keywords: Sequence[str], title_phrase: str
) -> str:
    for entity in entities:
        if entity:
            return entity

    for keyword in keywords:
        if keyword and keyword[0].isupper():
            return keyword

    title_tokens = title_phrase.split()
    return title_tokens[0] if title_tokens else ""


def _choose_primary_keyword(
    keywords: Sequence[str], primary_entity: str, modifier_terms: set[str]
) -> str:
    best_keyword = ""
    best_score: int | None = None
    for keyword in keywords:
        if not keyword:
            continue
        if primary_entity and _equivalent_phrase(keyword, primary_entity):
            continue
        if not _has_min_meaningful_tokens(keyword, 1):
            continue
        score = 0
        keyword_key = keyword.lower()
        if keyword_key in modifier_terms:
            score -= 3
        if " " in keyword:
            score += 2
        if keyword.isupper():
            score += 2
        if any(char.isupper() for char in keyword):
            score += 1
        if keyword.isalpha() and len(keyword) <= 3:
            score += 2
        alnum = re.sub(r"[^A-Za-z0-9]+", "", keyword)
        if len(alnum) >= 4:
            score += 1
        if best_score is None or score > best_score:
            best_keyword = keyword
            best_score = score
    return best_keyword
    return ""


def _append_tail(base: str, tail: str) -> str:
    if not base:
        return tail
    if tail.lower() in base.lower():
        return base
    return f"{base} {tail}"


def _equivalent_phrase(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _normalize_text(left).lower() == _normalize_text(right).lower()


def _collect_topic_hints(title_phrase: str, keywords: Sequence[str], entities: Sequence[str]) -> list[str]:
    combined = " ".join([title_phrase] + list(keywords or []) + list(entities or [])).lower()
    hints: list[str] = []

    if "ai" in combined or "artificial intelligence" in combined:
        hints.extend(["AI announcement", "AI launch", "AI update"])

    if "chip" in combined or "gpu" in combined or "semiconductor" in combined:
        if "ai" in combined:
            hints.append("AI chip announcement")
        hints.append("chip launch")

    if "data center" in combined or "datacenter" in combined:
        if "ai" in combined:
            hints.append("data center AI launch")
        else:
            hints.append("data center launch")

    if "earnings" in combined:
        hints.append("earnings call")

    return _dedupe_strings(hints)


def _add_fallback_queries(
    queries: list[str],
    primary_entity: str,
    base_phrase: str,
    min_queries: int,
    min_tokens: int,
) -> list[str]:
    fillers = [
        "latest news",
        "news video",
        "breaking news",
        "official announcement video",
        "event footage",
    ]

    base = base_phrase
    base = _normalize_query(base)
    seen = {query.lower() for query in queries}

    for filler in fillers:
        if len(queries) >= min_queries:
            break
        candidate = f"{base} {filler}" if base else filler
        candidate = _normalize_query(candidate)
        if primary_entity and not _contains_entity(candidate, primary_entity):
            candidate = _normalize_query(f"{primary_entity} {candidate}")
        if not candidate:
            continue
        if not _has_min_meaningful_tokens(candidate, min_tokens):
            continue
        for variant in _expand_with_intent_suffixes(candidate):
            key = variant.lower()
            if key in seen:
                continue
            queries.append(variant)
            seen.add(key)
            if len(queries) >= min_queries:
                break

    return queries
