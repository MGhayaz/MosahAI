"""Query builder for MosahAI video intelligence searches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


@dataclass(slots=True)
class QueryBuilder:
    min_queries: int = 5
    max_queries: int = 10
    max_keywords: int = 6
    max_entities: int = 4
    max_title_words: int = 10

    def generate_queries(
        self, title: str | None, keywords: Sequence[str] | None, entities: Sequence[str] | None
    ) -> list[str]:
        clean_title = _normalize_text(title)
        title_phrase = _trim_words(clean_title, self.max_title_words)

        keyword_terms = _normalize_terms(keywords)
        entity_terms = _normalize_terms(entities)

        primary_entity = _choose_primary_entity(entity_terms, keyword_terms, title_phrase)
        if primary_entity:
            entity_terms = [primary_entity] + [e for e in entity_terms if e != primary_entity]

        queries: list[str] = []

        if title_phrase:
            base = title_phrase
            if primary_entity and not _contains_entity(base, primary_entity):
                base = f"{primary_entity} {base}"
            queries.extend(
                [
                    base,
                    _append_tail(base, "news"),
                    _append_tail(base, "video"),
                ]
            )

        for keyword in keyword_terms[: self.max_keywords]:
            if primary_entity:
                queries.append(f"{primary_entity} {keyword}")
                queries.append(f"{primary_entity} {keyword} news")
            else:
                queries.append(keyword)

        for left, right in combinations(keyword_terms[:4], 2):
            phrase = f"{left} {right}"
            if primary_entity:
                queries.append(f"{primary_entity} {phrase}")
            else:
                queries.append(phrase)

        modifiers = ["announcement", "launch", "delay", "conference", "press conference", "update"]
        for modifier in modifiers:
            if len(queries) >= self.max_queries:
                break
            if primary_entity:
                queries.append(f"{primary_entity} {modifier}")

        queries = [_normalize_whitespace(q) for q in queries if q and q.strip()]
        queries = _dedupe_strings(queries)

        if primary_entity:
            enforced: list[str] = []
            for query in queries:
                if not _contains_entity(query, primary_entity):
                    enforced.append(_normalize_whitespace(f"{primary_entity} {query}"))
                else:
                    enforced.append(query)
            queries = _dedupe_strings(enforced)

        queries = [q for q in queries if q]

        if len(queries) < self.min_queries:
            queries = _add_fallback_queries(
                queries,
                primary_entity,
                title_phrase,
                self.min_queries,
            )

        return queries[: self.max_queries]


def _normalize_terms(terms: Sequence[str] | None) -> list[str]:
    if not terms:
        return []
    cleaned: list[str] = []
    for term in terms:
        value = _normalize_text(term)
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


def _append_tail(base: str, tail: str) -> str:
    if not base:
        return tail
    if tail.lower() in base.lower():
        return base
    return f"{base} {tail}"


def _add_fallback_queries(
    queries: list[str],
    primary_entity: str,
    title_phrase: str,
    min_queries: int,
) -> list[str]:
    fillers = [
        "latest",
        "news",
        "video",
        "announcement",
        "press conference",
        "update",
    ]

    base = primary_entity or title_phrase
    base = _normalize_whitespace(base)

    for filler in fillers:
        if len(queries) >= min_queries:
            break
        candidate = f"{base} {filler}" if base else filler
        candidate = _normalize_whitespace(candidate)
        if primary_entity and not _contains_entity(candidate, primary_entity):
            candidate = _normalize_whitespace(f"{primary_entity} {candidate}")
        if candidate and candidate.lower() not in {q.lower() for q in queries}:
            queries.append(candidate)

    return queries
