"""Debugging and inspection utilities for MosahAI media intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from mosahai.media_intelligence.video_engine.engine import (
    ArticleVideoAgent,
    TwitterVideoAgent,
    VideoCandidate,
    VideoIntelligenceEngine,
    VideoQueryContext,
    VideoSourceAgent,
    YouTubeVideoAgent,
    _coerce_datetime,
    _safe_int,
)


@dataclass(slots=True)
class MediaDebugInspector:
    engine: VideoIntelligenceEngine | None = None
    max_preview_items: int = 5

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = VideoIntelligenceEngine()

    def inspect_news(
        self,
        batch_id: str,
        news_id: str,
        title: str,
        keywords: Sequence[str] | None,
        entities: Sequence[str] | None,
    ) -> dict[str, Any]:
        context = VideoQueryContext(
            batch_id=str(batch_id),
            news_id=str(news_id),
            news_title=str(title or ""),
            keywords=[str(k) for k in (keywords or []) if str(k).strip()],
            entities=[str(e) for e in (entities or []) if str(e).strip()],
            summary=None,
            now=datetime.now(timezone.utc),
        )

        queries = self.engine.build_queries(context)
        self._print_queries(queries)

        raw_by_agent: dict[str, list[dict[str, Any]]] = {}
        candidates_by_agent: dict[str, list[VideoCandidate]] = {}

        for agent in self.engine.agents:
            raw_results, candidates = self._collect_agent_debug(agent, queries, context)
            raw_by_agent[agent.source_name] = raw_results
            candidates_by_agent[agent.source_name] = candidates

        self._print_raw_counts(raw_by_agent)

        combined_candidates: list[VideoCandidate] = []
        for agent_candidates in candidates_by_agent.values():
            combined_candidates.extend(agent_candidates)

        relevance_filtered = self.engine._apply_relevance_filter(
            combined_candidates, context.news_title
        )
        self._print_stage_count("After relevance filter", relevance_filtered)

        quality_filtered = self.engine._apply_quality_filter(relevance_filtered)
        self._print_stage_count("After quality filter", quality_filtered)

        deduped = (
            self.engine.dedup_engine.remove_duplicates(quality_filtered)
            if self.engine.dedup_engine
            else list(quality_filtered)
        )
        self._print_stage_count("After deduplication", deduped)

        ranked = self.engine._rank_candidates(deduped, context)
        self._print_ranked(ranked)

        return {
            "queries": queries,
            "raw_by_agent": raw_by_agent,
            "candidates_by_agent": candidates_by_agent,
            "after_relevance_filter": relevance_filtered,
            "after_quality_filter": quality_filtered,
            "after_deduplication": deduped,
            "ranked": ranked,
        }

    def _collect_agent_debug(
        self,
        agent: VideoSourceAgent,
        queries: Sequence[str],
        context: VideoQueryContext,
    ) -> tuple[list[dict[str, Any]], list[VideoCandidate]]:
        raw_results: list[dict[str, Any]] = []
        candidates: list[VideoCandidate] = []

        for query in queries:
            try:
                results = agent.search(query, max_results=self.engine.max_results_per_query)
            except Exception:
                results = []
            raw_results.extend([dict(result) for result in results])

            try:
                filtered_results = agent.filter_results(results, context)
            except Exception:
                filtered_results = []

            try:
                ranked_results = agent.rank_results(filtered_results, context=context, query=query)
            except Exception:
                ranked_results = filtered_results

            if hasattr(agent, "top_k_per_query"):
                ranked_results = ranked_results[: getattr(agent, "top_k_per_query")]

            for result in ranked_results:
                media_url = agent.extract_media_url(result)
                if not media_url:
                    continue
                score = self.engine.score_result(result, agent.source_name, context)
                candidates.append(
                    VideoCandidate(
                        source=agent.source_name,
                        url=str(media_url),
                        score=score,
                        title=result.get("title") or result.get("text"),
                        published_at=_coerce_datetime(result.get("published_at")),
                        duration_seconds=_safe_int(result.get("duration_seconds")),
                        raw=result.get("raw", {}),
                    )
                )

        candidates = self.engine._dedupe_candidates(candidates)
        return raw_results, candidates

    def _print_queries(self, queries: Sequence[str]) -> None:
        print("[DEBUG]\nQueries generated:")
        for query in queries:
            print(f"* {query}")
        if not queries:
            print("(no queries generated)")

    def _print_raw_counts(self, raw_by_agent: dict[str, list[dict[str, Any]]]) -> None:
        print("\n[DEBUG]\nRaw candidates by agent:")
        for agent_name, items in raw_by_agent.items():
            print(f"{agent_name.title()} results: {len(items)}")
            for preview in items[: self.max_preview_items]:
                title = preview.get("title") or preview.get("text") or "(no title)"
                url = preview.get("url") or preview.get("media_url") or preview.get("tweet_url") or ""
                print(f"- {title} | {url}")

    def _print_stage_count(self, label: str, candidates: Sequence[VideoCandidate]) -> None:
        print(f"\n[DEBUG]\n{label}: {len(candidates)}")

    def _print_ranked(self, ranked: Sequence[VideoCandidate]) -> None:
        print("\n[DEBUG]\nTop ranked candidates:")
        if not ranked:
            print("(none)")
            return
        for candidate in ranked[: self.max_preview_items]:
            print(f"- {candidate.url}")
