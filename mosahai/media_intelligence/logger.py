"""Structured logging for MosahAI media intelligence."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class MediaEngineLogger:
    log_path: str = "logs/media_engine.log"
    logger_name: str = "mosahai.media_engine"

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.logger_name)
        self._logger.setLevel(logging.INFO)
        self._ensure_handler()

    def search_started(self, *, batch_id: str, news_title: str, queries: list[str]) -> None:
        self.log_event(
            "search_started",
            batch_id=batch_id,
            news_title=news_title,
            query_count=len(queries),
        )

    def agent_result(
        self,
        *,
        batch_id: str,
        news_title: str,
        agent: str,
        query: str,
        status: str,
        result_count: int,
    ) -> None:
        self.log_event(
            "agent_result",
            batch_id=batch_id,
            news_title=news_title,
            agent=agent,
            query=query,
            status=status,
            result_count=result_count,
        )

    def candidate_found(
        self,
        *,
        batch_id: str,
        news_title: str,
        agent: str,
        count: int,
    ) -> None:
        self.log_event(
            "candidate_found",
            batch_id=batch_id,
            news_title=news_title,
            agent=agent,
            count=count,
        )

    def ranking_results(
        self,
        *,
        batch_id: str,
        news_title: str,
        top_urls: list[str],
        total_candidates: int,
    ) -> None:
        self.log_event(
            "ranking_results",
            batch_id=batch_id,
            news_title=news_title,
            total_candidates=total_candidates,
            top_urls="|".join(top_urls),
        )

    def video_downloaded(
        self,
        *,
        batch_id: str,
        news_id: str,
        url: str,
        output_path: str,
    ) -> None:
        self.log_event(
            "video_downloaded",
            batch_id=batch_id,
            news_id=news_id,
            url=url,
            output_path=output_path,
        )

    def log_event(self, event: str, **fields: Any) -> None:
        payload = " ".join(f"{key}={_stringify(value)}" for key, value in fields.items())
        timestamp = datetime.now(timezone.utc).isoformat()
        self._logger.info("[MEDIA_ENGINE] ts=%s event=%s %s", timestamp, event, payload)

    def _ensure_handler(self) -> None:
        log_path = self.log_path
        if log_path and not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
                return

        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\n", " ").replace("\r", " ")
