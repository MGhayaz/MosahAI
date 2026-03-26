import os
import re

import requests


class VisualResourceIntelligentAssetAcquisitionEngine:
    """Extract and cache a representative visual asset from a news article."""

    def __init__(self, assets_dir="temp_assets", request_timeout_seconds=10):
        self.assets_dir = str(assets_dir).strip() or "temp_assets"
        self.request_timeout_seconds = max(1, int(request_timeout_seconds))
        self.article_image_extractor = None

    def _safe_filename(self, short_id):
        base = str(short_id or "").strip()
        if not base:
            base = "unknown_short"
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        return f"{safe_base}.jpg"

    def extract_visual_asset(
        self,
        news_url: str,
        short_id: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str | None:
        normalized_news_url = str(news_url or "").strip()
        if not normalized_news_url:
            return None

        try:
            if self.article_image_extractor is None:
                from mosahai.media_intelligence.image_pipeline.article_image_extractor import (
                    ArticleImageExtractor,
                )

                self.article_image_extractor = ArticleImageExtractor(
                    timeout_seconds=self.request_timeout_seconds
                )
            candidates = self.article_image_extractor.extract(normalized_news_url)
        except Exception as exc:
            print(f"[VISUAL WARN] Failed to extract article image. url={normalized_news_url} error={exc}")
            return None

        image_url = ""
        if candidates:
            image_url = str(getattr(candidates[0], "url", "") or "").strip()

        if not image_url:
            return None

        try:
            target_dir = str(output_dir).strip() if output_dir else self.assets_dir
            os.makedirs(target_dir, exist_ok=True)

            if filename:
                safe_name = os.path.basename(str(filename))
            else:
                safe_name = self._safe_filename(short_id)
            local_file_path = os.path.join(target_dir, safe_name)

            from mosahai.media_intelligence.image_downloader import HEADERS

            response = requests.get(
                image_url,
                timeout=self.request_timeout_seconds,
                headers=HEADERS,
            )
            response.raise_for_status()

            with open(local_file_path, "wb") as handle:
                handle.write(response.content)

            return local_file_path
        except Exception as exc:
            print(f"[VISUAL WARN] Failed to download/store image. image_url={image_url} error={exc}")
            return None
