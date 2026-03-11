import os
import re

import requests


class VisualResourceIntelligentAssetAcquisitionEngine:
    """Extract and cache a representative visual asset from a news article."""

    def __init__(self, assets_dir="temp_assets", request_timeout_seconds=10):
        self.assets_dir = str(assets_dir).strip() or "temp_assets"
        self.request_timeout_seconds = max(1, int(request_timeout_seconds))

    def _safe_filename(self, short_id):
        base = str(short_id or "").strip()
        if not base:
            base = "unknown_short"
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        return f"{safe_base}.jpg"

    def extract_visual_asset(self, news_url: str, short_id: str) -> str | None:
        normalized_news_url = str(news_url or "").strip()
        if not normalized_news_url:
            return None

        try:
            from newspaper import Article
        except Exception as exc:
            print(f"[VISUAL WARN] newspaper3k unavailable. error={exc}")
            return None

        try:
            article = Article(normalized_news_url)
            article.download()
            article.parse()
            image_url = str(getattr(article, "top_image", "") or "").strip()
        except Exception as exc:
            print(f"[VISUAL WARN] Failed to parse article for image. url={normalized_news_url} error={exc}")
            return None

        if not image_url:
            return None

        try:
            os.makedirs(self.assets_dir, exist_ok=True)
            filename = self._safe_filename(short_id)
            local_file_path = os.path.join(self.assets_dir, filename)

            response = requests.get(
                image_url,
                timeout=self.request_timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MosahAI/3.0)"},
            )
            response.raise_for_status()

            with open(local_file_path, "wb") as handle:
                handle.write(response.content)

            return local_file_path
        except Exception as exc:
            print(f"[VISUAL WARN] Failed to download/store image. image_url={image_url} error={exc}")
            return None
