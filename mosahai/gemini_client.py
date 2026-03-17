import json
from typing import Dict, Any, Optional

import google.generativeai as genai

try:
    from .api_key_manager import APIKeyManager
except Exception:  # pragma: no cover - fallback for missing local manager module
    APIKeyManager = Any  # type: ignore[assignment]

from config import GEMINI_MODEL
from .logger import setup_logger


def safe_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_json(text: str) -> dict:
    parsed = safe_parse(text)
    if isinstance(parsed, dict) and parsed:
        return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = safe_parse(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    return {}


def _generate_text(prompt: str, api_key: str, model_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt, generation_config=config or {}, **kwargs)
    if not response or not hasattr(response, "text"):
        return ""
    return getattr(response, "text", "") or ""


def generate_json(prompt: str, api_key: str, model_name: str) -> dict:
    text = _generate_text(
        prompt=prompt,
        api_key=api_key,
        model_name=model_name,
        config={"response_mime_type": "application/json"},
    )
    if not text:
        return {}
    return _extract_json(text.strip())

class GeminiClient:
    def __init__(self, manager: APIKeyManager, model: str = GEMINI_MODEL):
        self.manager = manager
        self.model = model
        self.logger = setup_logger('GEMINI_CLIENT')
    
    def generate_content(self, prompt: str, config: Dict[str, Any] = None, 
                        **kwargs) -> str:
        if config is None:
            config = {}
        
        def request_fn(key: str) -> str:
            return _generate_text(
                prompt=prompt,
                api_key=key,
                model_name=self.model,
                config=config,
                **kwargs,
            )
        
        result = self.manager.execute_with_key_rotation(request_fn)
        self.logger.info('Gemini request succeeded')
        return result
