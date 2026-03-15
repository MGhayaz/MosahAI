import google.generativeai as genai
from typing import Dict, Any
from .api_key_manager import APIKeyManager, State
from .logger import setup_logger

class GeminiClient:
    def __init__(self, manager: APIKeyManager, model: str = 'gemini-1.5-flash'):
        self.manager = manager
        self.model = model
        self.logger = setup_logger('GEMINI_CLIENT')
    
    def generate_content(self, prompt: str, config: Dict[str, Any] = None, 
                        **kwargs) -> str:
        if config is None:
            config = {}
        
        def request_fn(key: str) -> str:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt, generation_config=config, **kwargs)
            return response.text
        
        result = self.manager.execute_with_key_rotation(request_fn)
        self.logger.info('Gemini request succeeded')
        return result
