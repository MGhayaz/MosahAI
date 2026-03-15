#!/usr/bin/env python3
"""
Demo/integration for MosahAI API Key Manager + Gemini Client.
Replace placeholders in key_pool.json with real keys.
"""

import time
import sys
sys.path.insert(0, '.')
from mosahai.api_key_manager import APIKeyManager
from mosahai.gemini_client import GeminiClient

def demo():
    # Init manager
    manager = APIKeyManager()
    
    # Mock Gemini (since placeholders)
def mock_gemini(key):
    if 'placeholder' in key:
        raise Exception('Mock 429 rate limit')  # Simulate
    time.sleep(0.1)
    return f"Mock Gemini response using {key[-4:]}..."
    
    # Test execution
    try:
        result = manager.execute_with_key_rotation(mock_gemini)
        print(f"SUCCESS: {result}")
    except Exception as e:
        print(f"Failed as expected (placeholders): {e}")
    
    # Real Gemini (uncomment w/ real keys)
    # client = GeminiClient(manager)
    # response = client.generate_content("Hello world")
    # print(response)

if __name__ == '__main__':
    demo()
