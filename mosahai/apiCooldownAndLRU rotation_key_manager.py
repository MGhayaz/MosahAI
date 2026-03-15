import json
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any

from .logger import setup_logger

@dataclass(slots=True)
class State:
    key: str
    last_used: float = 0.0
    cooldown_until: float = 0.0
    failure_count: int = 0
    timestamps: list[float] = field(default_factory=list)

class APIKeyManager:
    def __init__(self, key_pool_path: str = 'key_pool.json', 
                 max_rpm: int = 3, window_seconds: int = 60,
                 cooldown_429: int = 60, cb_pause: int = 120):
        self.logger = setup_logger('KEY_MANAGER')
        self.key_pool_path = key_pool_path
        self.keys = self._load_keys()
        self.states: Dict[str, State] = {}
        self.max_rpm = max_rpm
        self.window_seconds = window_seconds
        self.cooldown_429 = cooldown_429
        self.cb_pause = cb_pause
        self._lock = threading.Lock()
        self._init_states()
    
    def _load_keys(self) -> List[str]:
        try:
            with open(self.key_pool_path, 'r') as f:
                data = json.load(f)
            keys = data.get('gemini_keys', [])
            self.logger.info(f'Loaded {len(keys)} Gemini keys from {self.key_pool_path}')
            if len(keys) == 0:
                raise ValueError('No Gemini keys found in key_pool.json')
            return keys
        except Exception as e:
            self.logger.error(f'Failed to load keys: {e}')
            raise
    
    def _init_states(self):
        with self._lock:
            self.states = {key: State(key=key) for key in self.keys}
    
    def _prune_timestamps(self, state: State, now: float):
        while state.timestamps and now - state.timestamps[0] > self.window_seconds:
            state.timestamps.pop(0)
    
    def _is_healthy(self, state: State) -> bool:
        now = time.time()
        with self._lock:
            self._prune_timestamps(state, now)
            recent = len(state.timestamps)
            return now >= state.cooldown_until and recent < self.max_rpm
    
    def _log_metrics(self):
        now = time.time()
        active = cooling = rpm_blocked = total = 0
        with self._lock:
            for state in self.states.values():
                self._prune_timestamps(state, now)
                total += 1
                if now < state.cooldown_until:
                    cooling += 1
                elif len(state.timestamps) >= self.max_rpm:
                    rpm_blocked += 1
                else:
                    active += 1
        self.logger.info(f'Metrics - Active: {active}, Cooling: {cooling}, RPM-blocked: {rpm_blocked}/{total}')
    
    def _get_candidates(self) -> List[State]:
        self._log_metrics()
        candidates = [state for state in self.states.values() if self._is_healthy(state)]
        candidates.sort(key=lambda s: s.last_used)
        return candidates
    
    def get_next_key(self) -> Optional[str]:
        candidates = self._get_candidates()
        if not candidates:
            self.logger.warning('No healthy keys available')
            return None
        state = candidates[0]  # LRU
        with self._lock:
            now = time.time()
            self._prune_timestamps(state, now)
            state.timestamps.append(now)
            state.last_used = now
        self.logger.info(f'Using key {state.key[:8]}...')
        return state.key
    
    def _extract_status(self, exc: Exception) -> Optional[int]:
        exc_str = str(exc).lower()
        for code in ['429', '503']:
            if code in exc_str:
                return int(code)
        return None
    
    def execute_with_key_rotation(self, request_fn: Callable[[str], Any], 
                                max_retries: int = 3) -> Any:
        for attempt in range(len(self.keys)):
            candidates = self._get_candidates()
            if not candidates:
                self.logger.warning('Circuit breaker: No healthy keys. Pausing...')
                time.sleep(self.cb_pause)
                continue
            
            # Try up to max_retries per round
            for retry in range(max_retries):
                state = candidates[retry % len(candidates)]
                key = state.key
                try:
                    self.logger.info(f'Executing with key {state.key[:8]} (retry {retry+1})')
                    result = request_fn(key)
                    # Success: reset failures
                    with self._lock:
                        state.failure_count = 0
                    return result
                except Exception as exc:
                    status = self._extract_status(exc)
                    now = time.time()
                    with self._lock:
                        self._prune_timestamps(state, now)
                        if status == 429:
                            state.cooldown_until = now + self.cooldown_429
                            self.logger.warning(f'Key {state.key[:8]} 429 cooldown until {datetime.fromtimestamp(state.cooldown_until).strftime("%H:%M:%S")}')
                        elif status == 503:
                            self.logger.warning(f'Key {state.key[:8]} 503 transient, retrying next')
                        else:
                            self.logger.error(f'Key {state.key[:8]} failed: {exc}')
                        state.failure_count += 1
        
        self.logger.error('Circuit breaker: All keys failed. Pausing...')
        time.sleep(self.cb_pause)
        raise RuntimeError('All keys exhausted after retries')
