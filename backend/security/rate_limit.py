"""Simple in-process sliding-window rate limiter (per user). Use a gateway/Redis limiter for multi-instance deployments."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


WINDOW_SECONDS = 60
SWEEP_SECONDS = 60          # how often keys with no hit in the last window are dropped (bounded memory)


class RateLimiter:
    def __init__(self, per_minute: int, clock=time.monotonic) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._clock = clock
        self._last_sweep = clock()

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = self._clock()
        with self._lock:
            if now - self._last_sweep >= SWEEP_SECONDS:
                self._sweep(now)
            hits = self._hits[key]
            while hits and now - hits[0] > WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True

    def _sweep(self, now: float) -> None:
        """Drop every key whose newest hit is older than the window (one entry per active client only)."""
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] > WINDOW_SECONDS]:
            del self._hits[key]
        self._last_sweep = now

    def __len__(self) -> int:
        return len(self._hits)
