"""
rate_limit.py — Sliding-window rate limiter.

Tracks request timestamps per IP. Raises HTTP 429 when limit exceeded.
No external dependencies.
"""
import time
import threading
from collections import defaultdict, deque
from fastapi import Request, HTTPException


class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests    = max_requests
        self.window_seconds  = window_seconds
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets[key]
            # evict old timestamps
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.max_requests} requests "
                           f"per {self.window_seconds}s. Try again shortly.",
                    headers={"Retry-After": str(self.window_seconds)},
                )
            bucket.append(now)

    def get_remaining(self, key: str) -> int:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets[key]
            count = sum(1 for ts in bucket if ts >= cutoff)
            return max(0, self.max_requests - count)


# Singletons for different endpoint groups
backtest_limiter  = RateLimiter(max_requests=20, window_seconds=60)
portfolio_limiter = RateLimiter(max_requests=10, window_seconds=60)
watchlist_limiter = RateLimiter(max_requests=60, window_seconds=60)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
