"""
tests/test_rate_limit.py
"""
import sys, os
import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rate_limit import RateLimiter


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.check("user1")  # should not raise

    def test_blocks_on_limit_exceeded(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("user1")
        with pytest.raises(HTTPException) as exc:
            limiter.check("user1")
        assert exc.value.status_code == 429

    def test_different_keys_independent(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("user1")
        limiter.check("user1")
        # user2 should still have full quota
        limiter.check("user2")
        limiter.check("user2")

    def test_get_remaining_decrements(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.get_remaining("user1") == 10
        limiter.check("user1")
        assert limiter.get_remaining("user1") == 9

    def test_429_includes_retry_after_header(self):
        limiter = RateLimiter(max_requests=1, window_seconds=30)
        limiter.check("user1")
        with pytest.raises(HTTPException) as exc:
            limiter.check("user1")
        assert "Retry-After" in exc.value.headers
