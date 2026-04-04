"""
tests/test_cache.py
"""
import time
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cache import TTLCache


@pytest.fixture
def c():
    return TTLCache()


class TestTTLCache:
    def test_set_and_get(self, c):
        c.set("key", "value", ttl=10)
        assert c.get("key") == "value"

    def test_missing_key_returns_none(self, c):
        assert c.get("nonexistent") is None

    def test_expired_key_returns_none(self, c):
        c.set("key", "value", ttl=0)
        time.sleep(0.01)
        assert c.get("key") is None

    def test_overwrite(self, c):
        c.set("key", "a", ttl=10)
        c.set("key", "b", ttl=10)
        assert c.get("key") == "b"

    def test_delete(self, c):
        c.set("key", "val", ttl=10)
        c.delete("key")
        assert c.get("key") is None

    def test_clear(self, c):
        c.set("a", 1, ttl=10)
        c.set("b", 2, ttl=10)
        c.clear()
        assert c.get("a") is None
        assert c.get("b") is None

    def test_stats(self, c):
        c.set("a", 1, ttl=10)
        c.set("b", 2, ttl=0)
        time.sleep(0.01)
        stats = c.stats()
        assert stats["alive_keys"] == 1

    def test_stores_complex_objects(self, c):
        import pandas as pd
        s = pd.Series([1, 2, 3])
        c.set("series", s, ttl=10)
        result = c.get("series")
        assert list(result) == [1, 2, 3]
