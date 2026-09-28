"""Tests for ResponseCache (moved from bioportal/)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from molf_interp.sources._common.cache import ResponseCache


@pytest.fixture()
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache")


def test_key_deterministic(cache: ResponseCache) -> None:
    k1 = cache._key("https://example.com/foo", {"a": 1, "b": 2})
    k2 = cache._key("https://example.com/foo", {"b": 2, "a": 1})
    assert k1 == k2


def test_key_distinct_for_different_inputs(cache: ResponseCache) -> None:
    k1 = cache._key("https://example.com/foo", {"a": 1})
    k2 = cache._key("https://example.com/bar", {"a": 1})
    assert k1 != k2


def test_key_none_equals_empty_dict(cache: ResponseCache) -> None:
    assert cache._key("https://example.com", None) == cache._key("https://example.com", {})


def test_put_and_get_round_trip(cache: ResponseCache) -> None:
    payload: dict[str, Any] = {"foo": "bar", "count": 42}
    cache.put("https://example.com/x", {"p": 1}, payload)
    result = cache.get("https://example.com/x", {"p": 1})
    assert result == payload


def test_get_miss_returns_none(cache: ResponseCache) -> None:
    assert cache.get("https://example.com/missing") is None


def test_invalidate_removes_entry(cache: ResponseCache) -> None:
    cache.put("https://example.com/y", None, {"data": True})
    assert cache.invalidate("https://example.com/y") is True
    assert cache.get("https://example.com/y") is None


def test_invalidate_missing_returns_false(cache: ResponseCache) -> None:
    assert cache.invalidate("https://example.com/nonexistent") is False


def test_files_stored_in_sharded_layout(cache: ResponseCache) -> None:
    url = "https://example.com/shard-test"
    cache.put(url, None, {"x": 1})
    key = cache._key(url, None)
    expected = cache._root / key[:2] / f"{key}.json"
    assert expected.exists()


def test_concurrent_puts_no_corruption(tmp_path: Path) -> None:
    c = ResponseCache(tmp_path / "conc")
    url1 = "https://example.com/conc1"
    url2 = "https://example.com/conc2"

    async def _run() -> None:
        await asyncio.gather(
            asyncio.to_thread(c.put, url1, None, {"k": "v1"}),
            asyncio.to_thread(c.put, url2, None, {"k": "v2"}),
        )

    asyncio.run(_run())
    assert c.get(url1) == {"k": "v1"}
    assert c.get(url2) == {"k": "v2"}


def test_stats(cache: ResponseCache) -> None:
    cache.put("https://example.com/a", None, {"a": 1})
    cache.put("https://example.com/b", None, {"b": 2})
    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["bytes"] > 0
