from molf_interp.io.cache import ContentCache, compute_cache_key


def test_cache_key_deterministic():
    key1 = compute_cache_key({"a": 1, "b": 2})
    key2 = compute_cache_key({"a": 1, "b": 2})
    assert key1 == key2


def test_cache_key_order_independent():
    key1 = compute_cache_key({"a": 1, "b": 2})
    key2 = compute_cache_key({"b": 2, "a": 1})
    assert key1 == key2


def test_cache_key_differs_for_different_inputs():
    key1 = compute_cache_key({"a": 1})
    key2 = compute_cache_key({"a": 2})
    assert key1 != key2


def test_cache_path_under_root(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    key = compute_cache_key({"x": 42})
    path = cache.path_for(key)
    assert path.parent == tmp_path / "cache"
    assert path.name.startswith(key)
