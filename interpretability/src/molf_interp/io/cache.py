"""Content-hash-based cache path resolution."""

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_cache_key(components: dict[str, Any]) -> str:
    """Compute a SHA-256 cache key from a dict of components.

    Args:
        components: Arbitrary key-value pairs describing the computation.

    Returns:
        64-character hex digest uniquely identifying the input.
    """
    canonical = json.dumps(components, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ContentCache:
    """Resolves file paths for content-addressed cache entries.

    Does not implement read/write; only key derivation and path resolution.
    """

    def __init__(self, root: Path) -> None:
        """Initialise the cache, creating root if it does not exist.

        Args:
            root: Directory under which all cache files live.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str, suffix: str = ".parquet") -> Path:
        """Return the cache file path for a given key.

        Args:
            key: Cache key (typically from compute_cache_key).
            suffix: File extension for the cached artifact.

        Returns:
            Absolute path where the artifact should be stored.
        """
        return self._root / f"{key}{suffix}"

    def exists(self, key: str, suffix: str = ".parquet") -> bool:
        """Check whether a cached artifact exists on disk.

        Args:
            key: Cache key.
            suffix: File extension.

        Returns:
            True if the file is present, False otherwise.
        """
        return self.path_for(key, suffix).exists()

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return f"ContentCache(root={self._root})"
