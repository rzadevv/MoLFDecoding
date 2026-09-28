"""On-disk JSON cache for HTTP responses — shared across NCBI and BioPortal clients."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


class ResponseCache:
    """On-disk cache for HTTP JSON or JSON-wrapped responses.

    Layout:
        {cache_dir}/{first_two_hex_chars}/{full_hash}.json

    Each file stores:
        {"url": str, "params": dict, "response": dict|list, "fetched_at": iso}

    Write safety: uses tempfile + atomic rename so concurrent async tasks
    within the same process cannot corrupt a cache entry.
    """

    def __init__(self, root: Path) -> None:
        """Initialise the cache rooted at root, creating it if absent.

        Args:
            root: Directory to store cached response files.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str, params: dict[str, Any] | None) -> str:
        """Compute a stable SHA-256 key for url + sorted params.

        Args:
            url: The request URL.
            params: Optional query parameters; None and {} hash identically.

        Returns:
            64-character lowercase hex digest.
        """
        normalized = urlencode(sorted((params or {}).items()))
        canonical = f"{url}?{normalized}" if normalized else url
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Return the cached response payload, or None on a miss.

        Args:
            url: The request URL.
            params: Optional query parameters.

        Returns:
            The cached response dict, or None if not cached.
        """
        p = self._path(self._key(url, params))
        if not p.exists():
            return None
        try:
            envelope: dict[str, Any] = json.loads(p.read_text())
            return dict(envelope["response"])
        except (KeyError, json.JSONDecodeError):
            return None

    def put(
        self,
        url: str,
        params: dict[str, Any] | None,
        response: dict[str, Any],
    ) -> None:
        """Cache a successful response. Uses atomic rename for safety.

        Args:
            url: The request URL.
            params: Optional query parameters.
            response: The response body to cache.
        """
        key = self._key(url, params)
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        envelope = {
            "url": url,
            "params": params or {},
            "response": response,
            "fetched_at": datetime.now(tz=UTC).isoformat(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=dest.parent,
            suffix=".tmp",
            delete=False,
        ) as fh:
            json.dump(envelope, fh)
            tmp_path = Path(fh.name)

        tmp_path.replace(dest)

    def invalidate(self, url: str, params: dict[str, Any] | None = None) -> bool:
        """Remove a cached entry.

        Args:
            url: The request URL.
            params: Optional query parameters.

        Returns:
            True if an entry was removed, False if it was not present.
        """
        p = self._path(self._key(url, params))
        if p.exists():
            p.unlink()
            return True
        return False

    def stats(self) -> dict[str, int]:
        """Return total cached entries and total bytes on disk.

        Returns:
            Dict with keys 'entries' and 'bytes'.
        """
        total_entries = 0
        total_bytes = 0
        for f in self._root.rglob("*.json"):
            total_entries += 1
            total_bytes += f.stat().st_size
        return {"entries": total_entries, "bytes": total_bytes}
