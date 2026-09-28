"""SapBERT embedding model wrapper with on-disk parquet caching."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from molf_interp.curation.config import SapBertConfig

_EMBEDDING_DIM = 768
_CACHE_SCHEMA = pa.schema(
    [
        pa.field("text_hash", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
    ]
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SapBert:
    """SapBERT wrapper with lazy loading and parquet embedding cache."""

    def __init__(self, config: SapBertConfig) -> None:
        self._config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._cache: dict[str, npt.NDArray[Any]] = {}
        self._cache_dirty: bool = False
        self._last_saved_size: int = 0
        self._loaded_embedding_dim: int | None = None

    def load(self) -> None:
        """Load model and tokenizer. Idempotent."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        t0 = time.monotonic()
        device = self._config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        torch.set_num_threads(os.cpu_count() or 1)

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self._model = AutoModel.from_pretrained(self._config.model_name).to(device)
        self._model.eval()
        self._loaded_embedding_dim = self._model.config.hidden_size
        # Merge any entries already in self._cache (from pre-load encode calls)
        # with the disk cache, without losing in-memory entries
        disk_cache = self._load_cache()
        disk_cache.update(self._cache)  # in-memory entries take priority
        self._cache = disk_cache
        self._last_saved_size = len(self._cache)
        logger.info(
            "Loaded SapBERT '{}' on {} in {:.1f}s (cache: {} entries)",
            self._config.model_name,
            device,
            time.monotonic() - t0,
            len(self._cache),
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (derived from model if loaded, else default 768)."""
        if self._loaded_embedding_dim is not None:
            return self._loaded_embedding_dim
        return _EMBEDDING_DIM

    def _cache_path(self) -> Path:
        return self._config.cache_dir / "embeddings.parquet"

    def _load_cache(self) -> dict[str, npt.NDArray[Any]]:
        path = self._cache_path()
        if not path.exists():
            return {}
        try:
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            d = table.to_pydict()
            return {
                h: np.array(e, dtype=np.float32)
                for h, e in zip(d["text_hash"], d["embedding"], strict=True)
            }
        except Exception as exc:
            logger.warning("Failed to load SapBERT cache: {}; starting fresh", exc)
            return {}

    def _save_cache(self, *, force: bool = False) -> None:
        """Persist the cache to disk.

        Args:
            force: When False (default) saves only if >200 new entries accumulated.
                   When True, always writes if dirty. Call flush() at end of each stage.
        """
        if not self._cache_dirty:
            return
        new_entries = len(self._cache) - self._last_saved_size
        if not force and new_entries < 200:
            return
        path = self._cache_path()
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "text_hash": list(self._cache.keys()),
                "embedding": [e.tolist() for e in self._cache.values()],
            },
            schema=_CACHE_SCHEMA,
        )
        pq.write_table(table, path)  # type: ignore[no-untyped-call]
        self._cache_dirty = False
        self._last_saved_size = len(self._cache)

    def flush(self) -> None:
        """Force-flush cache to disk. Call at the end of each pipeline stage."""
        self._save_cache(force=True)

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> npt.NDArray[Any]:
        """Encode texts to L2-normalized [N, 768] float32 numpy array.

        Loads model on first call. Uses cache for previously encoded texts.

        Args:
            texts: Sequence of strings to encode.
            show_progress: Unused; reserved for future tqdm integration.

        Returns:
            Float32 array of shape [N, 768].
        """
        import torch

        texts_list = list(texts)
        n = len(texts_list)
        if n == 0:
            return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)

        hashes = [_sha256(t) for t in texts_list]
        result = np.zeros((n, _EMBEDDING_DIM), dtype=np.float32)

        # Load on-disk cache before first miss check so cached entries are found
        if not self._cache:
            disk_cache = self._load_cache()
            self._cache.update(disk_cache)
            self._last_saved_size = len(self._cache)

        # Populate from cache
        miss_indices: list[int] = []
        for i, h in enumerate(hashes):
            if h in self._cache:
                result[i] = self._cache[h]
            else:
                miss_indices.append(i)

        if miss_indices:
            self.load()
            # Re-check after load() populates full cache (handles race with _load_cache)
            still_miss: list[int] = []
            for i in miss_indices:
                if hashes[i] in self._cache:
                    result[i] = self._cache[hashes[i]]
                else:
                    still_miss.append(i)
            miss_indices = still_miss

        if miss_indices:
            miss_texts = [texts_list[i] for i in miss_indices]
            batch_size = self._config.batch_size

            for batch_start in range(0, len(miss_texts), batch_size):
                batch = miss_texts[batch_start : batch_start + batch_size]
                toks = self._tokenizer.batch_encode_plus(
                    batch,
                    padding="max_length",
                    max_length=self._config.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                toks = {k: v.to(self._device) for k, v in toks.items()}
                with torch.inference_mode():
                    cls = self._model(**toks)[0][:, 0, :].cpu().numpy()
                # L2 normalize
                norms = np.linalg.norm(cls, axis=-1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                cls = (cls / norms).astype(np.float32)

                for j, (idx, h) in enumerate(
                    zip(
                        miss_indices[batch_start : batch_start + batch_size],
                        [_sha256(t) for t in batch],
                        strict=True,
                    )
                ):
                    result[idx] = cls[j]
                    self._cache[h] = cls[j]
                    self._cache_dirty = True

            self._save_cache()

        return result

    def cosine_similarity(self, a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute cosine similarities between two L2-normalized arrays.

        Args:
            a: Array of shape [N, D], L2-normalized.
            b: Array of shape [M, D], L2-normalized.

        Returns:
            Similarity matrix of shape [N, M].
        """
        result: npt.NDArray[Any] = a @ b.T
        return result

    def encode_and_cache_only(self, texts: Sequence[str]) -> None:
        """Pre-populate cache without returning the result.

        Args:
            texts: Texts to encode and cache.
        """
        self.encode(texts)
