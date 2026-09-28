"""Shared fixtures for curation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from molf_interp.curation.config import (
    CanonicalizeConfig,
    CurationConfig,
    DedupConfig,
    LLMAdjudicationConfig,
    PromptGenConfig,
    SapBertConfig,
    TierRefineConfig,
    VisualFilterConfig,
)
from molf_interp.curation.sapbert import SapBert


@pytest.fixture
def tmp_config(tmp_path: Path) -> CurationConfig:
    """Return a CurationConfig wired to tmp_path."""
    return CurationConfig(
        input_path=tmp_path / "raw.parquet",
        output_dir=tmp_path / "out",
        stage_cache_dir=tmp_path / "stages",
        sapbert=SapBertConfig(cache_dir=tmp_path / "sapbert_cache"),
        dedup=DedupConfig(method="agglomerative", similarity_threshold=0.85),
        visual_filter=VisualFilterConfig(
            reference_bank_dir=tmp_path / "reference_bank",
            similarity_keep_threshold=0.55,
            similarity_drop_threshold=0.35,
        ),
        llm=LLMAdjudicationConfig(enabled=False),
        tier_refine=TierRefineConfig(
            uberon_table_path=tmp_path / "uberon.parquet",
            organ_default="universal",
        ),
        canonicalize=CanonicalizeConfig(id_offset=10000, sentence_case=True),
        prompt_gen=PromptGenConfig(
            templates_path=Path("configs/curation/prompt_templates.yaml"),
        ),
    )


def make_mock_sapbert(embedding_dim: int = 768) -> SapBert:
    """Return a SapBert mock that returns deterministic normalized embeddings."""
    mock = MagicMock(spec=SapBert)
    mock.embedding_dim = embedding_dim

    def _encode(texts: Any, **kwargs: Any) -> np.ndarray:
        texts_list = list(texts)
        n = len(texts_list)
        rng = np.random.default_rng(seed=abs(hash(str(texts_list))) % (2**31))
        embs = rng.standard_normal((n, embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / norms).astype(np.float32)

    mock.encode.side_effect = _encode

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a @ b.T).astype(np.float32)

    mock.cosine_similarity.side_effect = _cosine_sim
    return mock  # type: ignore[return-value]


def make_raw_concepts_df(
    n: int = 10,
    tier: str = "H1_morphology",
    source: str = "ncit",
    embedding_dim: int = 768,
) -> pd.DataFrame:
    """Make a small synthetic raw_concepts dataframe with embeddings."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, embedding_dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = (embs / norms).astype(np.float32)

    return pd.DataFrame(
        {
            "source_name": [source] * n,
            "source_id": [f"{source}:{i}" for i in range(n)],
            "preferred_label": [f"concept {i} label" for i in range(n)],
            "synonyms": [[] for _ in range(n)],
            "definition": [None] * n,
            "parent_ids": [[] for _ in range(n)],
            "candidate_tier": [tier] * n,
            "extra": [[] for _ in range(n)],
            "embedding": [embs[i] for i in range(n)],
        }
    )
