"""Tests for Stage 2: cross-source dedup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from molf_interp.curation.config import CurationConfig, DedupConfig
from molf_interp.curation.stage_2_dedup import _pick_canonical, run_stage_2
from tests.curation.conftest import make_mock_sapbert


def _sim_cluster_df(
    labels: list[str],
    source: str,
    tier: str,
    embedding_dim: int = 768,
) -> pd.DataFrame:
    """Build dataframe where first N-1 rows have near-identical embeddings."""
    n = len(labels)
    base = np.random.default_rng(42).standard_normal(embedding_dim).astype(np.float32)
    base /= np.linalg.norm(base)
    embs = []
    for i in range(n):
        noise = np.random.default_rng(i).standard_normal(embedding_dim).astype(np.float32) * 0.02
        e = base + noise
        e /= np.linalg.norm(e)
        embs.append(e)

    return pd.DataFrame(
        {
            "source_name": [source] * n,
            "source_id": [f"{source}:{i}" for i in range(n)],
            "preferred_label": labels,
            "synonyms": [[] for _ in range(n)],
            "definition": [None] * n,
            "parent_ids": [[] for _ in range(n)],
            "candidate_tier": [tier] * n,
            "extra": [[] for _ in range(n)],
            "embedding": embs,
        }
    )


def test_similar_labels_same_tier_collapsed(tmp_config: object) -> None:
    cfg = CurationConfig(dedup=DedupConfig(method="agglomerative", similarity_threshold=0.85))
    # Build df with near-identical embeddings so agglomerative clustering merges them
    n = 3
    base = np.ones(768, dtype=np.float32) / np.sqrt(768)
    # Tiny noise so vectors are very similar (cosine > 0.99)
    embs = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        noise = rng.standard_normal(768).astype(np.float32) * 0.001
        e = base + noise
        e /= np.linalg.norm(e)
        embs.append(e)

    df = pd.DataFrame(
        {
            "source_name": ["ncit"] * n,
            "source_id": [f"ncit:{i}" for i in range(n)],
            "preferred_label": [
                "coagulative necrosis",
                "Coagulative Necrosis",
                "necrosis coagulative",
            ],
            "synonyms": [[] for _ in range(n)],
            "definition": [None] * n,
            "parent_ids": [[] for _ in range(n)],
            "candidate_tier": ["H1_morphology"] * n,
            "extra": [[] for _ in range(n)],
            "embedding": embs,
        }
    )
    sapbert = make_mock_sapbert()

    result = run_stage_2(df, cfg, sapbert)
    assert len(result) < len(df)


def test_similar_labels_different_tiers_not_collapsed(tmp_config: object) -> None:
    cfg = CurationConfig(dedup=DedupConfig(method="agglomerative", similarity_threshold=0.85))
    # Use a single near-identical embedding for each — but different tiers
    base = np.ones(768, dtype=np.float32) / np.sqrt(768)
    df = pd.DataFrame(
        {
            "source_name": ["ncit", "pubmed"],
            "source_id": ["ncit:0", "pubmed:0"],
            "preferred_label": ["necrosis", "necrosis"],
            "synonyms": [[], []],
            "definition": [None, None],
            "parent_ids": [[], []],
            "candidate_tier": ["H1_morphology", "H2_cell_type"],
            "extra": [[], []],
            "embedding": [base, base],
        }
    )
    sapbert = make_mock_sapbert()

    result = run_stage_2(df, cfg, sapbert)
    # One per tier — dedup is within each tier
    assert len(result) == 2


def test_canonical_priority_snomed_over_pubmed() -> None:
    cfg = DedupConfig()
    df = pd.DataFrame(
        {
            "source_name": ["pubmed", "snomed"],
            "preferred_label": ["necrosis ab", "necrosis"],
        }
    )
    idx = _pick_canonical(df, cfg)
    # snomed has higher priority (lower rank)
    assert df.loc[idx, "source_name"] == "snomed"
