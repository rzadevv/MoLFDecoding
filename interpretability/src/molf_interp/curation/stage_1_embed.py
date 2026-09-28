"""Stage 1: embed preferred_label with SapBERT."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from loguru import logger

from molf_interp.curation.config import CurationConfig
from molf_interp.curation.sapbert import SapBert

_TIER_SUFFIX: dict[str, str] = {
    "H2_gene_program": "gene program",
    "H2_pathway": "pathway",
}


def _msigdb_embed_text(label: str, tier: str) -> str:
    """Return tier-suffixed label for MSigDB rows.

    Appends 'gene program' or 'pathway' so the surface form aligns with the reference
    naming style (e.g. 'epithelial mesenchymal transition gene program').
    """
    suffix = _TIER_SUFFIX.get(tier, "")
    if suffix:
        return f"{label} {suffix}"
    return label


def run_stage_1(
    raw_concepts: pd.DataFrame,
    config: CurationConfig,
    sapbert: SapBert,
) -> pd.DataFrame:
    """Add `embedding` column (numpy float32 array) to raw_concepts.

    For MSigDB-sourced rows, averages two embeddings (original label +
    tier-suffixed label) before L2-normalizing, pulling them toward the reference
    bank's region of SapBERT space without losing semantic content.

    Args:
        raw_concepts: Input dataframe with at least a 'preferred_label' and
            'source_name' column.
        config: CurationConfig instance.
        sapbert: SapBert instance for encoding.

    Returns:
        Copy of raw_concepts with 'embedding' column added.
    """
    t0 = time.monotonic()
    labels = raw_concepts["preferred_label"].tolist()
    source_names = raw_concepts["source_name"].tolist()
    tiers = raw_concepts["candidate_tier"].tolist()

    logger.info("Stage 1: embedding {} labels with SapBERT…", len(labels))

    # Base embeddings (all rows)
    base_embs = sapbert.encode(labels, show_progress=True)

    # MSigDB rows: also embed the tier-suffixed text and average
    msigdb_mask = [s.startswith("msigdb_") for s in source_names]
    msigdb_indices = [i for i, m in enumerate(msigdb_mask) if m]

    if msigdb_indices:
        suffixed_texts = [_msigdb_embed_text(labels[i], tiers[i]) for i in msigdb_indices]
        suffixed_embs = sapbert.encode(suffixed_texts)

        for k, i in enumerate(msigdb_indices):
            avg = base_embs[i] + suffixed_embs[k]
            norm = float(np.linalg.norm(avg))
            if norm > 0:
                base_embs[i] = (avg / norm).astype(np.float32)

        logger.info(
            "Stage 1: rehydrated {} MSigDB embeddings with tier-suffix averaging",
            len(msigdb_indices),
        )

    df = raw_concepts.copy()
    df["embedding"] = [base_embs[i] for i in range(len(base_embs))]
    elapsed = time.monotonic() - t0
    logger.info(
        "Stage 1 done: {} embeddings, dim={}, {:.1f}s",
        len(df),
        sapbert.embedding_dim,
        elapsed,
    )
    return df
