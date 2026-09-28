"""Tests for Stage 3 v2: KNN + negative-class proximity filter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from molf_interp.curation.calibration import CalibratedThresholds, TierThresholds
from molf_interp.curation.config import CurationConfig
from molf_interp.curation.negative_classes import NegativeClassCentroids
from molf_interp.curation.stage_3_visual_filter import (
    _compute_knn_agreement,
    _decide,
    run_stage_3,
)
from tests.curation.conftest import make_mock_sapbert

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dim_zeros(dim: int = 768) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _unit(idx: int, dim: int = 768) -> np.ndarray:
    v = _dim_zeros(dim)
    v[idx] = 1.0
    return v


def _neg_centroids_all_zeros(dim: int = 768) -> NegativeClassCentroids:
    return NegativeClassCentroids(
        disease=_dim_zeros(dim),
        anatomy=_dim_zeros(dim),
        procedure=_dim_zeros(dim),
        clinical_metadata=_dim_zeros(dim),
    )


def _thresholds(keep: float, drop: float, knn_agree: float = 0.5) -> CalibratedThresholds:
    t = TierThresholds(
        keep_threshold=keep, drop_threshold=drop, knn_tier_agreement_threshold=knn_agree
    )
    return CalibratedThresholds(
        h1_morphology=t,
        h2_cell_type=t,
        h2_niche=t,
        h2_gene_program=t,
        h2_pathway=t,
    )


def _make_bank_with_su_embs(
    su_embeddings: list[np.ndarray],
    reference_tier: str = "H1_morphology",
) -> MagicMock:
    from molf_interp.concepts.schemas import Tier

    tier_enum = Tier(reference_tier)
    concepts = []
    for i, _emb in enumerate(su_embeddings):
        c = MagicMock()
        c.concept_name = f"su_{i}"
        c.tier = tier_enum
        concepts.append(c)

    bank = MagicMock()
    bank.filter_concepts.side_effect = lambda tier=None, **kw: (
        tuple(concepts) if tier == tier_enum else ()
    )
    return bank


def _make_df(
    embeddings: list[np.ndarray],
    tier: str = "H1_morphology",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "preferred_label": [f"concept_{i}" for i in range(len(embeddings))],
            "candidate_tier": [tier] * len(embeddings),
            "source_name": ["ncit"] * len(embeddings),
            "embedding": embeddings,
        }
    )


# ---------------------------------------------------------------------------
# _decide (backward-compat helper)
# ---------------------------------------------------------------------------


def test_decide_keep() -> None:
    assert _decide(0.7, 0.55, 0.35) == "keep"


def test_decide_drop() -> None:
    assert _decide(0.2, 0.55, 0.35) == "drop"


def test_decide_ambiguous() -> None:
    assert _decide(0.45, 0.55, 0.35) == "ambiguous"


# ---------------------------------------------------------------------------
# _compute_knn_agreement
# ---------------------------------------------------------------------------


def test_knn_agreement_all_same_tier() -> None:
    dim = 8
    cand = np.array([_unit(0, dim)])
    su_embs = np.array([_unit(0, dim), _unit(0, dim), _unit(1, dim)])
    ref_tiers = ["H1_morphology", "H1_morphology", "H1_morphology"]
    agreements, _tops = _compute_knn_agreement(
        cand, ["H1_morphology"], su_embs, ref_tiers, k=3, batch_size=256
    )
    assert agreements[0] == pytest.approx(1.0)


def test_knn_agreement_half_matching() -> None:
    dim = 8
    # 5 reference concepts for same tier, 5 for different
    su_embs = np.vstack([_unit(0, dim)] * 5 + [_unit(1, dim)] * 5)
    ref_tiers = ["H1_morphology"] * 5 + ["H2_cell_type"] * 5
    cand = np.array([_unit(0, dim)])  # closest to all 5 H1 + varies
    # With k=10 all 10 neighbors found; exactly 5/10 are same tier
    agreements, _ = _compute_knn_agreement(
        cand, ["H1_morphology"], su_embs, ref_tiers, k=10, batch_size=256
    )
    assert agreements[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# run_stage_3 — decision categories
# ---------------------------------------------------------------------------


def test_drop_negative_proximity() -> None:
    """Candidate closer to negative class than to reference tier → drop_negative_proximity."""
    dim = 768
    sapbert = make_mock_sapbert()

    # Reference centroid roughly along dim=0; negative strongly along same dim
    su_emb = _unit(0, dim)
    bank = _make_bank_with_su_embs([su_emb])
    df = _make_df([_unit(0, dim)])

    # Negative class centroid also along dim=0 → max_neg_sim == ref_sim
    neg = NegativeClassCentroids(
        disease=_unit(0, dim),
        anatomy=_dim_zeros(dim),
        procedure=_dim_zeros(dim),
        clinical_metadata=_dim_zeros(dim),
    )

    # With mock sapbert, the centroid is computed from su_embs via sapbert.encode
    # Override to return controlled embeddings
    sapbert.encode.side_effect = lambda texts, **kw: np.array(
        [su_emb] * len(texts), dtype=np.float32
    )

    result = run_stage_3(
        df,
        CurationConfig(),
        sapbert,
        reference_bank=bank,
        negative_centroids=neg,
        thresholds=_thresholds(keep=0.5, drop=0.1),
    )
    # The candidate should be dropped (negative proximity ≥ reference similarity)
    assert len(result) == 0 or "drop" in result["visual_filter_decision"].iloc[0]


def test_drop_low_similarity() -> None:
    """Candidate with very low similarity to reference tier centroid → drop_low_similarity."""
    dim = 768
    sapbert = make_mock_sapbert()

    su_emb = _unit(0, dim)
    bank = _make_bank_with_su_embs([su_emb])

    # Candidate orthogonal to reference → similarity ≈ 0
    cand_emb = _unit(1, dim)
    df = _make_df([cand_emb])

    sapbert.encode.side_effect = lambda texts, **kw: np.array(
        [su_emb] * len(texts), dtype=np.float32
    )

    # With no negative centroids and drop_threshold=0.5 > sim≈0 → drop_low_similarity
    result = run_stage_3(
        df,
        CurationConfig(),
        sapbert,
        reference_bank=bank,
        thresholds=_thresholds(keep=0.8, drop=0.5),
    )
    assert len(result) == 0


def test_keep_high_similarity_high_agreement() -> None:
    """Candidate with high sim and high knn agreement → keep."""
    dim = 768
    sapbert = make_mock_sapbert()

    su_emb = _unit(0, dim)
    bank = _make_bank_with_su_embs([su_emb] * 10)

    # Candidate identical to reference concepts → sim=1.0, agreement=1.0
    cand_emb = _unit(0, dim)
    df = _make_df([cand_emb])

    sapbert.encode.side_effect = lambda texts, **kw: np.array(
        [su_emb] * len(texts), dtype=np.float32
    )

    result = run_stage_3(
        df,
        CurationConfig(),
        sapbert,
        reference_bank=bank,
        negative_centroids=_neg_centroids_all_zeros(),
        thresholds=_thresholds(keep=0.5, drop=0.1, knn_agree=0.5),
    )
    assert len(result) == 1
    assert result["visual_filter_decision"].iloc[0] == "keep"


def test_ambiguous_midrange_similarity() -> None:
    """Candidate between drop and keep thresholds → ambiguous."""
    dim = 768
    sapbert = make_mock_sapbert()

    su_emb = _unit(0, dim)
    bank = _make_bank_with_su_embs([su_emb] * 10)

    cand_emb = _unit(0, dim)  # sim=1.0 with centroid
    df = _make_df([cand_emb])

    sapbert.encode.side_effect = lambda texts, **kw: np.array(
        [su_emb] * len(texts), dtype=np.float32
    )

    # Set keep_threshold > 1.0 so nothing can keep; drop_threshold < 0 so nothing drops
    result = run_stage_3(
        df,
        CurationConfig(),
        sapbert,
        reference_bank=bank,
        negative_centroids=_neg_centroids_all_zeros(),
        thresholds=_thresholds(keep=1.5, drop=-0.5),
    )
    assert len(result) == 1
    assert result["visual_filter_decision"].iloc[0] == "ambiguous"


def test_drop_low_tier_agreement() -> None:
    """Candidate with low KNN tier agreement → drop_low_tier_agreement."""
    dim = 768
    sapbert = make_mock_sapbert()

    # Reference bank: mix of H1 and H2 concepts
    su_emb_h1 = _unit(0, dim)
    su_emb_h2 = _unit(1, dim)
    bank = MagicMock()
    from molf_interp.concepts.schemas import Tier

    def _filter(tier=None, **kw):  # type: ignore[return]
        if tier == Tier.H1_MORPHOLOGY:
            c = MagicMock()
            c.concept_name = "h1_concept"
            return (c,)
        if tier == Tier.H2_CELL_TYPE:
            c = MagicMock()
            c.concept_name = "h2_concept"
            return (c,) * 9  # 9 H2 vs 1 H1 → low agreement for H1 candidate
        return ()

    bank.filter_concepts.side_effect = _filter

    su_names_h1 = ["h1_concept"]

    def _encode(texts, **kw):  # type: ignore[return]
        texts_list = list(texts)
        out = []
        for t in texts_list:
            if t in su_names_h1 or t == "h1_concept":
                out.append(su_emb_h1)
            else:
                out.append(su_emb_h2)
        return np.array(out, dtype=np.float32)

    sapbert.encode.side_effect = _encode

    # Candidate similar to H1 reference concept but surrounded by H2 neighbors
    cand_emb = _unit(0, dim)  # same as H1
    df = _make_df([cand_emb], tier="H1_morphology")

    result = run_stage_3(
        df,
        CurationConfig(),
        sapbert,
        reference_bank=bank,
        negative_centroids=_neg_centroids_all_zeros(),
        thresholds=_thresholds(keep=0.5, drop=-0.5, knn_agree=0.99),
    )
    # knn_agreement = 1/10 < 0.99 → drop_low_tier_agreement
    assert len(result) == 0 or result["visual_filter_decision"].iloc[0].startswith("drop")
