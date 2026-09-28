"""Tests for threshold calibration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from molf_interp.curation.calibration import (
    CalibratedThresholds,
    TierThresholds,
    calibrate_thresholds,
    load_calibrated_thresholds,
    save_calibrated_thresholds,
)
from molf_interp.curation.negative_classes import NegativeClassCentroids


def _make_neg_centroids(dim: int = 768) -> NegativeClassCentroids:
    """Negative centroids orthogonal to axis-0 and axis-1 (where reference concepts cluster)."""

    def _axis(i: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0
        return v

    return NegativeClassCentroids(
        disease=_axis(2),
        anatomy=_axis(3),
        procedure=_axis(4),
        clinical_metadata=_axis(5),
    )


def _make_clustered_sapbert(dim: int = 768) -> MagicMock:
    """Sapbert mock: H1 concepts cluster near e_0, H2CT near e_1."""
    from unittest.mock import MagicMock

    from molf_interp.curation.sapbert import SapBert

    mock = MagicMock(spec=SapBert)
    mock.embedding_dim = dim
    rng = np.random.default_rng(42)

    def _encode(texts: object, **kwargs: object) -> np.ndarray:
        out = []
        for t in list(texts):  # type: ignore[call-overload]
            if str(t).startswith("H1_"):
                base = np.zeros(dim, dtype=np.float32)
                base[0] = 1.0
            elif str(t).startswith("H2CT_"):
                base = np.zeros(dim, dtype=np.float32)
                base[1] = 1.0
            else:
                base = rng.standard_normal(dim).astype(np.float32)
            noise = rng.standard_normal(dim).astype(np.float32) * 0.001
            v = base + noise
            v = (v / np.linalg.norm(v)).astype(np.float32)
            out.append(v)
        return np.array(out, dtype=np.float32)

    mock.encode.side_effect = _encode
    mock.cosine_similarity.side_effect = lambda a, b: (a @ b.T).astype(np.float32)
    return mock  # type: ignore[return-value]


def _make_su_bank_with_concepts() -> MagicMock:
    """Minimal ConceptBank-like mock with 3 H1 + 3 H2_cell_type concepts."""
    from molf_interp.concepts.schemas import Tier

    def _concept(name: str, tier: Tier) -> MagicMock:
        c = MagicMock()
        c.concept_name = name
        c.tier = tier
        return c

    h1_concepts = [_concept(f"H1_{i}", Tier.H1_MORPHOLOGY) for i in range(3)]
    h2ct_concepts = [_concept(f"H2CT_{i}", Tier.H2_CELL_TYPE) for i in range(3)]
    empty: list[MagicMock] = []

    bank = MagicMock()
    bank.filter_concepts.side_effect = lambda tier=None, **kw: {
        Tier.H1_MORPHOLOGY: tuple(h1_concepts),
        Tier.H2_CELL_TYPE: tuple(h2ct_concepts),
        Tier.H2_NICHE: tuple(empty),
        Tier.H2_GENE_PROGRAM: tuple(empty),
        Tier.H2_PATHWAY: tuple(empty),
    }.get(tier, tuple(empty))
    return bank


def test_calibrate_thresholds_returns_valid_structure() -> None:
    sapbert = _make_clustered_sapbert()
    bank = _make_su_bank_with_concepts()
    neg = _make_neg_centroids()

    thresholds = calibrate_thresholds(bank, sapbert, neg)

    assert isinstance(thresholds, CalibratedThresholds)
    for tier_str in ("H1_morphology", "H2_cell_type", "H2_niche", "H2_gene_program", "H2_pathway"):
        t = thresholds.for_tier(tier_str)
        assert isinstance(t, TierThresholds)


def test_calibrate_thresholds_keep_gt_drop() -> None:
    """For non-degenerate input: keep_threshold > drop_threshold."""
    sapbert = _make_clustered_sapbert()
    bank = _make_su_bank_with_concepts()
    neg = _make_neg_centroids()

    # Tiers with < 2 concepts get fallback thresholds (0.55/0.35) which always satisfy keep>drop.
    thresholds = calibrate_thresholds(bank, sapbert, neg)
    for tier_str in ("H1_morphology", "H2_cell_type"):
        t = thresholds.for_tier(tier_str)
        assert t.keep_threshold > t.drop_threshold, (
            f"{tier_str}: keep={t.keep_threshold} <= drop={t.drop_threshold}"
        )


def test_calibrate_higher_percentile_gives_stricter_threshold() -> None:
    """p50 keep_threshold >= p25 keep_threshold."""
    sapbert = _make_clustered_sapbert()
    bank = _make_su_bank_with_concepts()
    neg = _make_neg_centroids()

    t25 = calibrate_thresholds(bank, sapbert, neg, keep_percentile=25)
    t50 = calibrate_thresholds(bank, sapbert, neg, keep_percentile=50)

    for tier_str in ("H1_morphology", "H2_cell_type"):
        # Allow tiny float noise when embeddings are nearly identical
        assert t50.for_tier(tier_str).keep_threshold >= t25.for_tier(tier_str).keep_threshold - 1e-4


def test_save_load_roundtrip(tmp_path: Path) -> None:
    thresholds = CalibratedThresholds(
        h1_morphology=TierThresholds(keep_threshold=0.70, drop_threshold=0.40),
        h2_cell_type=TierThresholds(keep_threshold=0.65, drop_threshold=0.38),
        h2_niche=TierThresholds(keep_threshold=0.60, drop_threshold=0.35),
        h2_gene_program=TierThresholds(keep_threshold=0.80, drop_threshold=0.45),
        h2_pathway=TierThresholds(keep_threshold=0.68, drop_threshold=0.37),
    )
    path = tmp_path / "thresholds.yaml"
    save_calibrated_thresholds(thresholds, path)
    loaded = load_calibrated_thresholds(path)

    for tier_str in ("H1_morphology", "H2_cell_type", "H2_niche", "H2_gene_program", "H2_pathway"):
        orig = thresholds.for_tier(tier_str)
        back = loaded.for_tier(tier_str)
        assert abs(orig.keep_threshold - back.keep_threshold) < 1e-6
        assert abs(orig.drop_threshold - back.drop_threshold) < 1e-6


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_calibrated_thresholds(tmp_path / "nonexistent.yaml")


def test_for_tier_unknown_raises() -> None:
    t = CalibratedThresholds(
        h1_morphology=TierThresholds(keep_threshold=0.70, drop_threshold=0.40),
        h2_cell_type=TierThresholds(keep_threshold=0.65, drop_threshold=0.38),
        h2_niche=TierThresholds(keep_threshold=0.60, drop_threshold=0.35),
        h2_gene_program=TierThresholds(keep_threshold=0.80, drop_threshold=0.45),
        h2_pathway=TierThresholds(keep_threshold=0.68, drop_threshold=0.37),
    )
    with pytest.raises(KeyError):
        t.for_tier("UNKNOWN_TIER")
