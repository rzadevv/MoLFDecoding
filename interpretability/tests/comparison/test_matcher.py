"""Tests for cross-bank greedy bipartite matching."""

from __future__ import annotations

import numpy as np

from molf_interp.comparison.matcher import match_banks
from molf_interp.concepts.schemas import (
    ConceptBank,
    ProvenanceSource,
    Tier,
)
from tests.comparison.conftest import MockSapBert, _build_bank, _make_concept


def _bank_from_pairs(
    pairs: list[tuple[str, str, Tier]],
    id_prefix: str,
    provenance: ProvenanceSource,
) -> ConceptBank:
    """Build a ConceptBank from (concept_id, name, tier) triples."""
    concepts = [
        _make_concept(
            cid,
            name,
            tier,
            provenance,
            organ="universal" if tier == Tier.H1_MORPHOLOGY else None,
            level="tissue" if tier == Tier.H1_MORPHOLOGY else None,
            concept_type=None if tier == Tier.H1_MORPHOLOGY else _tier_to_concept_type(tier),
        )
        for cid, name, tier in pairs
    ]
    return _build_bank(concepts, provenance)


def _tier_to_concept_type(tier: Tier) -> str:
    mapping = {
        Tier.H2_CELL_TYPE: "cell_type",
        Tier.H2_NICHE: "niche",
        Tier.H2_GENE_PROGRAM: "gene_program",
        Tier.H2_PATHWAY: "pathway",
    }
    return mapping[tier]


def _unit_vec(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= float(np.linalg.norm(v))
    return v


def _tilted_vec(base: np.ndarray, target_cosine: float) -> np.ndarray:
    """Return a unit vector at approximately target_cosine from base."""
    dim = base.shape[0]
    orth = np.zeros(dim, dtype=np.float32)
    # Orthogonal component: subtract projection, then normalise
    orth[0] = 1.0
    orth -= float(np.dot(orth, base)) * base
    orth_norm = float(np.linalg.norm(orth))
    if orth_norm < 1e-8:
        orth[1] = 1.0
        orth -= float(np.dot(orth, base)) * base
        orth_norm = float(np.linalg.norm(orth))
    orth /= orth_norm
    sin_theta = float(np.sqrt(max(0.0, 1.0 - target_cosine**2)))
    v = target_cosine * base + sin_theta * orth
    v /= float(np.linalg.norm(v))
    return v


def test_perfect_matches_yield_cosine_one(tiny_reference_bank, tiny_harvested_bank) -> None:
    """Identical labels → cosine 1.0, all 4 identical pairs match."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    # 4 identical pairs: coagulative necrosis, nuclear pleomorphism, macrophage, plasma cell
    identical_su = {"MOR-0001", "MOR-0002", "CTY-0001", "CTY-0002"}
    matched_ref = {m.reference_concept_id for m in matches if not m.is_cross_tier}
    assert identical_su == matched_ref
    for m in matches:
        if m.reference_concept_id in identical_su:
            assert abs(m.similarity - 1.0) < 1e-4


def test_greedy_conflict_lower_sim_unmatched() -> None:
    """Ref_A and Ref_B both closest to Auto_X; only Ref_A (higher sim) wins."""
    base = _unit_vec(seed=1)
    vec_a = _tilted_vec(base, 0.96)  # Su_A ~ Auto_X: 0.96
    vec_b = _tilted_vec(base, 0.90)  # Su_B ~ Auto_X: 0.90
    vec_x = base  # Auto_X

    presets = {
        "Su_A": vec_a,
        "Su_B": vec_b,
        "Auto_X": vec_x,
    }
    mock = MockSapBert(presets=presets)

    ref = _bank_from_pairs(
        [("MOR-0001", "Su_A", Tier.H1_MORPHOLOGY), ("MOR-0002", "Su_B", Tier.H1_MORPHOLOGY)],
        "MOR-",
        ProvenanceSource.REFERENCE,
    )
    auto = _bank_from_pairs(
        [("MOR-10001", "Auto_X", Tier.H1_MORPHOLOGY)],
        "MOR-1",
        ProvenanceSource.HARVESTED,
    )

    matches = match_banks(ref, auto, mock, threshold=0.85, cross_tier_threshold=None)
    assert len(matches) == 1
    assert matches[0].reference_concept_id == "MOR-0001"
    assert matches[0].harvested_concept_id == "MOR-10001"


def test_below_threshold_no_match() -> None:
    """Pair with sim 0.80 does not match at threshold=0.85."""
    base = _unit_vec(seed=42)
    vec_ref = _tilted_vec(base, 0.80)
    vec_auto = base

    presets = {"low_sim_ref": vec_ref, "low_sim_auto": vec_auto}
    mock = MockSapBert(presets=presets)

    ref = _bank_from_pairs(
        [("MOR-0001", "low_sim_ref", Tier.H1_MORPHOLOGY)],
        "",
        ProvenanceSource.REFERENCE,
    )
    auto = _bank_from_pairs(
        [("MOR-10001", "low_sim_auto", Tier.H1_MORPHOLOGY)],
        "",
        ProvenanceSource.HARVESTED,
    )
    matches = match_banks(ref, auto, mock, threshold=0.85, cross_tier_threshold=None)
    assert len(matches) == 0


def test_cross_tier_match() -> None:
    """Unmatched ref[H1] and unmatched harv[H2_cell_type] with sim 0.94 → cross-tier match."""
    base = _unit_vec(seed=99)
    vec_ref = _tilted_vec(base, 0.94)
    vec_auto = base

    presets = {"h1_ref_concept": vec_ref, "h2_auto_concept": vec_auto}
    mock = MockSapBert(presets=presets)

    ref = _bank_from_pairs(
        [("MOR-0001", "h1_ref_concept", Tier.H1_MORPHOLOGY)],
        "",
        ProvenanceSource.REFERENCE,
    )
    auto = _bank_from_pairs(
        [("CTY-10001", "h2_auto_concept", Tier.H2_CELL_TYPE)],
        "",
        ProvenanceSource.HARVESTED,
    )
    matches = match_banks(ref, auto, mock, threshold=0.85, cross_tier_threshold=0.92)
    ct_matches = [m for m in matches if m.is_cross_tier]
    assert len(ct_matches) == 1
    assert ct_matches[0].reference_concept_id == "MOR-0001"
    assert ct_matches[0].harvested_concept_id == "CTY-10001"
    assert ct_matches[0].reference_tier == "H1_morphology"
    assert ct_matches[0].harvested_tier == "H2_cell_type"


def test_cross_tier_disabled() -> None:
    """Same cross-tier scenario with cross_tier_threshold=None → no cross-tier match."""
    base = _unit_vec(seed=99)
    vec_ref = _tilted_vec(base, 0.94)
    presets = {"h1_ref_concept": vec_ref, "h2_auto_concept": base}
    mock = MockSapBert(presets=presets)

    ref = _bank_from_pairs(
        [("MOR-0001", "h1_ref_concept", Tier.H1_MORPHOLOGY)],
        "",
        ProvenanceSource.REFERENCE,
    )
    auto = _bank_from_pairs(
        [("CTY-10001", "h2_auto_concept", Tier.H2_CELL_TYPE)],
        "",
        ProvenanceSource.HARVESTED,
    )
    matches = match_banks(ref, auto, mock, threshold=0.85, cross_tier_threshold=None)
    assert all(not m.is_cross_tier for m in matches)


def test_empty_reference_bank(tiny_harvested_bank) -> None:
    """Empty reference bank → empty match list, no exception."""
    empty_ref = ConceptBank(concepts=(), prompts=(), provenance=ProvenanceSource.REFERENCE)
    mock = MockSapBert()
    matches = match_banks(empty_ref, tiny_harvested_bank, mock, threshold=0.85)
    assert matches == []


def test_empty_auto_bank(tiny_reference_bank) -> None:
    """Empty harvested bank → empty match list, no exception."""
    empty_auto = ConceptBank(concepts=(), prompts=(), provenance=ProvenanceSource.HARVESTED)
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, empty_auto, mock, threshold=0.85)
    assert matches == []
