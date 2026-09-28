"""Shared fixtures for comparison module tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from molf_interp.concepts.schemas import (
    Concept,
    ConceptBank,
    ConceptPrompt,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)


def _make_concept(
    concept_id: str,
    concept_name: str,
    tier: Tier,
    provenance: ProvenanceSource,
    organ: str | None = None,
    level: str | None = None,
    concept_type: str | None = None,
    category: str = "test",
    subcategory: str = "test",
    source: str = "test",
) -> Concept:
    return Concept(
        concept_id=concept_id,
        concept_name=concept_name,
        tier=tier,
        category=category,
        subcategory=subcategory,
        source=source,
        provenance=provenance,
        organ=organ,
        level=level,
        concept_type=concept_type,
    )


def _make_prompts(concept: Concept) -> list[ConceptPrompt]:
    return [
        ConceptPrompt(
            concept_id=concept.concept_id,
            concept_name=concept.concept_name,
            tier=concept.tier,
            species=sp,
            prompt_text=f"H&E image showing {concept.concept_name}",
            species_status=SpeciesStatus.SHARED,
        )
        for sp in Species
    ]


def _build_bank(concepts: list[Concept], provenance: ProvenanceSource) -> ConceptBank:
    prompts: list[ConceptPrompt] = []
    for c in concepts:
        prompts.extend(_make_prompts(c))
    return ConceptBank(concepts=tuple(concepts), prompts=tuple(prompts), provenance=provenance)


@pytest.fixture
def tiny_reference_bank() -> ConceptBank:
    """Tiny reference bank: 2 H1 + 2 H2_cell_type + 1 H2_gene_program + 1 H2_pathway."""
    concepts = [
        _make_concept(
            "MOR-0001",
            "coagulative necrosis",
            Tier.H1_MORPHOLOGY,
            ProvenanceSource.REFERENCE,
            organ="universal",
            level="tissue",
        ),
        _make_concept(
            "MOR-0002",
            "nuclear pleomorphism",
            Tier.H1_MORPHOLOGY,
            ProvenanceSource.REFERENCE,
            organ="universal",
            level="cellular",
        ),
        _make_concept(
            "CTY-0001",
            "macrophage",
            Tier.H2_CELL_TYPE,
            ProvenanceSource.REFERENCE,
            concept_type="cell_type",
        ),
        _make_concept(
            "CTY-0002",
            "plasma cell",
            Tier.H2_CELL_TYPE,
            ProvenanceSource.REFERENCE,
            concept_type="cell_type",
        ),
        _make_concept(
            "GPR-0001",
            "hallmark myc targets v1",
            Tier.H2_GENE_PROGRAM,
            ProvenanceSource.REFERENCE,
            concept_type="gene_program",
        ),
        _make_concept(
            "PWY-0001",
            "pi3k akt mtor signaling",
            Tier.H2_PATHWAY,
            ProvenanceSource.REFERENCE,
            concept_type="pathway",
        ),
    ]
    return _build_bank(concepts, ProvenanceSource.REFERENCE)


@pytest.fixture
def tiny_harvested_bank() -> ConceptBank:
    """Tiny harvested bank: 4 matching reference + 4 novel."""
    concepts = [
        # Matching reference entries (identical names → cosine 1.0 with mock_sapbert)
        _make_concept(
            "MOR-10001",
            "coagulative necrosis",
            Tier.H1_MORPHOLOGY,
            ProvenanceSource.HARVESTED,
            organ="universal",
            level="tissue",
        ),
        _make_concept(
            "MOR-10002",
            "nuclear pleomorphism",
            Tier.H1_MORPHOLOGY,
            ProvenanceSource.HARVESTED,
            organ="universal",
            level="cellular",
        ),
        _make_concept(
            "CTY-10001",
            "macrophage",
            Tier.H2_CELL_TYPE,
            ProvenanceSource.HARVESTED,
            concept_type="cell_type",
        ),
        _make_concept(
            "CTY-10002",
            "plasma cell",
            Tier.H2_CELL_TYPE,
            ProvenanceSource.HARVESTED,
            concept_type="cell_type",
        ),
        # Novel entries (unique names → low cosine with reference bank)
        _make_concept(
            "MOR-10003",
            "zxqv_unique_morphology_alpha",
            Tier.H1_MORPHOLOGY,
            ProvenanceSource.HARVESTED,
            organ="universal",
            level="tissue",
        ),
        _make_concept(
            "MOR-10004",
            "zxqv_unique_morphology_beta",
            Tier.H1_MORPHOLOGY,
            ProvenanceSource.HARVESTED,
            organ="universal",
            level="tissue",
        ),
        _make_concept(
            "CTY-10003",
            "zxqv_unique_celltype_gamma",
            Tier.H2_CELL_TYPE,
            ProvenanceSource.HARVESTED,
            concept_type="cell_type",
        ),
        _make_concept(
            "GPR-10001",
            "hallmark apoptosis",
            Tier.H2_GENE_PROGRAM,
            ProvenanceSource.HARVESTED,
            concept_type="gene_program",
        ),
    ]
    return _build_bank(concepts, ProvenanceSource.HARVESTED)


class MockSapBert:
    """Deterministic mock SapBERT using MD5-hash-based unit vectors.

    Same string → same vector → cosine 1.0.
    Different strings → near-orthogonal vectors → cosine ~0.0 (with high probability).

    Preset embeddings can be injected via the ``presets`` dict to control
    exact cosine values in matcher tests.
    """

    def __init__(self, presets: dict[str, npt.NDArray[Any]] | None = None) -> None:
        self._presets: dict[str, npt.NDArray[Any]] = dict(presets or {})
        self._dim = 768

    def _vec(self, text: str) -> npt.NDArray[Any]:
        if text in self._presets:
            return self._presets[text]
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        v = rng.randn(self._dim).astype(np.float32)
        v /= float(np.linalg.norm(v))
        return v

    def encode(self, texts: Sequence[str], **_: Any) -> npt.NDArray[Any]:
        """Return [N, 768] float32 array of L2-normalised embeddings."""
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def cosine_similarity(self, a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute cosine similarity matrix [N, M]."""
        result: npt.NDArray[Any] = a @ b.T
        return result

    def flush(self) -> None:
        """No-op flush for interface compatibility."""


@pytest.fixture
def mock_sapbert() -> MockSapBert:
    """MockSapBert with no presets (hash-only)."""
    return MockSapBert()
