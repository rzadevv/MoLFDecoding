"""End-to-end integration test for the curation pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
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
from molf_interp.curation.config import CurationConfig, LLMAdjudicationConfig
from molf_interp.curation.pipeline import run_curation_pipeline


@pytest.mark.integration
def test_pipeline_integration_synthetic(tmp_path: Path) -> None:
    """Run full pipeline on 50-row synthetic input. LLM disabled."""
    from molf_interp.concepts.store import write_concept_bank
    from tests.curation.conftest import make_mock_sapbert

    # Build synthetic input
    rng = np.random.default_rng(0)
    n = 50
    embs = rng.standard_normal((n, 768)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    df = pd.DataFrame(
        {
            "source_name": (
                ["ncit"] * 20 + ["snomed"] * 15 + ["pubmed"] * 10 + ["cell_ontology"] * 5
            ),
            "source_id": [f"id:{i}" for i in range(n)],
            "preferred_label": [f"concept {i}" for i in range(n)],
            "synonyms": [[] for _ in range(n)],
            "definition": [None] * n,
            "parent_ids": [[] for _ in range(n)],
            "candidate_tier": (
                ["H1_morphology"] * 20
                + ["H1_morphology"] * 15
                + ["H2_cell_type"] * 10
                + ["H2_pathway"] * 5
            ),
            "extra": [[] for _ in range(n)],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    df.to_parquet(raw_path)

    # Write minimal reference bank
    reference_bank_dir = tmp_path / "reference_bank"
    reference_bank_dir.mkdir()

    su_concepts = []
    su_prompts = []
    for i, (tier, prefix, ctype, organ, level) in enumerate(
        [
            (Tier.H1_MORPHOLOGY, "MOR", None, "lung", "tissue"),
            (Tier.H2_CELL_TYPE, "CTY", "cell_type", None, None),
            (Tier.H2_PATHWAY, "PWY", "pathway", None, None),
        ]
    ):
        cid = f"{prefix}-{i + 1:04d}"
        c = Concept(
            concept_id=cid,
            concept_name=f"ref concept {i}",
            tier=tier,
            category="test",
            subcategory="test",
            source="reference",
            provenance=ProvenanceSource.REFERENCE,
            organ=organ,
            level=level,
            concept_type=ctype,
        )
        su_concepts.append(c)
        for sp in Species:
            su_prompts.append(
                ConceptPrompt(
                    concept_id=cid,
                    concept_name=c.concept_name,
                    tier=tier,
                    species=sp,
                    prompt_text=f"test prompt {i} {sp.value}",
                    species_status=SpeciesStatus.SHARED,
                )
            )

    reference_bank = ConceptBank(
        concepts=tuple(su_concepts),
        prompts=tuple(su_prompts),
        provenance=ProvenanceSource.REFERENCE,
    )
    write_concept_bank(reference_bank, reference_bank_dir)

    # Minimal uberon table
    uberon_path = tmp_path / "uberon.parquet"
    pd.DataFrame({"preferred_label": ["lung"], "synonyms": [["pulmonary"]]}).to_parquet(uberon_path)

    cfg = CurationConfig(
        input_path=raw_path,
        output_dir=tmp_path / "out",
        stage_cache_dir=tmp_path / "stages",
        llm=LLMAdjudicationConfig(enabled=False),
    )
    cfg = cfg.model_copy(
        update={
            # Thresholds set to -1/-2 so random mock embeddings pass the visual filter
            # (random 768-d unit vectors have cosine sim ≈ 0; real thresholds are 0.55/0.35)
            "visual_filter": cfg.visual_filter.model_copy(
                update={
                    "reference_bank_dir": reference_bank_dir,
                    "similarity_keep_threshold": -1.0,
                    "similarity_drop_threshold": -2.0,
                }
            ),
            "tier_refine": cfg.tier_refine.model_copy(update={"uberon_table_path": uberon_path}),
            "sapbert": cfg.sapbert.model_copy(update={"cache_dir": tmp_path / "sapbert_cache"}),
        }
    )

    mock_sapbert = make_mock_sapbert()

    with patch("molf_interp.curation.pipeline.SapBert", return_value=mock_sapbert):
        bank, _out_dir = run_curation_pipeline(cfg)

    assert isinstance(bank, ConceptBank)
    assert bank.provenance == ProvenanceSource.HARVESTED
    assert len(bank.concepts) > 0
    # All concept IDs should have at least 4 digits
    for c in bank.concepts:
        num = c.concept_id.split("-")[1]
        assert len(num) >= 4
