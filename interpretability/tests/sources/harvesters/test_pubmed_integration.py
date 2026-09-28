"""Integration tests for the PubMed harvester — require real NCBI_API_KEY and scispaCy."""

from __future__ import annotations

import os

import pytest

from molf_interp.sources.harvesters.pubmed import (
    PubMedHarvestConfig,
    PubMedQuery,
    harvest_pubmed,
)
from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.config import NCBIConfig
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts


def _has_ncbi_key() -> bool:
    return bool(os.environ.get("NCBI_API_KEY"))


def _has_scispacy() -> bool:
    try:
        import spacy  # type: ignore[import]

        spacy.load("en_ner_bionlp13cg_md")
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _has_ncbi_key(), reason="NCBI_API_KEY not set")
@pytest.mark.skipif(not _has_scispacy(), reason="en_ner_bionlp13cg_md model not installed")
async def test_harvest_pubmed_produces_concepts(tmp_path: pytest.TempPathFactory) -> None:
    config = PubMedHarvestConfig(
        queries=(
            PubMedQuery(
                name="he_test",
                query='"hematoxylin and eosin"[tiab]',
                max_results=20,
            ),
        ),
        output_path=tmp_path / "pubmed.parquet",  # type: ignore[operator]
        min_entity_occurrences=1,
    )
    ncbi_cfg = NCBIConfig(cache_dir=tmp_path / "cache")  # type: ignore[call-arg]
    async with NCBIClient.from_config(ncbi_cfg) as client:
        count = await harvest_pubmed(config, client)

    assert count >= 5
    records = read_raw_concepts(tmp_path / "pubmed.parquet")  # type: ignore[operator]
    assert all(r.source_name == "pubmed" for r in records)
    morphology_records = [r for r in records if r.candidate_tier == CandidateTier.H1_MORPHOLOGY]
    assert len(morphology_records) >= 1
