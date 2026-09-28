"""Integration tests for harvesters against the real BioPortal API and MSigDB."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.bioportal.config import BioPortalConfig
from molf_interp.sources.harvesters.msigdb import (
    MSigDBCollection,
    MSigDBHarvestConfig,
    harvest_msigdb,
)
from molf_interp.sources.harvesters.ncit import NCItHarvestConfig, harvest_ncit
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts


@pytest.fixture()
def _require_api_key() -> None:
    load_dotenv()
    if not os.environ.get("BIOPORTAL_API_KEY"):
        pytest.skip("BIOPORTAL_API_KEY not set")


@pytest.fixture()
def real_bioportal_config(tmp_path: Path) -> BioPortalConfig:
    return BioPortalConfig(
        cache_dir=tmp_path / "bp_cache",
        rate_limit_per_second=5,
        max_retries=2,
    )


@pytest.mark.integration()
async def test_ncit_real(
    real_bioportal_config: BioPortalConfig,
    _require_api_key: None,
    tmp_path: Path,
) -> None:
    """Harvest NCIt C7057 subtree; expect >100 records."""
    config = NCItHarvestConfig(
        subtree_roots=("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7057",),
        output_path=tmp_path / "ncit.parquet",
        apply_obvious_reject_filter=False,
    )
    async with BioPortalClient.from_config(real_bioportal_config) as client:
        count = await harvest_ncit(config, client)
    assert count > 100
    records = read_raw_concepts(tmp_path / "ncit.parquet")
    assert len(records) == count
    assert all(r.source_name == "ncit" for r in records)


@pytest.mark.integration()
async def test_msigdb_real(tmp_path: Path) -> None:
    """Download hallmark GMT; expect exactly 50 gene sets."""
    config = MSigDBHarvestConfig(
        collections=(MSigDBCollection.HALLMARK,),
        cache_dir=tmp_path / "msigdb",
        output_path=tmp_path / "msigdb.parquet",
    )
    count = harvest_msigdb(config)
    assert count == 50
    records = read_raw_concepts(tmp_path / "msigdb.parquet")
    assert all(r.candidate_tier == CandidateTier.H2_GENE_PROGRAM for r in records)


@pytest.mark.integration()
async def test_snomed_real(
    real_bioportal_config: BioPortalConfig,
    _require_api_key: None,
    tmp_path: Path,
) -> None:
    """Check SNOMED access status — logs result, does not assert True/False."""
    async with BioPortalClient.from_config(real_bioportal_config) as client:
        accessible = await client.can_access("SNOMEDCT")
    # Log-only test: account may or may not have SNOMED access.
    # Presence of the log message is not checked here — CI captures it.
    assert isinstance(accessible, bool)
