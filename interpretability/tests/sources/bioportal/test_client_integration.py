"""Integration tests for BioPortalClient against the real API."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.bioportal.config import BioPortalConfig

_NCIT_ROOT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3262"
_pytestmark = pytest.mark.integration


@pytest.fixture()
def real_config(tmp_path: Path) -> BioPortalConfig:
    return BioPortalConfig(
        cache_dir=tmp_path / "cache",
        rate_limit_per_second=5,
        max_retries=2,
    )


@pytest.fixture()
def _require_api_key() -> None:
    load_dotenv()
    if not os.environ.get("BIOPORTAL_API_KEY"):
        pytest.skip("BIOPORTAL_API_KEY not set")


@pytest.mark.integration()
async def test_can_access_ncit(real_config: BioPortalConfig, _require_api_key: None) -> None:
    async with BioPortalClient.from_config(real_config) as client:
        assert await client.can_access("NCIT") is True


@pytest.mark.integration()
async def test_get_ontology_ncit(real_config: BioPortalConfig, _require_api_key: None) -> None:
    async with BioPortalClient.from_config(real_config) as client:
        data = await client.get_ontology("NCIT")
    assert data["acronym"] == "NCIT"


@pytest.mark.integration()
async def test_get_class_neoplasm(real_config: BioPortalConfig, _require_api_key: None) -> None:
    async with BioPortalClient.from_config(real_config) as client:
        cls = await client.get_class("NCIT", _NCIT_ROOT)
    assert cls.pref_label == "Neoplasm"


@pytest.mark.integration()
async def test_iter_descendants_yields_items(
    real_config: BioPortalConfig, _require_api_key: None
) -> None:
    items = []
    async with BioPortalClient.from_config(real_config) as client:
        async for cls in client.iter_descendants("NCIT", _NCIT_ROOT):
            items.append(cls)
            if len(items) >= 10:
                break
    assert len(items) > 0


@pytest.mark.integration()
async def test_search_lymphocytic(real_config: BioPortalConfig, _require_api_key: None) -> None:
    results = []
    async with BioPortalClient.from_config(real_config) as client:
        search_iter = await client.search("lymphocytic infiltration", ontologies=["NCIT"])
        async for result in search_iter:
            results.append(result)
            if len(results) >= 20:
                break
    matches = [r for r in results if "lymphocyt" in r.pref_label.lower()]
    assert len(matches) > 0
