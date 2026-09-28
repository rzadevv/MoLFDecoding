"""Shared fixtures for harvester tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pytest
import respx

from molf_interp.sources._common.cache import ResponseCache
from molf_interp.sources._common.ratelimit import TokenBucket
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.bioportal.config import BioPortalConfig
from tests.sources.bioportal.conftest import mk_collection_response

_BASE = "https://data.bioontology.org"


@pytest.fixture()
def bioportal_config(tmp_path: Path) -> BioPortalConfig:
    """BioPortalConfig with fast settings for unit tests."""
    return BioPortalConfig(
        cache_dir=tmp_path / "bp_cache",
        rate_limit_per_second=1000,
        rate_limit_burst=1000,
        max_retries=1,
        timeout_seconds=5.0,
    )


@pytest.fixture()
def fake_api_key() -> str:
    """Fake API key for unit tests."""
    return "test-api-key-harvest"


@pytest.fixture()
def mock_client_factory(
    tmp_path: Path,
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
) -> Any:
    """Returns a factory: given a respx router, creates a BioPortalClient for testing."""

    def _make(router: respx.MockRouter) -> BioPortalClient:
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(router.handler),
            timeout=5.0,
            follow_redirects=True,
        )
        cache = ResponseCache(tmp_path / "bp_cache")
        limiter = TokenBucket(rate_per_second=1000, burst=1000)
        return BioPortalClient(
            config=bioportal_config,
            api_key=fake_api_key,
            http_client=http_client,
            cache=cache,
            rate_limiter=limiter,
        )

    return _make


def make_descendants_page(
    ont: str,
    root_iri: str,
    items: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Return (path, page_json) for mocking a single-page descendants response."""
    encoded = quote(root_iri, safe="")
    path = f"/ontologies/{ont}/classes/{encoded}/descendants"
    page = mk_collection_response(items)
    page["links"] = {"nextPage": None}
    return path, page


def synthetic_gmt_content(num_sets: int = 5, genes_per_set: int = 10) -> str:
    """Generate a synthetic MSigDB GMT string for testing.

    Args:
        num_sets: Number of gene sets to generate.
        genes_per_set: Number of gene symbols per set.

    Returns:
        Newline-separated GMT string.
    """
    lines = []
    for i in range(num_sets):
        set_name = f"HALLMARK_TEST_SET_{i}"
        url = f"http://www.example.com/msigdb/{set_name}"
        genes = [f"GENE{i}_{j}" for j in range(genes_per_set)]
        lines.append("\t".join([set_name, url, *genes]))
    return "\n".join(lines) + "\n"
