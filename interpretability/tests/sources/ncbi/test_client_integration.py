"""Integration tests for NCBIClient — require real NCBI_API_KEY."""

from __future__ import annotations

import os

import pytest

from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.config import NCBIConfig


def _has_ncbi_key() -> bool:
    return bool(os.environ.get("NCBI_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_ncbi_key(), reason="NCBI_API_KEY not set")
async def test_esearch_returns_results(tmp_path: pytest.TempPathFactory) -> None:
    cfg = NCBIConfig(cache_dir=tmp_path / "cache")  # type: ignore[call-arg]
    async with NCBIClient.from_config(cfg) as client:
        result = await client.esearch_pubmed('"hematoxylin and eosin"[tiab]', max_results=5)
    assert result.count > 0
    assert len(result.pmids) == 5


@pytest.mark.integration
@pytest.mark.skipif(not _has_ncbi_key(), reason="NCBI_API_KEY not set")
async def test_efetch_yields_abstracts_with_titles(tmp_path: pytest.TempPathFactory) -> None:
    cfg = NCBIConfig(cache_dir=tmp_path / "cache")  # type: ignore[call-arg]
    async with NCBIClient.from_config(cfg) as client:
        result = await client.esearch_pubmed('"hematoxylin and eosin"[tiab]', max_results=5)
        pmids = list(result.pmids[:3])
        abstracts = [ab async for ab in client.efetch_pubmed_abstracts(pmids=pmids)]
    assert len(abstracts) == 3
    assert all(ab.title for ab in abstracts)
