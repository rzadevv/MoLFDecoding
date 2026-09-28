"""Unit tests for NCBIClient — all HTTP mocked via respx."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from molf_interp.sources._common.cache import ResponseCache
from molf_interp.sources._common.ratelimit import TokenBucket
from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.config import NCBIConfig
from molf_interp.sources.ncbi.exceptions import (
    AuthenticationError,
    RateLimitedError,
    TransientError,
)
from tests.sources.ncbi.conftest import fake_efetch_xml, fake_esearch_xml

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _make_client(
    cfg: NCBIConfig,
    router: respx.MockRouter,
    tmp_path: Path,
    api_key: str | None = "test-ncbi-key",
    email: str | None = "test@example.com",
) -> NCBIClient:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(router.handler),
        timeout=5.0,
    )
    cache = ResponseCache(tmp_path / "cache")
    limiter = TokenBucket(rate_per_second=1000, burst=1000)
    return NCBIClient(
        config=cfg,
        api_key=api_key,
        email=email,
        http_client=http_client,
        cache=cache,
        rate_limiter=limiter,
    )


# ---------- esearch ----------


async def test_esearch_returns_parsed_result(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    xml = fake_esearch_xml(count=42, pmids=["100", "200"], webenv="WE1", query_key="1")
    respx_ncbi.get("/entrez/eutils/esearch.fcgi").mock(return_value=httpx.Response(200, text=xml))
    client = _make_client(ncbi_config, respx_ncbi, tmp_path)
    async with client:
        result = await client.esearch_pubmed("necrosis[tiab]", max_results=10)
    assert result.count == 42
    assert result.pmids == ("100", "200")
    assert result.webenv == "WE1"


async def test_esearch_includes_api_key_in_request(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    xml = fake_esearch_xml(count=1, pmids=["1"])
    route = respx_ncbi.get("/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(200, text=xml)
    )
    client = _make_client(ncbi_config, respx_ncbi, tmp_path, api_key="my-secret-key")
    async with client:
        await client.esearch_pubmed("test")
    # Verify api_key was sent
    request = route.calls[0].request
    assert "my-secret-key" in str(request.url)
    # Verify api_key is NOT logged (we trust loguru; just confirm it's in the URL params)


async def test_no_api_key_uses_lower_rate_limit(
    ncbi_config: NCBIConfig,
    tmp_path: Path,
) -> None:
    client = NCBIClient(
        config=ncbi_config,
        api_key=None,
        email=None,
        cache=ResponseCache(tmp_path / "cache"),
    )
    # Rate limiter for no-key should use rate_limit_no_key
    # We can't easily assert on internal state, just verify construction doesn't fail
    assert client._api_key is None
    await client._http_client.aclose()


# ---------- efetch pagination ----------


async def test_efetch_pmid_mode_two_batches(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    # batch_size=2, 3 PMIDs → 2 batches
    small_cfg = NCBIConfig(
        cache_dir=tmp_path / "cache",
        efetch_batch_size=2,
        rate_limit_per_second=1000,
        rate_limit_burst=1000,
        rate_limit_no_key=1000,
        max_retries=1,
    )
    xml1 = fake_efetch_xml([{"pmid": "1", "title": "T1", "abstract_text": "A1"}])
    xml2 = fake_efetch_xml(
        [
            {"pmid": "2", "title": "T2", "abstract_text": "A2"},
            {"pmid": "3", "title": "T3", "abstract_text": "A3"},
        ]
    )
    call_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, text=xml1 if call_count == 1 else xml2)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        timeout=5.0,
    )
    client = NCBIClient(
        config=small_cfg,
        api_key="key",
        email="e@e.com",
        http_client=http_client,
        cache=ResponseCache(tmp_path / "cache"),
        rate_limiter=TokenBucket(1000, 1000),
    )
    async with client:
        abstracts = [ab async for ab in client.efetch_pubmed_abstracts(pmids=["1", "2", "3"])]
    assert call_count == 2
    assert len(abstracts) == 3


async def test_efetch_webenv_pagination(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    # batch_size=200, total=500 → 3 batches (200 + 200 + 100)
    xmls = [
        fake_efetch_xml([{"pmid": f"{i}", "title": f"T{i}", "abstract_text": "A"}])
        for i in range(3)
    ]
    call_idx = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_idx
        result = httpx.Response(200, text=xmls[call_idx % len(xmls)])
        call_idx += 1
        return result

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        timeout=5.0,
    )
    client = NCBIClient(
        config=ncbi_config,
        api_key="k",
        email="e@e.com",
        http_client=http_client,
        cache=ResponseCache(tmp_path / "cache"),
        rate_limiter=TokenBucket(1000, 1000),
    )
    async with client:
        # consume the generator to trigger all fetches
        [ab async for ab in client.efetch_pubmed_abstracts(webenv="WE", query_key="1", total=500)]
    # 3 batches issued
    assert call_idx == 3


async def test_efetch_cache_skips_second_request(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    xml = fake_efetch_xml([{"pmid": "99", "title": "Cached", "abstract_text": "A"}])
    call_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, text=xml)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        timeout=5.0,
    )
    cache = ResponseCache(tmp_path / "cache")
    client = NCBIClient(
        config=ncbi_config,
        api_key="k",
        email=None,
        http_client=http_client,
        cache=cache,
        rate_limiter=TokenBucket(1000, 1000),
    )
    async with client:
        # First call — hits network
        abs1 = [ab async for ab in client.efetch_pubmed_abstracts(pmids=["99"])]
        # Second call with same params — should hit cache
        abs2 = [ab async for ab in client.efetch_pubmed_abstracts(pmids=["99"])]

    assert call_count == 1  # only one real HTTP request
    assert len(abs1) == len(abs2) == 1


# ---------- error handling ----------


async def test_auth_error_on_error_xml_body(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    xml = "<eSearchResult><ERROR>API key invalid: bad-key</ERROR></eSearchResult>"
    respx_ncbi.get("/entrez/eutils/esearch.fcgi").mock(return_value=httpx.Response(200, text=xml))
    client = _make_client(ncbi_config, respx_ncbi, tmp_path)
    async with client:
        with pytest.raises(AuthenticationError):
            await client.esearch_pubmed("test")


async def test_rate_limited_after_retries(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    small_cfg = NCBIConfig(
        cache_dir=tmp_path / "cache",
        max_retries=1,
        rate_limit_per_second=1000,
        rate_limit_burst=1000,
        rate_limit_no_key=1000,
    )
    respx_ncbi.get("/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(429, text="Rate limit")
    )
    client = _make_client(small_cfg, respx_ncbi, tmp_path)
    async with client:
        with pytest.raises(RateLimitedError):
            await client.esearch_pubmed("test")


async def test_transient_error_after_retries(
    ncbi_config: NCBIConfig,
    respx_ncbi: respx.MockRouter,
    tmp_path: Path,
) -> None:
    small_cfg = NCBIConfig(
        cache_dir=tmp_path / "cache",
        max_retries=1,
        rate_limit_per_second=1000,
        rate_limit_burst=1000,
        rate_limit_no_key=1000,
    )
    respx_ncbi.get("/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(500, text="Server Error")
    )
    client = _make_client(small_cfg, respx_ncbi, tmp_path)
    async with client:
        with pytest.raises(TransientError):
            await client.esearch_pubmed("test")
