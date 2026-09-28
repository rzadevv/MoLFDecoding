"""Unit tests for BioPortalClient — all HTTP mocked via respx."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pytest
import respx

from molf_interp.sources._common.cache import ResponseCache
from molf_interp.sources._common.ratelimit import TokenBucket
from molf_interp.sources.bioportal.client import BioPortalClient, _encode_class_iri
from molf_interp.sources.bioportal.config import BioPortalConfig
from molf_interp.sources.bioportal.exceptions import (
    AuthenticationError,
    LicensedOntologyError,
    NotFoundError,
    TransientError,
)
from tests.sources.bioportal.conftest import mk_class_json, mk_collection_response

_BASE = "https://data.bioontology.org"
_NCIT_IRI = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3262"
_NCIT_ENCODED = quote(_NCIT_IRI, safe="")


def _make_client(
    cfg: BioPortalConfig,
    api_key: str,
    router: respx.MockRouter,
    tmp_path: Path,
) -> BioPortalClient:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(router.handler),
        timeout=5.0,
        follow_redirects=True,
    )
    cache = ResponseCache(tmp_path / "cache")
    limiter = TokenBucket(rate_per_second=1000, burst=1000)
    return BioPortalClient(
        config=cfg,
        api_key=api_key,
        http_client=http_client,
        cache=cache,
        rate_limiter=limiter,
    )


# ---------- IRI encoding ----------


def test_encode_class_iri() -> None:
    encoded = _encode_class_iri(_NCIT_IRI)
    assert encoded == "http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C3262"


# ---------- get_ontology ----------


async def test_get_ontology_success(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(
            return_value=httpx.Response(200, json={"acronym": "NCIT", "name": "NCI Thesaurus"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            data = await client.get_ontology("NCIT")
        assert data["acronym"] == "NCIT"


async def test_get_ontology_sends_auth_header(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"acronym": "NCIT"})

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(side_effect=_handler)
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            await client.get_ontology("NCIT")

    assert f"apikey token={fake_api_key}" in captured_headers.get("authorization", "")


async def test_get_ontology_404_raises_not_found(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/FAKE").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            with pytest.raises(NotFoundError):
                await client.get_ontology("FAKE")


async def test_get_ontology_401_raises_auth_error(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            with pytest.raises(AuthenticationError):
                await client.get_ontology("NCIT")


async def test_get_ontology_403_licensed_raises(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    body = {"errors": ["You must agree to the license agreement to access SNOMEDCT"]}
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/SNOMEDCT").mock(return_value=httpx.Response(403, json=body))
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            with pytest.raises(LicensedOntologyError) as exc_info:
                await client.get_ontology("SNOMEDCT")
    assert exc_info.value.ontology_acronym == "SNOMEDCT"


# ---------- can_access ----------


async def test_can_access_true(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(
            return_value=httpx.Response(200, json={"acronym": "NCIT"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            result = await client.can_access("NCIT")
    assert result is True


async def test_can_access_false_on_403(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/SNOMEDCT").mock(
            return_value=httpx.Response(403, json={"error": "license required"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            result = await client.can_access("SNOMEDCT")
    assert result is False


# ---------- get_class ----------


async def test_get_class_url_encodes_iri(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    expected_path = f"/ontologies/NCIT/classes/{_NCIT_ENCODED}"
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(expected_path).mock(
            return_value=httpx.Response(200, json=mk_class_json(_NCIT_IRI, "Neoplasm"))
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            cls = await client.get_class("NCIT", _NCIT_IRI)
    assert cls.pref_label == "Neoplasm"
    assert cls.iri == _NCIT_IRI


# ---------- iter_descendants pagination ----------


async def test_iter_descendants_follows_next_page(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    encoded_root = quote("http://example.org/Root", safe="")
    base_path = f"/ontologies/TEST/classes/{encoded_root}/descendants"

    cls1 = mk_class_json("http://example.org/A", "ClassA")
    cls2 = mk_class_json("http://example.org/B", "ClassB")
    cls3 = mk_class_json("http://example.org/C", "ClassC")

    page1_next = f"{_BASE}{base_path}?page=2&pagesize=200"
    page2_next = f"{_BASE}{base_path}?page=3&pagesize=200"

    page1 = {
        "page": 1,
        "pageCount": 3,
        "totalCount": 3,
        "links": {"nextPage": page1_next, "prevPage": None},
        "collection": [cls1],
    }
    page2 = {
        "page": 2,
        "pageCount": 3,
        "totalCount": 3,
        "links": {"nextPage": page2_next, "prevPage": None},
        "collection": [cls2],
    }
    page3 = {
        "page": 3,
        "pageCount": 3,
        "totalCount": 3,
        "links": {"nextPage": None, "prevPage": None},
        "collection": [cls3],
    }

    pages = {"1": page1, "2": page2, "3": page3}

    def _dispatch(request: httpx.Request) -> httpx.Response:
        p = request.url.params.get("page", "1")
        return httpx.Response(200, json=pages[str(p)])

    with respx.mock(assert_all_called=False) as router:
        router.route(method="GET").mock(side_effect=_dispatch)

        results = []
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            async for cls in client.iter_descendants("TEST", "http://example.org/Root"):
                results.append(cls.pref_label)

    assert results == ["ClassA", "ClassB", "ClassC"]


async def test_iter_descendants_skips_obsolete_by_default(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    encoded = quote("http://example.org/R", safe="")
    path = f"/ontologies/TEST/classes/{encoded}/descendants"
    live = mk_class_json("http://example.org/Live", "LiveClass", obsolete=False)
    dead = mk_class_json("http://example.org/Dead", "DeadClass", obsolete=True)
    page = mk_collection_response([live, dead])
    page["links"] = {"nextPage": None}

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        results = []
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            async for cls in client.iter_descendants("TEST", "http://example.org/R"):
                results.append(cls.pref_label)

    assert results == ["LiveClass"]


async def test_iter_descendants_includes_obsolete_when_flag_set(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    encoded = quote("http://example.org/R2", safe="")
    path = f"/ontologies/TEST/classes/{encoded}/descendants"
    live = mk_class_json("http://example.org/Live2", "LiveClass2", obsolete=False)
    dead = mk_class_json("http://example.org/Dead2", "DeadClass2", obsolete=True)
    page = mk_collection_response([live, dead])
    page["links"] = {"nextPage": None}

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        results = []
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            async for cls in client.iter_descendants(
                "TEST", "http://example.org/R2", include_obsolete=True
            ):
                results.append(cls.pref_label)

    assert set(results) == {"LiveClass2", "DeadClass2"}


# ---------- iter_children ----------


async def test_iter_children_pagination(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    encoded = quote("http://example.org/Parent", safe="")
    path = f"/ontologies/TEST/classes/{encoded}/children"
    child1 = mk_class_json("http://example.org/Child1", "Child1")
    child2 = mk_class_json("http://example.org/Child2", "Child2")
    page = mk_collection_response([child1, child2])
    page["links"] = {"nextPage": None}

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        results = []
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            async for cls in client.iter_children("TEST", "http://example.org/Parent"):
                results.append(cls.pref_label)

    assert set(results) == {"Child1", "Child2"}


# ---------- search ----------


async def test_search_query_params(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    captured_params: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(dict(request.url.params))
        return httpx.Response(200, json=mk_collection_response([]))

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/search").mock(side_effect=_handler)
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            search_iter = await client.search("necrosis", ontologies=["NCIT", "CL"])
            async for _ in search_iter:
                break

    assert captured_params.get("q") == "necrosis"
    assert "NCIT" in captured_params.get("ontologies", "")
    assert "CL" in captured_params.get("ontologies", "")


# ---------- caching ----------


async def test_cached_request_skips_network(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        route = router.get("/ontologies/NCIT").mock(
            return_value=httpx.Response(200, json={"acronym": "NCIT"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            await client.get_ontology("NCIT")
            await client.get_ontology("NCIT")  # second call — cache hit

    assert route.call_count == 1


# ---------- retries ----------


async def test_429_triggers_retry_then_succeeds(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    responses = [
        httpx.Response(429, json={"error": "rate limited"}),
        httpx.Response(200, json={"acronym": "NCIT"}),
    ]
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(side_effect=responses)
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            data = await client.get_ontology("NCIT")
    assert data["acronym"] == "NCIT"


async def test_persistent_500_raises_transient_error(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
) -> None:
    # max_retries=2 → 3 total attempts
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(
            return_value=httpx.Response(500, json={"error": "server error"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            with pytest.raises(TransientError):
                await client.get_ontology("NCIT")


# ---------- log redaction ----------


async def test_api_key_not_in_logs(
    bioportal_config: BioPortalConfig,
    fake_api_key: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from loguru import logger

    log_output = StringIO()
    logger.remove()
    logger.add(log_output, level="DEBUG")

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/NCIT").mock(
            return_value=httpx.Response(200, json={"acronym": "NCIT"})
        )
        async with _make_client(bioportal_config, fake_api_key, router, tmp_path) as client:
            await client.get_ontology("NCIT")

    logs = log_output.getvalue()
    assert fake_api_key not in logs
