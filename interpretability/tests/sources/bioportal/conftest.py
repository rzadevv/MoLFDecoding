"""Shared fixtures for BioPortal tests."""

from pathlib import Path
from typing import Any

import pytest
import respx

from molf_interp.sources.bioportal.config import BioPortalConfig


@pytest.fixture()
def bioportal_config(tmp_path: Path) -> BioPortalConfig:
    """BioPortalConfig with fast settings for unit tests."""
    return BioPortalConfig(
        cache_dir=tmp_path / "cache",
        rate_limit_per_second=100,
        rate_limit_burst=100,
        max_retries=2,
        timeout_seconds=5.0,
    )


@pytest.fixture()
def fake_api_key() -> str:
    """Fake API key for unit tests."""
    return "test-api-key-deadbeef"


@pytest.fixture()
def respx_mock() -> respx.MockRouter:
    """Pre-configured respx router for https://data.bioontology.org."""
    with respx.mock(base_url="https://data.bioontology.org", assert_all_called=False) as router:
        yield router


def mk_class_json(
    iri: str = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3262",
    pref_label: str = "Neoplasm",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a realistic BioPortal class JSON payload.

    Args:
        iri: Class IRI.
        pref_label: Primary label.
        **overrides: Fields to override in the default payload.

    Returns:
        Dict mimicking a BioPortal class JSON response.
    """
    base: dict[str, Any] = {
        "@id": iri,
        "@type": "http://www.w3.org/2002/07/owl#Class",
        "prefLabel": pref_label,
        "synonym": ["Tumor", "Tumour"],
        "definition": ["A new growth of tissue in which cell multiplication is uncontrolled."],
        "obsolete": False,
        "cui": ["C0027651"],
        "semanticType": ["T191"],
        "parents": [],
        "links": {
            "self": f"https://data.bioontology.org/ontologies/NCIT/classes/{iri}",
            "ontology": "https://data.bioontology.org/ontologies/NCIT",
            "children": f"https://data.bioontology.org/ontologies/NCIT/classes/{iri}/children",
            "descendants": f"https://data.bioontology.org/ontologies/NCIT/classes/{iri}/descendants",
        },
    }
    base.update(overrides)
    return base


def mk_collection_response(
    items: list[dict[str, Any]],
    page: int = 1,
    page_count: int = 1,
    total_count: int | None = None,
) -> dict[str, Any]:
    """Build a BioPortal paginated collection response envelope.

    Args:
        items: Items for the collection array.
        page: Current page number (1-indexed).
        page_count: Total page count.
        total_count: Total items across all pages.

    Returns:
        Dict mimicking a BioPortal paginated response.
    """
    _base = "https://data.bioontology.org/ontologies/NCIT/classes/ROOT/descendants"
    next_page: str | None = f"{_base}?page={page + 1}&pagesize=200" if page < page_count else None
    return {
        "page": page,
        "pageCount": page_count,
        "totalCount": total_count or len(items),
        "prevPage": None,
        "nextPage": next_page,
        "links": {
            "nextPage": next_page,
            "prevPage": None,
        },
        "collection": items,
    }
