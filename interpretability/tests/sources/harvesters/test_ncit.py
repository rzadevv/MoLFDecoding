"""Unit tests for the NCIt harvester."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import respx

from molf_interp.sources.harvesters.ncit import NCItHarvestConfig, harvest_ncit
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts
from tests.sources.bioportal.conftest import mk_class_json
from tests.sources.harvesters.conftest import make_descendants_page

_NCIT_ROOT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7057"
_BASE = "https://data.bioontology.org"


def _ncit_config(tmp_path: Path, **overrides: Any) -> NCItHarvestConfig:
    defaults: dict[str, Any] = {
        "subtree_roots": (_NCIT_ROOT,),
        "output_path": tmp_path / "ncit.parquet",
        "apply_obvious_reject_filter": False,
    }
    defaults.update(overrides)
    return NCItHarvestConfig(**defaults)


async def test_single_subtree_obsolete_skipped(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(f"http://example.org/C{i}", f"Class {i}", obsolete=(i == 4)) for i in range(5)
    ]
    path, page = make_descendants_page("NCIT", _NCIT_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            count = await harvest_ncit(_ncit_config(tmp_path), client)

    assert count == 4
    records = read_raw_concepts(tmp_path / "ncit.parquet")
    assert len(records) == 4


async def test_source_name_and_tier(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [mk_class_json(f"http://example.org/C{i}", f"Class {i}") for i in range(3)]
    path, page = make_descendants_page("NCIT", _NCIT_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            await harvest_ncit(_ncit_config(tmp_path), client)

    records = read_raw_concepts(tmp_path / "ncit.parquet")
    assert all(r.source_name == "ncit" for r in records)
    assert all(r.candidate_tier == CandidateTier.H1_MORPHOLOGY for r in records)


async def test_obvious_reject_filter_drops_labels(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json("http://example.org/Keep", "coagulative necrosis"),
        mk_class_json("http://example.org/Drop", "patient survival rate"),
    ]
    path, page = make_descendants_page("NCIT", _NCIT_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        config = _ncit_config(tmp_path, apply_obvious_reject_filter=True)
        async with mock_client_factory(router) as client:
            count = await harvest_ncit(config, client)

    assert count == 1
    records = read_raw_concepts(tmp_path / "ncit.parquet")
    assert records[0].preferred_label == "coagulative necrosis"


async def test_two_subtree_roots_summed(mock_client_factory: Any, tmp_path: Path) -> None:
    root2 = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12219"
    items1 = [mk_class_json(f"http://example.org/A{i}", f"Class A{i}") for i in range(3)]
    items2 = [mk_class_json(f"http://example.org/B{i}", f"Class B{i}") for i in range(2)]
    path1, page1 = make_descendants_page("NCIT", _NCIT_ROOT, items1)
    path2, page2 = make_descendants_page("NCIT", root2, items2)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path1).mock(return_value=httpx.Response(200, json=page1))
        router.get(path2).mock(return_value=httpx.Response(200, json=page2))
        config = _ncit_config(tmp_path, subtree_roots=(_NCIT_ROOT, root2))
        async with mock_client_factory(router) as client:
            count = await harvest_ncit(config, client)

    assert count == 5


async def test_extra_contains_cui_and_semantic_types(
    mock_client_factory: Any, tmp_path: Path
) -> None:
    item = mk_class_json("http://example.org/WithCUI", "Neoplasm")
    item["cui"] = ["C0027651", "C0027652"]
    item["semanticType"] = ["T191", "T200"]
    path, page = make_descendants_page("NCIT", _NCIT_ROOT, [item])

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            await harvest_ncit(_ncit_config(tmp_path), client)

    records = read_raw_concepts(tmp_path / "ncit.parquet")
    assert len(records) == 1
    assert "C0027651" in records[0].extra.get("cui", "")
    assert "T191" in records[0].extra.get("semantic_types", "")


async def test_duplicate_descendants_across_roots_not_deduplicated(
    mock_client_factory: Any, tmp_path: Path
) -> None:
    root2 = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12219"
    shared = mk_class_json("http://example.org/Shared", "Shared Term")
    path1, page1 = make_descendants_page("NCIT", _NCIT_ROOT, [shared])
    path2, page2 = make_descendants_page("NCIT", root2, [shared])

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path1).mock(return_value=httpx.Response(200, json=page1))
        router.get(path2).mock(return_value=httpx.Response(200, json=page2))
        config = _ncit_config(tmp_path, subtree_roots=(_NCIT_ROOT, root2))
        async with mock_client_factory(router) as client:
            count = await harvest_ncit(config, client)

    assert count == 2  # dedup is curation's job
