"""Unit tests for the UBERON organ harvester."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import respx

from molf_interp.sources.harvesters.uberon import UberonHarvestConfig, harvest_uberon
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts
from tests.sources.bioportal.conftest import mk_class_json
from tests.sources.harvesters.conftest import make_descendants_page

_UBERON_ROOT = "http://purl.obolibrary.org/obo/UBERON_0000062"
_BASE = "https://data.bioontology.org"


def _uberon_config(tmp_path: Path, **overrides: Any) -> UberonHarvestConfig:
    defaults: dict[str, Any] = {
        "subtree_roots": (_UBERON_ROOT,),
        "output_path": tmp_path / "uberon.parquet",
    }
    defaults.update(overrides)
    return UberonHarvestConfig(**defaults)


async def test_source_name_is_uberon(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(f"http://purl.obolibrary.org/obo/UBERON_{i:07d}", f"Organ {i}")
        for i in range(3)
    ]
    path, page = make_descendants_page("UBERON", _UBERON_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            await harvest_uberon(_uberon_config(tmp_path), client)

    records = read_raw_concepts(tmp_path / "uberon.parquet")
    assert all(r.source_name == "uberon" for r in records)


async def test_candidate_tier_is_organ_vocab(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [mk_class_json("http://purl.obolibrary.org/obo/UBERON_0002048", "lung")]
    path, page = make_descendants_page("UBERON", _UBERON_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            count = await harvest_uberon(_uberon_config(tmp_path), client)

    assert count == 1
    records = read_raw_concepts(tmp_path / "uberon.parquet")
    assert records[0].candidate_tier == CandidateTier.ORGAN_VOCAB


async def test_obsolete_skipped(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(
            "http://purl.obolibrary.org/obo/UBERON_0000001", "live organ", obsolete=False
        ),
        mk_class_json(
            "http://purl.obolibrary.org/obo/UBERON_0000002", "obsolete organ", obsolete=True
        ),
    ]
    path, page = make_descendants_page("UBERON", _UBERON_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            count = await harvest_uberon(_uberon_config(tmp_path), client)

    assert count == 1
