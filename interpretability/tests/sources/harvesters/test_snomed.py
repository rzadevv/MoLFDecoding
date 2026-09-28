"""Unit tests for the SNOMED CT harvester (including graceful degradation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import respx

from molf_interp.sources.harvesters.snomed import SnomedHarvestConfig, harvest_snomed
from molf_interp.sources.raw_concept import read_raw_concepts
from tests.sources.bioportal.conftest import mk_class_json
from tests.sources.harvesters.conftest import make_descendants_page

_SNOMED_ROOT = "http://purl.bioontology.org/ontology/SNOMEDCT/49755003"
_BASE = "https://data.bioontology.org"


def _snomed_config(tmp_path: Path, **overrides: Any) -> SnomedHarvestConfig:
    defaults: dict[str, Any] = {
        "subtree_roots": (_SNOMED_ROOT,),
        "output_path": tmp_path / "snomed.parquet",
    }
    defaults.update(overrides)
    return SnomedHarvestConfig(**defaults)


async def test_not_accessible_writes_empty_parquet(
    mock_client_factory: Any, tmp_path: Path
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/SNOMEDCT").mock(
            return_value=httpx.Response(403, json={"errors": ["license agreement required"]})
        )
        config = _snomed_config(tmp_path)
        async with mock_client_factory(router) as client:
            count = await harvest_snomed(config, client)

    assert count == 0
    records = read_raw_concepts(tmp_path / "snomed.parquet")
    assert len(records) == 0


async def test_not_accessible_returns_zero_no_exception(
    mock_client_factory: Any, tmp_path: Path
) -> None:
    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/SNOMEDCT").mock(
            return_value=httpx.Response(403, json={"error": "license required"})
        )
        config = _snomed_config(tmp_path)
        async with mock_client_factory(router) as client:
            count = await harvest_snomed(config, client)

    assert count == 0


async def test_accessible_harvests_normally(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(f"http://purl.bioontology.org/ontology/SNOMEDCT/{i}", f"Morphology {i}")
        for i in range(4)
    ]
    path, page = make_descendants_page("SNOMEDCT", _SNOMED_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get("/ontologies/SNOMEDCT").mock(
            return_value=httpx.Response(200, json={"acronym": "SNOMEDCT"})
        )
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        config = _snomed_config(tmp_path)
        async with mock_client_factory(router) as client:
            count = await harvest_snomed(config, client)

    assert count == 4
    records = read_raw_concepts(tmp_path / "snomed.parquet")
    assert all(r.source_name == "snomed" for r in records)
