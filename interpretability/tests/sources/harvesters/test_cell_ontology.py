"""Unit tests for the Cell Ontology harvester."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import respx

from molf_interp.sources.harvesters.cell_ontology import (
    CellOntologyHarvestConfig,
    harvest_cell_ontology,
)
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts
from tests.sources.bioportal.conftest import mk_class_json
from tests.sources.harvesters.conftest import make_descendants_page

_CL_ROOT = "http://purl.obolibrary.org/obo/CL_0000000"
_BASE = "https://data.bioontology.org"


def _cl_config(tmp_path: Path, **overrides: Any) -> CellOntologyHarvestConfig:
    defaults: dict[str, Any] = {
        "subtree_roots": (_CL_ROOT,),
        "output_path": tmp_path / "cl.parquet",
    }
    defaults.update(overrides)
    return CellOntologyHarvestConfig(**defaults)


async def test_source_name_is_cell_ontology(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(f"http://purl.obolibrary.org/obo/CL_{i:07d}", f"Cell {i}") for i in range(3)
    ]
    path, page = make_descendants_page("CL", _CL_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            await harvest_cell_ontology(_cl_config(tmp_path), client)

    records = read_raw_concepts(tmp_path / "cl.parquet")
    assert all(r.source_name == "cell_ontology" for r in records)


async def test_candidate_tier_is_h2_cell_type(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [mk_class_json("http://purl.obolibrary.org/obo/CL_0000001", "neuron")]
    path, page = make_descendants_page("CL", _CL_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            count = await harvest_cell_ontology(_cl_config(tmp_path), client)

    assert count == 1
    records = read_raw_concepts(tmp_path / "cl.parquet")
    assert records[0].candidate_tier == CandidateTier.H2_CELL_TYPE


async def test_obsolete_skipped(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json("http://purl.obolibrary.org/obo/CL_0000001", "Live Cell", obsolete=False),
        mk_class_json("http://purl.obolibrary.org/obo/CL_0000002", "Dead Cell", obsolete=True),
    ]
    path, page = make_descendants_page("CL", _CL_ROOT, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            count = await harvest_cell_ontology(_cl_config(tmp_path), client)

    assert count == 1
