"""Unit tests for the Gene Ontology harvester."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import respx

from molf_interp.sources.harvesters.gene_ontology import (
    GeneOntologyHarvestConfig,
    harvest_gene_ontology,
)
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts
from tests.sources.bioportal.conftest import mk_class_json
from tests.sources.harvesters.conftest import make_descendants_page

_GO_PROLIF = "http://purl.obolibrary.org/obo/GO_0008283"
_GO_ACTIV = "http://purl.obolibrary.org/obo/GO_0001775"
_BASE = "https://data.bioontology.org"


def _go_config(tmp_path: Path, **overrides: Any) -> GeneOntologyHarvestConfig:
    defaults: dict[str, Any] = {
        "subtree_roots": (_GO_PROLIF,),
        "output_path": tmp_path / "go.parquet",
    }
    defaults.update(overrides)
    return GeneOntologyHarvestConfig(**defaults)


async def test_source_name_is_gene_ontology(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(f"http://purl.obolibrary.org/obo/GO_{i:07d}", f"GO Term {i}")
        for i in range(3)
    ]
    path, page = make_descendants_page("GO", _GO_PROLIF, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            await harvest_gene_ontology(_go_config(tmp_path), client)

    records = read_raw_concepts(tmp_path / "go.parquet")
    assert all(r.source_name == "gene_ontology" for r in records)


async def test_candidate_tier_is_h2_gene_program(mock_client_factory: Any, tmp_path: Path) -> None:
    items = [
        mk_class_json(
            "http://purl.obolibrary.org/obo/GO_0008284", "positive regulation of cell proliferation"
        )
    ]
    path, page = make_descendants_page("GO", _GO_PROLIF, items)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path).mock(return_value=httpx.Response(200, json=page))
        async with mock_client_factory(router) as client:
            count = await harvest_gene_ontology(_go_config(tmp_path), client)

    assert count == 1
    records = read_raw_concepts(tmp_path / "go.parquet")
    assert records[0].candidate_tier == CandidateTier.H2_GENE_PROGRAM


async def test_two_subtree_roots(mock_client_factory: Any, tmp_path: Path) -> None:
    items1 = [mk_class_json(f"http://example.org/G{i}", f"GO {i}") for i in range(2)]
    items2 = [mk_class_json(f"http://example.org/A{i}", f"Act {i}") for i in range(3)]
    path1, page1 = make_descendants_page("GO", _GO_PROLIF, items1)
    path2, page2 = make_descendants_page("GO", _GO_ACTIV, items2)

    with respx.mock(base_url=_BASE, assert_all_called=False) as router:
        router.get(path1).mock(return_value=httpx.Response(200, json=page1))
        router.get(path2).mock(return_value=httpx.Response(200, json=page2))
        config = _go_config(tmp_path, subtree_roots=(_GO_PROLIF, _GO_ACTIV))
        async with mock_client_factory(router) as client:
            count = await harvest_gene_ontology(config, client)

    assert count == 5
