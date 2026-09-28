"""Unit tests for the MSigDB harvester."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from molf_interp.sources.harvesters.msigdb import (
    MSigDBCollection,
    MSigDBHarvestConfig,
    harvest_msigdb,
    humanize_msigdb_set_name,
)
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts
from tests.sources.harvesters.conftest import synthetic_gmt_content

# ---------- humanize_msigdb_set_name ----------


def test_humanize_hallmark() -> None:
    assert (
        humanize_msigdb_set_name("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION")
        == "Epithelial mesenchymal transition"
    )


def test_humanize_reactome_tca() -> None:
    result = humanize_msigdb_set_name("REACTOME_TCA_CYCLE")
    assert result == "Tca cycle"


def test_humanize_kegg() -> None:
    result = humanize_msigdb_set_name("KEGG_GLYCOLYSIS_GLUCONEOGENESIS")
    assert result == "Glycolysis gluconeogenesis"


def test_humanize_no_prefix() -> None:
    # No underscore → treated as body; sentence-cased to "Standalone".
    result = humanize_msigdb_set_name("STANDALONE")
    assert result == "Standalone"


# ---------- GMT parsing ----------


def _msigdb_config(tmp_path: Path, **overrides: Any) -> MSigDBHarvestConfig:
    defaults: dict[str, Any] = {
        "collections": (MSigDBCollection.HALLMARK,),
        "cache_dir": tmp_path / "gmt_cache",
        "output_path": tmp_path / "msigdb.parquet",
        "msigdb_version": "test.v1",
    }
    defaults.update(overrides)
    return MSigDBHarvestConfig(**defaults)


def test_parse_synthetic_gmt(tmp_path: Path) -> None:
    gmt_content = synthetic_gmt_content(num_sets=5, genes_per_set=10)
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    gmt_file = cache_dir / "h.all.vtest.v1.symbols.gmt"
    gmt_file.write_text(gmt_content, encoding="utf-8")

    config = _msigdb_config(tmp_path)
    count = harvest_msigdb(config)

    assert count == 5
    records = read_raw_concepts(tmp_path / "msigdb.parquet")
    assert len(records) == 5


def test_gene_list_preserved_in_extra(tmp_path: Path) -> None:
    gmt_content = "HALLMARK_MY_SET\thttp://example.com\tGENE1\tGENE2\tGENE3\n"
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "h.all.vtest.v1.symbols.gmt").write_text(gmt_content, encoding="utf-8")

    config = _msigdb_config(tmp_path)
    harvest_msigdb(config)

    records = read_raw_concepts(tmp_path / "msigdb.parquet")
    assert "GENE1" in records[0].extra.get("genes", "")
    assert "GENE3" in records[0].extra.get("genes", "")


def test_set_name_kept_as_synonym(tmp_path: Path) -> None:
    gmt_content = "HALLMARK_APOPTOSIS\thttp://example.com\tGENE1\n"
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "h.all.vtest.v1.symbols.gmt").write_text(gmt_content, encoding="utf-8")

    config = _msigdb_config(tmp_path)
    harvest_msigdb(config)

    records = read_raw_concepts(tmp_path / "msigdb.parquet")
    assert "HALLMARK_APOPTOSIS" in records[0].synonyms


def test_line_with_no_genes_is_skipped(tmp_path: Path) -> None:
    gmt_content = "HALLMARK_EMPTY\thttp://example.com\nHALLMARK_OK\thttp://example.com\tGENE1\n"
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "h.all.vtest.v1.symbols.gmt").write_text(gmt_content, encoding="utf-8")

    config = _msigdb_config(tmp_path)
    count = harvest_msigdb(config)

    assert count == 1


def test_candidate_tier_hallmark_is_gene_program(tmp_path: Path) -> None:
    gmt_content = "HALLMARK_TEST\thttp://example.com\tGENE1\n"
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "h.all.vtest.v1.symbols.gmt").write_text(gmt_content, encoding="utf-8")

    config = _msigdb_config(tmp_path)
    harvest_msigdb(config)

    records = read_raw_concepts(tmp_path / "msigdb.parquet")
    assert records[0].candidate_tier == CandidateTier.H2_GENE_PROGRAM


def test_download_cached_on_second_call(tmp_path: Path) -> None:
    """Second harvest_msigdb call must not re-download the GMT file."""
    gmt_content = "HALLMARK_ONCE\thttp://example.com\tGENE1\n"
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    with patch("molf_interp.sources.harvesters.msigdb._download_gmt") as mock_dl:
        # First call: file absent → download writes it
        def _write_file(url: str, dest: Path, timeout: float) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(gmt_content, encoding="utf-8")

        mock_dl.side_effect = _write_file

        config = _msigdb_config(tmp_path)
        harvest_msigdb(config)
        first_count = mock_dl.call_count

        # Second call: file exists → should NOT call _download_gmt again
        config2 = _msigdb_config(tmp_path)
        harvest_msigdb(config2)
        second_count = mock_dl.call_count

    assert first_count == 1
    assert second_count == 1  # no additional download


def test_404_raises_file_not_found(tmp_path: Path) -> None:
    with patch(
        "molf_interp.sources.harvesters.msigdb._download_gmt",
        side_effect=FileNotFoundError(
            "MSigDB GMT not found at 'http://...'. Check msigdb_version."
        ),
    ):
        config = _msigdb_config(tmp_path)
        with pytest.raises(FileNotFoundError, match="msigdb_version"):
            harvest_msigdb(config)


def test_source_name_format(tmp_path: Path) -> None:
    gmt_content = "HALLMARK_X\thttp://example.com\tGENE1\n"
    cache_dir = tmp_path / "gmt_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "h.all.vtest.v1.symbols.gmt").write_text(gmt_content, encoding="utf-8")

    config = _msigdb_config(tmp_path)
    harvest_msigdb(config)

    records = read_raw_concepts(tmp_path / "msigdb.parquet")
    assert records[0].source_name == "msigdb_h.all"
