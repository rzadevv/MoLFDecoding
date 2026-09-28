"""Tests for gene_overlap helpers."""

from __future__ import annotations

import numpy as np

from molf_interp.curation.gene_overlap import (
    can_merge_gene_programs,
    jaccard_overlap,
    parse_gene_list,
)

# ---------------------------------------------------------------------------
# jaccard_overlap
# ---------------------------------------------------------------------------


def test_jaccard_identical() -> None:
    assert jaccard_overlap({"FOXP3", "CD8A"}, {"FOXP3", "CD8A"}) == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard_overlap({"FOXP3"}, {"CD8A"}) == 0.0


def test_jaccard_partial() -> None:
    result = jaccard_overlap({"A", "B", "C"}, {"B", "C", "D"})
    assert abs(result - 2 / 4) < 1e-9


def test_jaccard_both_empty() -> None:
    assert jaccard_overlap(set(), set()) == 0.0


def test_jaccard_one_empty() -> None:
    assert jaccard_overlap({"A"}, set()) == 0.0


# ---------------------------------------------------------------------------
# parse_gene_list
# ---------------------------------------------------------------------------


def test_parse_gene_list_dict() -> None:
    result = parse_gene_list({"genes": "FOXP3, CD8A, CD4"})
    assert result == {"FOXP3", "CD8A", "CD4"}


def test_parse_gene_list_list_of_pairs() -> None:
    extra = [np.array(["genes", "FOXP3,CD8A,PDCD1"], dtype=object)]
    assert parse_gene_list(extra) == {"FOXP3", "CD8A", "PDCD1"}


def test_parse_gene_list_no_genes_key_returns_empty() -> None:
    extra = [np.array(["msigdb_url", "http://example.com"], dtype=object)]
    assert parse_gene_list(extra) == set()


def test_parse_gene_list_empty_extra_returns_empty() -> None:
    assert parse_gene_list([]) == set()


def test_parse_gene_list_none_returns_empty() -> None:
    assert parse_gene_list(None) == set()


def test_parse_gene_list_uppercase_normalization() -> None:
    result = parse_gene_list({"genes": "foxp3,Cd8a"})
    assert result == {"FOXP3", "CD8A"}


# ---------------------------------------------------------------------------
# can_merge_gene_programs
# ---------------------------------------------------------------------------


def _msigdb_record(genes: str, source: str = "msigdb_h.all") -> dict:  # type: ignore[type-arg]
    return {
        "source_name": source,
        "extra": [np.array(["genes", genes], dtype=object)],
    }


def test_can_merge_high_name_high_gene_overlap() -> None:
    a = _msigdb_record("FOXP3,CD8A,PDCD1,CTLA4,LAG3")
    b = _msigdb_record("FOXP3,CD8A,PDCD1,CTLA4,TIM3")
    # 4/6 Jaccard = 0.667 > 0.5; name_sim > threshold
    assert can_merge_gene_programs(a, b, name_similarity=0.9, name_threshold=0.85)


def test_cannot_merge_high_name_low_gene_overlap() -> None:
    # Hallmark-EMT vs Gavish-EMT: same name, different gene composition
    a = _msigdb_record("VIM,FN1,CDH2,SNAI1,TWIST1")
    b = _msigdb_record("MMP2,ACTA2,COL1A1,FN1,ITGB1")
    # Only FN1 shared: Jaccard = 1/9 < 0.5
    assert not can_merge_gene_programs(a, b, name_similarity=0.92, name_threshold=0.85)


def test_can_merge_low_name_similarity() -> None:
    a = _msigdb_record("A,B,C")
    b = _msigdb_record("A,B,C")
    assert not can_merge_gene_programs(a, b, name_similarity=0.50, name_threshold=0.85)


def test_can_merge_one_record_no_gene_list() -> None:
    # One MSigDB record has no gene list → fall back to name similarity
    a = _msigdb_record("FOXP3,CD8A")
    b = {"source_name": "msigdb_h.all", "extra": []}
    assert can_merge_gene_programs(a, b, name_similarity=0.9, name_threshold=0.85)


def test_can_merge_non_msigdb_uses_name_only() -> None:
    a = {"source_name": "ncit", "extra": []}
    b = {"source_name": "snomed", "extra": []}
    # Both non-MSigDB: gene-overlap not checked
    assert can_merge_gene_programs(a, b, name_similarity=0.9, name_threshold=0.85)
