"""Tests for RawConcept model, parquet I/O, and merge utility."""

from __future__ import annotations

from pathlib import Path

import pytest

from molf_interp.sources.raw_concept import (
    CandidateTier,
    RawConcept,
    merge_raw_concept_files,
    read_raw_concepts,
    write_raw_concepts,
)


def _make_concept(**overrides: object) -> RawConcept:
    defaults: dict[str, object] = {
        "source_name": "ncit",
        "source_id": "http://example.org/C123",
        "preferred_label": "Test Concept",
        "synonyms": ("Alias One", "Alias Two"),
        "definition": "A test definition.",
        "parent_ids": ("http://example.org/Parent",),
        "candidate_tier": CandidateTier.H1_MORPHOLOGY,
        "extra": {"cui": "C0000001", "semantic_types": "T191"},
    }
    defaults.update(overrides)
    return RawConcept(**defaults)  # type: ignore[arg-type]


def test_roundtrip_all_fields(tmp_path: Path) -> None:
    concept = _make_concept()
    path = tmp_path / "test.parquet"
    write_raw_concepts([concept], path)
    records = read_raw_concepts(path)
    assert len(records) == 1
    got = records[0]
    assert got.source_name == concept.source_name
    assert got.source_id == concept.source_id
    assert got.preferred_label == concept.preferred_label
    assert got.synonyms == concept.synonyms
    assert got.definition == concept.definition
    assert got.parent_ids == concept.parent_ids
    assert got.candidate_tier == concept.candidate_tier
    assert got.extra == concept.extra


def test_empty_synonyms_roundtrip(tmp_path: Path) -> None:
    concept = _make_concept(synonyms=())
    path = tmp_path / "empty_syn.parquet"
    write_raw_concepts([concept], path)
    records = read_raw_concepts(path)
    assert records[0].synonyms == ()


def test_empty_records_write_read(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    write_raw_concepts([], path)
    records = read_raw_concepts(path)
    assert records == ()


def test_definition_none_roundtrip(tmp_path: Path) -> None:
    concept = _make_concept(definition=None)
    path = tmp_path / "nodef.parquet"
    write_raw_concepts([concept], path)
    records = read_raw_concepts(path)
    assert records[0].definition is None


def test_extra_empty_roundtrip(tmp_path: Path) -> None:
    concept = _make_concept(extra={})
    path = tmp_path / "noextra.parquet"
    write_raw_concepts([concept], path)
    records = read_raw_concepts(path)
    assert records[0].extra == {}


def test_multiple_concepts_roundtrip(tmp_path: Path) -> None:
    concepts = [
        _make_concept(source_id=f"http://example.org/C{i}", preferred_label=f"Concept {i}")
        for i in range(10)
    ]
    path = tmp_path / "multi.parquet"
    write_raw_concepts(concepts, path)
    records = read_raw_concepts(path)
    assert len(records) == 10
    labels = {r.preferred_label for r in records}
    assert labels == {f"Concept {i}" for i in range(10)}


def test_merge_concatenates_rows(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    write_raw_concepts(
        [_make_concept(source_id="http://example.org/A", preferred_label="A")], path_a
    )
    write_raw_concepts(
        [
            _make_concept(source_id="http://example.org/B1", preferred_label="B1"),
            _make_concept(source_id="http://example.org/B2", preferred_label="B2"),
        ],
        path_b,
    )
    merged = tmp_path / "merged.parquet"
    count = merge_raw_concept_files([path_a, path_b], merged)
    assert count == 3
    records = read_raw_concepts(merged)
    assert len(records) == 3


def test_merge_skips_missing_paths(tmp_path: Path) -> None:
    path_a = tmp_path / "exists.parquet"
    write_raw_concepts([_make_concept(preferred_label="Exists")], path_a)
    merged = tmp_path / "merged.parquet"
    count = merge_raw_concept_files([path_a, tmp_path / "missing.parquet"], merged)
    assert count == 1


def test_merge_empty_input(tmp_path: Path) -> None:
    merged = tmp_path / "merged.parquet"
    count = merge_raw_concept_files([], merged)
    assert count == 0
    records = read_raw_concepts(merged)
    assert records == ()


def test_preferred_label_validator_empty_raises() -> None:
    with pytest.raises(ValueError):
        _make_concept(preferred_label="")


def test_preferred_label_validator_whitespace_only_raises() -> None:
    with pytest.raises(ValueError):
        _make_concept(preferred_label="   ")


def test_preferred_label_stripped() -> None:
    concept = _make_concept(preferred_label="  Trimmed  ")
    assert concept.preferred_label == "Trimmed"
