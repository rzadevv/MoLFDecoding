"""Tests for OrganNormalizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from molf_interp.curation.organ_normalizer import OrganNormalizer


def _make_empty_uberon(tmp_path: Path) -> Path:
    p = tmp_path / "uberon.parquet"
    pd.DataFrame({"preferred_label": [], "synonyms": []}).to_parquet(p)
    return p


def test_exact_match(tmp_path: Path) -> None:
    uberon_path = _make_empty_uberon(tmp_path)
    norm = OrganNormalizer(uberon_path)
    result = norm.normalize("lung")
    assert result == "lung"


def test_substring_match(tmp_path: Path) -> None:
    uberon_path = _make_empty_uberon(tmp_path)
    norm = OrganNormalizer(uberon_path)
    result = norm.normalize("pulmonary adenocarcinoma")
    assert result == "lung"


def test_no_match_returns_none(tmp_path: Path) -> None:
    uberon_path = _make_empty_uberon(tmp_path)
    norm = OrganNormalizer(uberon_path)
    result = norm.normalize("myxoid stroma")
    assert result is None


def test_infer_from_concept_fallback(tmp_path: Path) -> None:
    uberon_path = _make_empty_uberon(tmp_path)
    norm = OrganNormalizer(uberon_path)
    result = norm.infer_from_concept("myxoid stroma", [], default="universal")
    assert result == "universal"
