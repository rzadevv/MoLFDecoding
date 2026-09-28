"""Tests for level_classifier."""

from __future__ import annotations

from molf_interp.curation.level_classifier import classify_level


def test_tissue_level() -> None:
    assert classify_level("coagulative necrosis") == "tissue"


def test_cellular_level() -> None:
    assert classify_level("mitotic figure") == "cellular"


def test_subcellular_level() -> None:
    assert classify_level("nuclear pleomorphism") == "subcellular"


def test_extracellular_level() -> None:
    assert classify_level("mucin pool") == "extracellular"


def test_stromal_level() -> None:
    assert classify_level("desmoplastic stroma") == "stromal"


def test_default_is_tissue() -> None:
    assert classify_level("xyz unknown feature") == "tissue"


def test_word_boundary_nucleated_not_subcellular() -> None:
    # "nucleated" should NOT match "nucleus"
    result = classify_level("binucleated giant cell")
    assert result == "cellular"  # matches "cell" before "nucleus"
