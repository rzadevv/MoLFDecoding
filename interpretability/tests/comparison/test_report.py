"""Tests for comparison report generation."""

from __future__ import annotations

from pathlib import Path

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.comparison.matcher import match_banks
from molf_interp.comparison.merger import merge_banks
from molf_interp.comparison.report import generate_report
from tests.comparison.conftest import MockSapBert


def _run_pipeline(tiny_reference_bank, tiny_harvested_bank, threshold=0.85):
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=threshold)
    return merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)


def test_report_generates_without_error(
    tmp_path: Path, tiny_reference_bank, tiny_harvested_bank
) -> None:
    """generate_report runs to completion on tiny fixture."""
    result = _run_pipeline(tiny_reference_bank, tiny_harvested_bank)
    cfg = ComparisonConfig()
    out = tmp_path / "report.md"
    generate_report(result, cfg, out)
    assert out.exists()
    assert out.stat().st_size > 100


def test_all_section_headers_present(
    tmp_path: Path, tiny_reference_bank, tiny_harvested_bank
) -> None:
    """All 9 section headers appear in the report."""
    result = _run_pipeline(tiny_reference_bank, tiny_harvested_bank)
    cfg = ComparisonConfig()
    out = tmp_path / "report.md"
    generate_report(result, cfg, out)
    content = out.read_text()
    for heading in [
        "## 1. Per-tier coverage",
        "## 2. Match similarity distribution",
        "## 3. Matched examples",
        "## 4. Reference-only concepts",
        "## 5. Harvested-only concepts",
        "## 6. Top",
        "## 7. Top",
        "## 8. Cross-tier matches",
        "## 9. Discussion",
    ]:
        assert heading in content, f"Missing section: {heading!r}"


def test_tier_table_has_correct_rows(
    tmp_path: Path, tiny_reference_bank, tiny_harvested_bank
) -> None:
    """Per-tier table contains one row per reference tier in the fixture."""
    result = _run_pipeline(tiny_reference_bank, tiny_harvested_bank)
    cfg = ComparisonConfig()
    out = tmp_path / "report.md"
    generate_report(result, cfg, out)
    content = out.read_text()
    for tier in ["H1_morphology", "H2_cell_type", "H2_gene_program", "H2_pathway"]:
        assert tier in content


def test_sampling_is_seeded_reproducible(
    tmp_path: Path, tiny_reference_bank, tiny_harvested_bank
) -> None:
    """Two runs with same seed produce identical output."""
    result = _run_pipeline(tiny_reference_bank, tiny_harvested_bank)
    cfg = ComparisonConfig()

    out1 = tmp_path / "report1.md"
    out2 = tmp_path / "report2.md"
    generate_report(result, cfg, out1, rng_seed=42)
    generate_report(result, cfg, out2, rng_seed=42)

    assert out1.read_text() == out2.read_text()


def test_different_seeds_may_differ(
    tmp_path: Path, tiny_reference_bank, tiny_harvested_bank
) -> None:
    """Two runs with different seeds can (and for realistic banks, do) differ."""
    result = _run_pipeline(tiny_reference_bank, tiny_harvested_bank)
    cfg = ComparisonConfig()
    out = tmp_path / "report_alt.md"
    generate_report(result, cfg, out, rng_seed=99)
    assert out.exists()
