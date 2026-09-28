"""Tests for merge_banks and write_merged_bank."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from molf_interp.comparison.matcher import match_banks
from molf_interp.comparison.merger import merge_banks, write_merged_bank
from tests.comparison.conftest import MockSapBert


def test_concept_count(tiny_reference_bank, tiny_harvested_bank) -> None:
    """Merged total == reference_total + harvested_total (no dedup)."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)
    assert len(result.merged_concepts) == len(tiny_reference_bank) + len(tiny_harvested_bank)


def test_cross_bank_status_column(tiny_reference_bank, tiny_harvested_bank) -> None:
    """cross_bank_status is correctly assigned for matched / reference_only / harvested_only."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)

    matched_ref = {m.reference_concept_id for m in matches}
    matched_harv = {m.harvested_concept_id for m in matches}

    for _, row in result.merged_concepts.iterrows():
        cid = row["concept_id"]
        prov = row["provenance"]
        status = row["cross_bank_status"]
        if prov == "reference":
            expected = "matched" if cid in matched_ref else "reference_only"
        else:
            expected = "matched" if cid in matched_harv else "harvested_only"
        assert status == expected, f"{cid}: expected {expected}, got {status}"


def test_prompt_count(tiny_reference_bank, tiny_harvested_bank) -> None:
    """Merged prompts count == total prompts across both banks."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)
    expected = len(tiny_reference_bank.prompts) + len(tiny_harvested_bank.prompts)
    assert len(result.merged_prompts) == expected


def test_write_and_read_roundtrip(tmp_path: Path, tiny_reference_bank, tiny_harvested_bank) -> None:
    """Write → read preserves merged_concepts exactly."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)
    write_merged_bank(result, tmp_path)

    loaded = pd.read_parquet(tmp_path / "merged_concepts.parquet")
    pd.testing.assert_frame_equal(
        result.merged_concepts.reset_index(drop=True),
        loaded.reset_index(drop=True),
        check_like=True,
    )


def test_summary_tier_counts_add_up(tiny_reference_bank, tiny_harvested_bank) -> None:
    """Summary tier counts sum to totals."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)
    s = result.summary
    assert s["merged_total"] == s["reference_total"] + s["harvested_total"]
    assert s["matches_total"] == len(matches)


def test_write_creates_all_output_files(
    tmp_path: Path, tiny_reference_bank, tiny_harvested_bank
) -> None:
    """write_merged_bank creates concepts, prompts, matches, summary files."""
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)
    write_merged_bank(result, tmp_path)

    assert (tmp_path / "merged_concepts.parquet").exists()
    assert (tmp_path / "merged_prompts.parquet").exists()
    assert (tmp_path / "matches.parquet").exists()
    assert (tmp_path / "summary.json").exists()
