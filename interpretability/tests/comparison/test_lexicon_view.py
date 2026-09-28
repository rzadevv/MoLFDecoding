"""Tests for primary lexicon view (View A) construction."""

from __future__ import annotations

import pandas as pd
import pytest

from molf_interp.comparison.lexicon_view import (
    LexiconViewConfig,
    build_primary_lexicon_view,
    compute_nearest_ref_within_tier,
)
from tests.comparison.conftest import MockSapBert

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mc(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal merged_concepts DataFrame."""
    return pd.DataFrame(rows)


def _su_row(cid: str, name: str, tier: str = "H1_morphology") -> dict:
    return {
        "concept_id": cid,
        "concept_name": name,
        "tier": tier,
        "provenance": "reference",
        "cross_bank_status": "reference_only",
    }


def _auto_row(
    cid: str,
    name: str,
    tier: str = "H1_morphology",
    status: str = "harvested_only",
) -> dict:
    return {
        "concept_id": cid,
        "concept_name": name,
        "tier": tier,
        "provenance": "harvested",
        "cross_bank_status": status,
    }


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "reference_concept_id",
            "harvested_concept_id",
            "tier",
            "is_cross_tier",
            "discordance_class",
        ]
    )


def _within_tier_match(su_id: str, auto_id: str, tier: str = "H1_morphology") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_concept_id": su_id,
                "harvested_concept_id": auto_id,
                "tier": tier,
                "is_cross_tier": False,
                "discordance_class": "exact",
            }
        ]
    )


def _cross_tier_match(su_id: str, auto_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_concept_id": su_id,
                "harvested_concept_id": auto_id,
                "tier": "H1_morphology",
                "is_cross_tier": True,
                "discordance_class": "cross_tier_h1_vs_h2_cell_type",
            }
        ]
    )


# ---------------------------------------------------------------------------
# Reference concepts always included
# ---------------------------------------------------------------------------


def test_su_concepts_always_included() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "x"),  # 1-char label — would fail label_too_short for auto
            _su_row("SU-002", "coagulative necrosis"),
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert result.in_primary_lexicon["SU-001"] is True or bool(result.in_primary_lexicon["SU-001"])
    assert bool(result.in_primary_lexicon["SU-002"])


# ---------------------------------------------------------------------------
# Automated within-tier match → included
# ---------------------------------------------------------------------------


def test_automated_within_tier_match_included() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "Coagulative necrosis", status="matched"),
        ]
    )
    matches = _within_tier_match("SU-001", "AUTO-001")
    result = build_primary_lexicon_view(mc, matches, MockSapBert(), LexiconViewConfig())
    assert bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == ""


# ---------------------------------------------------------------------------
# Cross-tier matched automated NOT auto-included; falls back to gates
# ---------------------------------------------------------------------------


def test_cross_tier_matched_not_auto_included_falls_to_gates() -> None:
    """Cross-tier match does NOT bypass the quality gates.
    The automated concept here has a long label that would normally exclude it.
    """
    long_label = "one two three four five six seven eight nine"  # 9 tokens > 8
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", long_label, tier="H2_cell_type", status="matched"),
        ]
    )
    matches = _cross_tier_match("SU-001", "AUTO-001")
    result = build_primary_lexicon_view(mc, matches, MockSapBert(), LexiconViewConfig())
    # Should be excluded because label is too long
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "label_too_long"


# ---------------------------------------------------------------------------
# Label filter tests
# ---------------------------------------------------------------------------


def test_label_too_short_excluded() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "xx"),  # 2 chars < 4
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "label_too_short"
    assert result.summary["by_exclusion_reason"]["label_too_short"] == 1


def test_label_too_long_excluded() -> None:
    long_label = "one two three four five six seven eight nine"  # 9 tokens
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", long_label),
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "label_too_long"
    assert result.summary["by_exclusion_reason"]["label_too_long"] == 1


def test_excluded_substring() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "Abacavir transmembrane transport"),
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "excluded_substring"


def test_excluded_substring_case_insensitive() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "Some Downstream Events pathway"),
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "excluded_substring"


# ---------------------------------------------------------------------------
# Definition / synonyms filter (only when columns present)
# ---------------------------------------------------------------------------


def test_no_definition_no_synonyms_excluded_when_columns_present() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            {
                **_auto_row("AUTO-001", "valid label name"),
                "definition": None,
                "synonyms": [],
            },
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "no_definition_no_synonyms"


def test_has_definition_passes_filter() -> None:
    """A concept with a definition passes the definition/synonyms filter and
    proceeds to the SapBERT similarity check. MockSapBert gives ~1.0 for
    the identical label, so it should be included."""
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            {
                **_auto_row("AUTO-001", "coagulative necrosis"),  # identical → sim ~1.0
                "definition": "A form of necrosis",
                "synonyms": [],
            },
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    assert bool(result.in_primary_lexicon["AUTO-001"])


def test_definition_synonyms_filter_skipped_when_columns_absent() -> None:
    """Without definition/synonyms columns the filter is silently skipped."""
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "coagulative necrosis"),  # no definition/synonyms cols
        ]
    )
    cfg = LexiconViewConfig(require_definition_or_synonyms=True)
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), cfg)
    # Should NOT be excluded for no_definition_no_synonyms
    assert result.reason_excluded["AUTO-001"] != "no_definition_no_synonyms"


# ---------------------------------------------------------------------------
# SapBERT similarity filter
# ---------------------------------------------------------------------------


def test_low_reference_neighbor_similarity_excluded() -> None:
    """AUTO-001 has a unique random label → low cosine to reference → excluded."""
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "zxqv_unique_morphology_xyz_99999"),
        ]
    )
    cfg = LexiconViewConfig(reference_neighbor_similarity_threshold=0.70)
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), cfg)
    assert not bool(result.in_primary_lexicon["AUTO-001"])
    assert result.reason_excluded["AUTO-001"] == "low_reference_neighbor_similarity"


def test_high_su_neighbor_similarity_included() -> None:
    """AUTO-001 with identical label to a reference concept → sim ≈ 1.0 → included."""
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis"),
            _auto_row("AUTO-001", "coagulative necrosis"),
        ]
    )
    cfg = LexiconViewConfig(reference_neighbor_similarity_threshold=0.70)
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), cfg)
    assert bool(result.in_primary_lexicon["AUTO-001"])


# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------


def test_summary_tier_counts_sum_to_total() -> None:
    mc = _make_mc(
        [
            _su_row("SU-001", "coagulative necrosis", "H1_morphology"),
            _su_row("SU-002", "macrophage", "H2_cell_type"),
            _auto_row("AUTO-001", "coagulative necrosis", "H1_morphology"),
        ]
    )
    result = build_primary_lexicon_view(mc, _empty_matches(), MockSapBert(), LexiconViewConfig())
    total_from_tiers = sum(result.summary["by_tier"].values())
    assert total_from_tiers == result.summary["total"]


def test_all_exclusion_reasons_in_summary() -> None:
    result = build_primary_lexicon_view(
        _make_mc([_su_row("SU-001", "coagulative necrosis")]),
        _empty_matches(),
        MockSapBert(),
        LexiconViewConfig(),
    )
    expected = {
        "label_too_short",
        "label_too_long",
        "excluded_substring",
        "no_definition_no_synonyms",
        "low_reference_neighbor_similarity",
    }
    assert expected == set(result.summary["by_exclusion_reason"].keys())


# ---------------------------------------------------------------------------
# compute_nearest_ref_within_tier
# ---------------------------------------------------------------------------


def test_nearest_ref_nan_for_tiers_with_no_ref() -> None:
    """Concepts in tiers where the reference bank has zero concepts should get NaN."""
    harv_df = pd.DataFrame(
        [{"concept_id": "AUTO-001", "concept_name": "senescence niche", "tier": "H2_niche"}]
    )
    ref_df = pd.DataFrame(
        [{"concept_id": "SU-001", "concept_name": "coagulative necrosis", "tier": "H1_morphology"}]
    )
    result = compute_nearest_ref_within_tier(harv_df, ref_df, MockSapBert())
    assert pd.isna(result["AUTO-001"])


def test_nearest_ref_identical_label_returns_one() -> None:
    """Identical label → MockSapBert gives cosine 1.0."""
    harv_df = pd.DataFrame(
        [
            {
                "concept_id": "AUTO-001",
                "concept_name": "coagulative necrosis",
                "tier": "H1_morphology",
            }
        ]
    )
    ref_df = pd.DataFrame(
        [{"concept_id": "SU-001", "concept_name": "coagulative necrosis", "tier": "H1_morphology"}]
    )
    result = compute_nearest_ref_within_tier(harv_df, ref_df, MockSapBert())
    assert pytest.approx(result["AUTO-001"], abs=1e-4) == 1.0


def test_nearest_ref_empty_harvested_df() -> None:
    harv_df = pd.DataFrame(columns=["concept_id", "concept_name", "tier"])
    ref_df = pd.DataFrame(
        [{"concept_id": "SU-001", "concept_name": "coagulative necrosis", "tier": "H1_morphology"}]
    )
    result = compute_nearest_ref_within_tier(harv_df, ref_df, MockSapBert())
    assert len(result) == 0
