"""Tests for Stage 5 category classifier v2 (regression + new keywords)."""

from __future__ import annotations

from molf_interp.curation.stage_5_tier_refine import _classify_h1_category

# ---------------------------------------------------------------------------
# Regression: breast Papillary DCIS must be "organ_specific", not "microenvironment"
# (The old bug was the "infiltrat" keyword causing a false-positive microenvironment match.)
# ---------------------------------------------------------------------------


def test_breast_dcis_is_organ_specific() -> None:
    assert (
        _classify_h1_category(
            "breast Papillary Ductal Carcinoma In Situ",
            [],
            level="tissue",
            organ="breast",
        )
        == "organ_specific"
    ), "Breast DCIS was incorrectly classified (expected organ_specific)"


# ---------------------------------------------------------------------------
# Regression: "lymphocytic infiltration" must NOT be microenvironment
# Verifies "infiltrat" keyword was removed from MICROENVIRONMENT_KEYWORDS.
# ---------------------------------------------------------------------------


def test_lymphocytic_infiltration_is_default() -> None:
    assert (
        _classify_h1_category(
            "lymphocytic infiltration",
            ["infiltrating lymphocytes"],
            level="tissue",
            organ="universal",
        )
        == "default"
    )


# ---------------------------------------------------------------------------
# Microenvironment positive cases
# ---------------------------------------------------------------------------


def test_tertiary_lymphoid_structure_is_microenvironment() -> None:
    assert (
        _classify_h1_category("tertiary lymphoid structure", [], level="tissue", organ="universal")
        == "microenvironment"
    )


def test_microenvironment_keyword_in_label() -> None:
    assert (
        _classify_h1_category("tumor microenvironment", [], level="tissue", organ="universal")
        == "microenvironment"
    )


def test_niche_keyword_in_label() -> None:
    assert (
        _classify_h1_category("perivascular niche", [], level="tissue", organ="universal")
        == "microenvironment"
    )


def test_immune_desert_is_microenvironment() -> None:
    assert (
        _classify_h1_category("immune desert phenotype", [], level="tissue", organ="universal")
        == "microenvironment"
    )


def test_peritumoral_is_microenvironment() -> None:
    assert (
        _classify_h1_category("peritumoral inflammation", [], level="tissue", organ="universal")
        == "microenvironment"
    )


# ---------------------------------------------------------------------------
# Artifact positive cases
# ---------------------------------------------------------------------------


def test_crush_artifact_is_artifact() -> None:
    assert (
        _classify_h1_category("crush artifact", [], level="tissue", organ="universal") == "artifact"
    )


def test_fixation_artifact_is_artifact() -> None:
    assert (
        _classify_h1_category("cold fixation artifact", [], level="tissue", organ="universal")
        == "artifact"
    )


def test_autolysis_is_artifact() -> None:
    assert _classify_h1_category("autolysis", [], level="cellular", organ="universal") == "artifact"


def test_artifact_in_synonym() -> None:
    assert (
        _classify_h1_category(
            "postmortem change",
            ["postmortem artifact", "autolysis artifact"],
            level="tissue",
            organ="universal",
        )
        == "artifact"
    )


# ---------------------------------------------------------------------------
# Organ-specific
# ---------------------------------------------------------------------------


def test_lung_adenocarcinoma_organ_specific() -> None:
    assert (
        _classify_h1_category("lung adenocarcinoma", [], level="tissue", organ="lung")
        == "organ_specific"
    )


def test_organ_specific_requires_tissue_level() -> None:
    assert (
        _classify_h1_category("hepatocellular steatosis", [], level="cellular", organ="liver")
        == "default"
    )


def test_universal_organ_not_organ_specific() -> None:
    assert _classify_h1_category("necrosis", [], level="tissue", organ="universal") == "default"


def test_none_organ_not_organ_specific() -> None:
    assert _classify_h1_category("necrosis", [], level="tissue", organ=None) == "default"


# ---------------------------------------------------------------------------
# Default catch-all
# ---------------------------------------------------------------------------


def test_necrosis_with_universal_organ_is_default() -> None:
    assert _classify_h1_category("necrosis", [], level="tissue", organ="universal") == "default"


def test_nuclear_pleomorphism_is_default() -> None:
    assert (
        _classify_h1_category("nuclear pleomorphism", [], level="cellular", organ="universal")
        == "default"
    )


# ---------------------------------------------------------------------------
# Priority: artifact before microenvironment
# ---------------------------------------------------------------------------


def test_artifact_wins_over_microenvironment() -> None:
    # Label contains both "artifact" and "niche"
    result = _classify_h1_category(
        "tissue fold artifact in niche region", [], level="tissue", organ="universal"
    )
    assert result == "artifact"
