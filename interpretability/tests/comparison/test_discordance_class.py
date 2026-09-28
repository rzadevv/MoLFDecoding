"""Tests for _classify_discordance helper in matcher.py."""

from __future__ import annotations

from molf_interp.comparison.matcher import _classify_discordance


def test_within_tier_is_exact() -> None:
    assert _classify_discordance(None, None, False) == "exact"


def test_same_tier_not_cross_tier() -> None:
    assert _classify_discordance("H1_morphology", "H1_morphology", False) == "exact"


def test_h1_vs_h2_cell_type() -> None:
    assert (
        _classify_discordance("H1_morphology", "H2_cell_type", True)
        == "cross_tier_h1_vs_h2_cell_type"
    )


def test_h1_vs_h2_cell_type_order_independent() -> None:
    assert (
        _classify_discordance("H2_cell_type", "H1_morphology", True)
        == "cross_tier_h1_vs_h2_cell_type"
    )


def test_program_vs_pathway() -> None:
    assert (
        _classify_discordance("H2_gene_program", "H2_pathway", True)
        == "cross_tier_program_vs_pathway"
    )


def test_program_vs_pathway_order_independent() -> None:
    assert (
        _classify_discordance("H2_pathway", "H2_gene_program", True)
        == "cross_tier_program_vs_pathway"
    )


def test_other_cross_tier() -> None:
    assert _classify_discordance("H1_morphology", "H2_pathway", True) == "cross_tier_other"


def test_h2_niche_cross_tier_is_other() -> None:
    assert _classify_discordance("H1_morphology", "H2_niche", True) == "cross_tier_other"
