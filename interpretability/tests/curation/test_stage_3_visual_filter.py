"""Tests for Stage 3: visual filter."""

from __future__ import annotations

from molf_interp.curation.config import VisualFilterConfig
from molf_interp.curation.stage_3_visual_filter import _decide, _get_tier_thresholds


def test_decide_keep() -> None:
    assert _decide(0.7, 0.55, 0.35) == "keep"


def test_decide_drop() -> None:
    assert _decide(0.2, 0.55, 0.35) == "drop"


def test_decide_ambiguous() -> None:
    assert _decide(0.45, 0.55, 0.35) == "ambiguous"


def test_per_tier_threshold_override() -> None:
    cfg = VisualFilterConfig(
        per_tier_thresholds={"H1_morphology": (0.6, 0.4)},
    )
    keep_t, drop_t = _get_tier_thresholds("H1_morphology", cfg)
    assert keep_t == 0.6
    assert drop_t == 0.4


def test_per_tier_threshold_fallback() -> None:
    cfg = VisualFilterConfig()
    keep_t, drop_t = _get_tier_thresholds("H2_pathway", cfg)
    assert keep_t == cfg.similarity_keep_threshold
    assert drop_t == cfg.similarity_drop_threshold
