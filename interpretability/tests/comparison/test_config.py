"""Tests for ComparisonConfig loading and defaults."""

from __future__ import annotations

from pathlib import Path

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.io.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config(Path("configs/comparison/default.yaml"), ComparisonConfig)
    assert cfg.match_threshold == 0.85
    assert cfg.cross_tier_match_threshold == 0.92
    assert cfg.enable_cross_tier_search is True
    assert cfg.umap_n_neighbors == 15
    assert cfg.examples_per_tier == 5
    assert cfg.top_k_unmatched == 20


def test_default_config_paths() -> None:
    cfg = load_config(Path("configs/comparison/default.yaml"), ComparisonConfig)
    assert "reference" in str(cfg.reference_bank_dir)
    assert "harvested" in str(cfg.harvested_bank_dir)
    assert "comparison" in str(cfg.output_dir)


def test_config_defaults_without_file() -> None:
    cfg = ComparisonConfig()
    assert cfg.match_threshold == 0.85
    assert cfg.plot_figsize == (14.0, 10.0)
    assert cfg.umap_max_points == 6000
