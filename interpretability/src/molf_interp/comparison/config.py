"""Configuration for cross-bank comparison."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from molf_interp.curation.config import SapBertConfig
from molf_interp.io.config import BaseConfig


class ComparisonConfig(BaseConfig):
    """Configuration for cross-bank comparison."""

    # Input banks
    reference_bank_dir: Path = Path("data/cache/concept_bank/reference")
    harvested_bank_dir: Path = Path("data/cache/concept_bank/harvested")

    # Matching parameters
    match_threshold: float = 0.85
    """Cosine similarity above which two cross-bank concepts are considered matched."""

    cross_tier_match_threshold: float = 0.92
    """Higher threshold for cross-tier matches (flagged as tier-disagreement)."""

    enable_cross_tier_search: bool = True
    """If True, search for cross-tier matches above cross_tier_match_threshold."""

    # SapBERT
    sapbert_config: SapBertConfig = Field(default_factory=SapBertConfig)

    # Output
    output_dir: Path = Path("data/outputs/comparison")

    # UMAP visualization
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_random_state: int = 42
    umap_max_points: int = 6000
    plot_dpi: int = 150
    plot_figsize: tuple[float, float] = (14.0, 10.0)

    # Report content
    examples_per_tier: int = 5
    top_k_unmatched: int = 20
