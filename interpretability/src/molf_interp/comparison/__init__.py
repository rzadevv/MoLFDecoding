"""Cross-bank comparison: matcher, merger, report, and UMAP visualization."""

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.comparison.lexicon_view import (
    LexiconViewConfig,
    LexiconViewResult,
    build_primary_lexicon_view,
    compute_nearest_ref_within_tier,
)
from molf_interp.comparison.matcher import ConceptMatch, compute_unmatched_nearest_sims, match_banks
from molf_interp.comparison.merger import MergedBankResult, merge_banks, write_merged_bank
from molf_interp.comparison.report import generate_report
from molf_interp.comparison.visualization import (
    UmapResult,
    compute_umap_projection,
    render_umap_plot,
)

__all__ = [
    "ComparisonConfig",
    "ConceptMatch",
    "LexiconViewConfig",
    "LexiconViewResult",
    "MergedBankResult",
    "UmapResult",
    "build_primary_lexicon_view",
    "compute_nearest_ref_within_tier",
    "compute_umap_projection",
    "compute_unmatched_nearest_sims",
    "generate_report",
    "match_banks",
    "merge_banks",
    "render_umap_plot",
    "write_merged_bank",
]
