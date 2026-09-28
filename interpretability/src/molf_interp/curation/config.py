"""Configuration models for the curation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from molf_interp.curation.negative_classes import NegativeClassConfig
from molf_interp.io.config import BaseConfig


class SapBertConfig(BaseConfig):
    """SapBERT model configuration."""

    model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    max_length: int = 25
    batch_size: int = 128
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    cache_dir: Path = Path("data/cache/sapbert")


class DedupConfig(BaseConfig):
    """Stage 2: cross-source dedup within each candidate_tier."""

    similarity_threshold: float = 0.85
    method: Literal["agglomerative", "hdbscan"] = "hdbscan"
    min_cluster_size: int = 2
    gene_overlap_threshold: float = 0.5
    canonical_source_priority: tuple[str, ...] = (
        "snomed",
        "ncit",
        "cell_ontology",
        "gene_ontology",
        "msigdb_h.all",
        "msigdb_c2.cp.reactome",
        "msigdb_c2.cp.kegg_legacy",
        "pubmed",
    )


class VisualFilterConfig(BaseConfig):
    """Stage 3: SapBERT-based H&E-visibility filter."""

    reference_bank_dir: Path = Path("data/cache/concept_bank/reference")
    # Global fallback thresholds used when calibrated_thresholds_path is absent or
    # when running tests that override them to extreme values.
    similarity_keep_threshold: float = 0.55
    similarity_drop_threshold: float = 0.35
    reference_centroid_method: Literal["mean", "topk_max"] = "mean"
    topk: int = 5
    # Per-tier override map (tier_str → (keep_t, drop_t)); takes priority over both
    # calibrated thresholds and global fallback when set.
    per_tier_thresholds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # KNN parameters
    knn_k: int = 10
    knn_batch_size: int = 256


class LLMAdjudicationConfig(BaseConfig):
    """Stage 4: Ollama LLM adjudication."""

    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b-instruct"
    timeout_seconds: float = 60.0
    max_concurrent: int = 1
    batch_size: int = 8
    prompt_template_path: Path = Path("configs/curation/llm_prompt_template.txt")
    fail_on_unreachable: bool = False


class TierRefineConfig(BaseConfig):
    """Stage 5: tier reassignment + H1 metadata."""

    reassignment_enabled: bool = True
    reassignment_min_neighbor_count: int = 3
    reassignment_similarity_floor: float = 0.50
    uberon_table_path: Path = Path("data/cache/harvest/uberon_organs.parquet")
    organ_default: str = "universal"


class CanonicalizeConfig(BaseConfig):
    """Stage 6: ID minting."""

    id_offset: int = 10000
    sentence_case: bool = True


class PromptGenConfig(BaseConfig):
    """Stage 7: cross-species prompt generation."""

    species: tuple[str, ...] = ("Homo_sapiens", "Mus_musculus")
    templates_path: Path = Path("configs/curation/prompt_templates.yaml")


class CurationConfig(BaseConfig):
    """Top-level curation pipeline configuration."""

    input_path: Path = Path("data/cache/harvest/all_raw_concepts.parquet")
    output_dir: Path = Path("data/cache/concept_bank/harvested")
    stage_cache_dir: Path = Path("data/cache/curation/stages")

    sapbert: SapBertConfig = Field(default_factory=SapBertConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    visual_filter: VisualFilterConfig = Field(default_factory=VisualFilterConfig)
    llm: LLMAdjudicationConfig = Field(default_factory=LLMAdjudicationConfig)
    tier_refine: TierRefineConfig = Field(default_factory=TierRefineConfig)
    canonicalize: CanonicalizeConfig = Field(default_factory=CanonicalizeConfig)
    prompt_gen: PromptGenConfig = Field(default_factory=PromptGenConfig)
    negative_classes: NegativeClassConfig = Field(default_factory=NegativeClassConfig)

    # Path to calibrated_thresholds.yaml produced by `curate calibrate`.
    # When the file is absent Stage 3 falls back to VisualFilterConfig global thresholds.
    calibrated_thresholds_path: Path = Path("configs/curation/calibrated_thresholds.yaml")

    # Fraction of "keep" candidates sampled as LLM verification probes (matches default.yaml).
    llm_verification_sample_rate: float = 0.0
    llm_verification_seed: int = 42

    # When True, adjudicate ALL "keep" rows (matches default.yaml production setting).
    llm_adjudicate_keeps_too: bool = True
