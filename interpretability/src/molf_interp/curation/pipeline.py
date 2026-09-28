"""Curation pipeline orchestrator."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from molf_interp.concepts.schemas import (
    Concept,
    ConceptBank,
    ConceptPrompt,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)
from molf_interp.concepts.store import read_concept_bank, write_concept_bank
from molf_interp.curation.calibration import CalibratedThresholds, load_calibrated_thresholds
from molf_interp.curation.config import CurationConfig
from molf_interp.curation.negative_classes import NegativeClassCentroids
from molf_interp.curation.organ_normalizer import OrganNormalizer
from molf_interp.curation.sapbert import SapBert
from molf_interp.curation.stage_1_embed import run_stage_1
from molf_interp.curation.stage_2_dedup import run_stage_2
from molf_interp.curation.stage_3_visual_filter import run_stage_3
from molf_interp.curation.stage_4_llm_adjudicate import run_stage_4
from molf_interp.curation.stage_5_tier_refine import run_stage_5
from molf_interp.curation.stage_6_canonicalize import run_stage_6
from molf_interp.curation.stage_7_prompt_gen import run_stage_7

# Base schema for columns shared across all stage parquets.
# Explicit typing prevents pandas from degrading empty list/map columns to null types.
_BASE_SCHEMA_FIELDS: list[pa.Field] = [
    pa.field("source_name", pa.string(), nullable=False),
    pa.field("source_id", pa.string(), nullable=False),
    pa.field("preferred_label", pa.string(), nullable=False),
    pa.field("synonyms", pa.list_(pa.string()), nullable=False),
    pa.field("definition", pa.string(), nullable=True),
    pa.field("parent_ids", pa.list_(pa.string()), nullable=False),
    pa.field("candidate_tier", pa.string(), nullable=False),
    pa.field("extra", pa.map_(pa.string(), pa.string()), nullable=False),
]

# Per-stage additions layered on top of the base schema.
_STAGE_SCHEMA_ADDITIONS: dict[int, list[pa.Field]] = {
    3: [
        pa.field("visual_filter_decision", pa.string(), nullable=False),
        pa.field("reference_tier_similarity", pa.float32(), nullable=True),
    ],
    4: [
        pa.field("visual_filter_decision", pa.string(), nullable=False),
        pa.field("reference_tier_similarity", pa.float32(), nullable=True),
        pa.field("llm_decision", pa.string(), nullable=False),
        pa.field("llm_confidence", pa.string(), nullable=False),
        pa.field("llm_justification", pa.string(), nullable=False),
        pa.field("llm_was_verification_probe", pa.bool_(), nullable=False),
    ],
    5: [
        pa.field("visual_filter_decision", pa.string(), nullable=False),
        pa.field("reference_tier_similarity", pa.float32(), nullable=True),
        pa.field("llm_decision", pa.string(), nullable=False),
        pa.field("llm_confidence", pa.string(), nullable=False),
        pa.field("llm_justification", pa.string(), nullable=False),
        pa.field("llm_was_verification_probe", pa.bool_(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("organ", pa.string(), nullable=True),
        pa.field("level", pa.string(), nullable=True),
        pa.field("category", pa.string(), nullable=True),
        pa.field("concept_type", pa.string(), nullable=True),
    ],
    6: [
        pa.field("visual_filter_decision", pa.string(), nullable=False),
        pa.field("reference_tier_similarity", pa.float32(), nullable=True),
        pa.field("llm_decision", pa.string(), nullable=False),
        pa.field("llm_confidence", pa.string(), nullable=False),
        pa.field("llm_justification", pa.string(), nullable=False),
        pa.field("llm_was_verification_probe", pa.bool_(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("organ", pa.string(), nullable=True),
        pa.field("level", pa.string(), nullable=True),
        pa.field("category", pa.string(), nullable=True),
        pa.field("concept_type", pa.string(), nullable=True),
        pa.field("concept_id", pa.string(), nullable=False),
    ],
}


_STAGE_NAMES = {
    1: "embed",
    2: "dedup",
    3: "visual_filter",
    4: "llm_adjudicate",
    5: "tier_refine",
    6: "canonicalize",
    7: "prompt_gen",
}


def _stage_cache_path(config: CurationConfig, stage: int) -> Path:
    name = _STAGE_NAMES[stage]
    return config.stage_cache_dir / f"stage_{stage}_{name}.parquet"


def _save_stage(df: pd.DataFrame, config: CurationConfig, stage: int) -> None:
    """Write a stage cache parquet with explicit schema preservation.

    pandas roundtrip degrades empty list/map columns to list<null> / list<list>.
    We declare the schema explicitly so types round-trip cleanly even when the
    column is entirely empty for this stage.

    Args:
        df: Stage output DataFrame (may contain 'embedding' column).
        config: CurationConfig instance (provides stage_cache_dir).
        stage: Stage number 1-7.
    """
    path = _stage_cache_path(config, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop embedding: too large and stores as list<float32> which bloats the file.
    save_df = df.drop(columns=["embedding"], errors="ignore")

    # Build explicit schema: base fields present in df + stage-specific additions.
    known_additions = _STAGE_SCHEMA_ADDITIONS.get(stage, [])
    declared_fields: list[pa.Field] = []
    for field in _BASE_SCHEMA_FIELDS:
        if field.name in save_df.columns:
            declared_fields.append(field)
    for field in known_additions:
        if field.name in save_df.columns:
            declared_fields.append(field)
    # Remaining columns (not in declared schema) go in as inferred types.
    declared_col_names = {f.name for f in declared_fields}
    extra_cols = [c for c in save_df.columns if c not in declared_col_names]

    if declared_fields:
        declared_schema = pa.schema(declared_fields)
        # Cast only the declared columns; let pyarrow infer the rest.
        try:
            declared_df = save_df[[f.name for f in declared_fields]]
            declared_table = pa.Table.from_pandas(declared_df, schema=declared_schema, safe=False)
            if extra_cols:
                extra_table = pa.Table.from_pandas(save_df[extra_cols])
                # Merge columns (not rows) from the extra table into the declared table
                for col_name in extra_cols:
                    declared_table = declared_table.append_column(
                        col_name, extra_table.column(col_name)
                    )
                table = declared_table
                # Reorder to match save_df column order
                table = table.select([c for c in save_df.columns if c in table.schema.names])
            else:
                table = declared_table
            pq.write_table(table, path)  # type: ignore[no-untyped-call]
        except Exception:
            # Fallback: plain pandas parquet if schema cast fails
            save_df.to_parquet(path, index=False)
    else:
        save_df.to_parquet(path, index=False)

    logger.info("Saved stage {} cache: {}", stage, path)


def _load_stage(config: CurationConfig, stage: int) -> pd.DataFrame:
    path = _stage_cache_path(config, stage)
    if not path.exists():
        raise FileNotFoundError(f"Stage {stage} cache not found: {path}")
    return pd.read_parquet(path)


def _nan_str(val: object, default: str = "") -> str:
    """Return str(val) or default when val is None/NaN/empty."""
    if val is None:
        return default
    if isinstance(val, float) and pd.isna(val):
        return default
    s = str(val).strip()
    return s if s else default


def _nan_none(val: object) -> str | None:
    """Return str(val) or None when val is None/NaN/empty."""
    result = _nan_str(val)
    return result if result else None


def _df_to_concept_bank(
    concepts_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
) -> ConceptBank:
    """Convert stage 6 + 7 dataframes to ConceptBank."""
    concepts = []
    for _, row in concepts_df.iterrows():
        tier_val = str(row["tier"])
        try:
            tier = Tier(tier_val)
        except ValueError:
            logger.warning("Skipping concept with unknown tier: {}", tier_val)
            continue

        # Determine category/subcategory
        # category comes from Stage 5 (_classify_h1_category for H1, tier-type for H2)
        # subcategory is source_name for provenance traceability
        if tier == Tier.H1_MORPHOLOGY:
            category = _nan_str(row.get("category"), "default")
            subcategory = _nan_str(row.get("source_name"), "automated")
        else:
            category = _nan_str(row.get("concept_type"), tier_val)
            subcategory = _nan_str(row.get("source_name"), "automated")

        try:
            concept = Concept(
                concept_id=str(row["concept_id"]),
                concept_name=str(row["preferred_label"]),
                tier=tier,
                category=category,
                subcategory=subcategory,
                source=_nan_str(row.get("source_name"), "automated"),
                provenance=ProvenanceSource.HARVESTED,
                organ=_nan_none(row.get("organ")),
                level=_nan_none(row.get("level")),
                concept_type=_nan_none(row.get("concept_type")),
            )
            concepts.append(concept)
        except Exception as exc:
            logger.warning("Skipping invalid concept {}: {}", row.get("concept_id"), exc)

    valid_ids = {c.concept_id for c in concepts}

    prompts = []
    for _, row in prompts_df.iterrows():
        cid = str(row["concept_id"])
        if cid not in valid_ids:
            continue
        tier_val = str(row["tier"])
        try:
            tier = Tier(tier_val)
        except ValueError:
            continue
        try:
            prompt = ConceptPrompt(
                concept_id=cid,
                concept_name=str(row["concept_name"]),
                tier=tier,
                species=Species(str(row["species"])),
                prompt_text=str(row["prompt_text"]),
                species_status=SpeciesStatus(str(row["species_status"])),
            )
            prompts.append(prompt)
        except Exception as exc:
            logger.warning("Skipping invalid prompt {}: {}", cid, exc)

    return ConceptBank(
        concepts=tuple(concepts),
        prompts=tuple(prompts),
        provenance=ProvenanceSource.HARVESTED,
    )


def run_curation_pipeline(
    config: CurationConfig,
    *,
    resume_from: int | None = None,
    save_stages: bool = True,
) -> tuple[ConceptBank, Path]:
    """Run the full curation pipeline. Returns (ConceptBank, output_dir).

    Args:
        config: CurationConfig instance.
        resume_from: If set, resume from this stage number (previous stage cache
            must exist). None means start from scratch.
        save_stages: Whether to save per-stage parquet checkpoints.

    Returns:
        Tuple of (ConceptBank, output_dir Path).
    """
    t_total = time.monotonic()
    logger.info("Starting curation pipeline (resume_from={})", resume_from)

    sapbert = SapBert(config.sapbert)
    organ_normalizer = OrganNormalizer(config.tier_refine.uberon_table_path, sapbert)

    # Load calibrated thresholds if available (graceful fallback to global thresholds)
    calibrated_thresholds: CalibratedThresholds | None = None
    try:
        calibrated_thresholds = load_calibrated_thresholds(config.calibrated_thresholds_path)
        logger.info("Loaded calibrated thresholds from {}", config.calibrated_thresholds_path)
    except FileNotFoundError:
        logger.info(
            "No calibrated_thresholds.yaml found at {}; Stage 3 will use global thresholds.",
            config.calibrated_thresholds_path,
        )

    # Load negative-class centroids if pre-computed
    negative_centroids: NegativeClassCentroids | None = None
    neg_centroids_dir = config.negative_classes.cache_dir / "centroids"
    try:
        negative_centroids = NegativeClassCentroids.load(neg_centroids_dir)
        logger.info("Loaded negative-class centroids from {}", neg_centroids_dir)
    except FileNotFoundError:
        logger.info(
            "No pre-computed negative centroids at {}; Stage 3 negative-proximity "
            "check disabled. Run `curate calibrate` to build them.",
            neg_centroids_dir,
        )

    # Load input or resume from cached stage
    if resume_from is not None and resume_from > 1:
        logger.info("Resuming from stage {} cache…", resume_from - 1)
        df = _load_stage(config, resume_from - 1)
        # Re-embed if embedding column missing
        if "embedding" not in df.columns:
            logger.info("Re-embedding from stage 1 cache for resume…")
            embed_df = run_stage_1(df, config, sapbert)
            df["embedding"] = embed_df["embedding"]
        start_stage = resume_from
    else:
        if not config.input_path.exists():
            raise FileNotFoundError(f"Pipeline input not found: {config.input_path}")
        df = pd.read_parquet(config.input_path)
        logger.info("Loaded {} raw concepts from {}", len(df), config.input_path)
        start_stage = 1

    stage_times: dict[int, float] = {}

    def _run_stage(stage: int, fn: Callable[..., pd.DataFrame], *args: object) -> pd.DataFrame:
        t0 = time.monotonic()
        result: pd.DataFrame = fn(*args)
        elapsed = time.monotonic() - t0
        stage_times[stage] = elapsed
        sapbert.flush()  # persist any new embeddings computed during this stage
        if save_stages:
            _save_stage(result, config, stage)
        return result

    # Load reference bank once and share across stages 3 and 5
    reference_bank = None
    if start_stage <= 5:
        try:
            reference_bank = read_concept_bank(config.visual_filter.reference_bank_dir)
            logger.info("Loaded reference bank from {}", config.visual_filter.reference_bank_dir)
        except Exception as exc:
            logger.warning("Could not load reference bank: {}; stages will load individually", exc)

    if start_stage <= 1:
        df = _run_stage(1, run_stage_1, df, config, sapbert)

    if start_stage <= 2:
        df = _run_stage(2, run_stage_2, df, config, sapbert)

    if start_stage <= 3:
        df = _run_stage(
            3,
            run_stage_3,
            df,
            config,
            sapbert,
            reference_bank,
            negative_centroids,
            calibrated_thresholds,
        )

    if start_stage <= 4:
        df = _run_stage(4, run_stage_4, df, config)

    if start_stage <= 5:
        df = _run_stage(5, run_stage_5, df, config, sapbert, reference_bank, organ_normalizer)

    if start_stage <= 6:
        df = _run_stage(6, run_stage_6, df, config)

    # Stage 7 generates prompt rows (different structure — don't save as replacement of df)
    prompts_df: pd.DataFrame
    if start_stage <= 7:
        t0 = time.monotonic()
        prompts_df = run_stage_7(df, config)
        stage_times[7] = time.monotonic() - t0
        if save_stages:
            prompts_path = config.stage_cache_dir / "stage_7_prompt_gen_prompts.parquet"
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_df.to_parquet(prompts_path, index=False)
    else:
        prompts_df = pd.read_parquet(config.stage_cache_dir / "stage_7_prompt_gen_prompts.parquet")

    logger.info("Converting to ConceptBank…")
    bank = _df_to_concept_bank(df, prompts_df)
    logger.info(
        "ConceptBank: {} concepts, {} prompts",
        len(bank.concepts),
        len(bank.prompts),
    )

    output_dir = config.output_dir
    write_concept_bank(bank, output_dir)
    logger.info("ConceptBank written to {}", output_dir)

    total = time.monotonic() - t_total
    logger.info(
        "Pipeline complete in {:.1f}s. Stage times: {}",
        total,
        {f"stage_{k}": f"{v:.1f}s" for k, v in stage_times.items()},
    )
    return bank, output_dir
