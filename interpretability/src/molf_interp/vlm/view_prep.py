"""Prepare View A concept prompts for VLM text embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

CLINICAL_DROP_IDS: frozenset[str] = frozenset(
    {
        "MOR-11275",
        "MOR-11279",
        "MOR-11920",
        "MOR-12987",
    }
)

_CLINICAL_DROP_NAMES_BY_ID: dict[str, str] = {
    "MOR-11275": "Grade 2 clear cell renal cell carcinoma",
    "MOR-11279": "Grade 3 clear cell renal cell carcinoma",
    "MOR-11920": "Microsatellite stable endometrial carcinoma",
    "MOR-12987": "Squamous intraepithelial neoplasia grade 2",
}

VISUAL_TIERS: frozenset[str] = frozenset(
    {
        "H1_morphology",
        "H2_cell_type",
        "H2_niche",
    }
)

TRANSCRIPTOMIC_TIERS: frozenset[str] = frozenset(
    {
        "H2_gene_program",
        "H2_pathway",
    }
)

_DEDUP_MAP_COLUMNS: list[str] = [
    "duplicate_concept_id",
    "canonical_concept_id",
    "tier",
    "concept_name",
    "duplicate_provenance",
    "canonical_provenance",
    "reason",
]

_READY_PROMPT_COLUMNS: list[str] = [
    "concept_id",
    "concept_name",
    "tier",
    "embedding_set",
    "species",
    "prompt_text",
    "species_status",
]


@dataclass(frozen=True)
class ViewPrepResult:
    """Summary of View A preparation output."""

    n_concepts_before: int
    n_clinical_dropped: int
    n_dedup_dropped: int
    n_concepts_after: int
    n_visual_concepts: int
    n_transcriptomic_concepts: int
    n_prompts_before: int
    n_prompts_after: int
    n_human_only_prompt_rows_before: int
    n_human_only_mouse_rows_excluded: int


def assign_embedding_set(tier: str) -> str:
    """Map a View A tier to an embedding set.

    Args:
        tier: View A tier name.

    Returns:
        ``"visual"`` for H1_morphology, H2_cell_type, and H2_niche.
        ``"transcriptomic"`` for H2_gene_program and H2_pathway.

    Raises:
        ValueError: If tier is not one of the known View A tiers.
    """
    if tier in VISUAL_TIERS:
        return "visual"
    if tier in TRANSCRIPTOMIC_TIERS:
        return "transcriptomic"
    raise ValueError(f"Unknown View A tier: {tier!r}")


def concept_id_numeric_part(concept_id: str) -> int:
    """Return the numeric component of a concept ID.

    Examples:
        MOR-0005 -> 5
        MOR-10450 -> 10450
        CTY-0073 -> 73

    Raises:
        ValueError: If the concept ID does not contain a numeric component.
    """
    match = re.search(r"\d+", concept_id)
    if match is None:
        raise ValueError(f"Concept ID has no numeric component: {concept_id!r}")
    return int(match.group(0))


def normalize_concept_name(name: str) -> str:
    """Normalize a concept name for duplicate detection.

    Uses case-insensitive matching and whitespace normalization. It does not
    apply stemming, synonym expansion, or punctuation deletion.
    """
    return " ".join(name.strip().casefold().split())


def _clinical_drop_names() -> frozenset[str]:
    return frozenset(normalize_concept_name(name) for name in _CLINICAL_DROP_NAMES_BY_ID.values())


def _clinical_drop_mask(df: pd.DataFrame) -> pd.Series:
    ids = df["concept_id"].astype(str)
    names = df["concept_name"].astype(str).map(normalize_concept_name)

    expected_names = ids.map(
        lambda concept_id: normalize_concept_name(_CLINICAL_DROP_NAMES_BY_ID.get(concept_id, ""))
    )
    legacy_id_and_name_match = (
        ids.isin(CLINICAL_DROP_IDS) & expected_names.ne("") & names.eq(expected_names)
    )
    return legacy_id_and_name_match | names.isin(_clinical_drop_names())


def compute_dedup_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic same-tier duplicate mappings.

    Groups concepts by ``(tier, normalized concept_name)``. For every group with
    more than one concept, reference provenance is kept before harvested
    provenance. If provenance is tied, the lowest numeric concept ID wins.

    Returns:
        A dataframe with one row per dropped duplicate concept and columns:
        duplicate_concept_id, canonical_concept_id, tier, concept_name,
        duplicate_provenance, canonical_provenance, and reason.
    """
    if df.empty:
        return pd.DataFrame(columns=_DEDUP_MAP_COLUMNS)

    working = df.copy()
    working["_normalized_concept_name"] = working["concept_name"].map(normalize_concept_name)
    rows: list[dict[str, str]] = []

    grouped = working.groupby(["tier", "_normalized_concept_name"], sort=True, dropna=False)
    for (_, _), group in grouped:
        if len(group) <= 1:
            continue

        group = group.copy()
        group["_provenance_rank"] = group["provenance"].map(
            lambda value: 0 if value == "reference" else 1
        )
        group["_concept_id_numeric"] = group["concept_id"].map(concept_id_numeric_part)
        group = group.sort_values(
            by=["_provenance_rank", "_concept_id_numeric", "concept_id"],
            kind="mergesort",
        )

        canonical = group.iloc[0]
        duplicates = group.iloc[1:]
        for _, duplicate in duplicates.iterrows():
            rows.append(
                {
                    "duplicate_concept_id": str(duplicate["concept_id"]),
                    "canonical_concept_id": str(canonical["concept_id"]),
                    "tier": str(duplicate["tier"]),
                    "concept_name": str(duplicate["concept_name"]),
                    "duplicate_provenance": str(duplicate["provenance"]),
                    "canonical_provenance": str(canonical["provenance"]),
                    "reason": "same_tier_normalized_name_duplicate",
                }
            )

    return pd.DataFrame(rows, columns=_DEDUP_MAP_COLUMNS)


def compute_dedup_drops(df: pd.DataFrame) -> frozenset[str]:
    """Return concept IDs to drop for duplicate-name resolution."""
    dedup_map = compute_dedup_mapping(df)
    return frozenset(cast("list[str]", dedup_map["duplicate_concept_id"].tolist()))


def _read_view_a_concepts(concepts_path: Path) -> pd.DataFrame:
    concepts = pd.read_parquet(concepts_path)
    if "in_primary_lexicon" in concepts.columns:
        concepts = concepts[concepts["in_primary_lexicon"]].copy()
    else:
        concepts = concepts.copy()
    return concepts


def _validate_ready_prompts(ready_prompts: pd.DataFrame) -> None:
    if list(ready_prompts.columns) != _READY_PROMPT_COLUMNS:
        raise ValueError(f"Ready prompts schema mismatch: {list(ready_prompts.columns)!r}")
    if ready_prompts["prompt_text"].isna().any():
        raise ValueError("Ready prompts contain null prompt_text values")
    if ready_prompts["prompt_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Ready prompts contain empty prompt_text values")
    if ready_prompts.duplicated(subset=["concept_id", "species"]).any():
        raise ValueError("Ready prompts contain duplicate (concept_id, species) rows")
    if (
        ready_prompts["concept_name"]
        .astype(str)
        .map(normalize_concept_name)
        .isin(_clinical_drop_names())
        .any()
    ):
        raise ValueError("Ready prompts contain clinical drop names")
    if not set(ready_prompts["embedding_set"].unique()).issubset({"visual", "transcriptomic"}):
        raise ValueError("Ready prompts contain unknown embedding_set values")
    human_only_mouse = ready_prompts["species"].eq("Mus_musculus") & ready_prompts[
        "species_status"
    ].eq("human_only")
    if human_only_mouse.any():
        raise ValueError("Ready prompts contain human_only mouse rows")


def prepare_view_a(
    concepts_path: Path,
    prompts_path: Path,
    output_concepts_path: Path,
    output_prompts_path: Path,
    ready_prompts_path: Path,
    dedup_map_path: Path,
    dry_run: bool = False,
) -> ViewPrepResult:
    """Prepare View A for VLM text embedding.

    Args:
        concepts_path: Merged concepts parquet path.
        prompts_path: Merged prompts parquet path.
        output_concepts_path: Destination for cleaned concepts with embedding_set.
        output_prompts_path: Destination for cleaned prompts with embedding_set.
        ready_prompts_path: Destination for flat prompts ready for VLM encoding.
        dedup_map_path: Destination for duplicate-to-canonical mapping.
        dry_run: When True, computes counts without writing files.

    Returns:
        Summary counts for the preparation run.
    """
    concepts = _read_view_a_concepts(concepts_path)
    prompts = pd.read_parquet(prompts_path)
    view_a_ids = set(cast("list[str]", concepts["concept_id"].tolist()))
    prompts = prompts[prompts["concept_id"].isin(view_a_ids)].copy()

    n_concepts_before = len(concepts)
    n_prompts_before = len(prompts)

    clinical_mask = _clinical_drop_mask(concepts)
    n_clinical_dropped = int(clinical_mask.sum())
    concepts = concepts[~clinical_mask].copy()

    dedup_map = compute_dedup_mapping(concepts)
    dedup_drop_ids = frozenset(cast("list[str]", dedup_map["duplicate_concept_id"].tolist()))
    concepts = concepts[~concepts["concept_id"].isin(dedup_drop_ids)].copy()

    concepts["embedding_set"] = concepts["tier"].map(assign_embedding_set)
    embedding_by_concept = dict(
        zip(
            cast("list[str]", concepts["concept_id"].tolist()),
            cast("list[str]", concepts["embedding_set"].tolist()),
            strict=True,
        )
    )

    cleaned_ids = set(cast("list[str]", concepts["concept_id"].tolist()))
    cleaned_prompts = prompts[prompts["concept_id"].isin(cleaned_ids)].copy()
    cleaned_prompts["embedding_set"] = cleaned_prompts["concept_id"].map(embedding_by_concept)

    human_only_mouse = cleaned_prompts["species"].eq("Mus_musculus") & cleaned_prompts[
        "species_status"
    ].eq("human_only")
    human_only_rows = cleaned_prompts["species_status"].eq("human_only")

    ready_prompts = cleaned_prompts.loc[~human_only_mouse, _READY_PROMPT_COLUMNS].copy()
    _validate_ready_prompts(ready_prompts)

    result = ViewPrepResult(
        n_concepts_before=n_concepts_before,
        n_clinical_dropped=n_clinical_dropped,
        n_dedup_dropped=len(dedup_drop_ids),
        n_concepts_after=len(concepts),
        n_visual_concepts=int(concepts["embedding_set"].eq("visual").sum()),
        n_transcriptomic_concepts=int(concepts["embedding_set"].eq("transcriptomic").sum()),
        n_prompts_before=n_prompts_before,
        n_prompts_after=len(ready_prompts),
        n_human_only_prompt_rows_before=int(human_only_rows.sum()),
        n_human_only_mouse_rows_excluded=int(human_only_mouse.sum()),
    )

    if dry_run:
        return result

    for path in [output_concepts_path, output_prompts_path, ready_prompts_path, dedup_map_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    concepts.to_parquet(output_concepts_path, index=False)
    cleaned_prompts.to_parquet(output_prompts_path, index=False)
    ready_prompts.to_parquet(ready_prompts_path, index=False)
    dedup_map.to_parquet(dedup_map_path, index=False)

    return result
