"""Ingester for the three-TSV reference concept bank format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from molf_interp.concepts.schemas import (
    _H2_CONCEPT_TYPE_BY_TIER,
    Concept,
    ConceptBank,
    ConceptPrompt,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)
from molf_interp.concepts.validation import validate_consistency

_MORPHOLOGY_FILE = "concept_bank_morphology.tsv"
_TRANSCRIPTOMICS_FILE = "concept_bank_transcriptomics.tsv"
_CROSS_SPECIES_FILE = "concept_bank_cross_species.tsv"

_EXPECTED_MORPHOLOGY_ROWS = 418
_EXPECTED_TRANSCRIPTOMICS_ROWS = 588
_EXPECTED_CROSS_SPECIES_ROWS = 2012

# Inverse of _H2_CONCEPT_TYPE_BY_TIER: "cell_type" -> Tier.H2_CELL_TYPE
_TIER_BY_CONCEPT_TYPE: dict[str, Tier] = {v: k for k, v in _H2_CONCEPT_TYPE_BY_TIER.items()}


def _parse_morphology_row(row: pd.Series) -> Concept:
    """Parse one row from the morphology TSV into a Concept.

    Args:
        row: A pandas Series with morphology TSV columns.

    Returns:
        A validated Concept at tier H1_MORPHOLOGY.
    """
    return Concept(
        concept_id=row["concept_id"],
        concept_name=row["concept_name"],
        tier=Tier.H1_MORPHOLOGY,
        category=row["category"],
        subcategory=row["subcategory"],
        source=row["source"],
        provenance=ProvenanceSource.REFERENCE,
        organ=row["organ"],
        level=row["level"],
        concept_type=None,
    )


def _parse_transcriptomics_row(row: pd.Series) -> Concept:
    """Parse one row from the transcriptomics TSV into a Concept.

    Args:
        row: A pandas Series with transcriptomics TSV columns.

    Returns:
        A validated Concept at the appropriate H2 tier.
    """
    concept_type = row["concept_type"]
    tier = _TIER_BY_CONCEPT_TYPE[concept_type]
    return Concept(
        concept_id=row["concept_id"],
        concept_name=row["concept_name"],
        tier=tier,
        category=row["category"],
        subcategory=row["subcategory"],
        source=row["source"],
        provenance=ProvenanceSource.REFERENCE,
        organ=None,
        level=None,
        concept_type=concept_type,
    )


def _parse_cross_species_row(row: pd.Series) -> ConceptPrompt:
    """Parse one row from the cross-species TSV into a ConceptPrompt.

    Args:
        row: A pandas Series with cross-species TSV columns.

    Returns:
        A validated ConceptPrompt.
    """
    return ConceptPrompt(
        concept_id=row["concept_id"],
        concept_name=row["concept_name"],
        tier=Tier(row["hierarchy"]),
        species=Species(row["species"]),
        prompt_text=row["prompt_text"],
        species_status=SpeciesStatus(row["species_status"]),
    )


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).apply(
        lambda col: col.str.strip() if col.dtype == object else col
    )


def load_reference_bank(input_dir: Path) -> ConceptBank:
    """Load the three-TSV reference concept bank from input_dir.

    Args:
        input_dir: Directory containing the three reference TSV files.

    Returns:
        A validated ConceptBank with provenance=REFERENCE.

    Raises:
        FileNotFoundError: if any of the three TSVs is missing.
        pydantic.ValidationError: if any row fails schema validation.
        ConsistencyError: if cross-table checks fail.
    """
    for fname in (_MORPHOLOGY_FILE, _TRANSCRIPTOMICS_FILE, _CROSS_SPECIES_FILE):
        p = input_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Required reference TSV not found: {p}")

    morph_df = _read_tsv(input_dir / _MORPHOLOGY_FILE)
    trans_df = _read_tsv(input_dir / _TRANSCRIPTOMICS_FILE)
    cross_df = _read_tsv(input_dir / _CROSS_SPECIES_FILE)

    logger.info(
        "Reference TSV row counts: morphology={}, transcriptomics={}, cross_species={}",
        len(morph_df),
        len(trans_df),
        len(cross_df),
    )
    _warn_if_unexpected(len(morph_df), _EXPECTED_MORPHOLOGY_ROWS, _MORPHOLOGY_FILE)
    _warn_if_unexpected(len(trans_df), _EXPECTED_TRANSCRIPTOMICS_ROWS, _TRANSCRIPTOMICS_FILE)
    _warn_if_unexpected(len(cross_df), _EXPECTED_CROSS_SPECIES_ROWS, _CROSS_SPECIES_FILE)

    concepts: list[Concept] = [_parse_morphology_row(row) for _, row in morph_df.iterrows()]
    concepts += [_parse_transcriptomics_row(row) for _, row in trans_df.iterrows()]

    prompts: list[ConceptPrompt] = [_parse_cross_species_row(row) for _, row in cross_df.iterrows()]

    tier_counts: dict[str, int] = {}
    for c in concepts:
        tier_counts[c.tier.value] = tier_counts.get(c.tier.value, 0) + 1
    logger.info("Concepts per tier: {}", tier_counts)

    human_only = sum(
        1
        for p in prompts
        if p.species_status == SpeciesStatus.HUMAN_ONLY and p.species == Species.HOMO_SAPIENS
    )
    logger.info("Total prompts: {}, human_only concepts: {}", len(prompts), human_only)

    validate_consistency(concepts, prompts)
    return ConceptBank(
        concepts=tuple(concepts),
        prompts=tuple(prompts),
        provenance=ProvenanceSource.REFERENCE,
    )


def _warn_if_unexpected(actual: int, expected: int, fname: str) -> None:
    if actual != expected:
        logger.warning(
            "{} has {} rows (expected {}); bank may have grown legitimately",
            fname,
            actual,
            expected,
        )
