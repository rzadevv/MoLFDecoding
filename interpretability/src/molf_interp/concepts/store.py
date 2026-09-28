"""Producer-agnostic parquet read/write for ConceptBank."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from molf_interp.concepts.schemas import (
    Concept,
    ConceptBank,
    ConceptPrompt,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)
from molf_interp.concepts.validation import validate_consistency

_CONCEPTS_SCHEMA = pa.schema(
    [
        pa.field("concept_id", pa.string(), nullable=False),
        pa.field("concept_name", pa.string(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("organ", pa.string(), nullable=True),
        pa.field("level", pa.string(), nullable=True),
        pa.field("concept_type", pa.string(), nullable=True),
    ]
)

_PROMPTS_SCHEMA = pa.schema(
    [
        pa.field("concept_id", pa.string(), nullable=False),
        pa.field("concept_name", pa.string(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("species", pa.string(), nullable=False),
        pa.field("prompt_text", pa.string(), nullable=False),
        pa.field("species_status", pa.string(), nullable=False),
    ]
)

_CONCEPTS_PATH = "concepts.parquet"
_PROMPTS_PATH = "prompts.parquet"

_PROVENANCE_MIGRATION: dict[str, str] = {
    "su_pathologist": "reference",
    "automated_harvest": "harvested",
}


def migrate_provenance_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy provenance strings to current enum values.

    Maps ``"su_pathologist"`` → ``"reference"`` and
    ``"automated_harvest"`` → ``"harvested"``. Current values pass through
    unchanged. Raises ``ValueError`` for unrecognised values.

    Legacy compatibility: parquets written before the enum rename carry the old
    values. Call this on any DataFrame read from an existing concepts.parquet
    before constructing ``Concept`` objects.
    """
    if "provenance" not in df.columns:
        return df
    known = set(_PROVENANCE_MIGRATION) | {"reference", "harvested"}
    unknown = set(df["provenance"].unique()) - known
    if unknown:
        raise ValueError(f"Unrecognised provenance values: {unknown}")
    df = df.copy()
    df["provenance"] = df["provenance"].map(lambda v: _PROVENANCE_MIGRATION.get(v, v))
    return df


def write_concept_bank(bank: ConceptBank, output_dir: Path) -> None:
    """Write concepts.parquet and prompts.parquet under output_dir.

    Args:
        bank: The ConceptBank to persist.
        output_dir: Directory to write parquet files into (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts_table = pa.table(
        {
            "concept_id": [c.concept_id for c in bank.concepts],
            "concept_name": [c.concept_name for c in bank.concepts],
            "tier": [c.tier.value for c in bank.concepts],
            "category": [c.category for c in bank.concepts],
            "subcategory": [c.subcategory for c in bank.concepts],
            "source": [c.source for c in bank.concepts],
            "provenance": [c.provenance.value for c in bank.concepts],
            "organ": [c.organ for c in bank.concepts],
            "level": [c.level for c in bank.concepts],
            "concept_type": [c.concept_type for c in bank.concepts],
        },
        schema=_CONCEPTS_SCHEMA,
    )
    pq.write_table(  # type: ignore[no-untyped-call]  # pyarrow stubs incomplete
        concepts_table, output_dir / _CONCEPTS_PATH, compression="snappy"
    )

    prompts_table = pa.table(
        {
            "concept_id": [p.concept_id for p in bank.prompts],
            "concept_name": [p.concept_name for p in bank.prompts],
            "tier": [p.tier.value for p in bank.prompts],
            "species": [p.species.value for p in bank.prompts],
            "prompt_text": [p.prompt_text for p in bank.prompts],
            "species_status": [p.species_status.value for p in bank.prompts],
        },
        schema=_PROMPTS_SCHEMA,
    )
    pq.write_table(  # type: ignore[no-untyped-call]  # pyarrow stubs incomplete
        prompts_table, output_dir / _PROMPTS_PATH, compression="snappy"
    )


def read_concept_bank(input_dir: Path) -> ConceptBank:
    """Read concepts.parquet and prompts.parquet from input_dir.

    Args:
        input_dir: Directory containing concepts.parquet and prompts.parquet.

    Returns:
        A fully validated ConceptBank.

    Raises:
        FileNotFoundError: if either parquet file is missing.
    """
    concepts_path = input_dir / _CONCEPTS_PATH
    prompts_path = input_dir / _PROMPTS_PATH
    if not concepts_path.exists():
        raise FileNotFoundError(f"concepts.parquet not found at {concepts_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.parquet not found at {prompts_path}")

    ct = pq.read_table(concepts_path)  # type: ignore[no-untyped-call]  # pyarrow stubs incomplete
    cd = ct.to_pydict()
    cd["provenance"] = [_PROVENANCE_MIGRATION.get(v, v) for v in cd["provenance"]]
    n = len(cd["concept_id"])
    concepts = tuple(
        Concept(
            concept_id=cd["concept_id"][i],
            concept_name=cd["concept_name"][i],
            tier=Tier(cd["tier"][i]),
            category=cd["category"][i],
            subcategory=cd["subcategory"][i],
            source=cd["source"][i],
            provenance=ProvenanceSource(cd["provenance"][i]),
            organ=cd["organ"][i],
            level=cd["level"][i],
            concept_type=cd["concept_type"][i],
        )
        for i in range(n)
    )

    pt = pq.read_table(prompts_path)  # type: ignore[no-untyped-call]  # pyarrow stubs incomplete
    pd_ = pt.to_pydict()
    m = len(pd_["concept_id"])
    prompts = tuple(
        ConceptPrompt(
            concept_id=pd_["concept_id"][i],
            concept_name=pd_["concept_name"][i],
            tier=Tier(pd_["tier"][i]),
            species=Species(pd_["species"][i]),
            prompt_text=pd_["prompt_text"][i],
            species_status=SpeciesStatus(pd_["species_status"][i]),
        )
        for i in range(m)
    )

    provenances = {c.provenance for c in concepts}
    provenance = next(iter(provenances)) if len(provenances) == 1 else ProvenanceSource.REFERENCE

    validate_consistency(concepts, prompts)
    return ConceptBank(concepts=concepts, prompts=prompts, provenance=provenance)
