"""RawConcept — the lowest-level harvested term record, pre-curation."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, field_validator

from molf_interp.io.config import BaseConfig

_RAW_CONCEPTS_SCHEMA = pa.schema(
    [
        pa.field("source_name", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("preferred_label", pa.string(), nullable=False),
        pa.field("synonyms", pa.list_(pa.string()), nullable=False),
        pa.field("definition", pa.string(), nullable=True),
        pa.field("parent_ids", pa.list_(pa.string()), nullable=False),
        pa.field("candidate_tier", pa.string(), nullable=False),
        pa.field("extra", pa.map_(pa.string(), pa.string()), nullable=False),
    ]
)


class CandidateTier(StrEnum):
    """Hint from a harvester about which reference tier this term likely belongs to.

    The curation stage makes the final call and may override.
    """

    H1_MORPHOLOGY = "H1_morphology"
    H2_CELL_TYPE = "H2_cell_type"
    H2_NICHE = "H2_niche"
    H2_GENE_PROGRAM = "H2_gene_program"
    H2_PATHWAY = "H2_pathway"
    UNKNOWN = "unknown"
    ORGAN_VOCAB = "organ_vocab"


class RawConcept(BaseConfig):
    """A single harvested term, pre-curation, pre-tiering, pre-prompt.

    Source-agnostic: works for ontology classes (BioPortal), gene sets (MSigDB),
    NER entities (PubMed/scispaCy), or any future source.
    """

    source_name: str
    source_id: str
    preferred_label: str
    synonyms: tuple[str, ...] = ()
    definition: str | None = None
    parent_ids: tuple[str, ...] = ()
    candidate_tier: CandidateTier
    extra: dict[str, str] = Field(default_factory=dict)

    @field_validator("preferred_label")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        """Validate preferred_label is non-empty after stripping whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("preferred_label must be non-empty after stripping")
        return v


def _extra_to_dict(raw: object) -> dict[str, str]:
    """Convert a pyarrow map column value back to a plain dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    # PyArrow map type returns a list of (key, value) tuples or {"key": k, "value": v} dicts.
    result: dict[str, str] = {}
    for item in raw:  # type: ignore[attr-defined]
        if isinstance(item, dict):
            result[str(item["key"])] = str(item["value"])
        else:
            result[str(item[0])] = str(item[1])
    return result


def write_raw_concepts(records: Sequence[RawConcept], path: Path) -> None:
    """Write RawConcepts to a single snappy-compressed parquet file.

    Args:
        records: Concepts to write. May be empty (creates a valid empty file).
        path: Destination file path (not a directory).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "source_name": pa.array([c.source_name for c in records], type=pa.string()),
            "source_id": pa.array([c.source_id for c in records], type=pa.string()),
            "preferred_label": pa.array([c.preferred_label for c in records], type=pa.string()),
            "synonyms": pa.array([list(c.synonyms) for c in records], type=pa.list_(pa.string())),
            "definition": pa.array([c.definition for c in records], type=pa.string()),
            "parent_ids": pa.array(
                [list(c.parent_ids) for c in records], type=pa.list_(pa.string())
            ),
            "candidate_tier": pa.array([c.candidate_tier.value for c in records], type=pa.string()),
            "extra": pa.array(
                [list(c.extra.items()) for c in records],
                type=pa.map_(pa.string(), pa.string()),
            ),
        },
        schema=_RAW_CONCEPTS_SCHEMA,
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        table, str(path), compression="snappy"
    )


def read_raw_concepts(path: Path) -> tuple[RawConcept, ...]:
    """Read RawConcepts from a parquet file.

    Args:
        path: Parquet file path.

    Returns:
        Immutable tuple of RawConcept instances.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw concepts file not found: {path}")
    table = pq.read_table(  # type: ignore[no-untyped-call]
        str(path), schema=_RAW_CONCEPTS_SCHEMA
    )
    d = table.to_pydict()
    n = len(d["source_name"])
    return tuple(
        RawConcept(
            source_name=d["source_name"][i],
            source_id=d["source_id"][i],
            preferred_label=d["preferred_label"][i],
            synonyms=tuple(d["synonyms"][i] or []),
            definition=d["definition"][i],
            parent_ids=tuple(d["parent_ids"][i] or []),
            candidate_tier=CandidateTier(d["candidate_tier"][i]),
            extra=_extra_to_dict(d["extra"][i]),
        )
        for i in range(n)
    )


def merge_raw_concept_files(paths: Sequence[Path], output_path: Path) -> int:
    """Concatenate multiple raw-concept parquet files into one.

    Does NOT dedup — curation handles deduplication downstream.

    Args:
        paths: Source parquet file paths. Non-existent paths are silently skipped.
        output_path: Destination parquet file path.

    Returns:
        Total row count written.
    """
    tables = [
        pq.read_table(str(p), schema=_RAW_CONCEPTS_SCHEMA)  # type: ignore[no-untyped-call]
        for p in paths
        if p.exists()
    ]
    if not tables:
        merged = pa.table(
            {field.name: pa.array([], type=field.type) for field in _RAW_CONCEPTS_SCHEMA},
            schema=_RAW_CONCEPTS_SCHEMA,
        )
    else:
        merged = pa.concat_tables(tables)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        merged, str(output_path), compression="snappy"
    )
    return int(merged.num_rows)
