"""Parquet I/O for concept embedding artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


@dataclass(frozen=True, slots=True)
class EmbeddingArtifact:
    """A single concept embedding stored as a frozen record."""

    concept_id: str
    species: str
    tier: str
    view: str
    prompt_text: str
    embedding: torch.Tensor  # shape (D,)


def _serialize_tensor(t: torch.Tensor) -> list[float]:
    return t.flatten().tolist()


def _deserialize_tensor(data: list[float]) -> torch.Tensor:
    return torch.tensor(data)


# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------


def write_embeddings(artifacts: list[EmbeddingArtifact], out_path: Path) -> None:
    """Write a list of embedding artifacts to a partitioned Parquet dataset.

    Schema: concept_id, species, tier, view, prompt_text, embedding (list[float])
    A JSON sidecar at ``out_path.with_suffix(".json")`` is also written with
    global metadata (D, n_rows, created_at).
    """
    import json
    from datetime import UTC, datetime

    rows = []
    dim = None
    for art in artifacts:
        rows.append(
            {
                "concept_id": art.concept_id,
                "species": art.species,
                "tier": art.tier,
                "view": art.view,
                "prompt_text": art.prompt_text,
                "embedding": _serialize_tensor(art.embedding),
            }
        )
        if dim is None:
            dim = art.embedding.shape[0]

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)  # type: ignore[no-untyped-call]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")  # type: ignore[no-untyped-call]

    meta = {
        "D": dim,
        "n_rows": len(rows),
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------


def read_embeddings(in_path: Path) -> list[EmbeddingArtifact]:
    """Read a Parquet embedding file back into EmbeddingArtifact objects."""
    table = pq.read_table(in_path)  # type: ignore[no-untyped-call]
    df = table.to_pandas()

    artifacts: list[EmbeddingArtifact] = []
    for _, row in df.iterrows():
        artifacts.append(
            EmbeddingArtifact(
                concept_id=row["concept_id"],
                species=row["species"],
                tier=row["tier"],
                view=row["view"],
                prompt_text=row["prompt_text"],
                embedding=_deserialize_tensor(row["embedding"].tolist()),
            )
        )
    return artifacts


def stack_embeddings(artifacts: list[EmbeddingArtifact]) -> tuple[torch.Tensor, list[str]]:
    """Stack embeddings into a single (N, D) tensor, returning concept_ids in order."""
    tensors = [art.embedding for art in artifacts]
    concept_ids = [art.concept_id for art in artifacts]
    return torch.stack(tensors, dim=0), concept_ids
