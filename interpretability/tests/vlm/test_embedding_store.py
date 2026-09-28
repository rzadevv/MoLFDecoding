"""Tests for the embedding store module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from molf_interp.vlm.embedding_store import (
    EmbeddingArtifact,
    read_embeddings,
    stack_embeddings,
    write_embeddings,
)


class TestEmbeddingArtifact:
    def test_frozen_and_slotted(self) -> None:
        art = EmbeddingArtifact(
            concept_id="MOR-0001",
            species="Homo_sapiens",
            tier="H1_morphology",
            view="A",
            prompt_text="Test prompt",
            embedding=torch.zeros(512),
        )
        with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
            art.concept_id = "X"  # type: ignore[annotation-type-mismatch]

    def test_shape_preserved(self) -> None:
        emb = torch.randn(128)
        art = EmbeddingArtifact(
            concept_id="CTY-0001",
            species="Mus_musculus",
            tier="H2_cell_type",
            view="A",
            prompt_text="A cell type",
            embedding=emb,
        )
        assert art.embedding.shape == (128,)


class TestWriteReadRoundTrip:
    def test_round_trip_single_artifact(self, tmp_path: Path) -> None:
        emb = torch.randn(512)
        artifacts = [
            EmbeddingArtifact(
                concept_id="MOR-0001",
                species="Homo_sapiens",
                tier="H1_morphology",
                view="A",
                prompt_text="Fibrosis with collagen deposition.",
                embedding=emb,
            )
        ]
        out = tmp_path / "embeddings.parquet"
        write_embeddings(artifacts, out)
        assert out.exists()
        assert out.with_suffix(".json").exists()

        read_back = read_embeddings(out)
        assert len(read_back) == 1
        assert read_back[0].concept_id == "MOR-0001"
        assert torch.allclose(read_back[0].embedding, emb)

    def test_round_trip_multiple_different_tiers(self, tmp_path: Path) -> None:
        artifacts = [
            EmbeddingArtifact(
                concept_id="MOR-0001",
                species="Homo_sapiens",
                tier="H1_morphology",
                view="A",
                prompt_text="Morphology prompt",
                embedding=torch.randn(512),
            ),
            EmbeddingArtifact(
                concept_id="CTY-0001",
                species="Mus_musculus",
                tier="H2_cell_type",
                view="A",
                prompt_text="Cell type prompt",
                embedding=torch.randn(512),
            ),
            EmbeddingArtifact(
                concept_id="PWY-0001",
                species="Homo_sapiens",
                tier="H2_pathway",
                view="A",
                prompt_text="Pathway prompt",
                embedding=torch.randn(512),
            ),
        ]
        out = tmp_path / "multi_tier.parquet"
        write_embeddings(artifacts, out)
        read_back = read_embeddings(out)
        assert len(read_back) == 3
        assert {a.concept_id for a in read_back} == {"MOR-0001", "CTY-0001", "PWY-0001"}

    def test_sidecar_json_metadata(self, tmp_path: Path) -> None:
        artifacts = [
            EmbeddingArtifact(
                concept_id="MOR-0001",
                species="Homo_sapiens",
                tier="H1_morphology",
                view="A",
                prompt_text="Test",
                embedding=torch.randn(512),
            )
            for _ in range(7)
        ]
        out = tmp_path / "sidecar_test.parquet"
        write_embeddings(artifacts, out)

        sidecar = out.with_suffix(".json")
        meta = json.loads(sidecar.read_text())
        assert meta["D"] == 512
        assert meta["n_rows"] == 7
        assert "created_at" in meta


class TestStackEmbeddings:
    def test_stacks_correctly(self) -> None:
        artifacts = [
            EmbeddingArtifact(
                concept_id=f"ID-{i:04d}",
                species="Homo_sapiens",
                tier="H1_morphology",
                view="A",
                prompt_text=f"Prompt {i}",
                embedding=torch.randn(128),
            )
            for i in range(10)
        ]
        tensor, ids = stack_embeddings(artifacts)
        assert tensor.shape == (10, 128)
        assert ids == [f"ID-{i:04d}" for i in range(10)]

    def test_empty_raises(self) -> None:
        with pytest.raises(RuntimeError):
            stack_embeddings([])
