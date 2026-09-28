"""Tests for SapBert wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from molf_interp.curation.config import SapBertConfig
from molf_interp.curation.sapbert import SapBert


def _make_config(tmp_path: Path) -> SapBertConfig:
    return SapBertConfig(
        model_name="fake-model",
        cache_dir=tmp_path / "sapbert_cache",
        batch_size=4,
    )


def _make_fake_model(embedding_dim: int = 768) -> tuple[MagicMock, MagicMock]:
    """Return (tokenizer_mock, model_mock) that produce deterministic embeddings."""
    import torch

    tokenizer = MagicMock()

    def _encode_plus(texts: list[str], **kwargs: object) -> dict[str, object]:
        n = len(texts)
        max_length = int(kwargs.get("max_length", 25))
        return {
            "input_ids": torch.zeros(n, max_length, dtype=torch.long),
            "attention_mask": torch.ones(n, max_length, dtype=torch.long),
        }

    tokenizer.batch_encode_plus.side_effect = _encode_plus

    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    def _forward(**kwargs: object) -> list[object]:
        batch_size = kwargs["input_ids"].shape[0]  # type: ignore[union-attr]
        hidden = torch.randn(batch_size, 25, embedding_dim)
        return [hidden]

    model.side_effect = lambda **kw: _forward(**kw)
    return tokenizer, model


class TestSapBert:
    def test_load_idempotent(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        sb = SapBert(config)
        tokenizer, model = _make_fake_model()

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModel.from_pretrained", return_value=model),
        ):
            sb.load()
            sb.load()  # second call should be no-op

        assert model.to.call_count == 1

    def test_encode_shape_and_normalized(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        sb = SapBert(config)
        tokenizer, model = _make_fake_model()

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModel.from_pretrained", return_value=model),
        ):
            result = sb.encode(["necrosis", "fibrosis"])

        assert result.shape == (2, 768)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_cache_reuse(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        sb = SapBert(config)
        tokenizer, model = _make_fake_model()
        call_count = 0

        def _model_call(**kw: object) -> object:
            nonlocal call_count
            import torch

            batch_size = kw["input_ids"].shape[0]  # type: ignore[union-attr]
            call_count += 1
            return [torch.randn(batch_size, 25, 768)]

        model.side_effect = lambda **kw: _model_call(**kw)

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModel.from_pretrained", return_value=model),
        ):
            sb.encode(["necrosis", "fibrosis"])
            sb.encode(["necrosis", "fibrosis", "adenocarcinoma"])

        # Only "adenocarcinoma" should trigger a new model call
        # (necrosis and fibrosis are cached)
        assert call_count == 2  # initial 2 texts + 1 new text (batched separately)

    def test_cosine_similarity(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        sb = SapBert(config)
        # L2-normalized vectors
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = sb.cosine_similarity(a, b)
        assert result.shape == (1, 2)
        assert abs(result[0, 0] - 1.0) < 1e-5
        assert abs(result[0, 1] - 0.0) < 1e-5
