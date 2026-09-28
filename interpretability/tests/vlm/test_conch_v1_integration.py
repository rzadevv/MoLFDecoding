"""Integration tests for ConchV1 requiring a real model download."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molf_interp.vlm.base import VLMConfig
from molf_interp.vlm.conch_v1 import ConchV1

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_token(tmp_path: Path) -> None:
    import os

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        pytest.skip("HF_TOKEN not set in environment")


class TestConchV1RealModel:
    def test_load_returns_512_dim_text_embeddings(self, tmp_path: Path) -> None:
        cfg = VLMConfig(
            model_name="conch_v1",
            device="cpu",
            hf_token_env_var="HF_TOKEN",
            cache_dir=tmp_path,
        )
        vlm = ConchV1(cfg)
        vlm._load()

        texts = ["This is a test sentence."]
        emb = vlm._encode_text_batch(texts)
        assert emb.shape == (1, 512)
        assert emb.dtype == torch.float32

    def test_preprocess_is_idempotent(self, tmp_path: Path) -> None:
        from PIL import Image

        cfg = VLMConfig(model_name="conch_v1", device="cpu")
        vlm = ConchV1(cfg)
        pp = vlm.preprocess
        img = Image.new("RGB", (224, 224))
        t1 = pp(img)
        t2 = pp(img)
        assert t1.shape == t2.shape == (3, 224, 224)

    def test_encode_texts_round_trip(self, tmp_path: Path) -> None:
        cfg = VLMConfig(
            model_name="conch_v1",
            device="cpu",
            hf_token_env_var="HF_TOKEN",
            cache_dir=tmp_path,
        )
        vlm = ConchV1(cfg)
        vlm._load()

        texts = ["stromal tissue", "epithelial cells", "immune infiltration"]
        emb = vlm.encode_texts(texts)
        assert emb.shape == (3, 512)
        # Two identical texts should give near-identical embeddings
        emb2 = vlm.encode_texts(texts)
        cosine = torch.nn.functional.cosine_similarity(emb[0], emb2[0], dim=0)
        assert cosine > 0.99
