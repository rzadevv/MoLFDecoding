"""Unit tests for ConchV1 that do not require a real model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision import transforms

from molf_interp.vlm.base import VLMConfig
from molf_interp.vlm.conch_v1 import ConchV1


class TestConchV1Preprocess:
    def test_preprocess_returns_correct_pipeline(self) -> None:
        cfg = VLMConfig(model_name="conch_v1", device="cpu")
        vlm = ConchV1(cfg)
        pp = vlm.preprocess
        assert isinstance(pp, transforms.Compose)

    def test_preprocess_idempotent(self) -> None:
        cfg = VLMConfig(model_name="conch_v1", device="cpu")
        vlm = ConchV1(cfg)
        pp1 = vlm.preprocess
        pp2 = vlm.preprocess
        assert pp1 is pp2


class TestConchV1ConfigValidation:
    def test_missing_token_raises(self, tmp_path: Path) -> None:
        cfg = VLMConfig(
            model_name="conch_v1",
            device="cpu",
            hf_token_env_var="HF_TOKEN_MISSING_xyz",
            cache_dir=tmp_path,
        )
        vlm = ConchV1(cfg)
        with pytest.raises(EnvironmentError, match="HF_TOKEN_MISSING_xyz"):
            vlm._load()

    def test_device_from_config(self) -> None:
        cfg = VLMConfig(model_name="conch_v1", device="cpu")
        vlm = ConchV1(cfg)
        assert vlm.device == torch.device("cpu")


class TestConchV1EncodeBeforeLoad:
    def test_encode_text_batch_raises_if_not_loaded(self) -> None:
        cfg = VLMConfig(model_name="conch_v1", device="cpu")
        vlm = ConchV1(cfg)
        with pytest.raises(RuntimeError, match="not loaded"):
            vlm._encode_text_batch(["test"])

    def test_encode_image_batch_raises_if_not_loaded(self) -> None:
        cfg = VLMConfig(model_name="conch_v1", device="cpu")
        vlm = ConchV1(cfg)
        with pytest.raises(RuntimeError, match="not loaded"):
            vlm._encode_image_batch(torch.zeros(1, 3, 224, 224))


class TestConchV1RegistryIntegration:
    def test_conch_v1_registered(self) -> None:
        from molf_interp.vlm.registry import _REGISTRY

        assert "conch_v1" in _REGISTRY
        assert _REGISTRY["conch_v1"] is ConchV1

    @patch.dict("os.environ", {"HF_TOKEN": "fake_token"}, clear=False)
    @patch("molf_interp.vlm.conch_v1.create_model_from_pretrained")
    @patch("molf_interp.vlm.conch_v1.get_tokenizer")
    @patch("molf_interp.vlm.conch_v1._tokenize")
    def test_load_wires_model_and_tokenizer(
        self,
        mock_tokenize: MagicMock,
        mock_get_tokenizer: MagicMock,
        mock_create: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        mock_model.encode_image.return_value = torch.zeros(1, 512)
        mock_model.encode_text.return_value = torch.zeros(1, 512)
        mock_create.return_value = (mock_model, None)

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        mock_tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)

        cfg = VLMConfig(
            model_name="conch_v1",
            device="cpu",
            hf_token_env_var="HF_TOKEN",
            cache_dir=tmp_path,
        )
        vlm = ConchV1(cfg)
        vlm._load()

        assert vlm._model is mock_model
        assert vlm._tokenizer is mock_tokenizer
        mock_model.to.assert_called_once_with(torch.device("cpu"))
        mock_model.eval.assert_called_once()
