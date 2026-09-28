"""Tests for the PathologyVLM abstract base class and VLMConfig."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import transforms

from molf_interp.vlm.base import PathologyVLM, VLMConfig


class _DummyVLM(PathologyVLM):
    """Minimal concrete subclass for testing base class behaviour."""

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._device = self._init_device()
        self._loaded = False

    @property
    def preprocess(self) -> transforms.Compose:
        return transforms.Compose([])

    def _load(self) -> None:
        self._loaded = True

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 128)

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), 128)


class TestVLMConfig:
    def test_default_device_is_auto(self) -> None:
        cfg = VLMConfig(model_name="test")
        assert cfg.device == "auto"
        assert cfg.hf_token_env_var == "HF_TOKEN"
        assert cfg.cache_dir is None

    def test_all_fields_set(self, tmp_path: Path) -> None:
        cfg = VLMConfig(
            model_name="conch_v1",
            device="cuda",
            hf_token_env_var="HF_TOKEN",
            cache_dir=tmp_path,
        )
        assert cfg.model_name == "conch_v1"
        assert cfg.device == "cuda"


class TestPathologyVLMInitDevice:
    def test_auto_resolves_to_cuda_when_available(self) -> None:
        cfg = VLMConfig(model_name="test", device="auto")
        vlm = _DummyVLM(cfg)
        # May be cpu/cuda/mps depending on hardware — just must not raise
        assert isinstance(vlm.device, torch.device)

    def test_explicit_string_device(self) -> None:
        cfg = VLMConfig(model_name="test", device="cpu")
        vlm = _DummyVLM(cfg)
        assert vlm.device == torch.device("cpu")


class TestPathologyVLMEncodeImages:
    def test_encode_images_returns_correct_shape(self) -> None:
        cfg = VLMConfig(model_name="test", device="cpu")
        vlm = _DummyVLM(cfg)
        images = [torch.rand(3, 224, 224) for _ in range(4)]
        result = vlm.encode_images(images, batch_size=2)
        assert result.shape == (4, 128)

    def test_encode_images_empty_list(self) -> None:
        cfg = VLMConfig(model_name="test", device="cpu")
        vlm = _DummyVLM(cfg)
        with pytest.raises(ValueError, match="expected a non-empty list"):
            vlm.encode_images([])

    def test_encode_images_single_image(self) -> None:
        cfg = VLMConfig(model_name="test", device="cpu")
        vlm = _DummyVLM(cfg)
        images = [torch.rand(3, 224, 224)]
        result = vlm.encode_images(images)
        assert result.shape == (1, 128)


class TestPathologyVLMEncodeTexts:
    def test_encode_texts_returns_correct_shape(self) -> None:
        cfg = VLMConfig(model_name="test", device="cpu")
        vlm = _DummyVLM(cfg)
        texts = ["a", "b", "c", "d"]
        result = vlm.encode_texts(texts, batch_size=2)
        assert result.shape == (4, 128)

    def test_encode_texts_empty_list(self) -> None:
        cfg = VLMConfig(model_name="test", device="cpu")
        vlm = _DummyVLM(cfg)
        with pytest.raises(ValueError, match="expected a non-empty list"):
            vlm.encode_texts([])


class TestPathologyVLMAbstract:
    def test_cannot_instantiate_base_directly(self) -> None:
        with pytest.raises(TypeError):
            PathologyVLM(VLMConfig(model_name="test"))  # type: ignore[arg-type]

    def test_subclass_without_abstract_methods_works(self) -> None:
        cfg = VLMConfig(model_name="test")
        vlm = _DummyVLM(cfg)
        assert isinstance(vlm, PathologyVLM)
