"""Pytest fixtures for VLM tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from molf_interp.vlm.base import VLMConfig


@pytest.fixture
def conch_v1_config(tmp_path: Path) -> VLMConfig:
    """Minimal VLMConfig pointing to a temp cache dir."""
    return VLMConfig(
        model_name="conch_v1",
        device="cpu",
        hf_token_env_var="HF_TOKEN",
        cache_dir=tmp_path,
    )


@pytest.fixture
def fake_image() -> Image.Image:
    """A 224x224 RGB image."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def fake_image_tensor(fake_image: Image.Image) -> torch.Tensor:
    """A preprocessed image tensor (3, 224, 224), mean/std normalised."""
    from torchvision import transforms

    t = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return t(fake_image)


@pytest.fixture
def stub_device() -> torch.device:
    """Force CPU device for unit tests."""
    return torch.device("cpu")
