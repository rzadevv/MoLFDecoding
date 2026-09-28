"""Abstract base class for pathology VLM wrappers and shared configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torchvision import transforms

from molf_interp.io.config import BaseConfig


class VLMConfig(BaseConfig):
    """Shared configuration fields for any VLM backend.

    Subclasses may extend this with model-specific fields.
    """

    model_name: str
    device: str = "auto"
    hf_token_env_var: str = "HF_TOKEN"
    cache_dir: Path | None = None
    arch: str | None = None
    checkpoint: str | None = None
    embedding_dim: int = 512


class PathologyVLM(ABC):
    """Abstract base class for vision-language pathology models.

    Subclasses must implement ``_load``, ``_encode_image_batch``, and
    ``_encode_text_batch``. All public ``encode_*`` methods wrap those
    protected methods with device management, batching, and inference mode.
    """

    config: VLMConfig
    _device: torch.device

    @property
    @abstractmethod
    def preprocess(self) -> transforms.Compose:
        """Return the preprocessing transform pipeline for images."""

    # ------------------------------------------------------------------
    # Protected interface — implement in subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _load(self) -> None:
        """Load model weights and tokenizer into ``self.device``."""

    @abstractmethod
    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of preprocessed image tensors.

        Args:
            images: Tensor of shape (B, 3, H, W), already on ``self.device``.

        Returns:
            Tensor of shape (B, D) — one D-dimensional embedding per image.
        """

    @abstractmethod
    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text strings.

        Args:
            texts: List of B text strings.

        Returns:
            Tensor of shape (B, D) — one D-dimensional embedding per text.
        """

    # ------------------------------------------------------------------
    # Public interface — uses protected methods + batching / inference mode
    # ------------------------------------------------------------------

    def encode_images(self, images: list[torch.Tensor], batch_size: int = 8) -> torch.Tensor:
        """Encode a list of image tensors with batching and inference mode.

        Args:
            images: List of tensors with shape (3, H, W).
            batch_size: Number of images per encode call.

        Returns:
            Stacked tensor (N, D).
        """
        with torch.inference_mode():
            device = self.device
            embeddings: list[torch.Tensor] = []
            for i in range(0, len(images), batch_size):
                batch = [images[j] for j in range(i, min(i + batch_size, len(images)))]
                batch_t = torch.stack(batch, dim=0).to(device)
                embeddings.append(self._encode_image_batch(batch_t).cpu())
            return torch.cat(embeddings, dim=0)

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode a list of text strings with batching and inference mode.

        Args:
            texts: List of text strings.
            batch_size: Number of texts per encode call.

        Returns:
            Stacked tensor (N, D).
        """
        with torch.inference_mode():
            device = self.device
            embeddings: list[torch.Tensor] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                emb = self._encode_text_batch(batch).to(device)
                embeddings.append(emb.cpu())
            return torch.cat(embeddings, dim=0)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return the torch device used by this model."""
        return self._device

    def _init_device(self) -> torch.device:
        """Resolve device from config string (auto → cuda/mps/cpu)."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)
