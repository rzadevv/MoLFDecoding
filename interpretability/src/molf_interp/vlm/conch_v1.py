"""CONCH v1 implementation — ViT-B/16 vision encoder + 12-layer text transformer."""

from __future__ import annotations

import os

import torch
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from conch.open_clip_custom import tokenize as _tokenize
from dotenv import load_dotenv
from torchvision import transforms

from molf_interp.vlm.base import PathologyVLM, VLMConfig

_CONCH_ARCH = "conch_ViT-B-16"
_CONCH_CHECKPOINT = "hf_hub:MahmoodLab/conch"


class ConchV1(PathologyVLM):
    """CONCH v1 wrapper.

    Architecture: ViT-B/16 vision encoder (≈90 M params) + 12-layer text
    transformer (≈110 M params) → 512-d joint projection space.
    License: CC-BY-NC-ND-4.0.
    """

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._device: torch.device = self._init_device()
        self._model = None
        self._tokenizer = None
        self._transform: transforms.Compose | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def preprocess(self) -> transforms.Compose:
        """Return the CONCH preprocessing transform pipeline."""
        if self._transform is None:
            self._transform = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return self._transform

    # ------------------------------------------------------------------
    # Protected interface
    # ------------------------------------------------------------------

    def _load(self) -> None:
        load_dotenv()
        hf_token = os.environ.get(self.config.hf_token_env_var, "")
        if not hf_token:
            raise OSError(
                f"Hugging Face token not found in env var {self.config.hf_token_env_var!r}. "
                "Set it or specify hf_token_env_var in the config."
            )

        arch = self.config.arch or _CONCH_ARCH
        checkpoint = self.config.checkpoint or _CONCH_CHECKPOINT

        model, _ = create_model_from_pretrained(
            arch,
            checkpoint,
            hf_auth_token=hf_token,
        )
        self._tokenizer = get_tokenizer()

        model.to(self._device)
        model.eval()  # type: ignore[attr-defined]
        self._model = model

    # ------------------------------------------------------------------
    # Encode helpers — assume model is loaded and already on correct device
    # ------------------------------------------------------------------

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not loaded — call _load() first")
        return self._model.encode_image(images, proj_contrast=True, normalize=True)  # type: ignore[no-any-return]

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not loaded — call _load() first")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — call _load() first")

        tokens = _tokenize(texts=texts, tokenizer=self._tokenizer).to(self._device)
        return self._model.encode_text(tokens)  # type: ignore[no-any-return]
