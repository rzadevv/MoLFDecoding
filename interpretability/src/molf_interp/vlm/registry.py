"""VLM model registry and loader."""

from __future__ import annotations

from molf_interp.vlm.base import PathologyVLM, VLMConfig
from molf_interp.vlm.conch_v1 import ConchV1

_REGISTRY: dict[str, type[PathologyVLM]] = {
    "conch_v1": ConchV1,
}


def load_vlm(config: VLMConfig) -> PathologyVLM:
    """Instantiate and load the requested VLM.

    Args:
        config: Must have ``model_name`` matching a registered VLM.

    Returns:
        Loaded ``PathologyVLM`` instance with ``_load()`` already called.

    Raises:
        ValueError: If ``model_name`` is not in the registry.
        RuntimeError: If loading fails.
    """
    model_name = config.model_name
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. Available: {list(_REGISTRY.keys())}")
    vlm: PathologyVLM = _REGISTRY[model_name](config)  # type: ignore[call-arg]
    vlm._load()
    return vlm
