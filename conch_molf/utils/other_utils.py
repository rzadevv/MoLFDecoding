import os
import random
from typing import Any

import numpy as np
import torch


def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:  # noqa: ANN401
    """Return ``config[keys[0]][keys[1]]...``, or ``default`` if a key is missing or None."""
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current if current is not None else default


def seed_everything(seed: int = 2025) -> None:
    """Seed Python, NumPy and PyTorch RNGs and enable deterministic cuDNN/cuBLAS behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
