"""Patch-embedding extraction with pathology foundation models (PFMs).

Supported models: UNI, UNI-v2, H-optimus-0, Virchow2, Prov-GigaPath and CONCH.
Run-2 uses CONCH (512-d, L2-normalised projection into the joint image-text
space). Gated models need the HuggingFace token in ``HF_TOKEN``.

Usage (from ``conch_molf/``):
    python utils/pfm_preprocesing.py --config configs/training_config_conch_run2.yml [--split test]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from huggingface_hub import hf_hub_download, login
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm import tqdm

try:
    from conch.open_clip_custom import create_model_from_pretrained
except ImportError:
    create_model_from_pretrained = None

# Allow `python utils/pfm_preprocesing.py` from conch_molf/ as well as package-style imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.hest_utils import read_assets_from_h5

Transform = Callable[[Image.Image], torch.Tensor]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _hf_login() -> str | None:
    """Log in to HuggingFace if ``HF_TOKEN`` is set; return the token."""
    token = os.getenv("HF_TOKEN")
    if token:
        login(token)
    return token


def _download_weights(repo_id: str, local_dir: str, force: bool) -> str:
    """Download ``pytorch_model.bin`` from ``repo_id`` unless it is already present."""
    path = os.path.join(local_dir, "pytorch_model.bin")
    if force or not os.path.exists(path):
        print(f"[INFO_preprocessing] Downloading {repo_id} weights...")
        os.makedirs(local_dir, exist_ok=True)
        hf_hub_download(
            repo_id, filename="pytorch_model.bin", local_dir=local_dir, force_download=force
        )
    return path


def load_uni_v2(FIRST_TIME: bool = False) -> tuple[torch.nn.Module, Transform]:  # noqa: N803
    """UNI-v2 (ViT-g/14, 1536-d); ``FIRST_TIME`` forces a fresh weight download."""
    _hf_login()
    weights = _download_weights("MahmoodLab/UNI2-h", "Pathology_FM/ckpts/uni2-h/", force=FIRST_TIME)
    model = timm.create_model(
        pretrained=False,
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    print("[INFO_preprocessing] UNI2-h model loaded")
    return model.eval(), transform


def load_uni_v1(FIRST_TIME: bool = False) -> tuple[torch.nn.Module, Transform]:  # noqa: N803
    """UNI (ViT-L/16, 1024-d); ``FIRST_TIME`` forces a fresh weight download."""
    _hf_login()
    weights = _download_weights("MahmoodLab/UNI", "Pathology_FM/ckpts/uni/", force=FIRST_TIME)
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    print("[INFO_preprocessing] UNI model loaded")
    return model.eval(), transform


def load_hoptimus0() -> tuple[torch.nn.Module, Transform]:
    """H-optimus-0 (ViT-g/14)."""
    _hf_login()
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )
    print("[INFO_preprocessing] hoptimus0 model loaded")
    return model.eval(), transform


def load_virchow2() -> tuple[torch.nn.Module, Transform]:
    """Virchow2 (ViT-H/14) with its published preprocessing."""
    _hf_login()
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print("[INFO_preprocessing] virchow2 model loaded")
    return model.eval(), transform


def load_gigapath() -> tuple[torch.nn.Module, Transform]:
    """Prov-GigaPath tile encoder (ViT-g/14)."""
    _hf_login()
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    print("[INFO_preprocessing] gigapath model loaded")
    return model.eval(), transform


def load_conch() -> tuple[torch.nn.Module, Transform]:
    """CONCH v1 (ViT-B/16); embeddings are taken from ``encode_image`` in :func:`main`."""
    token = _hf_login()
    if create_model_from_pretrained is None:
        raise ImportError("The 'conch' package is required for PFM_name 'conch'.")
    model, _ = create_model_from_pretrained(
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=token
    )
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    print("[INFO_preprocessing] conch model loaded")
    return model.eval(), transform


PFM_LOADERS: dict[str, Callable[[], tuple[torch.nn.Module, Transform]]] = {
    "univ2": load_uni_v2,
    "uni": load_uni_v1,
    "virchow2": load_virchow2,
    "hoptimus0": load_hoptimus0,
    "gigapath": load_gigapath,
    "conch": load_conch,
}


def get_PFM_model(config: dict[str, Any]) -> tuple[torch.nn.Module, Transform]:  # noqa: N802
    """Load the foundation model named by ``config["PFM_name"]`` and its transform."""
    name = config["PFM_name"]
    if name not in PFM_LOADERS:
        raise ValueError(f"[ERROR_preprocessing] Unknown PFM name: {name}")
    return PFM_LOADERS[name]()


def decode_barcode(b: Any) -> str:  # noqa: ANN401
    """Normalise a spot/cell barcode read from HDF5 (bytes, numpy scalar or str) to ``str``."""
    if isinstance(b, np.ndarray):
        b = b.item()
    if isinstance(b, bytes):
        return b.decode()
    if isinstance(b, np.bytes_):
        return str(b.astype(str))
    if isinstance(b, str) and b.startswith("[b'") and b.endswith("']"):
        return b[3:-2]  # stringified list of bytes, e.g. "[b'AAACAACGAATAGTTC-1']"
    return str(b)


def get_all_sample_ids(config: dict[str, Any], return_split: str | None = None) -> list[str]:
    """Sample IDs from the split CSVs; ``return_split`` selects one (default: train+val+test)."""
    split_path = config["paths_config"]["splits_path"]
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"[ERROR_preprocessing] {split_path} does not exist.")
    splits = {}
    for split in ("train", "val", "test", "test_mouse"):
        csv_path = os.path.join(split_path, f"{split}_split.csv")
        if os.path.exists(csv_path):
            splits[split] = pd.read_csv(csv_path, header=0)["sample_id"].tolist()
    if return_split is None:
        return splits["train"] + splits["val"] + splits["test"]
    if return_split not in splits:
        raise FileNotFoundError(
            f"[ERROR_preprocessing] no {return_split}_split.csv in {split_path}"
        )
    return splits[return_split]


def main(config: dict[str, Any], return_split: str | None = None, batch_size: int = 32) -> None:
    """Encode every patch of every selected slide with the configured foundation model.

    Reads ``<patches_path>/<sample_id>.h5`` and writes a ``{barcode: embedding}`` dict to
    ``<embeddings_path>/<sample_id>_embeddings.pt``; slides that are already encoded are skipped.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches_path = config["paths_config"]["patches_path"]
    save_dir = config["paths_config"]["embeddings_path"]
    os.makedirs(save_dir, exist_ok=True)

    model, transform = get_PFM_model(config)
    model = model.to(device)
    for sample_id in tqdm(
        get_all_sample_ids(config, return_split=return_split), desc="Preprocessing"
    ):
        save_path = os.path.join(save_dir, f"{sample_id}_embeddings.pt")
        if os.path.exists(save_path):
            continue
        asset_path = os.path.join(patches_path, f"{sample_id}.h5")
        if not os.path.exists(asset_path):
            print(f"[INFO_preprocessing] Skipping {sample_id}: {asset_path} not found.")
            continue
        assets, _ = read_assets_from_h5(asset_path)
        patches = assets["img"]
        barcode = assets["barcode"] if "barcode" in assets else assets["barcodes"]
        assert patches.shape[0] == len(barcode), (
            f"[ERROR_preprocessing] {patches.shape[0]} != {len(barcode)}"
        )

        embeddings = {}
        with torch.no_grad():
            for start in range(0, patches.shape[0], batch_size):
                indices = range(start, min(start + batch_size, patches.shape[0]))
                batch = [
                    transform(Image.fromarray(np.asarray(patches[j]).astype(np.uint8)))
                    for j in indices
                ]  # each 3 x 224 x 224
                batch_tensor = torch.stack(batch).to(device)
                if config["PFM_name"] == "conch":
                    batch_emb = model.encode_image(batch_tensor, proj_contrast=True, normalize=True)
                else:
                    batch_emb = model(batch_tensor)
                for n, j in enumerate(indices):
                    embeddings[decode_barcode(barcode[j])] = batch_emb[n].detach().cpu()

        torch.save(embeddings, save_path)
        print(f"[INFO_preprocessing] Saved {len(embeddings)} embeddings to {save_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patch embeddings with a pathology foundation model."
    )
    parser.add_argument("--config", default="configs/training_config_conch_run2.yml")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "test_mouse"],
        default=None,
        help="Encode only this split (default: train + val + test).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, return_split=args.split, batch_size=args.batch_size)
