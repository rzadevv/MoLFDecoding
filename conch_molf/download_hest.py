"""Download the HEST-1k data used in this project from HuggingFace.

Fetches, for every slide in the selected splits of data/splits/, the patch file
`patches/<sample_id>.h5` and the expression file `st/<sample_id>.h5ad`, plus the
HEST metadata table `HEST_v1_1_0.csv`, from the `MahmoodLab/hest` dataset on
HuggingFace (CC BY-NC-SA 4.0). No token is required; HF_TOKEN is used if set.
Files that already exist are skipped.

Afterwards, set `paths_config.patches_path` to `<out-dir>/patches` and
`paths_config.st_path` to `<out-dir>/st` in the training config.

Usage:
  python download_hest.py --out-dir data/hest               # all 504 slides
  python download_hest.py --out-dir data/hest --splits test  # 23 test slides
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "MahmoodLab/hest"
METADATA = "HEST_v1_1_0.csv"
SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def sample_ids(splits: list[str]) -> list[str]:
    """Sample IDs of the given splits, in split-file order."""
    ids: list[str] = []
    for split in splits:
        ids += pd.read_csv(SPLITS_DIR / f"{split}_split.csv")["sample_id"].tolist()
    return ids


def fetch(filename: str, out_dir: Path, token: str | None) -> None:
    """Download one file from the HEST dataset into ``out_dir`` unless it already exists."""
    if (out_dir / filename).exists():
        return
    hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=out_dir, token=token)


def main() -> None:
    """Parse arguments and download metadata plus the selected assets for every slide."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data/hest"))
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"]
    )
    parser.add_argument("--assets", nargs="+", choices=["patches", "st"], default=["patches", "st"])
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    ids = sample_ids(args.splits)
    print(f"{len(ids)} slides from {', '.join(args.splits)} -> {args.out_dir}")
    fetch(METADATA, args.out_dir, token)
    suffix = {"patches": ".h5", "st": ".h5ad"}
    for i, sid in enumerate(ids, 1):
        for asset in args.assets:
            fetch(f"{asset}/{sid}{suffix[asset]}", args.out_dir, token)
        print(f"[{i}/{len(ids)}] {sid}")


if __name__ == "__main__":
    main()
