#!/usr/bin/env python3

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(
    "/data/cat/ws/rzmi154h-molf-pathology/"
    "conch_molf_production"
)

RUN2 = ROOT / "run2_code"
EVAL = ROOT / "run2_evaluation"

os.chdir(RUN2)

import sys
sys.path.insert(0, str(RUN2))

from utils.custom_dataset import get_expression_and_mask
from utils.hest_utils import load_gene_list

CONFIG = ROOT / "run2_frozen/config.yml"
META = RUN2 / "data/Hest_Bench/HEST_v1_1_0.csv"
SPLITS = RUN2 / "data/Hest_Bench/hest_pathway_clean_split_0"

OUT_SLIDE = EVAL / "TARGET_DISTRIBUTION_PER_SLIDE.csv"
OUT_GROUP = EVAL / "TARGET_DISTRIBUTION_BY_TECH_SPLIT.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


with open(CONFIG) as f:
    config = yaml.safe_load(f)

gene_list = load_gene_list(config)

emb_path = Path(config["paths_config"]["embeddings_path"])
st_path = Path(config["paths_config"]["st_path"])

if not emb_path.is_absolute():
    emb_path = RUN2 / emb_path

if not st_path.is_absolute():
    st_path = RUN2 / st_path

meta = read_csv(META)
meta_by_id = {str(r["id"]): r for r in meta}

slide_rows = []

for split in ["train", "val", "test"]:

    split_rows = read_csv(SPLITS / f"{split}_split.csv")

    print()
    print(f"=== {split.upper()} : {len(split_rows)} slides ===")

    for i, s in enumerate(split_rows, 1):

        sid = str(s["sample_id"])

        if sid not in meta_by_id:
            raise RuntimeError(f"{sid}: missing HEST metadata")

        technology = meta_by_id[sid]["st_technology"]

        emb_file = emb_path / f"{sid}_embeddings.pt"
        h5ad_file = st_path / f"{sid}.h5ad"

        if not emb_file.exists():
            raise FileNotFoundError(emb_file)

        if not h5ad_file.exists():
            raise FileNotFoundError(h5ad_file)

        embeddings_dict = torch.load(
            emb_file,
            map_location="cpu",
            weights_only=True,
        )

        expr, mask = get_expression_and_mask(
            str(h5ad_file),
            embeddings_dict,
            gene_list,
        )

        valid = mask > 0.5
        x = expr[valid].astype(np.float64)

        if x.size == 0:
            raise RuntimeError(f"{sid}: zero valid target entries")

        if not np.isfinite(x).all():
            raise RuntimeError(f"{sid}: non-finite target")

        n_patches = expr.shape[0]
        measured_genes = int((mask.sum(axis=0) > 0).sum())

        mean = float(x.mean())
        sd = float(x.std())
        rms = float(np.sqrt(np.mean(x * x)))
        zero_frac = float(np.mean(x == 0.0))

        q50 = float(np.quantile(x, 0.50))
        q90 = float(np.quantile(x, 0.90))
        q95 = float(np.quantile(x, 0.95))
        q99 = float(np.quantile(x, 0.99))
        vmax = float(x.max())
        vmin = float(x.min())

        slide_rows.append({
            "split": split,
            "sample_id": sid,
            "technology": technology,
            "organ": meta_by_id[sid].get("organ", ""),
            "n_patches": n_patches,
            "measured_genes": measured_genes,
            "valid_entries": int(x.size),
            "mean": mean,
            "sd": sd,
            "rms": rms,
            "zero_fraction": zero_frac,
            "q50": q50,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "min": vmin,
            "max": vmax,
        })

        print(
            f"[{i:03d}/{len(split_rows):03d}] "
            f"{sid:<10s} "
            f"{technology:<22s} "
            f"genes={measured_genes:4d} "
            f"mean={mean:.4f} "
            f"sd={sd:.4f} "
            f"rms={rms:.4f} "
            f"zero={zero_frac:.3f}"
        )


fields = list(slide_rows[0].keys())

with open(OUT_SLIDE, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(slide_rows)


groups = defaultdict(list)

for r in slide_rows:
    groups[(r["split"], r["technology"])].append(r)


group_rows = []

print()
print("=== TARGET DISTRIBUTION BY SPLIT × TECHNOLOGY ===")
print(
    f"{'split':<6s} {'technology':<24s} "
    f"{'N':>4s} {'genes':>9s} "
    f"{'mean':>9s} {'sd':>9s} {'rms':>9s} "
    f"{'zero':>9s} {'q95':>9s}"
)

for (split, tech), rr in sorted(groups.items()):

    def mean_field(k):
        return float(np.mean([float(r[k]) for r in rr]))

    row = {
        "split": split,
        "technology": tech,
        "n_slides": len(rr),
        "mean_measured_genes": mean_field("measured_genes"),
        "mean_target": mean_field("mean"),
        "mean_sd": mean_field("sd"),
        "mean_rms": mean_field("rms"),
        "mean_zero_fraction": mean_field("zero_fraction"),
        "mean_q95": mean_field("q95"),
        "mean_q99": mean_field("q99"),
    }

    group_rows.append(row)

    print(
        f"{split:<6s} {tech:<24s} "
        f"{len(rr):4d} "
        f"{row['mean_measured_genes']:9.1f} "
        f"{row['mean_target']:9.4f} "
        f"{row['mean_sd']:9.4f} "
        f"{row['mean_rms']:9.4f} "
        f"{row['mean_zero_fraction']:9.4f} "
        f"{row['mean_q95']:9.4f}"
    )


with open(OUT_GROUP, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(group_rows[0].keys()),
    )
    w.writeheader()
    w.writerows(group_rows)


print()
print(f"SAVED: {OUT_SLIDE}")
print(f"SAVED: {OUT_GROUP}")
print("TARGET DISTRIBUTION AUDIT: PASS")
