#!/usr/bin/env python3

import csv
import inspect
import math
import os
import sys
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

sys.path.insert(0, str(RUN2))
os.chdir(RUN2)

from models.genevae_v2 import GeneTransformerVAEV2
from utils.custom_dataset import get_expression_and_mask
from utils.hest_utils import load_gene_list

RUN2_CONFIG = ROOT / "run2_frozen/config.yml"
META = RUN2 / "data/Hest_Bench/HEST_v1_1_0.csv"
SPLITS = RUN2 / "data/Hest_Bench/hest_pathway_clean_split_0"

OUT_SLIDE = EVAL / "GENEVAE_V2_RECON_PER_SLIDE.csv"
OUT_GROUP = EVAL / "GENEVAE_V2_RECON_BY_TECH_SPLIT.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def resolve(base, p):
    p = Path(p)
    if p.is_absolute():
        return p
    return base / p


with open(RUN2_CONFIG) as f:
    cfg = yaml.safe_load(f)

print("=== RUN2 GENE-VAE SETTINGS ===")
for k, v in cfg.items():
    if "gene_vae" in str(k).lower():
        print(f"{k}: {v}")

pretrained_path = Path(cfg["gene_vae_pretrained_path"])
if not pretrained_path.is_absolute():
    pretrained_path = RUN2 / pretrained_path

checkpoint_path = pretrained_path / cfg["gene_vae_checkpoint_file"]

# FlowMatching constructs this from the frozen GeneVAE directory.
config_candidates = []

for key in [
    "gene_vae_config_file",
    "gene_vae_config_filename",
]:
    if key in cfg:
        config_candidates.append(pretrained_path / cfg[key])

config_candidates += [
    pretrained_path / "config.yml",
    pretrained_path / "config.yaml",
    pretrained_path / "training_config.yml",
    pretrained_path / "genevae_v2_config.yml",
]

genevae_config_path = next(
    (p for p in config_candidates if p.exists()),
    None,
)

if genevae_config_path is None:
    raise FileNotFoundError(
        "Could not locate GeneVAE-v2 config in "
        f"{pretrained_path}. Candidates={config_candidates}"
    )

if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

print()
print("GeneVAE directory :", pretrained_path)
print("GeneVAE config    :", genevae_config_path)
print("GeneVAE checkpoint:", checkpoint_path)

with open(genevae_config_path) as f:
    gv_cfg = yaml.safe_load(f)

model_cfg = gv_cfg["model"]

print()
print("GeneTransformerVAEV2 signature:")
print(inspect.signature(GeneTransformerVAEV2))

# Construct using the frozen GeneVAE config plus the explicit
# Run-2 dimensionality contract.
sig = inspect.signature(GeneTransformerVAEV2)

valid_args = {
    k: v
    for k, v in model_cfg.items()
    if k in sig.parameters
}

valid_args["n_genes"] = int(
    cfg["gene_vae_expected_n_genes"]
)

valid_args["latent_dim"] = int(
    cfg["gene_vae_expected_latent_dim"]
)

print()
print("GeneVAE constructor args:", valid_args)

vae = GeneTransformerVAEV2(**valid_args)

if vae.n_genes != int(cfg["gene_vae_expected_n_genes"]):
    raise RuntimeError(
        f"GeneVAE n_genes mismatch: {vae.n_genes}"
    )

if vae.latent_dim != int(cfg["gene_vae_expected_latent_dim"]):
    raise RuntimeError(
        f"GeneVAE latent_dim mismatch: {vae.latent_dim}"
    )

ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

print()
print("Checkpoint type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Checkpoint keys:", sorted(ckpt.keys()))

    state = None

    for key in [
        "model_state",
        "model_state_dict",
        "state_dict",
        "model",
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            print("Using checkpoint state key:", key)
            break

    if state is None:
        # It might itself be a plain state_dict.
        if all(isinstance(k, str) for k in ckpt.keys()):
            if any(k.startswith(("encoder", "decoder", "gene_", "latent_"))
                   for k in ckpt.keys()):
                state = ckpt

    if state is None:
        raise RuntimeError(
            f"Could not identify model state in checkpoint keys "
            f"{sorted(ckpt.keys())}"
        )
else:
    raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

vae.load_state_dict(state, strict=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this audit; refusing accidental CPU execution"
    )

device = torch.device("cuda")
vae = vae.to(device)
vae.eval()

for p in vae.parameters():
    p.requires_grad_(False)

print("GeneVAE checkpoint load: PASS")
print("Audit device:", device)
print("GPU:", torch.cuda.get_device_name(0))

gene_list = load_gene_list(cfg)

if len(gene_list) != 1386:
    raise RuntimeError(f"Expected 1386 genes, got {len(gene_list)}")

emb_path = resolve(RUN2, cfg["paths_config"]["embeddings_path"])
st_path = resolve(RUN2, cfg["paths_config"]["st_path"])

meta_rows = read_csv(META)
meta_by_id = {str(r["id"]): r for r in meta_rows}

slide_rows = []


def masked_metrics(gt, pred, mask):
    valid = mask > 0.5

    y = gt[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)

    if y.size == 0:
        raise RuntimeError("No valid values")

    err = p - y

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    target_energy = float(np.mean(y ** 2))
    nmse_energy = (
        mse / target_energy
        if target_energy > 0
        else float("nan")
    )

    # Gene-wise spatial PCC within the slide.
    pccs = []

    measured_gene = mask[0] > 0.5

    for j in np.where(measured_gene)[0]:
        yy = gt[:, j].astype(np.float64)
        pp = pred[:, j].astype(np.float64)

        if (
            len(yy) >= 2
            and np.std(yy) > 0
            and np.std(pp) > 0
        ):
            r = np.corrcoef(yy, pp)[0, 1]
            if np.isfinite(r):
                pccs.append(float(r))

    mean_pcc = (
        float(np.mean(pccs))
        if pccs
        else float("nan")
    )

    return mse, mae, nmse_energy, mean_pcc, len(pccs)


with torch.inference_mode():

    for split in ["train", "val", "test"]:

        split_rows = read_csv(SPLITS / f"{split}_split.csv")

        print()
        print(f"=== {split.upper()} : {len(split_rows)} slides ===")

        for i, s in enumerate(split_rows, 1):

            sid = str(s["sample_id"])

            if sid not in meta_by_id:
                raise RuntimeError(f"{sid}: missing metadata")

            tech = meta_by_id[sid]["st_technology"]

            emb_file = emb_path / f"{sid}_embeddings.pt"
            h5ad_file = st_path / f"{sid}.h5ad"

            emb_dict = torch.load(
                emb_file,
                map_location="cpu",
                weights_only=True,
            )

            expr, mask = get_expression_and_mask(
                str(h5ad_file),
                emb_dict,
                gene_list,
            )

            x = (
                torch.from_numpy(expr)
                .float()
                .unsqueeze(0)
                .to(device)
            )
            m = (
                torch.from_numpy(mask)
                .float()
                .unsqueeze(0)
                .to(device)
            )

            # Deterministic reconstruction:
            # z = posterior mean, not a random posterior sample.
            recon, z_mean, z_log_var = vae(
                x,
                m,
                sample=False,
            )

            pred = recon.squeeze(0).cpu().numpy()

            if not np.isfinite(pred).all():
                raise RuntimeError(f"{sid}: non-finite reconstruction")

            mse, mae, nmse, pcc, valid_pcc_genes = masked_metrics(
                expr,
                pred,
                mask,
            )

            valid_entries = int(mask.sum())
            measured_genes = int((mask.sum(axis=0) > 0).sum())

            slide_rows.append({
                "split": split,
                "sample_id": sid,
                "technology": tech,
                "organ": meta_by_id[sid].get("organ", ""),
                "n_patches": int(expr.shape[0]),
                "measured_genes": measured_genes,
                "valid_entries": valid_entries,
                "masked_mse": mse,
                "masked_mae": mae,
                "energy_normalized_mse": nmse,
                "mean_gene_pcc": pcc,
                "valid_pcc_genes": valid_pcc_genes,
            })

            print(
                f"[{i:03d}/{len(split_rows):03d}] "
                f"{sid:<10s} "
                f"{tech:<23s} "
                f"MSE={mse:8.5f} "
                f"nMSE={nmse:7.4f} "
                f"PCC={pcc:7.4f}"
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
print("=== GENEVAE-V2 DETERMINISTIC RECONSTRUCTION ===")
print(
    f"{'split':<6s} "
    f"{'technology':<24s} "
    f"{'N':>4s} "
    f"{'MSE':>10s} "
    f"{'MAE':>10s} "
    f"{'nMSE':>10s} "
    f"{'PCC':>10s}"
)

for (split, tech), rr in sorted(groups.items()):

    # Micro MSE/MAE: weight slides by measured entries.
    total_n = sum(r["valid_entries"] for r in rr)

    mse = sum(
        r["masked_mse"] * r["valid_entries"]
        for r in rr
    ) / total_n

    mae = sum(
        r["masked_mae"] * r["valid_entries"]
        for r in rr
    ) / total_n

    # For normalized error/PCC report slide-level mean;
    # these are descriptive and avoid large slides dominating.
    nmse_vals = [
        r["energy_normalized_mse"]
        for r in rr
        if math.isfinite(r["energy_normalized_mse"])
    ]

    pcc_vals = [
        r["mean_gene_pcc"]
        for r in rr
        if math.isfinite(r["mean_gene_pcc"])
    ]

    nmse = float(np.mean(nmse_vals))
    pcc = float(np.mean(pcc_vals))

    row = {
        "split": split,
        "technology": tech,
        "n_slides": len(rr),
        "masked_mse_micro": mse,
        "masked_mae_micro": mae,
        "mean_slide_energy_normalized_mse": nmse,
        "mean_slide_gene_pcc": pcc,
    }

    group_rows.append(row)

    print(
        f"{split:<6s} "
        f"{tech:<24s} "
        f"{len(rr):4d} "
        f"{mse:10.6f} "
        f"{mae:10.6f} "
        f"{nmse:10.6f} "
        f"{pcc:10.6f}"
    )


with open(OUT_GROUP, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(group_rows[0].keys()),
    )
    w.writeheader()
    w.writerows(group_rows)


# ------------------------------------------------------------
# Direct comparison with final Run-2 test results.
# ------------------------------------------------------------

molf_path = EVAL / "FINAL_TEST_PER_SLIDE_5SEED.csv"

if molf_path.exists():

    molf_rows = {
        r["sample_id"]: r
        for r in read_csv(molf_path)
    }

    test_recon = [
        r for r in slide_rows
        if r["split"] == "test"
    ]

    print()
    print("=== TEST: GENEVAE RECON vs FINAL MOLF ===")
    print(
        f"{'slide':<10s} "
        f"{'tech':<10s} "
        f"{'GV2_MSE':>10s} "
        f"{'MoLF_MSE':>10s} "
        f"{'ratio':>9s}"
    )

    for r in sorted(
        test_recon,
        key=lambda z: float(
            molf_rows[z["sample_id"]]["masked_mse_mean"]
        ),
        reverse=True,
    ):
        sid = r["sample_id"]
        molf_mse = float(molf_rows[sid]["masked_mse_mean"])
        gv_mse = r["masked_mse"]

        ratio = (
            molf_mse / gv_mse
            if gv_mse > 0
            else float("inf")
        )

        print(
            f"{sid:<10s} "
            f"{r['technology'][:10]:<10s} "
            f"{gv_mse:10.5f} "
            f"{molf_mse:10.5f} "
            f"{ratio:9.2f}"
        )


print()
print(f"SAVED: {OUT_SLIDE}")
print(f"SAVED: {OUT_GROUP}")
print("GENEVAE-V2 RECONSTRUCTION AUDIT: PASS")
