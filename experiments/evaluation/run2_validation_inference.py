#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


EXPECTED_CKPT_SHA = (
    "dbe23d4e2f670749ee746272337a21d3d14e49fc6dab34dd43bea05fb727965d"
)
EXPECTED_CONFIG_SHA = (
    "ba6e43d95a6acaa9a4a4fe0b2ca3f0e4a5d5d0fa8a35c0f233a5d7f3417cfd0c"
)

EXPECTED_SOURCE_SHA = {
    "models/flow_matching.py":
        "4e2191cc581466ee5b97d687abfc1c7b8a8f7802f89718e49a088d11396d8d58",
    "models/genevae_v2.py":
        "963f881c01ef230644c954583abaf132bb81dda2565c6da123e470db89462a13",
    "utils/training_utils.py":
        "0cb6944afd4ccad43fa8f5ac55553bdd71c00f570322a4a282aad1a8b8265c32",
    "utils/custom_dataset.py":
        "c849f2fb65ad9221d4a401bff5f01f6508d3185afb9f9e13986c2949e5c7fdec",
    "utils/hest_utils.py":
        "dee179841684f723a0a13b9bfdeda1fd5965390211108034484334bdd3489a55",
}

FIXED_METHODS = {"euler", "midpoint", "heun3", "rk4"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def slide_seed(base_seed, sample_id):
    """
    Deterministic seed depending only on:
      base inference seed + sample ID.

    Therefore every solver / CFG setting receives the exact same x0
    for a particular slide.
    """
    payload = f"{base_seed}|{sample_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def masked_metrics(pred, target, mask):
    """
    pred,target,mask: numpy [P,G]

    Returns:
      global valid-entry sums
      per-gene PCC
      per-gene SSE/SAE/count
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch pred={pred.shape}, target={target.shape}, mask={mask.shape}"
        )

    valid = mask > 0.5

    if not np.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction encountered")
    if not np.isfinite(target[valid]).all():
        raise FloatingPointError("Non-finite valid ground truth encountered")

    diff = pred - target

    valid_count_gene = valid.sum(axis=0).astype(np.int64)

    # Valid-entry MSE / MAE components.
    sq = np.where(valid, diff * diff, 0.0)
    ab = np.where(valid, np.abs(diff), 0.0)

    gene_sse = sq.sum(axis=0, dtype=np.float64)
    gene_sae = ab.sum(axis=0, dtype=np.float64)

    total_sse = float(gene_sse.sum(dtype=np.float64))
    total_sae = float(gene_sae.sum(dtype=np.float64))
    total_n = int(valid_count_gene.sum())

    # Vectorized Pearson across patches for each gene.
    # Pearson's common n-1 factor cancels from numerator/denominator.
    x = np.where(valid, pred, 0.0)
    y = np.where(valid, target, 0.0)

    n = valid_count_gene.astype(np.float64)

    sx = x.sum(axis=0, dtype=np.float64)
    sy = y.sum(axis=0, dtype=np.float64)
    sxx = (x * x).sum(axis=0, dtype=np.float64)
    syy = (y * y).sum(axis=0, dtype=np.float64)
    sxy = (x * y).sum(axis=0, dtype=np.float64)

    safe_n = np.maximum(n, 1.0)

    cov_num = sxy - (sx * sy / safe_n)
    varx_num = sxx - (sx * sx / safe_n)
    vary_num = syy - (sy * sy / safe_n)

    # Protect only against tiny floating-point negatives.
    varx_num = np.maximum(varx_num, 0.0)
    vary_num = np.maximum(vary_num, 0.0)

    scale_x = np.maximum(sxx, 1.0)
    scale_y = np.maximum(syy, 1.0)

    varx_ok = varx_num > (1e-12 * scale_x)
    vary_ok = vary_num > (1e-12 * scale_y)

    pcc_valid = (n >= 2) & varx_ok & vary_ok

    pcc = np.full(pred.shape[1], np.nan, dtype=np.float64)
    denom = np.sqrt(varx_num[pcc_valid] * vary_num[pcc_valid])
    pcc[pcc_valid] = cov_num[pcc_valid] / denom

    # Numerical guard.
    pcc[pcc_valid] = np.clip(pcc[pcc_valid], -1.0, 1.0)

    measured_gene = n > 0
    gt_constant = (n >= 2) & (~vary_ok)
    pred_constant = (n >= 2) & (~varx_ok)

    return {
        "total_sse": total_sse,
        "total_sae": total_sae,
        "total_n": total_n,
        "gene_sse": gene_sse,
        "gene_sae": gene_sae,
        "gene_n": valid_count_gene,
        "gene_pcc": pcc,
        "measured_gene_count": int(measured_gene.sum()),
        "valid_pcc_count": int(np.isfinite(pcc).sum()),
        "gt_constant_count": int(gt_constant.sum()),
        "pred_constant_count": int(pred_constant.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run-2 VALIDATION-ONLY inference evaluator."
    )

    parser.add_argument("--code-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(FIXED_METHODS),
    )
    parser.add_argument("--num-steps", required=True, type=int)
    parser.add_argument("--guidance-scale", required=True, type=float)
    parser.add_argument("--noise-seed", required=True, type=int)

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all 96 validation slides; positive = smoke-test prefix only",
    )

    args = parser.parse_args()

    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")

    if args.guidance_scale <= 0:
        raise ValueError("--guidance-scale must be > 0")

    code_root = Path(args.code_root).resolve()
    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()

    print("=== RUN-2 VALIDATION INFERENCE ===")
    print("This evaluator is hard-coded to split='val'.")
    print("The held-out test split cannot be requested.\n")

    # ------------------------------------------------------------------
    # Provenance / anti-drift checks.
    # ------------------------------------------------------------------

    actual_ckpt_sha = sha256(ckpt_path)
    actual_config_sha = sha256(config_path)

    if actual_ckpt_sha != EXPECTED_CKPT_SHA:
        raise RuntimeError(
            f"Wrong Run-2 checkpoint SHA:\n"
            f"expected {EXPECTED_CKPT_SHA}\n"
            f"actual   {actual_ckpt_sha}"
        )

    if actual_config_sha != EXPECTED_CONFIG_SHA:
        raise RuntimeError(
            f"Wrong Run-2 config SHA:\n"
            f"expected {EXPECTED_CONFIG_SHA}\n"
            f"actual   {actual_config_sha}"
        )

    for rel, expected in EXPECTED_SOURCE_SHA.items():
        p = code_root / rel
        actual = sha256(p)
        if actual != expected:
            raise RuntimeError(
                f"Run-2 source drift detected: {rel}\n"
                f"expected {expected}\n"
                f"actual   {actual}"
            )

    print("Critical frozen-artifact/source hashes: PASS")

    sys.path.insert(0, str(code_root))

    # The frozen training config intentionally contains repository-relative
    # paths such as data/Hallmark/... and data/HEST1K/....
    # Resolve them exactly as training did by evaluating from code_root.
    os.chdir(code_root)
    print(f"Working directory: {Path.cwd()}")

    from flow_matching.path.scheduler import CondOTScheduler
    from flow_matching.path import AffineProbPath
    from flow_matching.solver import ODESolver

    from utils.hest_utils import load_gene_list
    from utils.training_utils import get_emb_dim, get_model
    from utils.custom_dataset import PrecomputedEmbeddingDataset

    import flow_matching

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["model"].lower() != "molf":
        raise ValueError(f"Expected model=molf, got {config['model']}")

    if config["PFM_name"].lower() != "conch":
        raise ValueError(f"Expected PFM_name=conch, got {config['PFM_name']}")

    if bool(config.get("debug", False)):
        raise ValueError("Production config unexpectedly has debug=True")

    # ------------------------------------------------------------------
    # Deterministic global state.
    # Source noise itself is generated separately per slide below.
    # ------------------------------------------------------------------

    random.seed(args.noise_seed)
    np.random.seed(args.noise_seed)
    torch.manual_seed(args.noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.noise_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for production validation inference")

    device = torch.device("cuda")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(
        "flow_matching:",
        getattr(flow_matching, "__version__", "unknown"),
    )

    # ------------------------------------------------------------------
    # VALIDATION DATA ONLY.
    # ------------------------------------------------------------------

    gene_list = load_gene_list(config)
    gene_dim = len(gene_list)

    val_dataset = PrecomputedEmbeddingDataset(
        config=config,
        split="val",
        gene_list=gene_list,
        single_oncotree_code=None,
        DEBUG=False,
    )

    if len(val_dataset) != 96:
        raise RuntimeError(
            f"Expected exactly 96 validation slides, found {len(val_dataset)}"
        )

    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    n_eval = len(val_dataset) if args.limit == 0 else min(args.limit, len(val_dataset))

    print(f"Validation slides available: {len(val_dataset)}")
    print(f"Validation slides evaluated: {n_eval}")
    print(f"Genes in joint target list: {gene_dim}")

    # Never instantiate train or test datasets.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # MODEL.
    # ------------------------------------------------------------------

    emb_dim = get_emb_dim(config)
    flow_path = AffineProbPath(scheduler=CondOTScheduler())

    model = get_model(
        config=config,
        gene_dim=gene_dim,
        emb_dim=emb_dim,
        device=device,
        flow_path=flow_path,
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state")

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    human_epoch = int(checkpoint["epoch"]) + 1
    best_epoch = int(checkpoint.get("best_epoch", -1))
    best_val_loss = float(checkpoint.get("best_val_loss", float("nan")))

    if human_epoch != 239 or best_epoch != 239:
        raise RuntimeError(
            f"Wrong checkpoint metadata: human_epoch={human_epoch}, "
            f"best_epoch={best_epoch}"
        )

    print(f"Checkpoint human epoch: {human_epoch}")
    print(f"Checkpoint best epoch: {best_epoch}")
    print(f"Checkpoint fixed-MC train-selection val: {best_val_loss:.12f}")

    # ------------------------------------------------------------------
    # Clean ODE protocol.
    #
    # num_steps=N means exactly intended fixed integration resolution 1/N.
    # We only request the final state at t=1.
    # ------------------------------------------------------------------

    time_grid = torch.tensor(
        [0.0, 1.0],
        device=device,
        dtype=torch.float32,
    )
    step_size = 1.0 / float(args.num_steps)

    solver = ODESolver(velocity_model=model.get_velocity)

    out_dir.mkdir(parents=True, exist_ok=False)

    slide_rows = []

    total_sse = 0.0
    total_sae = 0.0
    total_n = 0

    gene_sse = np.zeros(gene_dim, dtype=np.float64)
    gene_sae = np.zeros(gene_dim, dtype=np.float64)
    gene_n = np.zeros(gene_dim, dtype=np.int64)

    gene_pcc_sum = np.zeros(gene_dim, dtype=np.float64)
    gene_pcc_count = np.zeros(gene_dim, dtype=np.int64)

    print()
    print(
        f"Protocol: method={args.method} "
        f"steps={args.num_steps} "
        f"step_size={step_size:.8f} "
        f"guidance={args.guidance_scale} "
        f"noise_seed={args.noise_seed}"
    )
    print()

    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            if i >= n_eval:
                break

            embeddings, _, genes_gt, genes_mask, onco_onehot, sample_id = batch
            sample_id = str(sample_id[0])

            embeddings = embeddings.to(device, non_blocking=True)
            genes_gt = genes_gt.to(device, non_blocking=True)
            genes_mask = genes_mask.to(device, non_blocking=True)
            onco_onehot = onco_onehot.to(device, non_blocking=True)

            B, P, _ = embeddings.shape

            if B != 1:
                raise RuntimeError(f"Expected batch size 1, got {B}")

            # Create x0 on CPU using a deterministic slide-specific seed.
            # This guarantees identical starting noise across all inference
            # hyperparameter settings, independent of how many model calls
            # a particular solver makes.
            sseed = slide_seed(args.noise_seed, sample_id)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(sseed)

            x_init = torch.randn(
                (B, P, model.gene_latent_dim),
                generator=gen,
                dtype=torch.float32,
                device="cpu",
            ).to(device)

            latent_final = solver.sample(
                x_init=x_init,
                step_size=step_size,
                method=args.method,
                time_grid=time_grid,
                return_intermediates=False,
                enable_grad=False,
                emb=embeddings,
                onco_onehot=onco_onehot,
                guidance_scale=float(args.guidance_scale),
            )

            predicted_genes = model.gene_autoencoder.decode(
                latent_final,
                genes_mask,
            )

            if predicted_genes.shape != genes_gt.shape:
                raise RuntimeError(
                    f"{sample_id}: prediction shape {predicted_genes.shape} "
                    f"!= GT shape {genes_gt.shape}"
                )

            pred_np = predicted_genes.squeeze(0).float().cpu().numpy()
            gt_np = genes_gt.squeeze(0).float().cpu().numpy()
            mask_np = genes_mask.squeeze(0).float().cpu().numpy()

            m = masked_metrics(
                pred=pred_np,
                target=gt_np,
                mask=mask_np,
            )

            if m["total_n"] <= 0:
                raise RuntimeError(f"{sample_id}: no valid gene entries")

            slide_mse = m["total_sse"] / m["total_n"]
            slide_mae = m["total_sae"] / m["total_n"]
            slide_rmse = math.sqrt(slide_mse)

            valid_pcc = np.isfinite(m["gene_pcc"])
            if valid_pcc.any():
                slide_mean_pcc = float(np.mean(m["gene_pcc"][valid_pcc]))
                slide_median_pcc = float(np.median(m["gene_pcc"][valid_pcc]))
            else:
                slide_mean_pcc = float("nan")
                slide_median_pcc = float("nan")

            total_sse += m["total_sse"]
            total_sae += m["total_sae"]
            total_n += m["total_n"]

            gene_sse += m["gene_sse"]
            gene_sae += m["gene_sae"]
            gene_n += m["gene_n"]

            gene_pcc_sum[valid_pcc] += m["gene_pcc"][valid_pcc]
            gene_pcc_count[valid_pcc] += 1

            slide_rows.append({
                "sample_id": sample_id,
                "n_patches": int(P),
                "valid_entries": int(m["total_n"]),
                "measured_genes": int(m["measured_gene_count"]),
                "valid_pcc_genes": int(m["valid_pcc_count"]),
                "gt_constant_genes": int(m["gt_constant_count"]),
                "pred_constant_genes": int(m["pred_constant_count"]),
                "masked_mse": float(slide_mse),
                "masked_rmse": float(slide_rmse),
                "masked_mae": float(slide_mae),
                "mean_gene_pcc": finite_or_none(slide_mean_pcc),
                "median_gene_pcc": finite_or_none(slide_median_pcc),
                "noise_seed": int(sseed),
            })

            print(
                f"[{i+1:03d}/{n_eval:03d}] "
                f"{sample_id:12s} "
                f"P={P:6d} "
                f"genes={m['measured_gene_count']:4d} "
                f"PCCgenes={m['valid_pcc_count']:4d} "
                f"MSE={slide_mse:.6f} "
                f"PCC={slide_mean_pcc:.6f}",
                flush=True,
            )

            del (
                embeddings,
                genes_gt,
                genes_mask,
                onco_onehot,
                x_init,
                latent_final,
                predicted_genes,
            )

    if len(slide_rows) != n_eval:
        raise RuntimeError(
            f"Expected {n_eval} evaluated slides, got {len(slide_rows)}"
        )

    # ------------------------------------------------------------------
    # Aggregate metrics.
    # ------------------------------------------------------------------

    micro_mse = total_sse / total_n
    micro_rmse = math.sqrt(micro_mse)
    micro_mae = total_sae / total_n

    slide_pcc_values = [
        r["mean_gene_pcc"]
        for r in slide_rows
        if r["mean_gene_pcc"] is not None
    ]

    macro_slide_pcc = (
        float(np.mean(slide_pcc_values))
        if slide_pcc_values else float("nan")
    )

    median_slide_pcc = (
        float(np.median(slide_pcc_values))
        if slide_pcc_values else float("nan")
    )

    gene_mean_pcc = np.full(gene_dim, np.nan, dtype=np.float64)
    gene_has_pcc = gene_pcc_count > 0
    gene_mean_pcc[gene_has_pcc] = (
        gene_pcc_sum[gene_has_pcc] / gene_pcc_count[gene_has_pcc]
    )

    macro_gene_pcc = (
        float(np.mean(gene_mean_pcc[gene_has_pcc]))
        if gene_has_pcc.any() else float("nan")
    )

    # ------------------------------------------------------------------
    # Save slide metrics.
    # ------------------------------------------------------------------

    slide_csv = out_dir / "per_slide_metrics.csv"

    with open(slide_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(slide_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(slide_rows)

    # ------------------------------------------------------------------
    # Save per-gene metrics.
    # ------------------------------------------------------------------

    gene_csv = out_dir / "per_gene_metrics.csv"

    with open(gene_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gene",
            "valid_entries",
            "masked_mse",
            "masked_mae",
            "slides_with_valid_pcc",
            "mean_slide_pcc",
        ])

        for g, gene in enumerate(gene_list):
            if gene_n[g] > 0:
                gmse = gene_sse[g] / gene_n[g]
                gmae = gene_sae[g] / gene_n[g]
            else:
                gmse = float("nan")
                gmae = float("nan")

            writer.writerow([
                gene,
                int(gene_n[g]),
                finite_or_none(gmse),
                finite_or_none(gmae),
                int(gene_pcc_count[g]),
                finite_or_none(gene_mean_pcc[g]),
            ])

    summary = {
        "evaluation_split": "val",
        "test_split_used": False,
        "n_validation_slides_available": 96,
        "n_slides_evaluated": int(n_eval),
        "smoke_test": bool(args.limit > 0),

        "method": args.method,
        "num_steps": int(args.num_steps),
        "step_size": float(step_size),
        "guidance_scale": float(args.guidance_scale),
        "noise_seed": int(args.noise_seed),

        "checkpoint_human_epoch": int(human_epoch),
        "checkpoint_best_epoch": int(best_epoch),
        "checkpoint_training_selection_val": float(best_val_loss),

        "checkpoint_sha256": actual_ckpt_sha,
        "config_sha256": actual_config_sha,

        "masked_mse_micro": float(micro_mse),
        "masked_rmse_micro": float(micro_rmse),
        "masked_mae_micro": float(micro_mae),

        "mean_slide_mean_gene_pcc": finite_or_none(macro_slide_pcc),
        "median_slide_mean_gene_pcc": finite_or_none(median_slide_pcc),
        "mean_gene_mean_slide_pcc": finite_or_none(macro_gene_pcc),

        "total_valid_gene_entries": int(total_n),
        "genes_with_at_least_one_valid_pcc": int(gene_has_pcc.sum()),

        "torch_version": torch.__version__,
        "flow_matching_version":
            getattr(flow_matching, "__version__", "unknown"),
        "cuda_device": torch.cuda.get_device_name(0),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("=== VALIDATION SUMMARY ===")
    print(f"slides                   : {n_eval}")
    print(f"masked MSE               : {micro_mse:.9f}")
    print(f"masked RMSE              : {micro_rmse:.9f}")
    print(f"masked MAE               : {micro_mae:.9f}")
    print(f"mean slide gene-PCC      : {macro_slide_pcc:.9f}")
    print(f"median slide gene-PCC    : {median_slide_pcc:.9f}")
    print(f"mean gene slide-PCC      : {macro_gene_pcc:.9f}")
    print(f"valid gene entries       : {total_n}")
    print(f"genes with valid PCC     : {int(gene_has_pcc.sum())}")
    print()
    print(f"Results: {out_dir}")
    print("VALIDATION INFERENCE: PASS")
    print("TEST SET USED: NO")


if __name__ == "__main__":
    main()
