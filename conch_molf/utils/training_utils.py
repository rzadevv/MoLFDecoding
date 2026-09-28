import os
import random
import math
import numpy as np
import torch
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm

import torch
from torch.amp import autocast

from utils.pfm_preprocesing import (
    load_uni_v2,
    load_uni_v1,
    load_hoptimus0,
    load_virchow2,
    load_gigapath,
)
from utils.hest_utils import load_gene_list
from models.flow_matching import LatentFlow,LatentFlow_CFG
from models.baseline_models import SimpleMLP 


def capture_rng_state():
    """Capture RNG state so validation/checkpointing can be resume-safe."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path, **kwargs):
    """Save a checkpoint atomically, including optimizer/scheduler state."""
    checkpoint = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": float(best_val_loss),
        "rng_state": capture_rng_state(),
    }
    checkpoint.update(kwargs)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"[INFO] Checkpoint saved atomically to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """Load model/training state and restore RNG state when available."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if (
        scheduler is not None
        and checkpoint.get("scheduler_state") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if checkpoint.get("rng_state") is not None:
        restore_rng_state(checkpoint["rng_state"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    extra_info = {
        k: v
        for k, v in checkpoint.items()
        if k not in [
            "epoch", "model_state", "optimizer_state", "scheduler_state",
            "best_val_loss", "rng_state"
        ]
    }

    print(f"[INFO] Checkpoint loaded from {path}, resuming at epoch index {start_epoch}")
    return start_epoch, best_val_loss, extra_info


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 50)
    print(f"[INFO] Total parameters   : {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print("=" * 50)


def get_emb_dim(config):
    if config["PFM_name"] in ["univ2", "hoptimus0", "gigapath"]:
        emb_dim = 1536
    elif config["PFM_name"] == "uni":
        emb_dim = 1024
    elif config["PFM_name"] == "virchow2":
        emb_dim = 1280
    elif config["PFM_name"] == "conch":
        emb_dim = 512
    else:
        raise ValueError(f"[FLOW_ST_ERROR] Unknown PFM name: {config['PFM_name']}")
    return emb_dim


def get_PFM_model(config):
    if config["PFM_name"].lower() in ["univ2"]:
        # Load the pretrained pathology foundation model and transform
        PFM_model, transform = load_uni_v2(FIRST_TIME=False)
    elif config["PFM_name"].lower() == "uni":
        PFM_model, transform = load_uni_v1(FIRST_TIME=False)
    elif config["PFM_name"].lower() == "virchow2":
        PFM_model, transform = load_virchow2()
    elif config["PFM_name"].lower() == "hoptimus0":
        PFM_model, transform = load_hoptimus0()
    elif config["PFM_name"].lower() == "gigapath":
        PFM_model, transform = load_gigapath()
    else:
        raise ValueError(f"[FLOW_ST_ERROR] Unknown PFM name: {config['PFM_name']}")
    return PFM_model, transform


def get_model(config, gene_dim, emb_dim, device, flow_path=None):
    # velocity field model init
    model_name = config["model"]
    print(f"[INFO_MDOEL] Initializing model: {model_name}")
    # if model_name.lower() == "latentflow":
    #     model = LatentFlow(
    #         config=config,
    #         gene_dim=gene_dim,
    #         time_dim=1,
    #         img_emb_dim=emb_dim,
    #         flow_path=flow_path,
    #     ).to(device)
    
    if model_name.lower() == "molf":
        model = LatentFlow_CFG(
            config=config,
            gene_dim=gene_dim,
            time_dim=1,
            img_emb_dim=emb_dim,
            flow_path=flow_path,
        ).to(device)
    elif model_name.lower() == "simplemlp":
        model = SimpleMLP(
            embedding_dim=emb_dim,
            hidden_features_1=config["hidden_dim"][0],
            hidden_features_2=config["hidden_dim"][1],
            output_features=gene_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    return model


def handle_tensor_batch(model, batch, model_name: str):
    """
    Batch handler for a non-graph model that uses padded tensors.
    """
    # Get the target device from the model itself.
    device = next(model.parameters()).device

    # 1. Unpack the batch tuple. This order MUST match the return order of your pad_collate_fn.
    #  embeddings, coords, gene_tensor, gene_mask_tensor, oncotree_onehoted, sample_id
    embeddings, _, genes, gene_mask_tensor, onco_onehot, sample_id = batch

    # 2. Move each TENSOR to the GPU individually.
    #    Note: sample_ids is a list of strings, it stays on the CPU.
    embeddings = embeddings.to(device)
    genes = genes.to(device)
    gene_mask_tensor = gene_mask_tensor.to(device)
    onco_onehot = onco_onehot.to(device)

    patch_validity = gene_mask_tensor.sum(dim=-1) > 0  # Shape: (B, P)
    if not torch.any(patch_validity):
        print("\n\n!!!!!!!!!! DEBUG TRIGGERED: BAD BATCH DETECTED !!!!!!!!!!")
        print(f"SAMPLE {sample_id} - This entire batch has a mask that will result in all patches being padded.")
        print(f"Shape of x_1: {genes.shape}")
        print(f"Shape of gene_mask_patches: {gene_mask_tensor.shape}")
        
        # Print statistics of the input data that's causing the high KL loss
        print(f"Stats for x_1: mean={genes.mean():.4f}, std={genes.std():.4f}, min={genes.min():.4f}, max={genes.max():.4f}")
        
        # Print the validity mask to confirm it's all False
        print("Patch validity (should be all False):")
        print(patch_validity)

        # It's useful to see the sum of the mask per patch
        print("Sum of mask per patch (should be all 0):")
        print(gene_mask_tensor.sum(dim=-1))

    # forward(self, x_1, emb,  gene_mask_patches, onco_onehot):
    if model_name == "molf":
        total_loss, loss_dict, moe_load_balancing_logits = model(
            x_1=genes,  # The padded input features
            emb=embeddings,
            gene_mask_patches=gene_mask_tensor,  # The padded ground truth targets
            onco_onehot=onco_onehot
        )
    elif model_name == "simplemlp":
        total_loss, loss_dict,= model(
            x=embeddings,
            y=genes,
        )
        moe_load_balancing_logits = None
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return total_loss, loss_dict, moe_load_balancing_logits, sample_id



def unified_train_epoch(
    model, dataloader, optimizer, scheduler, scaler, batch_handler_fn, epoch,
    bad_batch_dir, model_name, save_bad_batches=True,
):
    """Run one strict FP32 training epoch and average only successful updates."""
    model.train()

    epoch_total_loss = 0.0
    epoch_loss_components = {}
    successful_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} Training")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)

        try:
            # Stage-2 training is intentionally full FP32 for numerical stability.
            total_loss, loss_dict_for_logging, moe_load_balancing_logits, sample_id = (
                batch_handler_fn(model, batch, model_name)
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"non-finite loss={total_loss.item()} sample={sample_id}"
                )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm={grad_norm} sample={sample_id}"
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        except Exception as e:
            if save_bad_batches:
                torch.save(
                    batch,
                    os.path.join(bad_batch_dir, f"epoch{epoch}_iter{i}_error.pt"),
                )
            # Run 2 is fail-fast: never silently train through a corrupted batch.
            raise RuntimeError(
                f"Training failed at epoch={epoch + 1}, batch={i}, sample={locals().get('sample_id', 'unknown')}: {e}"
            ) from e

        batch_log_data = {
            "batch/total_loss": float(total_loss.item()),
            **{k: float(v) for k, v in loss_dict_for_logging.items()},
        }
        pbar.set_postfix(**batch_log_data)

        epoch_total_loss += float(total_loss.item())
        for key, value in loss_dict_for_logging.items():
            epoch_loss_components[key] = (
                epoch_loss_components.get(key, 0.0) + float(value)
            )
        successful_batches += 1

    if successful_batches != len(dataloader):
        raise RuntimeError(
            f"Expected {len(dataloader)} successful batches, got {successful_batches}"
        )

    epoch_summary = {
        "train/total_loss": epoch_total_loss / successful_batches,
        "train/successful_batches": successful_batches,
    }
    for key, value in epoch_loss_components.items():
        epoch_summary[f"train/{key}"] = value / successful_batches

    for group_idx, group in enumerate(optimizer.param_groups):
        epoch_summary[f"train/lr_group_{group_idx}"] = float(group["lr"])

    return epoch_summary


def _set_eval_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_epoch(
    model, dataloader, batch_handler_fn, epoch, model_name, description="val",
    mc_repeats=1, eval_seed=31001,
):
    """
    Deterministic Monte-Carlo validation. Each epoch uses the same fixed random
    draws, while the caller's RNG state is restored afterward so validation does
    not alter subsequent training randomness.
    """
    if mc_repeats < 1:
        raise ValueError("mc_repeats must be >= 1")

    model.eval()
    saved_rng_state = capture_rng_state()

    repeat_total = [0.0 for _ in range(mc_repeats)]
    epoch_loss_components = {}
    total_calls = 0

    if getattr(model, "use_moe", False):
        num_experts = model.num_experts
        device = next(model.parameters()).device
        total_expert_counts = torch.zeros(num_experts, device=device)
        router_entropy_sum = 0.0
        router_margin_sum = 0.0
        router_top1_sum = 0.0
        router_token_count = 0
    else:
        total_expert_counts = None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} {description}")

    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                batch_totals = []
                for rep in range(mc_repeats):
                    # Same slide/replicate receives the same random draw every epoch.
                    forward_seed = int(eval_seed + rep * 100_000 + batch_idx)
                    _set_eval_seed(forward_seed)

                    total_loss, loss_dict, moe_logits, _ = batch_handler_fn(
                        model, batch, model_name
                    )
                    total = float(total_loss.item())
                    if not math.isfinite(total):
                        raise RuntimeError(
                            f"Non-finite {description} loss at batch={batch_idx}, rep={rep}"
                        )

                    repeat_total[rep] += total
                    batch_totals.append(total)
                    total_calls += 1

                    for key, value in loss_dict.items():
                        epoch_loss_components[key] = (
                            epoch_loss_components.get(key, 0.0) + float(value)
                        )

                    if total_expert_counts is not None and moe_logits is not None:
                        k = model.velocity_predictor.gating.k
                        probs = torch.softmax(moe_logits, dim=-1)
                        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
                        total_expert_counts += torch.bincount(
                            top_indices.flatten(), minlength=num_experts
                        )
                        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                        entropy = entropy / math.log(num_experts)
                        router_entropy_sum += float(entropy.sum().item())
                        router_top1_sum += float(top_probs[:, 0].sum().item())
                        if k >= 2:
                            router_margin_sum += float(
                                (top_probs[:, 0] - top_probs[:, 1]).sum().item()
                            )
                        router_token_count += int(probs.shape[0])

                pbar.set_postfix(total_loss=sum(batch_totals) / len(batch_totals))
    finally:
        restore_rng_state(saved_rng_state)

    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Empty evaluation dataloader")

    repeat_means = [x / num_batches for x in repeat_total]
    avg_total = float(np.mean(repeat_means))
    mc_sd = float(np.std(repeat_means, ddof=1)) if mc_repeats > 1 else 0.0

    prefix = description.lower()
    epoch_summary = {
        f"{prefix}/total_loss": avg_total,
        f"{prefix}/total_loss_mc_sd": mc_sd,
        f"{prefix}/mc_repeats": int(mc_repeats),
    }
    for key, value in epoch_loss_components.items():
        epoch_summary[f"{prefix}/{key}"] = value / total_calls

    if total_expert_counts is not None:
        total_routed = float(total_expert_counts.sum().item())
        if total_routed > 0:
            utilization = total_expert_counts / total_routed
            for i, util in enumerate(utilization):
                epoch_summary[f"{prefix}/expert_{i}_utilization_pct"] = (
                    float(util.item()) * 100.0
                )
        if router_token_count > 0:
            epoch_summary[f"{prefix}/router_normalized_entropy"] = (
                router_entropy_sum / router_token_count
            )
            epoch_summary[f"{prefix}/router_top1_probability"] = (
                router_top1_sum / router_token_count
            )
            epoch_summary[f"{prefix}/router_top1_top2_margin"] = (
                router_margin_sum / router_token_count
            )

    return epoch_summary

