import os
import sys
import datetime
import argparse

import torch
from torch.utils.data import DataLoader as StandardDataLoader
from transformers import get_cosine_schedule_with_warmup
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from ruamel.yaml import YAML

from utils.hest_utils import load_gene_list
from utils.training_utils import (
    get_emb_dim,
    get_model,
    handle_tensor_batch,
    unified_train_epoch,
    evaluate_epoch,
    print_model_params,
    save_checkpoint,
    load_checkpoint,
)
from utils.custom_dataset import PrecomputedEmbeddingDataset
from utils.other_utils import seed_everything


def _validate_run2_config(config):
    assert config["model"].lower() == "molf"
    assert config["PFM_name"].lower() == "conch"
    assert int(config["batch_size"]) == 1
    assert config["gene_vae_pretrained_path"]
    assert config["gene_vae_format"] == "GeneVAE-v2"
    assert config["gene_vae_latent_target"] == "posterior_sample"
    assert int(config["gene_vae_expected_human_epoch"]) == 929
    assert int(config["gene_vae_expected_n_genes"]) == 1386
    assert int(config["gene_vae_expected_latent_dim"]) == 128
    assert int(config["mixture_of_expert"]["num_experts"]) == 6
    assert int(config["mixture_of_expert"]["top_k"]) == 2
    assert config["spatial_encoding"].get("use_sequence_positional_encoding", False) is False
    assert int(config["validation"]["mc_repeats"]) >= 1
    assert int(config["num_epochs"]) > int(config["early_stop_patience"])


def _wandb_init(config, log_dir, continuous_training):
    if not config.get("wandb_enabled", False):
        print("[INFO_TRAIN] W&B disabled for Run 2.")
        return None

    import wandb

    project = config["wandb_project_name"]
    run_id_path = os.path.join(log_dir, "wandb_run_id.txt")
    if continuous_training and os.path.exists(run_id_path):
        run_id = open(run_id_path).read().strip()
        run = wandb.init(project=project, id=run_id, resume="allow", config=config)
    else:
        run = wandb.init(project=project, config=config)
        with open(run_id_path, "w") as f:
            f.write(run.id)
    return wandb


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config_path", "-f", type=str)
    group.add_argument("--log_dir", "-l", type=str)
    parser.add_argument("--continue_training", "-c", action="store_true")
    parser.add_argument(
        "--stop_after_epoch", type=int, default=None,
        help="Optional 1-based epoch at which to stop cleanly after checkpointing (pilot use).",
    )
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    continuous_training = args.continue_training
    log_dir = args.log_dir

    if not continuous_training:
        with open(args.config_path, "r") as f:
            config = yaml.load(f)
        _validate_run2_config(config)

        run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("training_log", f"run_{run_time}_{config['model']}_run2")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.yml"), "w") as f:
            yaml.dump(config, f)
    else:
        if not os.path.exists(log_dir):
            raise FileNotFoundError(log_dir)
        with open(os.path.join(log_dir, "config.yml"), "r") as f:
            config = yaml.load(f)
        _validate_run2_config(config)

    log_path = os.path.join(log_dir, "training_run.txt")
    sys.stdout = open(log_path, "a" if continuous_training else "w", buffering=1)
    sys.stderr = sys.stdout

    seed = int(config.get("training_seed", 2025))
    seed_everything(seed=seed)
    torch.autograd.set_detect_anomaly(bool(config.get("detect_anomaly", False)))

    print("[INFO_TRAIN] Starting corrected CONCH-MoLF Run 2")
    print(f"[INFO_TRAIN] seed={seed}")
    print(f"[INFO_TRAIN] detect_anomaly={config.get('detect_anomaly', False)}")
    print("[INFO_TRAIN] TEST SET IS NOT USED DURING TRAINING/VALIDATION")

    bad_batches_dir = os.path.join(log_dir, "bad_batches")
    os.makedirs(bad_batches_dir, exist_ok=True)

    wandb = _wandb_init(config, log_dir, continuous_training)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_TRAIN] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO_TRAIN] GPU: {torch.cuda.get_device_name(0)}")

    gene_list = load_gene_list(config)
    gene_dim = len(gene_list)
    emb_dim = get_emb_dim(config)
    print(f"[INFO_TRAIN] gene_dim={gene_dim}, emb_dim={emb_dim}")

    train_dataset = PrecomputedEmbeddingDataset(
        config, split="train", gene_list=gene_list, DEBUG=config["debug"]
    )
    val_dataset = PrecomputedEmbeddingDataset(
        config, split="val", gene_list=gene_list, DEBUG=config["debug"]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = StandardDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=train_generator,
        num_workers=int(config.get("num_workers", 6)),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = StandardDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=int(config.get("num_workers", 6)),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(
        f"[INFO_TRAIN] Train dataset size: {len(train_dataset)}, "
        f"Validation dataset size: {len(val_dataset)}"
    )

    flow_path = AffineProbPath(scheduler=CondOTScheduler())
    model = get_model(config, gene_dim, emb_dim, device, flow_path)
    print_model_params(model)

    gating_params = []
    main_model_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "velocity_predictor.gating" in name:
            gating_params.append(param)
        else:
            main_model_params.append(param)

    if not gating_params:
        raise RuntimeError("Run 2 expected MoE gating parameters but found none")

    main_lr = float(config["lr"])
    gating_lr = float(config["mixture_of_expert"]["gating_lr"])
    optimizer = torch.optim.AdamW(
        [
            {"params": main_model_params, "lr": main_lr},
            {"params": gating_params, "lr": gating_lr},
        ],
        weight_decay=float(config["weight_decay"]),
    )

    if config["debug"]:
        config["num_epochs"] = min(int(config["num_epochs"]), 10)

    num_epochs = int(config["num_epochs"])
    num_training_steps = num_epochs * len(train_loader)
    warmup_fraction = float(config["scheduler"]["warmup_fraction"])
    warmup_steps = int(round(warmup_fraction * num_training_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"[INFO_TRAIN] Scheduler=cosine_with_warmup")
    print(f"[INFO_TRAIN] Total training steps={num_training_steps}")
    print(f"[INFO_TRAIN] Warmup steps={warmup_steps} ({warmup_steps / len(train_loader):.1f} epochs)")

    if continuous_training:
        checkpoint_path = os.path.join(log_dir, "ckpt", "flow_checkpoint_latest.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        start_epoch, best_val_loss, extra = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        best_epoch = int(extra.get("best_epoch", 0))
        early_stop_counter = int(extra.get("early_stop_counter", 0))
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = 0
        early_stop_counter = 0

    early_stop_patience = int(config["early_stop_patience"])
    early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))
    val_mc_repeats = int(config["validation"]["mc_repeats"])
    val_seed = int(config["validation"]["seed"])

    print(
        f"[INFO_TRAIN] early_stop_patience={early_stop_patience}, "
        f"min_delta={early_stop_min_delta}"
    )
    print(
        f"[INFO_TRAIN] fixed validation MC repeats={val_mc_repeats}, seed={val_seed}"
    )

    ckpt_dir = os.path.join(log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        # Resume-safe deterministic but different shuffle every epoch.
        train_generator.manual_seed(seed + epoch)

        epoch_summary = unified_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            batch_handler_fn=handle_tensor_batch,
            epoch=epoch,
            bad_batch_dir=bad_batches_dir,
            model_name=config["model"].lower(),
            save_bad_batches=True,
        )
        epoch_summary["epoch"] = epoch + 1

        val_summary = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            batch_handler_fn=handle_tensor_batch,
            epoch=epoch,
            model_name=config["model"].lower(),
            description="val",
            mc_repeats=val_mc_repeats,
            eval_seed=val_seed,
        )
        val_summary["epoch"] = epoch + 1

        current_val = float(val_summary["val/total_loss"])
        improved = current_val < (best_val_loss - early_stop_min_delta)

        if improved:
            best_val_loss = current_val
            best_epoch = epoch + 1
            early_stop_counter = 0
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,  # zero-based internal epoch; load_checkpoint resumes at +1
                path=best_path,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                early_stop_counter=early_stop_counter,
                validation_mc_repeats=val_mc_repeats,
                validation_seed=val_seed,
            )
            print(
                f"[INFO_TRAIN] Saved new best model at epoch {best_epoch} "
                f"with fixed-MC val_loss {best_val_loss:.6f}"
            )
        else:
            early_stop_counter += 1
            print(
                f"[INFO_TRAIN] No improvement for {early_stop_counter} epoch(s). "
                f"current={current_val:.6f}, best={best_val_loss:.6f}"
            )

        # Save latest only AFTER validation/best/counter updates so resume state is exact.
        latest_path = os.path.join(ckpt_dir, "flow_checkpoint_latest.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            path=latest_path,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            early_stop_counter=early_stop_counter,
            validation_mc_repeats=val_mc_repeats,
            validation_seed=val_seed,
        )

        if wandb is not None:
            wandb.log({**epoch_summary, **val_summary, "best_epoch": best_epoch})

        print(
            f"[INFO_TRAIN] Epoch {epoch + 1}: "
            f"train={epoch_summary['train/total_loss']:.6f}, "
            f"val={current_val:.6f}, "
            f"val_mc_sd={val_summary['val/total_loss_mc_sd']:.6f}, "
            f"lr_main={optimizer.param_groups[0]['lr']:.8g}, "
            f"lr_gate={optimizer.param_groups[1]['lr']:.8g}"
        )

        if early_stop_counter >= early_stop_patience:
            print(f"[INFO_TRAIN] Early stopping triggered at epoch {epoch + 1}.")
            break

        if args.stop_after_epoch is not None and (epoch + 1) >= args.stop_after_epoch:
            print(
                f"[INFO_TRAIN] Pilot stop requested after epoch {epoch + 1}; "
                "latest checkpoint is resume-ready."
            )
            break

    if wandb is not None:
        wandb.finish()
    print(
        f"[INFO_TRAIN] Training completed. best_epoch={best_epoch}, "
        f"best_fixed_mc_val={best_val_loss:.6f}"
    )


if __name__ == "__main__":
    main()
