#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


# ============================================================
# FIXED PATHS / PROTOCOL
# ============================================================

ROOT = Path(
    "/data/cat/ws/rzmi154h-molf-pathology/"
    "conch_molf_production"
)

RUN2 = ROOT / "run2_code"
FREEZE = ROOT / "run2_frozen"
OUTROOT = ROOT / "run2_interpretability" / "all23"

OLDROOT = Path(
    "/data/horse/ws/rzmi154h-molf-recovered/"
    "rzmi154h-molf_workspace-1785452536/"
    "MOLFDECODE/molf-interpretability"
)

CONFIG = FREEZE / "config.yml"
CHECKPOINT = FREEZE / "run2_best_epoch239.pt"

CONCEPT_PARQUET = (
    OLDROOT
    / "data/outputs/vlm/conch_text_embeddings/"
      "view_a_concept_embeddings.parquet"
)

READY_PARQUET = (
    OLDROOT
    / "data/cache/concept_embeddings/"
      "view_a_ready.parquet"
)


EXPECTED_TEST_IDS = (
    "INT13",
    "INT14",
    "INT15",
    "INT24",
    "MEND154",
    "MEND156",
    "MEND157",
    "MEND158",
    "MEND159",
    "MEND160",
    "MEND161",
    "MEND162",
    "NCBI643",
    "NCBI684",
    "TENX116",
    "TENX117",
    "TENX141",
    "TENX147",
    "TENX148",
    "TENX149",
    "TENX99",
    "ZEN48",
    "ZEN49",
)


BASE_SEEDS = (
    41001,
    42001,
    43001,
    44001,
    45001,
)

GUIDANCE_SCALE = 1.5
STEP_SIZE = 0.5
STEP_TIMES = (0.0, 0.5)

NUM_EXPERTS = 6
TOP_K_ROUTER = 2
TOP_K_PATCHES = 256
TOP_N_CONCEPTS = 50


# ============================================================
# FROZEN HASHES
# ============================================================

EXPECTED_CKPT_SHA = (
    "dbe23d4e2f670749ee746272337a21d3"
    "d14e49fc6dab34dd43bea05fb727965d"
)

EXPECTED_CONFIG_SHA = (
    "ba6e43d95a6acaa9a4a4fe0b2ca3f0"
    "e4a5d5d0fa8a35c0f233a5d7f3417cfd0c"
)

EXPECTED_FLOW_SHA = (
    "4e2191cc581466ee5b97d687abfc1c7b"
    "8a8f7802f89718e49a088d11396d8d58"
)

EXPECTED_CONCEPT_SHA = (
    "7025a95ad665ea7b8f120b66e6e7cbd"
    "54879666fd9029f24e618d532454df352"
)

EXPECTED_READY_SHA = (
    "3839a446984d3c70ed44b393b5df52e"
    "030c6ea713ac29930c796131ddcc588f6"
)


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def slide_seed(base_seed, sample_id):
    payload = f"{base_seed}|{sample_id}".encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            "little",
        )
        % (2**63 - 1)
    )


def decode_value(x):
    if isinstance(x, np.ndarray):
        x = x.item()

    if isinstance(x, bytes):
        return x.decode()

    if isinstance(x, np.bytes_):
        return x.tobytes().decode()

    return str(x)


def normalize_rows(x):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0.0] = 1.0

    return (
        x / norms
    ).astype(
        np.float32,
        copy=False,
    )


def resolve_path(p):
    p = Path(p)

    if p.is_absolute():
        return p

    return RUN2 / p


def write_csv(path, fieldnames, rows):
    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CONCEPT BANK
# ============================================================

def load_concept_bank():

    # Use recovered PyArrow directly.
    candidate_sites = [
        OLDROOT / ".venv/lib/python3.11/site-packages",
        OLDROOT / ".venv/lib64/python3.11/site-packages",
    ]

    for site in candidate_sites:
        if (site / "pyarrow").exists():
            sys.path.append(str(site))
            break
    else:
        raise RuntimeError(
            "Recovered pyarrow not found"
        )

    import pyarrow
    import pyarrow.parquet as pq

    print(
        "PyArrow:",
        pyarrow.__version__,
    )

    concept_table = pq.read_table(
        CONCEPT_PARQUET
    )

    ready_table = pq.read_table(
        READY_PARQUET
    )

    concept_data = (
        concept_table.to_pydict()
    )

    ready_data = (
        ready_table.to_pydict()
    )


    ready_lookup = {}

    for i in range(
        ready_table.num_rows
    ):

        key = (
            str(
                ready_data[
                    "concept_id"
                ][i]
            ),
            str(
                ready_data[
                    "species"
                ][i]
            ),
        )

        if key in ready_lookup:
            raise RuntimeError(
                f"Duplicate ready concept key {key}"
            )

        ready_lookup[key] = {
            "concept_name":
                ready_data[
                    "concept_name"
                ][i],
            "embedding_set":
                ready_data[
                    "embedding_set"
                ][i],
        }


    concept_rows = []
    seen = set()

    for i in range(
        concept_table.num_rows
    ):

        concept_id = str(
            concept_data[
                "concept_id"
            ][i]
        )

        species = str(
            concept_data[
                "species"
            ][i]
        )

        key = (
            concept_id,
            species,
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate concept key {key}"
            )

        seen.add(key)

        if key not in ready_lookup:
            raise RuntimeError(
                f"Missing ready concept {key}"
            )

        meta = ready_lookup[key]

        if (
            species != "Homo_sapiens"
            or meta[
                "embedding_set"
            ] != "visual"
        ):
            continue

        embedding = np.asarray(
            concept_data[
                "embedding"
            ][i],
            dtype=np.float32,
        ).reshape(-1)

        if embedding.shape != (512,):
            raise RuntimeError(
                f"{key}: bad embedding "
                f"shape {embedding.shape}"
            )

        if not np.isfinite(
            embedding
        ).all():
            raise RuntimeError(
                f"{key}: non-finite embedding"
            )

        concept_rows.append({
            "concept_id":
                concept_id,
            "concept_name":
                str(
                    meta[
                        "concept_name"
                    ]
                ),
            "tier":
                str(
                    concept_data[
                        "tier"
                    ][i]
                ),
            "embedding_set":
                "visual",
            "species":
                species,
            "prompt_text":
                str(
                    concept_data[
                        "prompt_text"
                    ][i]
                ),
            "embedding":
                embedding,
        })


    if len(
        concept_rows
    ) != 2450:
        raise RuntimeError(
            "Expected 2450 visual "
            "Homo_sapiens concepts, "
            f"found {len(concept_rows)}"
        )


    text_matrix = normalize_rows(
        np.stack(
            [
                r["embedding"]
                for r in concept_rows
            ],
            axis=0,
        )
    )

    if text_matrix.shape != (
        2450,
        512,
    ):
        raise RuntimeError(
            f"Bad text matrix "
            f"{text_matrix.shape}"
        )

    return (
        concept_rows,
        text_matrix,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        required=True,
        choices=EXPECTED_TEST_IDS,
    )

    args = parser.parse_args()

    sample_id = args.sample_id


    # --------------------------------------------------------
    # Read-only frozen checks.
    # --------------------------------------------------------

    if (
        sha256(CHECKPOINT)
        != EXPECTED_CKPT_SHA
    ):
        raise RuntimeError(
            "Run-2 checkpoint hash mismatch"
        )

    if (
        sha256(CONFIG)
        != EXPECTED_CONFIG_SHA
    ):
        raise RuntimeError(
            "Run-2 config hash mismatch"
        )

    if (
        sha256(
            RUN2
            / "models/flow_matching.py"
        )
        != EXPECTED_FLOW_SHA
    ):
        raise RuntimeError(
            "flow_matching.py hash mismatch"
        )

    if (
        sha256(CONCEPT_PARQUET)
        != EXPECTED_CONCEPT_SHA
    ):
        raise RuntimeError(
            "CONCH concept bank hash mismatch"
        )

    if (
        sha256(READY_PARQUET)
        != EXPECTED_READY_SHA
    ):
        raise RuntimeError(
            "ready prompt bank hash mismatch"
        )

    print(
        "FROZEN ARTIFACT HASHES: PASS"
    )


    # --------------------------------------------------------
    # Temporary output dir.
    # Only renamed to final directory after every check passes.
    # --------------------------------------------------------

    final_dir = (
        OUTROOT
        / sample_id
    )

    tmp_dir = (
        OUTROOT
        / f".{sample_id}.tmp"
    )

    if tmp_dir.exists():
        shutil.rmtree(
            tmp_dir
        )

    tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Exact Run-2 imports/environment.
    # --------------------------------------------------------

    sys.path.insert(
        0,
        str(RUN2),
    )

    os.chdir(
        RUN2
    )


    from flow_matching.path.scheduler import CondOTScheduler
    from flow_matching.path import AffineProbPath
    from flow_matching.solver import ODESolver

    from utils.hest_utils import load_gene_list
    from utils.training_utils import (
        get_emb_dim,
        get_model,
    )
    from utils.custom_dataset import (
        PrecomputedEmbeddingDataset
    )


    with open(
        CONFIG
    ) as f:
        config = yaml.safe_load(f)


    if (
        config["model"].lower()
        != "molf"
    ):
        raise RuntimeError(
            "Expected model=molf"
        )

    if (
        config["PFM_name"].lower()
        != "conch"
    ):
        raise RuntimeError(
            "Expected PFM_name=conch"
        )


    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required"
        )

    device = torch.device(
        "cuda"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


    # --------------------------------------------------------
    # Full test dataset, exact expected IDs.
    # --------------------------------------------------------

    gene_list = load_gene_list(
        config
    )

    gene_dim = len(
        gene_list
    )


    dataset = PrecomputedEmbeddingDataset(
        config=config,
        split="test",
        gene_list=gene_list,
        single_oncotree_code=None,
        DEBUG=False,
    )


    if len(dataset) != 23:
        raise RuntimeError(
            f"Expected 23 test slides, "
            f"found {len(dataset)}"
        )


    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


    found_ids = []
    target_batch = None

    for batch in loader:

        sid = str(
            batch[-1][0]
        )

        found_ids.append(
            sid
        )

        if sid == sample_id:
            target_batch = batch


    if set(
        found_ids
    ) != set(
        EXPECTED_TEST_IDS
    ):
        raise RuntimeError(
            "Held-out test ID set mismatch"
        )


    if target_batch is None:
        raise RuntimeError(
            f"{sample_id} not found"
        )


    (
        embeddings,
        coords,
        genes_gt,
        genes_mask,
        onco_onehot,
        batch_sample_id,
    ) = target_batch


    batch_sample_id = str(
        batch_sample_id[0]
    )

    if (
        batch_sample_id
        != sample_id
    ):
        raise RuntimeError(
            "Sample ID mismatch"
        )


    embeddings = embeddings.to(
        device
    )

    onco_onehot = onco_onehot.to(
        device
    )

    B, P, D = embeddings.shape


    if B != 1:
        raise RuntimeError(
            f"Expected batch size 1, got {B}"
        )

    if D != 512:
        raise RuntimeError(
            f"Expected CONCH D=512, got {D}"
        )


    print(
        f"{sample_id}: "
        f"patches={P}, "
        f"CONCH_dim={D}, "
        f"genes={gene_dim}"
    )


    # --------------------------------------------------------
    # Exact patch identity:
    # patch index ↔ barcode ↔ H5 coords ↔ CONCH embedding.
    # --------------------------------------------------------

    emb_file = (
        resolve_path(
            config[
                "paths_config"
            ][
                "embeddings_path"
            ]
        )
        / f"{sample_id}_embeddings.pt"
    )

    patch_file = (
        resolve_path(
            config[
                "paths_config"
            ][
                "patches_path"
            ]
        )
        / f"{sample_id}.h5"
    )


    emb_dict = torch.load(
        emb_file,
        map_location="cpu",
        weights_only=True,
    )


    barcodes = [
        decode_value(k)
        for k in emb_dict.keys()
    ]


    patch_matrix = np.stack(
        [
            np.asarray(
                v.cpu(),
                dtype=np.float32,
            ).reshape(-1)
            for v in emb_dict.values()
        ],
        axis=0,
    )


    if patch_matrix.shape != (
        P,
        512,
    ):
        raise RuntimeError(
            "Patch embedding matrix shape "
            f"mismatch {patch_matrix.shape}"
        )


    with h5py.File(
        patch_file,
        "r",
    ) as f:

        h5_barcodes = [
            decode_value(x)
            for x in f[
                "barcode"
            ][:]
        ]

        h5_coords = np.asarray(
            f["coords"][:],
            dtype=np.int64,
        )


    if barcodes != h5_barcodes:
        raise RuntimeError(
            "Barcode order mismatch"
        )


    if (
        len(barcodes)
        != P
        or len(set(barcodes))
        != P
    ):
        raise RuntimeError(
            "Barcode count/uniqueness mismatch"
        )


    coords_np = (
        coords.squeeze(0)
        .cpu()
        .numpy()
        .astype(np.int64)
    )


    if not np.array_equal(
        coords_np,
        h5_coords,
    ):
        raise RuntimeError(
            "Coordinate order mismatch"
        )


    print(
        "PATCH INDEX ↔ BARCODE ↔ COORD "
        "↔ CONCH EMBEDDING: PASS"
    )


    # --------------------------------------------------------
    # Model load.
    # --------------------------------------------------------

    emb_dim = get_emb_dim(
        config
    )


    flow_path = AffineProbPath(
        scheduler=CondOTScheduler()
    )


    model = get_model(
        config=config,
        gene_dim=gene_dim,
        emb_dim=emb_dim,
        device=device,
        flow_path=flow_path,
    )


    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )


    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )


    model.to(
        device
    )

    model.eval()


    human_epoch = (
        int(
            checkpoint["epoch"]
        )
        + 1
    )


    if human_epoch != 239:
        raise RuntimeError(
            f"Expected epoch239, "
            f"got {human_epoch}"
        )


    print(
        "FROZEN RUN-2 EPOCH239 LOAD: PASS"
    )


    # --------------------------------------------------------
    # Allocate aggregate router statistics.
    # --------------------------------------------------------

    shape = (
        P,
        NUM_EXPERTS,
    )


    logit_sum = torch.zeros(
        shape,
        dtype=torch.float64,
    )

    logit_sq_sum = torch.zeros_like(
        logit_sum
    )

    prob_sum = torch.zeros_like(
        logit_sum
    )

    prob_sq_sum = torch.zeros_like(
        logit_sum
    )

    route_weight_sum = torch.zeros_like(
        logit_sum
    )

    top2_count = torch.zeros(
        shape,
        dtype=torch.int64,
    )

    top1_count = torch.zeros_like(
        top2_count
    )


    context_path = (
        tmp_dir
        / "router_contexts.csv"
    )


    context_fields = [
        "sample_id",
        "patch_index",
        "barcode",
        "coord_x",
        "coord_y",
        "base_seed",
        "slide_seed",
        "step_index",
        "t",
        "top1_expert",
        "top2_expert",
    ]


    for e in range(
        NUM_EXPERTS
    ):
        context_fields += [
            f"expert{e}_logit",
            f"expert{e}_prob",
            f"expert{e}_route_weight",
        ]


    num_contexts = 0


    # --------------------------------------------------------
    # Exact frozen trajectory + streaming context CSV.
    # --------------------------------------------------------

    with open(
        context_path,
        "w",
        newline="",
    ) as context_file:

        context_writer = csv.DictWriter(
            context_file,
            fieldnames=context_fields,
        )

        context_writer.writeheader()


        with torch.inference_mode():

            for base_seed in BASE_SEEDS:

                random.seed(
                    base_seed
                )

                np.random.seed(
                    base_seed
                )

                torch.manual_seed(
                    base_seed
                )

                torch.cuda.manual_seed_all(
                    base_seed
                )


                sseed = slide_seed(
                    base_seed,
                    sample_id,
                )


                generator = torch.Generator(
                    device="cpu"
                )

                generator.manual_seed(
                    sseed
                )


                x_init = torch.randn(
                    (
                        B,
                        P,
                        model.gene_latent_dim,
                    ),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ).to(
                    device
                )


                # Official frozen ODESolver.
                solver = ODESolver(
                    velocity_model=
                        model.get_velocity
                )


                official_final = (
                    solver.sample(
                        x_init=
                            x_init.clone(),
                        step_size=
                            STEP_SIZE,
                        method="euler",
                        time_grid=
                            torch.tensor(
                                [0.0, 1.0],
                                device=device,
                                dtype=torch.float32,
                            ),
                        return_intermediates=
                            False,
                        enable_grad=
                            False,
                        emb=
                            embeddings,
                        onco_onehot=
                            onco_onehot,
                        guidance_scale=
                            GUIDANCE_SCALE,
                    )
                )


                # Manual exact Euler trajectory
                # so the conditional router can
                # be recorded at the exact states.
                x = x_init.clone()


                for (
                    step_index,
                    t_value,
                ) in enumerate(
                    STEP_TIMES
                ):

                    t = torch.tensor(
                        [t_value],
                        device=device,
                        dtype=x.dtype,
                    )


                    # Conditional/image branch.
                    (
                        v_cond,
                        logits_cond,
                    ) = model._get_predictions(
                        z_t=x,
                        t=t,
                        emb=embeddings,
                        onco_onehot=
                            onco_onehot,
                    )


                    # Unconditional CFG branch.
                    uncond_emb = (
                        model
                        .unconditional_embedding
                        .expand(
                            B,
                            P,
                            -1,
                        )
                    )


                    (
                        v_uncond,
                        _,
                    ) = model._get_predictions(
                        z_t=x,
                        t=t,
                        emb=uncond_emb,
                        onco_onehot=
                            onco_onehot,
                    )


                    guided_velocity = (
                        v_uncond
                        + GUIDANCE_SCALE
                        * (
                            v_cond
                            - v_uncond
                        )
                    )


                    # Router statistics from
                    # CONDITIONAL branch.
                    probs = F.softmax(
                        logits_cond,
                        dim=-1,
                    )


                    (
                        top_raw,
                        top_idx,
                    ) = torch.topk(
                        logits_cond,
                        k=TOP_K_ROUTER,
                        dim=-1,
                    )


                    top_weights = F.softmax(
                        top_raw,
                        dim=-1,
                    )


                    route_weights = (
                        torch.zeros_like(
                            logits_cond
                        )
                    )


                    route_weights.scatter_(
                        1,
                        top_idx,
                        top_weights,
                    )


                    top2_mask = (
                        torch.zeros_like(
                            logits_cond,
                            dtype=torch.bool,
                        )
                    )


                    top2_mask.scatter_(
                        1,
                        top_idx,
                        True,
                    )


                    top1_mask = F.one_hot(
                        top_idx[:, 0],
                        num_classes=
                            NUM_EXPERTS,
                    ).bool()


                    logits_cpu = (
                        logits_cond
                        .double()
                        .cpu()
                    )

                    probs_cpu = (
                        probs
                        .double()
                        .cpu()
                    )

                    route_cpu = (
                        route_weights
                        .double()
                        .cpu()
                    )


                    logit_sum += (
                        logits_cpu
                    )

                    logit_sq_sum += (
                        logits_cpu ** 2
                    )

                    prob_sum += (
                        probs_cpu
                    )

                    prob_sq_sum += (
                        probs_cpu ** 2
                    )

                    route_weight_sum += (
                        route_cpu
                    )

                    top2_count += (
                        top2_mask
                        .cpu()
                        .long()
                    )

                    top1_count += (
                        top1_mask
                        .cpu()
                        .long()
                    )


                    top_idx_cpu = (
                        top_idx.cpu()
                    )


                    # Save every patch/context.
                    for patch_i in range(P):

                        row = {
                            "sample_id":
                                sample_id,
                            "patch_index":
                                patch_i,
                            "barcode":
                                barcodes[
                                    patch_i
                                ],
                            "coord_x":
                                int(
                                    h5_coords[
                                        patch_i,
                                        0
                                    ]
                                ),
                            "coord_y":
                                int(
                                    h5_coords[
                                        patch_i,
                                        1
                                    ]
                                ),
                            "base_seed":
                                base_seed,
                            "slide_seed":
                                int(
                                    sseed
                                ),
                            "step_index":
                                step_index,
                            "t":
                                t_value,
                            "top1_expert":
                                int(
                                    top_idx_cpu[
                                        patch_i,
                                        0
                                    ]
                                ),
                            "top2_expert":
                                int(
                                    top_idx_cpu[
                                        patch_i,
                                        1
                                    ]
                                ),
                        }


                        for e in range(
                            NUM_EXPERTS
                        ):

                            row[
                                f"expert{e}_logit"
                            ] = float(
                                logits_cpu[
                                    patch_i,
                                    e
                                ]
                            )

                            row[
                                f"expert{e}_prob"
                            ] = float(
                                probs_cpu[
                                    patch_i,
                                    e
                                ]
                            )

                            row[
                                f"expert{e}_route_weight"
                            ] = float(
                                route_cpu[
                                    patch_i,
                                    e
                                ]
                            )


                        context_writer.writerow(
                            row
                        )


                    num_contexts += 1


                    # Exact Euler update.
                    x = (
                        x
                        + STEP_SIZE
                        * guided_velocity
                    )


                max_abs = float(
                    (
                        x
                        - official_final
                    )
                    .abs()
                    .max()
                    .item()
                )


                print(
                    f"{sample_id} "
                    f"seed={base_seed} "
                    f"manual-vs-official "
                    f"max_abs={max_abs:.9g}"
                )


                if max_abs > 1e-5:
                    raise RuntimeError(
                        "Manual trajectory "
                        "does not match official "
                        f"solver: {max_abs}"
                    )


    if num_contexts != 10:
        raise RuntimeError(
            f"Expected 10 contexts, "
            f"got {num_contexts}"
        )


    print(
        "EXACT ROUTER-CAPTURE "
        "TRAJECTORY: PASS"
    )


    # --------------------------------------------------------
    # Aggregate router statistics.
    # --------------------------------------------------------

    mean_logits = (
        logit_sum
        / num_contexts
    )

    mean_probs = (
        prob_sum
        / num_contexts
    )

    prob_variance = (
        prob_sq_sum
        / num_contexts
        - mean_probs ** 2
    )

    prob_variance = torch.clamp(
        prob_variance,
        min=0.0,
    )

    sd_probs = torch.sqrt(
        prob_variance
    )


    logit_variance = (
        logit_sq_sum
        / num_contexts
        - mean_logits ** 2
    )

    logit_variance = torch.clamp(
        logit_variance,
        min=0.0,
    )

    sd_logits = torch.sqrt(
        logit_variance
    )


    mean_route_weights = (
        route_weight_sum
        / num_contexts
    )

    top2_frequency = (
        top2_count.double()
        / num_contexts
    )

    top1_frequency = (
        top1_count.double()
        / num_contexts
    )


    # Strong router invariants.
    prob_sum_check = (
        mean_probs.sum(
            dim=1
        )
    )

    route_sum_check = (
        mean_route_weights.sum(
            dim=1
        )
    )

    top2_sum_check = (
        top2_frequency.sum(
            dim=1
        )
    )


    if not torch.allclose(
        prob_sum_check,
        torch.ones_like(
            prob_sum_check
        ),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Router probabilities "
            "do not sum to 1"
        )


    if not torch.allclose(
        route_sum_check,
        torch.ones_like(
            route_sum_check
        ),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Mean route weights "
            "do not sum to 1"
        )


    if not torch.allclose(
        top2_sum_check,
        torch.full_like(
            top2_sum_check,
            2.0,
        ),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Top-2 frequencies "
            "do not sum to 2"
        )


    print(
        "ROUTER INVARIANTS: PASS"
    )


    # --------------------------------------------------------
    # Per-patch aggregate table.
    # --------------------------------------------------------

    aggregate_rows = []

    for patch_i in range(P):

        row = {
            "sample_id":
                sample_id,
            "patch_index":
                patch_i,
            "barcode":
                barcodes[
                    patch_i
                ],
            "coord_x":
                int(
                    h5_coords[
                        patch_i,
                        0
                    ]
                ),
            "coord_y":
                int(
                    h5_coords[
                        patch_i,
                        1
                    ]
                ),
            "n_contexts":
                num_contexts,
        }


        for e in range(
            NUM_EXPERTS
        ):

            row[
                f"expert{e}_mean_logit"
            ] = float(
                mean_logits[
                    patch_i,
                    e
                ]
            )

            row[
                f"expert{e}_sd_logit"
            ] = float(
                sd_logits[
                    patch_i,
                    e
                ]
            )

            row[
                f"expert{e}_mean_prob"
            ] = float(
                mean_probs[
                    patch_i,
                    e
                ]
            )

            row[
                f"expert{e}_sd_prob"
            ] = float(
                sd_probs[
                    patch_i,
                    e
                ]
            )

            row[
                f"expert{e}_mean_route_weight"
            ] = float(
                mean_route_weights[
                    patch_i,
                    e
                ]
            )

            row[
                f"expert{e}_top2_frequency"
            ] = float(
                top2_frequency[
                    patch_i,
                    e
                ]
            )

            row[
                f"expert{e}_top1_frequency"
            ] = float(
                top1_frequency[
                    patch_i,
                    e
                ]
            )


        aggregate_rows.append(
            row
        )


    aggregate_fields = list(
        aggregate_rows[0].keys()
    )


    write_csv(
        tmp_dir
        / "router_aggregate.csv",
        aggregate_fields,
        aggregate_rows,
    )


    # --------------------------------------------------------
    # Full affinity ranking + full natural ranking.
    # --------------------------------------------------------

    affinity_ranking_rows = []
    natural_ranking_rows = []

    top256_by_expert = {}


    for e in range(
        NUM_EXPERTS
    ):

        affinity_order = (
            torch.argsort(
                mean_probs[:, e],
                descending=True,
            )
            .cpu()
            .numpy()
            .astype(np.int64)
        )


        natural_order = sorted(
            range(P),
            key=lambda i: (
                float(
                    mean_route_weights[
                        i,
                        e
                    ]
                ),
                float(
                    top2_frequency[
                        i,
                        e
                    ]
                ),
                float(
                    mean_probs[
                        i,
                        e
                    ]
                ),
            ),
            reverse=True,
        )


        top_k_actual = min(
            TOP_K_PATCHES,
            P,
        )


        top256_by_expert[e] = (
            affinity_order[
                :top_k_actual
            ]
        )


        for rank, patch_i in enumerate(
            affinity_order,
            start=1,
        ):

            affinity_ranking_rows.append({
                "expert_id":
                    e,
                "rank":
                    rank,
                "sample_id":
                    sample_id,
                "patch_index":
                    int(
                        patch_i
                    ),
                "barcode":
                    barcodes[
                        int(
                            patch_i
                        )
                    ],
                "coord_x":
                    int(
                        h5_coords[
                            int(
                                patch_i
                            ),
                            0
                        ]
                    ),
                "coord_y":
                    int(
                        h5_coords[
                            int(
                                patch_i
                            ),
                            1
                        ]
                    ),
                "mean_logit":
                    float(
                        mean_logits[
                            patch_i,
                            e
                        ]
                    ),
                "sd_logit":
                    float(
                        sd_logits[
                            patch_i,
                            e
                        ]
                    ),
                "mean_router_prob":
                    float(
                        mean_probs[
                            patch_i,
                            e
                        ]
                    ),
                "sd_router_prob":
                    float(
                        sd_probs[
                            patch_i,
                            e
                        ]
                    ),
                "mean_route_weight":
                    float(
                        mean_route_weights[
                            patch_i,
                            e
                        ]
                    ),
                "top2_frequency":
                    float(
                        top2_frequency[
                            patch_i,
                            e
                        ]
                    ),
                "top1_frequency":
                    float(
                        top1_frequency[
                            patch_i,
                            e
                        ]
                    ),
            })


        for rank, patch_i in enumerate(
            natural_order,
            start=1,
        ):

            natural_ranking_rows.append({
                "expert_id":
                    e,
                "rank":
                    rank,
                "sample_id":
                    sample_id,
                "patch_index":
                    int(
                        patch_i
                    ),
                "barcode":
                    barcodes[
                        int(
                            patch_i
                        )
                    ],
                "coord_x":
                    int(
                        h5_coords[
                            int(
                                patch_i
                            ),
                            0
                        ]
                    ),
                "coord_y":
                    int(
                        h5_coords[
                            int(
                                patch_i
                            ),
                            1
                        ]
                    ),
                "mean_route_weight":
                    float(
                        mean_route_weights[
                            patch_i,
                            e
                        ]
                    ),
                "top2_frequency":
                    float(
                        top2_frequency[
                            patch_i,
                            e
                        ]
                    ),
                "top1_frequency":
                    float(
                        top1_frequency[
                            patch_i,
                            e
                        ]
                    ),
                "mean_router_prob":
                    float(
                        mean_probs[
                            patch_i,
                            e
                        ]
                    ),
            })


    write_csv(
        tmp_dir
        / "router_rankings_affinity.csv",
        list(
            affinity_ranking_rows[
                0
            ].keys()
        ),
        affinity_ranking_rows,
    )


    write_csv(
        tmp_dir
        / "router_rankings_natural.csv",
        list(
            natural_ranking_rows[
                0
            ].keys()
        ),
        natural_ranking_rows,
    )


    # --------------------------------------------------------
    # Top-256 table.
    # --------------------------------------------------------

    top256_rows = []


    for e in range(
        NUM_EXPERTS
    ):

        for rank, patch_i in enumerate(
            top256_by_expert[e],
            start=1,
        ):

            patch_i = int(
                patch_i
            )

            top256_rows.append({
                "expert_id":
                    e,
                "rank":
                    rank,
                "sample_id":
                    sample_id,
                "patch_index":
                    patch_i,
                "barcode":
                    barcodes[
                        patch_i
                    ],
                "coord_x":
                    int(
                        h5_coords[
                            patch_i,
                            0
                        ]
                    ),
                "coord_y":
                    int(
                        h5_coords[
                            patch_i,
                            1
                        ]
                    ),
                "affinity_score":
                    float(
                        mean_probs[
                            patch_i,
                            e
                        ]
                    ),
                "affinity_score_sd":
                    float(
                        sd_probs[
                            patch_i,
                            e
                        ]
                    ),
                "mean_logit":
                    float(
                        mean_logits[
                            patch_i,
                            e
                        ]
                    ),
                "actual_top2_frequency":
                    float(
                        top2_frequency[
                            patch_i,
                            e
                        ]
                    ),
                "actual_top1_frequency":
                    float(
                        top1_frequency[
                            patch_i,
                            e
                        ]
                    ),
                "actual_mean_route_weight":
                    float(
                        mean_route_weights[
                            patch_i,
                            e
                        ]
                    ),
            })


    write_csv(
        tmp_dir
        / "top256_affinity.csv",
        list(
            top256_rows[0].keys()
        ),
        top256_rows,
    )


    # --------------------------------------------------------
    # CONCH semantic alignment.
    # Save ALL 2450 concepts, not only Top-50.
    # --------------------------------------------------------

    (
        concept_rows,
        text_matrix,
    ) = load_concept_bank()


    normalized_patches = (
        normalize_rows(
            patch_matrix
        )
    )


    global_profile = (
        normalized_patches
        .mean(axis=0)
        .astype(np.float32)
    )


    concept_score_rows = []
    top50_rows = []
    semantic_summary_rows = []


    for e in range(
        NUM_EXPERTS
    ):

        indices = (
            top256_by_expert[e]
            .astype(np.int64)
        )


        topk_profile = (
            normalized_patches[
                indices
            ]
            .mean(axis=0)
            .astype(np.float32)
        )


        contrastive_profile = (
            topk_profile
            - global_profile
        ).astype(
            np.float32
        )


        topk_norm = float(
            np.linalg.norm(
                topk_profile
            )
        )

        contrastive_norm = float(
            np.linalg.norm(
                contrastive_profile
            )
        )


        if topk_norm == 0.0:
            raise RuntimeError(
                f"Expert {e}: "
                "zero top-k profile"
            )

        if contrastive_norm == 0.0:
            raise RuntimeError(
                f"Expert {e}: "
                "zero contrastive profile"
            )


        topk_n = (
            topk_profile
            / topk_norm
        ).astype(
            np.float32
        )


        contrastive_n = (
            contrastive_profile
            / contrastive_norm
        ).astype(
            np.float32
        )


        sim_mean = (
            topk_n
            @ text_matrix.T
        )


        sim_contrastive = (
            contrastive_n
            @ text_matrix.T
        )


        order = np.argsort(
            sim_contrastive
        )[::-1]


        rank_lookup = np.empty(
            len(
                concept_rows
            ),
            dtype=np.int64,
        )


        for rank, idx in enumerate(
            order,
            start=1,
        ):
            rank_lookup[
                int(idx)
            ] = rank


        for concept_idx, concept in enumerate(
            concept_rows
        ):

            row = {
                "sample_id":
                    sample_id,
                "expert_id":
                    e,
                "rank":
                    int(
                        rank_lookup[
                            concept_idx
                        ]
                    ),
                "concept_id":
                    concept[
                        "concept_id"
                    ],
                "concept_name":
                    concept[
                        "concept_name"
                    ],
                "tier":
                    concept[
                        "tier"
                    ],
                "embedding_set":
                    concept[
                        "embedding_set"
                    ],
                "species":
                    concept[
                        "species"
                    ],
                "similarity_topk_mean":
                    float(
                        sim_mean[
                            concept_idx
                        ]
                    ),
                "similarity_topk_contrastive":
                    float(
                        sim_contrastive[
                            concept_idx
                        ]
                    ),
                "prompt_text":
                    concept[
                        "prompt_text"
                    ],
            }


            concept_score_rows.append(
                row
            )


            if row[
                "rank"
            ] <= TOP_N_CONCEPTS:

                top50_rows.append(
                    row.copy()
                )


        top2_vals = (
            top2_frequency[
                indices,
                e
            ]
            .cpu()
            .numpy()
        )


        top1_vals = (
            top1_frequency[
                indices,
                e
            ]
            .cpu()
            .numpy()
        )


        semantic_summary_rows.append({
            "sample_id":
                sample_id,
            "expert_id":
                e,
            "n_patches":
                P,
            "top_k_patches":
                len(indices),
            "mean_route_weight_all_patches":
                float(
                    mean_route_weights[
                        :,
                        e
                    ].mean()
                ),
            "mean_router_prob_all_patches":
                float(
                    mean_probs[
                        :,
                        e
                    ].mean()
                ),
            "mean_top2_frequency_all_patches":
                float(
                    top2_frequency[
                        :,
                        e
                    ].mean()
                ),
            "mean_top1_frequency_all_patches":
                float(
                    top1_frequency[
                        :,
                        e
                    ].mean()
                ),
            "ever_top2_patches":
                int(
                    (
                        top2_frequency[
                            :,
                            e
                        ]
                        > 0
                    )
                    .sum()
                    .item()
                ),
            "always_top2_patches":
                int(
                    (
                        top2_frequency[
                            :,
                            e
                        ]
                        == 1
                    )
                    .sum()
                    .item()
                ),
            "top256_mean_top2_frequency":
                float(
                    np.mean(
                        top2_vals
                    )
                ),
            "top256_mean_top1_frequency":
                float(
                    np.mean(
                        top1_vals
                    )
                ),
            "top256_ever_top2":
                int(
                    np.sum(
                        top2_vals
                        > 0
                    )
                ),
            "top256_always_top2":
                int(
                    np.sum(
                        top2_vals
                        == 1
                    )
                ),
            "topk_profile_norm":
                topk_norm,
            "contrastive_profile_norm":
                contrastive_norm,
        })


    # Sort Top-50 into expert/rank order.
    top50_rows.sort(
        key=lambda r: (
            int(
                r[
                    "expert_id"
                ]
            ),
            int(
                r[
                    "rank"
                ]
            ),
        )
    )


    write_csv(
        tmp_dir
        / "concept_scores_all2450.csv",
        list(
            concept_score_rows[
                0
            ].keys()
        ),
        concept_score_rows,
    )


    write_csv(
        tmp_dir
        / "concept_rankings_top50.csv",
        list(
            top50_rows[
                0
            ].keys()
        ),
        top50_rows,
    )


    write_csv(
        tmp_dir
        / "expert_summary.csv",
        list(
            semantic_summary_rows[
                0
            ].keys()
        ),
        semantic_summary_rows,
    )


    # --------------------------------------------------------
    # Provenance manifest.
    # --------------------------------------------------------

    manifest = {
        "sample_id":
            sample_id,
        "status":
            "PASS",
        "n_patches":
            P,
        "n_contexts":
            num_contexts,
        "base_seeds":
            list(
                BASE_SEEDS
            ),
        "solver":
            "euler",
        "num_steps":
            2,
        "step_size":
            STEP_SIZE,
        "step_times":
            list(
                STEP_TIMES
            ),
        "guidance_scale":
            GUIDANCE_SCALE,
        "router_branch":
            "conditional",
        "num_experts":
            NUM_EXPERTS,
        "router_top_k":
            TOP_K_ROUTER,
        "semantic_top_k_patches":
            TOP_K_PATCHES,
        "semantic_concepts":
            2450,
        "semantic_embedding_set":
            "visual",
        "semantic_species":
            "Homo_sapiens",
        "semantic_rank_metric":
            "similarity_topk_contrastive",
        "checkpoint_sha256":
            sha256(
                CHECKPOINT
            ),
        "config_sha256":
            sha256(
                CONFIG
            ),
        "flow_matching_sha256":
            sha256(
                RUN2
                / "models/flow_matching.py"
            ),
        "concept_bank_sha256":
            sha256(
                CONCEPT_PARQUET
            ),
        "ready_bank_sha256":
            sha256(
                READY_PARQUET
            ),
    }


    with open(
        tmp_dir
        / "manifest.json",
        "w",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )


    # --------------------------------------------------------
    # Hash all result files.
    # --------------------------------------------------------

    result_files = sorted(
        [
            p
            for p in tmp_dir.iterdir()
            if p.is_file()
            and p.name
            != "SHA256SUMS.txt"
        ],
        key=lambda p: p.name,
    )


    with open(
        tmp_dir
        / "SHA256SUMS.txt",
        "w",
    ) as f:

        for p in result_files:

            f.write(
                f"{sha256(p)}  "
                f"{p.name}\n"
            )


    with open(
        tmp_dir
        / "STATUS.txt",
        "w",
    ) as f:

        f.write(
            "PASS\n"
        )


    # --------------------------------------------------------
    # Atomic finalization.
    # --------------------------------------------------------

    if final_dir.exists():

        backup_dir = (
            OUTROOT
            / f".{sample_id}.previous"
        )

        if backup_dir.exists():
            shutil.rmtree(
                backup_dir
            )

        final_dir.rename(
            backup_dir
        )


    tmp_dir.rename(
        final_dir
    )


    print()
    print(
        "========================================"
    )

    print(
        f"{sample_id} FULL RUN-2 "
        "INTERPRETABILITY: PASS"
    )

    print(
        "========================================"
    )

    print(
        "Output:",
        final_dir,
    )


    print()
    print(
        "=== EXPERT ROUTING SUMMARY ==="
    )


    for row in semantic_summary_rows:

        print(
            f"E{row['expert_id']}: "
            f"route_weight="
            f"{row['mean_route_weight_all_patches']:.6f} "
            f"top2_freq="
            f"{row['mean_top2_frequency_all_patches']:.4f} "
            f"top256_top2="
            f"{row['top256_mean_top2_frequency']:.4f}"
        )


    print()
    print(
        "=== TOP 5 CONCEPTS ==="
    )


    for e in range(
        NUM_EXPERTS
    ):

        print()
        print(
            f"Expert {e}"
        )

        rows = [
            r
            for r in top50_rows
            if int(
                r[
                    "expert_id"
                ]
            ) == e
            and int(
                r[
                    "rank"
                ]
            ) <= 5
        ]

        for r in rows:

            print(
                f"{r['rank']:2d}. "
                f"{r['concept_name']} "
                f"({r['similarity_topk_contrastive']:+.6f})"
            )


if __name__ == "__main__":
    main()
