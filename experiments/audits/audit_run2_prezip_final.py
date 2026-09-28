#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/data/cat/ws/rzmi154h-molf-pathology/"
    "conch_molf_production"
)

RUN2 = ROOT / "run2_code"
FROZEN = ROOT / "run2_frozen"
GV2 = ROOT / "genevae_v2_frozen"
EVAL = ROOT / "run2_evaluation"

ALL23 = (
    ROOT
    / "run2_interpretability"
    / "all23"
)

AGG = ALL23 / "aggregate"

TRAIN_RUN = (
    RUN2
    / "training_log"
    / "run_20260827_212630_molf_run2"
)

SPLITS = (
    RUN2
    / "data/Hest_Bench"
    / "hest_pathway_clean_split_0"
)

HEST_META = (
    RUN2
    / "data/Hest_Bench"
    / "HEST_v1_1_0.csv"
)

CONCH = (
    RUN2
    / "data/HEST1K"
    / "conch"
)

PATCHES = ROOT / "patch_meta"
ST = ROOT / "st"

OUT = ROOT / "prezip_final_audit"


# ============================================================
# EXPECTED FROZEN HASHES
# ============================================================

EXPECTED = {
    "run2_checkpoint":
        "dbe23d4e2f670749ee746272337a21d3"
        "d14e49fc6dab34dd43bea05fb727965d",

    "run2_config":
        "ba6e43d95a6acaa9a4a4fe0b2ca3f0"
        "e4a5d5d0fa8a35c0f233a5d7f3417cfd0c",

    "genevae_checkpoint":
        "cf58dafe7e37987fb24a046676a1a04a"
        "24a2c992fb74eeef4203840d48f00cc6",

    "genevae_config":
        "3f7a1380fa1033504ca4a6f1c3ba1e5"
        "f7cad5e527addaf1578d88e5d579410d9",

    "final_protocol":
        "e3939b92785ed85c1c15d54fa37204a"
        "16a030a301badca1b4cc8189862118dfc",

    "final_test_result":
        "e8aac7e5debc4af1ac0c0139eb524571"
        "e022ae7d1772f509a55a08698bc19033",

    "posthoc_summary":
        "b7dc36d97152b68bcac12c0d4079656"
        "adb74ec0a6304129bcf181ceb904c3272",

    "flow_matching":
        "4e2191cc581466ee5b97d687abfc1c7b"
        "8a8f7802f89718e49a088d11396d8d58",

    "genevae_source":
        "963f881c01ef230644c954583abaf132b"
        "b81dda2565c6da123e470db89462a13",

    "train_source":
        "f67b3762868e62088b4f5e6e3889bcea"
        "8923e3abb4af88f98ae838b309d68119",

    "training_utils":
        "0cb6944afd4ccad43fa8f5ac55553bdd"
        "71c00f570322a4a282aad1a8b8265c32",

    "custom_dataset":
        "c849f2fb65ad9221d4a401bff5f01f65"
        "08d3185afb9f9e13986c2949e5c7fdec",

    "hest_utils":
        "dee179841684f723a0a13b9bfdeda1fd"
        "5965390211108034484334bdd3489a55",

    "final_test_evaluator":
        "6ea99e689b1088ea5b38a35937752558"
        "d8f938e95fab1a808bc92b0c88ab3af0",

    "val_evaluator":
        "7805c568c6b826aff839c0686f21d55a"
        "2169a6dfa389f6b48c1886bb9b59e085",

    "all23_runner":
        "04e74ff1669c5e892abc43a11fa23f50"
        "d8eea207ab26a6eba2676355d9b90706",

    "all23_auditor":
        "ff738274c2fd7525c8e9e7d0187d78e8"
        "e1958b44ff22f828082480badc70f68e",

    "all23_aggregator":
        "dfdcf7fbf4f0a69a6b4748797c2db9b"
        "7244997bd6d3ec8d6d9a3e931ed449caa",

    "all23_final_auditor":
        "56c5261410db9698d77e6a0ab1fd82ab"
        "97da7a27a5175f5839811d00bab3b9e6",
}


TEST_IDS = {
    "INT13", "INT14", "INT15", "INT24",
    "MEND154", "MEND156", "MEND157", "MEND158",
    "MEND159", "MEND160", "MEND161", "MEND162",
    "NCBI643", "NCBI684",
    "TENX116", "TENX117", "TENX141",
    "TENX147", "TENX148", "TENX149", "TENX99",
    "ZEN48", "ZEN49",
}

FINAL_SEEDS = [
    41001,
    42001,
    43001,
    44001,
    45001,
]


PASS = []
WARN = []


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


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def check_hash(label, path, expected):
    require(
        path.is_file(),
        f"{label}: missing {path}",
    )

    actual = sha256(path)

    require(
        actual == expected,
        f"{label}: SHA mismatch\n"
        f"expected={expected}\n"
        f"actual={actual}",
    )

    PASS.append(
        f"{label}: exact SHA"
    )


def decode_barcode(x):
    if isinstance(x, np.ndarray):
        x = x.item()

    if isinstance(x, bytes):
        return x.decode()

    if isinstance(x, np.bytes_):
        return x.tobytes().decode()

    return str(x)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# 1. FROZEN ARTIFACT HASHES
# ============================================================

print()
print("=" * 72)
print("1. FROZEN ARTIFACT PROVENANCE")
print("=" * 72)


check_hash(
    "Run2 best checkpoint",
    FROZEN / "run2_best_epoch239.pt",
    EXPECTED["run2_checkpoint"],
)

check_hash(
    "Run2 frozen config",
    FROZEN / "config.yml",
    EXPECTED["run2_config"],
)

check_hash(
    "GeneVAE-v2 checkpoint",
    GV2 / "genevae_v2_best_epoch929.pt",
    EXPECTED["genevae_checkpoint"],
)

check_hash(
    "GeneVAE-v2 config",
    GV2 / "config.yml",
    EXPECTED["genevae_config"],
)

check_hash(
    "Final inference protocol",
    EVAL / "FINAL_INFERENCE_PROTOCOL.txt",
    EXPECTED["final_protocol"],
)

check_hash(
    "Final test result",
    EVAL / "FINAL_TEST_RESULT.txt",
    EXPECTED["final_test_result"],
)

check_hash(
    "Posthoc diagnostic summary",
    EVAL / "RUN2_POSTHOC_DIAGNOSTIC_SUMMARY.txt",
    EXPECTED["posthoc_summary"],
)


source_checks = {
    RUN2 / "models/flow_matching.py":
        EXPECTED["flow_matching"],

    RUN2 / "models/genevae_v2.py":
        EXPECTED["genevae_source"],

    RUN2 / "train.py":
        EXPECTED["train_source"],

    RUN2 / "utils/training_utils.py":
        EXPECTED["training_utils"],

    RUN2 / "utils/custom_dataset.py":
        EXPECTED["custom_dataset"],

    RUN2 / "utils/hest_utils.py":
        EXPECTED["hest_utils"],
}


for path, expected in source_checks.items():
    check_hash(
        f"Live source {path.name}",
        path,
        expected,
    )


check_hash(
    "Final test evaluator",
    EVAL / "run2_final_test_inference.py",
    EXPECTED["final_test_evaluator"],
)

check_hash(
    "Validation evaluator",
    EVAL / "run2_validation_inference.py",
    EXPECTED["val_evaluator"],
)


print("FROZEN HASH AUDIT: PASS")


# ============================================================
# 2. SPLIT INTEGRITY
# ============================================================

print()
print("=" * 72)
print("2. SPLIT / VOCAB / TARGET-GENE INTEGRITY")
print("=" * 72)


split_df = {}

for split, expected_n in [
    ("train", 385),
    ("val", 96),
    ("test", 23),
]:

    p = SPLITS / f"{split}_split.csv"

    require(
        p.is_file(),
        f"Missing split: {p}",
    )

    df = pd.read_csv(p)

    require(
        len(df) == expected_n,
        f"{split}: rows={len(df)} != {expected_n}",
    )

    require(
        df["sample_id"].astype(str).nunique()
        == expected_n,
        f"{split}: duplicate sample IDs",
    )

    split_df[split] = df


train_ids = set(
    split_df["train"]["sample_id"].astype(str)
)

val_ids = set(
    split_df["val"]["sample_id"].astype(str)
)

test_ids = set(
    split_df["test"]["sample_id"].astype(str)
)


require(
    not (train_ids & val_ids),
    "train/val sample overlap",
)

require(
    not (train_ids & test_ids),
    "train/test sample overlap",
)

require(
    not (val_ids & test_ids),
    "val/test sample overlap",
)

require(
    test_ids == TEST_IDS,
    "Test ID set mismatch",
)

require(
    len(
        train_ids
        | val_ids
        | test_ids
    ) == 504,
    "Expected 504 total unique sample IDs",
)

PASS.append(
    "Split IDs: 385/96/23, all sample-ID disjoint"
)


# Reconstruct the OncoTree vocabulary using TRAIN + VAL ONLY.

expected_vocab = set()

for split in ["train", "val"]:

    df = split_df[split]

    require(
        "oncotree_code_filled"
        in df.columns,
        f"{split}: missing oncotree_code_filled",
    )

    expected_vocab.update(
        df[
            "oncotree_code_filled"
        ]
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )


vocab_path = (
    ROOT
    / "metadata"
    / "run2_oncotree_code_list.txt"
)

actual_vocab = {
    x.strip()
    for x in vocab_path.read_text().splitlines()
    if x.strip()
}


require(
    actual_vocab == expected_vocab,
    "OncoTree vocab is not exactly "
    "the train+val label union",
)

require(
    len(actual_vocab) == 29,
    f"Expected 29 OncoTree labels, "
    f"got {len(actual_vocab)}",
)


test_labels = set(
    split_df["test"][
        "oncotree_code_filled"
    ]
    .fillna("Unknown")
    .astype(str)
)


require(
    test_labels <= actual_vocab,
    "Some test labels are outside "
    "the train+val vocabulary",
)

PASS.append(
    "OncoTree vocabulary reconstructed exactly from train+val only"
)


# Target genes.

with open(
    RUN2
    / "data/Hallmark/Hallmark_n_HVG50_genes.yaml"
) as f:
    genes = yaml.safe_load(f)["genes"]


require(
    len(genes) == 1386,
    f"Expected 1386 target genes, "
    f"got {len(genes)}",
)

require(
    len(set(genes)) == 1386,
    "Duplicate genes in target list",
)

PASS.append(
    "Target gene list: 1386 unique genes"
)


print("SPLIT / VOCAB / GENE AUDIT: PASS")


# ============================================================
# 3. RELATED-SPECIMEN WARNING
# ============================================================

print()
print("=" * 72)
print("3. SPLIT RELATEDNESS CHECK")
print("=" * 72)


meta = pd.read_csv(HEST_META)

all_split_rows = []

for split in ["train", "val", "test"]:

    d = split_df[split][
        ["sample_id"]
    ].copy()

    d["split"] = split

    all_split_rows.append(d)


split_map = pd.concat(
    all_split_rows,
    ignore_index=True,
)

joined = split_map.merge(
    meta,
    left_on="sample_id",
    right_on="id",
    how="left",
)


require(
    joined["id"].notna().all(),
    "Some split IDs missing HEST metadata",
)


related = []


for _, test_row in joined[
    joined["split"] == "test"
].iterrows():

    patient = test_row.get("patient")
    study = test_row.get("dataset_title")

    if pd.isna(patient) or pd.isna(study):
        continue

    candidates = joined[
        (joined["split"] != "test")
        & (joined["patient"] == patient)
        & (joined["dataset_title"] == study)
    ]

    for _, other in candidates.iterrows():

        related.append({
            "test_sample":
                test_row["sample_id"],
            "other_split":
                other["split"],
            "other_sample":
                other["sample_id"],
            "patient":
                patient,
            "study":
                study,
            "test_subseries":
                str(
                    test_row.get(
                        "subseries",
                        ""
                    )
                ),
            "other_subseries":
                str(
                    other.get(
                        "subseries",
                        ""
                    )
                ),
            "test_technology":
                str(
                    test_row.get(
                        "st_technology",
                        ""
                    )
                ),
            "other_technology":
                str(
                    other.get(
                        "st_technology",
                        ""
                    )
                ),
        })


if related:

    WARN.append(
        "The official split is sample-ID disjoint "
        "but not fully patient/study/specimen independent."
    )

    related_csv = (
        OUT
        / "RELATED_TEST_TRAIN_VAL_SPECIMENS.csv"
    )

    with open(
        related_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                related[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            related
        )


    print(
        f"WARNING: found "
        f"{len(related)} test↔train/val "
        "same-study+patient metadata relationships."
    )

    print(
        "This does NOT create sample-ID leakage, "
        "but the test set must be described as "
        "'held-out slides', not patient-independent."
    )

else:

    print(
        "No same-study+patient relationships found."
    )


# ============================================================
# 4. ALL 504 INPUT ARTIFACTS
# ============================================================

print()
print("=" * 72)
print("4. FULL 504-SLIDE CONCH / PATCH INPUT AUDIT")
print("=" * 72)


all_ids = []

for split in ["train", "val", "test"]:
    for sid in split_df[split][
        "sample_id"
    ].astype(str):
        all_ids.append(
            (split, sid)
        )


require(
    len(all_ids) == 504,
    "Expected 504 slide records",
)


total_patches = 0


for index, (split, sid) in enumerate(
    all_ids,
    start=1,
):

    emb_path = (
        CONCH
        / f"{sid}_embeddings.pt"
    )

    h5_path = (
        PATCHES
        / f"{sid}.h5"
    )

    st_path = (
        ST
        / f"{sid}.h5ad"
    )


    require(
        emb_path.is_file(),
        f"{sid}: embedding missing",
    )

    require(
        h5_path.is_file(),
        f"{sid}: patch H5 missing",
    )

    require(
        st_path.is_file(),
        f"{sid}: H5AD missing",
    )


    emb = torch.load(
        emb_path,
        map_location="cpu",
        weights_only=True,
    )


    require(
        isinstance(emb, dict),
        f"{sid}: embedding file is not dict",
    )


    keys = [
        decode_barcode(k)
        for k in emb.keys()
    ]

    require(
        len(keys) > 0,
        f"{sid}: no patches",
    )

    require(
        len(keys) == len(set(keys)),
        f"{sid}: duplicate embedding barcodes",
    )


    values = list(
        emb.values()
    )


    for v in values:

        require(
            torch.is_tensor(v),
            f"{sid}: non-tensor embedding",
        )

        require(
            tuple(v.shape) == (512,),
            f"{sid}: embedding shape "
            f"{tuple(v.shape)}",
        )

        require(
            v.dtype == torch.float32,
            f"{sid}: embedding dtype "
            f"{v.dtype}",
        )

        require(
            torch.isfinite(v).all().item(),
            f"{sid}: non-finite CONCH embedding",
        )


    mat = torch.stack(
        values
    ).float()


    norms = mat.norm(
        dim=1
    )


    require(
        torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1e-3,
            rtol=1e-3,
        ),
        f"{sid}: CONCH embeddings "
        "not unit normalized",
    )


    with h5py.File(
        h5_path,
        "r",
    ) as h5:

        require(
            "barcode" in h5,
            f"{sid}: H5 barcode missing",
        )

        require(
            "coords" in h5,
            f"{sid}: H5 coords missing",
        )

        h5_keys = [
            decode_barcode(x)
            for x in h5[
                "barcode"
            ][:]
        ]

        coords = h5[
            "coords"
        ]


        require(
            len(h5_keys)
            == len(keys),
            f"{sid}: H5/embedding "
            "patch-count mismatch",
        )

        require(
            len(coords)
            == len(keys),
            f"{sid}: coords/embedding "
            "patch-count mismatch",
        )

        require(
            h5_keys == keys,
            f"{sid}: H5 barcode order "
            "!= CONCH dict order",
        )


    total_patches += len(keys)


    if (
        index == 1
        or index % 50 == 0
        or index == 504
    ):

        print(
            f"[{index:03d}/504] "
            f"{sid:10s} "
            f"patches={len(keys):6d} PASS",
            flush=True,
        )


require(
    total_patches == 1_076_882,
    f"Expected 1,076,882 total patches, "
    f"got {total_patches}",
)


integrity_tsv = (
    ROOT
    / "metadata/final_504_integrity.tsv"
)

integrity = pd.read_csv(
    integrity_tsv,
    sep="\t",
)


require(
    len(integrity) == 504,
    "final_504_integrity.tsv row count",
)

require(
    set(
        integrity["status"].astype(str)
    ) == {"PASS"},
    "Stored 504 integrity manifest "
    "contains failures",
)

require(
    set(
        integrity["sample_id"].astype(str)
    )
    == {
        sid
        for _, sid in all_ids
    },
    "Stored integrity manifest "
    "sample set mismatch",
)


PASS.append(
    "All 504 CONCH+patch inputs independently "
    "re-opened and validated"
)


print(
    f"FULL INPUT AUDIT: PASS "
    f"(504 slides, {total_patches:,} patches)"
)


# ============================================================
# 5. GENEVAE-v2 + FINAL MOLF CHECKPOINT
# ============================================================

print()
print("=" * 72)
print("5. GENEVAE-v2 / MOLF CHECKPOINT AUDIT")
print("=" * 72)


with open(
    GV2 / "config.yml"
) as f:
    gv_cfg = yaml.safe_load(f)


require(
    gv_cfg["model"][
        "positional_encoding"
    ] is False,
    "GeneVAE-v2 positional_encoding "
    "must be false",
)


gv_source = (
    GV2 / "genevae_v2.py"
).read_text()


require(
    "self.pos_encoder" not in gv_source,
    "GeneVAE-v2 source unexpectedly "
    "contains self.pos_encoder",
)


acceptance = json.loads(
    (
        GV2
        / "final_acceptance.json"
    ).read_text()
)


require(
    acceptance["status"] == "PASS",
    "GeneVAE-v2 acceptance != PASS",
)

require(
    acceptance["test_used"] is False,
    "GeneVAE-v2 acceptance used test",
)

require(
    int(
        acceptance[
            "checkpoint_epoch"
        ]
    ) == 929,
    "GeneVAE-v2 epoch mismatch",
)

require(
    float(
        acceptance[
            "permutation_max_rel_rms"
        ]
    ) < 1e-5,
    "GeneVAE-v2 permutation RMS too high",
)

require(
    float(
        acceptance[
            "permutation_min_patch_cosine"
        ]
    ) > 0.99999,
    "GeneVAE-v2 permutation cosine too low",
)


gv_ckpt = torch.load(
    GV2
    / "genevae_v2_best_epoch929.pt",
    map_location="cpu",
    weights_only=False,
)


require(
    gv_ckpt.get("format")
    == "GeneVAE-v2",
    "Wrong GeneVAE-v2 format",
)

require(
    int(gv_ckpt["epoch"]) + 1
    == 929,
    "GeneVAE-v2 checkpoint epoch mismatch",
)


for name, tensor in gv_ckpt[
    "model_state"
].items():

    require(
        torch.isfinite(
            tensor
        ).all().item(),
        f"GeneVAE tensor non-finite: {name}",
    )


molf_ckpt = torch.load(
    FROZEN
    / "run2_best_epoch239.pt",
    map_location="cpu",
    weights_only=False,
)


require(
    int(molf_ckpt["epoch"]) + 1
    == 239,
    "MoLF checkpoint human epoch !=239",
)

require(
    int(molf_ckpt["best_epoch"])
    == 239,
    "MoLF best_epoch !=239",
)

require(
    abs(
        float(
            molf_ckpt[
                "best_val_loss"
            ]
        )
        - 1.0135788025644918
    ) < 1e-12,
    "MoLF best val loss mismatch",
)


molf_state = molf_ckpt[
    "model_state"
]


for name, tensor in molf_state.items():

    if torch.is_tensor(tensor):

        require(
            torch.isfinite(
                tensor
            ).all().item(),
            f"MoLF tensor non-finite: {name}",
        )


# Exact frozen GeneVAE identity inside MoLF.

gv_state = gv_ckpt[
    "model_state"
]


embedded_keys = {
    name[
        len("gene_autoencoder."):
    ]:
        tensor
    for name, tensor
    in molf_state.items()
    if name.startswith(
        "gene_autoencoder."
    )
}


require(
    set(embedded_keys)
    == set(gv_state),
    "Embedded GeneVAE state-key mismatch",
)


max_diff = 0.0


for name in gv_state:

    a = embedded_keys[name]
    b = gv_state[name]

    require(
        torch.equal(a, b),
        f"Embedded GeneVAE differs: {name}",
    )

    max_diff = max(
        max_diff,
        float(
            (a - b)
            .abs()
            .max()
            .item()
        ),
    )


require(
    len(gv_state) == 28,
    f"Expected 28 GeneVAE tensors, "
    f"got {len(gv_state)}",
)


# Six experts present.

for e in range(6):

    prefix = (
        f"velocity_predictor."
        f"experts.{e}."
    )

    require(
        any(
            k.startswith(prefix)
            for k in molf_state
        ),
        f"Expert {e} missing",
    )


require(
    any(
        "velocity_predictor.gating.linear.weight"
        in k
        for k in molf_state
    ),
    "MoE gate missing",
)


with open(
    FROZEN / "config.yml"
) as f:
    run_cfg = yaml.safe_load(f)


require(
    run_cfg[
        "spatial_encoding"
    ][
        "use_sequence_positional_encoding"
    ] is False,
    "Run2 sequence PE must be false",
)


require(
    not any(
        "spatial_pos_encoder"
        in k
        for k in molf_state
    ),
    "Unexpected spatial positional "
    "encoder parameters in checkpoint",
)


PASS.append(
    "GeneVAE-v2 exactly embedded in final MoLF "
    "(28/28 tensors exact)"
)

PASS.append(
    "Final MoLF checkpoint finite and contains all 6 experts"
)


print(
    "GENEVAE-v2 / CHECKPOINT AUDIT: PASS"
)

print(
    f"Embedded GeneVAE max abs diff: "
    f"{max_diff:.3g}"
)


# ============================================================
# 6. TRAINING HISTORY
# ============================================================

print()
print("=" * 72)
print("6. TRAINING HISTORY AUDIT")
print("=" * 72)


log_path = (
    TRAIN_RUN
    / "training_run.txt"
)


require(
    log_path.is_file(),
    "Training log missing",
)


log_text = log_path.read_text(
    errors="replace"
)


bad_patterns = [
    "Traceback (most recent call last)",
    "Training failed at epoch=",
    "non-finite loss=",
    "non-finite gradient norm=",
    "BAD BATCH DETECTED",
]


for pattern in bad_patterns:

    require(
        pattern not in log_text,
        f"Training log contains: {pattern}",
    )


epoch_re = re.compile(
    r"\[INFO_TRAIN\] Epoch "
    r"(\d+): train=([0-9.eE+-]+), "
    r"val=([0-9.eE+-]+)"
)


epochs = [
    (
        int(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
    )
    for m in epoch_re.finditer(
        log_text
    )
]


require(
    len(epochs) == 319,
    f"Expected 319 epoch summaries, "
    f"got {len(epochs)}",
)

require(
    [
        x[0]
        for x in epochs
    ] == list(
        range(1, 320)
    ),
    "Epoch summaries are not contiguous 1..319",
)


# Independently replay significant-improvement logic.

best = float("inf")
best_epoch = None
min_delta = 1e-4


for epoch, _, val in epochs:

    if val < (
        best - min_delta
    ):

        best = val
        best_epoch = epoch


require(
    best_epoch == 239,
    f"Replayed early-stop selection "
    f"gives epoch {best_epoch}, not 239",
)


require(
    "Early stopping triggered at epoch 319."
    in log_text,
    "Expected early-stop marker missing",
)


bad_dir = (
    TRAIN_RUN
    / "bad_batches"
)


if bad_dir.exists():

    require(
        not any(
            bad_dir.iterdir()
        ),
        "bad_batches directory is not empty",
    )


PASS.append(
    "Training epochs 1..319 contiguous; "
    "epoch239 selection independently reproduced"
)


print(
    "TRAINING HISTORY AUDIT: PASS"
)


# ============================================================
# 7. FINAL TEST RAW-RESULT RECONSTRUCTION
# ============================================================

print()
print("=" * 72)
print("7. FINAL TEST RESULT RECONSTRUCTION")
print("=" * 72)


metric_keys = [
    "masked_mse_micro",
    "masked_rmse_micro",
    "masked_mae_micro",
    "mean_slide_mean_gene_pcc",
    "median_slide_mean_gene_pcc",
    "mean_gene_mean_slide_pcc",
]


seed_summaries = []


for seed in FINAL_SEEDS:

    d = (
        EVAL
        / "results"
        / (
            "final_test_euler2_"
            f"cfg1.5_seed{seed}"
        )
    )


    summary_path = (
        d / "summary.json"
    )

    slide_path = (
        d / "per_slide_metrics.csv"
    )


    require(
        summary_path.is_file(),
        f"Seed {seed}: summary missing",
    )

    require(
        slide_path.is_file(),
        f"Seed {seed}: per-slide CSV missing",
    )


    summary = json.loads(
        summary_path.read_text()
    )


    require(
        summary[
            "evaluation_split"
        ] == "test",
        f"Seed {seed}: wrong split",
    )

    require(
        int(
            summary[
                "n_slides_evaluated"
            ]
        ) == 23,
        f"Seed {seed}: n slides !=23",
    )

    require(
        summary["method"] == "euler",
        f"Seed {seed}: wrong method",
    )

    require(
        int(summary["num_steps"])
        == 2,
        f"Seed {seed}: wrong steps",
    )

    require(
        abs(
            float(
                summary[
                    "guidance_scale"
                ]
            )
            - 1.5
        ) < 1e-12,
        f"Seed {seed}: wrong CFG",
    )

    require(
        int(summary["noise_seed"])
        == seed,
        f"Seed {seed}: seed mismatch",
    )

    require(
        summary[
            "checkpoint_sha256"
        ]
        == EXPECTED[
            "run2_checkpoint"
        ],
        f"Seed {seed}: checkpoint SHA mismatch",
    )

    require(
        summary["config_sha256"]
        == EXPECTED["run2_config"],
        f"Seed {seed}: config SHA mismatch",
    )

    require(
        summary[
            "final_protocol_sha256"
        ]
        == EXPECTED[
            "final_protocol"
        ],
        f"Seed {seed}: protocol SHA mismatch",
    )


    slide_rows = read_csv(
        slide_path
    )


    ids = [
        r["sample_id"]
        for r in slide_rows
    ]


    require(
        len(ids) == 23,
        f"Seed {seed}: per-slide rows !=23",
    )

    require(
        len(ids) == len(set(ids)),
        f"Seed {seed}: duplicate slide IDs",
    )

    require(
        set(ids) == TEST_IDS,
        f"Seed {seed}: test ID mismatch",
    )


    for key in metric_keys:

        require(
            math.isfinite(
                float(summary[key])
            ),
            f"Seed {seed}: "
            f"nonfinite {key}",
        )


    seed_summaries.append(
        summary
    )


means = {}
sds = {}


for key in metric_keys:

    values = np.asarray(
        [
            float(s[key])
            for s in seed_summaries
        ],
        dtype=np.float64,
    )

    means[key] = float(
        values.mean()
    )

    sds[key] = float(
        values.std(
            ddof=1
        )
    )


expected_report = {
    "masked_mse_micro":
        (0.369907, 0.000198),

    "masked_rmse_micro":
        (0.608200, 0.000162),

    "masked_mae_micro":
        (0.367717, 0.000088),

    "mean_slide_mean_gene_pcc":
        (0.246431, 0.000409),

    "median_slide_mean_gene_pcc":
        (0.184601, 0.001584),

    "mean_gene_mean_slide_pcc":
        (0.166239, 0.000526),
}


for key, (
    report_mean,
    report_sd,
) in expected_report.items():

    require(
        abs(
            means[key]
            - report_mean
        ) < 5e-7,
        f"{key}: reported mean mismatch "
        f"{means[key]} vs {report_mean}",
    )

    require(
        abs(
            sds[key]
            - report_sd
        ) < 5e-7,
        f"{key}: reported SD mismatch "
        f"{sds[key]} vs {report_sd}",
    )


# Verify all-5-seed per-slide aggregate.

agg_slide = pd.read_csv(
    EVAL
    / "FINAL_TEST_PER_SLIDE_5SEED.csv"
)


require(
    len(agg_slide) == 23,
    "FINAL_TEST_PER_SLIDE_5SEED "
    "must have 23 rows",
)

require(
    set(
        agg_slide[
            "sample_id"
        ].astype(str)
    ) == TEST_IDS,
    "Final per-slide aggregate ID mismatch",
)


PASS.append(
    "Final 5-seed test result independently "
    "reconstructed from raw seed summaries"
)


print(
    "FINAL TEST RECONSTRUCTION: PASS"
)

for key in metric_keys:
    print(
        f"{key:32s} "
        f"{means[key]:.9f} ± "
        f"{sds[key]:.9f}"
    )


# ============================================================
# 8. FULL-DATA POST-HOC CHECKS
# ============================================================

print()
print("=" * 72)
print("8. POST-HOC FULL-DATA AUDIT")
print("=" * 72)


target = pd.read_csv(
    EVAL
    / "TARGET_DISTRIBUTION_PER_SLIDE.csv"
)


gv_recon = pd.read_csv(
    EVAL
    / "GENEVAE_V2_RECON_PER_SLIDE.csv"
)


require(
    len(target) == 504,
    f"Target distribution rows "
    f"{len(target)} !=504",
)

require(
    len(gv_recon) == 504,
    f"GeneVAE reconstruction rows "
    f"{len(gv_recon)} !=504",
)


require(
    set(
        target["sample_id"].astype(str)
    )
    == {
        sid
        for _, sid in all_ids
    },
    "Target audit sample set mismatch",
)


require(
    set(
        gv_recon["sample_id"].astype(str)
    )
    == {
        sid
        for _, sid in all_ids
    },
    "GeneVAE recon sample set mismatch",
)


PASS.append(
    "Target-distribution audit: all 504 slides"
)

PASS.append(
    "GeneVAE reconstruction audit: all 504 slides"
)


print(
    "FULL-DATA POST-HOC AUDIT: PASS"
)


# ============================================================
# 9. FULL 23-SLIDE INTERPRETABILITY
# ============================================================

print()
print("=" * 72)
print("9. ALL-23 NATURAL-ROUTING INTERPRETABILITY")
print("=" * 72)


check_hash(
    "All23 runner",
    ALL23
    / "run_all23_slide.py",
    EXPECTED["all23_runner"],
)

check_hash(
    "All23 per-slide auditor",
    ALL23
    / "audit_all23_outputs.py",
    EXPECTED["all23_auditor"],
)

check_hash(
    "All23 aggregator",
    ALL23
    / "aggregate_all23.py",
    EXPECTED["all23_aggregator"],
)

check_hash(
    "All23 final auditor",
    ALL23
    / "audit_final_all23_interpretability.py",
    EXPECTED["all23_final_auditor"],
)


status_file = (
    AGG
    / "FINAL_NATURAL_ROUTING_INTERPRETABILITY_STATUS.txt"
)


require(
    status_file.is_file(),
    "Final interpretability status missing",
)

require(
    "STATUS: PASS"
    in status_file.read_text(),
    "Final interpretability status != PASS",
)


for sid in sorted(
    TEST_IDS
):

    d = ALL23 / sid

    require(
        (d / "STATUS.txt")
        .read_text()
        .strip()
        == "PASS",
        f"{sid}: interpretability status != PASS",
    )


    manifest = json.loads(
        (
            d
            / "manifest.json"
        ).read_text()
    )


    require(
        manifest[
            "checkpoint_sha256"
        ]
        == EXPECTED[
            "run2_checkpoint"
        ],
        f"{sid}: interpretability "
        "checkpoint mismatch",
    )

    require(
        manifest[
            "config_sha256"
        ]
        == EXPECTED[
            "run2_config"
        ],
        f"{sid}: interpretability "
        "config mismatch",
    )

    require(
        int(
            manifest[
                "n_contexts"
            ]
        ) == 10,
        f"{sid}: contexts !=10",
    )

    require(
        int(
            manifest[
                "semantic_concepts"
            ]
        ) == 2450,
        f"{sid}: concepts !=2450",
    )

    require(
        manifest[
            "router_branch"
        ] == "conditional",
        f"{sid}: router branch "
        "not conditional",
    )


required_agg = [
    "routing_by_slide.csv",
    "routing_global_summary.csv",
    "routing_by_technology.csv",
    "semantic_global_all2450.csv",
    "semantic_global_top50.csv",
    "semantic_route_weighted_top50.csv",
    "semantic_slide_pair_stability.csv",
    "semantic_stability_summary.csv",
    "semantic_expert_pair_similarity.csv",
    "semantic_by_technology_all2450.csv",
    "semantic_by_technology_top25.csv",
    "ALL23_AGGREGATE_SUMMARY.txt",
    "FINAL_INTERPRETABILITY_SHA256SUMS.txt",
]


for name in required_agg:

    require(
        (AGG / name).is_file(),
        f"Missing aggregate output: {name}",
    )


figure_stems = [
    "run2_all23_routing_heatmap",
    "run2_all23_routing_by_technology",
    "run2_all23_semantic_stability",
    "run2_all23_expert_semantic_similarity",
    "run2_all23_semantic_macro_heatmap",
    "run2_all23_semantic_route_weighted_heatmap",
]


for stem in figure_stems:

    require(
        (
            AGG
            / "figures"
            / f"{stem}.png"
        ).is_file(),
        f"Missing figure {stem}.png",
    )

    require(
        (
            AGG
            / "figures"
            / f"{stem}.pdf"
        ).is_file(),
        f"Missing figure {stem}.pdf",
    )


routing = pd.read_csv(
    AGG
    / "routing_global_summary.csv"
)


require(
    len(routing) == 6,
    "Global routing rows !=6",
)


require(
    abs(
        routing[
            "slide_macro_mean_route_weight"
        ].astype(float).sum()
        - 1.0
    ) < 1e-5,
    "Macro routing does not sum to 1",
)


semantic_all = pd.read_csv(
    AGG
    / "semantic_global_all2450.csv"
)


require(
    len(semantic_all)
    == 6 * 2450,
    "Global semantic rows !=14700",
)


PASS.append(
    "Natural-routing interpretation complete: "
    "23/23 slides, 92,507 held-out patches"
)


WARN.append(
    "Interpretability router = conditional/image branch "
    "evaluated on states traversed by the frozen CFG trajectory; "
    "it is not a unique 'full-CFG router'."
)

WARN.append(
    "Top-256 semantic affinity is not the same as actual Top-2 "
    "utilization for under-routed experts; both were recorded separately."
)

WARN.append(
    "Visium/Xenium comparisons are descriptive and confounded "
    "with tissue/study/specimen composition; do not claim causality."
)


print(
    "ALL-23 INTERPRETABILITY AUDIT: PASS"
)


# ============================================================
# 10. STATIC EXECUTION-PATH CAVEATS
# ============================================================

print()
print("=" * 72)
print("10. CODE-SCOPE / DEAD-CODE NOTES")
print("=" * 72)


custom_source = (
    RUN2
    / "utils/custom_dataset.py"
).read_text()


hest_source = (
    RUN2
    / "utils/hest_utils.py"
).read_text()


# These are known legacy helpers and were not used by Run2.
require(
    "class HESTGraphDataset"
    in custom_source,
    "HESTGraphDataset source unexpectedly absent",
)

require(
    "def save_hdf5"
    in hest_source,
    "save_hdf5 source unexpectedly absent",
)


WARN.append(
    "Unused legacy HESTGraphDataset has a stale tuple-unpacking "
    "interface; it is never referenced by the Run2 training/evaluation path."
)

WARN.append(
    "Unused legacy save_hdf5 append branch contains a stale typo; "
    "it is never called by Run2."
)

WARN.append(
    "Run2 'use_spatial_encoding' enables permutation-equivariant "
    "patch-context attention; real x/y coordinates are not model inputs."
)

WARN.append(
    "CFG unconditional branch is image-unconditional but remains "
    "OncoTree-conditioned."
)

WARN.append(
    "normalize_adata() docstring mentions total-count normalization, "
    "but executed implementation performs log1p only; "
    "Run2/GeneVAE configuration is consistent with log1p-only preprocessing."
)


print(
    "STATIC EXECUTION-PATH REVIEW: PASS WITH DOCUMENTED CAVEATS"
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("FINAL VERDICT")
print("=" * 72)


report_lines = []

report_lines.append(
    "DECODINGMOLF CONCH-MOLF RUN-2 PRE-ZIP FORENSIC AUDIT"
)

report_lines.append(
    "=" * 58
)

report_lines.append("")
report_lines.append(
    "STATUS: PASS WITH DOCUMENTED SCIENTIFIC / LEGACY-CODE CAVEATS"
)

report_lines.append("")

report_lines.append(
    "No invalidating implementation defect was found in the "
    "executed Run-2 training, frozen inference, evaluation, "
    "or all-23 natural-routing interpretation path."
)

report_lines.append("")

report_lines.append("PASS CHECKS")
report_lines.append("-" * 30)

for item in PASS:
    report_lines.append(
        f"- {item}"
    )

report_lines.append("")
report_lines.append("CAVEATS / WORDING REQUIREMENTS")
report_lines.append("-" * 30)

for item in WARN:
    report_lines.append(
        f"- {item}"
    )

report_lines.append("")
report_lines.append(
    "FINAL CLAIM SCOPE"
)

report_lines.append("-" * 30)

report_lines.append(
    "- Legitimate claim: final performance on 23 held-out slide IDs."
)

report_lines.append(
    "- Do NOT claim patient-independent or specimen-independent generalization."
)

report_lines.append(
    "- Semantic concept alignment is descriptive/correlative, "
    "not causal proof of expert function."
)

report_lines.append(
    "- Test-set interpretation is post-hoc analysis of the frozen Run2 model; "
    "it must not be used to retune Run2."
)

report_lines.append(
    "- If these held-out-test observations guide a future model, "
    "that future model needs a fresh external/new held-out evaluation."
)


report = "\n".join(
    report_lines
) + "\n"


report_path = (
    OUT
    / "FINAL_RUN2_PREZIP_FORENSIC_AUDIT.txt"
)


report_path.write_text(
    report
)


json_path = (
    OUT
    / "FINAL_RUN2_PREZIP_FORENSIC_AUDIT.json"
)


json_path.write_text(
    json.dumps(
        {
            "status":
                "PASS_WITH_DOCUMENTED_CAVEATS",

            "pass_checks":
                PASS,

            "caveats":
                WARN,

            "split_counts": {
                "train": 385,
                "val": 96,
                "test": 23,
                "total_unique": 504,
            },

            "total_conch_patches":
                total_patches,

            "run2_checkpoint_sha256":
                EXPECTED[
                    "run2_checkpoint"
                ],

            "genevae_v2_checkpoint_sha256":
                EXPECTED[
                    "genevae_checkpoint"
                ],

            "final_protocol_sha256":
                EXPECTED[
                    "final_protocol"
                ],

            "final_test_result_sha256":
                EXPECTED[
                    "final_test_result"
                ],
        },
        indent=2,
    )
)


hash_path = (
    OUT
    / "FINAL_RUN2_PREZIP_AUDIT_SHA256SUMS.txt"
)


with open(
    hash_path,
    "w",
) as f:

    for p in [
        report_path,
        json_path,
    ]:

        f.write(
            f"{sha256(p)}  "
            f"{p.name}\n"
        )


print(report)

print(
    "REPORT:",
    report_path,
)

print(
    "JSON:",
    json_path,
)

print(
    "HASHES:",
    hash_path,
)

print()
print(
    "FINAL RUN2 PRE-ZIP FORENSIC AUDIT: PASS WITH DOCUMENTED CAVEATS"
)
