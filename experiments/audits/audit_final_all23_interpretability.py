#!/usr/bin/env python3

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(
    "/data/cat/ws/rzmi154h-molf-pathology/"
    "conch_molf_production/run2_interpretability/all23"
)

AGG = ROOT / "aggregate"
FIGS = AGG / "figures"

SLIDES = [
    "INT13","INT14","INT15","INT24",
    "MEND154","MEND156","MEND157","MEND158",
    "MEND159","MEND160","MEND161","MEND162",
    "NCBI643","NCBI684",
    "TENX116","TENX117","TENX141",
    "TENX147","TENX148","TENX149","TENX99",
    "ZEN48","ZEN49",
]

PER_SLIDE_REQUIRED = [
    "STATUS.txt",
    "router_contexts.csv",
    "router_aggregate.csv",
    "router_rankings_affinity.csv",
    "router_rankings_natural.csv",
    "top256_affinity.csv",
    "concept_scores_all2450.csv",
    "concept_rankings_top50.csv",
    "expert_summary.csv",
    "manifest.json",
    "SHA256SUMS.txt",
]

AGG_REQUIRED = [
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
    "SHA256SUMS.txt",
]

FIGURE_STEMS = [
    "run2_all23_routing_heatmap",
    "run2_all23_routing_by_technology",
    "run2_all23_semantic_stability",
    "run2_all23_expert_semantic_similarity",
    "run2_all23_semantic_macro_heatmap",
    "run2_all23_semantic_route_weighted_heatmap",
]


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


print("=" * 64)
print("FINAL RUN-2 ALL-23 NATURAL-ROUTING INTERPRETABILITY AUDIT")
print("=" * 64)
print()


# ============================================================
# 1. PER-SLIDE COMPLETENESS
# ============================================================

total_patches = 0

for sid in SLIDES:
    d = ROOT / sid

    require(
        d.is_dir(),
        f"{sid}: directory missing",
    )

    for name in PER_SLIDE_REQUIRED:
        require(
            (d / name).is_file(),
            f"{sid}: missing {name}",
        )

    require(
        (d / "STATUS.txt").read_text().strip()
        == "PASS",
        f"{sid}: STATUS != PASS",
    )

    manifest = json.loads(
        (d / "manifest.json").read_text()
    )

    require(
        manifest["sample_id"] == sid,
        f"{sid}: manifest mismatch",
    )

    require(
        int(manifest["n_contexts"]) == 10,
        f"{sid}: expected 10 router contexts",
    )

    require(
        int(manifest["num_experts"]) == 6,
        f"{sid}: expected 6 experts",
    )

    require(
        int(manifest["semantic_concepts"]) == 2450,
        f"{sid}: expected 2450 concepts",
    )

    total_patches += int(
        manifest["n_patches"]
    )


require(
    total_patches == 92507,
    f"Total patches {total_patches} != 92507",
)

print(
    f"PER-SLIDE OUTPUTS: PASS "
    f"(23/23 slides, {total_patches} patches)"
)


# ============================================================
# 2. AGGREGATE FILES
# ============================================================

for name in AGG_REQUIRED:
    p = AGG / name

    require(
        p.is_file(),
        f"Missing aggregate file: {name}",
    )

    require(
        p.stat().st_size > 0,
        f"Empty aggregate file: {name}",
    )

print("AGGREGATE FILE SET: PASS")


# ============================================================
# 3. AGGREGATE ROW COUNTS
# ============================================================

checks = {
    "routing_by_slide.csv":
        23 * 6,

    "routing_global_summary.csv":
        6,

    "routing_by_technology.csv":
        2 * 6,

    "semantic_global_all2450.csv":
        2450 * 6,

    "semantic_global_top50.csv":
        50 * 6,

    "semantic_route_weighted_top50.csv":
        50 * 6,

    "semantic_slide_pair_stability.csv":
        6 * 253,

    "semantic_expert_pair_similarity.csv":
        15,

    "semantic_by_technology_all2450.csv":
        2 * 6 * 2450,

    "semantic_by_technology_top25.csv":
        2 * 6 * 25,
}

for name, expected in checks.items():

    rows = read_csv(
        AGG / name
    )

    require(
        len(rows) == expected,
        f"{name}: rows={len(rows)} "
        f"expected={expected}",
    )

print("AGGREGATE ROW COUNTS: PASS")


# ============================================================
# 4. ROUTING INVARIANTS
# ============================================================

rows = read_csv(
    AGG / "routing_global_summary.csv"
)

require(
    len(rows) == 6,
    "routing_global_summary != 6 rows",
)

macro_sum = sum(
    float(
        r[
            "slide_macro_mean_route_weight"
        ]
    )
    for r in rows
)

weighted_sum = sum(
    float(
        r[
            "patch_weighted_route_weight"
        ]
    )
    for r in rows
)

require(
    abs(macro_sum - 1.0) < 1e-5,
    f"Macro route sum={macro_sum}",
)

require(
    abs(weighted_sum - 1.0) < 1e-5,
    f"Patch-weighted route sum={weighted_sum}",
)

print(
    "GLOBAL ROUTING INVARIANTS: PASS"
)


# ============================================================
# 5. SEMANTIC GLOBAL RANKS
# ============================================================

rows = read_csv(
    AGG / "semantic_global_all2450.csv"
)

for e in range(6):

    erows = [
        r for r in rows
        if int(r["expert_id"]) == e
    ]

    require(
        len(erows) == 2450,
        f"E{e}: global semantic count != 2450",
    )

    ranks = sorted(
        int(
            r["global_mean_rank"]
        )
        for r in erows
    )

    require(
        ranks == list(
            range(1, 2451)
        ),
        f"E{e}: invalid global ranks",
    )

print(
    "GLOBAL SEMANTIC RANKS: PASS"
)


# ============================================================
# 6. FIGURE SET
# ============================================================

for stem in FIGURE_STEMS:

    png = FIGS / f"{stem}.png"
    pdf = FIGS / f"{stem}.pdf"

    require(
        png.is_file(),
        f"Missing PNG: {png.name}",
    )

    require(
        pdf.is_file(),
        f"Missing PDF: {pdf.name}",
    )

    require(
        png.stat().st_size > 10_000,
        f"Suspiciously small PNG: {png.name}",
    )

    require(
        pdf.stat().st_size > 5_000,
        f"Suspiciously small PDF: {pdf.name}",
    )

print(
    "FIGURE SET: PASS "
    "(6 PNG + 6 PDF)"
)


# ============================================================
# 7. WRITE FINAL HASH MANIFEST
# ============================================================

all_final_files = []

for name in AGG_REQUIRED:
    all_final_files.append(
        AGG / name
    )

for stem in FIGURE_STEMS:
    all_final_files.append(
        FIGS / f"{stem}.png"
    )
    all_final_files.append(
        FIGS / f"{stem}.pdf"
    )

hash_file = (
    AGG / "FINAL_INTERPRETABILITY_SHA256SUMS.txt"
)

with open(hash_file, "w") as f:

    for p in sorted(
        all_final_files,
        key=lambda x: str(x),
    ):

        f.write(
            f"{sha256(p)}  "
            f"{p.relative_to(AGG)}\n"
        )


# ============================================================
# 8. WRITE FINAL STATUS
# ============================================================

status_file = (
    AGG
    / "FINAL_NATURAL_ROUTING_INTERPRETABILITY_STATUS.txt"
)

with open(status_file, "w") as f:

    f.write(
        "RUN-2 NATURAL-ROUTING INTERPRETABILITY\n"
    )
    f.write(
        "========================================\n"
    )
    f.write(
        "STATUS: PASS\n\n"
    )
    f.write(
        "Coverage:\n"
    )
    f.write(
        "- 23/23 held-out slides\n"
    )
    f.write(
        "- 92,507 patches\n"
    )
    f.write(
        "- 5 frozen seeds\n"
    )
    f.write(
        "- 2 Euler evaluation states per seed\n"
    )
    f.write(
        "- 6 MoLF experts\n"
    )
    f.write(
        "- Top-256 affinity patches per expert/slide\n"
    )
    f.write(
        "- 2,450 CONCH visual Homo_sapiens concepts\n"
    )
    f.write(
        "- cross-slide semantic stability\n"
    )
    f.write(
        "- Visium/Xenium routing breakdown\n"
    )
    f.write(
        "- aggregate semantic profiles\n"
    )
    f.write(
        "- final figures generated\n\n"
    )
    f.write(
        "Important scope note:\n"
    )
    f.write(
        "This status covers the natural-routing "
        "interpretability branch only.\n"
    )
    f.write(
        "Forced-expert/mechanistic interpretation "
        "is a separate analysis.\n"
    )


print()
print("=" * 64)
print(
    "FINAL NATURAL-ROUTING INTERPRETABILITY AUDIT: PASS"
)
print("=" * 64)

print()
print(
    "Status:",
    status_file,
)

print(
    "Hashes:",
    hash_file,
)
