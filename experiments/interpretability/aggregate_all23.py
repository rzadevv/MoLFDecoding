#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(
    "/data/cat/ws/rzmi154h-molf-pathology/"
    "conch_molf_production/run2_interpretability/all23"
)

OUT = ROOT / "aggregate"

SLIDES = [
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
]

NUM_EXPERTS = 6
NUM_CONCEPTS = 2450


def technology(sid):
    # All 7 TENX held-out slides are Xenium.
    # The remaining 16 held-out slides are Visium.
    return (
        "Xenium"
        if sid.startswith("TENX")
        else "Visium"
    )


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean()
    y = y - y.mean()

    denom = (
        np.sqrt(np.sum(x * x))
        * np.sqrt(np.sum(y * y))
    )

    if denom == 0.0:
        return np.nan

    return float(
        np.sum(x * y) / denom
    )


def rankdata(x):
    """
    Average ranks for ties.
    Ascending rank convention.
    """
    x = np.asarray(x)

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    i = 0

    while i < len(x):

        j = i + 1

        while (
            j < len(x)
            and x[order[j]]
            == x[order[i]]
        ):
            j += 1

        # 1-based average rank
        avg_rank = (
            (i + 1)
            + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    return ranks


def spearman(x, y):
    return pearson(
        rankdata(x),
        rankdata(y),
    )


def safe_mean(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )
    return float(np.mean(arr))


def safe_sd(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(arr) <= 1:
        return 0.0

    return float(
        np.std(
            arr,
            ddof=1,
        )
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "==============================================="
    )
    print(
        "RUN-2 ALL-23 AGGREGATION"
    )
    print(
        "==============================================="
    )
    print()


    # ========================================================
    # Load and verify all 23 slide outputs
    # ========================================================

    slide_manifests = {}
    routing = {}
    semantic_raw = {}

    canonical_concept_ids = None
    concept_meta = {}


    for slide_idx, sid in enumerate(
        SLIDES
    ):

        d = ROOT / sid

        require(
            d.is_dir(),
            f"{sid}: directory missing",
        )

        require(
            (d / "STATUS.txt")
            .read_text()
            .strip()
            == "PASS",
            f"{sid}: STATUS is not PASS",
        )


        manifest = json.loads(
            (d / "manifest.json")
            .read_text()
        )

        require(
            manifest["sample_id"] == sid,
            f"{sid}: manifest ID mismatch",
        )

        require(
            int(manifest["n_contexts"]) == 10,
            f"{sid}: context mismatch",
        )

        require(
            int(manifest["semantic_concepts"])
            == NUM_CONCEPTS,
            f"{sid}: concept count mismatch",
        )

        slide_manifests[sid] = manifest


        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

        summary = read_csv(
            d / "expert_summary.csv"
        )

        require(
            len(summary) == NUM_EXPERTS,
            f"{sid}: expected 6 expert rows",
        )

        routing[sid] = {}

        for row in summary:

            e = int(
                row["expert_id"]
            )

            routing[sid][e] = {
                k: float(v)
                if k
                not in (
                    "sample_id",
                    "expert_id",
                )
                else v
                for k, v in row.items()
                if k not in (
                    "sample_id",
                )
            }


        require(
            sorted(routing[sid].keys())
            == list(range(6)),
            f"{sid}: expert ID mismatch",
        )


        # ----------------------------------------------------
        # Semantic scores: all 2450 × 6
        # ----------------------------------------------------

        concept_rows = read_csv(
            d / "concept_scores_all2450.csv"
        )

        require(
            len(concept_rows)
            == NUM_CONCEPTS * NUM_EXPERTS,
            f"{sid}: semantic row count mismatch",
        )


        semantic_raw[sid] = {}


        for e in range(
            NUM_EXPERTS
        ):

            erows = [
                r
                for r in concept_rows
                if int(
                    r["expert_id"]
                ) == e
            ]

            require(
                len(erows) == NUM_CONCEPTS,
                f"{sid} E{e}: expected "
                f"{NUM_CONCEPTS} concepts",
            )


            by_id = {}

            for r in erows:

                cid = r["concept_id"]

                require(
                    cid not in by_id,
                    f"{sid} E{e}: "
                    f"duplicate {cid}",
                )

                by_id[cid] = r


                meta = (
                    r["concept_name"],
                    r["tier"],
                    r["embedding_set"],
                    r["species"],
                    r["prompt_text"],
                )


                if cid in concept_meta:

                    require(
                        concept_meta[cid]
                        == meta,
                        f"{sid}: concept metadata "
                        f"changed for {cid}",
                    )

                else:

                    concept_meta[cid] = meta


            ids = sorted(
                by_id.keys()
            )


            if canonical_concept_ids is None:

                canonical_concept_ids = ids

            else:

                require(
                    ids
                    == canonical_concept_ids,
                    f"{sid} E{e}: concept bank "
                    "differs from canonical bank",
                )


            scores = np.asarray(
                [
                    float(
                        by_id[cid][
                            "similarity_topk_contrastive"
                        ]
                    )
                    for cid
                    in canonical_concept_ids
                ],
                dtype=np.float64,
            )


            means = np.asarray(
                [
                    float(
                        by_id[cid][
                            "similarity_topk_mean"
                        ]
                    )
                    for cid
                    in canonical_concept_ids
                ],
                dtype=np.float64,
            )


            ranks = np.asarray(
                [
                    int(
                        by_id[cid][
                            "rank"
                        ]
                    )
                    for cid
                    in canonical_concept_ids
                ],
                dtype=np.int32,
            )


            require(
                np.isfinite(scores).all(),
                f"{sid} E{e}: nonfinite "
                "contrastive score",
            )

            require(
                np.isfinite(means).all(),
                f"{sid} E{e}: nonfinite "
                "mean score",
            )


            semantic_raw[sid][e] = {
                "contrastive":
                    scores,
                "topk_mean":
                    means,
                "ranks":
                    ranks,
            }


    require(
        len(canonical_concept_ids)
        == NUM_CONCEPTS,
        "Canonical concept bank size mismatch",
    )


    print(
        "23/23 slide inputs verified"
    )
    print(
        f"Concept bank: "
        f"{len(canonical_concept_ids)} concepts"
    )


    # ========================================================
    # ROUTING BY SLIDE
    # ========================================================

    routing_by_slide_rows = []


    for sid in SLIDES:

        P = int(
            slide_manifests[
                sid
            ]["n_patches"]
        )


        for e in range(
            NUM_EXPERTS
        ):

            r = routing[
                sid
            ][e]

            routing_by_slide_rows.append({
                "sample_id":
                    sid,
                "technology":
                    technology(sid),
                "n_patches":
                    P,
                "expert_id":
                    e,
                "mean_route_weight":
                    r[
                        "mean_route_weight_all_patches"
                    ],
                "mean_router_prob":
                    r[
                        "mean_router_prob_all_patches"
                    ],
                "mean_top2_frequency":
                    r[
                        "mean_top2_frequency_all_patches"
                    ],
                "mean_top1_frequency":
                    r[
                        "mean_top1_frequency_all_patches"
                    ],
                "ever_top2_patches":
                    int(
                        r[
                            "ever_top2_patches"
                        ]
                    ),
                "always_top2_patches":
                    int(
                        r[
                            "always_top2_patches"
                        ]
                    ),
                "top256_mean_top2_frequency":
                    r[
                        "top256_mean_top2_frequency"
                    ],
                "top256_mean_top1_frequency":
                    r[
                        "top256_mean_top1_frequency"
                    ],
            })


    write_csv(
        OUT / "routing_by_slide.csv",
        list(
            routing_by_slide_rows[
                0
            ].keys()
        ),
        routing_by_slide_rows,
    )


    # ========================================================
    # GLOBAL ROUTING SUMMARY
    # ========================================================

    routing_global_rows = []


    dominant_counts = {
        e: 0
        for e in range(
            NUM_EXPERTS
        )
    }


    for sid in SLIDES:

        dominant_e = max(
            range(NUM_EXPERTS),
            key=lambda e:
                routing[sid][e][
                    "mean_route_weight_all_patches"
                ],
        )

        dominant_counts[
            dominant_e
        ] += 1


    total_patches = sum(
        int(
            slide_manifests[
                sid
            ]["n_patches"]
        )
        for sid in SLIDES
    )


    for e in range(
        NUM_EXPERTS
    ):

        route_vals = np.asarray(
            [
                routing[sid][e][
                    "mean_route_weight_all_patches"
                ]
                for sid in SLIDES
            ],
            dtype=np.float64,
        )


        prob_vals = np.asarray(
            [
                routing[sid][e][
                    "mean_router_prob_all_patches"
                ]
                for sid in SLIDES
            ],
            dtype=np.float64,
        )


        top2_vals = np.asarray(
            [
                routing[sid][e][
                    "mean_top2_frequency_all_patches"
                ]
                for sid in SLIDES
            ],
            dtype=np.float64,
        )


        top1_vals = np.asarray(
            [
                routing[sid][e][
                    "mean_top1_frequency_all_patches"
                ]
                for sid in SLIDES
            ],
            dtype=np.float64,
        )


        patch_counts = np.asarray(
            [
                int(
                    slide_manifests[
                        sid
                    ]["n_patches"]
                )
                for sid in SLIDES
            ],
            dtype=np.float64,
        )


        patch_weighted_route = float(
            np.sum(
                route_vals
                * patch_counts
            )
            / np.sum(
                patch_counts
            )
        )


        patch_weighted_top2 = float(
            np.sum(
                top2_vals
                * patch_counts
            )
            / np.sum(
                patch_counts
            )
        )


        routing_global_rows.append({
            "expert_id":
                e,
            "n_slides":
                len(SLIDES),
            "total_patches":
                total_patches,
            "slide_macro_mean_route_weight":
                float(
                    route_vals.mean()
                ),
            "slide_macro_sd_route_weight":
                float(
                    route_vals.std(
                        ddof=1
                    )
                ),
            "slide_median_route_weight":
                float(
                    np.median(
                        route_vals
                    )
                ),
            "slide_min_route_weight":
                float(
                    route_vals.min()
                ),
            "slide_max_route_weight":
                float(
                    route_vals.max()
                ),
            "patch_weighted_route_weight":
                patch_weighted_route,
            "slide_macro_mean_router_prob":
                float(
                    prob_vals.mean()
                ),
            "slide_macro_mean_top2_frequency":
                float(
                    top2_vals.mean()
                ),
            "patch_weighted_top2_frequency":
                patch_weighted_top2,
            "slide_macro_mean_top1_frequency":
                float(
                    top1_vals.mean()
                ),
            "n_slides_route_gt_0.01":
                int(
                    np.sum(
                        route_vals > 0.01
                    )
                ),
            "n_slides_route_gt_0.05":
                int(
                    np.sum(
                        route_vals > 0.05
                    )
                ),
            "n_slides_route_gt_0.10":
                int(
                    np.sum(
                        route_vals > 0.10
                    )
                ),
            "dominant_slide_count":
                dominant_counts[e],
        })


    write_csv(
        OUT / "routing_global_summary.csv",
        list(
            routing_global_rows[
                0
            ].keys()
        ),
        routing_global_rows,
    )


    # ========================================================
    # ROUTING BY TECHNOLOGY
    # ========================================================

    routing_tech_rows = []


    for tech in (
        "Visium",
        "Xenium",
    ):

        tech_slides = [
            sid
            for sid in SLIDES
            if technology(sid)
            == tech
        ]


        for e in range(
            NUM_EXPERTS
        ):

            vals = np.asarray(
                [
                    routing[sid][e][
                        "mean_route_weight_all_patches"
                    ]
                    for sid
                    in tech_slides
                ],
                dtype=np.float64,
            )


            top2_vals = np.asarray(
                [
                    routing[sid][e][
                        "mean_top2_frequency_all_patches"
                    ]
                    for sid
                    in tech_slides
                ],
                dtype=np.float64,
            )


            counts = np.asarray(
                [
                    int(
                        slide_manifests[
                            sid
                        ][
                            "n_patches"
                        ]
                    )
                    for sid
                    in tech_slides
                ],
                dtype=np.float64,
            )


            routing_tech_rows.append({
                "technology":
                    tech,
                "expert_id":
                    e,
                "n_slides":
                    len(
                        tech_slides
                    ),
                "n_patches":
                    int(
                        counts.sum()
                    ),
                "slide_macro_mean_route_weight":
                    float(
                        vals.mean()
                    ),
                "slide_macro_sd_route_weight":
                    float(
                        vals.std(
                            ddof=1
                        )
                    ),
                "patch_weighted_route_weight":
                    float(
                        np.sum(
                            vals
                            * counts
                        )
                        / counts.sum()
                    ),
                "slide_macro_mean_top2_frequency":
                    float(
                        top2_vals.mean()
                    ),
                "patch_weighted_top2_frequency":
                    float(
                        np.sum(
                            top2_vals
                            * counts
                        )
                        / counts.sum()
                    ),
            })


    write_csv(
        OUT / "routing_by_technology.csv",
        list(
            routing_tech_rows[
                0
            ].keys()
        ),
        routing_tech_rows,
    )


    # ========================================================
    # BUILD FULL SEMANTIC TENSORS
    #
    # shape:
    #   experts × slides × concepts
    # ========================================================

    n_slides = len(
        SLIDES
    )


    contrastive = np.zeros(
        (
            NUM_EXPERTS,
            n_slides,
            NUM_CONCEPTS,
        ),
        dtype=np.float64,
    )


    topk_mean = np.zeros_like(
        contrastive
    )


    ranks = np.zeros(
        (
            NUM_EXPERTS,
            n_slides,
            NUM_CONCEPTS,
        ),
        dtype=np.int32,
    )


    for s_idx, sid in enumerate(
        SLIDES
    ):

        for e in range(
            NUM_EXPERTS
        ):

            contrastive[
                e,
                s_idx,
            ] = semantic_raw[
                sid
            ][e][
                "contrastive"
            ]

            topk_mean[
                e,
                s_idx,
            ] = semantic_raw[
                sid
            ][e][
                "topk_mean"
            ]

            ranks[
                e,
                s_idx,
            ] = semantic_raw[
                sid
            ][e][
                "ranks"
            ]


    # ========================================================
    # GLOBAL SEMANTIC AGGREGATION
    # ========================================================

    semantic_global_rows = []


    for e in range(
        NUM_EXPERTS
    ):

        # Natural router use of this expert
        # per slide. Kept separate from
        # semantic affinity, but also used
        # for one secondary weighted summary.
        route_weights = np.asarray(
            [
                routing[sid][e][
                    "mean_route_weight_all_patches"
                ]
                for sid in SLIDES
            ],
            dtype=np.float64,
        )


        route_weight_sum = float(
            route_weights.sum()
        )


        for c_idx, cid in enumerate(
            canonical_concept_ids
        ):

            vals = contrastive[
                e,
                :,
                c_idx,
            ]

            mean_vals = topk_mean[
                e,
                :,
                c_idx,
            ]

            rank_vals = ranks[
                e,
                :,
                c_idx,
            ]


            if route_weight_sum > 0:

                route_weighted_score = float(
                    np.sum(
                        vals
                        * route_weights
                    )
                    / route_weight_sum
                )

            else:

                route_weighted_score = (
                    float("nan")
                )


            (
                concept_name,
                tier,
                embedding_set,
                species,
                prompt_text,
            ) = concept_meta[
                cid
            ]


            semantic_global_rows.append({
                "expert_id":
                    e,
                "concept_id":
                    cid,
                "concept_name":
                    concept_name,
                "tier":
                    tier,
                "embedding_set":
                    embedding_set,
                "species":
                    species,
                "n_slides":
                    n_slides,
                "mean_contrastive_similarity":
                    float(
                        vals.mean()
                    ),
                "sd_contrastive_similarity":
                    float(
                        vals.std(
                            ddof=1
                        )
                    ),
                "median_contrastive_similarity":
                    float(
                        np.median(
                            vals
                        )
                    ),
                "min_contrastive_similarity":
                    float(
                        vals.min()
                    ),
                "max_contrastive_similarity":
                    float(
                        vals.max()
                    ),
                "mean_topk_similarity":
                    float(
                        mean_vals.mean()
                    ),
                "mean_rank":
                    float(
                        rank_vals.mean()
                    ),
                "median_rank":
                    float(
                        np.median(
                            rank_vals
                        )
                    ),
                "top10_count":
                    int(
                        np.sum(
                            rank_vals <= 10
                        )
                    ),
                "top10_fraction":
                    float(
                        np.mean(
                            rank_vals <= 10
                        )
                    ),
                "top50_count":
                    int(
                        np.sum(
                            rank_vals <= 50
                        )
                    ),
                "top50_fraction":
                    float(
                        np.mean(
                            rank_vals <= 50
                        )
                    ),
                "route_weighted_mean_contrastive_similarity":
                    route_weighted_score,
                "router_weight_sum_across_slides":
                    route_weight_sum,
                "prompt_text":
                    prompt_text,
            })


    # Global rank inside each expert:
    # primary = slide-equal macro mean
    for e in range(
        NUM_EXPERTS
    ):

        erows = [
            r
            for r in semantic_global_rows
            if int(
                r["expert_id"]
            ) == e
        ]


        erows.sort(
            key=lambda r:
                r[
                    "mean_contrastive_similarity"
                ],
            reverse=True,
        )


        for rank, row in enumerate(
            erows,
            start=1,
        ):

            row[
                "global_mean_rank"
            ] = rank


        erows_rw = sorted(
            erows,
            key=lambda r:
                r[
                    "route_weighted_mean_contrastive_similarity"
                ],
            reverse=True,
        )


        for rank, row in enumerate(
            erows_rw,
            start=1,
        ):

            row[
                "route_weighted_global_rank"
            ] = rank


    semantic_global_rows.sort(
        key=lambda r: (
            int(
                r["expert_id"]
            ),
            int(
                r[
                    "global_mean_rank"
                ]
            ),
        )
    )


    semantic_fields = [
        "expert_id",
        "global_mean_rank",
        "route_weighted_global_rank",
        "concept_id",
        "concept_name",
        "tier",
        "embedding_set",
        "species",
        "n_slides",
        "mean_contrastive_similarity",
        "sd_contrastive_similarity",
        "median_contrastive_similarity",
        "min_contrastive_similarity",
        "max_contrastive_similarity",
        "mean_topk_similarity",
        "mean_rank",
        "median_rank",
        "top10_count",
        "top10_fraction",
        "top50_count",
        "top50_fraction",
        "route_weighted_mean_contrastive_similarity",
        "router_weight_sum_across_slides",
        "prompt_text",
    ]


    write_csv(
        OUT
        / "semantic_global_all2450.csv",
        semantic_fields,
        semantic_global_rows,
    )


    global_top50 = [
        r
        for r in semantic_global_rows
        if int(
            r[
                "global_mean_rank"
            ]
        ) <= 50
    ]


    write_csv(
        OUT
        / "semantic_global_top50.csv",
        semantic_fields,
        global_top50,
    )


    route_weighted_top50 = [
        r
        for r in semantic_global_rows
        if int(
            r[
                "route_weighted_global_rank"
            ]
        ) <= 50
    ]


    route_weighted_top50.sort(
        key=lambda r: (
            int(
                r["expert_id"]
            ),
            int(
                r[
                    "route_weighted_global_rank"
                ]
            ),
        )
    )


    write_csv(
        OUT
        / "semantic_route_weighted_top50.csv",
        semantic_fields,
        route_weighted_top50,
    )


    # ========================================================
    # CROSS-SLIDE SEMANTIC STABILITY
    #
    # Compare each expert's complete 2450-score
    # semantic vector between every pair of slides.
    # 23 choose 2 = 253 pairs per expert.
    # ========================================================

    stability_rows = []


    ranked_vectors = np.zeros_like(
        contrastive
    )


    for e in range(
        NUM_EXPERTS
    ):

        for s_idx in range(
            n_slides
        ):

            ranked_vectors[
                e,
                s_idx,
            ] = rankdata(
                contrastive[
                    e,
                    s_idx,
                ]
            )


    for e in range(
        NUM_EXPERTS
    ):

        for i in range(
            n_slides
        ):

            for j in range(
                i + 1,
                n_slides,
            ):

                sid_a = SLIDES[i]
                sid_b = SLIDES[j]

                tech_a = technology(
                    sid_a
                )

                tech_b = technology(
                    sid_b
                )


                if tech_a == tech_b:

                    pair_type = (
                        f"{tech_a}-{tech_b}"
                    )

                else:

                    pair_type = (
                        "cross-technology"
                    )


                p = pearson(
                    contrastive[
                        e,
                        i,
                    ],
                    contrastive[
                        e,
                        j,
                    ],
                )


                s = pearson(
                    ranked_vectors[
                        e,
                        i,
                    ],
                    ranked_vectors[
                        e,
                        j,
                    ],
                )


                stability_rows.append({
                    "expert_id":
                        e,
                    "slide_a":
                        sid_a,
                    "technology_a":
                        tech_a,
                    "slide_b":
                        sid_b,
                    "technology_b":
                        tech_b,
                    "pair_type":
                        pair_type,
                    "pearson":
                        p,
                    "spearman":
                        s,
                })


    require(
        len(stability_rows)
        == NUM_EXPERTS * 253,
        "Pairwise stability row "
        "count mismatch",
    )


    write_csv(
        OUT
        / "semantic_slide_pair_stability.csv",
        list(
            stability_rows[
                0
            ].keys()
        ),
        stability_rows,
    )


    # ========================================================
    # STABILITY SUMMARY
    # ========================================================

    stability_summary_rows = []


    pair_types = [
        "all",
        "Visium-Visium",
        "Xenium-Xenium",
        "cross-technology",
    ]


    for e in range(
        NUM_EXPERTS
    ):

        erows = [
            r
            for r in stability_rows
            if int(
                r["expert_id"]
            ) == e
        ]


        for pair_type in pair_types:

            if pair_type == "all":

                rows = erows

            else:

                rows = [
                    r
                    for r in erows
                    if r["pair_type"]
                    == pair_type
                ]


            if not rows:
                continue


            pvals = [
                float(
                    r["pearson"]
                )
                for r in rows
            ]

            svals = [
                float(
                    r["spearman"]
                )
                for r in rows
            ]


            stability_summary_rows.append({
                "expert_id":
                    e,
                "pair_type":
                    pair_type,
                "n_pairs":
                    len(rows),
                "mean_pearson":
                    safe_mean(
                        pvals
                    ),
                "sd_pearson":
                    safe_sd(
                        pvals
                    ),
                "median_pearson":
                    float(
                        np.median(
                            pvals
                        )
                    ),
                "min_pearson":
                    float(
                        np.min(
                            pvals
                        )
                    ),
                "max_pearson":
                    float(
                        np.max(
                            pvals
                        )
                    ),
                "mean_spearman":
                    safe_mean(
                        svals
                    ),
                "sd_spearman":
                    safe_sd(
                        svals
                    ),
                "median_spearman":
                    float(
                        np.median(
                            svals
                        )
                    ),
                "min_spearman":
                    float(
                        np.min(
                            svals
                        )
                    ),
                "max_spearman":
                    float(
                        np.max(
                            svals
                        )
                    ),
            })


    write_csv(
        OUT
        / "semantic_stability_summary.csv",
        list(
            stability_summary_rows[
                0
            ].keys()
        ),
        stability_summary_rows,
    )


    # ========================================================
    # GLOBAL EXPERT-TO-EXPERT SEMANTIC SIMILARITY
    #
    # Compare the six aggregate mean profiles.
    # ========================================================

    expert_mean_profiles = (
        contrastive.mean(
            axis=1
        )
    )


    expert_pair_rows = []


    for e1 in range(
        NUM_EXPERTS
    ):

        for e2 in range(
            e1 + 1,
            NUM_EXPERTS,
        ):

            expert_pair_rows.append({
                "expert_a":
                    e1,
                "expert_b":
                    e2,
                "pearson_mean_profile":
                    pearson(
                        expert_mean_profiles[
                            e1
                        ],
                        expert_mean_profiles[
                            e2
                        ],
                    ),
                "spearman_mean_profile":
                    spearman(
                        expert_mean_profiles[
                            e1
                        ],
                        expert_mean_profiles[
                            e2
                        ],
                    ),
            })


    write_csv(
        OUT
        / "semantic_expert_pair_similarity.csv",
        list(
            expert_pair_rows[
                0
            ].keys()
        ),
        expert_pair_rows,
    )


    # ========================================================
    # TECHNOLOGY-SPECIFIC SEMANTIC AGGREGATES
    # ========================================================

    tech_semantic_rows = []


    for tech in (
        "Visium",
        "Xenium",
    ):

        indices = [
            i
            for i, sid
            in enumerate(
                SLIDES
            )
            if technology(sid)
            == tech
        ]


        for e in range(
            NUM_EXPERTS
        ):

            for c_idx, cid in enumerate(
                canonical_concept_ids
            ):

                vals = contrastive[
                    e,
                    indices,
                    c_idx,
                ]


                (
                    concept_name,
                    tier,
                    embedding_set,
                    species,
                    _,
                ) = concept_meta[
                    cid
                ]


                tech_semantic_rows.append({
                    "technology":
                        tech,
                    "expert_id":
                        e,
                    "concept_id":
                        cid,
                    "concept_name":
                        concept_name,
                    "tier":
                        tier,
                    "embedding_set":
                        embedding_set,
                    "species":
                        species,
                    "n_slides":
                        len(
                            indices
                        ),
                    "mean_contrastive_similarity":
                        float(
                            vals.mean()
                        ),
                    "sd_contrastive_similarity":
                        float(
                            vals.std(
                                ddof=1
                            )
                            if len(vals) > 1
                            else 0.0
                        ),
                })


    # rank concepts inside each technology/expert
    for tech in (
        "Visium",
        "Xenium",
    ):

        for e in range(
            NUM_EXPERTS
        ):

            rows = [
                r
                for r in tech_semantic_rows
                if (
                    r["technology"]
                    == tech
                    and int(
                        r["expert_id"]
                    ) == e
                )
            ]


            rows.sort(
                key=lambda r:
                    r[
                        "mean_contrastive_similarity"
                    ],
                reverse=True,
            )


            for rank, row in enumerate(
                rows,
                start=1,
            ):

                row[
                    "technology_rank"
                ] = rank


    tech_semantic_rows.sort(
        key=lambda r: (
            r["technology"],
            int(
                r["expert_id"]
            ),
            int(
                r[
                    "technology_rank"
                ]
            ),
        )
    )


    write_csv(
        OUT
        / "semantic_by_technology_all2450.csv",
        [
            "technology",
            "expert_id",
            "technology_rank",
            "concept_id",
            "concept_name",
            "tier",
            "embedding_set",
            "species",
            "n_slides",
            "mean_contrastive_similarity",
            "sd_contrastive_similarity",
        ],
        tech_semantic_rows,
    )


    tech_top25 = [
        r
        for r in tech_semantic_rows
        if int(
            r[
                "technology_rank"
            ]
        ) <= 25
    ]


    write_csv(
        OUT
        / "semantic_by_technology_top25.csv",
        [
            "technology",
            "expert_id",
            "technology_rank",
            "concept_id",
            "concept_name",
            "tier",
            "embedding_set",
            "species",
            "n_slides",
            "mean_contrastive_similarity",
            "sd_contrastive_similarity",
        ],
        tech_top25,
    )


    # ========================================================
    # SUMMARY TXT
    # ========================================================

    summary_path = (
        OUT
        / "ALL23_AGGREGATE_SUMMARY.txt"
    )


    with open(
        summary_path,
        "w",
    ) as f:

        f.write(
            "RUN-2 ALL-23 AGGREGATE SUMMARY\n"
        )
        f.write(
            "========================================\n\n"
        )

        f.write(
            f"Slides: {len(SLIDES)}\n"
        )
        f.write(
            f"Total patches: {total_patches}\n"
        )
        f.write(
            "Experts: 6\n"
        )
        f.write(
            "Concepts per expert/slide: 2450\n"
        )
        f.write(
            "Semantic metric: "
            "similarity_topk_contrastive\n\n"
        )


        f.write(
            "GLOBAL ROUTING\n"
        )
        f.write(
            "----------------------------------------\n"
        )


        for row in routing_global_rows:

            f.write(
                f"E{row['expert_id']}: "
                f"macro_route="
                f"{row['slide_macro_mean_route_weight']:.6f}, "
                f"patch_weighted_route="
                f"{row['patch_weighted_route_weight']:.6f}, "
                f"dominant_slides="
                f"{row['dominant_slide_count']}/23, "
                f"slides_route>0.05="
                f"{row['n_slides_route_gt_0.05']}/23\n"
            )


        f.write(
            "\nROUTING BY TECHNOLOGY\n"
        )
        f.write(
            "----------------------------------------\n"
        )


        for tech in (
            "Visium",
            "Xenium",
        ):

            f.write(
                f"\n{tech}\n"
            )

            rows = [
                r
                for r
                in routing_tech_rows
                if r["technology"]
                == tech
            ]

            for row in rows:

                f.write(
                    f"E{row['expert_id']}: "
                    f"macro_route="
                    f"{row['slide_macro_mean_route_weight']:.6f}, "
                    f"patch_weighted_route="
                    f"{row['patch_weighted_route_weight']:.6f}\n"
                )


        f.write(
            "\nSEMANTIC STABILITY\n"
        )
        f.write(
            "----------------------------------------\n"
        )


        for e in range(
            NUM_EXPERTS
        ):

            row = next(
                r
                for r
                in stability_summary_rows
                if int(
                    r["expert_id"]
                ) == e
                and r["pair_type"]
                == "all"
            )

            f.write(
                f"E{e}: "
                f"mean Pearson="
                f"{row['mean_pearson']:.6f}, "
                f"mean Spearman="
                f"{row['mean_spearman']:.6f}, "
                f"median Spearman="
                f"{row['median_spearman']:.6f}\n"
            )


        f.write(
            "\nTOP 10 GLOBAL CONCEPTS PER EXPERT\n"
        )
        f.write(
            "----------------------------------------\n"
        )


        for e in range(
            NUM_EXPERTS
        ):

            f.write(
                f"\nExpert {e}\n"
            )


            rows = [
                r
                for r
                in semantic_global_rows
                if int(
                    r["expert_id"]
                ) == e
                and int(
                    r[
                        "global_mean_rank"
                    ]
                ) <= 10
            ]


            rows.sort(
                key=lambda r:
                    int(
                        r[
                            "global_mean_rank"
                        ]
                    )
            )


            for r in rows:

                f.write(
                    f"{int(r['global_mean_rank']):2d}. "
                    f"{r['concept_name']} | "
                    f"mean={r['mean_contrastive_similarity']:.6f} | "
                    f"SD={r['sd_contrastive_similarity']:.6f} | "
                    f"top10={r['top10_count']}/23 | "
                    f"top50={r['top50_count']}/23\n"
                )


    # ========================================================
    # HASH DERIVED OUTPUTS
    # ========================================================

    output_files = sorted(
        [
            p
            for p in OUT.iterdir()
            if p.is_file()
            and p.name
            != "SHA256SUMS.txt"
        ],
        key=lambda p: p.name,
    )


    with open(
        OUT / "SHA256SUMS.txt",
        "w",
    ) as f:

        for p in output_files:

            f.write(
                f"{sha256(p)}  "
                f"{p.name}\n"
            )


    # ========================================================
    # STDOUT COMPACT RESULTS
    # ========================================================

    print()
    print(
        "=== GLOBAL ROUTING ==="
    )


    for row in routing_global_rows:

        print(
            f"E{row['expert_id']}: "
            f"macro="
            f"{row['slide_macro_mean_route_weight']:.4f} "
            f"patch-weighted="
            f"{row['patch_weighted_route_weight']:.4f} "
            f"dominant="
            f"{row['dominant_slide_count']}/23"
        )


    print()
    print(
        "=== ROUTING BY TECHNOLOGY ==="
    )


    for tech in (
        "Visium",
        "Xenium",
    ):

        print()
        print(
            tech
        )

        for row in routing_tech_rows:

            if (
                row["technology"]
                == tech
            ):

                print(
                    f"E{row['expert_id']}: "
                    f"macro="
                    f"{row['slide_macro_mean_route_weight']:.4f} "
                    f"patch-weighted="
                    f"{row['patch_weighted_route_weight']:.4f}"
                )


    print()
    print(
        "=== CROSS-SLIDE SEMANTIC STABILITY ==="
    )


    for e in range(
        NUM_EXPERTS
    ):

        row = next(
            r
            for r
            in stability_summary_rows
            if (
                int(
                    r["expert_id"]
                ) == e
                and r["pair_type"]
                == "all"
            )
        )


        print(
            f"E{e}: "
            f"Pearson="
            f"{row['mean_pearson']:.4f} "
            f"Spearman="
            f"{row['mean_spearman']:.4f} "
            f"medianSpearman="
            f"{row['median_spearman']:.4f}"
        )


    print()
    print(
        "=== TOP 5 GLOBAL CONCEPTS ==="
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
            for r
            in semantic_global_rows
            if (
                int(
                    r["expert_id"]
                ) == e
                and int(
                    r[
                        "global_mean_rank"
                    ]
                ) <= 5
            )
        ]


        rows.sort(
            key=lambda r:
                int(
                    r[
                        "global_mean_rank"
                    ]
                )
        )


        for r in rows:

            print(
                f"{int(r['global_mean_rank']):2d}. "
                f"{r['concept_name']} "
                f"mean="
                f"{r['mean_contrastive_similarity']:+.4f} "
                f"top10="
                f"{r['top10_count']}/23 "
                f"top50="
                f"{r['top50_count']}/23"
            )


    print()
    print(
        "==============================================="
    )
    print(
        "ALL23 AGGREGATION: PASS"
    )
    print(
        "==============================================="
    )
    print(
        "Output:",
        OUT,
    )


if __name__ == "__main__":
    main()
