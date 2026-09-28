#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(
    "/data/cat/ws/rzmi154h-molf-pathology/"
    "conch_molf_production/run2_interpretability/all23"
)

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

REQUIRED_FILES = [
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

EXPECTED_SEEDS = [
    41001,
    42001,
    43001,
    44001,
    45001,
]


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path):
    with open(
        path,
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def finite(x):
    return math.isfinite(
        float(x)
    )


def audit_slide(sid):

    d = ROOT / sid

    require(
        d.is_dir(),
        f"{sid}: output directory missing",
    )

    for name in REQUIRED_FILES:
        require(
            (d / name).is_file(),
            f"{sid}: missing {name}",
        )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    require(
        (d / "STATUS.txt")
        .read_text()
        .strip()
        == "PASS",
        f"{sid}: STATUS != PASS",
    )


    # --------------------------------------------------------
    # Manifest / frozen protocol
    # --------------------------------------------------------

    manifest = json.loads(
        (d / "manifest.json")
        .read_text()
    )

    require(
        manifest["sample_id"] == sid,
        f"{sid}: manifest sample mismatch",
    )

    require(
        manifest["status"] == "PASS",
        f"{sid}: manifest status mismatch",
    )

    require(
        int(manifest["n_contexts"]) == 10,
        f"{sid}: expected 10 contexts",
    )

    require(
        manifest["base_seeds"]
        == EXPECTED_SEEDS,
        f"{sid}: seed set mismatch",
    )

    require(
        manifest["solver"] == "euler",
        f"{sid}: solver mismatch",
    )

    require(
        int(manifest["num_steps"]) == 2,
        f"{sid}: num_steps mismatch",
    )

    require(
        abs(
            float(manifest["step_size"])
            - 0.5
        ) < 1e-12,
        f"{sid}: step size mismatch",
    )

    require(
        abs(
            float(manifest["guidance_scale"])
            - 1.5
        ) < 1e-12,
        f"{sid}: CFG mismatch",
    )

    require(
        manifest["router_branch"]
        == "conditional",
        f"{sid}: router branch mismatch",
    )

    require(
        int(manifest["num_experts"]) == 6,
        f"{sid}: expert count mismatch",
    )

    require(
        int(manifest["router_top_k"]) == 2,
        f"{sid}: router top-k mismatch",
    )

    require(
        int(manifest["semantic_concepts"])
        == 2450,
        f"{sid}: concept count mismatch",
    )

    require(
        manifest[
            "semantic_embedding_set"
        ] == "visual",
        f"{sid}: embedding set mismatch",
    )

    require(
        manifest["semantic_species"]
        == "Homo_sapiens",
        f"{sid}: species mismatch",
    )

    require(
        manifest["semantic_rank_metric"]
        == "similarity_topk_contrastive",
        f"{sid}: ranking metric mismatch",
    )


    P = int(
        manifest["n_patches"]
    )

    require(
        P > 0,
        f"{sid}: invalid patch count",
    )


    # --------------------------------------------------------
    # SHA256 result verification
    # --------------------------------------------------------

    with open(
        d / "SHA256SUMS.txt"
    ) as f:

        sha_lines = [
            line.strip()
            for line in f
            if line.strip()
        ]


    require(
        len(sha_lines) > 0,
        f"{sid}: empty SHA256SUMS",
    )


    for line in sha_lines:

        expected, filename = (
            line.split(None, 1)
        )

        filename = (
            filename.strip()
        )

        p = d / filename

        require(
            p.is_file(),
            f"{sid}: hashed file missing "
            f"{filename}",
        )

        actual = sha256(p)

        require(
            actual == expected,
            f"{sid}: SHA mismatch "
            f"{filename}",
        )


    # --------------------------------------------------------
    # CSV row-count checks
    # --------------------------------------------------------

    router_aggregate = read_csv(
        d / "router_aggregate.csv"
    )

    router_contexts = read_csv(
        d / "router_contexts.csv"
    )

    affinity = read_csv(
        d / "router_rankings_affinity.csv"
    )

    natural = read_csv(
        d / "router_rankings_natural.csv"
    )

    top256 = read_csv(
        d / "top256_affinity.csv"
    )

    concept_all = read_csv(
        d / "concept_scores_all2450.csv"
    )

    concept_top50 = read_csv(
        d / "concept_rankings_top50.csv"
    )

    expert_summary = read_csv(
        d / "expert_summary.csv"
    )


    require(
        len(router_aggregate) == P,
        f"{sid}: router_aggregate rows "
        f"{len(router_aggregate)} != {P}",
    )

    require(
        len(router_contexts) == P * 10,
        f"{sid}: router_contexts rows "
        f"{len(router_contexts)} "
        f"!= {P * 10}",
    )

    require(
        len(affinity) == P * 6,
        f"{sid}: affinity rows "
        f"{len(affinity)} != {P * 6}",
    )

    require(
        len(natural) == P * 6,
        f"{sid}: natural rows "
        f"{len(natural)} != {P * 6}",
    )


    K = min(
        256,
        P,
    )

    require(
        len(top256) == K * 6,
        f"{sid}: top256 rows "
        f"{len(top256)} != {K * 6}",
    )

    require(
        len(concept_all)
        == 2450 * 6,
        f"{sid}: all-concept rows "
        f"{len(concept_all)} != 14700",
    )

    require(
        len(concept_top50)
        == 50 * 6,
        f"{sid}: top50 rows "
        f"{len(concept_top50)} != 300",
    )

    require(
        len(expert_summary) == 6,
        f"{sid}: expert_summary rows != 6",
    )


    # --------------------------------------------------------
    # Router summary invariants
    # --------------------------------------------------------

    experts = sorted(
        int(r["expert_id"])
        for r in expert_summary
    )

    require(
        experts == list(range(6)),
        f"{sid}: expert IDs incorrect",
    )


    route_sum = sum(
        float(
            r[
                "mean_route_weight_all_patches"
            ]
        )
        for r in expert_summary
    )

    top2_sum = sum(
        float(
            r[
                "mean_top2_frequency_all_patches"
            ]
        )
        for r in expert_summary
    )

    top1_sum = sum(
        float(
            r[
                "mean_top1_frequency_all_patches"
            ]
        )
        for r in expert_summary
    )


    require(
        abs(route_sum - 1.0) < 1e-5,
        f"{sid}: global route weight "
        f"sum={route_sum}",
    )

    require(
        abs(top2_sum - 2.0) < 1e-5,
        f"{sid}: global top2 "
        f"sum={top2_sum}",
    )

    require(
        abs(top1_sum - 1.0) < 1e-5,
        f"{sid}: global top1 "
        f"sum={top1_sum}",
    )


    # --------------------------------------------------------
    # Top-256 integrity
    # --------------------------------------------------------

    for e in range(6):

        erows = [
            r
            for r in top256
            if int(
                r["expert_id"]
            ) == e
        ]

        require(
            len(erows) == K,
            f"{sid} E{e}: top256 count "
            f"{len(erows)} != {K}",
        )

        ranks = sorted(
            int(r["rank"])
            for r in erows
        )

        require(
            ranks == list(
                range(
                    1,
                    K + 1,
                )
            ),
            f"{sid} E{e}: invalid "
            "top256 ranks",
        )

        patch_ids = [
            int(
                r["patch_index"]
            )
            for r in erows
        ]

        require(
            len(set(patch_ids))
            == K,
            f"{sid} E{e}: duplicate "
            "top256 patches",
        )


    # --------------------------------------------------------
    # 2450-concept integrity for every expert
    # --------------------------------------------------------

    concept_sets = []


    for e in range(6):

        erows = [
            r
            for r in concept_all
            if int(
                r["expert_id"]
            ) == e
        ]

        require(
            len(erows) == 2450,
            f"{sid} E{e}: concept count "
            f"{len(erows)}",
        )


        ids = [
            r["concept_id"]
            for r in erows
        ]

        require(
            len(set(ids))
            == 2450,
            f"{sid} E{e}: duplicate "
            "concept IDs",
        )


        ranks = sorted(
            int(r["rank"])
            for r in erows
        )

        require(
            ranks
            == list(
                range(
                    1,
                    2451,
                )
            ),
            f"{sid} E{e}: invalid "
            "concept ranks",
        )


        for r in erows:

            s1 = float(
                r[
                    "similarity_topk_mean"
                ]
            )

            s2 = float(
                r[
                    "similarity_topk_contrastive"
                ]
            )

            require(
                finite(s1)
                and finite(s2),
                f"{sid} E{e}: nonfinite "
                "concept score",
            )

            require(
                -1.00001 <= s1 <= 1.00001,
                f"{sid} E{e}: invalid "
                f"cosine {s1}",
            )

            require(
                -1.00001 <= s2 <= 1.00001,
                f"{sid} E{e}: invalid "
                f"contrastive cosine {s2}",
            )


        concept_sets.append(
            set(ids)
        )


    reference_set = (
        concept_sets[0]
    )

    for e in range(
        1,
        6,
    ):

        require(
            concept_sets[e]
            == reference_set,
            f"{sid}: concept bank "
            f"differs for E{e}",
        )


    # --------------------------------------------------------
    # Top-50 integrity
    # --------------------------------------------------------

    for e in range(6):

        erows = [
            r
            for r in concept_top50
            if int(
                r["expert_id"]
            ) == e
        ]

        require(
            len(erows) == 50,
            f"{sid} E{e}: top50 count "
            f"{len(erows)}",
        )


        ranks = sorted(
            int(
                r["rank"]
            )
            for r in erows
        )


        require(
            ranks
            == list(
                range(
                    1,
                    51,
                )
            ),
            f"{sid} E{e}: top50 "
            "rank mismatch",
        )


    # --------------------------------------------------------
    # Return compact slide summary
    # --------------------------------------------------------

    route = {
        int(r["expert_id"]):
            float(
                r[
                    "mean_route_weight_all_patches"
                ]
            )
        for r in expert_summary
    }


    return {
        "sample_id":
            sid,
        "n_patches":
            P,
        "route":
            route,
    }


def main():

    print(
        "==============================================="
    )

    print(
        "RUN-2 ALL-23 INTERPRETABILITY OUTPUT AUDIT"
    )

    print(
        "==============================================="
    )

    print()


    results = []
    failures = []


    for sid in SLIDES:

        try:

            result = audit_slide(
                sid
            )

            results.append(
                result
            )

            route_text = " ".join(
                f"E{e}="
                f"{result['route'][e]:.4f}"
                for e in range(6)
            )

            print(
                f"PASS  {sid:8s} "
                f"P={result['n_patches']:5d}  "
                f"{route_text}"
            )

        except Exception as exc:

            failures.append(
                (
                    sid,
                    str(exc),
                )
            )

            print(
                f"FAIL  {sid:8s} "
                f"{exc}"
            )


    print()
    print(
        "==============================================="
    )

    print(
        f"PASS: {len(results)}/23"
    )

    print(
        f"FAIL: {len(failures)}/23"
    )

    print(
        "==============================================="
    )


    if failures:

        print()

        for sid, error in failures:

            print(
                f"{sid}: {error}"
            )

        raise SystemExit(1)


    total_patches = sum(
        r[
            "n_patches"
        ]
        for r in results
    )


    print()
    print(
        f"TOTAL PATCHES AUDITED: "
        f"{total_patches}"
    )

    print()

    print(
        "ALL 23 SLIDES HAVE:"
    )

    print(
        "  ✓ exact frozen Run-2 protocol"
    )

    print(
        "  ✓ 10 router contexts "
        "(5 seeds × 2 Euler states)"
    )

    print(
        "  ✓ complete router outputs"
    )

    print(
        "  ✓ six expert rankings"
    )

    print(
        "  ✓ Top-256 affinity sets"
    )

    print(
        "  ✓ all 2450 CONCH concepts "
        "for all six experts"
    )

    print(
        "  ✓ output SHA256 verification"
    )

    print()
    print(
        "ALL23 ARTIFACT AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
