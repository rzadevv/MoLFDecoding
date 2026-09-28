"""Build the Run-2 summary tables from the raw run outputs in results/.

Writes results/summary/:
  training_history.csv      per-epoch losses and learning rates (from the training log)
  protocol_selection.csv    every validation inference protocol that was evaluated
  test_metrics.csv          final held-out test metrics, mean and sample SD over 5 seeds
  validation_vs_test.csv    selected protocol (Euler, 2 steps, CFG 1.5), validation vs. test
  expert_lexicon.csv        top-10 CONCH concepts per expert with natural routing weight

Every recomputed number that also appears in the frozen Run-2 record
(results/README.md) is checked against it; the script fails on mismatch.

Usage: python scripts/summarize_results.py
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "summary"

SELECTED = {"method": "euler", "num_steps": 2, "guidance_scale": 1.5}
METRICS = {
    "masked_mse_micro": "Masked MSE",
    "masked_rmse_micro": "Masked RMSE",
    "masked_mae_micro": "Masked MAE",
    "mean_slide_mean_gene_pcc": "Mean slide PCC",
    "median_slide_mean_gene_pcc": "Median slide PCC",
    "mean_gene_mean_slide_pcc": "Mean gene PCC",
}
# Values reported in the frozen Run-2 record (mean, sample SD over 5 seeds).
FROZEN_TEST = {
    "masked_mse_micro": (0.369907283, 0.000197512),
    "masked_rmse_micro": (0.608200018, 0.000162363),
    "masked_mae_micro": (0.367716582, 0.000088076),
    "mean_slide_mean_gene_pcc": (0.246431064, 0.000409378),
    "median_slide_mean_gene_pcc": (0.184601000, 0.001584148),
    "mean_gene_mean_slide_pcc": (0.166238990, 0.000525587),
}
# Five-seed validation result of the selected protocol
# (results/evaluation/FINAL_INFERENCE_PROTOCOL.txt).
FROZEN_VAL = {
    "masked_mse_micro": (0.246389, 0.000091),
    "masked_mae_micro": (0.297388, 0.000050),
    "mean_slide_mean_gene_pcc": (0.223595, 0.000585),
    "median_slide_mean_gene_pcc": (0.159889, 0.003779),
    "mean_gene_mean_slide_pcc": (0.193725, 0.000523),
}
FROZEN_BEST_EPOCH, FROZEN_BEST_VAL = 239, 1.013579
MIN_DELTA = 1e-4  # early_stop_min_delta in conch_molf/configs/training_config_conch_run2.yml

EPOCH_RE = re.compile(
    r"^\[INFO_TRAIN\] Epoch (\d+): train=([\d.]+), val=([\d.]+), "
    r"val_mc_sd=([\d.]+), lr_main=([\d.e+-]+), lr_gate=([\d.e+-]+)"
)
BEST_RE = re.compile(r"^\[INFO_TRAIN\] Saved new best model at epoch (\d+)")


def training_history() -> pd.DataFrame:
    """Parse per-epoch losses from the training log and verify checkpoint selection."""
    rows, best_epochs = [], set()
    with gzip.open(RESULTS / "training" / "training_run.txt.gz", "rt") as f:
        for line in f:
            if m := EPOCH_RE.match(line):
                rows.append([int(m[1]), *map(float, m.group(2, 3, 4, 5, 6))])
            elif m := BEST_RE.match(line):
                best_epochs.add(int(m[1]))
    df = pd.DataFrame(
        rows, columns=["epoch", "train_loss", "val_loss", "val_mc_sd", "lr_main", "lr_gate"]
    )
    df["new_best"] = df["epoch"].isin(best_epochs)
    assert df["epoch"].is_unique and df["epoch"].tolist() == list(range(1, len(df) + 1))

    # Checkpoint selection rule from conch_molf/train.py: a new best requires an
    # improvement larger than early_stop_min_delta over the previous best.
    best_val, replayed = float("inf"), list[int]()
    for epoch, val in zip(df["epoch"], df["val_loss"], strict=True):
        if val < best_val - MIN_DELTA:
            best_val, replayed = val, [*replayed, epoch]
    assert replayed == sorted(best_epochs), "selection rule does not reproduce the log"
    assert replayed[-1] == FROZEN_BEST_EPOCH and round(best_val, 6) == FROZEN_BEST_VAL
    return df


def load_runs() -> pd.DataFrame:
    """Collect the summary of every validation and test inference run."""
    rows = []
    for path in sorted((RESULTS / "evaluation" / "runs").glob("*/summary.json")):
        s = json.loads(path.read_text())
        stage = path.parent.name.split("_")[0]
        rows.append(
            {
                "run": path.parent.name,
                "stage": "final_test" if stage == "final" else stage,
                "split": s["evaluation_split"],
                "method": s["method"],
                "num_steps": s["num_steps"],
                "guidance_scale": s["guidance_scale"],
                "noise_seed": s["noise_seed"],
                "n_slides": s["n_slides_evaluated"],
                **{k: s[k] for k in METRICS},
            }
        )
    return pd.DataFrame(rows)


def is_selected(df: pd.DataFrame) -> pd.Series:
    """Mask of runs that use the frozen inference protocol."""
    return (
        (df["method"] == SELECTED["method"])
        & (df["num_steps"] == SELECTED["num_steps"])
        & (df["guidance_scale"] == SELECTED["guidance_scale"])
    )


def test_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Final test metrics (mean and sample SD over 5 seeds), checked against the record."""
    test = runs[(runs["split"] == "test") & is_selected(runs)]
    assert len(test) == 5 and (test["n_slides"] == 23).all()
    out = pd.DataFrame(
        {
            "metric": [METRICS[k] for k in METRICS],
            "key": list(METRICS),
            "mean": [test[k].mean() for k in METRICS],
            "sample_sd": [test[k].std(ddof=1) for k in METRICS],
            "n_seeds": len(test),
        }
    )
    for _, r in out.iterrows():
        mean, sd = FROZEN_TEST[r["key"]]
        assert abs(r["mean"] - mean) < 5e-9 and abs(r["sample_sd"] - sd) < 5e-9, r
    return out


def validation_vs_test(runs: pd.DataFrame) -> pd.DataFrame:
    """Selected-protocol metrics on validation and test, 5 seeds each."""
    rows = []
    for split, n_slides in [("val", 96), ("test", 23)]:
        sel = runs[(runs["split"] == split) & (runs["n_slides"] == n_slides) & is_selected(runs)]
        assert len(sel) == 5, (split, len(sel))
        for k, name in METRICS.items():
            mean, sd = sel[k].mean(), sel[k].std(ddof=1)
            if split == "val" and k in FROZEN_VAL:
                assert (round(mean, 6), round(sd, 6)) == FROZEN_VAL[k], (k, mean, sd)
            rows.append(
                {
                    "split": split,
                    "n_slides": n_slides,
                    "n_seeds": len(sel),
                    "metric": name,
                    "mean": mean,
                    "sample_sd": sd,
                }
            )
    return pd.DataFrame(rows)


def expert_lexicon(top_n: int = 10) -> pd.DataFrame:
    """Top concepts per expert from the cross-slide aggregate, with route weights."""
    agg = RESULTS / "interpretability" / "aggregate"
    sem = pd.read_csv(agg / "semantic_global_top50.csv")
    route = pd.read_csv(agg / "routing_global_summary.csv")[
        ["expert_id", "slide_macro_mean_route_weight"]
    ]
    lex = sem[sem["global_mean_rank"] <= top_n][
        [
            "expert_id",
            "global_mean_rank",
            "concept_name",
            "tier",
            "mean_contrastive_similarity",
            "sd_contrastive_similarity",
            "top10_count",
            "n_slides",
        ]
    ].rename(columns={"global_mean_rank": "rank", "top10_count": "slides_in_top10"})
    return lex.merge(route, on="expert_id").sort_values(["expert_id", "rank"])


def main() -> None:
    """Write all summary tables to results/summary/."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    tables = {
        "training_history.csv": training_history(),
        "protocol_selection.csv": runs[runs["split"] == "val"].drop(columns="split"),
        "test_metrics.csv": test_metrics(runs).drop(columns="key"),
        "validation_vs_test.csv": validation_vs_test(runs),
        "expert_lexicon.csv": expert_lexicon(),
    }
    for name, df in tables.items():
        df.to_csv(OUT / name, index=False, float_format="%.9g")
        print(f"wrote {OUT.relative_to(ROOT) / name} ({len(df)} rows)")
    print("All recomputed values match the frozen Run-2 record.")


if __name__ == "__main__":
    main()
