"""Generate the Run-2 figures in figures/ from the result tables.

Inputs are the tables in results/summary/ (written by summarize_results.py and
filter_concepts.py) and the frozen outputs in results/. Each figure is saved as
PNG and PDF.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
AGG = RESULTS / "interpretability" / "aggregate"
FIG = ROOT / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TECH_COLOR = {"Visium": BLUE, "Xenium": ORANGE}
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelcolor": TEXT,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` as PNG and PDF under figures/ and close it."""
    for ext in ("png", "pdf"):
        # No creation timestamp in PDFs, so re-running the script is byte-reproducible.
        metadata = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white", metadata=metadata)
    plt.close(fig)
    print(f"wrote figures/{name}.png|pdf")


def fig1_training_curve() -> None:
    """Fig. 1: training and validation loss with the selected checkpoint."""
    h = pd.read_csv(SUMMARY / "training_history.csv")
    best = h[h["new_best"]].iloc[-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(h["epoch"], h["train_loss"], color=BLUE, lw=1.6, label="Training")
    ax.plot(h["epoch"], h["val_loss"], color=ORANGE, lw=1.6, label="Validation (fixed-MC)")
    ax.axvline(best["epoch"], color=MUTED, lw=0.9, ls="--")
    ax.scatter(
        [best["epoch"]],
        [best["val_loss"]],
        s=36,
        color=ORANGE,
        zorder=3,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.annotate(
        f"selected checkpoint\nepoch {int(best['epoch'])}, val {best['val_loss']:.4f}",
        (best["epoch"], best["val_loss"]),
        xytext=(-150, 42),
        textcoords="offset points",
        color=TEXT,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (flow + gene reconstruction)")
    ax.set_xlim(0, h["epoch"].max() + 2)
    ax.set_title("CONCH-MoLF Run-2 training (early stopping, patience 80)", loc="left")
    ax.legend(loc="upper right")
    save(fig, "fig1_training_curve")


def fig2_protocol_selection() -> None:
    """Fig. 2: validation MSE vs. PCC for all evaluated inference protocols."""
    p = pd.read_csv(SUMMARY / "protocol_selection.csv")
    p = p[p["n_slides"] == 96]
    g = (
        p.groupby(["method", "num_steps", "guidance_scale"])[
            ["masked_mse_micro", "mean_slide_mean_gene_pcc"]
        ]
        .agg(["mean", "count"])
        .reset_index()
    )
    g.columns = ["method", "steps", "cfg", "mse", "n", "pcc", "_"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    solver = g[g["cfg"] == 1.0]
    cfg = g[(g["method"] == "euler") & (g["steps"] == 2)].sort_values("cfg")
    ax.scatter(
        solver["mse"],
        solver["pcc"],
        s=40,
        color=MUTED,
        zorder=3,
        label="Solver / step count (CFG 1.0)",
    )
    offsets = {5: (-8, 5), 10: (-8, 5), 25: (-10, 10), 50: (-4, -16)}
    for _, r in solver[solver["method"] == "euler"].iterrows():
        if r["steps"] != 2:
            ax.annotate(
                f"Euler {int(r['steps'])} steps",
                (r["mse"], r["pcc"]),
                xytext=offsets[int(r["steps"])],
                textcoords="offset points",
                fontsize=7.5,
                color=MUTED,
                ha="right",
            )
    rk4 = solver[solver["method"] == "rk4"]
    ax.annotate(
        f"RK4 {'/'.join(str(int(s)) for s in rk4['steps'])} steps",
        (rk4["mse"].mean(), rk4["pcc"].mean()),
        xytext=(8, -3),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
        ha="left",
    )
    ax.set_xlim(right=rk4["mse"].max() + 0.0075)
    ax.plot(cfg["mse"], cfg["pcc"], color=BLUE, lw=1.6, zorder=2)
    ax.scatter(
        cfg["mse"],
        cfg["pcc"],
        s=44,
        color=BLUE,
        zorder=3,
        edgecolor="white",
        linewidth=1.5,
        label="Euler 2 steps, CFG sweep",
    )
    for _, r in cfg.iterrows():
        ax.annotate(
            f"CFG {r['cfg']:g}",
            (r["mse"], r["pcc"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=TEXT,
        )
    sel = cfg[cfg["cfg"] == 1.5].iloc[0]
    ax.scatter(
        [sel["mse"]],
        [sel["pcc"]],
        s=150,
        facecolor="none",
        edgecolor=ORANGE,
        lw=2,
        zorder=4,
        label="Selected protocol",
    )
    ax.set_xlabel("Masked MSE (lower is better)")
    ax.set_ylabel("Mean slide PCC (higher is better)")
    ax.set_title("Inference-protocol selection on 96 validation slides", loc="left")
    ax.legend(loc="center left", fontsize=8, bbox_to_anchor=(0.0, 0.42))
    save(fig, "fig2_protocol_selection")


def fig3_validation_vs_test() -> None:
    """Fig. 3: selected protocol on validation vs. held-out test."""
    v = pd.read_csv(SUMMARY / "validation_vs_test.csv")
    panels = [
        ("Error (lower is better)", ["Masked MSE", "Masked MAE"]),
        ("Correlation (higher is better)", ["Mean slide PCC", "Median slide PCC", "Mean gene PCC"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [2, 3]})
    for ax, (title, metrics) in zip(axes, panels, strict=True):
        x = np.arange(len(metrics))
        for i, (split, color, label) in enumerate(
            [("val", BLUE, "Validation (96 slides)"), ("test", ORANGE, "Held-out test (23 slides)")]
        ):
            d = v[v["split"] == split].set_index("metric").loc[metrics]
            bars = ax.bar(
                x + (i - 0.5) * 0.38,
                d["mean"],
                width=0.36,
                color=color,
                label=label,
                yerr=d["sample_sd"],
                error_kw={"lw": 0.8, "ecolor": TEXT},
            )
            ax.bar_label(
                bars, labels=[f"{m:.3f}" for m in d["mean"]], padding=2, fontsize=7.5, color=TEXT
            )
        ax.set_xticks(x, metrics)
        ax.set_title(title, loc="left")
        ax.margins(y=0.15)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean over 5 noise seeds")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Frozen protocol (Euler, 2 steps, CFG 1.5): validation vs. held-out test",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig3_validation_vs_test")


def fig4_per_slide_pcc() -> None:
    """Fig. 4: per-slide test PCC, coloured by platform."""
    d = pd.read_csv(RESULTS / "evaluation" / "FINAL_TEST_PER_SLIDE_5SEED.csv")
    meta = pd.read_csv(RESULTS / "evaluation" / "FINAL_TEST_METADATA_JOIN.csv")[
        ["sample_id", "st_technology", "oncotree_code_filled", "organ"]
    ]
    d = d.merge(meta, on="sample_id").sort_values("mean_gene_pcc_mean")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    y = np.arange(len(d))
    ax.barh(
        y,
        d["mean_gene_pcc_mean"],
        xerr=d["mean_gene_pcc_sd"],
        height=0.7,
        color=[TECH_COLOR[t] for t in d["st_technology"]],
        error_kw={"lw": 0.8, "ecolor": TEXT},
    )
    ax.set_yticks(
        y,
        [f"{s}  ({o})" for s, o in zip(d["sample_id"], d["oncotree_code_filled"], strict=True)],
        fontsize=7.5,
    )
    ax.axvline(d["mean_gene_pcc_mean"].mean(), color=MUTED, lw=0.9, ls="--")
    ax.annotate(
        f"mean {d['mean_gene_pcc_mean'].mean():.3f}",
        (d["mean_gene_pcc_mean"].mean(), len(d) - 0.6),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TECH_COLOR.values()]
    ax.legend(handles, TECH_COLOR.keys(), loc="lower right", title="Platform", fontsize=8)
    ax.set_xlabel("Mean gene PCC per slide (mean ± SD over 5 seeds)")
    ax.set_title("Held-out test performance by slide", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_per_slide_pcc")


def fig5_routing() -> None:
    """Fig. 5: mean route weight per expert and slide, with slide means."""
    r = pd.read_csv(AGG / "routing_by_slide.csv")
    m = r.pivot(index="sample_id", columns="expert_id", values="mean_route_weight")
    tech = r.drop_duplicates("sample_id").set_index("sample_id")["technology"]
    order = sorted(m.index, key=lambda s: (tech[s], s))
    m = m.loc[order]
    g = pd.read_csv(AGG / "routing_global_summary.csv").set_index("expert_id")
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(5.6, 7.4), gridspec_kw={"height_ratios": [5, 1.1]}, sharex=True
    )
    im = ax.imshow(m.values, cmap=SEQ, vmin=0, vmax=max(0.7, m.values.max()), aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if val > 0.4 else TEXT,
            )
    ax.set_yticks(range(len(m)), [f"{s}" for s in m.index], fontsize=7.5)
    n_vis = int((tech[m.index] == "Visium").sum())
    ax.axhline(n_vis - 0.5, color="white", lw=3)
    ax.text(
        m.shape[1] - 0.45,
        n_vis / 2 - 0.5,
        "Visium",
        rotation=270,
        va="center",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        m.shape[1] - 0.45,
        n_vis + (len(m) - n_vis) / 2 - 0.5,
        "Xenium",
        rotation=270,
        va="center",
        fontsize=8,
        color=MUTED,
    )
    ax.grid(False)
    ax.set_title("Natural routing: mean route weight per expert and slide", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.08)
    cb.set_label("Mean route weight", color=MUTED)
    cb.outline.set_visible(False)  # type: ignore[operator]  # matplotlib stubs mistype outline
    w = g["slide_macro_mean_route_weight"]
    bars = axb.bar(
        range(6),
        w,
        yerr=g["slide_macro_sd_route_weight"],
        color=BLUE,
        width=0.6,
        error_kw={"lw": 0.8, "ecolor": TEXT},
    )
    axb.bar_label(bars, labels=[f"{x:.3f}" for x in w], padding=2, fontsize=7.5, color=TEXT)
    axb.set_xticks(range(6), [f"Expert {i}" for i in range(6)])
    axb.set_ylabel("Slide mean")
    axb.set_ylim(0, max(0.62, (w + g["slide_macro_sd_route_weight"]).max() * 1.1))
    axb.grid(axis="x", visible=False)
    save(fig, "fig5_routing")


def _lexicon_figure(table: str, title: str, name: str, top_n: int = 5) -> None:
    """Top CONCH concepts per expert from a lexicon table in results/summary/."""
    lex = pd.read_csv(SUMMARY / table)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
    for e, ax in zip(range(6), axes.ravel(), strict=True):
        d = lex[(lex["expert_id"] == e) & (lex["rank"] <= top_n)].sort_values(
            "rank", ascending=False
        )
        y = np.arange(len(d))
        ax.barh(
            y,
            d["mean_contrastive_similarity"],
            xerr=d["sd_contrastive_similarity"],
            height=0.6,
            color=BLUE,
            error_kw={"lw": 0.7, "ecolor": MUTED},
        )
        names = [
            f"{n if len(n) <= 30 else n[:28] + '…'}  ({int(k)}/23)"
            for n, k in zip(d["concept_name"], d["slides_in_top10"], strict=True)
        ]
        ax.set_yticks(y, names, fontsize=7.5)
        w = d["slide_macro_mean_route_weight"].iloc[0]
        ax.set_title(f"Expert {e}  ·  route weight {w:.3f}", loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        title,
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
    fig.supxlabel(
        "Mean contrastive CONCH similarity (± SD over 23 slides); "
        "(n/23) = slides on which the concept is in the expert's top 10",
        fontsize=8.5,
        color=TEXT,
    )
    fig.tight_layout()
    save(fig, name)


def fig6_expert_lexicon() -> None:
    """Fig. 6: top CONCH concepts per expert, all 2,450 concepts (frozen result)."""
    _lexicon_figure(
        "expert_lexicon.csv",
        "Top CONCH concepts per expert, aggregated over the 23 held-out slides (all concepts)",
        "fig6_expert_lexicon",
    )


def fig6b_expert_lexicon_filtered() -> None:
    """Fig. 6b: as Fig. 6, without the concepts in data/concept_exclusions.csv."""
    _lexicon_figure(
        "expert_lexicon_filtered.csv",
        "Top CONCH concepts per expert, excluding non-human, cytology-only and "
        "marker-defined concepts",
        "fig6b_expert_lexicon_filtered",
    )


def fig7_semantic_stability() -> None:
    """Fig. 7: cross-slide rank correlation of expert concept profiles."""
    s = pd.read_csv(AGG / "semantic_stability_summary.csv")
    types = [
        ("all", BLUE, "All slide pairs"),
        ("Visium-Visium", ORANGE, "Visium–Visium"),
        ("Xenium-Xenium", AQUA, "Xenium–Xenium"),
        ("cross-technology", YELLOW, "Visium–Xenium"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(6)
    width = 0.2
    for i, (t, color, label) in enumerate(types):
        d = s[s["pair_type"] == t].set_index("expert_id").loc[list(range(6))]
        ax.bar(
            x + (i - 1.5) * width, d["mean_spearman"], width=width - 0.02, color=color, label=label
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"Expert {i}" for i in range(6)])
    ax.set_ylabel("Mean Spearman ρ between\nslide concept profiles")
    ax.set_title("Cross-slide stability of expert concept profiles (2,450 concepts)", loc="left")
    ax.legend(ncol=4, loc="upper left", fontsize=8, bbox_to_anchor=(0, 1.0))
    ax.margins(y=0.2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig7_semantic_stability")


def main() -> None:
    """Generate all figures."""
    FIG.mkdir(exist_ok=True)
    fig1_training_curve()
    fig2_protocol_selection()
    fig3_validation_vs_test()
    fig4_per_slide_pcc()
    fig5_routing()
    fig6_expert_lexicon()
    fig6b_expert_lexicon_filtered()
    fig7_semantic_stability()


if __name__ == "__main__":
    main()
