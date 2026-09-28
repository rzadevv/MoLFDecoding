"""Re-rank the expert concepts without concepts that cannot be valid H&E findings.

The concept list contains Cell Ontology entries that do not exist in human tissue,
cytology-only findings, and cell subsets defined only by molecular markers (see
docs/concept_bank.md). A concept's score does not depend on the other concepts,
so the rankings can be recomputed without them from the per-slide scores in
results/interpretability/<sample_id>/concept_scores_all2450.csv; the frozen
results are not changed.

The script first recomputes the frozen aggregate (global ranks, per-slide top-10
counts, per-pair and mean cross-slide stability) with no concepts removed and stops if it does not
match results/interpretability/aggregate/. It then writes:

  data/concept_exclusions.csv                     excluded concepts with reasons
  results/summary/expert_lexicon_filtered.csv     top-10 concepts per expert
  results/summary/semantic_stability_filtered.csv cross-slide stability per expert
  results/summary/semantic_slide_pairs_filtered.csv the same, per slide pair

Usage: python scripts/filter_concepts.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERP = ROOT / "results" / "interpretability"
AGG = INTERP / "aggregate"
SUMMARY = ROOT / "results" / "summary"

# Species-restricted Cell Ontology labels: mouse-only "(mmus)" and "(sensu X)" for
# non-vertebrate taxa. "(sensu vertebrata)" applies to humans and is kept.
ORGANISM_RESTRICTED = re.compile(r"\(mmus\)|\(sensu (?!vertebrata|mammalia)[^)]*\)", re.IGNORECASE)

# Cell types defined by surface markers, transcripts or functional state; on H&E
# they are indistinguishable from their parent cell type. Applied to cell types only.
MARKER_DEFINED = re.compile(
    r"\bCD\d+|alpha-beta|gamma-delta|\w+-(positive|negative|low|high)\b|\bGr1\b|\bsca1\b|\bIge\b"
    r"|regulatory T|T regulatory|\bTreg\b|\bTr1\b|\bTc1\b|T-helper|helper T|\bTh\d+\b"
    r"|memory (T|B)|naive (T|regulatory)|effector T|exhausted T|cytotoxic T|NK T|invariant T"
    r"|innate lymphoid|\bIlc1\b|natural helper lymphocyte|unswitched memory|CD56(bright|dim)",
    re.IGNORECASE,
)

# Cell types from other organisms, identified by reading all Cell Ontology entries.
NON_HUMAN = {
    "Adaxial cell": "zebrafish embryonic somite",
    "Amnioserosal cell": "Drosophila embryo",
    "Arcade cell": "C. elegans pharynx",
    "Bottle cell": "amphibian gastrula",
    "Chorionic girdle cell": "horse placenta",
    "Coelomocyte": "invertebrate body cavity",
    "Collar cell": "sponge",
    "Cystoblast": "Drosophila germline",
    "Ecdysteroid secreting cell": "insect",
    "Eggshell secreting cell": "insect",
    "Embryonic plasmatocyte": "Drosophila blood cell",
    "Excretory cell": "C. elegans",
    "Fat body cell": "insect",
    "Ganglion mother cell": "Drosophila neurogenesis",
    "Garland cell": "Drosophila",
    "Interneuromast cell": "fish lateral line",
    "Kenyon cell": "insect brain",
    "Larval midgut cell": "insect larva",
    "Lymph gland plasmatocyte": "Drosophila blood cell",
    "Malpighian tubule tip cell": "insect",
    "Morula cell": "ascidian",
    "Nephrocyte": "Drosophila",
    "Neuromast supporting cell": "fish lateral line",
    "Nucleate erythrocyte": "non-mammalian; mammalian erythrocytes are anucleate",
    "Nucleated thrombocyte": "non-mammalian; mammals have platelets",
    "Oenocyte": "insect",
    "Pericardial nephrocyte": "Drosophila",
    "Procrystal cell": "Drosophila blood cell",
    "Pronephric nephron tubule epithelial cell": "embryonic kidney of fish and amphibians",
    "Pronephric podocyte": "embryonic kidney of fish and amphibians",
    "Scolopale cell": "insect sensory organ",
    "Spongiotrophoblast cell": "rodent placenta",
    "Xanthophore cell": "fish and amphibian pigment cell",
}
CYTOLOGY_ONLY = {"Clue cell": "cervicovaginal smear finding, not a tissue feature"}

TOP_N = 10


def load_scores() -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-slide contrastive scores as an array [expert, slide, concept], plus frozen ranks."""
    slides = sorted(p.parent.name for p in INTERP.glob("*/concept_scores_all2450.csv"))
    first = pd.read_csv(INTERP / slides[0] / "concept_scores_all2450.csv")
    concepts = first[first.expert_id == 0][["concept_id", "concept_name", "tier"]].reset_index(
        drop=True
    )
    ids = concepts.concept_id.tolist()
    n_exp = first.expert_id.nunique()
    scores = np.zeros((n_exp, len(slides), len(ids)))
    ranks = np.zeros_like(scores, dtype=np.int64)
    for s, sid in enumerate(slides):
        df = pd.read_csv(INTERP / sid / "concept_scores_all2450.csv")
        for e in range(n_exp):
            d = df[df.expert_id == e].set_index("concept_id").loc[ids]
            scores[e, s] = d.similarity_topk_contrastive.to_numpy()
            ranks[e, s] = d["rank"].to_numpy()
    return slides, concepts, scores, ranks


def exclusions(concepts: pd.DataFrame) -> pd.DataFrame:
    """Apply the exclusion rules; every excluded concept gets a category and a reason."""
    rows = []
    for _, c in concepts.iterrows():
        name = c.concept_name
        if name in NON_HUMAN:
            rows.append((c.concept_id, name, c.tier, "non-human", NON_HUMAN[name]))
        elif ORGANISM_RESTRICTED.search(name):
            rows.append((c.concept_id, name, c.tier, "non-human", "species-restricted label"))
        elif name in CYTOLOGY_ONLY:
            rows.append((c.concept_id, name, c.tier, "cytology only", CYTOLOGY_ONLY[name]))
        elif c.tier == "H2_cell_type" and MARKER_DEFINED.search(name):
            rows.append(
                (c.concept_id, name, c.tier, "marker-defined", "not distinguishable on H&E")
            )
    missing = (set(NON_HUMAN) | set(CYTOLOGY_ONLY)) - set(concepts.concept_name)
    assert not missing, f"listed concepts not found: {sorted(missing)}"
    return pd.DataFrame(rows, columns=["concept_id", "concept_name", "tier", "category", "reason"])


def ranks_desc(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each concept within each (expert, slide), highest score first."""
    order = np.argsort(-scores, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, scores.shape[-1] + 1), axis=-1)
    return ranks


def global_table(concepts: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Mean score, per-slide top-10 counts and global rank for every expert and concept."""
    ranks = ranks_desc(scores)
    rows = []
    for e in range(scores.shape[0]):
        d = concepts.copy()
        d["expert_id"] = e
        d["mean_contrastive_similarity"] = scores[e].mean(axis=0)
        d["sd_contrastive_similarity"] = scores[e].std(axis=0, ddof=1)
        d["slides_in_top10"] = (ranks[e] <= TOP_N).sum(axis=0)
        d["n_slides"] = scores.shape[1]
        d = d.sort_values("mean_contrastive_similarity", ascending=False, kind="stable")
        d["rank"] = np.arange(1, len(d) + 1)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def slide_pairs(slides: list[str], scores: np.ndarray) -> pd.DataFrame:
    """Pearson and Spearman correlation of concept profiles for every slide pair and expert."""
    tech = ["Xenium" if s.startswith("TENX") else "Visium" for s in slides]
    ranked = pd.DataFrame(scores.reshape(-1, scores.shape[-1])).rank(axis=1).to_numpy()
    ranked = ranked.reshape(scores.shape)
    rows = []
    for e in range(scores.shape[0]):
        for i in range(len(slides)):
            for j in range(i + 1, len(slides)):
                rows.append(
                    {
                        "expert_id": e,
                        "slide_a": slides[i],
                        "slide_b": slides[j],
                        "pair_type": (
                            f"{tech[i]}-{tech[j]}" if tech[i] == tech[j] else "cross-technology"
                        ),
                        "pearson": np.corrcoef(scores[e, i], scores[e, j])[0, 1],
                        "spearman": np.corrcoef(ranked[e, i], ranked[e, j])[0, 1],
                    }
                )
    return pd.DataFrame(rows)


def stability_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean correlations per expert, over all slide pairs and per pair type."""
    rows = []
    for e, df in pairs.groupby("expert_id"):
        for kind in ["all", "Visium-Visium", "Xenium-Xenium", "cross-technology"]:
            sub = df if kind == "all" else df[df.pair_type == kind]
            rows.append(
                {
                    "expert_id": e,
                    "pair_type": kind,
                    "n_pairs": len(sub),
                    "mean_pearson": sub.pearson.mean(),
                    "mean_spearman": sub.spearman.mean(),
                }
            )
    return pd.DataFrame(rows)


def check_against_frozen(
    concepts: pd.DataFrame, scores: np.ndarray, ranks: np.ndarray, slides: list[str]
) -> None:
    """Stop unless the unfiltered recomputation matches the frozen aggregate."""
    # The per-slide analysis used an unstable sort, so concepts with identical scores
    # may appear in either order; everything else must match exactly.
    mine = ranks_desc(scores)
    for e, s, c in np.argwhere(mine != ranks):
        tied = scores[e, s] == scores[e, s, c]
        assert tied.sum() > 1, "per-slide ranks do not match"
        assert set(mine[e, s, tied]) == set(ranks[e, s, tied]), "per-slide ranks do not match"
    ours = global_table(concepts, scores).set_index(["expert_id", "concept_id"])
    ref = pd.read_csv(AGG / "semantic_global_all2450.csv").set_index(["expert_id", "concept_id"])
    ref = ref.loc[ours.index]
    assert (ours["rank"] == ref["global_mean_rank"]).all(), "global ranks do not match"
    assert (ours["slides_in_top10"] == ref["top10_count"]).all(), "top-10 counts do not match"
    assert np.allclose(ours.mean_contrastive_similarity, ref.mean_contrastive_similarity)
    pairs = slide_pairs(slides, scores)
    ref_pairs = pd.read_csv(AGG / "semantic_slide_pair_stability.csv")
    merged = pairs.merge(ref_pairs, on=["expert_id", "slide_a", "slide_b"], suffixes=("", "_ref"))
    assert len(merged) == len(pairs) == len(ref_pairs), "slide pairs do not match"
    for col in ("pearson", "spearman"):
        assert np.allclose(merged[col], merged[f"{col}_ref"]), f"pair {col} does not match"
    stab = stability_summary(pairs).set_index(["expert_id", "pair_type"])
    ref_stab = pd.read_csv(AGG / "semantic_stability_summary.csv").set_index(
        ["expert_id", "pair_type"]
    )
    for col in ("mean_pearson", "mean_spearman"):
        assert np.allclose(stab[col], ref_stab.loc[stab.index, col]), f"{col} does not match"


def main() -> None:
    """Build the exclusion list, validate the recomputation, and write the filtered tables."""
    slides, concepts, scores, ranks = load_scores()
    check_against_frozen(concepts, scores, ranks, slides)
    print("Unfiltered recomputation matches the frozen aggregate.")

    excluded = exclusions(concepts)
    excluded.sort_values(["category", "concept_name"]).to_csv(
        ROOT / "data" / "concept_exclusions.csv", index=False
    )
    keep = ~concepts.concept_id.isin(excluded.concept_id).to_numpy()
    kept, kept_scores = concepts[keep].reset_index(drop=True), scores[:, :, keep]

    route = pd.read_csv(AGG / "routing_global_summary.csv")[
        ["expert_id", "slide_macro_mean_route_weight"]
    ]
    lexicon = global_table(kept, kept_scores)
    lexicon = lexicon[lexicon["rank"] <= TOP_N].merge(route, on="expert_id")
    lexicon = lexicon[
        [
            "expert_id",
            "rank",
            "concept_name",
            "tier",
            "mean_contrastive_similarity",
            "sd_contrastive_similarity",
            "slides_in_top10",
            "n_slides",
            "slide_macro_mean_route_weight",
        ]
    ].sort_values(["expert_id", "rank"])
    lexicon.to_csv(SUMMARY / "expert_lexicon_filtered.csv", index=False, float_format="%.9g")
    pairs = slide_pairs(slides, kept_scores)
    pairs.to_csv(SUMMARY / "semantic_slide_pairs_filtered.csv", index=False, float_format="%.9g")
    stability_summary(pairs).to_csv(
        SUMMARY / "semantic_stability_filtered.csv", index=False, float_format="%.9g"
    )

    counts = excluded.category.value_counts().to_dict()
    print(f"Excluded {len(excluded)} of {len(concepts)} concepts: {counts}")
    print(f"Kept {int(keep.sum())}; wrote data/concept_exclusions.csv and results/summary/")


if __name__ == "__main__":
    main()
