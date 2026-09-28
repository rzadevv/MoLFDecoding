"""Markdown comparison report generator."""

from __future__ import annotations

import importlib.metadata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.comparison.merger import MergedBankResult


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt_sim(v: float) -> str:
    return f"{v:.3f}"


def _tier_table_row(tier: str, stats: dict[str, Any]) -> str:
    ref = stats["reference_total"]
    harv = stats["harvested_total"]
    matched = stats["matched"]
    ref_only = stats["reference_only"]
    harv_only = stats["harvested_only"]
    recall = stats["reference_recall"]
    return (
        f"| {tier} | {ref} | {harv} | {matched} | {ref_only} | {harv_only} | {_fmt_pct(recall)} |"
    )


def generate_report(
    result: MergedBankResult,
    config: ComparisonConfig,
    output_path: Path,
    rng_seed: int = 42,
) -> None:
    """Generate the cross-bank comparison report as a markdown document.

    Sections:
      1. Overview
      2. Per-tier coverage table
      3. Match similarity distribution
      4. Matched examples (random N per tier)
      5. Reference-only concepts (N per tier)
      6. Harvested-only concepts (N per tier)
      7. Top-K hardest reference gaps
      8. Top-K most-novel harvested additions
      9. Cross-tier matches
      10. Discussion stubs

    The report uses seeded random sampling for reproducibility.

    Args:
        result: MergedBankResult from merge_banks().
        config: ComparisonConfig controlling example counts and thresholds.
        output_path: Destination .md file.
        rng_seed: RNG seed for reproducible sampling.
    """
    rng = np.random.RandomState(rng_seed)
    s = result.summary
    by_tier: dict[str, dict[str, Any]] = s.get("by_tier", {})
    all_tiers = sorted(by_tier)

    try:
        version = importlib.metadata.version("molf-interp")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"

    now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []

    # Header
    lines += [
        "# Cross-bank comparison: reference bank vs harvested bank",
        "",
        f"**Generated:** {now}",
        f"**Reference bank:** {s['reference_total']} concepts (provenance: reference)",
        f"**Harvested bank:** {s['harvested_total']} concepts (provenance: harvested)",
        f"**Merged:** {s['merged_total']} concepts",
        f"**Matches found:** {s['matches_total']} "
        f"(within-tier: {s['matches_within_tier']}; cross-tier: {s['matches_cross_tier']})",
        f"**Overall reference recall:** {_fmt_pct(s['overall_reference_recall'])} "
        "(fraction of reference concepts with a harvested match)",
        "",
    ]

    # 1. Per-tier coverage
    lines += [
        "## 1. Per-tier coverage",
        "",
        "| Tier | Reference | Harvested | Matched | Ref-only | Harv-only | Ref recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    total_su = total_auto = total_matched = total_ref_only = total_harv_only = 0
    for tier in all_tiers:
        ts = by_tier[tier]
        lines.append(_tier_table_row(tier, ts))
        total_su += ts["reference_total"]
        total_auto += ts["harvested_total"]
        total_matched += ts["matched"]
        total_ref_only += ts["reference_only"]
        total_harv_only += ts["harvested_only"]
    overall_recall = total_matched / total_su if total_su > 0 else 0.0
    lines.append(
        f"| **Total** | **{total_su}** | **{total_auto}** | **{total_matched}** |"
        f" **{total_ref_only}** | **{total_harv_only}** | **{_fmt_pct(overall_recall)}** |"
    )
    lines.append("")

    # Cross-tier callout (always shown when matches exist)
    _ct_df = result.matches
    if not _ct_df.empty and "discordance_class" in _ct_df.columns:
        n_ct = int((_ct_df["is_cross_tier"] == True).sum())  # noqa: E712
        n_h1_ct = int((_ct_df["discordance_class"] == "cross_tier_h1_vs_h2_cell_type").sum())
        if n_ct > 0:
            lines += [
                f"> **Cross-tier matches:** {n_ct} concepts matched across different tiers"
                f" (see Section 8). {n_h1_ct} are 'cells as morphology pattern' vs"
                " cell taxonomy — a systematic tier-assignment difference between"
                " expert curation and ontology-based automation.",
                "",
            ]

    # 1a. Primary lexicon view (only when view_a data is present in summary)
    view_a: dict[str, Any] | None = s.get("view_a")
    if view_a is not None:
        va_total = view_a["total"]
        merged_total = s["merged_total"]
        va_pct = va_total / merged_total if merged_total > 0 else 0.0
        by_exc = view_a.get("by_exclusion_reason", {})
        va_by_tier: dict[str, int] = view_a.get("by_tier", {})

        lines += [
            "## 1a. Primary Lexicon View (View A) — input to Clinical Lexicon generation",
            "",
            "View A is the subset of the merged bank for Task 3.1 Clinical Lexicon"
            " generation. Concepts here are either from the reference bank, validated by"
            " cross-bank overlap, or passed label quality gates AND SapBERT similarity"
            " to the reference bank's same-tier vocabulary.",
            "",
            f"View A total: **{va_total}** concepts ({_fmt_pct(va_pct)} of merged bank)",
            "",
            "| Tier | Reference | Harv matched | Harv ref-adjacent | View A total |",
            "|---|---:|---:|---:|---:|",
        ]
        va_tier_ref_total = 0
        va_tier_matched_total = 0
        va_tier_adj_total = 0
        va_tier_total_total = 0
        for tier in all_tiers:
            tier_su_n = by_tier[tier]["reference_total"] if tier in by_tier else 0
            tier_va_n = va_by_tier.get(tier, 0)
            # matched = concepts in both reference and harvested matches for this tier
            tier_matched_n = by_tier[tier]["matched"] if tier in by_tier else 0
            tier_adj_n = max(0, tier_va_n - tier_su_n - tier_matched_n)
            lines.append(
                f"| {tier} | {tier_su_n} | {tier_matched_n} | {tier_adj_n} | {tier_va_n} |"
            )
            va_tier_ref_total += tier_su_n
            va_tier_matched_total += tier_matched_n
            va_tier_adj_total += tier_adj_n
            va_tier_total_total += tier_va_n
        lines.append(
            f"| **Total** | **{va_tier_ref_total}** | **{va_tier_matched_total}**"
            f" | **{va_tier_adj_total}** | **{va_tier_total_total}** |"
        )
        min_len = getattr(config, "exclude_labels_shorter_than", 4)
        max_tok = getattr(config, "exclude_label_token_count_above", 8)
        sim_thr = getattr(config, "reference_neighbor_similarity_threshold", 0.70)
        lines += [
            "",
            "Exclusions (automated concepts not entering View A):",
            "",
            "| Reason | Count |",
            "|---|---:|",
            f"| Label too short (< {min_len} chars) | {by_exc.get('label_too_short', 0)} |",
            f"| Label too long (> {max_tok} tokens) | {by_exc.get('label_too_long', 0)} |",
            f"| Excluded substring | {by_exc.get('excluded_substring', 0)} |",
            f"| No definition or synonyms | {by_exc.get('no_definition_no_synonyms', 0)} |",
            f"| Low ref-neighbor similarity (< {sim_thr:.2f})"
            f" | {by_exc.get('low_reference_neighbor_similarity', 0)} |",
            "",
        ]

    # 2. Match similarity distribution
    lines += [
        "## 2. Match similarity distribution",
        "",
        "| Tier | Mean | Median | p25 | p75 | p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    matches_df = result.matches
    for tier in all_tiers:
        tier_sims = (
            matches_df[matches_df["tier"] == tier]["similarity"].tolist()
            if not matches_df.empty
            else []
        )
        tier_sims_f = [float(x) for x in tier_sims]
        mean = _fmt_sim(float(np.mean(tier_sims_f)) if tier_sims_f else 0.0)
        med = _fmt_sim(_percentile(tier_sims_f, 50))
        p25 = _fmt_sim(_percentile(tier_sims_f, 25))
        p75 = _fmt_sim(_percentile(tier_sims_f, 75))
        p90 = _fmt_sim(_percentile(tier_sims_f, 90))
        lines.append(f"| {tier} | {mean} | {med} | {p25} | {p75} | {p90} |")
    lines.append("")

    # 3. Matched examples
    lines += [
        "## 3. Matched examples (random samples per tier)",
        "",
        "These are concepts where the reference label has a close harvested equivalent.",
        "",
    ]
    if not matches_df.empty:
        for tier in all_tiers:
            tier_m = matches_df[matches_df["tier"] == tier]
            if tier_m.empty:
                continue
            n = min(config.examples_per_tier, len(tier_m))
            sample = tier_m.sample(n=n, random_state=int(rng.randint(0, 2**31)))
            lines += [
                f"### {tier}",
                "",
                "| Reference label | Harvested label | Similarity |",
                "|---|---|---:|",
            ]
            for _, row in sample.iterrows():
                sim_str = f"{float(row['similarity']):.3f}"
                lines.append(f"| {row['reference_label']} | {row['harvested_label']} | {sim_str} |")
            lines.append("")

    # 4. Reference-only concepts
    lines += [
        "## 4. Reference-only concepts (random samples per tier)",
        "",
        "Concepts in the reference bank that the harvested bank did not match.",
        "These represent gaps in automated curation.",
        "",
    ]
    ref_only_df = result.merged_concepts[
        result.merged_concepts["cross_bank_status"] == "reference_only"
    ]
    for tier in all_tiers:
        tier_ref_only = ref_only_df[ref_only_df["tier"] == tier]
        if tier_ref_only.empty:
            continue
        n = min(config.examples_per_tier, len(tier_ref_only))
        sample = tier_ref_only.sample(n=n, random_state=int(rng.randint(0, 2**31)))
        lines.append(f"### {tier}")
        lines.append("")
        for _, row in sample.iterrows():
            lines.append(f"- {row['concept_name']}")
        lines.append("")

    # 5. Harvested-only concepts
    lines += [
        "## 5. Harvested-only concepts (random samples per tier)",
        "",
        "Concepts the harvested bank contains with no reference near-equivalent.",
        "These represent additions beyond the reference vocabulary.",
        "",
    ]
    harv_only_df = result.merged_concepts[
        result.merged_concepts["cross_bank_status"] == "harvested_only"
    ]
    for tier in all_tiers:
        tier_harv_only = harv_only_df[harv_only_df["tier"] == tier]
        if tier_harv_only.empty:
            continue
        n = min(config.examples_per_tier, len(tier_harv_only))
        sample = tier_harv_only.sample(n=n, random_state=int(rng.randint(0, 2**31)))
        lines.append(f"### {tier}")
        lines.append("")
        for _, row in sample.iterrows():
            lines.append(f"- {row['concept_name']}")
        lines.append("")

    # 6. Top-K hardest reference gaps
    lines += [
        f"## 6. Top {config.top_k_unmatched} hardest reference gaps",
        "",
        "Reference concepts whose nearest harvested neighbor has the LOWEST similarity.",
        "These are the highest-priority gaps for future automated curation work.",
        "",
    ]
    unmatched_nearest: dict[str, Any] = s.get("unmatched_nearest", {})
    if unmatched_nearest:
        ref_gap_rows: list[dict[str, Any]] = []
        for _, row in ref_only_df.iterrows():
            cid = str(row["concept_id"])
            if cid in unmatched_nearest:
                entry = unmatched_nearest[cid]
                nearest_name = _lookup_concept_name(result.merged_concepts, entry["nearest_id"])
                ref_gap_rows.append(
                    {
                        "concept": row["concept_name"],
                        "tier": row["tier"],
                        "nearest_auto": nearest_name,
                        "sim": float(entry["sim"]),
                    }
                )
        ref_gap_rows.sort(key=lambda r: r["sim"])
        top_k = ref_gap_rows[: config.top_k_unmatched]
        if top_k:
            lines += [
                "| Reference concept | Tier | Nearest harvested | Sim |",
                "|---|---|---|---:|",
            ]
            for r in top_k:
                lines.append(
                    f"| {r['concept']} | {r['tier']} | {r['nearest_auto']} | {r['sim']:.3f} |"
                )
    else:
        # Fallback: show random reference-only concepts sorted by tier
        ref_gap_fallback = ref_only_df.sample(
            n=min(config.top_k_unmatched, len(ref_only_df)),
            random_state=int(rng.randint(0, 2**31)),
        )
        lines += [
            "| Reference concept | Tier |",
            "|---|---|",
        ]
        for _, row in ref_gap_fallback.iterrows():
            lines.append(f"| {row['concept_name']} | {row['tier']} |")
    lines.append("")

    # 7. Top-K most-novel harvested additions
    lines += [
        f"## 7. Top {config.top_k_unmatched} most-novel harvested additions",
        "",
        "Harvested concepts whose nearest reference neighbor has the LOWEST similarity.",
        "Sanity check: are these legitimately novel, or noise that survived curation?",
        "",
    ]
    if unmatched_nearest:
        harv_novel_rows: list[dict[str, Any]] = []
        for _, row in harv_only_df.iterrows():
            cid = str(row["concept_id"])
            if cid in unmatched_nearest:
                entry = unmatched_nearest[cid]
                nearest_name = _lookup_concept_name(result.merged_concepts, entry["nearest_id"])
                harv_novel_rows.append(
                    {
                        "concept": row["concept_name"],
                        "tier": row["tier"],
                        "nearest_ref": nearest_name,
                        "sim": float(entry["sim"]),
                    }
                )
        harv_novel_rows.sort(key=lambda r: r["sim"])
        top_k = harv_novel_rows[: config.top_k_unmatched]
        if top_k:
            lines += [
                "| Harvested concept | Tier | Nearest reference | Sim |",
                "|---|---|---|---:|",
            ]
            for r in top_k:
                lines.append(
                    f"| {r['concept']} | {r['tier']} | {r['nearest_ref']} | {r['sim']:.3f} |"
                )
    else:
        harv_novel_fallback = harv_only_df.sample(
            n=min(config.top_k_unmatched, len(harv_only_df)),
            random_state=int(rng.randint(0, 2**31)),
        )
        lines += [
            "| Harvested concept | Tier |",
            "|---|---|",
        ]
        for _, row in harv_novel_fallback.iterrows():
            lines.append(f"| {row['concept_name']} | {row['tier']} |")
    lines.append("")

    # 8. Cross-tier matches
    lines += [
        "## 8. Cross-tier matches",
        "",
    ]
    if not matches_df.empty:
        ct_df = matches_df[matches_df["is_cross_tier"] == True]  # noqa: E712
    else:
        ct_df = pd.DataFrame()

    if ct_df.empty:
        lines.append(
            f"No cross-tier matches found above {config.cross_tier_match_threshold} similarity."
        )
    else:
        lines.append(
            f"{len(ct_df)} concepts matched across different tiers (above "
            f"{config.cross_tier_match_threshold} similarity). These indicate either (a) ambiguous "
            f"concepts spanning tiers, or (b) tier-assignment disagreements between banks."
        )
        lines += [
            "",
            "| Reference label (tier) | Harvested label (tier) | Sim |",
            "|---|---|---:|",
        ]
        for _, row in ct_df.head(config.top_k_unmatched).iterrows():
            su_str = f"{row['reference_label']} ({row['reference_tier']})"
            auto_str = f"{row['harvested_label']} ({row['harvested_tier']})"
            lines.append(f"| {su_str} | {auto_str} | {float(row['similarity']):.3f} |")
    lines.append("")

    # 9. Discussion
    lines += [
        "## 9. Discussion",
        "",
        "- Tier-by-tier coverage and what gaps tell us about source diversity.",
        "- Why H2_niche has 0% reference recall (documented harvest-side limitation).",
        "- Why H2_gene_program has lower reference recall than H1_morphology.",
        "- Cross-tier match patterns — are they consistent enough to suggest one bank's",
        "  tier assignment is wrong?",
        "- Harvested-only concepts — net positive or noise?",
        "",
        "---",
        "",
        f"*Report generated by molf-interp v{version} | "
        f"Reference: {config.reference_bank_dir} | Harvested: {config.harvested_bank_dir}*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _lookup_concept_name(merged_concepts: pd.DataFrame, concept_id: str) -> str:
    """Look up concept_name by concept_id in merged_concepts DataFrame.

    Args:
        merged_concepts: The merged concepts DataFrame.
        concept_id: Concept ID to look up.

    Returns:
        The concept_name string, or the concept_id if not found.
    """
    row = merged_concepts[merged_concepts["concept_id"] == concept_id]
    if row.empty:
        return concept_id
    return str(row.iloc[0]["concept_name"])
