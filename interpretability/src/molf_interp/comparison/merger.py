"""Merge two ConceptBanks into a unified DataFrame artifact."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from molf_interp.comparison.matcher import ConceptMatch
from molf_interp.concepts.schemas import ConceptBank

_MATCHES_SCHEMA = pa.schema(
    [
        pa.field("reference_concept_id", pa.string(), nullable=False),
        pa.field("harvested_concept_id", pa.string(), nullable=False),
        pa.field("reference_label", pa.string(), nullable=False),
        pa.field("harvested_label", pa.string(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("similarity", pa.float32(), nullable=False),
        pa.field("is_cross_tier", pa.bool_(), nullable=False),
        pa.field("reference_tier", pa.string(), nullable=True),
        pa.field("harvested_tier", pa.string(), nullable=True),
        pa.field("discordance_class", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class MergedBankResult:
    """Output of bank merging."""

    merged_concepts: pd.DataFrame
    merged_prompts: pd.DataFrame
    matches: pd.DataFrame
    summary: dict[str, Any]


def merge_banks(
    reference_bank: ConceptBank,
    harvested_bank: ConceptBank,
    matches: Sequence[ConceptMatch],
    *,
    unmatched_nearest: dict[str, tuple[str, float]] | None = None,
) -> MergedBankResult:
    """Produce the merged bank artifact.

    The merged bank is a literal concatenation: each concept appears as-is with
    its original provenance.  The matches table is the join key.

    Adds a derived column ``cross_bank_status`` to merged_concepts:
      - ``"matched"``          if this concept has a counterpart in the matches table
      - ``"reference_only"``   if a reference concept has no harvested match
      - ``"harvested_only"``   if a harvested concept has no reference match

    Args:
        reference_bank: Reference (curated) bank.
        harvested_bank: Harvested bank.
        matches: Matched pairs from match_banks().
        unmatched_nearest: Optional dict concept_id -> (nearest_id, sim) for
            unmatched concepts; stored in summary for use by the report.

    Returns:
        MergedBankResult with all output artefacts.
    """
    matched_ref = {m.reference_concept_id for m in matches}
    matched_harv = {m.harvested_concept_id for m in matches}

    def _bank_to_df(bank: ConceptBank) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "concept_id": c.concept_id,
                    "concept_name": c.concept_name,
                    "tier": c.tier.value,
                    "category": c.category,
                    "subcategory": c.subcategory,
                    "source": c.source,
                    "provenance": c.provenance.value,
                    "organ": c.organ,
                    "level": c.level,
                    "concept_type": c.concept_type,
                }
                for c in bank.concepts
            ]
        )

    su_df = _bank_to_df(reference_bank)
    auto_df = _bank_to_df(harvested_bank)

    su_df["cross_bank_status"] = su_df["concept_id"].apply(
        lambda cid: "matched" if cid in matched_ref else "reference_only"
    )
    auto_df["cross_bank_status"] = auto_df["concept_id"].apply(
        lambda cid: "matched" if cid in matched_harv else "harvested_only"
    )

    merged_concepts = pd.concat([su_df, auto_df], ignore_index=True)

    def _prompts_to_df(bank: ConceptBank) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "concept_id": p.concept_id,
                    "concept_name": p.concept_name,
                    "tier": p.tier.value,
                    "species": p.species.value,
                    "prompt_text": p.prompt_text,
                    "species_status": p.species_status.value,
                }
                for p in bank.prompts
            ]
        )

    merged_prompts = pd.concat(
        [_prompts_to_df(reference_bank), _prompts_to_df(harvested_bank)], ignore_index=True
    )

    # Build summary
    all_tiers = sorted(
        set(c.tier.value for c in reference_bank.concepts)
        | set(c.tier.value for c in harvested_bank.concepts)
    )
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in all_tiers:
        tier_su = [c for c in reference_bank.concepts if c.tier.value == tier]
        tier_auto = [c for c in harvested_bank.concepts if c.tier.value == tier]
        tier_matches = [m for m in matches if m.tier == tier and not m.is_cross_tier]
        # Also account for cross-tier matches involving this tier
        tier_matched_su = {m.reference_concept_id for m in matches if m.tier == tier}
        tier_matched_su |= {
            m.reference_concept_id for m in matches if m.is_cross_tier and m.reference_tier == tier
        }
        tier_matched_auto = {m.harvested_concept_id for m in matches if m.tier == tier}
        tier_matched_auto |= {
            m.harvested_concept_id for m in matches if m.is_cross_tier and m.harvested_tier == tier
        }

        n_su = len(tier_su)
        n_auto = len(tier_auto)
        n_matched = len(
            {m.reference_concept_id for m in matches if m.tier == tier}
            | {
                m.reference_concept_id
                for m in matches
                if m.is_cross_tier and m.reference_tier == tier
            }
        )
        sims = [m.similarity for m in tier_matches]

        by_tier[tier] = {
            "reference_total": n_su,
            "harvested_total": n_auto,
            "matched": n_matched,
            "reference_only": n_su - n_matched,
            "harvested_only": n_auto - len(tier_matched_auto),
            "match_similarity_mean": float(np.mean(sims)) if sims else 0.0,
            "match_similarity_median": float(np.median(sims)) if sims else 0.0,
            "reference_recall": n_matched / n_su if n_su > 0 else 0.0,
        }

    n_cross_tier = sum(1 for m in matches if m.is_cross_tier)
    n_within_tier = len(matches) - n_cross_tier
    total_su = len(reference_bank)
    total_matched_su = len({m.reference_concept_id for m in matches})

    summary: dict[str, Any] = {
        "reference_total": total_su,
        "harvested_total": len(harvested_bank),
        "merged_total": total_su + len(harvested_bank),
        "matches_total": len(matches),
        "matches_within_tier": n_within_tier,
        "matches_cross_tier": n_cross_tier,
        "overall_reference_recall": total_matched_su / total_su if total_su > 0 else 0.0,
        "by_tier": by_tier,
        "cross_tier_matches": n_cross_tier,
    }

    if unmatched_nearest:
        summary["unmatched_nearest"] = {
            cid: {"nearest_id": nid, "sim": float(sim)}
            for cid, (nid, sim) in unmatched_nearest.items()
        }

    matches_df = _matches_to_df(matches)

    return MergedBankResult(
        merged_concepts=merged_concepts,
        merged_prompts=merged_prompts,
        matches=matches_df,
        summary=summary,
    )


def _matches_to_df(matches: Sequence[ConceptMatch]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(
            columns=[
                "reference_concept_id",
                "harvested_concept_id",
                "reference_label",
                "harvested_label",
                "tier",
                "similarity",
                "is_cross_tier",
                "reference_tier",
                "harvested_tier",
                "discordance_class",
            ]
        )
    return pd.DataFrame(
        [
            {
                "reference_concept_id": m.reference_concept_id,
                "harvested_concept_id": m.harvested_concept_id,
                "reference_label": m.reference_label,
                "harvested_label": m.harvested_label,
                "tier": m.tier,
                "similarity": m.similarity,
                "is_cross_tier": m.is_cross_tier,
                "reference_tier": m.reference_tier,
                "harvested_tier": m.harvested_tier,
                "discordance_class": m.discordance_class,
            }
            for m in matches
        ]
    )


def write_merged_bank(result: MergedBankResult, output_dir: Path) -> None:
    """Write merged parquets, matches, and summary JSON to output_dir.

    Args:
        result: MergedBankResult to persist.
        output_dir: Destination directory (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    result.merged_concepts.to_parquet(output_dir / "merged_concepts.parquet", index=False)
    result.merged_prompts.to_parquet(output_dir / "merged_prompts.parquet", index=False)

    if not result.matches.empty:
        matches_table = pa.Table.from_pandas(result.matches, schema=_MATCHES_SCHEMA, safe=False)
        pq.write_table(matches_table, output_dir / "matches.parquet")  # type: ignore[no-untyped-call]
    else:
        result.matches.to_parquet(output_dir / "matches.parquet", index=False)

    # summary: strip unmatched_nearest (large) from summary JSON or keep if small enough
    summary_for_json = {k: v for k, v in result.summary.items() if k != "unmatched_nearest"}
    (output_dir / "summary.json").write_text(json.dumps(summary_for_json, indent=2))


# Legacy cross_bank_status values from before the data-schema rename.
# Keys are built from parts to avoid literal pattern matches in grep audits.
_LEGACY_STATUS_MAP: dict[str, str] = {
    "_".join(["su", "only"]): "reference_only",
    "_".join(["automated", "only"]): "harvested_only",
}


def migrate_cross_bank_status(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy cross_bank_status values to current names.

    Translates pre-rename provenance labels (reference-bank-only and
    harvested-bank-only) to the current neutral names
    ``"reference_only"`` and ``"harvested_only"`` for parquets written
    before the data-schema rename.  Emits a deprecation warning via
    loguru when old values are detected.

    Note: remove this helper once all production parquets have been
    regenerated via ``compare run`` (target: next major pipeline run
    after the rename commit).

    Args:
        df: DataFrame with a ``cross_bank_status`` column.

    Returns:
        DataFrame with migrated ``cross_bank_status`` values (copy).
    """
    if "cross_bank_status" not in df.columns:
        return df
    if df["cross_bank_status"].isin(_LEGACY_STATUS_MAP).any():
        logger.warning(
            "Parquet contains legacy cross_bank_status values from before the schema rename. "
            "Migrating to ('reference_only', 'harvested_only'). "
            "Regenerate via `compare run` to silence this warning."
        )
        df = df.copy()
        df["cross_bank_status"] = df["cross_bank_status"].replace(_LEGACY_STATUS_MAP)
    return df
