"""Stage 4: batched LLM adjudication with verification probing."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
from loguru import logger

from molf_interp.curation.config import CurationConfig
from molf_interp.curation.llm_adjudicator import CandidateForLLM, LLMAdjudicator


def _syns(row: pd.Series) -> tuple[str, ...]:  # type: ignore[type-arg]
    v = row.get("synonyms")
    return tuple(str(s) for s in v) if isinstance(v, list | np.ndarray) else ()


def _nan_to_none(val: object) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _build_candidate(row: pd.Series, row_idx: int) -> CandidateForLLM:  # type: ignore[type-arg]
    return CandidateForLLM(
        candidate_id=str(row_idx),
        label=str(row["preferred_label"]),
        synonyms=_syns(row),
        definition=_nan_to_none(row.get("definition")),
        candidate_tier=str(row["candidate_tier"]),
        reference_tier_similarity=float(
            row.get("reference_tier_similarity", row.get("su_similarity", 0.0))
        ),
    )


async def _run_adjudication_v2(
    df: pd.DataFrame,
    config: CurationConfig,
) -> pd.DataFrame:
    adjudicator = LLMAdjudicator(config.llm)

    async with adjudicator:
        if not await adjudicator.health_check():
            if config.llm.fail_on_unreachable:
                raise RuntimeError(
                    "Ollama health check failed; set fail_on_unreachable=false to skip."
                )
            logger.warning(
                "Stage 4: Ollama unreachable; dropping ambiguous + probe rows and returning."
            )
            df = df.copy()
            df["llm_decision"] = "skipped"
            df["llm_confidence"] = "skipped"
            df["llm_justification"] = ""
            df["llm_was_verification_probe"] = False
            return df[df["visual_filter_decision"] != "ambiguous"].copy()

        ambiguous_mask = df["visual_filter_decision"] == "ambiguous"
        keep_mask = df["visual_filter_decision"] == "keep"

        if config.llm_adjudicate_keeps_too:
            # Full-keep mode: adjudicate every keep row, no sampling
            probe_mask = keep_mask.copy()
            logger.info(
                "Stage 4: adjudicate_keeps_too=True — sending all {} keep rows to LLM",
                int(keep_mask.sum()),
            )
        else:
            # Verification probe: random sample of keep rows
            probe_mask = pd.Series(False, index=df.index)
            rate = config.llm_verification_sample_rate
            if rate > 0 and keep_mask.sum() > 0:
                rng = np.random.default_rng(config.llm_verification_seed)
                keep_indices = df.index[keep_mask].tolist()
                n_probe = max(1, int(len(keep_indices) * rate))
                probe_chosen = rng.choice(
                    keep_indices, size=min(n_probe, len(keep_indices)), replace=False
                )
                probe_mask[probe_chosen] = True

        adjudicate_mask = ambiguous_mask | probe_mask
        adjudicate_df = df[adjudicate_mask]

        candidates = [_build_candidate(row, idx) for idx, row in adjudicate_df.iterrows()]

        logger.info(
            "Stage 4: adjudicating {} candidates ({} ambiguous + {} probes)…",
            len(candidates),
            int(ambiguous_mask.sum()),
            int(probe_mask.sum()),
        )

        decisions = await adjudicator.adjudicate_all(
            candidates,
            batch_size=config.llm.batch_size,
            max_concurrent=config.llm.max_concurrent,
        )

    df = df.copy()
    df["llm_decision"] = "skipped"
    df["llm_confidence"] = "skipped"
    df["llm_justification"] = ""
    df["llm_was_verification_probe"] = False

    for idx, _row in adjudicate_df.iterrows():
        cid = str(idx)
        if cid in decisions:
            d = decisions[cid]
            df.at[idx, "llm_decision"] = d.decision
            df.at[idx, "llm_confidence"] = d.confidence
            df.at[idx, "llm_justification"] = d.justification
        df.at[idx, "llm_was_verification_probe"] = bool(probe_mask[idx])

    return df


def run_stage_4(df: pd.DataFrame, config: CurationConfig) -> pd.DataFrame:
    """LLM adjudication: batched Ollama with verification-probe sampling.

    Pipeline:
      1. If LLM disabled: drop all ambiguous rows, add skipped columns, return.
      2. Run health check; if unreachable and fail_on_unreachable=False: same as disabled.
      3. Adjudicate ambiguous rows + 5% random sample of keep rows (probe).
      4. Drop: ambiguous→drop AND keep_probe→drop (false-positives caught).

    Args:
        df: DataFrame from Stage 3 with 'visual_filter_decision' column.
        config: CurationConfig instance.

    Returns:
        DataFrame with llm_decision, llm_confidence, llm_justification,
        llm_was_verification_probe columns added.
    """
    t0 = time.monotonic()
    ambiguous_count = int((df["visual_filter_decision"] == "ambiguous").sum())

    df = df.copy()
    df["llm_decision"] = "skipped"
    df["llm_confidence"] = "skipped"
    df["llm_justification"] = ""
    df["llm_was_verification_probe"] = False

    if not config.llm.enabled:
        logger.info(
            "Stage 4: LLM disabled; dropping {} ambiguous candidates",
            ambiguous_count,
        )
        result = df[df["visual_filter_decision"] != "ambiguous"].copy()
        elapsed = time.monotonic() - t0
        logger.info("Stage 4 done: {} rows, {:.1f}s", len(result), elapsed)
        return result

    keep_count = int((df["visual_filter_decision"] == "keep").sum())
    nothing_to_do = (
        ambiguous_count == 0
        and config.llm_verification_sample_rate == 0
        and not (config.llm_adjudicate_keeps_too and keep_count > 0)
    )
    if nothing_to_do:
        logger.info("Stage 4: nothing to adjudicate")
        return df

    try:
        df = asyncio.run(_run_adjudication_v2(df, config))
    except RuntimeError:
        # Already inside an event loop (e.g. Jupyter notebook) — use the existing loop
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(_run_adjudication_v2(df, config))

    # Drop ambiguous candidates where LLM said drop
    drop_ambiguous = (df["visual_filter_decision"] == "ambiguous") & (df["llm_decision"] == "drop")
    # Drop keep-probes where LLM flagged as drop (false-positives)
    drop_probes = df["llm_was_verification_probe"] & (df["llm_decision"] == "drop")
    drop_total = drop_ambiguous | drop_probes

    kept_from_ambiguous = int(
        ((df["visual_filter_decision"] == "ambiguous") & (df["llm_decision"] == "keep")).sum()
    )
    probes_dropped = int(drop_probes.sum())

    result = df[~drop_total].copy()
    elapsed = time.monotonic() - t0
    logger.info(
        "Stage 4 done: {} ambiguous → {} kept ({}%), {} probe false-positives dropped, {:.1f}s",
        ambiguous_count,
        kept_from_ambiguous,
        round(100 * kept_from_ambiguous / max(1, ambiguous_count)),
        probes_dropped,
        elapsed,
    )
    return result
