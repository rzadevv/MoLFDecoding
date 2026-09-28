"""Tests for Stage 4: LLM adjudication."""

from __future__ import annotations

import pandas as pd

from molf_interp.curation.config import CurationConfig, LLMAdjudicationConfig
from molf_interp.curation.stage_4_llm_adjudicate import run_stage_4


def _make_df_with_ambiguous(n_keep: int = 5, n_ambiguous: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_keep):
        rows.append(
            {
                "preferred_label": f"necrosis {i}",
                "synonyms": [],
                "definition": None,
                "candidate_tier": "H1_morphology",
                "visual_filter_decision": "keep",
            }
        )
    for i in range(n_ambiguous):
        rows.append(
            {
                "preferred_label": f"ambiguous {i}",
                "synonyms": [],
                "definition": None,
                "candidate_tier": "H1_morphology",
                "visual_filter_decision": "ambiguous",
            }
        )
    return pd.DataFrame(rows)


def test_llm_disabled_drops_ambiguous() -> None:
    cfg = CurationConfig(llm=LLMAdjudicationConfig(enabled=False))
    df = _make_df_with_ambiguous(n_keep=5, n_ambiguous=3)
    result = run_stage_4(df, cfg)
    assert len(result) == 5
    assert (result["visual_filter_decision"] == "keep").all()


def test_llm_disabled_no_ambiguous_is_noop() -> None:
    cfg = CurationConfig(llm=LLMAdjudicationConfig(enabled=False))
    df = _make_df_with_ambiguous(n_keep=5, n_ambiguous=0)
    result = run_stage_4(df, cfg)
    assert len(result) == 5
