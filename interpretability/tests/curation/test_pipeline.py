"""Regression tests for pipeline._save_stage parquet schema preservation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from molf_interp.curation.config import CurationConfig
from molf_interp.curation.pipeline import _save_stage


def _make_stage3_df() -> pd.DataFrame:
    """Minimal stage-3 DataFrame with empty parent_ids and extra columns."""
    return pd.DataFrame(
        {
            "source_name": ["ncit"],
            "source_id": ["C001"],
            "preferred_label": ["necrosis"],
            "synonyms": [[]],
            "definition": [None],
            "parent_ids": [[]],
            "candidate_tier": ["H1_morphology"],
            "extra": [[]],
            "visual_filter_decision": ["keep"],
            "reference_tier_similarity": [0.6],
        }
    )


def test_save_stage_parent_ids_type(tmp_path: Path) -> None:
    """parent_ids must round-trip as list<string>, not list<null>."""
    cfg = CurationConfig(stage_cache_dir=tmp_path / "stages")
    df = _make_stage3_df()
    _save_stage(df, cfg, stage=3)

    path = tmp_path / "stages" / "stage_3_visual_filter.parquet"
    schema = pq.read_schema(path)
    field = schema.field("parent_ids")
    # pyarrow uses "element" as the list child name
    assert str(field.type) == "list<element: string>", (
        f"parent_ids type degraded: expected list<element: string>, got {field.type}"
    )


def test_save_stage_extra_type(tmp_path: Path) -> None:
    """extra must round-trip as map<string, string>, not list<list<...>>."""
    cfg = CurationConfig(stage_cache_dir=tmp_path / "stages")
    df = _make_stage3_df()
    _save_stage(df, cfg, stage=3)

    path = tmp_path / "stages" / "stage_3_visual_filter.parquet"
    schema = pq.read_schema(path)
    field = schema.field("extra")
    assert "map" in str(field.type).lower(), (
        f"extra type degraded: expected map<string, string>, got {field.type}"
    )


def test_save_stage_roundtrip_values(tmp_path: Path) -> None:
    """Data values survive the parquet roundtrip."""
    cfg = CurationConfig(stage_cache_dir=tmp_path / "stages")
    df = _make_stage3_df()
    _save_stage(df, cfg, stage=3)

    path = tmp_path / "stages" / "stage_3_visual_filter.parquet"
    loaded = pd.read_parquet(path)
    assert loaded["preferred_label"].iloc[0] == "necrosis"
    assert loaded["visual_filter_decision"].iloc[0] == "keep"
    assert list(loaded["parent_ids"].iloc[0]) == []
