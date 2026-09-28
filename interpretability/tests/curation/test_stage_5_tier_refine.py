"""Tests for Stage 5: tier refine."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from molf_interp.curation.config import CurationConfig
from molf_interp.curation.organ_normalizer import OrganNormalizer
from molf_interp.curation.stage_5_tier_refine import run_stage_5
from tests.curation.conftest import make_mock_sapbert


def test_organ_vocab_dropped(tmp_config: object) -> None:
    cfg = CurationConfig()
    sapbert = make_mock_sapbert()

    df = pd.DataFrame(
        {
            "source_name": ["uberon"],
            "source_id": ["uberon:1"],
            "preferred_label": ["lung"],
            "synonyms": [[]],
            "definition": [None],
            "candidate_tier": ["organ_vocab"],
            "extra": [[]],
            "embedding": [np.ones(768, dtype=np.float32) / np.sqrt(768)],
        }
    )

    from molf_interp.concepts.schemas import ConceptBank

    mock_bank = MagicMock(spec=ConceptBank)
    mock_bank.concepts = ()

    mock_organ = MagicMock(spec=OrganNormalizer)
    mock_organ.infer_from_concept.return_value = "universal"

    result = run_stage_5(df, cfg, sapbert, reference_bank=mock_bank, organ_normalizer=mock_organ)
    assert len(result) == 0
