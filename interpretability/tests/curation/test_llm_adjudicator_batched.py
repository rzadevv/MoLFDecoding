"""Tests for the batched LLM adjudicator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from molf_interp.curation.config import LLMAdjudicationConfig
from molf_interp.curation.llm_adjudicator import (
    CandidateForLLM,
    LLMAdjudicator,
    LLMBatchRequest,
    LLMDecision,
)


def _cfg(**kwargs) -> LLMAdjudicationConfig:  # type: ignore[type-arg]
    defaults = dict(
        enabled=True,
        base_url="http://localhost:11434",
        model="qwen2.5:3b-instruct",
        timeout_seconds=5.0,
        max_concurrent=1,
        batch_size=8,
        prompt_template_path="configs/curation/llm_prompt_template.txt",
    )
    defaults.update(kwargs)
    return LLMAdjudicationConfig(**defaults)


def _candidate(cid: str, label: str = "necrosis") -> CandidateForLLM:
    return CandidateForLLM(
        candidate_id=cid,
        label=label,
        synonyms=(),
        definition=None,
        candidate_tier="H1_morphology",
        reference_tier_similarity=0.75,
    )


def _batch(*cids: str) -> LLMBatchRequest:
    return LLMBatchRequest(
        batch_id="test_batch",
        candidates=tuple(_candidate(cid) for cid in cids),
    )


def _make_response(decisions: list[dict]) -> str:  # type: ignore[type-arg]
    return json.dumps({"decisions": decisions})


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_returns_false_on_404() -> None:
    adj = LLMAdjudicator(_cfg())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client
        result = await adj.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_health_check_returns_true_when_model_present() -> None:
    adj = LLMAdjudicator(_cfg())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(
            return_value={"models": [{"name": "qwen2.5:3b-instruct:latest"}]}
        )
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client
        result = await adj.health_check()
    assert result is True


# ---------------------------------------------------------------------------
# adjudicate_batch — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjudicate_batch_parses_response() -> None:
    adj = LLMAdjudicator(_cfg())
    batch = _batch("1", "2")

    response_json = _make_response(
        [
            {"candidate_id": "1", "decision": "keep", "confidence": "high", "justification": "ok"},
            {"candidate_id": "2", "decision": "drop", "confidence": "low", "justification": "bad"},
        ]
    )

    adj._call_ollama = AsyncMock(return_value=response_json)  # type: ignore[method-assign]
    decisions = await adj.adjudicate_batch(batch)

    assert len(decisions) == 2
    assert decisions[0].candidate_id == "1"
    assert decisions[0].decision == "keep"
    assert decisions[1].decision == "drop"


# ---------------------------------------------------------------------------
# adjudicate_batch — retry on malformed JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjudicate_batch_retries_on_malformed_json() -> None:
    adj = LLMAdjudicator(_cfg())
    batch = _batch("1")

    good_response = _make_response(
        [
            {"candidate_id": "1", "decision": "keep", "confidence": "high", "justification": "ok"},
        ]
    )

    # First call returns garbage, second returns valid JSON
    adj._call_ollama = AsyncMock(side_effect=["not json {{{", good_response])  # type: ignore[method-assign]
    decisions = await adj.adjudicate_batch(batch)

    assert decisions[0].decision == "keep"
    assert adj._call_ollama.call_count == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# adjudicate_batch — default-drop on double failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjudicate_batch_defaults_to_drop_on_double_failure() -> None:
    adj = LLMAdjudicator(_cfg())
    batch = _batch("1", "2")

    adj._call_ollama = AsyncMock(return_value=None)  # type: ignore[method-assign]
    decisions = await adj.adjudicate_batch(batch)

    assert all(d.decision == "drop" for d in decisions)
    assert all(d.confidence == "low" for d in decisions)


# ---------------------------------------------------------------------------
# adjudicate_all — batch count and output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjudicate_all_correct_batch_count() -> None:
    adj = LLMAdjudicator(_cfg())

    candidates = [_candidate(str(i)) for i in range(20)]
    call_count = 0

    async def _mock_batch(batch: LLMBatchRequest) -> list[LLMDecision]:
        nonlocal call_count
        call_count += 1
        return [
            LLMDecision(
                candidate_id=c.candidate_id,
                decision="keep",
                confidence="high",
                justification="ok",
            )
            for c in batch.candidates
        ]

    adj.adjudicate_batch = _mock_batch  # type: ignore[method-assign]
    result = await adj.adjudicate_all(candidates, batch_size=8, show_progress=False)

    # 20 candidates / 8 per batch = 3 batches (8+8+4)
    assert call_count == 3
    assert len(result) == 20


@pytest.mark.asyncio
async def test_adjudicate_all_returns_all_ids() -> None:
    adj = LLMAdjudicator(_cfg())
    candidates = [_candidate(str(i)) for i in range(5)]

    async def _mock_batch(batch: LLMBatchRequest) -> list[LLMDecision]:
        return [
            LLMDecision(
                candidate_id=c.candidate_id,
                decision="keep",
                confidence="high",
                justification="ok",
            )
            for c in batch.candidates
        ]

    adj.adjudicate_batch = _mock_batch  # type: ignore[method-assign]
    result = await adj.adjudicate_all(candidates, batch_size=8, show_progress=False)
    assert set(result) == {"0", "1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# justification truncation
# ---------------------------------------------------------------------------


def test_llm_decision_justification_truncated() -> None:
    d = LLMDecision(
        candidate_id="1",
        decision="keep",
        confidence="high",
        justification="x" * 300,
    )
    assert len(d.justification) == 200
