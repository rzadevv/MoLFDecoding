"""Batched Ollama adjudicator for ambiguous and verification-probe candidates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import field_validator

from molf_interp.curation.config import LLMAdjudicationConfig
from molf_interp.io.config import BaseConfig


class CandidateForLLM(BaseConfig):
    """One candidate's inputs for the LLM adjudicator."""

    candidate_id: str
    label: str
    synonyms: tuple[str, ...]
    definition: str | None
    candidate_tier: str
    reference_tier_similarity: float


class LLMDecision(BaseConfig):
    """One LLM decision per candidate."""

    candidate_id: str
    decision: Literal["keep", "drop"]
    confidence: Literal["high", "medium", "low"]
    justification: str

    @field_validator("justification")
    @classmethod
    def _truncate(cls, v: str) -> str:
        """Truncate justification to 200 chars for safety."""
        return v.strip()[:200]


class LLMBatchRequest(BaseConfig):
    """One batched LLM request containing up to batch_size candidates."""

    batch_id: str
    candidates: tuple[CandidateForLLM, ...]


def _render_candidates_block(candidates: Sequence[CandidateForLLM]) -> str:
    lines: list[str] = []
    for c in candidates:
        syns = ", ".join(c.synonyms) if c.synonyms else "(none)"
        lines.append(
            f'[id={c.candidate_id}] term: "{c.label}"\n'
            f"  candidate_tier: {c.candidate_tier}\n"
            f"  synonyms: {syns}\n"
            f"  similarity_to_reference_tier: {c.reference_tier_similarity:.2f}"
        )
    return "\n\n".join(lines)


def _default_drop(candidate_id: str, reason: str) -> LLMDecision:
    return LLMDecision(
        candidate_id=candidate_id,
        decision="drop",
        confidence="low",
        justification=reason[:200],
    )


class LLMAdjudicator:
    """Batched Ollama adjudicator with health-check and retry on parse failure."""

    def __init__(self, config: LLMAdjudicationConfig) -> None:
        """Initialize with adjudication config.

        Args:
            config: LLMAdjudicationConfig instance.
        """
        self._config = config
        self._template: str | None = None

    async def __aenter__(self) -> LLMAdjudicator:
        """Enter async context and verify Ollama is reachable."""
        if not await self.health_check():
            if self._config.fail_on_unreachable:
                raise RuntimeError(
                    f"Ollama not reachable or model {self._config.model!r} not pulled. "
                    "Run `ollama serve` and `ollama pull <model>`."
                )
            logger.warning(
                "Ollama health check failed for model {}. Adjudication may fail.",
                self._config.model,
            )
        return self

    async def __aexit__(self, *args: object) -> None:
        """No cleanup needed."""

    async def health_check(self) -> bool:
        """Check whether Ollama is reachable and the configured model is available.

        Returns:
            True if the Ollama server responded and the model is in /api/tags.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._config.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self._config.model in m for m in models)
        except Exception as exc:
            logger.debug("Ollama health check error: {}", exc)
            return False

    def _load_template(self) -> str:
        if self._template is None:
            template_path = Path(self._config.prompt_template_path)
            if not template_path.exists():
                raise FileNotFoundError(
                    f"LLM prompt template not found: {template_path}. "
                    f"Check llm.prompt_template_path in your CurationConfig."
                )
            self._template = template_path.read_text()
        return self._template

    def _build_prompt(self, candidates: Sequence[CandidateForLLM]) -> str:
        template = self._load_template()
        block = _render_candidates_block(candidates)
        return template.format(candidates_block=block)

    async def _call_ollama(self, prompt: str) -> str | None:
        try:
            import ollama

            client = ollama.AsyncClient(
                host=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
            response = await client.chat(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return str(response.message.content)
        except Exception as exc:
            logger.warning("Ollama call failed: {}", exc)
            return None

    def _parse_decisions(
        self,
        raw: str | None,
        expected_ids: list[str],
    ) -> list[LLMDecision] | None:
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            raw_decisions = data.get("decisions", [])
            by_id: dict[str, LLMDecision] = {}
            for item in raw_decisions:
                try:
                    d = LLMDecision(**item)
                    by_id[d.candidate_id] = d
                except Exception:
                    pass
            # All expected IDs must be present
            if not all(cid in by_id for cid in expected_ids):
                return None
            return [by_id[cid] for cid in expected_ids]
        except Exception:
            return None

    async def adjudicate_batch(self, batch: LLMBatchRequest) -> list[LLMDecision]:
        """Send one batch to Ollama and parse decisions.

        Retries once on parse failure. Falls back to default-drop decisions on
        second failure.

        Args:
            batch: LLMBatchRequest with up to batch_size candidates.

        Returns:
            List of LLMDecision in the same order as batch.candidates.
        """
        candidates = list(batch.candidates)
        expected_ids = [c.candidate_id for c in candidates]
        prompt = self._build_prompt(candidates)

        raw = await self._call_ollama(prompt)
        decisions = self._parse_decisions(raw, expected_ids)

        if decisions is None:
            retry_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."
            raw2 = await self._call_ollama(retry_prompt)
            decisions = self._parse_decisions(raw2, expected_ids)

        if decisions is None:
            logger.warning(
                "Batch {} failed to parse after retry; defaulting all {} to drop",
                batch.batch_id,
                len(candidates),
            )
            decisions = [
                _default_drop(cid, "LLM parse failure after retry") for cid in expected_ids
            ]

        return decisions

    async def adjudicate_all(
        self,
        candidates: Sequence[CandidateForLLM],
        *,
        batch_size: int = 8,
        max_concurrent: int = 1,
        show_progress: bool = True,
    ) -> dict[str, LLMDecision]:
        """Process all candidates in batches with concurrency control.

        Args:
            candidates: All candidates to adjudicate.
            batch_size: Candidates per Ollama call (default 8).
            max_concurrent: Maximum simultaneous Ollama calls.
            show_progress: Show tqdm progress bar if True.

        Returns:
            Dict mapping candidate_id → LLMDecision.
        """
        all_cands = list(candidates)
        batches: list[LLMBatchRequest] = []
        for i in range(0, len(all_cands), batch_size):
            chunk = all_cands[i : i + batch_size]
            batches.append(
                LLMBatchRequest(
                    batch_id=f"batch_{i // batch_size}",
                    candidates=tuple(chunk),
                )
            )

        sem = asyncio.Semaphore(max_concurrent)

        async def _guarded(b: LLMBatchRequest) -> list[LLMDecision]:
            async with sem:
                return await self.adjudicate_batch(b)

        tasks = [_guarded(b) for b in batches]

        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio  # type: ignore[import-untyped]

                results_nested: list[list[LLMDecision]] = await tqdm_asyncio.gather(
                    *tasks, desc="LLM adjudication"
                )
            except ImportError:
                results_nested = await asyncio.gather(*tasks)  # type: ignore[assignment]
        else:
            results_nested = await asyncio.gather(*tasks)  # type: ignore[assignment]

        out: dict[str, LLMDecision] = {}
        for batch_decisions in results_nested:
            for d in batch_decisions:
                out[d.candidate_id] = d
        return out
