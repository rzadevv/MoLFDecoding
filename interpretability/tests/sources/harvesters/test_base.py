"""Tests for harvester base utilities."""

from __future__ import annotations

from molf_interp.sources.harvesters.base import (
    is_obviously_non_morphological,
    normalize_synonyms,
    truncate_definition,
)


class TestNormalizeSynonyms:
    def test_strips_whitespace(self) -> None:
        result = normalize_synonyms(["  foo  ", "bar "])
        assert "foo" in result
        assert "bar" in result

    def test_drops_empties(self) -> None:
        result = normalize_synonyms(["", "  ", "valid"])
        assert result == ("valid",)

    def test_deduplicates_case_insensitively(self) -> None:
        result = normalize_synonyms(["Necrosis", "NECROSIS", "necrosis"])
        assert len(result) == 1
        assert result[0] == "Necrosis"

    def test_preserves_first_seen_casing(self) -> None:
        result = normalize_synonyms(["Foo Bar", "foo bar"])
        assert len(result) == 1
        assert result[0] == "Foo Bar"

    def test_sorts_case_insensitively(self) -> None:
        result = normalize_synonyms(["zebra", "Apple", "mango"])
        assert result == ("Apple", "mango", "zebra")

    def test_drops_label_exact_match(self) -> None:
        result = normalize_synonyms(["Necrosis", "Tumor"], drop_label="Necrosis")
        assert "Necrosis" not in result
        assert "Tumor" in result

    def test_drops_label_case_insensitive(self) -> None:
        result = normalize_synonyms(["NECROSIS", "Valid"], drop_label="necrosis")
        assert len([s for s in result if s.lower() == "necrosis"]) == 0

    def test_empty_input(self) -> None:
        assert normalize_synonyms([]) == ()

    def test_no_drop_label(self) -> None:
        result = normalize_synonyms(["A", "B"])
        assert len(result) == 2


class TestTruncateDefinition:
    def test_none_returns_none(self) -> None:
        assert truncate_definition(None) is None

    def test_short_text_unchanged(self) -> None:
        text = "Short definition."
        assert truncate_definition(text) == text

    def test_truncates_at_sentence_boundary(self) -> None:
        sentence = "First sentence. "
        long_text = sentence + "A" * 490
        result = truncate_definition(long_text, max_chars=30)
        assert result == "First sentence."

    def test_hard_truncate_when_no_sentence(self) -> None:
        text = "A" * 600
        result = truncate_definition(text, max_chars=500)
        assert len(result) == 500  # type: ignore[arg-type]

    def test_exact_limit_unchanged(self) -> None:
        text = "A" * 500
        assert truncate_definition(text, max_chars=500) == text


class TestIsObviouslyNonMorphological:
    def test_morphological_term_returns_false(self) -> None:
        assert is_obviously_non_morphological("coagulative necrosis") is False

    def test_lesion_ambiguous_returns_false(self) -> None:
        assert is_obviously_non_morphological("lesion") is False

    def test_patient_returns_true(self) -> None:
        assert is_obviously_non_morphological("patient age >70") is True

    def test_survival_returns_true(self) -> None:
        assert is_obviously_non_morphological("5-year survival") is True

    def test_stage_returns_true(self) -> None:
        assert is_obviously_non_morphological("tumor stage") is True

    def test_genetic_mutation_returns_true(self) -> None:
        assert is_obviously_non_morphological("genetic mutation analysis") is True

    def test_case_insensitive(self) -> None:
        assert is_obviously_non_morphological("PATIENT HISTORY") is True
