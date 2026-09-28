"""Tests for NCBIConfig."""

from pathlib import Path

import pytest

from molf_interp.sources.ncbi.config import NCBIConfig


def test_defaults() -> None:
    cfg = NCBIConfig()
    assert cfg.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    assert cfg.api_key_env == "NCBI_API_KEY"
    assert cfg.email_env == "NCBI_EMAIL"
    assert cfg.rate_limit_per_second == 8
    assert cfg.efetch_batch_size == 200
    assert cfg.max_retries == 5


def test_yaml_loading(tmp_path: Path) -> None:
    yaml_text = """\
cache_dir: /tmp/pubmed
rate_limit_per_second: 5
"""
    f = tmp_path / "ncbi.yaml"
    f.write_text(yaml_text)
    from molf_interp.io.config import load_config

    cfg = load_config(f, NCBIConfig)
    assert cfg.cache_dir == Path("/tmp/pubmed")
    assert cfg.rate_limit_per_second == 5


def test_extra_field_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NCBIConfig(unknown_field="x")  # type: ignore[call-arg]
