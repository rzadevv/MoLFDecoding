"""Tests for BioPortalConfig."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from molf_interp.io.config import load_config
from molf_interp.sources.bioportal.config import BioPortalConfig


def test_defaults() -> None:
    cfg = BioPortalConfig()
    assert cfg.base_url == "https://data.bioontology.org"
    assert cfg.api_key_env == "BIOPORTAL_API_KEY"
    assert cfg.rate_limit_per_second == 10
    assert cfg.rate_limit_burst == 5
    assert cfg.timeout_seconds == 30.0
    assert cfg.max_retries == 5
    assert cfg.page_size == 200
    assert cfg.cache_dir == Path("data/cache/bioportal")


def test_yaml_loading(tmp_path: Path) -> None:
    yaml_text = """\
base_url: https://data.bioontology.org
api_key_env: BIOPORTAL_API_KEY
rate_limit_per_second: 8
rate_limit_burst: 3
timeout_seconds: 15.0
max_retries: 3
page_size: 100
cache_dir: /tmp/cache
user_agent: "test-agent"
"""
    config_file = tmp_path / "bioportal.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(config_file, BioPortalConfig)
    assert cfg.rate_limit_per_second == 8
    assert cfg.page_size == 100
    assert cfg.cache_dir == Path("/tmp/cache")


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        BioPortalConfig(unknown_field="oops")  # type: ignore[call-arg]
