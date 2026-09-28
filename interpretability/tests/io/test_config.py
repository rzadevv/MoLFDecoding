import pytest
from pydantic import ValidationError

from molf_interp.io.config import BaseConfig, HelloConfig, load_config


def test_load_valid_yaml(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("name: alice\ngreeting: hi\n")
    result = load_config(cfg_file, HelloConfig)
    assert result.name == "alice"
    assert result.greeting == "hi"


def test_load_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_config(missing, HelloConfig)


def test_load_extra_fields_raises(tmp_path):
    cfg_file = tmp_path / "extra.yaml"
    cfg_file.write_text("name: bob\ngreeting: hey\nunknown_field: oops\n")
    with pytest.raises(ValidationError):
        load_config(cfg_file, HelloConfig)


def test_base_config_is_frozen():
    class SampleConfig(BaseConfig):
        value: int

    cfg = SampleConfig(value=1)
    with pytest.raises(ValidationError):
        cfg.value = 2  # type: ignore[misc]
