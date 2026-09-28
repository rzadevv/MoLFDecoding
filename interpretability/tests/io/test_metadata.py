import json
from datetime import UTC, datetime

from molf_interp.io.metadata import RunMetadata, collect_git_info, write_metadata


def _sample_meta() -> RunMetadata:
    return RunMetadata(
        run_id="abc123",
        git_sha="deadbeef",
        git_dirty=False,
        config_hash="cafebabe",
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        command="hello",
        python_version="3.11.0",
    )


def test_run_metadata_roundtrip():
    meta = _sample_meta()
    raw = meta.model_dump(mode="json")
    restored = RunMetadata.model_validate(raw)
    assert restored.run_id == meta.run_id
    assert restored.command == meta.command


def test_write_metadata_creates_valid_json(tmp_path):
    meta = _sample_meta()
    out = tmp_path / "meta.json"
    write_metadata(meta, out)
    data = json.loads(out.read_text())
    assert "run_id" in data
    assert data["command"] == "hello"
    assert data["git_sha"] == "deadbeef"


def test_collect_git_info_non_git_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha, dirty = collect_git_info()
    assert sha is None
    assert dirty is False
