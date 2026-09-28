"""Run metadata collection and persistence."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from molf_interp.io.config import BaseConfig


class RunMetadata(BaseConfig):
    """Captures provenance for a single pipeline run.

    Attributes:
        run_id: UUID4 hex string uniquely identifying the run.
        git_sha: Current HEAD commit SHA, or None if unavailable.
        git_dirty: True if the working tree has uncommitted changes.
        config_hash: SHA-256 hash of the config used.
        started_at: UTC timestamp when the run began.
        finished_at: UTC timestamp when the run ended, or None if still running.
        command: CLI sub-command name, e.g. "embed-concepts".
        python_version: Python interpreter version string.
    """

    run_id: str
    git_sha: str | None
    git_dirty: bool
    config_hash: str
    started_at: datetime
    finished_at: datetime | None = None
    command: str
    python_version: str


def collect_git_info() -> tuple[str | None, bool]:
    """Retrieve the current git SHA and dirty status.

    Returns:
        A tuple of (sha_or_none, is_dirty). Returns (None, False) if git
        is unavailable or the directory is not a repository.
    """
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha: str | None = sha_result.stdout.strip()

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        is_dirty = bool(status_result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False
    return sha, is_dirty


def write_metadata(metadata: RunMetadata, path: Path) -> None:
    """Serialise RunMetadata to a pretty-printed JSON file.

    Args:
        metadata: The metadata object to persist.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # mode="json" converts datetime fields to ISO 8601 strings automatically
    raw = metadata.model_dump(mode="json")
    path.write_text(json.dumps(raw, indent=2))


def _python_version() -> str:
    """Return the current Python version string."""
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"
