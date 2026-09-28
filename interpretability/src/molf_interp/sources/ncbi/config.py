"""NCBI E-utilities client configuration."""

from __future__ import annotations

from pathlib import Path

from molf_interp.io.config import BaseConfig


class NCBIConfig(BaseConfig):
    """NCBI E-utilities client configuration.

    The actual API key and email are never stored here — only the env var names.
    """

    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key_env: str = "NCBI_API_KEY"
    email_env: str = "NCBI_EMAIL"
    tool_name: str = "molf-interp"
    rate_limit_per_second: int = 8
    rate_limit_burst: int = 3
    rate_limit_no_key: int = 2
    timeout_seconds: float = 60.0
    max_retries: int = 5
    efetch_batch_size: int = 200
    cache_dir: Path = Path("data/cache/pubmed")
    user_agent: str = "molf-interp/0.1 (research; contact via NCBI_EMAIL)"
