"""BioPortal client configuration model."""

from __future__ import annotations

from pathlib import Path

from molf_interp.io.config import BaseConfig


class BioPortalConfig(BaseConfig):
    """BioPortal client configuration.

    The actual API key is never stored here; only the env var name that
    holds it. Load the key at runtime via python-dotenv + os.environ.
    """

    base_url: str = "https://data.bioontology.org"
    api_key_env: str = "BIOPORTAL_API_KEY"
    # Cap at 10 to leave 5 req/s headroom under the 15/s server limit.
    rate_limit_per_second: int = 10
    rate_limit_burst: int = 5
    timeout_seconds: float = 30.0
    max_retries: int = 5
    page_size: int = 200
    cache_dir: Path = Path("data/cache/bioportal")
    user_agent: str = "molf-interp/0.1 (https://github.com/.../molf-interpretability)"
