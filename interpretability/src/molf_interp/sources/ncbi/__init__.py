"""NCBI E-utilities client package."""

from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.config import NCBIConfig
from molf_interp.sources.ncbi.exceptions import (
    AuthenticationError,
    MalformedResponseError,
    NCBIError,
    RateLimitedError,
    TransientError,
)
from molf_interp.sources.ncbi.models import ESearchResult, PubMedAbstract, PubMedAuthor

__all__ = [
    "AuthenticationError",
    "ESearchResult",
    "MalformedResponseError",
    "NCBIClient",
    "NCBIConfig",
    "NCBIError",
    "PubMedAbstract",
    "PubMedAuthor",
    "RateLimitedError",
    "TransientError",
]
