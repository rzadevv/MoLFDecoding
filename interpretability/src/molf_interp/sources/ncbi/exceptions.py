"""NCBI E-utilities exception hierarchy."""

from __future__ import annotations


class NCBIError(Exception):
    """Base class for all NCBI E-utilities errors."""


class AuthenticationError(NCBIError):
    """Invalid or missing NCBI API key."""


class RateLimitedError(NCBIError):
    """Rate limit exceeded after all retries."""


class TransientError(NCBIError):
    """Transient server error (5xx) persisted after all retries."""


class MalformedResponseError(NCBIError):
    """NCBI returned an unexpected XML / response shape."""
