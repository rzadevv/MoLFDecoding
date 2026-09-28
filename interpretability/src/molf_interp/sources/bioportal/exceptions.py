"""Custom exception hierarchy for BioPortal client errors."""

from __future__ import annotations


class BioPortalError(Exception):
    """Base for all BioPortal client errors."""


class AuthenticationError(BioPortalError):
    """Raised on HTTP 401. API key missing, invalid, or revoked."""


class LicensedOntologyError(BioPortalError):
    """Raised on HTTP 403 for a licensed ontology (e.g. SNOMED CT).

    Distinct from AuthenticationError so harvesters can implement fallback
    logic (e.g. SNOMED-locked → use NCIt morphology subtree instead).
    """

    def __init__(self, ontology_acronym: str, message: str | None = None) -> None:
        """Initialise with the acronym of the gated ontology.

        Args:
            ontology_acronym: Ontology acronym that requires a license (e.g. "SNOMEDCT").
            message: Optional human-readable explanation.
        """
        super().__init__(message or f"License required to access ontology: {ontology_acronym}")
        self.ontology_acronym = ontology_acronym


class NotFoundError(BioPortalError):
    """Raised on HTTP 404 for a class/ontology that does not exist."""


class RateLimitedError(BioPortalError):
    """Raised on HTTP 429 after retries are exhausted (should be rare)."""


class TransientError(BioPortalError):
    """Raised on HTTP 5xx after retries are exhausted."""
