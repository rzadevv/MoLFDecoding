"""Async BioPortal HTTP client with cache, rate limiting, and retries."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, cast
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from molf_interp.sources._common.cache import ResponseCache
from molf_interp.sources._common.ratelimit import TokenBucket
from molf_interp.sources.bioportal.config import BioPortalConfig
from molf_interp.sources.bioportal.exceptions import (
    AuthenticationError,
    BioPortalError,
    LicensedOntologyError,
    NotFoundError,
    RateLimitedError,
    TransientError,
)
from molf_interp.sources.bioportal.models import OntologyClass, SearchResult

_LICENSE_KEYWORDS = ("license", "terms of use", "agreement", "subscribe", "licensed")


def _encode_class_iri(iri: str) -> str:
    """URL-encode a class IRI for embedding in BioPortal path segments.

    Args:
        iri: Full class IRI.

    Returns:
        Percent-encoded string safe for use in URL paths.
    """
    return quote(iri, safe="")


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that warrant a retry."""
    if isinstance(exc, httpx.NetworkError | httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _map_http_error(
    exc: httpx.HTTPStatusError, ontology_acronym: str | None = None
) -> BioPortalError:
    """Translate an httpx status error to the appropriate BioPortalError subclass.

    Args:
        exc: The HTTP status error.
        ontology_acronym: Acronym hint for constructing LicensedOntologyError.

    Returns:
        A BioPortalError instance.
    """
    status = exc.response.status_code
    if status == 401:
        return AuthenticationError("HTTP 401: API key missing, invalid, or revoked")
    if status == 403:
        body = exc.response.text.lower()
        if any(kw in body for kw in _LICENSE_KEYWORDS) or ontology_acronym:
            acronym = ontology_acronym or "UNKNOWN"
            return LicensedOntologyError(acronym, f"HTTP 403: license required for {acronym}")
        return AuthenticationError("HTTP 403: access denied")
    if status == 404:
        return NotFoundError(f"HTTP 404: resource not found — {exc.request.url}")
    if status == 429:
        return RateLimitedError("HTTP 429: rate limit exceeded after retries")
    return TransientError(f"HTTP {status}: transient server error after retries")


class BioPortalClient:
    """Async HTTP client for the BioPortal REST API.

    Handles authentication, rate limiting, disk caching, and retries
    transparently. All public methods are async.

    Usage:
        async with BioPortalClient.from_config(cfg) as client:
            ont = await client.get_ontology("NCIT")
            async for cls in client.iter_descendants("NCIT", root_iri="http://..."):
                process(cls)
    """

    @classmethod
    def from_config(cls, config: BioPortalConfig) -> BioPortalClient:
        """Construct a client from config, reading the API key from the environment.

        Args:
            config: Client configuration.

        Returns:
            Configured BioPortalClient.

        Raises:
            OSError: If the API key env var is not set.
        """
        load_dotenv()
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise OSError(
                f"BioPortal API key not found. "
                f"Set {config.api_key_env!r} in the environment or .env file."
            )
        return cls(config=config, api_key=api_key)

    def __init__(
        self,
        config: BioPortalConfig,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        cache: ResponseCache | None = None,
        rate_limiter: TokenBucket | None = None,
    ) -> None:
        """Initialise with optional dependency injection for testing.

        Args:
            config: Client configuration.
            api_key: BioPortal API key (never logged).
            http_client: Pre-configured httpx client, or None to create one.
            cache: ResponseCache, or None to create from config.
            rate_limiter: TokenBucket, or None to create from config.
        """
        self._config = config
        self._api_key = api_key
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": config.user_agent},
        )
        self._cache = cache or ResponseCache(config.cache_dir)
        self._rate_limiter = rate_limiter or TokenBucket(
            rate_per_second=config.rate_limit_per_second,
            burst=config.rate_limit_burst,
        )
        self._base_url = config.base_url.rstrip("/")

    async def __aenter__(self) -> BioPortalClient:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client if we created it."""
        if self._owns_http_client:
            await self._http_client.aclose()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def get_ontology(self, acronym: str) -> dict[str, Any]:
        """Return raw ontology metadata from BioPortal.

        Args:
            acronym: Ontology acronym, e.g. "NCIT".

        Returns:
            Raw metadata dict.

        Raises:
            NotFoundError: Ontology not found.
            LicensedOntologyError: License required.
            AuthenticationError: Bad or missing API key.
        """
        return await self._request(f"/ontologies/{acronym}", ont_hint=acronym)

    async def can_access(self, acronym: str) -> bool:
        """Return True if the ontology is accessible with the current API key.

        Args:
            acronym: Ontology acronym.

        Returns:
            True on success, False on LicensedOntologyError or NotFoundError.
        """
        try:
            await self.get_ontology(acronym)
            return True
        except (LicensedOntologyError, NotFoundError):
            return False

    async def get_class(self, ontology_acronym: str, class_iri: str) -> OntologyClass:
        """Fetch a single ontology class by IRI.

        Args:
            ontology_acronym: Ontology acronym.
            class_iri: Full class IRI.

        Returns:
            Normalized OntologyClass.

        Raises:
            NotFoundError: Class not found.
        """
        encoded = _encode_class_iri(class_iri)
        path = f"/ontologies/{ontology_acronym}/classes/{encoded}"
        payload = await self._request(path, ont_hint=ontology_acronym)
        return OntologyClass.from_bioportal_json(payload, ontology_acronym)

    async def iter_descendants(
        self,
        ontology_acronym: str,
        root_iri: str,
        *,
        include_obsolete: bool = False,
    ) -> AsyncIterator[OntologyClass]:
        """Yield every descendant of root_iri, paginating transparently.

        Args:
            ontology_acronym: Ontology acronym.
            root_iri: Root class IRI.
            include_obsolete: Include obsolete classes when True.

        Yields:
            OntologyClass for each descendant.
        """
        encoded = _encode_class_iri(root_iri)
        path = f"/ontologies/{ontology_acronym}/classes/{encoded}/descendants"
        params: dict[str, Any] = {"pagesize": self._config.page_size, "page": 1}
        async for item in self._paginate(path, params, ont_hint=ontology_acronym):
            cls = OntologyClass.from_bioportal_json(item, ontology_acronym)
            if cls.obsolete and not include_obsolete:
                continue
            yield cls

    async def iter_children(
        self,
        ontology_acronym: str,
        parent_iri: str,
        *,
        include_obsolete: bool = False,
    ) -> AsyncIterator[OntologyClass]:
        """Yield direct children of parent_iri, paginating transparently.

        Args:
            ontology_acronym: Ontology acronym.
            parent_iri: Parent class IRI.
            include_obsolete: Include obsolete classes when True.

        Yields:
            OntologyClass for each child.
        """
        encoded = _encode_class_iri(parent_iri)
        path = f"/ontologies/{ontology_acronym}/classes/{encoded}/children"
        params: dict[str, Any] = {"pagesize": self._config.page_size, "page": 1}
        async for item in self._paginate(path, params, ont_hint=ontology_acronym):
            cls = OntologyClass.from_bioportal_json(item, ontology_acronym)
            if cls.obsolete and not include_obsolete:
                continue
            yield cls

    async def search(
        self,
        query: str,
        *,
        ontologies: Sequence[str] | None = None,
        require_exact_match: bool = False,
        include_definitions: bool = False,
        page_size: int | None = None,
    ) -> AsyncIterator[SearchResult]:
        """Paginated cross-ontology text search.

        Args:
            query: Search string.
            ontologies: Limit to these acronyms; None means all.
            require_exact_match: Add exact_match=true when True.
            include_definitions: Include definitions in response when True.
            page_size: Override the default page size.

        Returns:
            Async iterator of SearchResult objects.
        """
        params: dict[str, Any] = {
            "q": query,
            "pagesize": page_size or self._config.page_size,
            "page": 1,
        }
        if ontologies:
            params["ontologies"] = ",".join(ontologies)
        if require_exact_match:
            params["exact_match"] = "true"
        if include_definitions:
            params["include"] = "prefLabel,synonym,definition"
        return self._search_iter(params)

    async def _search_iter(self, params: dict[str, Any]) -> AsyncIterator[SearchResult]:
        """Internal search async generator.

        Args:
            params: Full query parameter dict for the /search endpoint.

        Yields:
            SearchResult for each matching class.
        """
        async for item in self._paginate("/search", params):
            iri = item.get("@id", "")
            pref_label = item.get("prefLabel", "")
            if not iri or not pref_label:
                continue
            links: dict[str, Any] = item.get("links", {}) or {}
            ont_link: str = links.get("ontology", "")
            acronym = ont_link.rstrip("/").split("/")[-1] if ont_link else ""
            score_raw = item.get("score")
            score: float | None = float(score_raw) if score_raw is not None else None
            yield SearchResult(
                iri=str(iri),
                ontology_acronym=acronym,
                pref_label=str(pref_label),
                score=score,
            )

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        ont_hint: str | None = None,
    ) -> dict[str, Any]:
        """Perform one HTTP GET with cache, rate limit, retries, and error mapping.

        Args:
            path: URL path relative to base_url.
            params: Optional query parameters.
            ont_hint: Ontology acronym for richer 403 error messages.

        Returns:
            Parsed JSON response dict.

        Raises:
            BioPortalError subclass on any non-2xx response.
        """
        url = f"{self._base_url}{path}"

        cached = self._cache.get(url, params)
        if cached is not None:
            logger.debug("Cache hit: {}", path)
            return cached

        await self._rate_limiter.acquire()
        response_data = await self._do_fetch(url, params, ont_hint=ont_hint)
        self._cache.put(url, params, response_data)
        return response_data

    async def _do_fetch(
        self,
        url: str,
        params: dict[str, Any] | None,
        *,
        ont_hint: str | None = None,
    ) -> dict[str, Any]:
        """Execute the HTTP GET with tenacity retries.

        Args:
            url: Absolute URL.
            params: Optional query parameters.
            ont_hint: Ontology acronym for error context.

        Returns:
            Parsed JSON response dict.
        """
        max_attempts = self._config.max_retries + 1
        client = self._http_client
        api_key = self._api_key

        @retry(  # type: ignore[misc]
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        )
        async def _attempt() -> dict[str, Any]:
            headers = {
                "Authorization": f"apikey token={api_key}",
                "Accept": "application/json",
            }
            logger.debug("GET {}", url)
            try:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if not _is_retryable(exc):
                    raise _map_http_error(exc, ont_hint) from exc
                raise
            return cast("dict[str, Any]", resp.json())

        try:
            return await _attempt()
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(exc, ont_hint) from exc

    async def _paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        ont_hint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield items across pages by following BioPortal's nextPage link.

        Args:
            path: URL path for the first page.
            params: Base query parameters.
            ont_hint: Ontology acronym for error context.

        Yields:
            Each item dict from each page's collection array.
        """
        data = await self._request(path, dict(params or {}), ont_hint=ont_hint)

        while True:
            collection: list[dict[str, Any]] = data.get("collection", [])
            for item in collection:
                yield item

            links: dict[str, Any] = data.get("links", {}) or {}
            next_page_url: str | None = links.get("nextPage")

            if not next_page_url:
                # Fall back to page-count check if nextPage absent
                current = int(data.get("page", 1))
                total = int(data.get("pageCount", 1))
                if current >= total:
                    break
                logger.warning("nextPage link missing; constructing next URL from page/pageCount")
                next_page_url = f"{self._base_url}{path}"
                next_params = dict(params or {})
                next_params["page"] = current + 1
                data = await self._request(path, next_params, ont_hint=ont_hint)
            else:
                # nextPage is a full URL with query string embedded
                data = await self._fetch_full_url(next_page_url, ont_hint=ont_hint)

    async def _fetch_full_url(
        self, full_url: str, *, ont_hint: str | None = None
    ) -> dict[str, Any]:
        """GET a fully-qualified URL (used when nextPage includes all params).

        Args:
            full_url: Complete URL including query string.
            ont_hint: Ontology acronym for error context.

        Returns:
            Parsed JSON response dict.
        """
        cached = self._cache.get(full_url)
        if cached is not None:
            logger.debug("Cache hit (full URL): {}", full_url)
            return cached

        await self._rate_limiter.acquire()
        result = await self._do_fetch(full_url, None, ont_hint=ont_hint)
        self._cache.put(full_url, None, result)
        return result
