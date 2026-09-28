"""Async NCBI E-utilities client with cache, rate limiting, and retries."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator, Sequence
from types import TracebackType
from typing import Any, Literal

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
from molf_interp.sources.ncbi.config import NCBIConfig
from molf_interp.sources.ncbi.exceptions import (
    AuthenticationError,
    MalformedResponseError,
    NCBIError,
    RateLimitedError,
    TransientError,
)
from molf_interp.sources.ncbi.models import ESearchResult, PubMedAbstract
from molf_interp.sources.ncbi.parser import (
    parse_efetch_pubmed_response,
    parse_esearch_response,
)

_AUTH_KEYWORDS = ("api key", "apikey", "invalid key")
_RATE_KEYWORDS = ("rate limit", "too many requests")

# Params excluded from cache key
_CACHE_EXCLUDE_PARAMS = frozenset({"api_key", "email", "tool"})


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that warrant a retry."""
    if isinstance(exc, httpx.NetworkError | httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _cache_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip auth/identity params so cache key is query-content-only."""
    return {k: v for k, v in params.items() if k not in _CACHE_EXCLUDE_PARAMS}


def _check_error_body(xml_text: str) -> None:
    """Raise NCBIError if the response body contains an NCBI <ERROR> element.

    NCBI sometimes returns HTTP 200 with error XML. We must detect these.

    Args:
        xml_text: Raw XML response text.

    Raises:
        AuthenticationError: If the error mentions an API key problem.
        RateLimitedError: If the error mentions rate limiting.
        MalformedResponseError: On other embedded errors.
    """
    if "<ERROR>" not in xml_text:
        return
    start = xml_text.find("<ERROR>") + len("<ERROR>")
    end = xml_text.find("</ERROR>", start)
    msg = xml_text[start:end].strip() if end > start else "Unknown NCBI error"
    lower = msg.lower()
    if any(kw in lower for kw in _AUTH_KEYWORDS):
        raise AuthenticationError(f"NCBI API key error: {msg}")
    if any(kw in lower for kw in _RATE_KEYWORDS):
        raise RateLimitedError(f"NCBI rate limit: {msg}")
    raise MalformedResponseError(f"NCBI error body: {msg}")


class NCBIClient:
    """Async NCBI E-utilities client focused on PubMed (db=pubmed).

    Usage:
        async with NCBIClient.from_config(cfg) as client:
            result = await client.esearch_pubmed(
                query='"hematoxylin and eosin"[tiab]',
                max_results=5000,
            )
            async for abstract in client.efetch_pubmed_abstracts(
                webenv=result.webenv,
                query_key=result.query_key,
                total=result.count,
            ):
                ...
    """

    @classmethod
    def from_config(cls, config: NCBIConfig) -> NCBIClient:
        """Construct a client from config, reading credentials from the environment.

        Args:
            config: Client configuration.

        Returns:
            Configured NCBIClient (api_key may be None if env var not set).
        """
        load_dotenv()
        api_key = os.environ.get(config.api_key_env) or None
        email = os.environ.get(config.email_env) or None
        if not api_key:
            logger.warning(
                "NCBI API key not set (env: {}). Rate limit: {}/s",
                config.api_key_env,
                config.rate_limit_no_key,
            )
        if not email:
            logger.warning(
                "NCBI_EMAIL not set (env: {}). NCBI may throttle anonymous traffic.",
                config.email_env,
            )
        return cls(config=config, api_key=api_key, email=email)

    def __init__(
        self,
        config: NCBIConfig,
        api_key: str | None,
        email: str | None,
        http_client: httpx.AsyncClient | None = None,
        cache: ResponseCache | None = None,
        rate_limiter: TokenBucket | None = None,
    ) -> None:
        """Initialise with optional dependency injection for testing.

        Args:
            config: Client configuration.
            api_key: NCBI API key, or None to fall back to lower rate limit.
            email: Contact email for NCBI politeness convention.
            http_client: Pre-configured httpx client, or None to create one.
            cache: ResponseCache, or None to create from config.
            rate_limiter: TokenBucket, or None to create from config.
        """
        self._config = config
        self._api_key = api_key
        self._email = email
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": config.user_agent},
        )
        self._cache = cache or ResponseCache(config.cache_dir)
        rate = config.rate_limit_per_second if api_key else config.rate_limit_no_key
        burst = config.rate_limit_burst if api_key else 1
        self._rate_limiter = rate_limiter or TokenBucket(
            rate_per_second=rate,
            burst=burst,
        )
        self._base_url = config.base_url.rstrip("/")

    async def __aenter__(self) -> NCBIClient:
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

    async def esearch_pubmed(
        self,
        query: str,
        *,
        max_results: int = 1000,
        date_range: tuple[str, str] | None = None,
        use_history: bool = True,
    ) -> ESearchResult:
        """Run an ESearch against PubMed.

        Args:
            query: E-utilities query string.
            max_results: Maximum records to return in the IdList.
            date_range: Optional (mindate, maxdate) as "YYYY/MM/DD".
            use_history: When True, sends usehistory=y for WebEnv/QueryKey.

        Returns:
            ESearchResult with count, pmids, and optionally webenv/query_key.
        """
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
        }
        if use_history:
            params["usehistory"] = "y"
        if date_range:
            params["mindate"] = date_range[0]
            params["maxdate"] = date_range[1]

        logger.info("ESearch PubMed: {!r} (max_results={})", query, max_results)
        xml_text = await self._request("esearch", params)
        result = parse_esearch_response(xml_text)
        logger.info("ESearch result: {} total records", result.count)
        return result

    async def efetch_pubmed_abstracts(
        self,
        *,
        pmids: Sequence[str] | None = None,
        webenv: str | None = None,
        query_key: str | None = None,
        total: int | None = None,
    ) -> AsyncGenerator[PubMedAbstract, None]:
        """Yield PubMedAbstract records in batches of efetch_batch_size.

        Either `pmids` OR (`webenv`, `query_key`, `total`) must be provided.

        Args:
            pmids: Explicit list of PMIDs to fetch.
            webenv: WebEnv from a prior ESearch with use_history=True.
            query_key: QueryKey from the same ESearch.
            total: Total records available (required with webenv/query_key).

        Yields:
            PubMedAbstract for each fetched record.

        Raises:
            ValueError: If neither pmids nor webenv+query_key+total is provided.
        """
        batch_size = self._config.efetch_batch_size

        if pmids is not None:
            pmid_list = list(pmids)
            n_batches = max(1, (len(pmid_list) + batch_size - 1) // batch_size)
            for batch_idx, i in enumerate(range(0, len(pmid_list), batch_size), 1):
                batch = pmid_list[i : i + batch_size]
                params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "xml",
                    "rettype": "abstract",
                }
                t0 = time.monotonic()
                xml_text = await self._request("efetch", params)
                abstracts = parse_efetch_pubmed_response(xml_text)
                elapsed = time.monotonic() - t0
                logger.info(
                    "efetch batch {}/{}: {} abstracts in {:.1f}s",
                    batch_idx,
                    n_batches,
                    len(abstracts),
                    elapsed,
                )
                for abstract in abstracts:
                    yield abstract

        elif webenv is not None and query_key is not None and total is not None:
            n_batches = max(1, (total + batch_size - 1) // batch_size)
            fetched = 0
            for batch_idx in range(1, n_batches + 1):
                retmax = min(batch_size, total - fetched)
                if retmax <= 0:
                    break
                params = {
                    "db": "pubmed",
                    "WebEnv": webenv,
                    "query_key": query_key,
                    "retstart": str(fetched),
                    "retmax": str(retmax),
                    "retmode": "xml",
                    "rettype": "abstract",
                }
                t0 = time.monotonic()
                xml_text = await self._request("efetch", params)
                abstracts = parse_efetch_pubmed_response(xml_text)
                elapsed = time.monotonic() - t0
                logger.info(
                    "efetch batch {}/{}: {} abstracts in {:.1f}s",
                    batch_idx,
                    n_batches,
                    len(abstracts),
                    elapsed,
                )
                for abstract in abstracts:
                    yield abstract
                    fetched += 1

        else:
            raise ValueError("Either pmids or (webenv, query_key, total) must be provided.")

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        endpoint: Literal["esearch", "efetch"],
        params: dict[str, Any],
    ) -> str:
        """Single E-utilities request returning raw XML text.

        Cache key = endpoint URL + sorted params excluding api_key/email/tool.
        On 429/5xx, retries with exponential backoff.

        Args:
            endpoint: E-utilities endpoint name.
            params: Query parameters (WITHOUT api_key/email/tool — added here).

        Returns:
            Raw XML response text.

        Raises:
            AuthenticationError: On bad API key (even with HTTP 200).
            RateLimitedError: After exhausted retries on 429.
            TransientError: After exhausted retries on 5xx.
        """
        url = f"{self._base_url}/{endpoint}.fcgi"
        cache_key_params = _cache_params(params)

        cached = self._cache.get(url, cache_key_params)
        if cached is not None:
            logger.debug("Cache hit: {} {}", endpoint, cache_key_params)
            return str(cached.get("_content", ""))

        # Build full params with auth
        full_params: dict[str, Any] = dict(params)
        if self._api_key:
            full_params["api_key"] = self._api_key
        if self._email:
            full_params["email"] = self._email
        full_params["tool"] = self._config.tool_name

        await self._rate_limiter.acquire()
        xml_text = await self._do_fetch(url, full_params)

        self._cache.put(url, cache_key_params, {"_content": xml_text})
        return xml_text

    async def _do_fetch(self, url: str, params: dict[str, Any]) -> str:
        """Execute the HTTP GET with tenacity retries.

        Args:
            url: Absolute URL.
            params: Full query parameters (including auth).

        Returns:
            Raw XML response text.

        Raises:
            NCBIError subclass on failure.
        """
        max_attempts = self._config.max_retries + 1
        client = self._http_client

        @retry(  # type: ignore[misc]
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        )
        async def _attempt() -> str:
            logger.debug("GET {}", url)
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if not _is_retryable(exc):
                    _map_http_error(exc)
                raise
            xml_text: str = resp.text
            _check_error_body(xml_text)
            return xml_text

        try:
            return str(await _attempt())
        except httpx.HTTPStatusError as exc:
            _map_http_error(exc)
            # unreachable — _map_http_error always raises
            raise NCBIError("Unexpected code path") from exc


def _map_http_error(exc: httpx.HTTPStatusError) -> None:
    """Translate an httpx status error to an NCBIError subclass and raise it.

    Args:
        exc: The HTTP status error to translate.

    Raises:
        AuthenticationError: On 401/403.
        RateLimitedError: On 429.
        TransientError: On other 5xx.
    """
    status = exc.response.status_code
    if status in (401, 403):
        raise AuthenticationError(f"HTTP {status}: credentials rejected by NCBI") from exc
    if status == 429:
        raise RateLimitedError("HTTP 429: rate limit exceeded after retries") from exc
    raise TransientError(f"HTTP {status}: transient server error after retries") from exc
