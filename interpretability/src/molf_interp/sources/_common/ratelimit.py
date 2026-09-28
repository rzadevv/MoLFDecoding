"""Async token-bucket rate limiter — shared across NCBI and BioPortal clients."""

from __future__ import annotations

from types import TracebackType

from aiolimiter import AsyncLimiter


class TokenBucket:
    """Async token bucket with burst capacity.

    Wraps aiolimiter.AsyncLimiter. Setting time_period=burst and
    max_rate=rate*burst gives a bucket that allows `burst` immediate
    acquires before the sustained rate kicks in.
    """

    def __init__(self, rate_per_second: int, burst: int = 1) -> None:
        """Initialise the token bucket.

        Args:
            rate_per_second: Sustained rate in tokens per second.
            burst: Number of tokens that can be consumed immediately.
        """
        # AsyncLimiter(max_rate, time_period) allows max_rate tokens per time_period seconds,
        # starting with max_rate tokens in the bucket (the burst capacity).
        # To get `burst` initial tokens at a sustained rate of `rate_per_second` ops/sec:
        #   max_rate = burst, time_period = burst / rate_per_second
        self._limiter = AsyncLimiter(
            max_rate=float(burst),
            time_period=float(burst) / rate_per_second,
        )

    async def acquire(self) -> None:
        """Block until a token is available."""
        await self._limiter.acquire()

    async def __aenter__(self) -> TokenBucket:
        """Acquire a token on entry."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """No-op on exit."""
