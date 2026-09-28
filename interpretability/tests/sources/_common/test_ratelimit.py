"""Tests for TokenBucket rate limiter (moved from bioportal/)."""

import time

from molf_interp.sources._common.ratelimit import TokenBucket


async def test_rate_enforced_sequential() -> None:
    bucket = TokenBucket(rate_per_second=10, burst=1)
    times = []
    for _ in range(5):
        await bucket.acquire()
        times.append(time.monotonic())

    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert all(g >= 0.09 for g in gaps), f"Gap too small: {gaps}"


async def test_burst_allows_immediate_acquires() -> None:
    burst = 3
    bucket = TokenBucket(rate_per_second=1, burst=burst)
    start = time.monotonic()
    for _ in range(burst):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    # burst tokens should be available immediately (well under 1 second)
    assert elapsed < 1.0, f"Burst too slow: {elapsed:.3f}s"


async def test_context_manager() -> None:
    bucket = TokenBucket(rate_per_second=100, burst=10)
    async with bucket:
        pass  # should not raise
