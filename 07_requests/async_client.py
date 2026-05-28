"""
async_client.py - Production-ready async client for the Anthropic API.

Uses asyncio, a semaphore for concurrency control, a token-bucket rate
limiter, and exponential-backoff retries to handle 429 / 500 responses.
Tracks per-instance statistics (requests sent, failures, average latency,
cumulative cost) for observability.

Demo (if __name__ == "__main__"): sends 10 prompts serially then the same
10 prompts fully in parallel.  Reports total time, total cost, and
throughput (requests / second) for both approaches side by side.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Pricing (USD per token)
# ---------------------------------------------------------------------------

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":            {"input": 0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    "claude-haiku-4-5-20251001":   {"input": 0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    "claude-sonnet-4-5":           {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-6":           {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-opus-4-7":             {"input": 5.00 / 1_000_000, "output": 25.00 / 1_000_000},
}
_DEFAULT_INPUT_PRICE:  float = 3.00 / 1_000_000
_DEFAULT_OUTPUT_PRICE: float = 15.00 / 1_000_000

_DEFAULT_MODEL:      str = "claude-haiku-4-5"
_DEFAULT_MAX_TOKENS: int = 150


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _MODEL_PRICING.get(
        model,
        {"input": _DEFAULT_INPUT_PRICE, "output": _DEFAULT_OUTPUT_PRICE},
    )
    return pricing["input"] * input_tokens + pricing["output"] * output_tokens


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompletionResult:
    """
    Outcome of a single completion request.

    Attributes
    ----------
    text:
        Model-generated text, or empty string on failure.
    model:
        Model ID used for the request.
    input_tokens:
        Prompt tokens billed by the API.
    output_tokens:
        Generated tokens billed by the API.
    latency_ms:
        Wall-clock time from initial dispatch (including any retry waits)
        until the final response was received.
    cost_usd:
        Estimated USD cost for this call.
    success:
        False when all retries were exhausted or a non-retryable error occurred.
    error:
        Error description when ``success`` is False, otherwise None.
    attempts:
        Total attempt count, including the initial try and all retries.
    """

    text:          str
    model:         str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    cost_usd:      float
    success:       bool
    error:         Optional[str] = None
    attempts:      int = 1


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    """
    Async token-bucket rate limiter.

    Tokens accumulate at ``rate_per_second`` up to a ``burst`` ceiling.
    Each :meth:`acquire` call consumes one token, suspending the caller
    via ``asyncio.sleep`` when the bucket is empty.

    Parameters
    ----------
    rate_per_second:
        Sustained refill rate (tokens / second).
    burst:
        Maximum bucket capacity.  Allows a short burst above the sustained
        rate if tokens have accumulated while the client was idle.
    """

    def __init__(self, rate_per_second: float, burst: float) -> None:
        self._rate:        float = rate_per_second
        self._burst:       float = burst
        self._tokens:      float = burst          # start full
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        while True:
            async with self._lock:
                now     = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens      = min(self._burst, self._tokens + elapsed * self._rate)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Compute how long until one token is available.
                wait = (1.0 - self._tokens) / self._rate

            # Sleep outside the lock so other waiters can progress.
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Internal stats container
# ---------------------------------------------------------------------------


class _ClientStats:
    """Thread-safe (asyncio-safe) counters for the client."""

    __slots__ = (
        "requests_sent",
        "requests_failed",
        "_successful_count",
        "_successful_latency_ms",
        "total_cost_usd",
        "_lock",
    )

    def __init__(self) -> None:
        self.requests_sent:          int   = 0
        self.requests_failed:        int   = 0
        self._successful_count:      int   = 0
        self._successful_latency_ms: float = 0.0
        self.total_cost_usd:         float = 0.0
        self._lock = asyncio.Lock()

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency of successful requests (ms).  0 if none succeeded."""
        if self._successful_count == 0:
            return 0.0
        return self._successful_latency_ms / self._successful_count

    async def record_success(self, latency_ms: float, cost_usd: float) -> None:
        async with self._lock:
            self.requests_sent          += 1
            self._successful_count      += 1
            self._successful_latency_ms += latency_ms
            self.total_cost_usd         += cost_usd

    async def record_failure(self) -> None:
        async with self._lock:
            self.requests_sent   += 1
            self.requests_failed += 1


# ---------------------------------------------------------------------------
# Async LLM client
# ---------------------------------------------------------------------------


class AsyncLLMClient:
    """
    Production-ready async client for the Anthropic Messages API.

    Wraps ``anthropic.AsyncAnthropic`` with:

    * **Concurrency control** — an ``asyncio.Semaphore`` caps in-flight
      requests at ``max_concurrent``.
    * **Rate limiting** — a token-bucket algorithm enforces
      ``requests_per_minute`` at a sustained rate, with an initial burst
      up to ``max_concurrent`` so a cold batch isn't artificially stalled.
    * **Automatic retry** — transient errors (HTTP 429, 500-504) are
      retried up to ``max_retries`` times with exponential back-off
      (1 s → 2 s → 4 s … capped at 30 s).
    * **Observability** — per-instance statistics via read-only properties.

    Parameters
    ----------
    max_concurrent:
        Maximum number of simultaneous in-flight requests.
    requests_per_minute:
        Sustained request-rate cap enforced by the token bucket.
    timeout_seconds:
        Per-attempt timeout.  Exceeded attempts raise ``asyncio.TimeoutError``
        and are retried like any other transient error.
    max_retries:
        Maximum retry attempts *after* the first failure (3 → up to 4 tries).
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        max_concurrent:      int           = 5,
        requests_per_minute: int           = 50,
        timeout_seconds:     int           = 30,
        max_retries:         int           = 3,
        api_key:             Optional[str] = None,
    ) -> None:
        self.max_concurrent      = max_concurrent
        self.requests_per_minute = requests_per_minute
        self.timeout_seconds     = timeout_seconds
        self.max_retries         = max_retries

        resolved_key   = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client   = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        rate_per_second = requests_per_minute / 60.0
        # Allow an initial burst up to max_concurrent so the first batch
        # of requests is not artificially serialised by the token bucket.
        burst = max(rate_per_second, float(max_concurrent))
        self._bucket = _TokenBucket(rate_per_second=rate_per_second, burst=burst)

        self._stats = _ClientStats()

    # ------------------------------------------------------------------
    # Read-only statistics
    # ------------------------------------------------------------------

    @property
    def requests_sent(self) -> int:
        """Total logical requests dispatched (not counting individual retries)."""
        return self._stats.requests_sent

    @property
    def requests_failed(self) -> int:
        """Requests that ultimately failed after exhausting all retries."""
        return self._stats.requests_failed

    @property
    def avg_latency_ms(self) -> float:
        """Mean wall-clock latency of successful requests in milliseconds."""
        return self._stats.avg_latency_ms

    @property
    def total_cost_usd(self) -> float:
        """Cumulative estimated USD cost across all successful requests."""
        return self._stats.total_cost_usd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True for transient errors that warrant a retry."""
        if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError)):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code in (429, 500, 502, 503, 504)
        if isinstance(exc, asyncio.TimeoutError):
            return True
        return False

    async def _call_api(
        self,
        prompt:     str,
        model:      str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """
        Dispatch one ``messages.create`` call.

        Returns
        -------
        tuple[str, int, int]
            ``(text, input_tokens, output_tokens)``

        Raises
        ------
        anthropic.APIError
            On any non-retryable API error.
        asyncio.TimeoutError
            When the response takes longer than ``timeout_seconds``.
        """
        response = await asyncio.wait_for(
            self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=self.timeout_seconds,
        )
        text = response.content[0].text if response.content else ""
        return text, response.usage.input_tokens, response.usage.output_tokens

    async def _complete_with_retry(
        self,
        prompt:     str,
        model:      str,
        max_tokens: int,
    ) -> CompletionResult:
        """Attempt the request up to ``max_retries + 1`` times."""
        backoff:  float             = 1.0
        last_exc: Optional[Exception] = None
        t_start = time.perf_counter()

        for attempt in range(1, self.max_retries + 2):  # 1 initial + max_retries
            try:
                text, in_tok, out_tok = await self._call_api(prompt, model, max_tokens)
                latency_ms = (time.perf_counter() - t_start) * 1000
                cost       = _calc_cost(model, in_tok, out_tok)
                await self._stats.record_success(latency_ms, cost)

                return CompletionResult(
                    text=text,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    success=True,
                    attempts=attempt,
                )

            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc) or attempt > self.max_retries:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        await self._stats.record_failure()
        return CompletionResult(
            text="",
            model=model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(last_exc),
            attempts=self.max_retries + 1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt:     str,
        model:      str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> CompletionResult:
        """
        Send one completion request with rate limiting, concurrency control,
        and automatic retry.

        The call first waits for a token from the rate-limiter bucket, then
        acquires the concurrency semaphore before touching the network.

        Parameters
        ----------
        prompt:
            User message to send to the model.
        model:
            Anthropic model ID.
        max_tokens:
            Upper bound on generated tokens.

        Returns
        -------
        CompletionResult
            Always returns a result object; inspect ``success`` to detect
            failures rather than catching exceptions.
        """
        await self._bucket.acquire()
        async with self._semaphore:
            return await self._complete_with_retry(prompt, model, max_tokens)

    async def batch_complete(
        self,
        prompts:    list[str],
        model:      str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> list[CompletionResult]:
        """
        Process all prompts concurrently while respecting rate and concurrency
        limits.

        All tasks are created immediately and scheduled together; the semaphore
        and token bucket internally serialise access to the API as required.
        The returned list preserves the original prompt ordering.

        Parameters
        ----------
        prompts:
            Ordered list of user messages to complete.
        model:
            Anthropic model ID applied to every request.
        max_tokens:
            Upper bound on generated tokens per request.

        Returns
        -------
        list[CompletionResult]
            One ``CompletionResult`` per prompt, in the original order.
        """
        tasks = [
            asyncio.create_task(self.complete(prompt, model, max_tokens))
            for prompt in prompts
        ]
        return list(await asyncio.gather(*tasks))

    async def close(self) -> None:
        """Close the underlying HTTP transport and release connections."""
        await self._client.close()


# ---------------------------------------------------------------------------
# Demo: serial vs. parallel
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PROMPTS: list[str] = [
        "In one sentence, what is a large language model?",
        "What is the difference between latency and throughput?",
        "Name three use cases for the Anthropic API.",
        "In one sentence, explain what rate limiting is.",
        "What is asyncio in Python? Answer in one sentence.",
        "Why is exponential backoff useful for retries? One sentence.",
        "What does 'concurrent requests' mean? Answer briefly.",
        "In one sentence, what is a token in NLP?",
        "Why is caching useful for LLM APIs? One sentence.",
        "In one sentence, what is cost optimisation in AI?",
    ]
    _MODEL:      str = "claude-haiku-4-5"
    _MAX_TOKENS: int = 80

    # ------------------------------------------------------------------
    # Serial run: one request at a time
    # ------------------------------------------------------------------

    async def _run_serial() -> tuple[list[CompletionResult], float, AsyncLLMClient]:
        client = AsyncLLMClient(
            max_concurrent=1,
            requests_per_minute=120,
            timeout_seconds=30,
        )
        results: list[CompletionResult] = []
        t0 = time.perf_counter()
        for i, prompt in enumerate(_PROMPTS, 1):
            print(f"  [serial] {i:>2}/{len(_PROMPTS)}  dispatching...", end="\r", flush=True)
            r = await client.complete(prompt, _MODEL, _MAX_TOKENS)
            results.append(r)
        elapsed = time.perf_counter() - t0
        print()
        return results, elapsed, client

    # ------------------------------------------------------------------
    # Parallel run: all requests concurrently
    # ------------------------------------------------------------------

    async def _run_parallel() -> tuple[list[CompletionResult], float, AsyncLLMClient]:
        client = AsyncLLMClient(
            max_concurrent=10,
            requests_per_minute=120,
            timeout_seconds=30,
        )
        t0 = time.perf_counter()
        results = await client.batch_complete(_PROMPTS, _MODEL, _MAX_TOKENS)
        elapsed = time.perf_counter() - t0
        return results, elapsed, client

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def _main() -> None:
        n = len(_PROMPTS)
        sep_wide  = "=" * 68
        sep_thin  = "-" * 68

        print(f"\n{sep_wide}")
        print(f"  ASYNC CLIENT DEMO  —  {n} prompts  |  model: {_MODEL}")
        print(sep_wide)

        # --- Serial -------------------------------------------------------
        print(f"\n  Running {n} requests SERIALLY …")
        serial_results, serial_elapsed, serial_client = await _run_serial()
        await serial_client.close()

        serial_ok       = sum(1 for r in serial_results if r.success)
        serial_cost     = serial_client.total_cost_usd
        serial_tput     = n / serial_elapsed
        serial_avg_lat  = serial_client.avg_latency_ms

        # --- Parallel -----------------------------------------------------
        print(f"  Running {n} requests in PARALLEL …")
        parallel_results, parallel_elapsed, parallel_client = await _run_parallel()
        await parallel_client.close()

        parallel_ok      = sum(1 for r in parallel_results if r.success)
        parallel_cost    = parallel_client.total_cost_usd
        parallel_tput    = n / parallel_elapsed
        parallel_avg_lat = parallel_client.avg_latency_ms

        speedup          = serial_elapsed / parallel_elapsed if parallel_elapsed > 0 else 0

        # --- Comparison table ---------------------------------------------
        print(f"\n{sep_wide}")
        print(f"  {'METRIC':<30}  {'SERIAL':>14}  {'PARALLEL':>14}")
        print(sep_thin)
        print(f"  {'Total time (s)':<30}  {serial_elapsed:>13.2f}s  {parallel_elapsed:>13.2f}s")
        print(f"  {'Throughput (req/s)':<30}  {serial_tput:>13.2f}   {parallel_tput:>13.2f} ")
        print(f"  {'Avg latency per req (ms)':<30}  {serial_avg_lat:>12.0f}ms  {parallel_avg_lat:>12.0f}ms")
        print(f"  {'Successful requests':<30}  {serial_ok:>13}    {parallel_ok:>13}  ")
        print(f"  {'Total cost (USD)':<30}  ${serial_cost:>13.6f}  ${parallel_cost:>13.6f}")
        print(sep_thin)
        print(f"  Speedup (serial / parallel):  {speedup:.2f}x")
        print(f"  Time saved:                   {serial_elapsed - parallel_elapsed:.2f}s  "
              f"({(1 - parallel_elapsed / serial_elapsed) * 100:.1f}% faster)")
        print(sep_wide)

        # --- Per-request detail -------------------------------------------
        print(f"\n  {'#':<3}  {'SERIAL lat':>10}  {'SERIAL ok':>9}  "
              f"{'PARALLEL lat':>12}  {'PARALLEL ok':>11}")
        print(f"  {'-' * 3}  {'-' * 10}  {'-' * 9}  {'-' * 12}  {'-' * 11}")
        for i, (sr, pr) in enumerate(zip(serial_results, parallel_results), 1):
            s_lat  = f"{sr.latency_ms:.0f}ms" if sr.success else "FAIL"
            p_lat  = f"{pr.latency_ms:.0f}ms" if pr.success else "FAIL"
            s_ok   = "ok" if sr.success else f"err({sr.attempts})"
            p_ok   = "ok" if pr.success else f"err({pr.attempts})"
            print(f"  {i:<3}  {s_lat:>10}  {s_ok:>9}  {p_lat:>12}  {p_ok:>11}")
        print()

    asyncio.run(_main())
