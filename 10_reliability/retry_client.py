"""
10_reliability/retry_client.py

Anthropic API client with intelligent retry, exponential backoff + jitter,
and sliding-window retry-budget protection against retry storms.

Retryable codes  : 429, 500, 502, 503, 529
Non-retryable    : 400, 401, 403, 404, 422  (fail immediately)

Delay formula::

    raw_delay = base_delay * 2 ** attempt_index
    if jitter: raw_delay += uniform(0, raw_delay * 0.10)
    delay = min(raw_delay, max_delay)
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRIABLE_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 529})
_NON_RETRIABLE_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 422})

_HTTP_REASON: dict[int, str] = {
    429: "Too Many Requests (rate limit)",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    529: "API Overloaded",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    422: "Unprocessable Entity",
}

_REPORT_WIDTH = 72


# ---------------------------------------------------------------------------
# .env loader (stdlib only)
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """Parse KEY=VALUE pairs from a .env file and populate ``os.environ``."""
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Mock infrastructure (demo scenarios only — no real HTTP)
# ---------------------------------------------------------------------------

class _SimulatedAPIError(Exception):
    """
    Stand-in for ``anthropic.APIStatusError`` in demos.

    Carries a ``status_code`` attribute so :class:`RetryClient` handles it
    identically to a real API error.
    """

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        reason = message or _HTTP_REASON.get(status_code, "Error")
        super().__init__(f"HTTP {status_code}: {reason}")


class _MockMessages:
    """
    Scripted replacement for ``client.messages``.

    Replays *script* items in order: raises exceptions, or returns a mock
    message object.  After the script is exhausted, all further calls succeed.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self._idx = 0

    def create(self, **_kwargs: Any) -> Any:
        if self._idx < len(self._script):
            item = self._script[self._idx]
            self._idx += 1
            if isinstance(item, Exception):
                raise item
        # default success response
        block = type("TextBlock", (), {"type": "text", "text": "Mock response."})()
        usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()
        return type("Message", (), {
            "id": "msg_mock_000",
            "model": "claude-haiku-4-5-20251001",
            "role": "assistant",
            "content": [block],
            "usage": usage,
            "stop_reason": "end_turn",
        })()


class _MockClient:
    """Minimal stand-in for ``anthropic.Anthropic``."""

    def __init__(self, script: list[Any]) -> None:
        self.messages = _MockMessages(script)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """Tunable parameters for :class:`RetryClient`."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter: bool = True
    budget_window_seconds: float = 60.0
    budget_max_retries: int = 10


@dataclass
class RetryEvent:
    """Immutable record of one retry attempt."""

    request_id: str
    attempt_number: int           # 1-based: first retry = 1
    error_code: int               # HTTP status code, or -1 for connection errors
    error_message: str
    delay_applied_seconds: float
    timestamp: str                # ISO-8601 UTC


@dataclass
class RetryStats:
    """Aggregate statistics returned by :meth:`RetryClient.get_retry_stats`."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_retries: int
    retried_requests: int          # requests that needed >= 1 retry
    retry_success_rate: float      # % of retried requests that eventually succeeded
    avg_retries_per_request: float
    budget_used: int
    budget_remaining: int
    retry_log: list[RetryEvent]


# ---------------------------------------------------------------------------
# Sliding-window retry budget
# ---------------------------------------------------------------------------

class _BudgetWindow:
    """
    Sliding-window guard against retry storms.

    Tracks retry timestamps over the last ``window_seconds`` and blocks
    further retries once ``max_retries`` have been consumed in that window.
    """

    def __init__(self, window_seconds: float, max_retries: int) -> None:
        self._window = window_seconds
        self._max = max_retries
        self._ts: list[float] = []

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._window
        self._ts = [t for t in self._ts if t > cutoff]

    def can_retry(self) -> bool:
        """Return True if the budget allows one more retry."""
        self._evict()
        return len(self._ts) < self._max

    def record(self) -> None:
        """Consume one retry unit from the budget."""
        self._ts.append(time.monotonic())

    @property
    def used(self) -> int:
        self._evict()
        return len(self._ts)

    @property
    def remaining(self) -> int:
        return max(0, self._max - self.used)


# ---------------------------------------------------------------------------
# RetryClient
# ---------------------------------------------------------------------------

class RetryClient:
    """
    Anthropic API client with exponential backoff, jitter, and budget control.

    Args:
        client:  An ``anthropic.Anthropic`` instance (or mock with a
                 ``.messages.create(**kwargs)`` interface).  When *None*,
                 a real client is constructed from ``ANTHROPIC_API_KEY``.
        config:  Retry parameters.  Defaults to :class:`RetryConfig`.

    Example::

        client = RetryClient()
        msg = client.complete("Explain backoff in one sentence.")
        print(msg.content[0].text)
    """

    def __init__(
        self,
        client: Any = None,
        config: Optional[RetryConfig] = None,
    ) -> None:
        _load_dotenv()
        self._client = client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._config = config or RetryConfig()
        self._budget = _BudgetWindow(
            window_seconds=self._config.budget_window_seconds,
            max_retries=self._config.budget_max_retries,
        )
        self._log: list[RetryEvent] = []
        # (request_id, retries_used, succeeded)
        self._records: list[tuple[str, int, bool]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_delay(self, attempt_index: int) -> float:
        """
        Compute sleep duration before re-attempting *attempt_index* (0-based).

        ``raw = base * 2^attempt_index``; optional jitter adds up to 10% on top;
        result is capped at ``max_delay_seconds``.
        """
        raw = self._config.base_delay_seconds * (2 ** attempt_index)
        if self._config.jitter:
            raw += random.uniform(0, raw * 0.10)
        return min(raw, self._config.max_delay_seconds)

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        """Return the HTTP status code from an API exception, or *None*."""
        return getattr(exc, "status_code", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
        request_id: Optional[str] = None,
        _dry_run_delay: bool = False,
    ) -> Any:
        """
        Send a completion request, retrying on transient errors.

        Args:
            prompt:          User message text.
            model:           Anthropic model ID.
            max_tokens:      Max tokens to generate.
            request_id:      Optional correlation ID logged on every retry.
            _dry_run_delay:  Skip ``time.sleep`` (used by demos and tests).

        Returns:
            ``anthropic.types.Message`` on success.

        Raises:
            Exception: Re-raises on non-retriable error or exhausted retries.
        """
        cfg = self._config
        req_id = request_id or f"req-{int(time.monotonic_ns() % 1_000_000):06d}"
        retries_used = 0

        for attempt in range(cfg.max_retries + 1):
            try:
                result = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._records.append((req_id, retries_used, True))
                return result

            except Exception as exc:
                code = self._status_code(exc)
                is_conn = isinstance(exc, anthropic.APIConnectionError)

                # Non-retriable status code: fail immediately
                if code is not None and code in _NON_RETRIABLE_CODES:
                    self._records.append((req_id, retries_used, False))
                    raise

                # Unknown exception type (not an API error): re-raise
                if code is None and not is_conn:
                    self._records.append((req_id, retries_used, False))
                    raise

                # Retries exhausted
                if attempt >= cfg.max_retries:
                    self._records.append((req_id, retries_used, False))
                    raise

                # Budget check
                if not self._budget.can_retry():
                    print(
                        f"  [budget] BLOCKED -- "
                        f"{self._budget.used}/{self._budget._max} retries used "
                        f"in the last {self._budget._window:.0f}s window. "
                        f"Failing fast to protect downstream."
                    )
                    self._records.append((req_id, retries_used, False))
                    raise

                delay = self._compute_delay(attempt)
                self._budget.record()
                retries_used += 1

                event = RetryEvent(
                    request_id=req_id,
                    attempt_number=attempt + 1,
                    error_code=code if code is not None else -1,
                    error_message=str(exc),
                    delay_applied_seconds=delay,
                    timestamp=datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )[:23] + "Z",
                )
                self._log.append(event)

                code_str = str(code) if code is not None else "conn"
                print(
                    f"  [retry] id={req_id}  "
                    f"attempt={attempt + 1}/{cfg.max_retries}  "
                    f"code={code_str}  "
                    f"delay={delay:.3f}s  "
                    f"budget={self._budget.remaining} left"
                )

                if not _dry_run_delay:
                    time.sleep(delay)

        # Unreachable, but satisfies static analysis
        raise RuntimeError("Unexpected exit from retry loop")

    def get_retry_stats(self) -> RetryStats:
        """
        Return aggregate statistics accumulated since the client was created.

        Returns:
            :class:`RetryStats` containing counters, rates, and full log.
        """
        total = len(self._records)
        succeeded = sum(1 for _, _, ok in self._records if ok)
        total_retries = sum(r for _, r, _ in self._records)
        retried = sum(1 for _, r, _ in self._records if r > 0)
        retried_ok = sum(1 for _, r, ok in self._records if r > 0 and ok)
        success_rate = (retried_ok / retried * 100.0) if retried else 0.0
        avg = (total_retries / total) if total else 0.0

        return RetryStats(
            total_requests=total,
            successful_requests=succeeded,
            failed_requests=total - succeeded,
            total_retries=total_retries,
            retried_requests=retried,
            retry_success_rate=success_rate,
            avg_retries_per_request=avg,
            budget_used=self._budget.used,
            budget_remaining=self._budget.remaining,
            retry_log=list(self._log),
        )


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", w: int = _REPORT_WIDTH) -> str:
    return char * w


def _bar(value: float, max_value: float, width: int = 20) -> str:
    filled = round(min(value, max_value) / max_value * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _show_delay_schedule(config: RetryConfig) -> None:
    """Print the theoretical delay for every possible attempt index."""
    print(_sep("="))
    print("  EXPONENTIAL BACKOFF DELAY SCHEDULE")
    print(_sep("="))
    print(
        f"  base={config.base_delay_seconds}s  "
        f"max={config.max_delay_seconds}s  "
        f"jitter={'on' if config.jitter else 'off'}  "
        f"max_retries={config.max_retries}"
    )
    print()
    print(
        f"  {'Attempt':>8}  {'Raw delay':>10}  "
        f"{'Jitter (+max)':>14}  {'Final delay':>12}  Bar"
    )
    print(f"  {_sep('-', _REPORT_WIDTH-2)}")

    # Seed for reproducible display
    rng = random.Random(42)
    for i in range(config.max_retries):
        raw = config.base_delay_seconds * (2 ** i)
        jitter_max = raw * 0.10 if config.jitter else 0.0
        jitter_sample = rng.uniform(0, jitter_max)
        final = min(raw + jitter_sample, config.max_delay_seconds)
        capped = " (capped)" if raw + jitter_sample > config.max_delay_seconds else ""
        bar = _bar(final, config.max_delay_seconds)
        print(
            f"  {'retry '+str(i+1):>8}  "
            f"{raw:>9.2f}s  "
            f"+{jitter_max:>12.3f}s  "
            f"{final:>11.3f}s  "
            f"{bar}{capped}"
        )
    print()


def _run_scenario(
    name: str,
    description: str,
    script: list[Any],
    config: RetryConfig,
    expect_success: bool,
) -> RetryStats:
    """Execute one demo scenario and print a structured result."""
    print(_sep("-"))
    print(f"  SCENARIO: {name}")
    print(f"  {description}")
    print(_sep("-"))

    rc = RetryClient(client=_MockClient(script), config=config)
    outcome = "?"
    try:
        msg = rc.complete(
            prompt="Test prompt",
            request_id=name.lower().replace(" ", "-"),
            _dry_run_delay=True,
        )
        text = msg.content[0].text
        outcome = f"SUCCESS -- \"{text}\""
    except _SimulatedAPIError as exc:
        outcome = f"FAILED   -- {exc}"
    except Exception as exc:
        outcome = f"FAILED   -- {type(exc).__name__}: {exc}"

    stats = rc.get_retry_stats()
    status_icon = "OK  " if expect_success else "FAIL"
    print(f"  [{status_icon}] Outcome : {outcome}")
    print(f"  Retries used   : {stats.total_retries}")
    print(f"  Budget used    : {stats.budget_used} / {config.budget_max_retries}")
    print()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Shared retry config for all scenarios (base_delay > 0 to show real
    # delay values; _dry_run_delay=True ensures no actual sleeping).
    CFG = RetryConfig(
        max_retries=3,
        base_delay_seconds=1.0,
        max_delay_seconds=30.0,
        jitter=True,
        budget_window_seconds=60.0,
        budget_max_retries=8,
    )

    # ------------------------------------------------------------------
    # Section 0: delay schedule table
    # ------------------------------------------------------------------
    _show_delay_schedule(CFG)

    # ------------------------------------------------------------------
    # Section 1: Scenario -- 429 rate limit, recovers on 3rd attempt
    # ------------------------------------------------------------------
    _run_scenario(
        name="Rate Limit Recovery",
        description="429 twice, then success on attempt 3",
        script=[
            _SimulatedAPIError(429),
            _SimulatedAPIError(429),
            # 3rd call succeeds (script exhausted -> mock returns success)
        ],
        config=CFG,
        expect_success=True,
    )

    # ------------------------------------------------------------------
    # Section 2: Scenario -- 500 server error, recovers on 2nd attempt
    # ------------------------------------------------------------------
    _run_scenario(
        name="Server Error Recovery",
        description="500 once, then success on attempt 2",
        script=[
            _SimulatedAPIError(500),
        ],
        config=CFG,
        expect_success=True,
    )

    # ------------------------------------------------------------------
    # Section 3: Scenario -- 400 bad request (non-retriable, no retry)
    # ------------------------------------------------------------------
    _run_scenario(
        name="Non-Retriable Error",
        description="400 Bad Request -- should fail immediately, zero retries",
        script=[
            _SimulatedAPIError(400),
        ],
        config=CFG,
        expect_success=False,
    )

    # ------------------------------------------------------------------
    # Section 4: Scenario -- 503 persists, retries exhausted
    # ------------------------------------------------------------------
    _run_scenario(
        name="Exhausted Retries",
        description="503 on every attempt -- fails after max_retries=3",
        script=[
            _SimulatedAPIError(503),
            _SimulatedAPIError(503),
            _SimulatedAPIError(503),
            _SimulatedAPIError(503),  # 4th = original + 3 retries
        ],
        config=CFG,
        expect_success=False,
    )

    # ------------------------------------------------------------------
    # Section 5: Scenario -- budget protection (tight window budget)
    #
    # Script uses None as a success sentinel between requests so each
    # request gets predictable errors regardless of shared script index.
    #
    # budget_max_retries=2: A uses 1 retry, B uses 1 retry (budget full),
    # C encounters an error and is BLOCKED immediately -- zero retries issued.
    # ------------------------------------------------------------------
    print(_sep("-"))
    print("  SCENARIO: Retry Storm Protection")
    print("  budget_max_retries=2: A and B each use 1 retry (budget full),")
    print("  Request C hits 500 but budget is exhausted -- immediately blocked.")
    print(_sep("-"))

    tight_cfg = RetryConfig(
        max_retries=3,
        base_delay_seconds=1.0,
        max_delay_seconds=30.0,
        jitter=False,
        budget_window_seconds=60.0,
        budget_max_retries=2,
    )
    storm_client = RetryClient(
        client=_MockClient([
            _SimulatedAPIError(429),  # A: error 1 -> retry (budget=1)
            None,                     # A: success
            _SimulatedAPIError(429),  # B: error 1 -> retry (budget=2, full)
            None,                     # B: success
            _SimulatedAPIError(500),  # C: error   -> budget FULL -> BLOCKED
        ]),
        config=tight_cfg,
    )

    for label, rid in [("A", "storm-A"), ("B", "storm-B"), ("C", "storm-C")]:
        try:
            storm_client.complete(
                "Test", request_id=rid, _dry_run_delay=True
            )
            print(f"  [OK  ] Request {label} succeeded")
        except Exception as exc:
            print(f"  [FAIL] Request {label} failed: {exc}")
    print()

    # ------------------------------------------------------------------
    # Section 6: Aggregate stats
    # ------------------------------------------------------------------
    print(_sep("="))
    print("  AGGREGATE RETRY STATISTICS  (storm scenario)")
    print(_sep("="))
    stats = storm_client.get_retry_stats()
    w = 36
    print(f"  {'Total requests':<{w}} {stats.total_requests}")
    print(f"  {'Successful requests':<{w}} {stats.successful_requests}")
    print(f"  {'Failed requests':<{w}} {stats.failed_requests}")
    print(f"  {'Total retries issued':<{w}} {stats.total_retries}")
    print(f"  {'Requests that needed >= 1 retry':<{w}} {stats.retried_requests}")
    print(f"  {'Retry success rate':<{w}} {stats.retry_success_rate:.1f}%")
    print(f"  {'Avg retries per request':<{w}} {stats.avg_retries_per_request:.2f}")
    print(f"  {'Budget used (60s window)':<{w}} {stats.budget_used} / {tight_cfg.budget_max_retries}")
    print(f"  {'Budget remaining':<{w}} {stats.budget_remaining}")

    if stats.retry_log:
        print()
        print(f"  RETRY EVENT LOG  ({len(stats.retry_log)} events)")
        print(_sep("-"))
        print(
            f"  {'Request ID':<14}  {'Attempt':>7}  {'Code':>5}  "
            f"{'Delay':>8}  Timestamp"
        )
        print(f"  {_sep('-', _REPORT_WIDTH-2)}")
        for ev in stats.retry_log:
            code_str = str(ev.error_code) if ev.error_code != -1 else "conn"
            print(
                f"  {ev.request_id:<14}  "
                f"{ev.attempt_number:>7}  "
                f"{code_str:>5}  "
                f"{ev.delay_applied_seconds:>7.3f}s  "
                f"{ev.timestamp}"
            )

    print()
    print(_sep("="))
