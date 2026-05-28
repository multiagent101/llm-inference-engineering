"""
10_reliability/circuit_breaker.py

Circuit breaker pattern for protecting downstream services from cascade
failures.  Three states model the lifecycle of service health:

    CLOSED   -- normal operation; failures are counted
    OPEN     -- all calls blocked immediately; no downstream calls made
    HALF_OPEN -- one probe call allowed; success closes, failure re-opens

Transitions::

    CLOSED  --[failure_threshold consecutive failures]--> OPEN
    OPEN    --[recovery_timeout elapsed]----------------> HALF_OPEN
    HALF_OPEN --[success_threshold successes]-----------> CLOSED
    HALF_OPEN --[any failure]--------------------------> OPEN
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

_REPORT_WIDTH = 72


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(Enum):
    """The three states of a circuit breaker."""
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitBreakerOpenError(Exception):
    """
    Raised when a call is attempted while the circuit is OPEN.

    Attributes:
        circuit_name:  Name of the circuit breaker that blocked the call.
        retry_after:   Estimated seconds until the circuit tries HALF_OPEN.
    """

    def __init__(self, circuit_name: str, retry_after: float) -> None:
        self.circuit_name = circuit_name
        self.retry_after = max(0.0, retry_after)
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. "
            f"Retry in {self.retry_after:.1f}s."
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    """Record of one circuit state change."""

    from_state: State
    to_state: State
    reason: str
    timestamp: str              # ISO-8601 UTC


@dataclass
class CallRecord:
    """Outcome of one call through the circuit."""

    timestamp: str
    state_at_call: State
    succeeded: bool
    blocked: bool
    latency_ms: float           # 0.0 for blocked calls
    error_type: Optional[str]   # None on success or block


# ---------------------------------------------------------------------------
# Mock clock (for tests and demo — no real sleeping required)
# ---------------------------------------------------------------------------

class _MockClock:
    """
    Manually-advanced monotonic clock substitute.

    Pass an instance as ``_now`` when constructing a :class:`CircuitBreaker`
    to control perceived time in demos and unit tests.
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        """Move the clock forward by *seconds*."""
        self._t += seconds


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Thread-safe circuit breaker wrapping arbitrary callables.

    Args:
        failure_threshold:         Consecutive failures before opening.
        recovery_timeout_seconds:  Seconds in OPEN before probing (HALF_OPEN).
        success_threshold:         Consecutive successes in HALF_OPEN to close.
        name:                      Human-readable identifier for logging.
        _now:                      Clock callable (default: ``time.monotonic``).
                                   Inject a :class:`_MockClock` for testing.

    Example::

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30)

        try:
            result = cb.call(requests.get, "https://api.example.com/v1/data")
        except CircuitBreakerOpenError as e:
            # Serve from cache, return degraded response, etc.
            result = fallback()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        success_threshold: int = 2,
        name: str = "default",
        _now: Optional[Callable[[], float]] = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._success_threshold = success_threshold
        self._name = name
        self._now = _now or time.monotonic

        # --- state machine ---
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._open_since: Optional[float] = None   # monotonic timestamp

        # --- statistics ---
        self._total_calls = 0           # ALL calls, including blocked
        self._successful_calls = 0
        self._failed_calls = 0
        self._blocked_calls = 0
        self._accumulated_open_time: float = 0.0
        self._state_history: list[StateTransition] = []
        self._call_log: list[CallRecord] = []

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Current circuit state (checks recovery timeout on each access)."""
        with self._lock:
            self._check_recovery_timeout()
            return self._state

    @property
    def state_history(self) -> list[StateTransition]:
        """Chronological list of all state transitions."""
        with self._lock:
            return list(self._state_history)

    @property
    def call_log(self) -> list[CallRecord]:
        """Chronological list of all call records."""
        with self._lock:
            return list(self._call_log)

    @property
    def time_in_open_state_seconds(self) -> float:
        """Total seconds the circuit has spent in OPEN state."""
        with self._lock:
            total = self._accumulated_open_time
            if self._state == State.OPEN and self._open_since is not None:
                total += self._now() - self._open_since
            return total

    @property
    def failure_rate(self) -> float:
        """Fraction of non-blocked calls that failed (0.0 – 1.0)."""
        with self._lock:
            attempted = self._successful_calls + self._failed_calls
            return (self._failed_calls / attempted) if attempted else 0.0

    @property
    def calls_blocked(self) -> int:
        """Number of calls rejected without reaching the downstream service."""
        with self._lock:
            return self._blocked_calls

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _check_recovery_timeout(self) -> None:
        """Silently transition OPEN -> HALF_OPEN when the timeout has passed."""
        if (
            self._state == State.OPEN
            and self._open_since is not None
            and self._now() - self._open_since >= self._recovery_timeout
        ):
            self._transition(State.HALF_OPEN, "Recovery timeout elapsed")

    def _transition(self, new_state: State, reason: str) -> None:
        """Record and apply a state change."""
        old_state = self._state

        # Accumulate time spent in OPEN before leaving it
        if old_state == State.OPEN and self._open_since is not None:
            self._accumulated_open_time += self._now() - self._open_since
            self._open_since = None

        if new_state == State.OPEN:
            self._open_since = self._now()

        self._state = new_state
        self._consecutive_failures = 0
        self._consecutive_successes = 0

        self._state_history.append(StateTransition(
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ) + "Z",
        ))

    def _record_success(self, call_state: State, latency_ms: float) -> None:
        """Update counters and possibly transition after a successful call."""
        self._successful_calls += 1
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        self._call_log.append(CallRecord(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S") + "Z",
            state_at_call=call_state,
            succeeded=True,
            blocked=False,
            latency_ms=latency_ms,
            error_type=None,
        ))
        if call_state == State.HALF_OPEN:
            if self._consecutive_successes >= self._success_threshold:
                self._transition(
                    State.CLOSED,
                    f"{self._success_threshold} consecutive successes in HALF_OPEN",
                )

    def _record_failure(
        self,
        call_state: State,
        latency_ms: float,
        error_type: str,
    ) -> None:
        """Update counters and possibly transition after a failed call."""
        self._failed_calls += 1
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        self._call_log.append(CallRecord(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S") + "Z",
            state_at_call=call_state,
            succeeded=False,
            blocked=False,
            latency_ms=latency_ms,
            error_type=error_type,
        ))
        if call_state == State.HALF_OPEN:
            self._transition(State.OPEN, "Failure during HALF_OPEN probe")
        elif call_state == State.CLOSED:
            if self._consecutive_failures >= self._failure_threshold:
                self._transition(
                    State.OPEN,
                    f"{self._failure_threshold} consecutive failures in CLOSED",
                )

    # ------------------------------------------------------------------
    # Public call interface
    # ------------------------------------------------------------------

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute *func* while respecting the current circuit state.

        - **CLOSED / HALF_OPEN**: calls *func*; tracks success/failure.
        - **OPEN**: raises :class:`CircuitBreakerOpenError` immediately.

        Args:
            func:    Callable to protect.
            *args:   Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            CircuitBreakerOpenError: Circuit is OPEN; call was blocked.
            Exception: Propagates any exception raised by *func*.
        """
        with self._lock:
            self._check_recovery_timeout()
            current_state = self._state
            self._total_calls += 1

            if current_state == State.OPEN:
                self._blocked_calls += 1
                elapsed = (
                    self._now() - self._open_since
                    if self._open_since is not None
                    else self._recovery_timeout
                )
                retry_after = self._recovery_timeout - elapsed
                self._call_log.append(CallRecord(
                    timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S") + "Z",
                    state_at_call=State.OPEN,
                    succeeded=False,
                    blocked=True,
                    latency_ms=0.0,
                    error_type="CircuitBreakerOpenError",
                ))
                raise CircuitBreakerOpenError(self._name, retry_after)

        # Execute the wrapped function outside the lock (may be slow I/O)
        t0 = self._now()
        try:
            result = func(*args, **kwargs)
            latency_ms = (self._now() - t0) * 1000.0
            with self._lock:
                self._record_success(current_state, latency_ms)
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            latency_ms = (self._now() - t0) * 1000.0
            with self._lock:
                self._record_failure(current_state, latency_ms, type(exc).__name__)
            raise

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    def get_health_report(self) -> str:
        """
        Return a multi-section plain-text health report.

        Includes current state, call statistics, open-time accounting,
        state-transition history, and actionable recommendations.
        """
        with self._lock:
            self._check_recovery_timeout()
            state = self._state
            cf = self._consecutive_failures
            cs = self._consecutive_successes
            total = self._total_calls
            succeeded = self._successful_calls
            failed = self._failed_calls
            blocked = self._blocked_calls
            fr = self.failure_rate
            open_time = self.time_in_open_state_seconds
            history = list(self._state_history)

        lines: list[str] = []
        sep = "=" * _REPORT_WIDTH
        thin = "-" * _REPORT_WIDTH

        def h(title: str) -> None:
            lines.append("")
            lines.append(f"  {title}")
            lines.append(thin)

        def kv(label: str, value: str, w: int = 38) -> None:
            lines.append(f"  {label:<{w}} {value}")

        lines.append(sep)
        lines.append(f"  CIRCUIT BREAKER HEALTH REPORT -- '{self._name}'")
        lines.append(sep)

        # -- Current state --
        h("CURRENT STATE")
        kv("State", state.name)
        kv("Consecutive failures", str(cf))
        kv("Consecutive successes", str(cs))
        kv("Failure threshold", str(self._failure_threshold))
        kv("Success threshold (HALF_OPEN)", str(self._success_threshold))
        kv("Recovery timeout", f"{self._recovery_timeout:.0f}s")

        if state == State.OPEN and self._open_since is not None:
            elapsed = self._now() - self._open_since
            remaining = max(0.0, self._recovery_timeout - elapsed)
            kv("Time in current OPEN window", f"{elapsed:.1f}s")
            kv("Time until HALF_OPEN", f"{remaining:.1f}s")

        # -- Call statistics --
        h("CALL STATISTICS")
        kv("Total calls", str(total))
        kv("Successful", str(succeeded))
        kv("Failed (reached service)", str(failed))
        kv("Blocked (circuit OPEN)", str(blocked))
        kv("Failure rate (excl. blocked)", f"{fr * 100:.1f}%")
        kv("Block rate", f"{blocked / total * 100:.1f}%" if total else "0.0%")
        kv("Total time in OPEN state", f"{open_time:.1f}s")

        # -- Transition history --
        h(f"STATE TRANSITION HISTORY  ({len(history)} transitions)")
        if history:
            for t in history:
                lines.append(
                    f"  {t.timestamp}  "
                    f"{t.from_state.name:<10} -> {t.to_state.name:<10}  "
                    f"{t.reason}"
                )
        else:
            lines.append("  No transitions yet.")

        # -- Recommendations --
        h("RECOMMENDATIONS")
        if state == State.CLOSED:
            if fr >= 0.50:
                lines.append("  WARNING: failure rate above 50%.")
                lines.append("  Investigate upstream service health immediately.")
                lines.append("  Consider reducing failure_threshold to trip faster.")
            elif fr >= 0.20:
                lines.append("  CAUTION: elevated failure rate.")
                lines.append("  Monitor upstream service; circuit may trip soon.")
            else:
                lines.append("  Circuit is healthy. No action required.")
        elif state == State.OPEN:
            lines.append("  Circuit is OPEN. Downstream calls are blocked.")
            lines.append("  Actions:")
            lines.append("    1. Check upstream service health dashboard.")
            lines.append("    2. Serve stale cache or degraded fallback to clients.")
            lines.append(
                f"    3. Expect HALF_OPEN probe in "
                f"{max(0.0, self._recovery_timeout - (self._now() - (self._open_since or self._now()))):.0f}s."
            )
        else:  # HALF_OPEN
            needed = self._success_threshold - cs
            lines.append(
                f"  Circuit is probing. "
                f"{needed} more success{'es' if needed != 1 else ''} needed to close."
            )
            lines.append("  Avoid high-traffic routing until circuit closes.")

        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _sep(char: str = "=", w: int = _REPORT_WIDTH) -> str:
    return char * w


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Simulated scenario
    # ------------------------------------------------------------------
    # Mock clock: each call advances time by TICK seconds.
    # Service recovery time: simulates ~2 minutes of outage.
    #
    # Config chosen for clear demo (not production defaults):
    #   failure_threshold=3  -- trip quickly so OPEN appears early
    #   recovery_timeout=30s -- short probe window (represents 30 min real)
    #   success_threshold=2  -- two clean probes to restore confidence
    #
    # Timeline (mock seconds):
    #   t=5-15:   Calls 1-3 fail; CLOSED -> OPEN at t=15
    #   t=20-40:  Calls 4-8 blocked (OPEN)
    #   t=45:     Timeout -> HALF_OPEN; probe fails -> OPEN
    #   t=50-70:  Calls 10-14 blocked (OPEN again)
    #   t=75:     Timeout -> HALF_OPEN; probe fails -> OPEN
    #   t=80-100: Calls 16-20 blocked (OPEN again)
    #   t=105:    Timeout -> HALF_OPEN; SERVICE RECOVERED; probe succeeds
    #   t=110:    Second probe succeeds -> HALF_OPEN -> CLOSED
    #   t=115+:   Calls succeed normally (CLOSED)
    # ------------------------------------------------------------------

    TICK = 5.0                  # simulated seconds per call
    SERVICE_RECOVERY_TIME = 105.0  # service becomes healthy at this mock time

    clock = _MockClock()

    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        success_threshold=2,
        name="payment-service",
        _now=clock,
    )

    def payment_service() -> str:
        """Simulated downstream service -- fails until SERVICE_RECOVERY_TIME."""
        if clock() < SERVICE_RECOVERY_TIME:
            raise ConnectionError("Payment gateway unavailable")
        return "payment_ok"

    print(_sep("="))
    print("  CIRCUIT BREAKER DEMO -- 2-minute service outage (mock clock)")
    print(f"  Tick: {TICK:.0f}s/call  |  Service recovers at t={SERVICE_RECOVERY_TIME:.0f}s (mock)")
    print(f"  failure_threshold={cb._failure_threshold}  |  "
          f"recovery_timeout={cb._recovery_timeout:.0f}s  |  "
          f"success_threshold={cb._success_threshold}")
    print(_sep("="))
    print(
        f"  {'#':>4}  {'t(s)':>5}  {'Before':>10}  "
        f"{'Result':<38}  {'After':>10}"
    )
    print(_sep("-"))

    prev_history_len = 0

    for call_num in range(1, 26):
        clock.advance(TICK)
        sim_time = clock()

        state_before = cb.state  # triggers timeout check

        result_str = ""
        try:
            cb.call(payment_service)
            result_str = "OK: payment_ok"
        except CircuitBreakerOpenError as exc:
            result_str = f"BLOCKED (retry in {exc.retry_after:.1f}s)"
        except ConnectionError:
            result_str = "FAIL: ConnectionError"

        state_after = cb.state

        print(
            f"  {call_num:>4}  {sim_time:>4.0f}s  "
            f"{state_before.name:>10}  "
            f"{result_str:<38}  "
            f"{state_after.name:>10}"
        )

        # Print any transitions that occurred during this call
        current_history = cb.state_history
        for t in current_history[prev_history_len:]:
            print(
                f"         {'':>5}  {'':>10}  "
                f"  *** {t.from_state.name} -> {t.to_state.name}: {t.reason}"
            )
        prev_history_len = len(current_history)

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------
    print()
    print(cb.get_health_report())
