"""
10_reliability/provider_failover.py

Automatic LLM provider failover with periodic health monitoring.

Providers are tried in priority order (lowest number = highest priority).
After a configurable number of consecutive in-call failures, a provider is
temporarily marked unhealthy and skipped until the background health-check
thread restores it.

Designed for multi-provider setups such as:
  - Anthropic (primary)  +  Azure OpenAI (secondary)
  - Any primary          +  local/mock  (fallback / testing)
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import anthropic

_REPORT_WIDTH = 74

# Tiers map to provider-specific model IDs inside each Provider config.
MODEL_TIERS = ("fast", "quality", "powerful")


# ---------------------------------------------------------------------------
# .env loader (stdlib only)
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """Parse KEY=VALUE pairs from a .env file and set ``os.environ``."""
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    """
    Configuration for one LLM provider.

    Args:
        name:                Human-readable identifier (e.g. "anthropic").
        api_key_env_var:     Environment variable that holds the API key.
        base_url:            API base URL (used for OpenAI-compatible providers).
        available_models:    Mapping from tier name to provider model ID.
                             Required tiers: "fast", "quality", "powerful".
        health_check_endpoint: URL polled by the health-check thread.
        priority:            Call order — lower number is tried first.
        timeout_seconds:     Per-call timeout.
        complete_override:   If set, replaces the real API call.  Signature:
                             ``(prompt: str, model_id: str) -> str``.
                             Used for mocks and tests.
        health_override:     If set, replaces the HTTP health check.  Signature:
                             ``() -> bool``.
    """

    name: str
    api_key_env_var: str
    base_url: str
    available_models: dict[str, str]
    health_check_endpoint: str
    priority: int
    timeout_seconds: float = 30.0
    complete_override: Optional[Callable[[str, str], str]] = field(
        default=None, repr=False
    )
    health_override: Optional[Callable[[], bool]] = field(
        default=None, repr=False
    )


# ---------------------------------------------------------------------------
# Runtime status and event dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProviderStatus:
    """Mutable runtime health and performance state for one provider."""

    provider: Provider
    is_healthy: bool = True
    last_health_check_ts: Optional[float] = None    # time.monotonic()
    last_health_check_wall: Optional[str] = None    # human-readable
    consecutive_call_failures: int = 0
    last_error: Optional[str] = None
    total_requests: int = 0
    successful_requests: int = 0
    total_latency_ms: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Fraction of requests that failed (0.0 – 1.0)."""
        return (
            (self.total_requests - self.successful_requests) / self.total_requests
            if self.total_requests
            else 0.0
        )

    @property
    def avg_latency_ms(self) -> float:
        """Average latency of successful requests."""
        return (
            self.total_latency_ms / self.successful_requests
            if self.successful_requests
            else 0.0
        )


@dataclass
class FailoverEvent:
    """Record of one failover incident."""

    timestamp: str
    reason: str
    original_provider: str    # highest-priority provider that should have handled it
    provider_used: str        # provider that actually handled it
    latency_impact_ms: float  # extra latency wasted on failed providers
    prompt_preview: str       # first 60 chars of the prompt


@dataclass
class FailoverResult:
    """Return value of :meth:`ProviderFailover.complete`."""

    response: Optional[str]
    provider_used: str
    providers_tried: list[str]
    failover_occurred: bool
    total_latency_ms: float
    per_provider_latency_ms: dict[str, float]
    success: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# ProviderFailover
# ---------------------------------------------------------------------------

class ProviderFailover:
    """
    Priority-ordered LLM provider pool with automatic failover.

    On each :meth:`complete` call, providers are tried in priority order.
    A provider that raises an exception is counted as failing; after
    ``unhealthy_threshold`` consecutive failures it is skipped on future
    calls until the health-check thread restores it.

    A background daemon thread runs periodic health checks every
    ``health_check_interval`` seconds.  Inject :class:`Provider` instances
    with a ``health_override`` callable to control health checks in tests.

    Args:
        providers:               Ordered (or unordered — sorted internally)
                                 list of :class:`Provider` configs.
        health_check_interval:   Seconds between background health sweeps.
        unhealthy_threshold:     Consecutive in-call failures before a
                                 provider is marked unhealthy.

    Example::

        failover = ProviderFailover([anthropic_provider, fallback_provider])
        result = failover.complete("Summarise this document.", model_tier="quality")
        if result.failover_occurred:
            print("Used fallback:", result.provider_used)
    """

    def __init__(
        self,
        providers: list[Provider],
        health_check_interval: float = 60.0,
        unhealthy_threshold: int = 3,
    ) -> None:
        _load_dotenv()
        self._providers = sorted(providers, key=lambda p: p.priority)
        self._health_interval = health_check_interval
        self._unhealthy_threshold = unhealthy_threshold
        self._status: dict[str, ProviderStatus] = {
            p.name: ProviderStatus(provider=p) for p in self._providers
        }
        self._failover_log: list[FailoverEvent] = []
        self._anthropic_clients: dict[str, anthropic.Anthropic] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="provider-health"
        )
        self._health_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        model_tier: str = "fast",
        max_tokens: int = 512,
    ) -> FailoverResult:
        """
        Send *prompt* to the highest-priority healthy provider.

        Falls over to the next provider on any exception.

        Args:
            prompt:     User message text.
            model_tier: One of ``"fast"``, ``"quality"``, ``"powerful"``.
            max_tokens: Max tokens to generate.

        Returns:
            :class:`FailoverResult` — always returned, even if all providers fail.
        """
        tried: list[str] = []
        per_latency: dict[str, float] = {}
        primary_name = self._providers[0].name  # highest-priority regardless of health
        t_total = time.monotonic()

        with self._lock:
            ordered_names = [p.name for p in self._providers]

        for name in ordered_names:
            with self._lock:
                status = self._status[name]
                if not status.is_healthy:
                    continue
                provider = status.provider

            model_id = provider.available_models.get(model_tier)
            if not model_id:
                continue

            t0 = time.monotonic()
            try:
                response = self._call_provider(provider, prompt, model_id, max_tokens)
                latency = (time.monotonic() - t0) * 1000.0
                per_latency[name] = latency
                tried.append(name)

                with self._lock:
                    status.total_requests += 1
                    status.successful_requests += 1
                    status.consecutive_call_failures = 0
                    status.total_latency_ms += latency

                failover_occurred = name != primary_name
                if failover_occurred:
                    wasted = sum(v for k, v in per_latency.items() if k != name)
                    self._record_failover(
                        reason=f"Provider '{primary_name}' unavailable",
                        original=primary_name,
                        used=name,
                        impact_ms=wasted,
                        prompt=prompt,
                    )

                return FailoverResult(
                    response=response,
                    provider_used=name,
                    providers_tried=tried,
                    failover_occurred=failover_occurred,
                    total_latency_ms=(time.monotonic() - t_total) * 1000.0,
                    per_provider_latency_ms=per_latency,
                    success=True,
                    error=None,
                )

            except Exception as exc:
                latency = (time.monotonic() - t0) * 1000.0
                per_latency[name] = latency
                tried.append(name)
                with self._lock:
                    status.total_requests += 1
                    status.consecutive_call_failures += 1
                    status.last_error = str(exc)
                    if status.consecutive_call_failures >= self._unhealthy_threshold:
                        if status.is_healthy:
                            status.is_healthy = False
                            self._log(
                                f"[health] '{name}' auto-marked UNHEALTHY "
                                f"after {self._unhealthy_threshold} consecutive failures."
                            )

        # All providers exhausted
        return FailoverResult(
            response=None,
            provider_used="none",
            providers_tried=tried,
            failover_occurred=len(tried) > 1,
            total_latency_ms=(time.monotonic() - t_total) * 1000.0,
            per_provider_latency_ms=per_latency,
            success=False,
            error=f"All providers failed or unhealthy: {tried}",
        )

    def run_health_checks_now(self) -> None:
        """Trigger a synchronous health check sweep (useful for testing/demos)."""
        for p in self._providers:
            self._check_one(p)

    def mark_unhealthy(self, provider_name: str, reason: str = "manual") -> None:
        """Manually mark a provider as unhealthy."""
        with self._lock:
            if provider_name in self._status:
                self._status[provider_name].is_healthy = False
                self._status[provider_name].last_error = reason
                self._log(f"[manual] '{provider_name}' marked UNHEALTHY: {reason}")

    def mark_healthy(self, provider_name: str) -> None:
        """Manually restore a provider to healthy status."""
        with self._lock:
            if provider_name in self._status:
                self._status[provider_name].is_healthy = True
                self._status[provider_name].consecutive_call_failures = 0
                self._log(f"[manual] '{provider_name}' restored to HEALTHY.")

    def get_failover_report(self) -> str:
        """
        Return a plain-text summary of provider health and failover history.
        """
        with self._lock:
            statuses = list(self._status.values())
            log = list(self._failover_log)

        lines: list[str] = []
        sep = "=" * _REPORT_WIDTH
        thin = "-" * _REPORT_WIDTH

        def h(title: str) -> None:
            lines.append("")
            lines.append(f"  {title}")
            lines.append(thin)

        def kv(label: str, value: str, w: int = 36) -> None:
            lines.append(f"  {label:<{w}} {value}")

        lines.append(sep)
        lines.append("  PROVIDER FAILOVER REPORT")
        lines.append(sep)

        h("PROVIDER STATUS SUMMARY")
        lines.append(
            f"  {'Provider':<18}  {'Pri':>3}  {'Health':>9}  "
            f"{'Requests':>9}  {'Fail%':>6}  {'AvgMs':>7}  Last Error"
        )
        lines.append(f"  {thin}")
        for st in sorted(statuses, key=lambda s: s.provider.priority):
            p = st.provider
            health = "HEALTHY" if st.is_healthy else "UNHEALTHY"
            fail_pct = f"{st.failure_rate * 100:.0f}%"
            avg_ms = f"{st.avg_latency_ms:.0f}" if st.successful_requests else "-"
            err = (st.last_error or "-")[:28]
            lines.append(
                f"  {p.name:<18}  {p.priority:>3}  {health:>9}  "
                f"{st.total_requests:>9}  {fail_pct:>6}  {avg_ms:>7}  {err}"
            )

        h(f"FAILOVER EVENT LOG  ({len(log)} events)")
        if log:
            for ev in log:
                lines.append(f"  {ev.timestamp}")
                lines.append(f"    Reason           : {ev.reason}")
                lines.append(f"    Original provider: {ev.original_provider}")
                lines.append(f"    Provider used    : {ev.provider_used}")
                lines.append(f"    Latency impact   : +{ev.latency_impact_ms:.1f}ms")
                lines.append(f"    Prompt preview   : \"{ev.prompt_preview}\"")
                lines.append("")
        else:
            lines.append("  No failover events recorded.")

        lines.append(sep)
        return "\n".join(lines)

    def stop(self) -> None:
        """Stop the background health-check thread."""
        self._stop.set()
        self._health_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_provider(
        self,
        provider: Provider,
        prompt: str,
        model_id: str,
        max_tokens: int,
    ) -> str:
        """Invoke one provider, returning the response text."""
        if provider.complete_override is not None:
            return provider.complete_override(prompt, model_id)

        client = self._get_anthropic_client(provider)
        msg = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _get_anthropic_client(self, provider: Provider) -> anthropic.Anthropic:
        """Return (cached) Anthropic client for *provider*."""
        if provider.name not in self._anthropic_clients:
            api_key = os.environ.get(provider.api_key_env_var, "")
            self._anthropic_clients[provider.name] = anthropic.Anthropic(
                api_key=api_key
            )
        return self._anthropic_clients[provider.name]

    def _check_one(self, provider: Provider) -> None:
        """Run health check for a single provider and update its status."""
        try:
            if provider.health_override is not None:
                healthy = provider.health_override()
            elif provider.health_check_endpoint:
                with urllib.request.urlopen(
                    provider.health_check_endpoint, timeout=5
                ) as resp:
                    healthy = 200 <= resp.status < 300
            else:
                healthy = True  # no endpoint configured: assume healthy
        except Exception as exc:
            healthy = False
            with self._lock:
                self._status[provider.name].last_error = str(exc)

        wall = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        with self._lock:
            st = self._status[provider.name]
            prev = st.is_healthy
            st.is_healthy = healthy
            st.last_health_check_ts = time.monotonic()
            st.last_health_check_wall = wall
            if healthy:
                st.consecutive_call_failures = 0
            if prev != healthy:
                arrow = "HEALTHY" if healthy else "UNHEALTHY"
                self._log(
                    f"[health] '{provider.name}' is now {arrow} "
                    f"(health check at {wall})"
                )

    def _health_loop(self) -> None:
        """Background daemon: sweep all providers every *_health_interval* s."""
        while not self._stop.wait(self._health_interval):
            for p in self._providers:
                self._check_one(p)

    def _record_failover(
        self,
        reason: str,
        original: str,
        used: str,
        impact_ms: float,
        prompt: str,
    ) -> None:
        ev = FailoverEvent(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            reason=reason,
            original_provider=original,
            provider_used=used,
            latency_impact_ms=impact_ms,
            prompt_preview=prompt[:60],
        )
        with self._lock:
            self._failover_log.append(ev)
        self._log(
            f"[failover] {original} -> {used}  "
            f"reason='{reason}'  impact=+{impact_ms:.0f}ms"
        )

    @staticmethod
    def _log(message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  {ts}  {message}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _sep(char: str = "=", w: int = _REPORT_WIDTH) -> str:
    return char * w


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # State shared by the mock overrides
    # ------------------------------------------------------------------
    _outage_active = True   # flip to False to simulate recovery

    # ------------------------------------------------------------------
    # Provider definitions
    # ------------------------------------------------------------------

    def _anthropic_complete(prompt: str, model_id: str) -> str:
        """Simulates Anthropic: raises during outage, otherwise calls real API."""
        if _outage_active:
            raise ConnectionError(
                "Anthropic API unreachable (simulated network outage)"
            )
        # Real call — uses actual API key from .env
        _load_dotenv()
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        msg = client.messages.create(
            model=model_id,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _anthropic_health() -> bool:
        return not _outage_active

    anthropic_provider = Provider(
        name="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        available_models={
            "fast":     "claude-haiku-4-5-20251001",
            "quality":  "claude-sonnet-4-6-20241022",
            "powerful": "claude-opus-4-7",
        },
        health_check_endpoint="https://status.anthropic.com",
        priority=1,
        complete_override=_anthropic_complete,
        health_override=_anthropic_health,
    )

    _mock_call_count = 0

    def _mock_complete(prompt: str, model_id: str) -> str:
        global _mock_call_count
        _mock_call_count += 1
        return (
            f"[MockLLM/{model_id}] Response #{_mock_call_count}: "
            f"{prompt[:40]}..."
        )

    mock_provider = Provider(
        name="mock-llm",
        api_key_env_var="MOCK_API_KEY",
        base_url="http://localhost:11434/v1",
        available_models={
            "fast":     "mock-haiku",
            "quality":  "mock-sonnet",
            "powerful": "mock-opus",
        },
        health_check_endpoint="http://localhost:11434/health",
        priority=2,
        complete_override=_mock_complete,
        health_override=lambda: True,
    )

    # ------------------------------------------------------------------
    # Build failover client (fast health checks for demo visibility)
    # ------------------------------------------------------------------
    failover = ProviderFailover(
        providers=[anthropic_provider, mock_provider],
        health_check_interval=2.0,   # 2s for demo; use 60s in production
        unhealthy_threshold=2,
    )

    PROMPTS = [
        "What is exponential backoff?",
        "Explain circuit breakers in one sentence.",
        "What is a retry storm?",
        "Define provider failover.",
        "What is mean time to recovery?",
    ]

    # ------------------------------------------------------------------
    # Phase 1: Primary outage — all calls fail over to mock
    # ------------------------------------------------------------------
    print(_sep("="))
    print("  PROVIDER FAILOVER DEMO")
    print(_sep("="))
    print("  Phase 1: Anthropic outage active -- expecting failover to mock-llm")
    print(_sep("-"))

    for i, prompt in enumerate(PROMPTS[:3], start=1):
        result = failover.complete(prompt, model_tier="fast")
        icon = "OK  " if result.success else "FAIL"
        fo = " [FAILOVER]" if result.failover_occurred else ""
        print(
            f"  [{icon}] call={i}  used={result.provider_used:<12}"
            f"  tried={result.providers_tried}"
            f"  {result.total_latency_ms:.1f}ms{fo}"
        )
        if result.response:
            print(f"         response: {result.response[:70]}")
        print()

    # ------------------------------------------------------------------
    # Phase 2: Health check sweep — marks anthropic unhealthy
    # ------------------------------------------------------------------
    print(_sep("-"))
    print("  Phase 2: Triggering health check sweep...")
    print(_sep("-"))
    failover.run_health_checks_now()
    time.sleep(0.1)  # let print buffer flush
    print()

    # ------------------------------------------------------------------
    # Phase 3: Still in outage -- provider already unhealthy, skipped fast
    # ------------------------------------------------------------------
    print(_sep("-"))
    print("  Phase 3: Anthropic still unhealthy -- routed directly to mock-llm")
    print(_sep("-"))

    result = failover.complete(PROMPTS[3], model_tier="quality")
    icon = "OK  " if result.success else "FAIL"
    fo = " [FAILOVER]" if result.failover_occurred else ""
    print(
        f"  [{icon}] call=4  used={result.provider_used:<12}"
        f"  tried={result.providers_tried}"
        f"  {result.total_latency_ms:.1f}ms{fo}"
    )
    if result.response:
        print(f"         response: {result.response[:70]}")
    print()

    # ------------------------------------------------------------------
    # Phase 4: Simulate recovery -- primary comes back
    # ------------------------------------------------------------------
    print(_sep("-"))
    print("  Phase 4: Anthropic recovers -- running health check...")
    print(_sep("-"))
    _outage_active = False
    failover.run_health_checks_now()
    time.sleep(0.1)
    print()

    result = failover.complete(PROMPTS[4], model_tier="fast")
    icon = "OK  " if result.success else "FAIL"
    fo = " [FAILOVER]" if result.failover_occurred else ""
    print(
        f"  [{icon}] call=5  used={result.provider_used:<12}"
        f"  tried={result.providers_tried}"
        f"  {result.total_latency_ms:.1f}ms{fo}"
    )
    if result.response:
        print(f"         response: {result.response[:70]}")
    print()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    failover.stop()
    print(failover.get_failover_report())
