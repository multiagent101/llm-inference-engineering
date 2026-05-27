#!/usr/bin/env python3
"""Exact-match LLM response cache with TTL, persistence, and cost analytics."""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Pricing — used only in benchmark() for cost projections
# ---------------------------------------------------------------------------

_INPUT_PRICE: dict[str, float] = {
    "claude-haiku-4-5":  0.80e-6,
    "claude-sonnet-4-5": 3.00e-6,
    "claude-opus-4":     15.00e-6,
    "gpt-4o-mini":       0.15e-6,
    "gpt-4o":            2.50e-6,
}
_OUTPUT_PRICE: dict[str, float] = {
    "claude-haiku-4-5":  4.00e-6,
    "claude-sonnet-4-5": 15.00e-6,
    "claude-opus-4":     75.00e-6,
    "gpt-4o-mini":       0.60e-6,
    "gpt-4o":            10.00e-6,
}
_CHARS_PER_TOKEN = 4  # rough approximation for cost estimation without tiktoken


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached response record."""

    key: str            # SHA-256(prompt | model)
    prompt_hash: str    # SHA-256(prompt) — enables prompt-only invalidation
    prompt_preview: str # first 100 chars of prompt
    model: str
    response: str
    created_at: str     # ISO-8601 UTC
    expires_at: str     # ISO-8601 UTC
    hit_count: int
    input_tokens: int   # actual or estimated
    output_tokens: int  # actual or estimated


@dataclass
class CacheStats:
    """Aggregated runtime statistics for one cache instance."""

    total_hits: int
    total_misses: int
    total_requests: int
    hit_rate_pct: float
    entries_count: int
    expired_entries: int
    memory_used_mb: float


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------

class ExactMatchCache:
    """
    Exact-match response cache for LLM API calls.

    Architecture
    ------------
    Primary storage is an in-memory ``dict``; the same data is persisted to
    a JSON file on every ``set`` call (configurable via ``auto_persist``).
    On instantiation, expired entries are purged immediately after loading.

    Cache key
    ---------
    ``SHA-256(prompt + "|" + model)`` — identical prompts on different models
    are stored independently.

    TTL
    ---
    Each entry carries an ``expires_at`` timestamp.  Expired entries are
    excluded by ``get``, removed by ``clear_expired``, and purged on ``_load``.
    """

    def __init__(
        self,
        persist_path: str = "exact_cache.json",
        default_ttl: int = 3600,
        auto_persist: bool = True,
    ) -> None:
        """
        Args:
            persist_path: Path to the JSON persistence file.
            default_ttl: Default TTL in seconds for new entries (default 1 h).
            auto_persist: If True, write to disk after every ``set`` / remove.
        """
        self.persist_path  = Path(persist_path)
        self.default_ttl   = default_ttl
        self.auto_persist  = auto_persist
        self._store: dict[str, dict] = {}
        self._total_hits   = 0
        self._total_misses = 0
        self._load()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(prompt: str, model: str) -> str:
        """Cache key: SHA-256(prompt|model)."""
        return hashlib.sha256(f"{prompt}|{model}".encode()).hexdigest()

    @staticmethod
    def _phash(prompt: str) -> str:
        """Prompt-only hash for cross-model invalidation."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _dt(s: str) -> datetime:
        return datetime.fromisoformat(s)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, prompt: str, model: str) -> Optional[str]:
        """
        Return the cached response for ``(prompt, model)``, or ``None``.

        A ``None`` return means either a cache miss or an expired entry.
        Expired entries are evicted on access.  Hit/miss counters are always
        updated.

        Args:
            prompt: Exact user prompt text.
            model: Model identifier (e.g. ``"claude-haiku-4-5"``).

        Returns:
            Cached response string, or ``None`` on miss / expiry.
        """
        key   = self._key(prompt, model)
        entry = self._store.get(key)

        if entry is None:
            self._total_misses += 1
            return None

        if self._dt(entry["expires_at"]) < self._now():
            del self._store[key]
            if self.auto_persist:
                self._save()
            self._total_misses += 1
            return None

        entry["hit_count"] += 1
        self._total_hits += 1
        return entry["response"]

    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        ttl_seconds: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        """
        Store a response in the cache.

        Args:
            prompt: The exact prompt text.
            model: Model identifier.
            response: The LLM response to cache.
            ttl_seconds: Entry lifetime in seconds.  Uses ``default_ttl`` if
                         omitted.
            input_tokens: Actual input token count (estimated from prompt
                          length when 0).
            output_tokens: Actual output token count (estimated when 0).

        Returns:
            The cache key (64-character SHA-256 hex string).
        """
        ttl  = ttl_seconds if ttl_seconds is not None else self.default_ttl
        key  = self._key(prompt, model)
        now  = self._now()
        in_t = input_tokens  or max(1, len(prompt)   // _CHARS_PER_TOKEN)
        ou_t = output_tokens or max(1, len(response) // _CHARS_PER_TOKEN)

        self._store[key] = {
            "key":            key,
            "prompt_hash":    self._phash(prompt),
            "prompt_preview": prompt[:100],
            "model":          model,
            "response":       response,
            "created_at":     now.isoformat(),
            "expires_at":     (now + timedelta(seconds=ttl)).isoformat(),
            "hit_count":      0,
            "input_tokens":   in_t,
            "output_tokens":  ou_t,
        }
        if self.auto_persist:
            self._save()
        return key

    def invalidate(self, prompt: str, model: Optional[str] = None) -> int:
        """
        Remove cached entries for ``prompt``.

        Args:
            prompt: The prompt to invalidate.
            model: When specified, only the entry for this model is removed.
                   When ``None``, all models cached for this prompt are removed.

        Returns:
            Number of entries actually removed.
        """
        if model is not None:
            key = self._key(prompt, model)
            if key not in self._store:
                return 0
            del self._store[key]
            if self.auto_persist:
                self._save()
            return 1

        ph      = self._phash(prompt)
        to_drop = [k for k, e in self._store.items() if e["prompt_hash"] == ph]
        for k in to_drop:
            del self._store[k]
        if to_drop and self.auto_persist:
            self._save()
        return len(to_drop)

    def clear_expired(self) -> int:
        """
        Remove all entries whose TTL has elapsed.

        Returns:
            Number of entries removed.
        """
        now     = self._now()
        to_drop = [
            k for k, e in self._store.items()
            if self._dt(e["expires_at"]) < now
        ]
        for k in to_drop:
            del self._store[k]
        if to_drop and self.auto_persist:
            self._save()
        return len(to_drop)

    def clear_all(self) -> int:
        """Remove every entry.  Returns count removed."""
        count = len(self._store)
        self._store.clear()
        if self.auto_persist:
            self._save()
        return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> CacheStats:
        """
        Return a snapshot of cache statistics.

        ``memory_used_mb`` is estimated from stored string lengths plus a
        constant per-entry overhead; it does not reflect Python object
        headers or JSON serialisation overhead.
        """
        now     = self._now()
        expired = sum(
            1 for e in self._store.values()
            if self._dt(e["expires_at"]) < now
        )
        reqs    = self._total_hits + self._total_misses
        hr_pct  = self._total_hits / reqs * 100 if reqs else 0.0
        mem     = sum(
            len(e.get("response", ""))
            + len(e.get("prompt_preview", ""))
            + 300           # per-entry metadata overhead estimate (bytes)
            for e in self._store.values()
        )
        return CacheStats(
            total_hits=self._total_hits,
            total_misses=self._total_misses,
            total_requests=reqs,
            hit_rate_pct=round(hr_pct, 2),
            entries_count=len(self._store),
            expired_entries=expired,
            memory_used_mb=round(mem / 1_048_576, 4),
        )

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(
        self,
        prompts: list[str],
        model: str = "claude-haiku-4-5",
        estimated_output_tokens: int = 200,
    ) -> None:
        """
        Simulate serving ``prompts`` with and without this cache and print
        a cost-comparison report.

        No API calls are made; token counts are estimated at 1 token per
        ``_CHARS_PER_TOKEN`` characters.

        Args:
            prompts: Request sequence (may contain duplicates).
            model: Pricing model to use for cost projection.
            estimated_output_tokens: Assumed output length per API call.
        """
        in_p  = _INPUT_PRICE.get(model,  1.00e-6)
        out_p = _OUTPUT_PRICE.get(model, 5.00e-6)

        seen: set[str] = set()
        hits = misses = 0
        cost_no_cache = cost_with_cache = 0.0

        # Track repeat frequency for histogram
        freq: dict[str, int] = {}
        for p in prompts:
            freq[p] = freq.get(p, 0) + 1

        for prompt in prompts:
            in_tok    = max(1, len(prompt) // _CHARS_PER_TOKEN)
            call_cost = in_tok * in_p + estimated_output_tokens * out_p
            cost_no_cache += call_cost
            if prompt in seen:
                hits += 1
            else:
                seen.add(prompt)
                misses += 1
                cost_with_cache += call_cost

        total    = len(prompts)
        unique   = len(seen)
        savings  = cost_no_cache - cost_with_cache
        save_pct = savings / cost_no_cache * 100 if cost_no_cache else 0.0

        # Repetition breakdown
        once    = sum(1 for c in freq.values() if c == 1)
        twice   = sum(1 for c in freq.values() if c == 2)
        more    = sum(1 for c in freq.values() if c >= 3)

        SEP  = "=" * 64
        THIN = "-" * 64
        print()
        print(SEP)
        print("  CACHE BENCHMARK")
        print(SEP)
        print(f"  Model:                   {model}")
        print(f"  Est. output tokens/call: {estimated_output_tokens}")
        print()
        print(f"  Total requests:          {total:>8,}")
        print(f"  Unique prompts:          {unique:>8,}  ({unique/total*100:.1f}%)")
        print(f"  Repeated requests:       {hits:>8,}  ({hits/total*100:.1f}%)")
        print()
        print(f"  Repetition breakdown:")
        print(f"    Seen exactly once:     {once:>8,}")
        print(f"    Seen exactly twice:    {twice:>8,}")
        print(f"    Seen 3+ times:         {more:>8,}")
        print()
        print(THIN)
        print(f"  Cost WITHOUT cache:      ${cost_no_cache:>10.6f}")
        print(f"  Cost WITH cache:         ${cost_with_cache:>10.6f}")
        print(f"  Savings:                 ${savings:>10.6f}  ({save_pct:.1f}%)")
        print(f"  API calls avoided:       {hits:>8,}")
        print(SEP)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from the JSON file and immediately purge expired ones."""
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._store = data
            self.clear_expired()
        except (json.JSONDecodeError, OSError, KeyError):
            self._store = {}

    def _save(self) -> None:
        """Atomically write the in-memory store to the JSON file."""
        try:
            tmp = self.persist_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2, ensure_ascii=False)
            tmp.replace(self.persist_path)
        except OSError as exc:
            print(
                f"[cache] WARNING: could not persist to {self.persist_path}: {exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        """Print a formatted statistics table to stdout."""
        s   = self.stats()
        SEP = "=" * 50
        print()
        print(SEP)
        print("  CACHE STATS")
        print(SEP)
        print(f"  Total requests:    {s.total_requests:>10,}")
        print(f"  Hits:              {s.total_hits:>10,}")
        print(f"  Misses:            {s.total_misses:>10,}")
        print(f"  Hit rate:          {s.hit_rate_pct:>9.1f}%")
        print(f"  Live entries:      {s.entries_count - s.expired_entries:>10,}")
        print(f"  Expired entries:   {s.expired_entries:>10,}")
        print(f"  Memory (est.):     {s.memory_used_mb:>9.4f} MB")
        print(SEP)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

_UNIQUE_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain machine learning in simple terms.",
    "How does transformer attention work?",
    "What are the main differences between Python lists and tuples?",
    "Summarize the key ideas of the Agile methodology.",
    "How do I reverse a string in Python?",
    "What is the CAP theorem in distributed systems?",
    "Explain the difference between TCP and UDP.",
    "What is a REST API and how does it work?",
    "How does gradient descent optimize neural networks?",
    "What is Docker and why is it useful?",
    "Explain the concept of recursion with an example.",
    "What are the SOLID principles in software engineering?",
    "How do I implement a binary search in Python?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain how public-key cryptography works.",
    "What is the time complexity of quicksort?",
    "How does the HTTP caching mechanism work?",
    "What is a decorator in Python?",
    "Explain the MapReduce programming model.",
    "What is the difference between a process and a thread?",
    "How does a hash table handle collisions?",
    "What is the difference between authentication and authorization?",
    "Explain what a Kubernetes pod is.",
    "How does git rebase differ from git merge?",
    "What is the Observer design pattern?",
    "How do I handle rate limiting in an API client?",
    "What is the difference between latency and throughput?",
    "Explain what a deadlock is and how to prevent it.",
    "What is the purpose of an API gateway?",
    "How does a load balancer choose which server to route to?",
    "What is the difference between synchronous and asynchronous code?",
    "Explain what dependency injection is.",
    "What is a bloom filter and when would you use one?",
    "How does the JIT compiler work?",
    "What is eventual consistency in distributed databases?",
    "Explain the difference between unit, integration, and e2e tests.",
    "What is a monorepo and what are its trade-offs?",
    "How does connection pooling improve database performance?",
    "What are the main features of TypeScript over JavaScript?",
    "What is a circuit breaker in microservices?",
    "How does an LRU cache work?",
    "Explain what CORS is and how it works.",
    "What is the difference between stack and heap memory?",
    "How do I paginate a large API response efficiently?",
    "What is a race condition and how do you detect it?",
    "Explain the concept of eventual consistency vs strong consistency.",
    "What is the difference between a message queue and an event bus?",
    "How does HTTPS protect data in transit?",
    "What is the purpose of a foreign key constraint in SQL?",
    "Explain what a webhook is.",
    "How does backpressure work in data streaming?",
    "What is the N+1 query problem in ORMs?",
    "Explain what a CDN does and when to use one.",
    "What is the difference between optimistic and pessimistic locking?",
    "How does the two-phase commit protocol work?",
    "What is A/B testing and how do you measure its results?",
    "Explain the difference between monolith and microservices architectures.",
    "What is feature flagging and why is it useful?",
    "How does WebSocket differ from HTTP?",
    "What is the Saga pattern for distributed transactions?",
    "Explain what a sparse index is in a database.",
    "What is tail call optimization?",
    "How do vector databases work?",
    "What is the difference between in-process and out-of-process caching?",
    "Explain what an idempotent API operation is.",
    "What is the purpose of the Content-Security-Policy header?",
    "How does semantic versioning work?",
    "What is the difference between encoding and encryption?",
    "Explain what a service mesh does.",
]


if __name__ == "__main__":
    import os
    import random

    CACHE_FILE = "exact_cache_demo.json"
    MODEL      = "claude-haiku-4-5"

    # Fresh run
    if Path(CACHE_FILE).exists():
        os.remove(CACHE_FILE)

    cache = ExactMatchCache(persist_path=CACHE_FILE, default_ttl=3600)

    # Build 100-request sequence: 70 unique + 30 repeats (shuffled)
    rng = random.Random(42)
    unique_pool = _UNIQUE_PROMPTS[:70]
    repeated    = rng.choices(unique_pool, k=30)
    all_requests: list[str] = unique_pool + repeated
    rng.shuffle(all_requests)

    print(f"\n{'='*60}")
    print("  EXACT-MATCH CACHE — simulation: 100 requests, ~30% repeats")
    print(f"{'='*60}")
    print(f"  Unique prompts in pool: {len(unique_pool)}")
    print(f"  Total requests:         {len(all_requests)}")
    print(f"  Persisting to:          {CACHE_FILE}")

    # Simulate serving requests
    api_call_count = 0
    log_lines: list[str] = []

    for i, prompt in enumerate(all_requests, 1):
        cached = cache.get(prompt, MODEL)

        if cached is not None:
            status = "HIT "
        else:
            # Simulate API response (no real API call)
            api_call_count += 1
            simulated_response = (
                f"[Simulated API response #{api_call_count}] "
                f"Answer to: {prompt[:55]}..."
            )
            cache.set(
                prompt, MODEL, simulated_response,
                ttl_seconds=3600,
                input_tokens=len(prompt) // _CHARS_PER_TOKEN,
                output_tokens=80,
            )
            status = "MISS"

        log_lines.append(f"  req {i:>3}  [{status}]  {prompt[:55]}")

    # Print first 20 log lines as sample
    print(f"\n  Sample of first 20 requests:")
    print(f"  {'-'*62}")
    for line in log_lines[:20]:
        print(line)
    if len(log_lines) > 20:
        print(f"  ... ({len(log_lines) - 20} more requests not shown)")

    print(f"\n  Actual API calls made: {api_call_count} / {len(all_requests)}")

    # Stats and benchmark
    cache.print_stats()
    cache.benchmark(all_requests, model=MODEL, estimated_output_tokens=150)

    # Demonstrate TTL expiry simulation
    print("\n  TTL EXPIRY DEMO")
    print("  " + "-" * 40)
    short_cache = ExactMatchCache(
        persist_path="ttl_demo.json", default_ttl=1, auto_persist=False
    )
    short_cache.set("test prompt", MODEL, "test response", ttl_seconds=1)
    hit1 = short_cache.get("test prompt", MODEL)
    print(f"  get() immediately:       {'HIT' if hit1 else 'MISS'}")

    import time
    time.sleep(1.1)
    hit2 = short_cache.get("test prompt", MODEL)
    print(f"  get() after 1.1 seconds: {'HIT' if hit2 else 'MISS (TTL expired)'}")
    evicted = short_cache.clear_expired()
    print(f"  clear_expired() removed: {evicted} entry")

    # Cleanup demo files
    for f in ["ttl_demo.json"]:
        if Path(f).exists():
            os.remove(f)
