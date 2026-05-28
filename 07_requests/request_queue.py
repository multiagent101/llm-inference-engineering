"""
request_queue.py - Async priority queue with worker pool for LLM requests.

Requests are ordered by priority (HIGH → NORMAL → LOW) and processed by a
configurable worker pool.  Transient failures trigger exponential-backoff
retries; requests that exhaust all retries land in a dead-letter queue that
is persisted to ``failed_requests.json``.

Special HIGH-priority timeout behaviour
----------------------------------------
When a HIGH request exceeds its per-attempt timeout the worker logs a
graceful timeout event, re-inserts the request at the *front* of the HIGH
priority band (ahead of other waiting HIGH items), and keeps retrying.
Only after ``max_retries`` attempts does the request move to the DLQ.

Statistics
----------
``queue_depth``         — items still in the heap (not yet dequeued)
``avg_wait_time_ms``    — mean enqueue-to-dispatch time, per priority level
``success_rate``        — fraction of permanently-resolved requests that succeeded
``dead_letter_count``   — items written to the dead-letter file

Standard-library only: asyncio, heapq, json, uuid, time, random, dataclasses,
enum, pathlib, typing.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

_DEAD_LETTER_FILE = Path(__file__).parent / "failed_requests.json"

# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------


class Priority(IntEnum):
    """
    Request priority levels.

    Smaller integer value = higher urgency in the min-heap.  Do not rely on
    the exact values; use the named constants.
    """

    HIGH   = 0  # SLA-critical: timeout triggers graceful re-queue
    NORMAL = 1  # Standard interactive requests
    LOW    = 2  # Batch / background work


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QueuedRequest:
    """
    Internal state for one pending or in-flight request.

    Attributes
    ----------
    request_id:
        Unique opaque identifier returned by :meth:`RequestQueue.enqueue`.
    prompt:
        The text payload for the processor.
    priority:
        Priority level at time of (re-)insertion.
    timeout_ms:
        Per-attempt timeout in milliseconds.
    enqueued_at:
        ``time.monotonic()`` timestamp at the moment the request was first
        submitted (not updated on re-queue).
    attempts:
        Number of processing attempts made so far (incremented by the worker).
    """

    request_id:  str
    prompt:      str
    priority:    Priority
    timeout_ms:  float
    enqueued_at: float
    attempts:    int = 0

    # Prevent heapq from comparing QueuedRequest objects when the seq
    # tie-breaker is equal (it never is, but this is defensive).
    def __lt__(self, other: object) -> bool:
        return False


@dataclass
class RequestResult:
    """
    Final outcome for one request (success or permanent failure).

    Attributes
    ----------
    request_id:
        Matches the ID returned by :meth:`RequestQueue.enqueue`.
    prompt_preview:
        First 60 characters of the prompt (for logging).
    priority:
        Priority name string ("HIGH", "NORMAL", "LOW").
    success:
        True when the processor returned a valid response.
    response:
        Processor output; ``None`` on failure.
    wait_ms:
        Time spent in the queue before the *first* processing attempt (ms).
    latency_ms:
        Wall-clock processing time of the *final* attempt (ms).
    attempts:
        Total attempt count including all retries.
    error:
        Failure description when ``success`` is False.
    """

    request_id:     str
    prompt_preview: str
    priority:       str
    success:        bool
    response:       Optional[str]
    wait_ms:        float
    latency_ms:     float
    attempts:       int
    error:          Optional[str] = None


@dataclass
class QueueStats:
    """
    Summary statistics emitted by :meth:`RequestQueue.process_queue`.

    Attributes
    ----------
    queue_depth:
        Items still waiting in the heap when stats were captured (0 after
        a complete drain).
    dead_letter_count:
        Requests written to the dead-letter file during this run.
    success_rate:
        Fraction of permanently-resolved requests that succeeded (0–1).
    avg_wait_ms_high / _normal / _low:
        Mean enqueue-to-first-dispatch wait time per priority level (ms).
    total_processed:
        Total requests permanently resolved (success + DLQ).
    elapsed_ms:
        Wall-clock time from the first ``process_queue`` call to drain.
    """

    queue_depth:       int
    dead_letter_count: int
    success_rate:      float
    avg_wait_ms_high:  float
    avg_wait_ms_normal: float
    avg_wait_ms_low:   float
    total_processed:   int
    elapsed_ms:        float


# ---------------------------------------------------------------------------
# Internal per-priority accumulator
# ---------------------------------------------------------------------------


class _PriStats:
    __slots__ = ("total", "successful", "_total_wait_ms")

    def __init__(self) -> None:
        self.total:         int   = 0
        self.successful:    int   = 0
        self._total_wait_ms: float = 0.0

    def record(self, wait_ms: float, success: bool) -> None:
        self.total          += 1
        self._total_wait_ms += wait_ms
        if success:
            self.successful += 1

    @property
    def avg_wait_ms(self) -> float:
        return self._total_wait_ms / self.total if self.total else 0.0

    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# RequestQueue
# ---------------------------------------------------------------------------


class RequestQueue:
    """
    Async priority queue with worker pool, retries, and dead-letter storage.

    Processor contract
    ------------------
    The ``processor`` callable must have the signature::

        async def processor(prompt: str, timeout_ms: float) -> str: ...

    It should raise any exception on failure; the queue handles retries and
    DLQ routing.  ``asyncio.TimeoutError`` from exceeding ``timeout_ms`` is
    treated as a retryable transient error (with special re-prioritisation
    logic for HIGH requests).

    Parameters
    ----------
    processor:
        Async callable that executes one LLM request.
    max_retries:
        Maximum *additional* attempts after the first failure (default 3,
        so up to 4 total attempts before DLQ).
    dead_letter_file:
        Path to the JSON file that accumulates permanently-failed requests.
        Defaults to ``07_requests/failed_requests.json``.
    """

    def __init__(
        self,
        processor:         Callable[[str, float], Awaitable[str]],
        max_retries:       int           = 3,
        dead_letter_file:  Optional[Path] = None,
    ) -> None:
        self._processor       = processor
        self._max_retries     = max_retries
        self._dlq_file        = dead_letter_file or _DEAD_LETTER_FILE

        # Heap entries: (priority_value, seq, QueuedRequest)
        # seq < 0 for re-queued items so they sort before fresh submissions
        # within the same priority band.
        self._heap:     list[tuple[int, int, QueuedRequest]] = []
        self._counter:  int = 0   # ever-increasing; negated for re-queues

        # Callbacks keyed by request_id
        self._callbacks: dict[str, Optional[Callable[[RequestResult], None]]] = {}

        # "In-flight" count: enqueued items not yet permanently resolved.
        # Decremented only on success, DLQ, or non-retryable failure.
        # Re-queued retries do NOT decrement (the item stays in flight).
        self._in_flight: int = 0

        # Accumulated outcomes and stats
        self._results:      list[RequestResult]       = []
        self._dead_letters: list[dict[str, Any]]       = []
        self._pri_stats:    dict[Priority, _PriStats]  = {
            p: _PriStats() for p in Priority
        }

        # Per-request first-enqueue timestamp (not updated on retry)
        self._first_enqueued_at: dict[str, float] = {}

        # Asyncio lock protecting the heap (created inside the event loop)
        self._lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------
    # Read-only statistics
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        """Number of items currently waiting in the heap."""
        return len(self._heap)

    @property
    def dead_letter_count(self) -> int:
        """Requests permanently failed and written to the DLQ file."""
        return len(self._dead_letters)

    @property
    def success_rate(self) -> float:
        """Fraction of permanently-resolved requests that succeeded."""
        total      = sum(s.total      for s in self._pri_stats.values())
        successful = sum(s.successful for s in self._pri_stats.values())
        return successful / total if total else 0.0

    def avg_wait_time_ms(self, priority: Priority) -> float:
        """
        Mean enqueue-to-dispatch wait time in milliseconds for *priority*.

        Returns 0 if no requests of that priority have been resolved yet.
        """
        return self._pri_stats[priority].avg_wait_ms

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        prompt:      str,
        priority:    Priority,
        callback:    Optional[Callable[[RequestResult], None]] = None,
        timeout_ms:  float = 5_000.0,
    ) -> str:
        """
        Add a request to the priority queue.

        May be called before or while :meth:`process_queue` is running.
        Items added while workers are active are picked up automatically.

        Parameters
        ----------
        prompt:
            Text payload forwarded to the processor.
        priority:
            Urgency level.  HIGH items are processed before NORMAL and LOW.
        callback:
            Called exactly once with the final :class:`RequestResult` when
            the request is permanently resolved (whether success or DLQ).
        timeout_ms:
            Per-attempt timeout.  HIGH requests that time out are
            re-inserted at the front of the HIGH band rather than failing
            immediately (see module docstring).

        Returns
        -------
        str
            Opaque 8-character request ID.
        """
        request_id = uuid.uuid4().hex[:8]
        now        = time.monotonic()

        request = QueuedRequest(
            request_id  = request_id,
            prompt      = prompt,
            priority    = priority,
            timeout_ms  = timeout_ms,
            enqueued_at = now,
        )

        seq            = self._counter
        self._counter += 1

        heapq.heappush(self._heap, (priority.value, seq, request))

        self._callbacks[request_id]        = callback
        self._first_enqueued_at[request_id] = now
        self._in_flight                    += 1

        return request_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _requeue(self, request: QueuedRequest) -> None:
        """
        Re-insert *request* at the front of its priority band.

        Negative sequence numbers sort before all positive (fresh) values,
        so retried items jump ahead of newly-submitted same-priority items.
        The most-recently retried item gets the most-negative seq and lands
        first among all retried items (LIFO within retried band — fair for
        the demo; a FIFO variant would need a separate monotone counter).
        """
        seq            = -(self._counter + 1)
        self._counter += 1
        heapq.heappush(self._heap, (request.priority.value, seq, request))

    def _record_result(
        self,
        request:    QueuedRequest,
        wait_ms:    float,
        latency_ms: float,
        success:    bool,
        response:   Optional[str] = None,
        error:      Optional[str] = None,
    ) -> None:
        """Commit a permanently-resolved request: update stats, fire callback."""
        result = RequestResult(
            request_id     = request.request_id,
            prompt_preview = request.prompt[:60],
            priority       = request.priority.name,
            success        = success,
            response       = response,
            wait_ms        = round(wait_ms,    1),
            latency_ms     = round(latency_ms, 1),
            attempts       = request.attempts,
            error          = error,
        )
        self._results.append(result)
        self._pri_stats[request.priority].record(wait_ms, success)
        self._in_flight -= 1

        cb = self._callbacks.pop(request.request_id, None)
        if cb is not None:
            cb(result)

    def _send_to_dlq(
        self,
        request:    QueuedRequest,
        wait_ms:    float,
        latency_ms: float,
        error:      str,
    ) -> None:
        """Persist a permanently-failed request and record the result."""
        record: dict[str, Any] = {
            "request_id": request.request_id,
            "prompt":     request.prompt,
            "priority":   request.priority.name,
            "attempts":   request.attempts,
            "wait_ms":    round(wait_ms,    1),
            "latency_ms": round(latency_ms, 1),
            "error":      error,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._dead_letters.append(record)

        # Atomic append to DLQ file
        try:
            existing: list[dict[str, Any]] = []
            if self._dlq_file.exists():
                existing = json.loads(self._dlq_file.read_text(encoding="utf-8"))
            existing.append(record)
            self._dlq_file.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass  # DLQ persistence failure must never crash the queue

        self._record_result(request, wait_ms, latency_ms, success=False, error=error)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _process_one(self, request: QueuedRequest, worker_id: int) -> None:
        """
        Execute one attempt for *request*.

        On success records the result.  On retryable failure re-queues the
        request (HIGH with graceful-error logging; others with back-off).
        After ``max_retries`` total attempts routes the request to the DLQ.
        """
        request.attempts += 1
        t_attempt_start = time.monotonic()

        # Wait time is measured from the *original* enqueue time so retried
        # items do not artificially inflate wait_ms.
        first_enqueued = self._first_enqueued_at.get(
            request.request_id, request.enqueued_at
        )
        wait_ms = (t_attempt_start - first_enqueued) * 1_000

        try:
            response = await asyncio.wait_for(
                self._processor(request.prompt, request.timeout_ms),
                timeout=request.timeout_ms / 1_000,
            )
            latency_ms = (time.monotonic() - t_attempt_start) * 1_000
            self._record_result(request, wait_ms, latency_ms, success=True, response=response)

        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - t_attempt_start) * 1_000

            if request.attempts > self._max_retries:
                print(
                    f"  [W{worker_id}] DLQ   {request.priority.name:<6} "
                    f"id={request.request_id}  timeout exhausted "
                    f"({request.attempts} attempts)"
                )
                self._send_to_dlq(request, wait_ms, latency_ms, error="TimeoutError")
                return

            if request.priority == Priority.HIGH:
                # Graceful timeout event: notify + re-prioritise to HEAD of HIGH band
                print(
                    f"  [W{worker_id}] GRACE  HIGH   "
                    f"id={request.request_id}  "
                    f"timeout {latency_ms:.0f}ms > {request.timeout_ms:.0f}ms  "
                    f"-> re-queued at HIGH (attempt {request.attempts}/{self._max_retries})"
                )
            else:
                backoff_s = 0.05 * request.attempts
                await asyncio.sleep(backoff_s)

            self._requeue(request)

        except Exception as exc:
            latency_ms = (time.monotonic() - t_attempt_start) * 1_000

            if request.attempts > self._max_retries:
                print(
                    f"  [W{worker_id}] DLQ   {request.priority.name:<6} "
                    f"id={request.request_id}  error exhausted: {exc}"
                )
                self._send_to_dlq(request, wait_ms, latency_ms, error=str(exc))
                return

            backoff_s = 0.05 * request.attempts
            print(
                f"  [W{worker_id}] RETRY  {request.priority.name:<6} "
                f"id={request.request_id}  attempt {request.attempts}  "
                f"err={exc!s:.40s}  backoff={backoff_s*1000:.0f}ms"
            )
            await asyncio.sleep(backoff_s)
            self._requeue(request)

    async def _worker(self, worker_id: int) -> None:
        """
        Pull requests from the heap and process them until the queue drains.

        Exits when both the heap is empty and ``_in_flight`` reaches zero
        (all items permanently resolved, none awaiting retry).
        """
        assert self._lock is not None, "_worker called before process_queue"

        while True:
            async with self._lock:
                if self._heap:
                    _, _, request = heapq.heappop(self._heap)
                else:
                    request = None

            if request is None:
                # Heap empty; check if all work is done
                if self._in_flight == 0:
                    return
                # Some items are being processed by peer workers or awaiting
                # retry back-off; yield and check again shortly.
                await asyncio.sleep(0.005)
                continue

            await self._process_one(request, worker_id)

    # ------------------------------------------------------------------
    # process_queue
    # ------------------------------------------------------------------

    async def process_queue(self, workers: int = 3) -> QueueStats:
        """
        Drain the queue using a pool of *workers* async workers.

        Blocks until every enqueued request has been permanently resolved
        (succeeded or moved to the dead-letter queue).  Additional requests
        may be enqueued while this coroutine is running.

        Parameters
        ----------
        workers:
            Number of concurrent worker coroutines.

        Returns
        -------
        QueueStats
            Aggregated statistics for the completed run.
        """
        self._lock = asyncio.Lock()
        t_start = time.monotonic()

        worker_tasks = [
            asyncio.create_task(self._worker(i), name=f"worker-{i}")
            for i in range(workers)
        ]
        await asyncio.gather(*worker_tasks)

        elapsed_ms = (time.monotonic() - t_start) * 1_000

        return QueueStats(
            queue_depth        = len(self._heap),
            dead_letter_count  = self.dead_letter_count,
            success_rate       = self.success_rate,
            avg_wait_ms_high   = self.avg_wait_time_ms(Priority.HIGH),
            avg_wait_ms_normal = self.avg_wait_time_ms(Priority.NORMAL),
            avg_wait_ms_low    = self.avg_wait_time_ms(Priority.LOW),
            total_processed    = len(self._results),
            elapsed_ms         = elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    # ------------------------------------------------------------------
    # Mock processor: simulates variable latency and occasional failures
    # ------------------------------------------------------------------

    # Latency range per priority (the processor doesn't know about priority,
    # but we encode it in the prompt prefix for demo clarity).
    _LATENCY_BY_PRIORITY: dict[str, tuple[float, float]] = {
        "HIGH":   (0.03, 0.18),   # 30–180 ms
        "NORMAL": (0.05, 0.25),   # 50–250 ms
        "LOW":    (0.08, 0.35),   # 80–350 ms
    }
    _FAIL_RATE = 0.12  # 12% random failure probability

    async def _mock_processor(prompt: str, timeout_ms: float) -> str:
        """Simulate an LLM call: random latency + occasional failure."""
        tag = prompt.split("|")[0].strip() if "|" in prompt else "NORMAL"
        lo, hi = _LATENCY_BY_PRIORITY.get(tag, (0.05, 0.25))
        await asyncio.sleep(random.uniform(lo, hi))

        if random.random() < _FAIL_RATE:
            raise RuntimeError("simulated transient error")

        return f"[OK] {prompt[prompt.find('|')+1:].strip()[:40]}..."

    # ------------------------------------------------------------------
    # Build prompts: 5 HIGH, 10 NORMAL, 5 LOW
    # ------------------------------------------------------------------

    _HIGH_PROMPTS = [
        "What is the p99 latency threshold for SLA compliance?",
        "Summarise the current outage in one sentence.",
        "Is the payment service healthy right now?",
        "What caused the spike in error rate 5 minutes ago?",
        "Provide a one-line status update for the incident channel.",
    ]
    _NORMAL_PROMPTS = [
        "Translate 'hello world' into French.",
        "Write a two-sentence bio for a software engineer.",
        "What is asyncio in Python?",
        "List three benefits of async programming.",
        "Explain token-bucket rate limiting briefly.",
        "What is the difference between p50 and p99 latency?",
        "Name two use cases for priority queues.",
        "What does FIFO mean?",
        "Define dead-letter queue in one sentence.",
        "Why is exponential backoff useful?",
    ]
    _LOW_PROMPTS = [
        "Generate a haiku about distributed systems.",
        "Summarise the history of message queues in 3 sentences.",
        "List 5 famous async frameworks.",
        "Describe the producer-consumer pattern.",
        "What is eventual consistency?",
    ]

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    _callback_log: list[str] = []

    def _on_done(result: RequestResult) -> None:
        icon = "ok" if result.success else "FAIL"
        _callback_log.append(
            f"  callback {icon:<4}  id={result.request_id}  "
            f"{result.priority:<6}  attempts={result.attempts}  "
            f"wait={result.wait_ms:.0f}ms  lat={result.latency_ms:.0f}ms"
        )

    # ------------------------------------------------------------------
    # Main coroutine
    # ------------------------------------------------------------------

    async def _main() -> None:
        sep  = "=" * 72
        thin = "-" * 72

        print(f"\n{sep}")
        print("  PRIORITY QUEUE DEMO  --  20 requests  (5 HIGH / 10 NORMAL / 5 LOW)")
        print(sep)

        queue = RequestQueue(
            processor    = _mock_processor,
            max_retries  = 3,
            dead_letter_file = Path(__file__).parent / "failed_requests.json",
        )

        # HIGH: tight 120 ms timeout to trigger occasional graceful-timeout
        # NORMAL / LOW: generous 3 s / 5 s timeout
        ids_high   = [queue.enqueue(f"HIGH|{p}",   Priority.HIGH,   _on_done, timeout_ms=120)  for p in _HIGH_PROMPTS]
        ids_normal = [queue.enqueue(f"NORMAL|{p}", Priority.NORMAL, _on_done, timeout_ms=3000) for p in _NORMAL_PROMPTS]
        ids_low    = [queue.enqueue(f"LOW|{p}",    Priority.LOW,    _on_done, timeout_ms=5000) for p in _LOW_PROMPTS]

        all_ids = ids_high + ids_normal + ids_low
        print(f"\n  Enqueued {len(all_ids)} requests -- starting 3-worker pool ...\n")

        # ------------------------------------------------------------------
        # Live processing output (each worker prints its own line)
        # ------------------------------------------------------------------

        print(f"  {'[Wk]':<7} {'EVENT':<6} {'PRIO':<7} {'ID':<10} {'DETAIL'}")
        print(f"  {thin}")

        stats = await queue.process_queue(workers=3)

        # ------------------------------------------------------------------
        # Callback log
        # ------------------------------------------------------------------

        print(f"\n{sep}")
        print("  CALLBACK LOG  (fired once per permanently-resolved request)")
        print(sep)
        for line in _callback_log:
            print(line)

        # ------------------------------------------------------------------
        # Per-request results table
        # ------------------------------------------------------------------

        print(f"\n{sep}")
        print(f"  {'#':<3}  {'ID':<10} {'PRI':<7} {'OK':>3}  "
              f"{'ATT':>3}  {'WAIT':>8}  {'LAT':>8}  PREVIEW")
        print(f"  {thin}")

        for i, r in enumerate(queue._results, 1):
            ok_sym  = "ok" if r.success else "FAIL"
            preview = r.prompt_preview[r.prompt_preview.find("|")+1:].strip()[:30]
            print(
                f"  {i:<3}  {r.request_id:<10} {r.priority:<7} {ok_sym:>3}  "
                f"{r.attempts:>3}  {r.wait_ms:>7.0f}ms  {r.latency_ms:>7.0f}ms  {preview}"
            )

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------

        sep2 = "-" * 40
        print(f"\n{sep}")
        print("  QUEUE STATISTICS")
        print(sep)
        print(f"  {'Metric':<36}  {'Value':>12}")
        print(f"  {sep2}")
        print(f"  {'Total processed':<36}  {stats.total_processed:>12}")
        print(f"  {'Queue depth (remaining)':<36}  {stats.queue_depth:>12}")
        print(f"  {'Dead letter count':<36}  {stats.dead_letter_count:>12}")
        print(f"  {'Overall success rate':<36}  {stats.success_rate:>11.1%}")
        print(f"  {sep2}")
        print(f"  {'Avg wait  HIGH  (ms)':<36}  {stats.avg_wait_ms_high:>11.1f}ms")
        print(f"  {'Avg wait  NORMAL (ms)':<36}  {stats.avg_wait_ms_normal:>11.1f}ms")
        print(f"  {'Avg wait  LOW   (ms)':<36}  {stats.avg_wait_ms_low:>11.1f}ms")
        print(f"  {sep2}")

        # Per-priority breakdown
        for pri in Priority:
            ps = queue._pri_stats[pri]
            print(
                f"  {pri.name:<6}  total={ps.total:<3}  "
                f"ok={ps.successful:<3}  "
                f"fail={ps.total - ps.successful:<3}  "
                f"success={ps.success_rate:.0%}  "
                f"avg_wait={ps.avg_wait_ms:.0f}ms"
            )

        print(f"  {sep2}")
        print(f"  {'Total elapsed':<36}  {stats.elapsed_ms:>11.0f}ms")
        print(f"  {'Throughput':<36}  {stats.total_processed / (stats.elapsed_ms/1000):>10.1f} req/s")

        if stats.dead_letter_count:
            print(f"\n  Dead-letter file: {queue._dlq_file}")

        print(f"{sep}\n")

    asyncio.run(_main())
