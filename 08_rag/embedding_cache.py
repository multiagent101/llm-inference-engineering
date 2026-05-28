"""
embedding_cache.py - Disk-backed embedding cache with SHA-256 keying.

Every embedding is stored as a single ``{sha256}.npy`` file alongside a
``cache_index.json`` that maps keys to metadata (model, text preview,
timestamps, access count).  Cache keys are deterministic:

    key = SHA256(text + "||" + model_name)

so the same text embedded with the same model always hits the same slot,
regardless of which process wrote it or when.

Savings model
-------------
sentence-transformers is a free local library, so the monetary saving is
zero.  To make the cost story concrete, the module also reports equivalent
savings **if you were using OpenAI text-embedding-3-small** ($0.02/1M
tokens): every cache hit is one embedding call avoided, and the cumulative
avoided token count drives a USD savings estimate.

Parallel batch compute
----------------------
``batch_get_or_compute`` reads the cache for every text concurrently
(ThreadPoolExecutor for disk I/O), collects all misses, encodes them in a
single ``model.encode()`` call (sentence-transformers handles internal
batching efficiently), then writes the new embeddings back to disk
concurrently.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------

_OPENAI_MODEL            = "text-embedding-3-small"
_OPENAI_PRICE_PER_TOKEN  = 0.020 / 1_000_000   # USD per token

_DEFAULT_MODEL    = "all-MiniLM-L6-v2"
_DEFAULT_CACHE_DIR = Path(__file__).parent / "embedding_cache"
_INDEX_FILENAME    = "cache_index.json"

# Approximate token multiplier: words × 1.3 ≈ BPE tokens for English
_WORDS_TO_TOKENS = 1.3


def _approx_tokens(text: str) -> int:
    """Estimate token count as whitespace words * 1.3, minimum 1."""
    return max(1, int(len(text.split()) * _WORDS_TO_TOKENS))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """
    Snapshot of cache performance for the current in-memory session.

    Attributes
    ----------
    hits:
        Requests served from the on-disk cache.
    misses:
        Requests that required a fresh embedding computation.
    hit_rate:
        ``hits / (hits + misses)`` as a fraction (0–1).
    bytes_saved:
        Embedding bytes that were read from cache rather than recomputed
        (each float32 embedding: ``dim * 4`` bytes per hit).
    compute_time_saved_ms:
        Estimated wall-clock milliseconds saved by cache hits, derived from
        the mean per-miss compute time.
    avg_miss_compute_time_ms:
        Mean time (ms) to embed one cache-miss text.
    api_calls_saved:
        Number of embedding API calls avoided — equal to ``hits``.  If you
        were using a paid API each hit is one billable call avoided.
    openai_tokens_saved:
        Estimated tokens that would have been sent to OpenAI for the hit
        texts (whitespace words * 1.3).
    openai_cost_saved_usd:
        ``openai_tokens_saved * $0.02/1M`` — equivalent monetary saving vs
        OpenAI ``text-embedding-3-small``.
    cache_entries:
        Total entries persisted on disk at the time of measurement.
    cache_size_bytes:
        Total bytes consumed by ``.npy`` files on disk.
    """

    hits:                      int
    misses:                    int
    hit_rate:                  float
    bytes_saved:               int
    compute_time_saved_ms:     float
    avg_miss_compute_time_ms:  float
    api_calls_saved:           int
    openai_tokens_saved:       int
    openai_cost_saved_usd:     float
    cache_entries:             int
    cache_size_bytes:          int


@dataclass
class SavingsProjection:
    """
    Monthly cost-savings extrapolation based on the current session.

    Attributes
    ----------
    session_duration_s:
        Wall-clock length of the recorded session (seconds).
    session_requests:
        Total embedding requests observed this session.
    session_hit_rate:
        Fraction of requests served from cache this session.
    projected_monthly_requests:
        ``session_requests / session_duration_s * 86400 * 30``.
    projected_monthly_hits:
        ``projected_monthly_requests * session_hit_rate``.
    projected_openai_tokens_saved:
        Tokens that would not be sent to OpenAI thanks to cache hits.
    projected_openai_cost_saved_usd:
        ``projected_openai_tokens_saved * $0.02/1M``.
    monthly_cost_without_cache_usd:
        Full OpenAI embedding cost for all projected monthly requests.
    monthly_cost_with_cache_usd:
        OpenAI embedding cost for only the projected monthly misses.
    monthly_savings_pct:
        ``(without - with) / without * 100``.
    """

    session_duration_s:              float
    session_requests:                int
    session_hit_rate:                float
    projected_monthly_requests:      int
    projected_monthly_hits:          int
    projected_openai_tokens_saved:   int
    projected_openai_cost_saved_usd: float
    monthly_cost_without_cache_usd:  float
    monthly_cost_with_cache_usd:     float
    monthly_savings_pct:             float


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """
    Disk-backed embedding cache with SHA-256 content addressing.

    Each text+model pair maps to a deterministic ``{sha256_key}.npy`` file.
    An accompanying ``cache_index.json`` stores metadata for every cached
    vector (timestamps, token estimate, access statistics).

    Thread safety
    -------------
    The in-memory index and session counters are protected by a
    ``threading.Lock``.  Concurrent reads from multiple threads are safe
    (each thread reads its own ``.npy`` file); concurrent writes are
    serialised through the lock.

    Parameters
    ----------
    cache_dir:
        Directory where ``.npy`` files and ``cache_index.json`` are stored.
        Created automatically if it does not exist.
    model_name:
        Default sentence-transformers model used when no model is specified
        in :meth:`get_or_compute` / :meth:`batch_get_or_compute`.
    max_workers:
        Thread-pool size for parallel cache I/O in :meth:`batch_get_or_compute`.
    """

    def __init__(
        self,
        cache_dir:   Optional[Path] = None,
        model_name:  str            = _DEFAULT_MODEL,
        max_workers: int            = 8,
    ) -> None:
        self._cache_dir  = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / _INDEX_FILENAME
        self._model_name = model_name
        self._max_workers = max_workers

        # Load or create the persistent index
        self._lock  = threading.Lock()
        self._index = self._load_index()

        # Session-level counters (reset on each __init__)
        self._hits:              int   = 0
        self._misses:            int   = 0
        self._bytes_saved:       int   = 0
        self._tokens_saved:      int   = 0
        self._miss_times_ms:     list[float] = []
        self._session_start:     float = time.monotonic()

        # Lazy model — loaded on first cache miss
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, dict]:
        """Load ``cache_index.json`` from disk; return empty dict if absent."""
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
                return raw.get("entries", {})
            except (json.JSONDecodeError, KeyError):
                return {}
        return {}

    def _save_index(self) -> None:
        """
        Atomically write the in-memory index to ``cache_index.json``.

        Uses a temp-file + ``os.replace`` to avoid leaving a half-written
        file if the process is interrupted mid-write.
        """
        payload = {"version": 1, "entries": self._index}
        tmp     = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(str(tmp), str(self._index_path))

    # ------------------------------------------------------------------
    # Key and path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(text: str, model: str) -> str:
        """
        Return the SHA-256 hex digest used as the cache key for *text*
        embedded with *model*.

        Parameters
        ----------
        text:
            The text to be embedded.
        model:
            Model identifier string (e.g. ``"all-MiniLM-L6-v2"``).

        Returns
        -------
        str
            64-character lowercase hex string.
        """
        return hashlib.sha256(f"{text}||{model}".encode()).hexdigest()

    def _npy_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npy"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Load and cache the sentence-transformer model lazily."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Single-text get_or_compute
    # ------------------------------------------------------------------

    def get_or_compute(
        self,
        text:       str,
        model:      Optional[str] = None,
    ) -> np.ndarray:
        """
        Return the embedding vector for *text*, computing it if not cached.

        Parameters
        ----------
        text:
            Input string to embed.
        model:
            Model name override.  Defaults to ``self._model_name``; changing
            this produces a different cache key so both embeddings coexist.

        Returns
        -------
        np.ndarray
            Float32 vector of shape ``(embedding_dim,)``.
        """
        m   = model or self._model_name
        key = self.cache_key(text, m)
        npy = self._npy_path(key)

        with self._lock:
            if key in self._index and npy.exists():
                # --- Cache hit ---
                vec = np.load(str(npy))
                self._hits           += 1
                self._bytes_saved    += vec.nbytes
                tok                   = self._index[key].get("token_estimate", 1)
                self._tokens_saved   += tok
                # Update access metadata
                self._index[key]["access_count"]  = self._index[key].get("access_count", 0) + 1
                self._index[key]["last_accessed"]  = datetime.now(timezone.utc).isoformat()
                self._save_index()
                return vec

        # --- Cache miss ---
        t0  = time.perf_counter()
        vec = self._get_model().encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        ms  = (time.perf_counter() - t0) * 1_000

        np.save(str(npy), vec)

        tok = _approx_tokens(text)
        entry = {
            "text_preview":    text[:60],
            "model":           m,
            "created_at":      datetime.now(timezone.utc).isoformat(),
            "last_accessed":   datetime.now(timezone.utc).isoformat(),
            "token_estimate":  tok,
            "bytes":           int(vec.nbytes),
            "access_count":    1,
        }

        with self._lock:
            self._misses          += 1
            self._miss_times_ms.append(ms)
            self._index[key]       = entry
            self._save_index()

        return vec

    # ------------------------------------------------------------------
    # Batch get_or_compute
    # ------------------------------------------------------------------

    def batch_get_or_compute(
        self,
        texts:  list[str],
        model:  Optional[str] = None,
    ) -> list[np.ndarray]:
        """
        Return embeddings for every text in *texts*, reading hits from cache
        and encoding all misses in a single batched ``model.encode()`` call.

        Order of the returned list mirrors the order of *texts*.

        Parallelism strategy
        --------------------
        1. Check the cache for every text concurrently
           (``ThreadPoolExecutor`` parallelises disk reads).
        2. Collect miss indices.
        3. Encode all miss texts in one ``model.encode(batch)`` call.
        4. Write new embeddings to disk concurrently.

        Parameters
        ----------
        texts:
            Ordered list of strings to embed.
        model:
            Model name override; defaults to ``self._model_name``.

        Returns
        -------
        list[np.ndarray]
            One float32 vector per input text, in the original order.
        """
        if not texts:
            return []

        m            = model or self._model_name
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        miss_indices: list[int]              = []

        # ---- Phase 1: parallel cache reads --------------------------------
        def _try_load(idx: int) -> Optional[np.ndarray]:
            key = self.cache_key(texts[idx], m)
            npy = self._npy_path(key)
            with self._lock:
                if key in self._index and npy.exists():
                    return (idx, np.load(str(npy)), key)
            return (idx, None, key)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(_try_load, i): i for i in range(len(texts))}
            for fut in concurrent.futures.as_completed(futures):
                idx, vec, key = fut.result()
                if vec is not None:
                    results[idx] = vec
                    with self._lock:
                        self._hits        += 1
                        self._bytes_saved += vec.nbytes
                        tok                = self._index[key].get("token_estimate", 1)
                        self._tokens_saved += tok
                        self._index[key]["access_count"] = (
                            self._index[key].get("access_count", 0) + 1
                        )
                        self._index[key]["last_accessed"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                else:
                    miss_indices.append(idx)

        # ---- Phase 2: batch encode misses ----------------------------------
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            t0         = time.perf_counter()
            miss_vecs  = self._get_model().encode(
                miss_texts, convert_to_numpy=True, show_progress_bar=False
            )
            ms_total   = (time.perf_counter() - t0) * 1_000
            ms_each    = ms_total / len(miss_texts)

            # ---- Phase 3: parallel cache writes ----------------------------
            def _write(pack: tuple[int, np.ndarray]) -> None:
                idx, vec = pack
                key      = self.cache_key(texts[idx], m)
                npy      = self._npy_path(key)
                np.save(str(npy), vec)
                tok   = _approx_tokens(texts[idx])
                entry = {
                    "text_preview":   texts[idx][:60],
                    "model":          m,
                    "created_at":     datetime.now(timezone.utc).isoformat(),
                    "last_accessed":  datetime.now(timezone.utc).isoformat(),
                    "token_estimate": tok,
                    "bytes":          int(vec.nbytes),
                    "access_count":   1,
                }
                with self._lock:
                    self._index[key] = entry

            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                list(ex.map(_write, zip(miss_indices, miss_vecs)))

            with self._lock:
                self._misses += len(miss_indices)
                self._miss_times_ms.extend([ms_each] * len(miss_indices))
                self._save_index()

            for list_pos, orig_idx in enumerate(miss_indices):
                results[orig_idx] = miss_vecs[list_pos]

        return [v for v in results if v is not None]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # invalidate_old
    # ------------------------------------------------------------------

    def invalidate_old(self, days: int = 30) -> int:
        """
        Delete cache entries not accessed within the last *days* days.

        Removes both the ``.npy`` file and the index entry for each stale
        record.

        Parameters
        ----------
        days:
            Entries whose ``last_accessed`` timestamp is older than this
            many days are evicted.

        Returns
        -------
        int
            Number of entries deleted.
        """
        cutoff  = datetime.now(timezone.utc) - timedelta(days=days)
        deleted = 0

        with self._lock:
            stale_keys = []
            for key, meta in self._index.items():
                try:
                    last_accessed = datetime.fromisoformat(meta["last_accessed"])
                    if last_accessed < cutoff:
                        stale_keys.append(key)
                except (KeyError, ValueError):
                    stale_keys.append(key)  # malformed entry → evict

            for key in stale_keys:
                npy = self._npy_path(key)
                if npy.exists():
                    npy.unlink()
                del self._index[key]
                deleted += 1

            if deleted:
                self._save_index()

        return deleted

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """
        Return a snapshot of cache performance for the current session.

        Compute-time-saved is estimated from the mean per-text encoding
        time of cache misses observed this session.  If no misses have
        occurred yet, this field is 0.

        Returns
        -------
        CacheStats
        """
        with self._lock:
            total_requests   = self._hits + self._misses
            hit_rate         = self._hits / total_requests if total_requests else 0.0
            avg_miss_ms      = (
                float(np.mean(self._miss_times_ms)) if self._miss_times_ms else 0.0
            )
            compute_saved_ms = avg_miss_ms * self._hits
            cost_saved       = self._tokens_saved * _OPENAI_PRICE_PER_TOKEN

            # Disk totals
            npy_files        = list(self._cache_dir.glob("*.npy"))
            cache_bytes      = sum(f.stat().st_size for f in npy_files)

            return CacheStats(
                hits                     = self._hits,
                misses                   = self._misses,
                hit_rate                 = round(hit_rate, 4),
                bytes_saved              = self._bytes_saved,
                compute_time_saved_ms    = round(compute_saved_ms, 1),
                avg_miss_compute_time_ms = round(avg_miss_ms, 1),
                api_calls_saved          = self._hits,
                openai_tokens_saved      = self._tokens_saved,
                openai_cost_saved_usd    = round(cost_saved, 8),
                cache_entries            = len(self._index),
                cache_size_bytes         = cache_bytes,
            )

    # ------------------------------------------------------------------
    # Monthly savings projection
    # ------------------------------------------------------------------

    def monthly_savings_projection(
        self,
        queries_per_day: Optional[float] = None,
        avg_tokens_per_text: Optional[float] = None,
    ) -> SavingsProjection:
        """
        Extrapolate current session statistics to a 30-day horizon.

        Parameters
        ----------
        queries_per_day:
            Override the per-day request rate.  If ``None``, the rate is
            computed from the current session: requests / elapsed seconds
            * 86400.
        avg_tokens_per_text:
            Override the mean tokens-per-text used for OpenAI cost
            estimation.  If ``None``, derived from the saved token totals.

        Returns
        -------
        SavingsProjection
        """
        stats           = self.get_stats()
        elapsed_s       = time.monotonic() - self._session_start
        total_requests  = stats.hits + stats.misses

        if total_requests == 0:
            # No data yet
            return SavingsProjection(
                session_duration_s=elapsed_s,
                session_requests=0,
                session_hit_rate=0.0,
                projected_monthly_requests=0,
                projected_monthly_hits=0,
                projected_openai_tokens_saved=0,
                projected_openai_cost_saved_usd=0.0,
                monthly_cost_without_cache_usd=0.0,
                monthly_cost_with_cache_usd=0.0,
                monthly_savings_pct=0.0,
            )

        rate_per_day    = queries_per_day or (total_requests / max(elapsed_s, 1) * 86_400)
        monthly_req     = int(rate_per_day * 30)

        avg_tok         = avg_tokens_per_text or (
            self._tokens_saved / stats.hits if stats.hits else 10.0
        )

        monthly_hits    = int(monthly_req * stats.hit_rate)
        monthly_misses  = monthly_req - monthly_hits
        tok_saved       = int(monthly_hits   * avg_tok)
        cost_saved      = tok_saved * _OPENAI_PRICE_PER_TOKEN
        full_cost       = int(monthly_req    * avg_tok) * _OPENAI_PRICE_PER_TOKEN
        miss_cost       = int(monthly_misses * avg_tok) * _OPENAI_PRICE_PER_TOKEN
        savings_pct     = (cost_saved / full_cost * 100) if full_cost > 0 else 0.0

        return SavingsProjection(
            session_duration_s              = round(elapsed_s, 1),
            session_requests                = total_requests,
            session_hit_rate                = stats.hit_rate,
            projected_monthly_requests      = monthly_req,
            projected_monthly_hits          = monthly_hits,
            projected_openai_tokens_saved   = tok_saved,
            projected_openai_cost_saved_usd = round(cost_saved, 6),
            monthly_cost_without_cache_usd  = round(full_cost,  6),
            monthly_cost_with_cache_usd     = round(miss_cost,  6),
            monthly_savings_pct             = round(savings_pct, 1),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Current session hit rate (0–1)."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def cache_entries(self) -> int:
        """Number of entries in the on-disk index."""
        with self._lock:
            return len(self._index)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import textwrap

    random.seed(42)

    # ------------------------------------------------------------------
    # Build 100 texts: 60 unique + 40 duplicates  →  40 % repetition rate
    # ------------------------------------------------------------------

    _TOPICS = [
        "embedding caching", "vector retrieval", "chunk size tuning",
        "reranking strategies", "token budget management",
        "context injection", "semantic similarity", "BM25 search",
        "prompt caching", "cosine distance", "FAISS indexing",
        "sentence transformers",
    ]
    _TEMPLATES = [
        "What is the role of {t} in production RAG pipelines?",
        "How does {t} affect LLM inference cost per query?",
        "Explain best practices for {t} in a high-throughput system.",
        "What are the main trade-offs when optimising {t}?",
        "How can {t} reduce end-to-end latency in RAG?",
    ]

    # 5 templates * 12 topics = 60 unique texts
    _UNIQUE = [tmpl.format(t=topic) for tmpl in _TEMPLATES for topic in _TOPICS]
    # 40 duplicates sampled from the first 40 unique texts
    _DUPES  = random.choices(_UNIQUE[:40], k=40)
    # Shuffle the full 100-text list
    _ALL    = _UNIQUE + _DUPES
    random.shuffle(_ALL)

    sep  = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print("  EMBEDDING CACHE DEMO")
    print(sep)
    print(f"  Total texts      : {len(_ALL)}")
    print(f"  Unique texts     : {len(_UNIQUE)}")
    print(f"  Duplicates       : {len(_DUPES)}  (expected hit rate ~{len(_DUPES)/len(_ALL):.0%})")
    print(f"  Model            : {_DEFAULT_MODEL}")
    print(f"  OpenAI reference : {_OPENAI_MODEL}  (${_OPENAI_PRICE_PER_TOKEN*1e6:.3f}/1M tok)")
    print()

    # ------------------------------------------------------------------
    # Initialise cache (clear stale entries from previous runs first)
    # ------------------------------------------------------------------

    cache = EmbeddingCache(
        cache_dir   = _DEFAULT_CACHE_DIR,
        model_name  = _DEFAULT_MODEL,
        max_workers = 8,
    )

    # Warm start: clear entries older than 0 days to get a clean demo
    removed = cache.invalidate_old(days=0)
    if removed:
        print(f"  Cleared {removed} stale entries from previous run.\n")

    # ------------------------------------------------------------------
    # Phase 1: process first 50 texts one-by-one to show live HIT/MISS
    # ------------------------------------------------------------------

    LIVE_SHOW = 50
    print(f"{sep}")
    print(f"  PHASE 1 -- Live processing (first {LIVE_SHOW} texts, get_or_compute)")
    print(sep)
    print(f"  {'#':>3}  {'Result':<6}  {'Key (12 chars)':<14}  "
          f"{'Tokens':>6}  {'Text preview'}")
    print(f"  {thin}")

    phase1_start = time.perf_counter()
    for i, text in enumerate(_ALL[:LIVE_SHOW], 1):
        hits_before = cache._hits
        vec = cache.get_or_compute(text)
        hit = cache._hits > hits_before
        key_short = cache.cache_key(text, _DEFAULT_MODEL)[:12]
        tok = _approx_tokens(text)
        tag = "HIT  " if hit else "MISS "
        print(
            f"  {i:>3}  {tag}  {key_short:<14}  "
            f"{tok:>6}  {text[:44]}..."
        )
    phase1_ms = (time.perf_counter() - phase1_start) * 1_000

    # ------------------------------------------------------------------
    # Phase 2: process remaining 50 as a batch (batch_get_or_compute)
    # ------------------------------------------------------------------

    print(f"\n{sep}")
    print(f"  PHASE 2 -- Batch processing (remaining {len(_ALL)-LIVE_SHOW} texts, "
          f"batch_get_or_compute)")
    print(sep)

    t0           = time.perf_counter()
    batch_vecs   = cache.batch_get_or_compute(_ALL[LIVE_SHOW:])
    batch_ms     = (time.perf_counter() - t0) * 1_000
    print(f"  Encoded {len(batch_vecs)} texts in {batch_ms:.0f}ms total  "
          f"({batch_ms/len(batch_vecs):.1f}ms/text avg)")
    print(f"  Returned shape: {batch_vecs[0].shape}  dtype={batch_vecs[0].dtype}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    stats = cache.get_stats()

    print(f"\n{sep}")
    print("  CACHE STATISTICS")
    print(sep)
    print(f"\n  {'Metric':<42}  {'Value':>16}")
    print(f"  {thin}")

    def _row(label: str, value: str) -> None:
        print(f"  {label:<42}  {value:>16}")

    _row("Total requests",                str(stats.hits + stats.misses))
    _row("Cache hits",                    str(stats.hits))
    _row("Cache misses",                  str(stats.misses))
    _row("Hit rate",                      f"{stats.hit_rate:.1%}")
    _row("Entries on disk",               str(stats.cache_entries))
    _row("Cache size on disk",            f"{stats.cache_size_bytes / 1024:.1f} KB")
    print(f"  {thin}")
    _row("Bytes saved (embeddings)",      f"{stats.bytes_saved / 1024:.1f} KB")
    _row("Compute time saved (est.)",     f"{stats.compute_time_saved_ms:.0f} ms")
    _row("Avg miss compute time",         f"{stats.avg_miss_compute_time_ms:.1f} ms")
    print(f"  {thin}")
    _row(f"Equiv. OpenAI tokens saved",   f"{stats.openai_tokens_saved:,}")
    _row(f"Equiv. OpenAI cost saved",     f"${stats.openai_cost_saved_usd:.7f}")

    # ------------------------------------------------------------------
    # Savings bar per category
    # ------------------------------------------------------------------

    def _pct_bar(value: float, total: float, width: int = 28) -> str:
        if total == 0:
            return "[" + "-" * width + "]"
        r = min(value / total, 1.0)
        f = round(r * width)
        return "[" + "#" * f + "-" * (width - f) + "]"

    total_req = stats.hits + stats.misses
    print(f"\n  HIT/MISS BREAKDOWN")
    print(f"  {thin}")
    print(f"  {'Hits  '} {_pct_bar(stats.hits,   total_req)}  {stats.hits:>3}/{total_req} "
          f"({stats.hit_rate:.1%})")
    print(f"  {'Misses'} {_pct_bar(stats.misses, total_req)}  {stats.misses:>3}/{total_req} "
          f"({1-stats.hit_rate:.1%})")

    # ------------------------------------------------------------------
    # Monthly savings projection
    # ------------------------------------------------------------------

    # Assume 1000 embedding queries per day (typical small RAG workload)
    proj = cache.monthly_savings_projection(queries_per_day=1_000)

    print(f"\n{sep}")
    print(f"  MONTHLY SAVINGS PROJECTION  (1,000 queries/day, "
          f"OpenAI {_OPENAI_MODEL})")
    print(sep)
    print(f"\n  {'Metric':<42}  {'Value':>16}")
    print(f"  {thin}")
    _row("Session duration",              f"{proj.session_duration_s:.1f}s")
    _row("Session hit rate",              f"{proj.session_hit_rate:.1%}")
    _row("Projected monthly requests",    f"{proj.projected_monthly_requests:,}")
    _row("Projected monthly cache hits",  f"{proj.projected_monthly_hits:,}")
    print(f"  {thin}")
    _row("Monthly cost WITHOUT cache",    f"${proj.monthly_cost_without_cache_usd:.4f}")
    _row("Monthly cost WITH cache",       f"${proj.monthly_cost_with_cache_usd:.4f}")
    _row("Monthly savings",               f"${proj.monthly_cost_without_cache_usd - proj.monthly_cost_with_cache_usd:.4f}")
    _row("Monthly savings %",             f"{proj.monthly_savings_pct:.1f}%")
    _row("OpenAI tokens saved / month",   f"{proj.projected_openai_tokens_saved:,}")
    _row("OpenAI cost saved / month",     f"${proj.projected_openai_cost_saved_usd:.4f}")

    # ------------------------------------------------------------------
    # Demo: verify vectors are identical across calls (correctness check)
    # ------------------------------------------------------------------

    sample_text = _UNIQUE[0]
    v1 = cache.get_or_compute(sample_text)
    v2 = cache.get_or_compute(sample_text)
    identical = np.allclose(v1, v2)

    print(f"\n{sep}")
    print("  CORRECTNESS CHECK")
    print(sep)
    print(f"  Same text, two calls -> vectors identical: {identical}")
    print(f"  Vector shape: {v1.shape}  dtype: {v1.dtype}  "
          f"L2-norm: {float(np.linalg.norm(v1)):.4f}")

    # ------------------------------------------------------------------
    # invalidate_old demo
    # ------------------------------------------------------------------

    stale_count = cache.invalidate_old(days=0)
    final_stats = cache.get_stats()
    print(f"\n  After invalidate_old(days=0): removed {stale_count} entries "
          f"({final_stats.cache_entries} remaining)")
    print(f"\n{sep}\n")
