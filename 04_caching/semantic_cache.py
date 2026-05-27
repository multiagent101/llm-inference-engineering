#!/usr/bin/env python3
"""Semantic cache using sentence-transformer embeddings for approximate prompt matching."""

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")

_EMBED_MODEL   = "all-MiniLM-L6-v2"
_EMBED_DIM     = 384
_DEFAULT_TTL   = 86_400        # 24 hours


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CacheHit:
    """A successful semantic cache lookup."""

    response: str
    similarity_score: float     # cosine similarity [0, 1]
    original_prompt: str        # the prompt that was stored in cache
    age_seconds: float          # how old the entry is
    from_exact_match: bool      # True when similarity == 1.0


@dataclass
class SemanticCacheStats:
    """Aggregated runtime statistics."""

    total_requests: int
    semantic_hits: int
    exact_hits: int
    misses: int
    semantic_hit_rate_pct: float
    avg_similarity_score: float    # mean over hit similarity scores
    threshold_used: float
    entries_count: int
    memory_used_mb: float          # embeddings + text, estimated


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Approximate-match LLM response cache backed by dense vector similarity.

    How it works
    ------------
    Each prompt is encoded into a 384-dim embedding via
    ``all-MiniLM-L6-v2``.  On ``get``, the query embedding is compared
    against all stored embeddings using vectorised cosine similarity.  If
    the closest match exceeds ``threshold``, its response is returned as a
    ``CacheHit`` without an API call.

    The full embedding matrix is kept in memory as a NumPy array for fast
    batch similarity computation (O(n × d) per lookup).

    Persistence
    -----------
    Entries and their embeddings are serialised to JSON on every ``set``
    call.  The embedding matrix is reconstructed from the stored float
    lists on load.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        persist_path: str = "semantic_cache.json",
        ttl_seconds: Optional[int] = _DEFAULT_TTL,
        model_name: str = _EMBED_MODEL,
    ) -> None:
        """
        Args:
            threshold: Minimum cosine similarity to accept a cache hit.
            persist_path: JSON file for persistent storage.
            ttl_seconds: Entry lifetime in seconds; ``None`` = never expire.
            model_name: HuggingFace sentence-transformer model name.
        """
        self.threshold      = threshold
        self.persist_path   = Path(persist_path)
        self.ttl_seconds    = ttl_seconds
        self._model_name    = model_name
        self._model: Optional[SentenceTransformer] = None

        # Parallel storage: entries[i] ↔ _emb_matrix[i]
        self._entries: list[dict]         = []
        self._emb_matrix: Optional[np.ndarray] = None   # shape (n, 384)

        # Runtime stats
        self._total_requests  = 0
        self._exact_hits      = 0
        self._semantic_hits   = 0
        self._hit_scores: list[float] = []

        self._load()

    # ------------------------------------------------------------------
    # Model & embedding helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"  [loading {self._model_name}]", flush=True)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts → float32 matrix of shape (len(texts), 384)."""
        return self._get_model().encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)

    def _cosine_sim(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Vectorised cosine similarity between ``query_emb`` and all cached
        embeddings.  Returns array of shape (n,); empty if cache is empty.
        """
        if self._emb_matrix is None or len(self._emb_matrix) == 0:
            return np.array([], dtype=np.float32)

        eps = 1e-10
        q   = query_emb / (np.linalg.norm(query_emb) + eps)
        C   = self._emb_matrix / (
            np.linalg.norm(self._emb_matrix, axis=1, keepdims=True) + eps
        )
        return (C @ q).astype(np.float32)   # shape (n,)

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _is_expired(self, entry: dict) -> bool:
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(entry["created_at"])
        return (self._now() - created).total_seconds() > self.ttl_seconds

    def _age_seconds(self, entry: dict) -> float:
        created = datetime.fromisoformat(entry["created_at"])
        return (self._now() - created).total_seconds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str) -> Optional[CacheHit]:
        """
        Return a cached response for the semantically closest stored prompt.

        The lookup performs:
        1. Embed the query prompt.
        2. Compute cosine similarity against all (non-expired) stored
           embeddings.
        3. If the best match ≥ ``threshold``, return a ``CacheHit``.
           Otherwise return ``None``.

        Args:
            prompt: The query prompt to look up.

        Returns:
            ``CacheHit`` with the cached response and metadata, or ``None``.
        """
        self._total_requests += 1

        if not self._entries:
            return None

        query_emb = self._embed([prompt])[0]
        sims      = self._cosine_sim(query_emb)

        # Filter expired entries from consideration
        valid_mask = np.array(
            [not self._is_expired(e) for e in self._entries], dtype=bool
        )
        if not valid_mask.any():
            return None

        sims[~valid_mask] = -1.0      # exclude expired
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.threshold:
            return None

        entry    = self._entries[best_idx]
        entry["hit_count"] += 1
        is_exact = best_sim > 0.9999

        if is_exact:
            self._exact_hits += 1
        else:
            self._semantic_hits += 1
        self._hit_scores.append(best_sim)

        return CacheHit(
            response=entry["response"],
            similarity_score=round(best_sim, 5),
            original_prompt=entry["prompt"],
            age_seconds=round(self._age_seconds(entry), 1),
            from_exact_match=is_exact,
        )

    def set(
        self,
        prompt: str,
        response: str,
        ttl_override: Optional[int] = None,
    ) -> None:
        """
        Store a prompt-response pair with its embedding.

        If the prompt is already cached (cosine similarity ≥ 0.9999), the
        existing entry is updated in-place instead of creating a duplicate.

        Args:
            prompt: The prompt to cache.
            response: The LLM response to store.
            ttl_override: Per-entry TTL in seconds; falls back to the
                          instance default when ``None``.
        """
        emb = self._embed([prompt])[0]

        # Overwrite near-duplicate entries (exact matches)
        if self._emb_matrix is not None and len(self._emb_matrix) > 0:
            sims = self._cosine_sim(emb)
            if len(sims) > 0:
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) > 0.9999:
                    self._entries[best_idx]["response"]   = response
                    self._entries[best_idx]["created_at"] = self._now().isoformat()
                    self._emb_matrix[best_idx]            = emb
                    self._save()
                    return

        entry = {
            "prompt":     prompt,
            "response":   response,
            "embedding":  emb.tolist(),
            "created_at": self._now().isoformat(),
            "hit_count":  0,
        }
        self._entries.append(entry)
        self._emb_matrix = (
            emb.reshape(1, -1)
            if self._emb_matrix is None
            else np.vstack([self._emb_matrix, emb])
        )
        self._save()

    def get_similar(self, prompt: str, top_k: int = 5) -> list[CacheHit]:
        """
        Return the ``top_k`` most similar cached responses, regardless of
        whether they exceed the threshold.

        Useful for exploring cache contents and debugging threshold choice.

        Args:
            prompt: Query prompt.
            top_k: Maximum number of results to return.

        Returns:
            List of CacheHit objects sorted by descending similarity score.
        """
        if not self._entries:
            return []

        query_emb = self._embed([prompt])[0]
        sims      = self._cosine_sim(query_emb)
        k         = min(top_k, len(sims))
        top_idx   = np.argsort(sims)[-k:][::-1]

        hits: list[CacheHit] = []
        for idx in top_idx:
            entry = self._entries[int(idx)]
            if not self._is_expired(entry):
                hits.append(CacheHit(
                    response=entry["response"],
                    similarity_score=round(float(sims[idx]), 5),
                    original_prompt=entry["prompt"],
                    age_seconds=round(self._age_seconds(entry), 1),
                    from_exact_match=float(sims[idx]) > 0.9999,
                ))
        return hits

    # ------------------------------------------------------------------
    # Threshold optimisation
    # ------------------------------------------------------------------

    def find_optimal_threshold(
        self,
        paraphrase_pairs: list[tuple[str, str]],
        thresholds: Optional[list[float]] = None,
        min_precision: float = 0.90,
    ) -> float:
        """
        Scan a range of thresholds and return the one that maximises F1
        while maintaining ``min_precision``.

        The cache must already be populated with the "ground truth" prompts
        before calling this method.

        Args:
            paraphrase_pairs: List of ``(query, expected_cached_prompt)`` —
                              the query is a paraphrase and the expected
                              cached prompt is what a correct hit should
                              return as ``original_prompt``.
            thresholds: Explicit list of thresholds to evaluate.  Defaults
                        to a linear sweep from 0.60 to 0.98 in 0.02 steps.
            min_precision: A threshold is rejected if its precision falls
                           below this value.

        Returns:
            The threshold with the best F1 (or highest recall among those
            meeting ``min_precision``).
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.60, 0.99, 0.02)]

        SEP  = "=" * 68
        THIN = "-" * 68
        print()
        print(SEP)
        print("  THRESHOLD OPTIMISATION ANALYSIS")
        print(SEP)
        print(f"  Test pairs:    {len(paraphrase_pairs)}")
        print(f"  Thresholds:    {len(thresholds)}  ({thresholds[0]:.2f} -> {thresholds[-1]:.2f})")
        print(f"  Min precision: {min_precision:.0%}")
        print()
        print(
            f"  {'Thresh':>7} {'Hits':>6} {'TP':>5} {'FP':>5} "
            f"{'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}"
        )
        print(f"  {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

        results: list[tuple[float, float, float, float]] = []  # (t, prec, rec, f1)

        for t in thresholds:
            tp = fp = fn = 0
            for query, expected_prompt in paraphrase_pairs:
                query_emb = self._embed([query])[0]
                sims      = self._cosine_sim(query_emb)
                if len(sims) == 0:
                    fn += 1
                    continue
                best_idx  = int(np.argmax(sims))
                best_sim  = float(sims[best_idx])
                if best_sim >= t:
                    matched = self._entries[best_idx]["prompt"]
                    if matched == expected_prompt:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1

            total   = tp + fp + fn
            hits    = tp + fp
            prec    = tp / hits       if hits else 0.0
            rec     = tp / (tp + fn)  if (tp + fn) else 0.0
            f1      = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            results.append((t, prec, rec, f1))

            flag = "  <-- " if prec >= min_precision else ""
            print(
                f"  {t:>7.2f} {hits:>6} {tp:>5} {fp:>5} {fn:>5} "
                f"{prec:>7.3f} {rec:>7.3f} {f1:>7.3f}{flag}"
            )

        # Best F1 among thresholds with sufficient precision
        valid    = [(t, p, r, f) for t, p, r, f in results if p >= min_precision]
        if not valid:
            valid = results
        best     = max(valid, key=lambda x: x[3])
        optimal  = best[0]

        print()
        print(f"  Optimal threshold: {optimal:.2f}  "
              f"(F1={best[3]:.3f}, precision={best[1]:.3f}, recall={best[2]:.3f})")
        print(SEP)
        return optimal

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> SemanticCacheStats:
        """Return an aggregated stats snapshot."""
        total_hits = self._exact_hits + self._semantic_hits
        reqs       = self._total_requests
        hr_pct     = total_hits / reqs * 100 if reqs else 0.0
        avg_sim    = float(np.mean(self._hit_scores)) if self._hit_scores else 0.0

        # Memory estimate: embeddings (float32) + text
        emb_bytes  = (self._emb_matrix.nbytes if self._emb_matrix is not None else 0)
        txt_bytes  = sum(len(e["prompt"]) + len(e["response"]) for e in self._entries)
        mem_mb     = (emb_bytes + txt_bytes) / 1_048_576

        return SemanticCacheStats(
            total_requests=reqs,
            semantic_hits=self._semantic_hits,
            exact_hits=self._exact_hits,
            misses=reqs - total_hits,
            semantic_hit_rate_pct=round(hr_pct, 2),
            avg_similarity_score=round(avg_sim, 4),
            threshold_used=self.threshold,
            entries_count=len(self._entries),
            memory_used_mb=round(mem_mb, 4),
        )

    def print_stats(self) -> None:
        """Print a formatted statistics snapshot to stdout."""
        s   = self.stats()
        SEP = "=" * 52
        print()
        print(SEP)
        print("  SEMANTIC CACHE STATS")
        print(SEP)
        print(f"  Threshold:         {s.threshold_used:.2f}")
        print(f"  Total requests:    {s.total_requests:>10,}")
        print(f"  Semantic hits:     {s.semantic_hits:>10,}")
        print(f"  Exact hits:        {s.exact_hits:>10,}")
        print(f"  Misses:            {s.misses:>10,}")
        print(f"  Hit rate:          {s.semantic_hit_rate_pct:>9.1f}%")
        print(f"  Avg similarity:    {s.avg_similarity_score:>10.4f}")
        print(f"  Entries:           {s.entries_count:>10,}")
        print(f"  Memory (est.):     {s.memory_used_mb:>9.4f} MB")
        print(SEP)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            tmp = self.persist_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
            tmp.replace(self.persist_path)
        except OSError as exc:
            print(f"[semantic_cache] WARNING: {exc}", file=sys.stderr)

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            self._entries = data
            if self._entries:
                embs             = np.array([e["embedding"] for e in self._entries], dtype=np.float32)
                self._emb_matrix = embs
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            self._entries    = []
            self._emb_matrix = None

    def clear(self) -> int:
        """Remove all entries. Returns count removed."""
        count = len(self._entries)
        self._entries.clear()
        self._emb_matrix = None
        self._save()
        return count


# ---------------------------------------------------------------------------
# Test data: 10 base questions × 5 paraphrases = 50 variants
# ---------------------------------------------------------------------------

_BASE_QA: dict[str, str] = {
    "How does transformer attention work?":
        "Transformer attention computes pairwise token relationships using "
        "query, key, and value matrices. Each token attends to all others "
        "via scaled dot-product attention, and multi-head attention runs "
        "several attention operations in parallel.",

    "What is the difference between Python lists and tuples?":
        "Lists are mutable ordered sequences; tuples are immutable. "
        "Tuples are faster for iteration and can be used as dict keys. "
        "Use lists when you need to modify the collection, tuples for fixed data.",

    "How do I implement binary search?":
        "Binary search repeatedly halves the search interval. Compare the "
        "target with the middle element; recurse or iterate on the left or "
        "right half. Time complexity: O(log n). Requires a sorted array.",

    "What is gradient descent?":
        "Gradient descent minimises a loss function by iteratively moving "
        "parameters in the direction of the negative gradient. The learning "
        "rate controls step size. Variants include SGD, Mini-batch, Adam.",

    "How does HTTPS encrypt web traffic?":
        "HTTPS uses TLS to encrypt traffic. A handshake negotiates a cipher "
        "suite and exchanges keys via asymmetric cryptography; subsequent "
        "data uses symmetric encryption for performance.",

    "What is a REST API?":
        "REST is an architectural style for web services. Clients make "
        "stateless HTTP requests (GET, POST, PUT, DELETE) to resource URLs. "
        "Responses are typically JSON. REST enforces uniform interfaces.",

    "How does machine learning work?":
        "ML systems learn patterns from labelled data. A model is trained by "
        "adjusting parameters to minimise prediction error. In supervised "
        "learning, input-output pairs drive this optimisation.",

    "What is Docker?":
        "Docker packages applications and their dependencies into portable "
        "containers. Containers share the host OS kernel but are isolated. "
        "This ensures consistent environments across dev, test, and prod.",

    "How does recursion work?":
        "A recursive function calls itself with a smaller sub-problem until "
        "it hits the base case. Each call is pushed onto the call stack. "
        "Tail recursion can be optimised to avoid stack overflow.",

    "What is the CAP theorem?":
        "CAP states that a distributed system can guarantee at most two of: "
        "Consistency, Availability, Partition Tolerance. In practice, "
        "networks can partition, so systems choose CP or AP trade-offs.",
}

_PARAPHRASES: list[tuple[str, str]] = [
    # Transformer attention (5)
    ("Explain the self-attention mechanism in neural networks.",           "How does transformer attention work?"),
    ("What is multi-head attention in transformer models?",               "How does transformer attention work?"),
    ("How is attention computed in a language model?",                    "How does transformer attention work?"),
    ("Can you describe the query-key-value attention mechanism?",         "How does transformer attention work?"),
    ("Why is the attention mechanism important in deep learning?",        "How does transformer attention work?"),
    # Python lists vs tuples (5)
    ("When should I use a list instead of a tuple in Python?",            "What is the difference between Python lists and tuples?"),
    ("Are Python tuples immutable compared to lists?",                    "What is the difference between Python lists and tuples?"),
    ("What are the performance differences between list and tuple?",      "What is the difference between Python lists and tuples?"),
    ("Which is faster in Python, a list or a tuple?",                     "What is the difference between Python lists and tuples?"),
    ("How do mutable and immutable sequences differ in Python?",          "What is the difference between Python lists and tuples?"),
    # Binary search (5)
    ("Write a binary search function in Python.",                         "How do I implement binary search?"),
    ("What is the time complexity of binary search?",                     "How do I implement binary search?"),
    ("How does bisect work in a sorted array?",                           "How do I implement binary search?"),
    ("Explain the divide-and-conquer approach to searching.",             "How do I implement binary search?"),
    ("How do I efficiently search a sorted list?",                        "How do I implement binary search?"),
    # Gradient descent (5)
    ("How does gradient descent train a neural network?",                 "What is gradient descent?"),
    ("What role does the learning rate play in gradient descent?",        "What is gradient descent?"),
    ("Explain stochastic gradient descent and mini-batch SGD.",           "What is gradient descent?"),
    ("How do we minimise a loss function using gradients?",               "What is gradient descent?"),
    ("What is the difference between SGD and Adam optimizer?",            "What is gradient descent?"),
    # HTTPS (5)
    ("What is TLS and how does it secure web connections?",               "How does HTTPS encrypt web traffic?"),
    ("How does the HTTPS handshake establish a secure channel?",          "How does HTTPS encrypt web traffic?"),
    ("Why is HTTPS more secure than plain HTTP?",                         "How does HTTPS encrypt web traffic?"),
    ("What certificates are used in HTTPS to prove identity?",            "How does HTTPS encrypt web traffic?"),
    ("How does SSL/TLS protect data in transit?",                         "How does HTTPS encrypt web traffic?"),
    # REST API (5)
    ("Explain the principles of RESTful web services.",                   "What is a REST API?"),
    ("How do REST APIs use HTTP methods like GET and POST?",              "What is a REST API?"),
    ("What is the difference between REST and SOAP?",                     "What is a REST API?"),
    ("How do I design a RESTful endpoint?",                               "What is a REST API?"),
    ("What makes an API stateless in the REST sense?",                    "What is a REST API?"),
    # Machine learning (5)
    ("Give me a simple explanation of how ML algorithms work.",           "How does machine learning work?"),
    ("What is supervised learning?",                                      "How does machine learning work?"),
    ("How do computers learn patterns from data?",                        "How does machine learning work?"),
    ("What is the difference between ML and rule-based programming?",     "How does machine learning work?"),
    ("Explain the concept of training a model on labelled examples.",     "How does machine learning work?"),
    # Docker (5)
    ("How do Docker containers differ from virtual machines?",            "What is Docker?"),
    ("What is the purpose of a Dockerfile?",                              "What is Docker?"),
    ("Why do developers containerise applications?",                      "What is Docker?"),
    ("How does Docker provide environment consistency?",                  "What is Docker?"),
    ("What problems does Docker solve in deployment?",                    "What is Docker?"),
    # Recursion (5)
    ("Explain recursive functions with a simple example.",                "How does recursion work?"),
    ("What is a base case and why does recursion need one?",              "How does recursion work?"),
    ("When should I use recursion instead of a loop?",                    "How does recursion work?"),
    ("How does the call stack behave during recursive calls?",            "How does recursion work?"),
    ("What is tail recursion and how does it avoid stack overflow?",      "How does recursion work?"),
    # CAP theorem (5)
    ("Explain consistency, availability, and partition tolerance.",       "What is the CAP theorem?"),
    ("How does CAP theorem apply to distributed databases?",              "What is the CAP theorem?"),
    ("What trade-offs does the CAP theorem force on system designers?",   "What is the CAP theorem?"),
    ("What is the difference between a CP and an AP system?",             "What is the CAP theorem?"),
    ("How does network partitioning affect distributed system design?",   "What is the CAP theorem?"),
]


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    CACHE_FILE = "semantic_cache_demo.json"

    # ----- Build cache with 10 base questions --------------------------------
    if Path(CACHE_FILE).exists():
        os.remove(CACHE_FILE)

    cache = SemanticCache(
        threshold=0.85,
        persist_path=CACHE_FILE,
        ttl_seconds=None,
    )

    SEP = "#" * 68
    print()
    print(SEP)
    print("  SEMANTIC CACHE - 50 paraphrase test")
    print(SEP)
    print(f"  Base questions:  {len(_BASE_QA)}")
    print(f"  Paraphrases:     {len(_PARAPHRASES)}")
    print(f"  Thresholds:      0.75, 0.85, 0.90, 0.95")
    print()

    print("  Loading sentence-transformer and embedding 10 base questions ...")
    for prompt, response in _BASE_QA.items():
        cache.set(prompt, response)
    print(f"  Cache populated: {cache.stats().entries_count} entries\n")

    # ----- Test each threshold -----------------------------------------------
    thresholds_to_test = [0.75, 0.85, 0.90, 0.95]
    results_table: list[tuple[float, int, int, float, float]] = []

    print(f"  {'Threshold':>10} {'Hits/50':>8} {'Correct':>8} {'Hit rate':>10} {'Precision':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for t in thresholds_to_test:
        cache.threshold = t
        hits = correct = 0

        for query, expected_base in _PARAPHRASES:
            hit = cache.get(query)
            if hit is not None:
                hits += 1
                if hit.original_prompt == expected_base:
                    correct += 1

        hit_rate  = hits / len(_PARAPHRASES) * 100
        precision = correct / hits * 100 if hits else 0.0
        results_table.append((t, hits, correct, hit_rate, precision))
        print(
            f"  {t:>10.2f} {hits:>8} {correct:>8} "
            f"{hit_rate:>9.1f}% {precision:>9.1f}%"
        )

        # Reset counters for next threshold test
        cache._total_requests = 0
        cache._semantic_hits  = 0
        cache._exact_hits     = 0
        cache._hit_scores     = []

    # ----- Detailed breakdown at threshold 0.85 ------------------------------
    cache.threshold = 0.85
    print(f"\n  DETAILED BREAKDOWN  (threshold = 0.85)")
    print(f"  {'Base question (short)':<42} {'Hits/5':>7} {'Prec':>7}")
    print(f"  {'-'*42} {'-'*7} {'-'*7}")

    for base_q in _BASE_QA:
        relevant = [(q, b) for q, b in _PARAPHRASES if b == base_q]
        hits = correct = 0
        for q, b in relevant:
            h = cache.get(q)
            if h:
                hits += 1
                if h.original_prompt == b:
                    correct += 1
        prec = correct / hits * 100 if hits else 0.0
        label = base_q[:42]
        print(f"  {label:<42} {hits:>5}/5  {prec:>6.0f}%")

    # ----- get_similar demo --------------------------------------------------
    print(f"\n  get_similar() DEMO  - top 3 matches for a new query")
    print(f"  Query: 'How is attention computed in transformers?'")
    cache.threshold = 0.00   # show all
    similars = cache.get_similar("How is attention computed in transformers?", top_k=3)
    for i, s in enumerate(similars, 1):
        print(f"  {i}. [{s.similarity_score:.4f}] {s.original_prompt[:60]}")

    # ----- find_optimal_threshold --------------------------------------------
    cache.threshold = 0.85
    cache._total_requests = cache._semantic_hits = cache._exact_hits = 0
    cache._hit_scores = []

    optimal = cache.find_optimal_threshold(
        _PARAPHRASES,
        thresholds=[round(t, 2) for t in np.arange(0.70, 0.98, 0.02)],
        min_precision=0.90,
    )
    print(f"\n  Recommended threshold for production: {optimal:.2f}")

    # ----- Final stats -------------------------------------------------------
    cache.threshold = optimal
    for query, _ in _PARAPHRASES:
        cache.get(query)
    cache.print_stats()

    # Cleanup
    if Path(CACHE_FILE).exists():
        os.remove(CACHE_FILE)
