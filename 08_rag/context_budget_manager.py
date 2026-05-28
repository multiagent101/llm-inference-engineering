"""
context_budget_manager.py - Token-budget-aware context selection for RAG.

Given a pool of retrieved chunks, this module selects the subset that fits
within a fixed token budget while maximising a quality score that rewards
both semantic relevance and proximity to the top of the retrieval ranking.

Scoring formula
---------------
    adjusted_score = relevance_score * (1 - position_penalty)
    position_penalty = (position / max(N-1, 1)) * decay_factor

where ``position`` is the 0-based rank in the retrieval results (0 = most
relevant) and ``decay_factor`` controls how aggressively late-ranked chunks
are discounted (default 0.30 — a chunk at rank N-1 loses 30 % of its score).

Allocation strategy
-------------------
Chunks are sorted by ``adjusted_score`` descending; a greedy loop adds
chunks until the next chunk would overflow the budget.  This is an O(N log N)
approximation of the 0/1 knapsack; it is optimal when chunk token counts are
roughly equal and slightly sub-optimal otherwise.

Three strategies are compared
-------------------------------
1. **Top-k naive**        — take the k highest-relevance chunks regardless
                            of total token count or budget.
2. **Budget manager**     — position-penalised greedy fill up to
                            ``budget_tokens`` using pre-computed relevance
                            scores.
3. **Full reranking**     — fresh sentence-transformer similarity scores,
                            ignoring the original retrieval order, then
                            budget-managed fill.

Cost integration
----------------
Imports ``_llm_cost``, ``_input_price`` from ``rag_cost_analyzer`` when
available; falls back to a built-in pricing table otherwise.
Token counting uses ``tiktoken`` (cl100k_base) — fast, API-free, ~95 %
accurate for Claude models.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Pricing — import from sibling module when available
# ---------------------------------------------------------------------------

try:
    from rag_cost_analyzer import _llm_cost, _input_price, _MODEL_PRICING  # type: ignore
    _PRICING_SOURCE = "rag_cost_analyzer"
except ImportError:
    _MODEL_PRICING: dict[str, dict[str, float]] = {
        "claude-haiku-4-5":  {"input": 0.80 / 1_000_000, "output":  4.00 / 1_000_000},
        "claude-sonnet-4-5": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-opus-4-7":   {"input": 5.00 / 1_000_000, "output": 25.00 / 1_000_000},
    }
    _DEFAULT_IN  = 3.00 / 1_000_000
    _DEFAULT_OUT = 15.00 / 1_000_000

    def _input_price(model: str) -> float:
        return _MODEL_PRICING.get(model, {"input": _DEFAULT_IN})["input"]

    def _llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        p = _MODEL_PRICING.get(model, {"input": _DEFAULT_IN, "output": _DEFAULT_OUT})
        return p["input"] * input_tokens + p["output"] * output_tokens

    _PRICING_SOURCE = "built-in"

_OUTPUT_FILE = Path(__file__).parent / "context_budget_analysis.json"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RankedChunk:
    """
    A retrieved text passage with its retrieval metadata.

    Attributes
    ----------
    text:
        The passage content.
    relevance_score:
        Relevance to the query as assigned by the retrieval system (0–1).
        For dense retrieval this is typically cosine similarity.
    token_count:
        Exact token count measured with tiktoken (cl100k_base).
    source:
        Document identifier (filename, URL, chunk ID, etc.).
    position:
        0-based rank in the retrieval result list (0 = highest relevance).
        Used to compute the position penalty in :meth:`ContextBudgetManager.allocate`.
    adjusted_score:
        Score after applying the position penalty; computed and set by
        :meth:`ContextBudgetManager.allocate`.  Defaults to ``relevance_score``
        before allocation is called.
    """

    text:            str
    relevance_score: float
    token_count:     int
    source:          str
    position:        int
    adjusted_score:  float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if self.adjusted_score == 0.0:
            self.adjusted_score = self.relevance_score


@dataclass
class StrategyResult:
    """
    Outcome of applying one context-selection strategy.

    Attributes
    ----------
    name:
        Human-readable strategy label.
    chunks_selected:
        Number of chunks included in the context.
    tokens_used:
        Sum of ``token_count`` for selected chunks.
    budget_tokens:
        The token budget this strategy was evaluated against.
    utilization_pct:
        ``tokens_used / budget_tokens * 100``.
    avg_relevance:
        Mean ``relevance_score`` of selected chunks.
    max_relevance:
        Highest ``relevance_score`` among selected chunks.
    min_relevance:
        Lowest ``relevance_score`` among selected chunks.
    avg_adjusted_score:
        Mean ``adjusted_score`` (after position penalty) of selected chunks.
    weighted_quality:
        Relevance-weighted quality: ``sum(score * tokens) / tokens_used``.
        Measures how much relevance is obtained per token spent.
    estimated_context_cost_usd:
        Estimated LLM input cost for the context tokens alone.
    selected_chunks:
        The actual :class:`RankedChunk` objects selected (ordered by
        ``adjusted_score`` descending).
    """

    name:                        str
    chunks_selected:             int
    tokens_used:                 int
    budget_tokens:               int
    utilization_pct:             float
    avg_relevance:               float
    max_relevance:               float
    min_relevance:               float
    avg_adjusted_score:          float
    weighted_quality:            float
    estimated_context_cost_usd:  float
    selected_chunks:             list[RankedChunk] = field(default_factory=list)


@dataclass
class StrategyComparison:
    """
    Side-by-side comparison of three context-selection strategies.

    Attributes
    ----------
    query:
        The query used for all strategies.
    budget_tokens:
        Shared token budget.
    model:
        Model used for cost estimation.
    total_candidate_chunks:
        Number of chunks in the input pool.
    total_candidate_tokens:
        Sum of token counts across all candidate chunks.
    results:
        One :class:`StrategyResult` per strategy, in definition order:
        top-k naive, budget manager, full reranking.
    best_strategy:
        Name of the strategy with the highest ``weighted_quality``.
    savings_budget_vs_topk_pct:
        Cost reduction of the budget-manager strategy vs. top-k naive.
        Positive means the budget manager is cheaper.
    savings_rerank_vs_topk_pct:
        Cost reduction of full-reranking strategy vs. top-k naive.
    recommendation:
        Plain-English recommendation string.
    """

    query:                        str
    budget_tokens:                int
    model:                        str
    total_candidate_chunks:       int
    total_candidate_tokens:       int
    results:                      list[StrategyResult]
    best_strategy:                str
    savings_budget_vs_topk_pct:   float
    savings_rerank_vs_topk_pct:   float
    recommendation:               str


# ---------------------------------------------------------------------------
# ContextBudgetManager
# ---------------------------------------------------------------------------


class ContextBudgetManager:
    """
    Token-budget-aware context selector for RAG pipelines.

    Uses a position-penalised relevance score to rank candidates and greedily
    fills a token budget, maximising quality per token spent.

    Parameters
    ----------
    budget_tokens:
        Default token budget for context selection.
    embedding_model:
        HuggingFace sentence-transformers model for computing fresh
        similarity scores in ``full reranking`` strategy.
    position_decay:
        Maximum fractional penalty applied to the last chunk in the ranking.
        0.0 = no decay (pure relevance ordering);
        1.0 = the last chunk has its score zeroed out.
        Default 0.30 means rank-last chunk loses 30 % of its score.
    token_encoding:
        tiktoken encoding name.  ``cl100k_base`` (GPT-4 / Claude approximation)
        gives ~95 % accuracy for Claude models without an API call.
    output_file:
        Where to write JSON analysis reports.
    """

    def __init__(
        self,
        budget_tokens:   int   = 2_000,
        embedding_model: str   = "all-MiniLM-L6-v2",
        position_decay:  float = 0.30,
        token_encoding:  str   = "cl100k_base",
        output_file:     Optional[Path] = None,
    ) -> None:
        self._budget         = budget_tokens
        self._position_decay = position_decay
        self._out_file       = output_file or _OUTPUT_FILE

        print(f"  Loading embedding model '{embedding_model}' ...", end=" ", flush=True)
        self._embedder       = SentenceTransformer(embedding_model)
        print("ready.")

        self._tokenizer      = tiktoken.get_encoding(token_encoding)
        self._encoding_name  = token_encoding

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokens in *text* using tiktoken.

        Parameters
        ----------
        text:
            Any string to measure.

        Returns
        -------
        int
            Token count (>= 1 for non-empty strings).
        """
        return max(1, len(self._tokenizer.encode(text)))

    # ------------------------------------------------------------------
    # Embedding and ranking helpers
    # ------------------------------------------------------------------

    def embed_and_rank(
        self,
        query:   str,
        texts:   list[str],
        sources: Optional[list[str]] = None,
    ) -> list[RankedChunk]:
        """
        Embed *texts* and *query* with sentence-transformers, rank by cosine
        similarity, and return a list of :class:`RankedChunk` sorted by
        descending similarity.

        Parameters
        ----------
        query:
            User question used as the similarity anchor.
        texts:
            Candidate passage strings.
        sources:
            Optional source labels parallel to *texts*.  Defaults to
            ``"chunk-{i}"`` if not supplied.

        Returns
        -------
        list[RankedChunk]
            One entry per text, sorted by ``relevance_score`` descending.
            ``position`` reflects this sorted order (0 = best match).
        """
        if sources is None:
            sources = [f"chunk-{i:02d}" for i in range(len(texts))]

        query_vec  = self._embedder.encode([query], convert_to_numpy=True)
        corpus_mat = self._embedder.encode(texts,   convert_to_numpy=True, show_progress_bar=False)
        sims       = cosine_similarity(query_vec, corpus_mat)[0]

        pairs = sorted(
            zip(sims, texts, sources),
            key=lambda x: x[0],
            reverse=True,
        )

        chunks: list[RankedChunk] = []
        for pos, (score, text, src) in enumerate(pairs):
            chunks.append(RankedChunk(
                text            = text,
                relevance_score = float(score),
                token_count     = self.count_tokens(text),
                source          = src,
                position        = pos,
                adjusted_score  = float(score),
            ))
        return chunks

    # ------------------------------------------------------------------
    # Position penalty and adjusted score
    # ------------------------------------------------------------------

    def _position_penalty(self, position: int, total: int) -> float:
        """
        Compute the fractional score penalty for a chunk at *position*.

        penalty = (position / max(total - 1, 1)) * decay_factor

        Chunk at position 0 gets penalty 0; chunk at position ``total-1``
        gets penalty equal to ``self._position_decay``.

        Parameters
        ----------
        position:
            0-based rank (0 = highest relevance in original ranking).
        total:
            Total number of candidate chunks.

        Returns
        -------
        float
            Penalty in [0, position_decay].
        """
        if total <= 1:
            return 0.0
        return (position / max(total - 1, 1)) * self._position_decay

    def _adjusted_score(self, chunk: RankedChunk, total: int) -> float:
        """
        Return the position-penalised score for *chunk*.

        score = relevance_score * (1 - position_penalty)
        """
        penalty = self._position_penalty(chunk.position, total)
        return chunk.relevance_score * (1.0 - penalty)

    # ------------------------------------------------------------------
    # allocate
    # ------------------------------------------------------------------

    def allocate(
        self,
        query:         str,
        chunks:        list[RankedChunk],
        budget_tokens: Optional[int] = None,
    ) -> list[RankedChunk]:
        """
        Select the best-scoring chunks that collectively fit within the token
        budget, using position-penalised relevance scores.

        Algorithm
        ---------
        1. Compute ``adjusted_score`` for every chunk.
        2. Sort candidates by ``adjusted_score`` descending.
        3. Greedy fill: add each chunk if it fits in the remaining budget.
        4. Return selected chunks ordered by ``adjusted_score`` descending
           (highest-quality context first in the assembled prompt).

        Parameters
        ----------
        query:
            User question (currently unused in scoring; reserved for future
            query-aware weighting extensions).
        chunks:
            Candidate pool produced by the retrieval system.
        budget_tokens:
            Token ceiling.  Defaults to ``self._budget``.

        Returns
        -------
        list[RankedChunk]
            Selected chunks sorted by ``adjusted_score`` descending.
            Each returned chunk has its ``adjusted_score`` field populated.
        """
        budget = budget_tokens if budget_tokens is not None else self._budget
        total  = len(chunks)

        # Compute and stamp adjusted scores
        scored: list[tuple[float, RankedChunk]] = []
        for chunk in chunks:
            adj = self._adjusted_score(chunk, total)
            chunk.adjusted_score = round(adj, 6)
            scored.append((adj, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected:    list[RankedChunk] = []
        tokens_used: int               = 0

        for _, chunk in scored:
            if tokens_used + chunk.token_count <= budget:
                selected.append(chunk)
                tokens_used += chunk.token_count

        return selected

    # ------------------------------------------------------------------
    # compare_strategies
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        query:         str,
        chunks:        list[RankedChunk],
        top_k:         int = 5,
        model:         str = "claude-haiku-4-5",
        budget_tokens: Optional[int] = None,
    ) -> StrategyComparison:
        """
        Compare three context-selection strategies on the same candidate pool.

        Strategies
        ----------
        **Top-k naive**
            Take the *top_k* chunks with the highest ``relevance_score``
            (original retrieval order), ignoring the token budget entirely.

        **Budget manager**
            Position-penalised greedy fill up to ``budget_tokens`` using
            the pre-computed ``relevance_score`` values from the chunks.

        **Full reranking**
            Re-embed all chunk texts with the sentence-transformer, compute
            fresh cosine-similarity scores against the query (ignoring the
            original ``relevance_score`` and ``position`` fields), then apply
            the budget manager to the re-scored pool.

        Parameters
        ----------
        query:
            User question to use as the relevance anchor.
        chunks:
            Candidate pool with pre-computed ``relevance_score`` values.
        top_k:
            Number of chunks for the naive top-k strategy.
        model:
            Anthropic model whose input-token price is used for cost estimates.
        budget_tokens:
            Token budget for the budget-manager and reranking strategies.
            Defaults to ``self._budget``.

        Returns
        -------
        StrategyComparison
            Structured comparison including per-strategy :class:`StrategyResult`
            and a plain-English recommendation.
        """
        budget = budget_tokens if budget_tokens is not None else self._budget
        in_price = _input_price(model)

        def _make_result(
            name:     str,
            selected: list[RankedChunk],
            bgt:      int,
        ) -> StrategyResult:
            if not selected:
                return StrategyResult(
                    name=name, chunks_selected=0, tokens_used=0,
                    budget_tokens=bgt, utilization_pct=0.0,
                    avg_relevance=0.0, max_relevance=0.0, min_relevance=0.0,
                    avg_adjusted_score=0.0, weighted_quality=0.0,
                    estimated_context_cost_usd=0.0, selected_chunks=[],
                )

            tokens_used   = sum(c.token_count     for c in selected)
            rel_scores    = [c.relevance_score     for c in selected]
            adj_scores    = [c.adjusted_score      for c in selected]
            wq            = (
                sum(c.relevance_score * c.token_count for c in selected) / tokens_used
                if tokens_used else 0.0
            )
            cost          = in_price * tokens_used

            return StrategyResult(
                name                       = name,
                chunks_selected            = len(selected),
                tokens_used                = tokens_used,
                budget_tokens              = bgt,
                utilization_pct            = round(tokens_used / bgt * 100, 1) if bgt else 0.0,
                avg_relevance              = round(float(np.mean(rel_scores)),  4),
                max_relevance              = round(float(np.max(rel_scores)),   4),
                min_relevance              = round(float(np.min(rel_scores)),   4),
                avg_adjusted_score         = round(float(np.mean(adj_scores)),  4),
                weighted_quality           = round(wq, 6),
                estimated_context_cost_usd = round(cost, 8),
                selected_chunks            = selected,
            )

        # --- Strategy 1: Top-k naive ---
        topk_sorted   = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        topk_selected = topk_sorted[:top_k]
        # Report utilization against the shared budget so all three strategies
        # are directly comparable (top-k may exceed 100 % when over-budget).
        r_topk = _make_result("top-k naive", topk_selected, budget)

        # --- Strategy 2: Budget manager ---
        # Work on a copy so we don't mutate the caller's adjusted_score fields
        chunks_copy = [
            RankedChunk(c.text, c.relevance_score, c.token_count, c.source, c.position)
            for c in chunks
        ]
        bm_selected = self.allocate(query, chunks_copy, budget_tokens=budget)
        r_budget    = _make_result("budget manager", bm_selected, budget)

        # --- Strategy 3: Full reranking ---
        t0          = time.perf_counter()
        query_vec   = self._embedder.encode([query],                      convert_to_numpy=True)
        corpus_mat  = self._embedder.encode([c.text for c in chunks],     convert_to_numpy=True,
                                             show_progress_bar=False)
        rerank_ms   = (time.perf_counter() - t0) * 1_000
        fresh_sims  = cosine_similarity(query_vec, corpus_mat)[0]

        reranked_pool: list[RankedChunk] = []
        for pos, (chunk, sim) in enumerate(
            sorted(zip(chunks, fresh_sims), key=lambda x: x[1], reverse=True)
        ):
            reranked_pool.append(RankedChunk(
                text            = chunk.text,
                relevance_score = float(sim),
                token_count     = chunk.token_count,
                source          = chunk.source,
                position        = pos,
            ))

        rr_selected = self.allocate(query, reranked_pool, budget_tokens=budget)
        r_rerank    = _make_result("full reranking", rr_selected, budget)
        r_rerank.name = f"full reranking ({rerank_ms:.0f}ms)"

        # --- Comparison ---
        results    = [r_topk, r_budget, r_rerank]
        best       = max(results, key=lambda r: r.weighted_quality)

        topk_cost  = r_topk.estimated_context_cost_usd
        sav_budget = (
            (topk_cost - r_budget.estimated_context_cost_usd) / topk_cost * 100
            if topk_cost > 0 else 0.0
        )
        sav_rerank = (
            (topk_cost - r_rerank.estimated_context_cost_usd) / topk_cost * 100
            if topk_cost > 0 else 0.0
        )

        total_tokens = sum(c.token_count for c in chunks)
        rec_parts    = [
            f"Best strategy: '{best.name}' "
            f"(weighted quality {best.weighted_quality:.4f}).",
        ]
        if sav_budget > 0:
            rec_parts.append(
                f"Budget manager saves {sav_budget:.1f}% vs top-k naive "
                f"({r_topk.tokens_used} -> {r_budget.tokens_used} tokens) "
                f"while using {r_budget.utilization_pct:.0f}% of the {budget}-token budget."
            )
        else:
            rec_parts.append(
                f"Budget manager uses {r_budget.tokens_used} tokens "
                f"({r_budget.utilization_pct:.0f}% of budget)."
            )

        return StrategyComparison(
            query                       = query,
            budget_tokens               = budget,
            model                       = model,
            total_candidate_chunks      = len(chunks),
            total_candidate_tokens      = total_tokens,
            results                     = results,
            best_strategy               = best.name,
            savings_budget_vs_topk_pct  = round(sav_budget, 1),
            savings_rerank_vs_topk_pct  = round(sav_rerank, 1),
            recommendation              = " ".join(rec_parts),
        )

    # ------------------------------------------------------------------
    # save_analysis
    # ------------------------------------------------------------------

    def save_analysis(self, comparison: StrategyComparison) -> Path:
        """
        Persist *comparison* to ``context_budget_analysis.json``.

        The ``selected_chunks`` lists are serialised as chunk previews
        (first 80 chars of text) to keep the file readable.

        Parameters
        ----------
        comparison:
            Result of :meth:`compare_strategies`.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        def _result_dict(r: StrategyResult) -> dict:
            d = asdict(r)
            d["selected_chunks"] = [
                {
                    "source":          c.source,
                    "position":        c.position,
                    "relevance_score": c.relevance_score,
                    "adjusted_score":  c.adjusted_score,
                    "token_count":     c.token_count,
                    "text_preview":    c.text[:80],
                }
                for c in r.selected_chunks
            ]
            return d

        payload = {
            "query":                   comparison.query,
            "budget_tokens":           comparison.budget_tokens,
            "model":                   comparison.model,
            "total_candidate_chunks":  comparison.total_candidate_chunks,
            "total_candidate_tokens":  comparison.total_candidate_tokens,
            "best_strategy":           comparison.best_strategy,
            "savings_budget_vs_topk_pct": comparison.savings_budget_vs_topk_pct,
            "savings_rerank_vs_topk_pct": comparison.savings_rerank_vs_topk_pct,
            "recommendation":          comparison.recommendation,
            "results":                 [_result_dict(r) for r in comparison.results],
        }
        self._out_file.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return self._out_file


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 20 candidate chunks: varied relevance and length
    # Designed so that naive top-k (by relevance alone) picks long high-sim
    # chunks that eat the budget, while the budget manager finds a better mix.
    # ------------------------------------------------------------------

    _QUERY = (
        "How does chunk size affect RAG pipeline cost and what strategies "
        "can reduce the number of tokens injected into the LLM context?"
    )

    # (text, source_label)  — mix of on-topic (high similarity) and off-topic
    _RAW_CHUNKS: list[tuple[str, str]] = [
        # --- Highly relevant, LONG (~150-200 tokens each) ---
        (
            "Chunk size is one of the most impactful hyperparameters in a RAG pipeline. "
            "Smaller chunks (64-128 words) enable precise retrieval but sacrifice surrounding "
            "context; larger chunks (512-1024 words) preserve discourse continuity but inflate "
            "the LLM context window and increase per-query cost proportionally. "
            "Most production teams benchmark three to five sizes on a representative query set "
            "and choose the configuration that maximises answer quality per dollar of LLM spend. "
            "A common outcome is that 256-word chunks offer the best quality/cost ratio for "
            "typical technical documentation.",
            "doc:chunking-guide"
        ),
        (
            "Token cost optimisation in RAG has three main levers. First, reducing chunk size "
            "directly lowers the number of context tokens injected per query — halving the chunk "
            "size roughly halves context-window spend. Second, reducing top-k (the number of "
            "retrieved chunks passed to the LLM) cuts costs proportionally; moving from top-5 "
            "to top-3 typically saves 30-40 % of context tokens with only a small precision "
            "loss. Third, context compression — either extractive (selecting key sentences) or "
            "abstractive (summarising with a small fast model) — can achieve 50-70 % token "
            "reduction while preserving the information needed to answer most queries.",
            "doc:cost-optimisation"
        ),
        (
            "Sliding-window chunking with overlap prevents information loss at chunk boundaries "
            "by sharing 10-20 % of words between consecutive windows. The trade-off is a larger "
            "index and longer embedding time, but retrieval recall — the fraction of relevant "
            "facts that appear in the top-k results — typically improves by 5-12 percentage "
            "points compared with non-overlapping splits. For cost-sensitive pipelines, the "
            "overlap should be minimised to the smallest value that preserves boundary context: "
            "experimentally, 10 % overlap recovers most of the benefit at a fraction of the "
            "index-size overhead.",
            "doc:chunking-overlap"
        ),
        # --- Highly relevant, SHORT (~50-70 tokens) ---
        (
            "Reducing top-k from 5 to 3 saves roughly 35 % of context tokens. "
            "For a 256-word chunk and claude-haiku-4-5 pricing, this cuts per-query "
            "context cost from ~$0.00026 to ~$0.00017 — a saving of $17 per 100k queries.",
            "doc:topk-savings"
        ),
        (
            "Context compression with a small model (e.g., claude-haiku-4-5) before passing "
            "context to a larger model reduces net token cost by 50-70 % on typical RAG queries.",
            "doc:compression-tip"
        ),
        (
            "Prompt caching on the Anthropic API stores repeated system-prompt prefixes at "
            "10 % of the normal input rate, cutting effective context cost by 40-60 % for "
            "workloads where the same document chunks are retrieved repeatedly.",
            "doc:prompt-caching"
        ),
        # --- Moderately relevant, MEDIUM (~80-100 tokens) ---
        (
            "Dense retrieval with sentence-transformers outperforms BM25 on most semantic "
            "benchmarks, but the embedding step adds 10-100 ms of latency per query when "
            "running locally on CPU. For high-throughput pipelines, pre-computing and caching "
            "chunk embeddings at index time eliminates per-query embedding cost entirely; "
            "only the query itself needs to be embedded at inference time.",
            "doc:dense-retrieval"
        ),
        (
            "Hybrid search combines dense vector similarity with sparse BM25 keyword matching "
            "using a weighted sum or reciprocal rank fusion. It consistently outperforms either "
            "modality alone on benchmarks with mixed keyword and semantic queries, at the cost "
            "of maintaining two indexes and a fusion layer.",
            "doc:hybrid-search"
        ),
        (
            "Approximate nearest-neighbour indexes (FAISS, HNSW, ScaNN) scale dense retrieval "
            "to millions of chunks with sub-millisecond query latency. Exact exhaustive search "
            "is faster for corpora under ~100k chunks due to the absence of index build overhead.",
            "doc:ann-indexes"
        ),
        # --- Moderately relevant, SHORT ---
        (
            "Re-ranking the top-20 retrieval results with a cross-encoder before selecting "
            "the final top-3 typically improves precision by 8-15 % over bi-encoder ranking alone.",
            "doc:reranking"
        ),
        (
            "Metadata filtering (date range, document type, author) during retrieval reduces "
            "the effective corpus size and improves both latency and relevance precision.",
            "doc:metadata-filter"
        ),
        # --- Weakly relevant, MEDIUM ---
        (
            "Vector databases such as Pinecone, Weaviate, Qdrant, and Chroma store embeddings "
            "alongside metadata and support filtered ANN search. Managed services eliminate "
            "operational overhead at the cost of a per-query and per-storage fee. Open-source "
            "alternatives run on-premise with zero marginal cost at scale.",
            "doc:vector-stores"
        ),
        (
            "Embedding model selection affects both retrieval quality and index build time. "
            "Larger models (e.g., all-mpnet-base-v2, 768-dim) score 2-4 % higher on BEIR "
            "benchmarks than smaller ones (all-MiniLM-L6-v2, 384-dim) but take 3-5x longer "
            "to embed the same corpus.",
            "doc:embedding-models"
        ),
        # --- Off-topic / low relevance, SHORT ---
        (
            "Kubernetes autoscaling separates the embedding service, vector store, and LLM "
            "gateway into independently scalable components, preventing LLM latency spikes "
            "from blocking the retrieval tier.",
            "doc:k8s-scaling"
        ),
        (
            "Distributed tracing with a propagated trace_id across all RAG service tiers "
            "enables root-cause analysis of latency anomalies without correlating separate logs.",
            "doc:observability"
        ),
        (
            "A/B testing RAG configurations requires hundreds to thousands of paired queries "
            "for statistical significance, depending on effect size and desired power.",
            "doc:ab-testing"
        ),
        # --- Off-topic, LONG (demonstrates budget manager skipping these) ---
        (
            "SLA monitoring for LLM inference tracks p50, p90, and p99 latency percentiles "
            "in a rolling window. Alerts fire on consecutive window violations rather than "
            "single-point anomalies to suppress false positives from transient API blips. "
            "A 5-minute window of 50 samples is sufficient to detect p99 breaches within "
            "two minutes of an incident starting. On-call escalation tiers are: investigate "
            "for p90 breach, escalate for p99 breach, page on-call for sustained p99 > 3x SLA.",
            "doc:sla-monitoring"
        ),
        (
            "RAGAS evaluation measures context precision, context recall, faithfulness, and "
            "answer relevancy. Context precision: fraction of retrieved passages that are "
            "genuinely relevant. Context recall: fraction of relevant facts that appear in "
            "the retrieved set. Both metrics are required for a complete RAG quality picture. "
            "Automated LLM judges (Claude-as-evaluator) correlate well with human ratings "
            "at a fraction of the annotation cost.",
            "doc:ragas-eval"
        ),
        (
            "Blue-green index deployments allow zero-downtime embedding-model upgrades: build "
            "the new index in parallel, validate on a shadow traffic slice, then switch "
            "atomically. Rolling back is instantaneous — simply point the query router back "
            "at the old index. Index version tags must be propagated to all downstream "
            "services to avoid stale-embedding mismatch bugs.",
            "doc:blue-green-index"
        ),
        (
            "Prompt injection via malicious content embedded in retrieved documents is a "
            "security risk specific to RAG systems. Mitigations include: output sanitisation, "
            "sandboxed LLM calls with restricted tool access, and user-identity-based "
            "retrieval-time access control so users can only surface documents they are "
            "authorised to read.",
            "doc:security"
        ),
    ]

    # ------------------------------------------------------------------
    # Helper: text progress bar
    # ------------------------------------------------------------------

    def _bar(filled: int, total: int, width: int = 28) -> str:
        if total == 0:
            return "[" + "-" * width + "]"
        ratio  = min(filled / total, 1.0)
        f      = round(ratio * width)
        return "[" + "#" * f + "-" * (width - f) + "]"

    # ------------------------------------------------------------------
    # Run demo
    # ------------------------------------------------------------------

    sep  = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print("  CONTEXT BUDGET MANAGER DEMO")
    print(sep)

    manager = ContextBudgetManager(
        budget_tokens   = 600,
        embedding_model = "all-MiniLM-L6-v2",
        position_decay  = 0.30,
    )

    print(f"\n  Pricing source : {_PRICING_SOURCE}")
    print(f"  Token encoding : {manager._encoding_name}")
    print(f"  Budget         : {manager._budget} tokens")
    print(f"  Position decay : {manager._position_decay}")
    print(f"  Query          : {_QUERY[:72]}...")

    # --- Build ranked chunks via sentence-transformers ---
    print(f"\n  Embedding {len(_RAW_CHUNKS)} chunks + query ...", end=" ", flush=True)
    t0     = time.perf_counter()
    chunks = manager.embed_and_rank(
        query   = _QUERY,
        texts   = [t for t, _ in _RAW_CHUNKS],
        sources = [s for _, s in _RAW_CHUNKS],
    )
    embed_ms = (time.perf_counter() - t0) * 1_000
    print(f"done ({embed_ms:.0f}ms).")

    total_candidate_tokens = sum(c.token_count for c in chunks)

    # --- Candidate pool table ---
    print(f"\n{sep}")
    print("  CANDIDATE POOL  (sorted by relevance, highest first)")
    print(sep)
    print(f"  {'#':>3}  {'Source':<26}  {'Sim':>6}  {'Toks':>5}  {'Text preview'}")
    print(f"  {thin}")
    for i, c in enumerate(chunks):
        print(
            f"  {i:>3}  {c.source:<26}  {c.relevance_score:>6.4f}  "
            f"{c.token_count:>5}  {c.text[:42]}..."
        )
    print(f"  {thin}")
    print(f"  Total: {len(chunks)} chunks / {total_candidate_tokens} tokens")

    # --- Run strategy comparison ---
    print(f"\n{sep}")
    print("  STRATEGY COMPARISON")
    print(sep)

    comparison = manager.compare_strategies(
        query         = _QUERY,
        chunks        = chunks,
        top_k         = 10,
        model         = "claude-haiku-4-5",
        budget_tokens = 600,
    )

    # Per-strategy summary table
    print(f"\n  {'Strategy':<36}  {'Chunks':>6}  {'Tokens':>7}  {'Budget%':>7}  "
          f"{'AvgSim':>7}  {'WgtQual':>8}  {'Cost$':>12}")
    print(f"  {thin}")
    for r in comparison.results:
        over = "  ** OVER BUDGET **" if r.tokens_used > comparison.budget_tokens else ""
        print(
            f"  {r.name:<36}  {r.chunks_selected:>6}  {r.tokens_used:>7}  "
            f"{r.utilization_pct:>6.1f}%  {r.avg_relevance:>7.4f}  "
            f"{r.weighted_quality:>8.6f}  ${r.estimated_context_cost_usd:>11.8f}{over}"
        )

    # --- Savings summary ---
    topk_cost  = comparison.results[0].estimated_context_cost_usd
    bm_cost    = comparison.results[1].estimated_context_cost_usd
    rr_cost    = comparison.results[2].estimated_context_cost_usd

    print(f"\n{sep}")
    print("  SAVINGS vs TOP-K NAIVE  (context tokens only)")
    print(sep)
    print(f"  {'Metric':<42}  {'Top-k':>10}  {'Budget Mgr':>10}  {'Full Rerank':>12}")
    print(f"  {thin}")
    print(f"  {'Tokens used':<42}  {comparison.results[0].tokens_used:>10}  "
          f"{comparison.results[1].tokens_used:>10}  {comparison.results[2].tokens_used:>12}")
    print(f"  {'Estimated context cost (USD)':<42}  ${topk_cost:>9.8f}  "
          f"${bm_cost:>9.8f}  ${rr_cost:>11.8f}")
    print(f"  {'Savings vs top-k':<42}  {'---':>10}  "
          f"{comparison.savings_budget_vs_topk_pct:>9.1f}%  "
          f"{comparison.savings_rerank_vs_topk_pct:>11.1f}%")
    print(f"  {'Weighted quality':<42}  "
          f"{comparison.results[0].weighted_quality:>10.6f}  "
          f"{comparison.results[1].weighted_quality:>10.6f}  "
          f"{comparison.results[2].weighted_quality:>12.6f}")

    # --- Budget manager: selected chunks detail ---
    bm_result  = comparison.results[1]
    rr_result  = comparison.results[2]

    print(f"\n{sep}")
    print("  BUDGET MANAGER -- SELECTED CHUNKS")
    print(sep)
    tokens_acc = 0
    for i, c in enumerate(bm_result.selected_chunks):
        tokens_acc += c.token_count
        bar = _bar(tokens_acc, comparison.budget_tokens)
        print(
            f"  {i+1:>2}. [{bar}] {tokens_acc:>4}/{comparison.budget_tokens} tok  "
            f"adj={c.adjusted_score:.4f}  sim={c.relevance_score:.4f}  "
            f"{c.source}"
        )
        print(f"      {c.text[:70]}...")

    # --- Full reranking: selected chunks detail ---
    print(f"\n{sep}")
    print("  FULL RERANKING -- SELECTED CHUNKS")
    print(sep)
    tokens_acc = 0
    for i, c in enumerate(rr_result.selected_chunks):
        tokens_acc += c.token_count
        bar = _bar(tokens_acc, comparison.budget_tokens)
        print(
            f"  {i+1:>2}. [{bar}] {tokens_acc:>4}/{comparison.budget_tokens} tok  "
            f"adj={c.adjusted_score:.4f}  sim={c.relevance_score:.4f}  "
            f"{c.source}"
        )
        print(f"      {c.text[:70]}...")

    # --- Recommendation ---
    print(f"\n{sep}")
    print("  RECOMMENDATION")
    print(sep)
    print(f"  Best strategy  : {comparison.best_strategy}")
    print(f"  Budget savings : {comparison.savings_budget_vs_topk_pct:.1f}% "
          f"(budget manager vs top-k naive)")
    print(f"  Rerank savings : {comparison.savings_rerank_vs_topk_pct:.1f}% "
          f"(full reranking vs top-k naive)")
    print()
    import textwrap
    for line in textwrap.wrap(comparison.recommendation, width=70):
        print(f"  {line}")

    # --- Save ---
    out_path = manager.save_analysis(comparison)
    print(f"\n  Analysis saved to: {out_path}")
    print(f"{sep}\n")
