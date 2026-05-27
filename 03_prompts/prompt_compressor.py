#!/usr/bin/env python3
"""Prompt compression using semantic similarity and structural analysis."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Pricing — input USD per token (output is unchanged by compression)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, float] = {
    "claude-haiku-4-5":  0.80e-6,
    "claude-sonnet-4-5": 3.00e-6,
    "claude-opus-4":     15.00e-6,
    "gpt-4o-mini":       0.15e-6,
    "gpt-4o":            2.50e-6,
}

Strategy = Literal[
    "remove_redundancy",
    "compress_instructions",
    "prune_context",
    "abbreviate_examples",
    "auto",
]

# Verbose sentence prefixes stripped by compress_instructions
_VERBOSE_PREFIXES: list[tuple[str, str]] = [
    (r"^Please ",            ""),
    (r"^Kindly ",            ""),
    (r"^I want you to ",     ""),
    (r"^I need you to ",     ""),
    (r"^I would like you to ", ""),
    (r"^You should ",        ""),
    (r"^Make sure (?:to |that )?",           ""),
    (r"^Always make sure (?:to |that )?",    ""),
    (r"^Remember (?:to |that )?",            ""),
    (r"^Always remember (?:to |that )?",     ""),
    (r"^Don't forget (?:to )?",              ""),
    (r"^Note that ",                         ""),
    (r"^It is (?:very |extremely |highly |absolutely )?"
     r"(?:important|crucial|critical|essential) that (?:you )?", ""),
]

# Full sentences that are boilerplate and should be dropped entirely
_BOILERPLATE_SENTENCES: list[str] = [
    r"(?:Feel free to )?(?:let me know|please let me know) if you (?:have (?:any )?)?(?:questions|need clarification)\.?",
    r"(?:Do not hesitate|don't hesitate) to (?:ask|reach out)\.?",
    r"I(?:'m| am) here to help\.?",
    r"Thank you for your (?:understanding|patience|cooperation)\.?",
    r"I hope this (?:helps|is helpful|clarifies things)\.?",
]
_BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _BOILERPLATE_SENTENCES),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Quantitative comparison of original vs compressed text."""

    original_tokens: int
    compressed_tokens: int
    token_reduction_pct: float
    chars_original: int
    chars_compressed: int
    chars_reduction_pct: float
    semantic_similarity: float             # 0–1; 1 = identical meaning
    cost_savings_usd: dict[str, float]     # model_id → USD saved per 1 000 calls


@dataclass
class CompressionResult:
    """Full output of one compression run."""

    strategy: str
    original_text: str
    compressed_text: str
    benchmark: BenchmarkResult


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class PromptCompressor:
    """
    Reduces prompt length while preserving semantic meaning, using four
    complementary strategies.

    Strategies
    ----------
    remove_redundancy
        Drops sentences whose cosine similarity to any already-kept sentence
        exceeds ``similarity_threshold`` (default 0.85).

    compress_instructions
        Strips verbose prefixes ("Please", "Make sure to", "It is crucial
        that you"), boilerplate closing sentences, then applies
        ``remove_redundancy`` with a slightly looser threshold (0.80).

    prune_context
        Given a query, keeps only the ``keep_top_n`` sentences most
        semantically similar to the query (useful for RAG contexts).

    abbreviate_examples
        Detects few-shot example blocks, then selects the ``keep_n`` most
        *diverse* examples using greedy max-min cosine distance.

    auto
        Heuristic selection: checks text structure and length to pick the
        most appropriate strategy.
    """

    _EMBED_MODEL = "all-MiniLM-L6-v2"
    _TIKTOKEN_ENC = "cl100k_base"

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        keep_top_n: int = 10,
        keep_examples: int = 3,
    ) -> None:
        """
        Args:
            similarity_threshold: Cosine similarity above which a sentence is
                                   considered redundant (``remove_redundancy``).
            keep_top_n: Maximum sentences retained by ``prune_context``.
            keep_examples: Maximum examples retained by ``abbreviate_examples``.
        """
        self.similarity_threshold = similarity_threshold
        self.keep_top_n            = keep_top_n
        self.keep_examples         = keep_examples

        self._model: Optional[SentenceTransformer] = None
        self._enc:   Optional[tiktoken.Encoding]   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"  [loading sentence-transformer: {self._EMBED_MODEL}]")
            self._model = SentenceTransformer(self._EMBED_MODEL)
        return self._model

    def _get_enc(self) -> tiktoken.Encoding:
        if self._enc is None:
            self._enc = tiktoken.get_encoding(self._TIKTOKEN_ENC)
        return self._enc

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._get_model().encode(texts, show_progress_bar=False)

    def _count_tokens(self, text: str) -> int:
        return len(self._get_enc().encode(text))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into individual sentences, respecting bullet points and newlines."""
        sentences: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Further split long lines on sentence boundaries
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\'])', line)
            sentences.extend(p.strip() for p in parts if p.strip())
        return sentences

    @staticmethod
    def _preview(text: str, width: int = 70) -> str:
        flat = text.replace("\n", " ")
        return flat[:width] + "..." if len(flat) > width else flat

    # ------------------------------------------------------------------
    # Strategy: remove_redundancy
    # ------------------------------------------------------------------

    def remove_redundancy(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Remove sentences that are semantically redundant.

        For each sentence (in order), it is dropped when its cosine
        similarity to any already-kept sentence exceeds ``threshold``.

        Args:
            text: Input text.
            threshold: Override ``self.similarity_threshold`` for this call.

        Returns:
            Deduplicated text with original sentence order preserved.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text

        thr = threshold if threshold is not None else self.similarity_threshold
        embeddings = self._embed(sentences)

        kept: list[int] = [0]
        for i in range(1, len(sentences)):
            kept_embs = embeddings[kept]
            sims = cosine_similarity(embeddings[i : i + 1], kept_embs)[0]
            if float(np.max(sims)) < thr:
                kept.append(i)

        separator = "\n" if "\n" in text else " "
        return separator.join(sentences[i] for i in kept)

    # ------------------------------------------------------------------
    # Strategy: compress_instructions
    # ------------------------------------------------------------------

    def compress_instructions(self, text: str) -> str:
        """
        Compress system/instruction prompts by removing verbose language.

        Steps
        -----
        1. Remove known boilerplate closing sentences entirely.
        2. Strip verbose prefixes from each remaining sentence.
        3. Apply ``remove_redundancy`` with threshold 0.80 to drop
           semantically equivalent instructions.

        Args:
            text: System prompt or instruction block.

        Returns:
            Shortened text that retains the original intent.
        """
        lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue

            # Drop full-line boilerplate
            if _BOILERPLATE_RE.fullmatch(stripped):
                continue

            # Strip verbose prefixes (iteratively, in case multiple apply)
            changed = True
            while changed:
                changed = False
                for pattern, replacement in _VERBOSE_PREFIXES:
                    new = re.sub(pattern, replacement, stripped, flags=re.IGNORECASE)
                    if new != stripped:
                        stripped = new.strip()
                        # Capitalise first letter after stripping prefix
                        if stripped:
                            stripped = stripped[0].upper() + stripped[1:]
                        changed = True

            # Remove inline boilerplate fragments
            stripped = _BOILERPLATE_RE.sub("", stripped).strip()
            if stripped:
                lines.append(stripped)

        compressed = "\n".join(lines).strip()
        # Final pass: deduplicate semantically similar instructions
        return self.remove_redundancy(compressed, threshold=0.80)

    # ------------------------------------------------------------------
    # Strategy: prune_context
    # ------------------------------------------------------------------

    def prune_context(
        self,
        text: str,
        query: str,
        keep_top_n: Optional[int] = None,
    ) -> str:
        """
        Keep only the sentences most relevant to a query.

        Each sentence is scored by its cosine similarity to the query
        embedding.  The top-N highest-scoring sentences are returned in
        their original order.

        Args:
            text: Long context to prune (e.g. RAG retrieved passages).
            query: The question or task the context should serve.
            keep_top_n: Override ``self.keep_top_n`` for this call.

        Returns:
            Pruned context containing only the most relevant sentences.
        """
        sentences = self._split_sentences(text)
        n = keep_top_n if keep_top_n is not None else self.keep_top_n
        if len(sentences) <= n:
            return text

        all_texts   = sentences + [query]
        embeddings  = self._embed(all_texts)
        sent_embs   = embeddings[:-1]
        query_emb   = embeddings[-1:]

        scores      = cosine_similarity(query_emb, sent_embs)[0]
        top_indices = sorted(np.argsort(scores)[-n:].tolist())  # keep original order
        separator   = "\n" if "\n" in text else " "
        return separator.join(sentences[i] for i in top_indices)

    # ------------------------------------------------------------------
    # Strategy: abbreviate_examples
    # ------------------------------------------------------------------

    def abbreviate_examples(
        self,
        text: str,
        keep_n: Optional[int] = None,
    ) -> str:
        """
        Reduce few-shot examples to the most *diverse* subset.

        Detects example blocks (separated by blank lines, numbered lists,
        or ``Example N:`` / ``Input:`` / ``Output:`` patterns) and selects
        ``keep_n`` examples that maximise coverage of the example space via
        a greedy max-min cosine-distance algorithm.

        Args:
            text: Text containing multiple few-shot examples.
            keep_n: Override ``self.keep_examples`` for this call.

        Returns:
            Text with redundant examples removed.
        """
        n = keep_n if keep_n is not None else self.keep_examples
        examples = self._parse_examples(text)
        if len(examples) <= n:
            return text

        embeddings = self._embed(examples)
        selected   = self._select_diverse(embeddings, n)
        return "\n\n".join(examples[i] for i in selected)

    def _parse_examples(self, text: str) -> list[str]:
        """Split text into individual example blocks using multiple heuristics."""
        # Heuristic 1: "Example N:" markers
        if re.search(r"^Example\s+\d+[:\.]", text, re.MULTILINE | re.IGNORECASE):
            blocks = re.split(r"(?=^Example\s+\d+[:\.])", text, flags=re.MULTILINE | re.IGNORECASE)
            result = [b.strip() for b in blocks if b.strip()]
            if len(result) >= 2:
                return result

        # Heuristic 2: Blank-line separated blocks (most common in few-shot)
        blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
        if len(blocks) >= 3:
            return blocks

        # Heuristic 3: Numbered list items "1. ..." "2. ..."
        items = re.findall(r"(?:^|\n)\d+\.\s+.+?(?=\n\d+\.|\Z)", text, re.DOTALL)
        if len(items) >= 2:
            return [item.strip() for item in items]

        # Heuristic 4: Input/Output pairs
        pairs = re.split(r"(?=\bInput:)", text, flags=re.IGNORECASE)
        if len(pairs) >= 3:
            return [p.strip() for p in pairs if p.strip()]

        return [text]

    @staticmethod
    def _select_diverse(embeddings: np.ndarray, n: int) -> list[int]:
        """
        Greedy max-min selection: each new example is the one least similar
        to any already-selected example (maximises diversity).
        """
        n = min(n, len(embeddings))
        selected = [0]

        while len(selected) < n:
            remaining = [i for i in range(len(embeddings)) if i not in selected]
            if not remaining:
                break
            sel_embs = embeddings[selected]
            # For each candidate, find its max similarity to any selected example
            max_sims = [
                float(np.max(cosine_similarity(embeddings[i : i + 1], sel_embs)))
                for i in remaining
            ]
            # Pick the candidate with the LOWEST max similarity (most diverse)
            best = remaining[int(np.argmin(max_sims))]
            selected.append(best)

        return sorted(selected)

    # ------------------------------------------------------------------
    # Strategy detection
    # ------------------------------------------------------------------

    def _detect_strategy(self, text: str) -> str:
        """Heuristic strategy selection based on text structure."""
        # Few-shot examples
        if re.search(
            r"(?:^Example\s+\d+[:\.]|^Input:\s|^\d+\.\s+Input:)",
            text, re.MULTILINE | re.IGNORECASE,
        ):
            return "abbreviate_examples"

        # Blank-line separated blocks with >= 4 blocks (likely examples)
        blocks = [b for b in text.split("\n\n") if b.strip()]
        if len(blocks) >= 4 and all(len(b.split()) < 150 for b in blocks):
            return "abbreviate_examples"

        # System prompt / instruction-heavy text
        if re.search(
            r"^(?:You are|Your role|Your task|Your goal|As an? |As the )",
            text, re.MULTILINE | re.IGNORECASE,
        ):
            return "compress_instructions"

        # Long context (likely RAG passages)
        if len(text.split()) > 300:
            return "prune_context"

        return "remove_redundancy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        text: str,
        strategy: Strategy = "auto",
        query: str = "",
    ) -> CompressionResult:
        """
        Compress ``text`` using the specified strategy.

        Args:
            text: The prompt to compress.
            strategy: One of the four strategies or ``"auto"`` for automatic
                      selection.  ``"prune_context"`` requires a non-empty
                      ``query``.
            query: Relevance anchor for ``prune_context``.  When ``strategy``
                   is ``"auto"`` and the text is long, the first sentence of
                   the text is used as a fallback query.

        Returns:
            CompressionResult with compressed text and benchmark metrics.
        """
        resolved = strategy if strategy != "auto" else self._detect_strategy(text)

        if resolved == "remove_redundancy":
            compressed = self.remove_redundancy(text)
        elif resolved == "compress_instructions":
            compressed = self.compress_instructions(text)
        elif resolved == "prune_context":
            effective_query = query or self._split_sentences(text)[0]
            compressed = self.prune_context(text, effective_query)
        elif resolved == "abbreviate_examples":
            compressed = self.abbreviate_examples(text)
        else:
            raise ValueError(f"Unknown strategy '{resolved}'.")

        return CompressionResult(
            strategy=resolved,
            original_text=text,
            compressed_text=compressed,
            benchmark=self.benchmark(text, compressed),
        )

    def benchmark(self, original: str, compressed: str) -> BenchmarkResult:
        """
        Compute quantitative metrics comparing original and compressed text.

        Metrics
        -------
        * Token counts (tiktoken cl100k_base).
        * Token and character reduction percentages.
        * Semantic similarity — cosine similarity of whole-text embeddings.
        * Cost savings per model for 1 000 API calls (input tokens only).

        Args:
            original: The unmodified prompt.
            compressed: The compressed prompt.

        Returns:
            BenchmarkResult with all metrics populated.
        """
        orig_tok = self._count_tokens(original)
        comp_tok = self._count_tokens(compressed)
        saved    = orig_tok - comp_tok
        tok_pct  = saved / orig_tok * 100 if orig_tok else 0.0

        orig_chars = len(original)
        comp_chars = len(compressed)
        char_pct   = (orig_chars - comp_chars) / orig_chars * 100 if orig_chars else 0.0

        # Semantic similarity between full texts
        embs = self._embed([original, compressed])
        sim  = float(cosine_similarity(embs[:1], embs[1:])[0][0])

        cost_savings = {
            model: saved * price * 1_000  # per 1 000 calls
            for model, price in MODEL_PRICING.items()
        }

        return BenchmarkResult(
            original_tokens=orig_tok,
            compressed_tokens=comp_tok,
            token_reduction_pct=round(tok_pct, 2),
            chars_original=orig_chars,
            chars_compressed=comp_chars,
            chars_reduction_pct=round(char_pct, 2),
            semantic_similarity=round(sim, 4),
            cost_savings_usd=cost_savings,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, result: CompressionResult) -> None:
        """Print a formatted before/after report to stdout."""
        SEP  = "=" * 72
        THIN = "-" * 72
        b    = result.benchmark

        print()
        print(SEP)
        print(f"  COMPRESSION REPORT  [ strategy: {result.strategy} ]")
        print(SEP)
        print(f"  Original tokens:   {b.original_tokens:>8,}")
        print(f"  Compressed tokens: {b.compressed_tokens:>8,}  ({b.token_reduction_pct:+.1f}%)")
        print(f"  Original chars:    {b.chars_original:>8,}")
        print(f"  Compressed chars:  {b.chars_compressed:>8,}  ({b.chars_reduction_pct:+.1f}%)")
        print(f"  Semantic similarity: {b.semantic_similarity:.4f}  (1.0 = identical meaning)")
        print()
        print(f"  Cost savings per 1,000 calls (input tokens only):")
        print(f"  {'-'*36}")
        for model, saving in b.cost_savings_usd.items():
            print(f"    {model:<22}  ${saving:.4f}")
        print()
        print(f"  ORIGINAL  ({b.original_tokens} tokens)")
        print(THIN)
        preview_orig = result.original_text[:300].replace("\n", " | ")
        print(f"  {preview_orig}{'...' if len(result.original_text) > 300 else ''}")
        print()
        print(f"  COMPRESSED  ({b.compressed_tokens} tokens)")
        print(THIN)
        preview_comp = result.compressed_text[:300].replace("\n", " | ")
        print(f"  {preview_comp}{'...' if len(result.compressed_text) > 300 else ''}")
        print(SEP)


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a helpful AI coding assistant. I want you to help developers write better code.
Please make sure to always write clean, well-documented code in all your responses.
It is very important that you follow best practices and established design patterns.
Please ensure that all code examples you provide are complete and immediately runnable.
You should always explain your reasoning clearly before writing any code.
Make sure to always add proper type hints to all Python functions and methods.
It is crucial that you check for edge cases in all your implementations.
Please remember to handle errors appropriately using try/except blocks where needed.
Don't forget to add docstrings to all functions, methods, and classes you write.
Always make sure your code is well-tested and provably correct before sharing it.
Remember that clean, readable code is always better than overly clever code.
Please try to keep your responses concise and to the point while still being complete.
You should provide working examples whenever possible to illustrate your explanations.
It is important that you follow the principle of least surprise in your implementations.
Make sure to consider performance implications when suggesting data structures.
Please always validate inputs and provide meaningful error messages to the user.
Let me know if you have any questions about what I am asking for at any time.
Feel free to ask for clarification if any part of the request is unclear.
"""

_RAG_CONTEXT = """\
The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair.
It was designed by Gustave Eiffel and stands 330 meters tall including its broadcast antenna.
The tower receives approximately 7 million visitors per year, making it the most-visited paid monument in the world.

Python is a high-level, interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
The language features dynamic typing and automatic memory management through garbage collection.

Transformer attention was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.
The self-attention mechanism allows models to weigh the importance of different tokens when processing a sequence.
Multi-head attention runs several attention operations in parallel, each learning different relationships.
The key-query-value formulation computes attention scores by taking the dot product of queries and keys.
Positional encodings are added to token embeddings to give the model information about token order.
The transformer architecture has become the foundation of modern large language models like GPT and Claude.

The Amazon rainforest covers approximately 5.5 million square kilometres across nine countries.
It produces 20 percent of the world's oxygen and is home to 10 percent of all species on Earth.
Deforestation rates have been increasing due to agricultural expansion and logging activities.
Climate scientists warn that loss of the Amazon could trigger irreversible tipping points in Earth's climate.

Gradient descent is an optimisation algorithm used to minimise a loss function in machine learning.
Stochastic gradient descent (SGD) updates model weights using a single sample or small mini-batch.
The learning rate controls how large each parameter update step is during training.
Adaptive optimisers like Adam combine momentum and per-parameter learning rates for faster convergence.
Batch normalisation stabilises training by normalising layer inputs to have zero mean and unit variance.
"""

_QUERY_FOR_RAG = "How does transformer attention work and what are its key components?"

_REDUNDANT_DOC = """\
Our API uses token-based authentication for all requests.
Token-based auth is the primary security mechanism for our REST API.
Every API call must include a valid authentication token in the request header.
You need to include an auth token with each request to our API endpoints.
The system uses tokens to verify the identity of API callers.
Authentication tokens expire after 24 hours and must be refreshed.
Tokens have a 24-hour lifetime and need renewal after expiry.
After 24 hours your token will expire and you will need to get a new one.
Rate limiting applies to all authenticated API endpoints.
The API enforces rate limits on authenticated requests.
All endpoints are subject to rate limiting per user account.
Errors are returned as JSON objects with a message and status code.
API errors are formatted as JSON with status and message fields.
When an error occurs the response is a JSON object containing error details.
"""

_FEW_SHOT_EXAMPLES = """\
Example 1:
Input: "The movie was absolutely fantastic, I loved every minute of it!"
Output: {"sentiment": "positive", "confidence": 0.98, "emotions": ["joy", "excitement"]}

Example 2:
Input: "It was okay I guess, nothing special but not terrible either."
Output: {"sentiment": "neutral", "confidence": 0.82, "emotions": ["indifference"]}

Example 3:
Input: "This film was great, I really enjoyed watching it from start to finish."
Output: {"sentiment": "positive", "confidence": 0.95, "emotions": ["joy", "satisfaction"]}

Example 4:
Input: "I hated every second of this movie, total waste of money and time."
Output: {"sentiment": "negative", "confidence": 0.99, "emotions": ["anger", "disappointment"]}

Example 5:
Input: "Not bad, had some good moments but also some slow parts."
Output: {"sentiment": "neutral", "confidence": 0.76, "emotions": ["mild_approval"]}

Example 6:
Input: "Absolutely loved it, one of the best movies I have ever seen in my life!"
Output: {"sentiment": "positive", "confidence": 0.97, "emotions": ["joy", "excitement", "admiration"]}
"""


if __name__ == "__main__":
    compressor = PromptCompressor(
        similarity_threshold=0.85,
        keep_top_n=8,
        keep_examples=3,
    )

    test_cases: list[tuple[str, str, str, str]] = [
        ("SYSTEM PROMPT (verbose instructions)",
         _SYSTEM_PROMPT, "compress_instructions", ""),
        ("RAG CONTEXT (prune to query)",
         _RAG_CONTEXT, "prune_context", _QUERY_FOR_RAG),
        ("REDUNDANT DOCUMENT",
         _REDUNDANT_DOC, "remove_redundancy", ""),
        ("FEW-SHOT EXAMPLES (abbreviate)",
         _FEW_SHOT_EXAMPLES, "abbreviate_examples", ""),
    ]

    SEP = "#" * 72
    print()
    print(SEP)
    print("  PROMPT COMPRESSOR — 4 strategies demonstrated")
    print(SEP)

    for label, text, strategy, query in test_cases:
        print(f"\n>>> {label}")
        result = compressor.compress(text, strategy=strategy, query=query)
        compressor.print_report(result)

    # Auto-strategy demo
    print(f"\n>>> AUTO-STRATEGY SELECTION on 4 prompts")
    print(SEP)
    print(f"  {'Prompt':<40} {'Detected strategy':<25} {'Token reduction':>16}")
    print(f"  {'-'*40} {'-'*25} {'-'*16}")
    for label, text, _, query in test_cases:
        r = compressor.compress(text, strategy="auto", query=query)
        print(
            f"  {label[:40]:<40} {r.strategy:<25} "
            f"{r.benchmark.token_reduction_pct:>+14.1f}%"
        )
    print(SEP)
