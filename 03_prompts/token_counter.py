#!/usr/bin/env python3
"""Token counting and cost estimation across Anthropic and OpenAI models."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import tiktoken
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Pricing and tokenizer configuration for one model."""

    model_id: str
    provider: str               # "anthropic" | "openai"
    input_usd_per_token: float
    output_usd_per_token: float
    encoding: str               # tiktoken encoding; empty for Anthropic models


# All Claude models share the same tokenizer; OpenAI models vary.
MODELS: dict[str, ModelSpec] = {
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-haiku-4-5": ModelSpec(
        "claude-haiku-4-5",  "anthropic", 0.80 / 1_000_000,  4.00 / 1_000_000, "",
    ),
    "claude-sonnet-4-5": ModelSpec(
        "claude-sonnet-4-5", "anthropic", 3.00 / 1_000_000, 15.00 / 1_000_000, "",
    ),
    "claude-opus-4": ModelSpec(
        "claude-opus-4",     "anthropic", 15.00 / 1_000_000, 75.00 / 1_000_000, "",
    ),
    # ── OpenAI (token counts via tiktoken — no OpenAI API key required) ──
    "gpt-4o": ModelSpec(
        "gpt-4o",      "openai",  2.50 / 1_000_000, 10.00 / 1_000_000, "o200k_base",
    ),
    "gpt-4o-mini": ModelSpec(
        "gpt-4o-mini", "openai",  0.15 / 1_000_000,  0.60 / 1_000_000, "o200k_base",
    ),
    "gpt-4-turbo": ModelSpec(
        "gpt-4-turbo", "openai", 10.00 / 1_000_000, 30.00 / 1_000_000, "cl100k_base",
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CountResult:
    """Token count for a single text / provider pair."""

    text_preview: str
    char_count: int
    provider: str
    token_count: int
    from_cache: bool


@dataclass
class CostEstimate:
    """Cost projection for a single model given a fixed output estimate."""

    model_id: str
    provider: str
    input_tokens: int
    estimated_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


@dataclass
class ModelComparison:
    """Full cross-model comparison for one input text."""

    text_preview: str
    char_count: int
    estimated_output_tokens: int
    estimates: list[CostEstimate]
    cheapest_model: str


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Counts tokens and estimates costs across Anthropic and OpenAI models.

    Anthropic token counts come from the official ``count_tokens`` API
    endpoint (exact, same tokenizer for all Claude models).  OpenAI counts
    are computed locally via ``tiktoken`` (no OpenAI API key required).

    Results are cached in memory by ``(tokenizer, sha256(text))`` to avoid
    redundant API calls or re-encoding of repeated texts.
    """

    # Anthropic models all share the same tokenizer; use the cheapest to call.
    _COUNT_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_output_tokens: int = 500,
    ) -> None:
        """
        Args:
            api_key: Anthropic API key.  Reads ``ANTHROPIC_API_KEY`` from
                     the environment if omitted.
            default_output_tokens: Assumed output length used when estimating
                                   costs without a known response size.
        """
        self.default_output_tokens = default_output_tokens
        self._client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        # Lazy-load tiktoken encodings so startup is fast when only using Anthropic
        self._encoders: dict[str, tiktoken.Encoding] = {}
        # Cache: key = (tokenizer_id, text_sha256) → token_count
        self._cache: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _encoder(self, encoding_name: str) -> tiktoken.Encoding:
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        return self._encoders[encoding_name]

    def _count_anthropic(self, text: str) -> int:
        """Count tokens via the Anthropic API (cached)."""
        cache_key = ("anthropic", self._sha256(text))
        if cache_key in self._cache:
            return self._cache[cache_key]
        response = self._client.messages.count_tokens(
            model=self._COUNT_MODEL,
            messages=[{"role": "user", "content": text}],
        )
        count = response.input_tokens
        self._cache[cache_key] = count
        return count

    def _count_tiktoken(self, text: str, encoding: str) -> int:
        """Count tokens via tiktoken (local, cached)."""
        cache_key = (encoding, self._sha256(text))
        if cache_key in self._cache:
            return self._cache[cache_key]
        count = len(self._encoder(encoding).encode(text))
        self._cache[cache_key] = count
        return count

    def _count_for_spec(self, text: str, spec: ModelSpec) -> int:
        """Dispatch to the correct counting backend for a ModelSpec."""
        if spec.provider == "anthropic":
            return self._count_anthropic(text)
        return self._count_tiktoken(text, spec.encoding)

    @staticmethod
    def _preview(text: str, width: int = 60) -> str:
        flat = text.replace("\n", " ")
        return flat[:width] + "..." if len(flat) > width else flat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, text: str, provider: str = "anthropic") -> CountResult:
        """
        Count tokens for ``text`` using the specified provider's tokenizer.

        For ``"anthropic"`` the result comes from the API endpoint (all Claude
        models share one tokenizer).  For ``"openai"`` the ``cl100k_base``
        encoding is used (GPT-4 / GPT-3.5 family).  Results are cached.

        Args:
            text: Input text to count.
            provider: ``"anthropic"`` or ``"openai"``.

        Returns:
            CountResult with the token count and cache status.

        Raises:
            ValueError: If ``provider`` is not recognised.
        """
        if provider == "anthropic":
            cache_key = ("anthropic", self._sha256(text))
            was_cached = cache_key in self._cache
            token_count = self._count_anthropic(text)
        elif provider == "openai":
            encoding = "cl100k_base"
            cache_key = (encoding, self._sha256(text))
            was_cached = cache_key in self._cache
            token_count = self._count_tiktoken(text, encoding)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")

        return CountResult(
            text_preview=self._preview(text),
            char_count=len(text),
            provider=provider,
            token_count=token_count,
            from_cache=was_cached,
        )

    def estimate_cost(
        self,
        text: str,
        model: str,
        estimated_output_tokens: Optional[int] = None,
    ) -> CostEstimate:
        """
        Estimate the API cost for ``text`` on a specific model.

        Args:
            text: Input prompt text.
            model: Model ID from ``MODELS`` (e.g. ``"claude-haiku-4-5"``).
            estimated_output_tokens: Expected response length in tokens.
                                     Falls back to ``default_output_tokens``.

        Returns:
            CostEstimate with per-component and total USD cost.

        Raises:
            KeyError: If ``model`` is not in the catalog.
        """
        if model not in MODELS:
            raise KeyError(f"Unknown model '{model}'. Available: {list(MODELS)}")

        spec = MODELS[model]
        out_tokens = estimated_output_tokens or self.default_output_tokens
        in_tokens = self._count_for_spec(text, spec)

        return CostEstimate(
            model_id=model,
            provider=spec.provider,
            input_tokens=in_tokens,
            estimated_output_tokens=out_tokens,
            input_cost_usd=in_tokens * spec.input_usd_per_token,
            output_cost_usd=out_tokens * spec.output_usd_per_token,
            total_cost_usd=(in_tokens * spec.input_usd_per_token
                            + out_tokens * spec.output_usd_per_token),
        )

    def compare_models(
        self,
        text: str,
        estimated_output_tokens: Optional[int] = None,
    ) -> ModelComparison:
        """
        Compare token counts and costs across all models in the catalog.

        Each model uses its own tokenizer (Anthropic API or tiktoken), so
        token counts may differ slightly between providers.

        Args:
            text: Input prompt text.
            estimated_output_tokens: Assumed response length for all models.

        Returns:
            ModelComparison sorted by ascending total cost, with the
            cheapest model identified.
        """
        out_tokens = estimated_output_tokens or self.default_output_tokens
        estimates: list[CostEstimate] = [
            self.estimate_cost(text, model_id, out_tokens)
            for model_id in MODELS
        ]
        estimates.sort(key=lambda e: e.total_cost_usd)

        return ModelComparison(
            text_preview=self._preview(text),
            char_count=len(text),
            estimated_output_tokens=out_tokens,
            estimates=estimates,
            cheapest_model=estimates[0].model_id,
        )

    def batch_count(
        self,
        texts: list[str],
        provider: str = "anthropic",
    ) -> list[CountResult]:
        """
        Count tokens for multiple texts, reusing the in-memory cache.

        Texts already seen in this session are returned instantly without
        an API call.  New texts are counted sequentially.

        Args:
            texts: List of input strings.
            provider: ``"anthropic"`` or ``"openai"``.

        Returns:
            List of CountResult, one per input text, in the same order.
        """
        return [self.count(text, provider=provider) for text in texts]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_comparison(self, comparison: ModelComparison) -> None:
        """Print a formatted comparison table to stdout."""
        SEP  = "=" * 80
        THIN = "-" * 80

        print()
        print(SEP)
        print(f'  TEXT COMPARISON  ({comparison.char_count:,} chars)')
        print(f'  Preview: "{comparison.text_preview}"')
        print(f'  Assumed output: {comparison.estimated_output_tokens:,} tokens')
        print(SEP)
        print(
            f"  {'Model':<22} {'Provider':<10} {'In tok':>8} "
            f"{'Input cost':>12} {'Output cost':>12} {'Total cost':>12}"
        )
        print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

        for est in comparison.estimates:
            marker = "  <-- cheapest" if est.model_id == comparison.cheapest_model else ""
            print(
                f"  {est.model_id:<22} {est.provider:<10} {est.input_tokens:>8,} "
                f"${est.input_cost_usd:>11.6f} "
                f"${est.output_cost_usd:>11.6f} "
                f"${est.total_cost_usd:>11.6f}"
                f"{marker}"
            )
        print(SEP)


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

_TEXTS: list[tuple[str, str]] = [
    (
        "MICRO (~5 tok)",
        "Hi!",
    ),
    (
        "SHORT (~30 tok)",
        "What are the main differences between supervised and unsupervised machine learning?",
    ),
    (
        "MEDIUM (~200 tok)",
        """\
You are an expert software architect. A startup is building a real-time analytics
platform that needs to ingest 100,000 events per second, store them for 90 days,
and serve sub-second queries over the last 7 days of data. Their current stack is
Python + PostgreSQL and they have three backend engineers.

What architecture would you recommend, and what are the top three trade-offs they
should be aware of? Keep the answer practical and focused on their constraints.\
""",
    ),
    (
        "LONG (~500 tok)",
        """\
# Code Review Request

Please review the following Python service and identify any bugs, performance issues,
security vulnerabilities, or design problems. Provide a prioritised list of findings
with severity (critical / high / medium / low) and suggested fixes.

```python
import os, pickle, hashlib
from flask import Flask, request, jsonify
from functools import lru_cache
import psycopg2

app = Flask(__name__)
DB_URL = os.environ.get("DATABASE_URL", "postgres://localhost/mydb")

def get_conn():
    return psycopg2.connect(DB_URL)

@lru_cache(maxsize=None)
def get_user(user_id):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cur.fetchone()

@app.route("/user/<user_id>")
def user_endpoint(user_id):
    user = get_user(user_id)
    if not user:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": user[0], "email": user[1], "data": user[2]})

@app.route("/cache", methods=["POST"])
def cache_object():
    obj  = request.json.get("object")
    data = pickle.dumps(obj)
    key  = hashlib.md5(data).hexdigest()
    open(f"/tmp/cache/{key}", "wb").write(data)
    return jsonify({"key": key})

@app.route("/load/<key>")
def load_object(key):
    path = f"/tmp/cache/{key}"
    return jsonify(pickle.loads(open(path, "rb").read()))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
```
""",
    ),
    (
        "VERY LONG (~1500 tok)",
        """\
# System Design Interview: Global URL Shortener

## Problem Statement
Design a URL shortening service (like bit.ly) that operates globally. The service
must handle 10 billion URLs in total storage, 100,000 new URLs created per day,
and 10 million URL redirects per second at peak.

## Functional Requirements
1. Given a long URL, generate a unique short URL (e.g., sho.rt/abc123)
2. Redirect users from short URL to the original long URL
3. Short URLs expire after 5 years by default, configurable up to 10 years
4. Custom aliases allowed (e.g., sho.rt/my-brand-campaign)
5. Analytics: total clicks, unique visitors, geographic distribution, referrers
6. URL validation and malware scanning before shortening

## Non-Functional Requirements
- Availability: 99.99% uptime (52 minutes downtime per year)
- Latency: p99 redirect latency < 10ms globally
- Consistency: eventually consistent for analytics, strongly consistent for URL resolution
- Security: no enumerable IDs, rate limiting, abuse prevention

## Questions to Answer

### 1. Capacity Estimation
Calculate storage requirements, bandwidth, and cache sizing. Show your math.

### 2. Short URL Generation
Compare these approaches:
- Counter-based with base62 encoding
- Random UUID truncated to 8 characters
- Consistent hashing of the long URL
- Snowflake-style distributed ID

What collision probability does each approach give us at 10B URLs?

### 3. Data Model
Design the database schema. Which database(s) would you choose and why?
Consider: write patterns, read patterns, geographic distribution.

### 4. Redirect Service Architecture
How do you achieve < 10ms p99 globally?
- CDN strategy
- Cache hierarchy (CDN edge / regional / origin)
- Cache invalidation on URL expiry or manual deletion

### 5. Analytics Pipeline
Design an analytics pipeline that:
- Counts 10M clicks/second without impacting redirect latency
- Provides near-real-time dashboards (< 30 second lag)
- Stores 5 years of click history
- Supports ad-hoc queries (e.g., "clicks from Germany on mobile in Q3 2024")

### 6. Failure Modes
What happens when:
- The primary database is unreachable?
- The cache layer is completely cold (e.g., after a regional failover)?
- A popular URL goes viral and gets 1M req/s (hot key problem)?
- A malicious user submits 10,000 URLs per minute?

### 7. Global Deployment
How would you deploy this across AWS us-east-1, eu-west-1, and ap-southeast-1?
Address: data residency requirements, active-active vs active-passive, conflict resolution.

Please provide a complete system design with architecture diagram description,
technology choices with justification, and the top 5 risks with mitigations.
""",
    ),
]


if __name__ == "__main__":
    counter = TokenCounter(default_output_tokens=500)

    SEP = "#" * 80
    print()
    print(SEP)
    print("  TOKEN COUNTER — Cross-Model Comparison (5 text sizes)")
    print(SEP)

    for label, text in _TEXTS:
        print(f"\n>>> {label}")
        comparison = counter.compare_models(text, estimated_output_tokens=500)
        counter.print_comparison(comparison)

    # Demonstrate batch_count
    print("\n\n>>> BATCH COUNT demo (5 texts, provider=anthropic, cache reuse)")
    texts_only = [t for _, t in _TEXTS]
    results    = counter.batch_count(texts_only, provider="anthropic")
    print(f"\n  {'Label':<22} {'Chars':>7} {'Tokens':>7} {'Cached'}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}")
    for (label, _), r in zip(_TEXTS, results):
        print(
            f"  {label:<22} {r.char_count:>7,} {r.token_count:>7,} "
            f"{'yes (cache hit)' if r.from_cache else 'no'}"
        )
    print()
