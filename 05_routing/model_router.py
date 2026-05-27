#!/usr/bin/env python3
"""
model_router.py -- Central routing engine that sends each LLM request to the
cheapest capable model, validates response quality via LLM-as-judge, and
falls back to a stronger model automatically when quality is insufficient.

Flow:
  classify -> call cheapest model -> quality gate -> [fallback if needed]
  -> record stats -> return RouterResponse
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

# sibling module in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from task_classifier import (
    ClassificationResult,
    TaskClassifier,
    TaskComplexity,
    _MODEL_MAP,
    _PRICING,
)

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered fallback chain: if quality gate fails, escalate one step
_FALLBACK_CHAIN: Dict[str, Optional[str]] = {
    "claude-haiku-4-5":  "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-opus-4-7",
    "claude-opus-4-7":   None,
}

# Conservative max_tokens per complexity tier
_MAX_TOKENS: Dict[TaskComplexity, int] = {
    TaskComplexity.SIMPLE:    256,
    TaskComplexity.STANDARD:  600,
    TaskComplexity.COMPLEX:  1200,
    TaskComplexity.EXPERT:   2000,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RouterResponse:
    """Complete result of one routed LLM request."""

    model_used: str                              # final model (after any fallback)
    response_text: str                           # LLM answer
    input_tokens: int                            # input tokens for the final call
    output_tokens: int                           # output tokens for the final call
    cost_usd: float                              # total cost incl. all retry + judge calls
    classification: ClassificationResult         # routing decision
    latency_ms: float                            # wall-clock time from classify to done
    quality_score: Optional[float] = None        # final LLM-as-judge score (1-10)
    initial_quality_score: Optional[float] = None  # score before fallback (if triggered)
    fallback_triggered: bool = False              # True when quality gate triggered retry
    fallback_from_model: Optional[str] = None    # model that produced the low-quality answer


@dataclass
class RouterStats:
    """Aggregated statistics across all requests processed by the router."""

    total_routed: int
    total_cost_usd: float
    total_cost_opus_equivalent: float   # what those requests would have cost on opus
    total_saved_vs_always_opus: float
    savings_pct: float
    fallback_count: int
    fallback_rate_pct: float
    avg_cost_per_request: float
    avg_latency_ms: float
    avg_quality_score: Optional[float]
    model_distribution: Dict[str, int]       # final model -> request count
    complexity_distribution: Dict[str, int]  # complexity level -> request count


@dataclass
class _Record:
    """Internal per-request record used to compute RouterStats."""

    model_used: str
    initial_model: str          # routing target before any fallback
    cost_usd: float
    opus_equiv_cost: float
    fallback: bool
    quality_score: Optional[float]
    latency_ms: float
    complexity: str


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Central LLM router with automatic model selection, quality gating, and
    fallback to stronger models when response quality is insufficient.

    Parameters
    ----------
    enable_quality_gate : bool
        When True, each non-opus response is scored by LLM-as-judge.
        Responses scoring below ``quality_threshold`` are retried with the
        next model in the fallback chain.  Default True.
    quality_threshold : float
        Minimum acceptable LLM-as-judge score (1-10).  Default 7.0.
    llm_confidence_threshold : float
        Forwarded to TaskClassifier: prompts with rule confidence below this
        value are re-classified by claude-haiku.  Default 0.70.
    """

    def __init__(
        self,
        enable_quality_gate: bool = True,
        quality_threshold: float = 7.0,
        llm_confidence_threshold: float = 0.70,
    ) -> None:
        self.enable_quality_gate   = enable_quality_gate
        self.quality_threshold     = quality_threshold
        self.classifier            = TaskClassifier(llm_confidence_threshold)
        self._client: Optional[anthropic.Anthropic] = None
        self._records: List[_Record] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy Anthropic client — created on first use."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        return self._client

    @staticmethod
    def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Return USD cost for a single API call."""
        inp_p, out_p = _PRICING.get(model, (15.00, 75.00))
        return (input_tokens * inp_p + output_tokens * out_p) / 1_000_000

    def _call_model(
        self,
        model: str,
        prompt: str,
        complexity: TaskComplexity,
    ) -> Tuple[str, int, int]:
        """
        Call the Anthropic API and return (response_text, input_tokens, output_tokens).

        Parameters
        ----------
        model : str
            Model ID to call.
        prompt : str
            User prompt.
        complexity : TaskComplexity
            Used to select the appropriate max_tokens ceiling.
        """
        max_tok = _MAX_TOKENS[complexity]
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tok,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return text, response.usage.input_tokens, response.usage.output_tokens

    def _judge_quality(self, prompt: str, response_text: str) -> float:
        """
        Score a model response on a 1-10 scale using claude-haiku as judge.

        Parameters
        ----------
        prompt : str
            The original user request.
        response_text : str
            The response to evaluate.

        Returns
        -------
        float
            Score in [1, 10].  Returns 8.0 on parse failure (assume OK).
        """
        # 2000-char window: enough to judge a 600-token response without
        # the judge penalising hard cut-offs on long answers
        p_snip = prompt[:400]
        r_snip = response_text[:2000]

        judge_msg = (
            f"User request:\n{p_snip}\n\n"
            f"Model response (first 2000 chars):\n{r_snip}\n\n"
            "Rate the QUALITY of what you can see on a 1-10 scale.\n"
            "  10 = accurate, complete for this request\n"
            "   8 = good answer with minor omissions\n"
            "   6 = partially answers but misses key points\n"
            "   4 = significant errors or mostly off-topic\n"
            "   1 = wrong or refuses without cause\n"
            "Do NOT penalise for cut-off at 2000 chars if what you see is good.\n"
            'Reply ONLY with JSON: {"score": N}'
        )

        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=60,
                system="You are an objective LLM response quality evaluator.",
                messages=[{"role": "user", "content": judge_msg}],
            )
            raw  = resp.content[0].text.strip()
            data = self._parse_json(raw)
            return min(10.0, max(1.0, float(data.get("score", 8.0))))
        except Exception:
            return 8.0  # conservative default on failure

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract first JSON object from text, tolerating markdown fences."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        prompt: str,
        force_model: Optional[str] = None,
    ) -> RouterResponse:
        """
        Route a prompt to the appropriate model and return the response.

        Workflow:
          1. Classify the prompt (rule-based, with optional haiku escalation).
          2. Call the recommended (or forced) model.
          3. If quality gate is enabled and model is not opus, score the response.
          4. If score < quality_threshold, retry with the next model in the chain.
          5. Record stats and return RouterResponse.

        Parameters
        ----------
        prompt : str
            The user prompt to answer.
        force_model : str, optional
            Override classification and always use this model ID.

        Returns
        -------
        RouterResponse
        """
        wall_t0 = time.perf_counter()

        # ── 1. Classify ───────────────────────────────────────────────
        cls_result    = self.classifier.classify(prompt)
        initial_model = force_model or cls_result.recommended_model
        complexity    = cls_result.complexity

        # ── 2. First call ─────────────────────────────────────────────
        resp_text, in_tok, out_tok = self._call_model(
            initial_model, prompt, complexity
        )
        total_cost = self._compute_cost(initial_model, in_tok, out_tok)

        final_model                      = initial_model
        quality_score: Optional[float]   = None
        initial_quality_score: Optional[float] = None
        fallback_triggered               = False
        fallback_from: Optional[str]     = None

        # ── 3. Quality gate ───────────────────────────────────────────
        # Requirement: accept if score > quality_threshold (i.e., strictly above).
        # Fallback when score <= quality_threshold.
        runs_gate = (
            self.enable_quality_gate
            and initial_model != "claude-opus-4-7"
            and force_model is None
        )
        if runs_gate:
            quality_score = self._judge_quality(prompt, resp_text)
            # ~350 in + 60 out tokens for the judge call on haiku
            total_cost += self._compute_cost("claude-haiku-4-5", 350, 60)

            if quality_score <= self.quality_threshold:  # NOT > threshold -> fallback
                fb_model = _FALLBACK_CHAIN.get(initial_model)
                if fb_model:
                    fallback_triggered    = True
                    fallback_from         = initial_model
                    initial_quality_score = quality_score  # preserve pre-fallback score

                    # ── 4. Fallback call ──────────────────────────────
                    fb_text, fb_in, fb_out = self._call_model(
                        fb_model, prompt, complexity
                    )
                    total_cost += self._compute_cost(fb_model, fb_in, fb_out)

                    # Re-judge the fallback response
                    fb_score   = self._judge_quality(prompt, fb_text)
                    total_cost += self._compute_cost("claude-haiku-4-5", 350, 60)

                    final_model   = fb_model
                    resp_text     = fb_text
                    in_tok        = fb_in
                    out_tok       = fb_out
                    quality_score = fb_score

        latency_ms = (time.perf_counter() - wall_t0) * 1000

        # ── 5. Opus equivalent for savings calculation ─────────────────
        opus_cost = self._compute_cost("claude-opus-4-7", in_tok, out_tok)

        # ── 6. Record ─────────────────────────────────────────────────
        self._records.append(_Record(
            model_used=final_model,
            initial_model=initial_model,
            cost_usd=total_cost,
            opus_equiv_cost=opus_cost,
            fallback=fallback_triggered,
            quality_score=quality_score,
            latency_ms=latency_ms,
            complexity=complexity.value,
        ))

        return RouterResponse(
            model_used=final_model,
            response_text=resp_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=total_cost,
            classification=cls_result,
            latency_ms=latency_ms,
            quality_score=quality_score,
            initial_quality_score=initial_quality_score,
            fallback_triggered=fallback_triggered,
            fallback_from_model=fallback_from,
        )

    def get_stats(self) -> RouterStats:
        """
        Return cumulative statistics across all routed requests.

        Returns
        -------
        RouterStats
        """
        n = len(self._records)
        if n == 0:
            return RouterStats(
                total_routed=0,
                total_cost_usd=0.0,
                total_cost_opus_equivalent=0.0,
                total_saved_vs_always_opus=0.0,
                savings_pct=0.0,
                fallback_count=0,
                fallback_rate_pct=0.0,
                avg_cost_per_request=0.0,
                avg_latency_ms=0.0,
                avg_quality_score=None,
                model_distribution={},
                complexity_distribution={},
            )

        total_cost = sum(r.cost_usd for r in self._records)
        total_opus = sum(r.opus_equiv_cost for r in self._records)
        saved      = total_opus - total_cost
        fallbacks  = sum(1 for r in self._records if r.fallback)

        scores = [r.quality_score for r in self._records if r.quality_score is not None]
        avg_q  = sum(scores) / len(scores) if scores else None

        model_dist: Dict[str, int] = {}
        for r in self._records:
            model_dist[r.model_used] = model_dist.get(r.model_used, 0) + 1

        cmplx_dist: Dict[str, int] = {}
        for r in self._records:
            cmplx_dist[r.complexity] = cmplx_dist.get(r.complexity, 0) + 1

        return RouterStats(
            total_routed=n,
            total_cost_usd=total_cost,
            total_cost_opus_equivalent=total_opus,
            total_saved_vs_always_opus=saved,
            savings_pct=(saved / total_opus * 100) if total_opus > 0 else 0.0,
            fallback_count=fallbacks,
            fallback_rate_pct=fallbacks / n * 100,
            avg_cost_per_request=total_cost / n,
            avg_latency_ms=sum(r.latency_ms for r in self._records) / n,
            avg_quality_score=avg_q,
            model_distribution=model_dist,
            complexity_distribution=cmplx_dist,
        )

    def print_stats(self) -> None:
        """Print a formatted statistics report to stdout."""
        s   = self.get_stats()
        W   = 68
        SEP = "=" * W
        SEP2 = "-" * W

        print()
        print(SEP)
        print("  MODEL ROUTER -- CUMULATIVE STATS")
        print(SEP)

        print(f"  Requests routed:      {s.total_routed}")
        print(f"  Total cost:           ${s.total_cost_usd:.6f}")
        print(f"  Opus equivalent:      ${s.total_cost_opus_equivalent:.6f}")
        print(f"  Saved vs always-opus: ${s.total_saved_vs_always_opus:.6f}  "
              f"({s.savings_pct:.1f}%)")
        print(f"  Avg cost / request:   ${s.avg_cost_per_request:.6f}")
        print(f"  Avg latency:          {s.avg_latency_ms:.0f} ms")

        if s.avg_quality_score is not None:
            print(f"  Avg quality score:    {s.avg_quality_score:.1f} / 10")
        print(f"  Fallbacks triggered:  {s.fallback_count} / {s.total_routed} "
              f"({s.fallback_rate_pct:.0f}%)")

        print()
        print(SEP2)
        print("  MODEL DISTRIBUTION  (final model after fallback)")
        print(SEP2)
        for model, count in sorted(s.model_distribution.items()):
            bar = "#" * count
            pct = count / s.total_routed * 100
            print(f"  {model:<22} {count:>3}  {pct:>5.0f}%  {bar}")

        print()
        print(SEP2)
        print("  COMPLEXITY DISTRIBUTION")
        print(SEP2)
        for level in ("SIMPLE", "STANDARD", "COMPLEX", "EXPERT"):
            count = s.complexity_distribution.get(level, 0)
            if count == 0:
                continue
            bar = "#" * count
            pct = count / s.total_routed * 100
            print(f"  {level:<10} {count:>3}  {pct:>5.0f}%  {bar}")

        print(SEP)
        print()


# ---------------------------------------------------------------------------
# __main__ -- 10-request mixed demo
# ---------------------------------------------------------------------------

_DEMO_PROMPTS: List[str] = [
    # SIMPLE (2) -> haiku
    "What is the capital of Japan?",
    "Translate 'thank you very much' to Italian.",

    # STANDARD (4) -> sonnet
    "Explain the difference between TCP and UDP protocols.",
    "What are the main advantages of using Docker containers?",
    "How does the Python garbage collector work?",
    "Describe the CAP theorem and give a real-world example of each combination.",

    # COMPLEX (3) -> sonnet
    (
        "Write a Python function implementing the Sieve of Eratosthenes "
        "to find all prime numbers up to N."
    ),
    "Implement a thread-safe singleton pattern in Python with double-checked locking.",
    (
        "Design a URL shortener system (like bit.ly): explain the key components, "
        "data model, and how to handle 10,000 redirects per second."
    ),

    # EXPERT (1) -> opus
    (
        "Analyze the trade-offs between event sourcing and traditional CRUD "
        "architectures for a high-traffic financial platform. Propose a hybrid "
        "design with specific recommendations on consistency, replay, and "
        "audit-log requirements."
    ),
]


def _short(text: str, n: int = 90) -> str:
    """Return first n ASCII-safe chars, newlines collapsed to spaces."""
    flat = text.replace("\n", " ")
    safe = flat.encode("ascii", errors="replace").decode("ascii")
    return safe[:n] + ("..." if len(safe) > n else "")


if __name__ == "__main__":
    router = ModelRouter(enable_quality_gate=True, quality_threshold=7.0)

    W    = 68
    SEP  = "=" * W
    SEP2 = "-" * W

    print()
    print(SEP)
    print("  MODEL ROUTER DEMO  --  10 requests")
    print(SEP)

    total = len(_DEMO_PROMPTS)
    for i, prompt in enumerate(_DEMO_PROMPTS, 1):
        print(f"\n  [{i}/{total}] Routing: {_short(prompt, 55)}")

        r = router.route(prompt)

        # classification line
        initial = r.fallback_from_model or r.model_used
        cls_str = (
            f"  Classified: {r.classification.complexity.value}"
            f" ({r.classification.confidence:.0%} conf, {r.classification.method_used})"
            f"  ->  initial={initial}"
        )
        print(cls_str)

        # fallback notice
        if r.fallback_triggered:
            init_s = (
                f"{r.initial_quality_score:.1f}/10"
                if r.initial_quality_score is not None else "?"
            )
            print(
                f"  ** FALLBACK: {r.fallback_from_model} scored {init_s}"
                f" (threshold >{router.quality_threshold:.0f})"
                f"  ->  retried on {r.model_used}"
            )

        # result line
        q_str = f"  quality={r.quality_score:.1f}/10" if r.quality_score else "  quality=n/a"
        print(
            f"  Model: {r.model_used:<22}"
            f"  tokens={r.input_tokens}in/{r.output_tokens}out"
            f"  cost=${r.cost_usd:.6f}"
            f"  lat={r.latency_ms:.0f}ms"
        )
        print(f"  Response: {_short(r.response_text, 80)}")
        print(f"  {q_str}")
        print(f"  {SEP2}")

    # Final stats
    router.print_stats()
