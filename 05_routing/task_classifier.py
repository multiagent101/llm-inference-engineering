#!/usr/bin/env python3
"""
task_classifier.py -- Route LLM requests to the cheapest capable model.

Classification flow:
  1. Fast rule-based heuristic (keyword scoring + structure signals).
  2. If confidence < threshold, escalate to claude-haiku for a short
     clarification call (~150 tokens).
  3. Return ClassificationResult with recommended model and cost estimate.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Pricing  (USD per 1M tokens, 2025-Q4)
# ---------------------------------------------------------------------------
_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-haiku-4-5":  (0.80,   4.00),
    "claude-sonnet-4-6": (3.00,  15.00),
    "claude-opus-4-7":   (15.00, 75.00),
}


class TaskComplexity(Enum):
    """Complexity tier for an incoming LLM request."""

    SIMPLE   = "SIMPLE"    # Factual lookups, translations, basic arithmetic
    STANDARD = "STANDARD"  # Explanations, summaries, how-things-work
    COMPLEX  = "COMPLEX"   # Code generation, debugging, system design
    EXPERT   = "EXPERT"    # Research, novel designs, security audits


# Cheapest model that can handle each complexity tier
_MODEL_MAP: Dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE:   "claude-haiku-4-5",
    TaskComplexity.STANDARD: "claude-sonnet-4-6",
    TaskComplexity.COMPLEX:  "claude-sonnet-4-6",
    TaskComplexity.EXPERT:   "claude-opus-4-7",
}

# Typical output tokens per tier (used for pre-call cost estimation)
_TYPICAL_OUTPUT: Dict[TaskComplexity, int] = {
    TaskComplexity.SIMPLE:    80,
    TaskComplexity.STANDARD: 350,
    TaskComplexity.COMPLEX:  700,
    TaskComplexity.EXPERT:  1_200,
}


@dataclass
class ClassificationResult:
    """Result of a single prompt classification."""

    complexity: TaskComplexity
    confidence: float          # 0.0-1.0; values below threshold triggered LLM
    reasoning: str             # Human-readable explanation
    recommended_model: str     # Model ID to use for this query
    estimated_cost_usd: float  # Estimated API cost for this single query
    method_used: str           # "rule" or "llm"


class TaskClassifier:
    """
    Hybrid classifier: fast heuristic first, haiku LLM for borderline cases.

    Parameters
    ----------
    llm_confidence_threshold : float
        Rule-based results with confidence below this value are re-classified
        by claude-haiku.  Default 0.70.
    """

    # -- keyword lists (checked in this order; most specific first) ----------
    _EXPERT_KWS: List[str] = [
        "research paper", "novel algorithm", "novel approach",
        "state of the art", "security audit", "penetration test",
        "gdpr", "soc2", "hipaa", "comprehensive framework",
        "transformer architecture", "reinforcement learning",
        "threat model", "novel attention", "write a research",
        "experimental design",
    ]
    _COMPLEX_KWS: List[str] = [
        "write a ",        # trailing space avoids "write about"
        "write code",
        "implement",       # also matches "implements", "implementation"
        "build a ",
        "create a system",
        "design a ",
        "design an ",
        "debug",
        "refactor",
        "rate-limit",
        "rate limit",
        "lru",
        "binary search tree",
        "complexity",      # time/space complexity
        "distributed cache",
        "distributed system",
        "jwt",
        "quicksort",
        "mergesort",
    ]
    _STANDARD_KWS: List[str] = [
        "explain", "describe", "summarize", "how does", "how do",
        "pros and cons", "difference between", "differences between",
        "compare", "what are the", "overview of", "list the",
        "why does", "why do",
    ]
    _SIMPLE_KWS: List[str] = [
        "capital of", "who wrote", "when was", "where is",
        "translate", "stand for", "meaning of", "definition of",
    ]

    def __init__(self, llm_confidence_threshold: float = 0.70) -> None:
        if not 0.0 < llm_confidence_threshold < 1.0:
            raise ValueError("llm_confidence_threshold must be in (0, 1)")
        self.llm_confidence_threshold = llm_confidence_threshold
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy Anthropic client — created on first LLM call."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        return self._client

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """~4 chars per token approximation."""
        return max(1, len(text) // 4)

    def _estimate_cost(
        self,
        complexity: TaskComplexity,
        input_tokens: int,
        model: Optional[str] = None,
    ) -> float:
        """Estimated USD cost for one query on the given (or recommended) model."""
        m = model or _MODEL_MAP[complexity]
        inp_p, out_p = _PRICING[m]
        out_t = _TYPICAL_OUTPUT[complexity]
        return (input_tokens * inp_p + out_t * out_p) / 1_000_000

    def _score_prompt(self, prompt: str) -> Tuple[int, str]:
        """
        Return (integer_score, reasoning_string).

        Score mapping:
          score < 0      -> SIMPLE
          0 <= score < 4 -> STANDARD
          4 <= score < 8 -> COMPLEX
          score >= 8     -> EXPERT
        """
        pl = prompt.lower()

        expert_hits  = sum(1 for k in self._EXPERT_KWS  if k in pl)
        complex_hits = sum(1 for k in self._COMPLEX_KWS if k in pl)
        std_hits     = sum(1 for k in self._STANDARD_KWS if k in pl)
        simple_hits  = sum(1 for k in self._SIMPLE_KWS  if k in pl)

        word_count = len(prompt.split())
        if   word_count <  10: len_bonus = -1
        elif word_count <  30: len_bonus =  0
        elif word_count <  80: len_bonus =  1
        elif word_count < 200: len_bonus =  2
        else:                  len_bonus =  3

        code_bonus = 3 if "```" in prompt else 0
        list_bonus = 2 if re.search(r"^\s*\d+\.", prompt, re.MULTILINE) else 0
        multi_q    = 2 if prompt.count("?") > 2 else 0

        score = (
            min(expert_hits  * 4, 12)
            + min(complex_hits * 2,  8)
            + min(std_hits,          4)   # each standard kw = +1, cap 4
            - min(simple_hits * 2,   6)
            + len_bonus + code_bonus + list_bonus + multi_q
        )

        parts: List[str] = []
        if expert_hits:  parts.append(f"expert_kws={expert_hits}")
        if complex_hits: parts.append(f"complex_kws={complex_hits}")
        if std_hits:     parts.append(f"std_kws={std_hits}")
        if simple_hits:  parts.append(f"simple_kws={simple_hits}")
        parts.append(f"words={word_count}")
        if code_bonus:   parts.append("code_block")
        if list_bonus:   parts.append("numbered_list")
        reasoning = ", ".join(parts) + f" => score={score}"

        return score, reasoning

    @staticmethod
    def _complexity_from_score(score: int) -> TaskComplexity:
        if score >= 8: return TaskComplexity.EXPERT
        if score >= 4: return TaskComplexity.COMPLEX
        if score >= 0: return TaskComplexity.STANDARD
        return TaskComplexity.SIMPLE

    @staticmethod
    def _confidence_from_score(score: int) -> float:
        """Confidence drops near the boundaries 0, 4, 8."""
        min_dist = min(abs(score - b) for b in (0, 4, 8))
        return {0: 0.62, 1: 0.73, 2: 0.82}.get(min_dist, 0.92)

    def _rule_based(self, prompt: str) -> ClassificationResult:
        score, reasoning = self._score_prompt(prompt)
        complexity = self._complexity_from_score(score)
        confidence = self._confidence_from_score(score)
        in_tok     = self._approx_tokens(prompt)
        return ClassificationResult(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            recommended_model=_MODEL_MAP[complexity],
            estimated_cost_usd=self._estimate_cost(complexity, in_tok),
            method_used="rule",
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse JSON from LLM response, tolerating markdown fences."""
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

    def _llm_based(
        self,
        prompt: str,
        rule_hint: Optional[ClassificationResult] = None,
    ) -> ClassificationResult:
        """
        Ask claude-haiku to classify the prompt.

        Parameters
        ----------
        prompt : str
            The user prompt to classify.
        rule_hint : ClassificationResult, optional
            Rule-based result provided as context; LLM may override it.

        Returns
        -------
        ClassificationResult with method_used='llm'.
        """
        hint = ""
        if rule_hint:
            hint = (
                f"\nRule pre-classification: {rule_hint.complexity.value} "
                f"(conf {rule_hint.confidence:.0%}) -- override if wrong."
            )

        system = (
            "You are a prompt complexity classifier for LLM request routing.\n"
            "Classify the prompt into one level:\n"
            "  SIMPLE   -- Factual lookups, translations, basic math.\n"
            "  STANDARD -- Explanations, summaries, how-things-work, comparisons.\n"
            "  COMPLEX  -- Code generation, debugging, system design, algorithms.\n"
            "  EXPERT   -- Research, novel designs, security audits, advanced analysis.\n\n"
            "Respond with ONLY valid JSON (no markdown):\n"
            '{"complexity":"SIMPLE|STANDARD|COMPLEX|EXPERT",'
            '"confidence":0.0-1.0,"reasoning":"one sentence"}'
        )

        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify this prompt:{hint}\n\n{prompt[:800]}",
                }
            ],
        )

        data = self._extract_json(response.content[0].text)

        raw_level = data.get("complexity", "STANDARD").upper().strip()
        try:
            complexity = TaskComplexity[raw_level]
        except KeyError:
            complexity = rule_hint.complexity if rule_hint else TaskComplexity.STANDARD

        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.80))))
        reasoning  = str(data.get("reasoning", response.content[0].text[:100]))
        in_tok     = self._approx_tokens(prompt)

        return ClassificationResult(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            recommended_model=_MODEL_MAP[complexity],
            estimated_cost_usd=self._estimate_cost(complexity, in_tok),
            method_used="llm",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, prompt: str, force_llm: bool = False) -> ClassificationResult:
        """
        Classify a prompt and return recommended model + estimated cost.

        Uses the rule-based classifier first; escalates to haiku LLM only
        when the rule confidence falls below ``llm_confidence_threshold``.

        Parameters
        ----------
        prompt : str
            The user prompt to classify.
        force_llm : bool
            Skip rule-based and go straight to LLM.  Default False.

        Returns
        -------
        ClassificationResult
        """
        result = self._rule_based(prompt)
        if force_llm or result.confidence < self.llm_confidence_threshold:
            result = self._llm_based(prompt, rule_hint=result)
        return result

    def what_if(self, prompt: str, hit_rate: float) -> ClassificationResult:
        """
        Classify ``prompt`` but adjust cost for an expected cache hit rate.

        Parameters
        ----------
        prompt : str
            The user prompt.
        hit_rate : float
            Fraction of identical/similar queries served from cache (0-1).

        Returns
        -------
        ClassificationResult with estimated_cost_usd scaled by (1 - hit_rate).
        """
        result = self.classify(prompt)
        result.estimated_cost_usd *= (1.0 - hit_rate)
        result.reasoning += f" [cache hit_rate={hit_rate:.0%}]"
        return result

    def benchmark(self, prompts: List[str]) -> None:
        """
        Classify every prompt and print a distribution + cost-savings report.

        Compares smart routing against always-haiku / always-sonnet /
        always-opus baselines.  Output token estimates are complexity-based
        and held constant across strategies so pricing is the only variable.

        Parameters
        ----------
        prompts : list[str]
            Prompts to evaluate.
        """
        W    = 72
        SEP  = "=" * W
        SEP2 = "-" * W

        results: List[ClassificationResult] = []
        llm_calls = 0

        # ── per-prompt table ────────────────────────────────────────────
        print()
        print(SEP)
        print(f"  TASK CLASSIFIER BENCHMARK  --  {len(prompts)} prompts")
        print(SEP)
        print(
            f"  {'#':>3}  {'Prompt (first 44 chars)':<44}  "
            f"{'Level':>8}  {'Conf':>5}  {'Via':>4}"
        )
        print(f"  {'-'*3}  {'-'*44}  {'-'*8}  {'-'*5}  {'-'*4}")

        for i, prompt in enumerate(prompts, 1):
            r = self.classify(prompt)
            results.append(r)
            if r.method_used == "llm":
                llm_calls += 1
            snippet = prompt.replace("\n", " ")[:44]
            print(
                f"  {i:>3}  {snippet:<44}  "
                f"{r.complexity.value:>8}  {r.confidence:>4.0%}  {r.method_used:>4}"
            )

        n = len(prompts)

        # ── distribution ────────────────────────────────────────────────
        print()
        print(SEP2)
        print("  DISTRIBUTION")
        print(SEP2)
        print(
            f"  {'Level':<10}  {'N':>4}  {'%':>5}  "
            f"{'AvgConf':>7}  {'Model':<20}  {'AvgCost':>12}"
        )
        print(f"  {'-'*10}  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*20}  {'-'*12}")

        for level in TaskComplexity:
            lr = [r for r in results if r.complexity == level]
            if not lr:
                continue
            cnt    = len(lr)
            avg_cf = sum(r.confidence for r in lr) / cnt
            avg_c  = sum(r.estimated_cost_usd for r in lr) / cnt
            print(
                f"  {level.value:<10}  {cnt:>4}  {cnt/n:>4.0%}  "
                f"  {avg_cf:>5.0%}  {_MODEL_MAP[level]:<20}  ${avg_c:>11.6f}"
            )

        print(
            f"\n  LLM escalations: {llm_calls}/{n} "
            f"({llm_calls/n:.0%} of prompts needed haiku clarification)"
        )

        # ── cost comparison ─────────────────────────────────────────────
        print()
        print(SEP2)
        print("  COST COMPARISON  (same output-token estimates, different model prices)")
        print(SEP2)

        totals: Dict[str, float] = {s: 0.0 for s in
                                    ("Smart Routing", "Always Haiku",
                                     "Always Sonnet", "Always Opus")}

        for prompt, r in zip(prompts, results):
            it = self._approx_tokens(prompt)
            totals["Smart Routing"] += r.estimated_cost_usd
            totals["Always Haiku"]  += self._estimate_cost(r.complexity, it, "claude-haiku-4-5")
            totals["Always Sonnet"] += self._estimate_cost(r.complexity, it, "claude-sonnet-4-6")
            totals["Always Opus"]   += self._estimate_cost(r.complexity, it, "claude-opus-4-7")

        opus = totals["Always Opus"]
        print(
            f"  {'Strategy':<20}  {'Total':>12}  {'Per prompt':>12}  "
            f"{'vs. always-opus':>15}"
        )
        print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*15}")

        order = ["Always Opus", "Always Sonnet", "Smart Routing", "Always Haiku"]
        for name in order:
            total  = totals[name]
            per_p  = total / n
            vs_str = "(baseline)" if name == "Always Opus" else \
                     f"-{(opus - total) / opus:.1%}"
            tag    = "  <-- recommended" if name == "Smart Routing" else ""
            print(
                f"  {name:<20}  ${total:>11.5f}  ${per_p:>11.7f}  "
                f"{vs_str:>15}{tag}"
            )

        smart  = totals["Smart Routing"]
        sonnet = totals["Always Sonnet"]
        expert_n = sum(1 for r in results if r.complexity == TaskComplexity.EXPERT)
        simple_n = sum(1 for r in results if r.complexity == TaskComplexity.SIMPLE)

        print()
        print(
            f"  Smart routing saves {(opus - smart) / opus:.1%} vs always-opus."
        )
        if sonnet > smart:
            print(
                f"  Smart routing is {(sonnet - smart) / sonnet:.1%} cheaper than "
                f"always-sonnet while keeping {expert_n} expert task(s) on opus."
            )
        print(
            f"  {simple_n} simple task(s) cheaply handled by haiku "
            f"instead of overpaying with sonnet/opus."
        )
        print(SEP)
        print()


# ---------------------------------------------------------------------------
# __main__ -- 20-prompt benchmark
# ---------------------------------------------------------------------------

_TEST_PROMPTS: List[str] = [
    # ── SIMPLE (5) ──────────────────────────────────────────────────────
    "What is the capital of France?",
    "Translate 'hello' to Spanish.",
    "What does API stand for?",
    "Who wrote Romeo and Juliet?",
    "What is 15% of 200?",

    # ── STANDARD (6) ────────────────────────────────────────────────────
    "Explain how HTTP cookies work.",
    "What are the pros and cons of microservices architecture?",
    "Summarize the main differences between SQL and NoSQL databases.",
    "Describe the gradient descent algorithm.",
    "What are the SOLID principles in software engineering?",
    "How does garbage collection work in Python?",

    # ── COMPLEX (6) ─────────────────────────────────────────────────────
    (
        "Write a Python function that implements a binary search tree "
        "with insert, delete, and search operations."
    ),
    (
        "Debug this Python code and explain what is wrong:\n"
        "```python\n"
        "def factorial(n):\n"
        "    if n == 0: return 1\n"
        "    return n * factorial(n)  # missing -1\n"
        "```"
    ),
    (
        "Design a rate-limiting system for a REST API that handles "
        "10,000 requests per second using a token bucket algorithm."
    ),
    (
        "Analyze the time and space complexity of quicksort vs mergesort "
        "and provide Python implementations of both."
    ),
    (
        "Implement a distributed cache with LRU eviction policy "
        "and TTL support in Python."
    ),
    (
        "Refactor this authentication module to use JWT tokens instead "
        "of session-based cookies, maintaining backward compatibility."
    ),

    # ── EXPERT (3) ──────────────────────────────────────────────────────
    (
        "Research and compare the latest transformer architectures "
        "(GPT-4, Claude, Gemini) and propose a novel attention mechanism "
        "that could improve efficiency for long-context tasks. Include "
        "mathematical formulation and expected performance gains."
    ),
    (
        "Design a comprehensive security audit framework for a multi-tenant "
        "SaaS application handling PII data, including threat modeling, "
        "penetration testing strategy, and compliance with GDPR and SOC2."
    ),
    (
        "Write a research paper outline on the application of reinforcement "
        "learning to optimize LLM inference routing, including methodology, "
        "experimental design, and expected contributions to the field."
    ),
]


if __name__ == "__main__":
    classifier = TaskClassifier(llm_confidence_threshold=0.70)
    classifier.benchmark(_TEST_PROMPTS)
