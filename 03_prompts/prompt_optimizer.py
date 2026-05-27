#!/usr/bin/env python3
"""Automatic A/B testing and ranking of prompt variants by quality-to-cost ratio."""

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import tiktoken
from dotenv import load_dotenv

# PromptCompressor lives in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from prompt_compressor import PromptCompressor

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 0.80e-6,  "output": 4.00e-6},
    "claude-sonnet-4-5": {"input": 3.00e-6,  "output": 15.00e-6},
    "claude-opus-4":     {"input": 15.00e-6, "output": 75.00e-6},
}

# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------

_JUDGE_TEMPLATE = """\
You are an expert quality evaluator for AI assistant responses.

The assistant operated under this system prompt:
<system_prompt>
{system_prompt}
</system_prompt>

User query:
<query>{query}</query>

Assistant response:
<response>{response}</response>

Rate the response 1-10 on:
- Helpfulness: does it fully solve the user's problem?
- Completeness: are all aspects of the query addressed?
- Clarity: well-structured and easy to understand?
- Tone: appropriate for the context in the system prompt?
- Conciseness: efficient, no unnecessary padding?

Reply with valid JSON only — no markdown fences, no extra text:
{{"score": <integer 1-10>, "reasoning": "<2-3 concise sentences>"}}"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """Evaluation outcome for one prompt variant."""

    variant_id: str
    strategy: str
    system_prompt: str
    system_tokens: int            # tiktoken count of the system prompt alone
    input_tokens: int             # API-reported input (system + user)
    output_tokens: int
    cost_usd: float               # actual cost of this API call
    response_text: str
    quality_score: float          # 1-10 from judge
    quality_reasoning: str
    latency_ms: float
    quality_per_kdollar: float    # quality_score / (cost_usd × 1 000)
    token_reduction_pct: float    # vs original (0.0 for the original baseline)
    timestamp: str


@dataclass
class OptimizationReport:
    """Full A/B test report across all evaluated variants."""

    original_prompt: str
    test_query: str
    model: str
    judge_model: str
    variants: list[VariantResult]   # sorted: best quality_per_kdollar first
    winner_id: str
    runner_up_id: str
    timestamp: str


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """
    Generates compressed variants of a prompt, evaluates each one with a
    real API call, and ranks them by quality-to-cost ratio.

    Pipeline
    --------
    1. ``generate_variants`` — five deterministic compressions of the
       original prompt (original + 4 algorithmic strategies).
    2. ``_run_single`` — calls the target model; records latency, tokens,
       actual cost.
    3. ``_judge`` — calls the judge model with system-prompt + query +
       response and extracts a structured JSON score (1-10).
    4. ``optimize`` — orchestrates the pipeline, sorts by
       ``quality_per_kdollar``, and writes ``optimization_results.json``.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        judge_model: str = "claude-haiku-4-5",
        output_file: str = "optimization_results.json",
        max_response_tokens: int = 512,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: Target model used to generate responses for each variant.
            judge_model: Model used to evaluate response quality.
            output_file: Path to the JSON file where results are appended.
            max_response_tokens: ``max_tokens`` passed to the target model.
            api_key: Anthropic API key; reads env var if omitted.
        """
        self.model               = model
        self.judge_model         = judge_model
        self.output_file         = Path(output_file)
        self.max_response_tokens = max_response_tokens
        self._client             = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self._compressor = PromptCompressor(similarity_threshold=0.82)
        self._enc        = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def generate_variants(self, prompt: str) -> list[tuple[str, str]]:
        """
        Produce up to five named variants of ``prompt``.

        Strategies
        ----------
        * ``original`` — unchanged baseline.
        * ``compress_instructions`` — strips verbose prefixes and boilerplate.
        * ``remove_redundancy`` — drops semantically near-duplicate sentences.
        * ``combined`` — ``compress_instructions`` applied to the
          ``remove_redundancy`` output for maximum algorithmic reduction.
        * ``bullet_only`` — keeps role definition, section headers, and
          structured list/bullet items; drops all prose paragraphs.

        Duplicate texts are silently dropped so the returned list always
        contains genuinely distinct variants.

        Returns:
            List of ``(strategy_name, prompt_text)`` tuples, original first.
        """
        seen: set[str] = set()
        variants: list[tuple[str, str]] = []

        def _add(name: str, text: str) -> None:
            key = text.strip()
            if key and key not in seen:
                seen.add(key)
                variants.append((name, text))

        _add("original",              prompt)
        _add("compress_instructions", self._compressor.compress_instructions(prompt))
        _add("remove_redundancy",     self._compressor.remove_redundancy(prompt))
        _add(
            "combined",
            self._compressor.compress_instructions(
                self._compressor.remove_redundancy(prompt)
            ),
        )
        bullet = self._make_bullet_only(prompt)
        if bullet:
            _add("bullet_only", bullet)

        return variants

    def _make_bullet_only(self, prompt: str) -> str:
        """
        Extract the structural skeleton of an instruction prompt.

        Keeps: role definition (first "You are…" sentence), section headers
        (short lines ending in ``:``) and all bullet / numbered-list items,
        plus lines that begin with imperative verbs.  Everything else is
        discarded.
        """
        kept: list[str] = []
        role_seen = False

        for line in prompt.splitlines():
            s = line.strip()
            if not s:
                continue
            if not role_seen and re.match(r"^You are\b", s, re.IGNORECASE):
                kept.append(s)
                role_seen = True
            elif s.endswith(":") and len(s.split()) <= 8:
                kept.append(s)
            elif re.match(r"^[-*•–]|\d+\.", s):
                kept.append(line)
            elif re.match(
                r"^(?:Always|Never|Avoid|Ensure|Use |Keep |Provide|Escalate|"
                r"End |Include|Acknowledge|Greet|Confirm|Offer)",
                s,
            ):
                kept.append(s)

        return "\n".join(kept).strip()

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _run_single(
        self, system_prompt: str, query: str
    ) -> tuple[str, float, int, int, float]:
        """
        Call the target model and return
        ``(response_text, latency_ms, input_tokens, output_tokens, cost_usd)``.
        """
        prices = MODEL_PRICING.get(self.model, MODEL_PRICING["claude-haiku-4-5"])
        t0  = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_response_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        latency_ms    = (time.perf_counter() - t0) * 1000
        response_text = msg.content[0].text
        in_tok        = msg.usage.input_tokens
        out_tok       = msg.usage.output_tokens
        cost          = in_tok * prices["input"] + out_tok * prices["output"]
        return response_text, latency_ms, in_tok, out_tok, cost

    def _judge(
        self, system_prompt: str, query: str, response: str
    ) -> tuple[float, str]:
        """
        Ask the judge model to score ``response`` on a 1-10 scale.

        Attempts JSON parsing first; falls back to regex if the model
        returns malformed output.  Score is clamped to [1, 10].

        Returns:
            ``(score, reasoning)``; score defaults to 5.0 on parse failure.
        """
        content = _JUDGE_TEMPLATE.format(
            system_prompt=system_prompt, query=query, response=response
        )
        msg = self._client.messages.create(
            model=self.judge_model,
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        raw = msg.content[0].text.strip()

        # Primary: JSON parse (handle optional markdown fences)
        try:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data      = json.loads(m.group())
                score     = float(data.get("score", 5))
                reasoning = str(data.get("reasoning", raw))
                return min(max(score, 1.0), 10.0), reasoning
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: regex
        m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
        score = float(m.group(1)) if m else 5.0
        return min(max(score, 1.0), 10.0), raw

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def optimize(self, prompt: str, test_query: str) -> OptimizationReport:
        """
        Run the full optimization pipeline for ``prompt``.

        For each variant:
        * Makes one real API call with the variant as the system prompt.
        * Asks the judge model to score the response.
        * Computes ``quality_per_kdollar = score / (cost_usd × 1_000)``.

        Results are sorted in descending order of ``quality_per_kdollar``
        and appended to ``optimization_results.json``.

        Args:
            prompt: Original system prompt to optimise.
            test_query: Representative user message for all variants.

        Returns:
            OptimizationReport with all ranked variants.
        """
        variants       = self.generate_variants(prompt)
        original_toks  = len(self._enc.encode(prompt))
        results: list[VariantResult] = []

        print(
            f"\n  {len(variants)} variants × 2 calls each "
            f"(response + judge) = {len(variants) * 2} API calls total\n"
        )

        for idx, (strategy, vprompt) in enumerate(variants, 1):
            sys_toks     = len(self._enc.encode(vprompt))
            reduction    = (1 - sys_toks / original_toks) * 100 if original_toks else 0.0
            variant_id   = f"v{idx}_{strategy}"

            print(
                f"  [{idx}/{len(variants)}] {strategy:<26} "
                f"{sys_toks:>4} tok ({reduction:+.1f}%)  ",
                end="", flush=True,
            )

            resp, lat, in_tok, out_tok, cost = self._run_single(vprompt, test_query)
            score, reasoning                 = self._judge(vprompt, test_query, resp)
            ratio = score / (cost * 1_000) if cost > 0 else 0.0

            print(f"score={score:.1f}/10  cost=${cost:.5f}  Q/k$={ratio:,.0f}")

            results.append(VariantResult(
                variant_id=variant_id,
                strategy=strategy,
                system_prompt=vprompt,
                system_tokens=sys_toks,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                response_text=resp,
                quality_score=score,
                quality_reasoning=reasoning,
                latency_ms=round(lat, 1),
                quality_per_kdollar=round(ratio, 1),
                token_reduction_pct=round(reduction, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        results.sort(key=lambda r: r.quality_per_kdollar, reverse=True)

        report = OptimizationReport(
            original_prompt=prompt,
            test_query=test_query,
            model=self.model,
            judge_model=self.judge_model,
            variants=results,
            winner_id=results[0].variant_id,
            runner_up_id=results[1].variant_id if len(results) > 1 else results[0].variant_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._save(report)
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, report: OptimizationReport) -> None:
        """Append ``report`` to ``optimization_results.json``."""
        existing: list[dict] = []
        if self.output_file.exists():
            try:
                with open(self.output_file, encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, list):
                        existing = d
            except (json.JSONDecodeError, OSError):
                pass
        existing.append(asdict(report))
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: OptimizationReport) -> None:
        """Print the ranked variant table and winner details to stdout."""
        SEP  = "=" * 84
        THIN = "-" * 84
        MEDALS = {1: "** WINNER **", 2: "2nd", 3: "3rd"}

        print()
        print(SEP)
        print("  PROMPT OPTIMIZATION — FINAL RANKING")
        print(SEP)
        print(f"  Model:       {report.model}")
        print(f"  Judge:       {report.judge_model}")
        print(f"  Test query:  \"{report.test_query[:72]}\"")
        print(f"  Variants:    {len(report.variants)}")
        print()
        print(
            f"  {'Rank':<5} {'Strategy':<26} {'Sys tok':>8} {'Reduction':>10} "
            f"{'Score':>7} {'Cost ($)':>10} {'Q / k$':>10}"
        )
        print(
            f"  {'-'*5} {'-'*26} {'-'*8} {'-'*10} "
            f"{'-'*7} {'-'*10} {'-'*10}"
        )
        for rank, r in enumerate(report.variants, 1):
            medal = MEDALS.get(rank, f"{rank}th")
            print(
                f"  {rank:<5} {r.strategy:<26} {r.system_tokens:>8,} "
                f"{r.token_reduction_pct:>+9.1f}% "
                f"{r.quality_score:>7.1f} "
                f"{r.cost_usd:>10.5f} "
                f"{r.quality_per_kdollar:>10,.0f}  {medal}"
            )

        winner = report.variants[0]
        print()
        print(THIN)
        print(f"  WINNER: {winner.strategy}")
        print(f"  Quality score:    {winner.quality_score:.1f} / 10")
        print(f"  Token reduction:  {winner.token_reduction_pct:+.1f}%  "
              f"({winner.system_tokens:,} tokens vs original)")
        print(f"  Cost per call:    ${winner.cost_usd:.5f}")
        print(f"  Q / k$ ratio:     {winner.quality_per_kdollar:,.0f}")
        print()
        print(f"  Judge reasoning:")
        print(f"  \"{winner.quality_reasoning[:250]}\"")
        print()
        print(f"  Response preview:")
        preview = winner.response_text[:280].replace("\n", " | ")
        print(f"  \"{preview}{'...' if len(winner.response_text) > 280 else ''}\"")
        print(SEP)
        print(f"\n  Full results saved to: {self.output_file.resolve()}")


# ---------------------------------------------------------------------------
# Demo prompt — customer support system prompt (~230 tokens)
# ---------------------------------------------------------------------------

_SUPPORT_SYSTEM_PROMPT = """\
You are a friendly and professional customer support representative for TechFlow,
a cloud-based project management SaaS platform trusted by over 50,000 teams worldwide.

Your core responsibilities:
- Resolve billing questions, subscription changes, cancellations, and refund requests
- Troubleshoot technical issues with login, integrations, notifications, and data export
- Guide customers through onboarding, feature setup, and workflow configuration
- Escalate complex issues to the appropriate internal team with a support ticket number

Communication guidelines:
- Always greet the customer warmly and acknowledge their frustration before offering solutions
- Use clear, accessible language; avoid technical jargon unless the customer uses it first
- Provide numbered step-by-step instructions for any process that requires multiple actions
- Keep responses focused: 2-4 short paragraphs maximum unless detailed troubleshooting is needed
- Always end every interaction with: "Is there anything else I can help you with today?"

Escalation policy:
- Billing disputes exceeding $200 must be escalated to billing@techflow.com with ticket reference
- Account security issues or suspected data breaches must be escalated immediately to security team
- Legal threats, GDPR requests, or compliance questions must be forwarded to legal@techflow.com
- Unresolved technical issues after two troubleshooting attempts require a priority support ticket

Tone: Empathetic, patient, solution-focused. Never argue with a customer or dismiss their concern.
Please make sure to always be polite and to always thank the customer for reaching out to us.
Remember to always confirm that the issue has been resolved before closing the conversation.
Don't forget to mention that premium support customers have access to 24/7 phone support.
"""

_TEST_QUERY = (
    "I was charged twice for my monthly subscription this billing cycle. "
    "I need a refund immediately — this is completely unacceptable and I'm "
    "considering cancelling my account entirely."
)

if __name__ == "__main__":
    optimizer = PromptOptimizer(
        model="claude-haiku-4-5",
        judge_model="claude-haiku-4-5",
        output_file="optimization_results.json",
        max_response_tokens=400,
    )

    SEP = "#" * 84
    print()
    print(SEP)
    print("  PROMPT OPTIMIZER — Customer Support System Prompt")
    print(SEP)
    print(f"  Original prompt: {len(optimizer._enc.encode(_SUPPORT_SYSTEM_PROMPT)):,} tokens")
    print(f"  Test query:      \"{_TEST_QUERY[:72]}...\"")

    report = optimizer.optimize(_SUPPORT_SYSTEM_PROMPT, _TEST_QUERY)
    optimizer.print_report(report)
