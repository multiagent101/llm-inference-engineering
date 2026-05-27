#!/usr/bin/env python3
"""Pre-call cost estimator for Anthropic API calls."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# USD per token for each model
DEFAULT_MODELS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {
        "input": 0.80 / 1_000_000,
        "output": 4.00 / 1_000_000,
    },
    "claude-sonnet-4-5": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    "claude-opus-4": {
        "input": 15.00 / 1_000_000,
        "output": 75.00 / 1_000_000,
    },
}


@dataclass
class ModelEstimate:
    """Cost estimate for a single model."""

    model: str
    input_tokens: int
    estimated_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


@dataclass
class CostEstimation:
    """Full cost estimation across all configured models."""

    prompt: str
    input_tokens: int
    estimated_output_tokens: int
    estimates: list[ModelEstimate]
    cheapest_model: str


class CostEstimator:
    """
    Estimates API call costs before sending requests.

    Uses the Anthropic token-counting endpoint to get exact input token counts,
    then projects total cost across all configured models given an expected
    output length. All Claude models share the same tokenizer, so a single
    count_tokens call is sufficient regardless of which model will be used.
    """

    def __init__(
        self,
        models: Optional[dict[str, dict[str, float]]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            models: Mapping of model ID → {"input": price_per_token, "output": price_per_token}.
                    Defaults to DEFAULT_MODELS if not provided.
            api_key: Anthropic API key. Reads ANTHROPIC_API_KEY from env if omitted.
        """
        self.models: dict[str, dict[str, float]] = models or DEFAULT_MODELS
        self.client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def estimate(
        self,
        prompt: str,
        estimated_output_tokens: int = 500,
        system: Optional[str] = None,
    ) -> CostEstimation:
        """
        Estimate cost for a prompt across all configured models.

        Calls the Anthropic token-counting API to get the exact input token
        count, then multiplies by each model's per-token price. Output cost
        is projected from ``estimated_output_tokens`` — actual usage will
        differ based on the real response length.

        Args:
            prompt: User message to estimate.
            estimated_output_tokens: Expected response length for cost projection.
            system: Optional system prompt included in the token count.

        Returns:
            CostEstimation with per-model breakdowns and the cheapest model name.

        Raises:
            anthropic.APIConnectionError: If the network request fails.
            anthropic.APIStatusError: If the API returns an error status.
            ValueError: If no models are configured.
        """
        if not self.models:
            raise ValueError("No models configured.")

        messages: list[dict] = [{"role": "user", "content": prompt}]
        count_kwargs: dict = {
            "model": next(iter(self.models)),
            "messages": messages,
        }
        if system:
            count_kwargs["system"] = system

        response = self.client.messages.count_tokens(**count_kwargs)
        input_tokens: int = response.input_tokens

        estimates: list[ModelEstimate] = []
        for model, prices in self.models.items():
            input_cost = input_tokens * prices["input"]
            output_cost = estimated_output_tokens * prices["output"]
            estimates.append(
                ModelEstimate(
                    model=model,
                    input_tokens=input_tokens,
                    estimated_output_tokens=estimated_output_tokens,
                    input_cost_usd=input_cost,
                    output_cost_usd=output_cost,
                    total_cost_usd=input_cost + output_cost,
                )
            )

        cheapest = min(estimates, key=lambda e: e.total_cost_usd)
        estimation = CostEstimation(
            prompt=prompt,
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            estimates=estimates,
            cheapest_model=cheapest.model,
        )

        self._print_report(estimation)
        return estimation

    def add_model(self, model_id: str, input_usd_per_1m: float, output_usd_per_1m: float) -> None:
        """
        Add or update a model's pricing configuration.

        Args:
            model_id: Anthropic model identifier (e.g. "claude-haiku-4-5").
            input_usd_per_1m: Input price in USD per 1 million tokens.
            output_usd_per_1m: Output price in USD per 1 million tokens.
        """
        self.models[model_id] = {
            "input": input_usd_per_1m / 1_000_000,
            "output": output_usd_per_1m / 1_000_000,
        }

    def _print_report(self, estimation: CostEstimation) -> None:
        """Print a formatted comparative cost table to stdout."""
        sep = "=" * 74
        preview = estimation.prompt[:65].replace("\n", " ")
        if len(estimation.prompt) > 65:
            preview += "..."

        print()
        print(sep)
        print("  COST ESTIMATION REPORT")
        print(sep)
        print(f"  Prompt:             {preview}")
        print(f"  Input tokens:       {estimation.input_tokens:>10,}")
        print(f"  Est. output tokens: {estimation.estimated_output_tokens:>10,}")
        print()
        print(
            f"  {'Model':<26} {'Input cost':>12} {'Output cost':>12} {'Total cost':>12}  "
        )
        print(f"  {'-'*26} {'-'*12} {'-'*12} {'-'*12}")
        for est in estimation.estimates:
            marker = " <-- cheapest" if est.model == estimation.cheapest_model else ""
            print(
                f"  {est.model:<26} "
                f"${est.input_cost_usd:>11.6f} "
                f"${est.output_cost_usd:>11.6f} "
                f"${est.total_cost_usd:>11.6f}"
                f"{marker}"
            )
        print(sep)


# ---------------------------------------------------------------------------
# Sample prompts for testing
# ---------------------------------------------------------------------------

_PROMPT_SHORT = (
    "What is the capital of France?"
)

_PROMPT_MEDIUM = """\
You are analyzing the performance characteristics of large language models in production environments.

Consider the following scenario: a company runs a customer support chatbot that handles approximately
10,000 conversations per day. Each conversation averages 5 turns, with user messages averaging 50 tokens
and assistant responses averaging 200 tokens. The system prompt is 300 tokens long.

Please answer the following questions:
1. What is the total daily token consumption (input + output)?
2. How should the company decide between Haiku, Sonnet, and Opus for this use case?
3. What caching strategies would you recommend to reduce costs?
4. How does latency differ between the three model tiers, and how does that affect user experience?
5. What monitoring metrics should the company track to optimize cost and quality over time?

Provide concrete numbers where possible and justify your recommendations.\
"""

_PROMPT_LONG = """\
# Technical Deep-Dive: Optimizing LLM Inference at Scale

## Background

You are a senior ML infrastructure engineer at a mid-sized tech company. Your team has been tasked
with building a production-grade LLM inference pipeline that must handle diverse workloads efficiently.
The system will serve multiple internal teams with very different requirements.

## Current Workloads

### Workload A — Real-time Customer Chat
- Volume: 50,000 requests/day
- Avg input tokens: 800 (includes 400-token system prompt + conversation history)
- Avg output tokens: 150
- Latency SLA: p95 < 2 seconds to first token
- Quality requirement: High — directly customer-facing

### Workload B — Document Summarization Pipeline
- Volume: 5,000 requests/day
- Avg input tokens: 8,000 (long documents)
- Avg output tokens: 500
- Latency SLA: p95 < 30 seconds total
- Quality requirement: Medium — reviewed by humans before use

### Workload C — Internal Code Review Assistant
- Volume: 2,000 requests/day
- Avg input tokens: 3,000 (code diffs + context)
- Avg output tokens: 800
- Latency SLA: p95 < 10 seconds
- Quality requirement: High — developers rely on accuracy

### Workload D — Batch Data Extraction
- Volume: 100,000 requests/day (overnight batch)
- Avg input tokens: 1,200
- Avg output tokens: 200
- Latency SLA: batch completes within 6 hours
- Quality requirement: Medium — structured output validated by schema

## Questions

1. **Model Selection Strategy**: For each workload (A, B, C, D), which Claude model tier would you
   recommend (Haiku, Sonnet, or Opus)? Justify each choice considering quality, latency, and cost.

2. **Cost Analysis**: Calculate the estimated monthly cost for each workload under your recommended
   model selection. Show your math clearly.

3. **Prompt Caching**: Which workloads would benefit most from prompt caching, and why? How would
   you implement caching for the system prompt in Workload A?

4. **Batching Strategy**: For Workload D, design a batching strategy that maximizes throughput within
   the 6-hour window while staying within rate limits. What batch sizes and concurrency levels would
   you use?

5. **Routing Logic**: Design a simple routing system that automatically selects the right model based
   on input complexity. What signals would you use (token count, topic classification, user tier)?
   Describe the decision tree.

6. **Fallback and Reliability**: What happens when the primary model tier is degraded or rate-limited?
   Design a fallback chain for each workload. How do you handle partial failures in streaming responses?

7. **Observability**: What metrics, logs, and traces would you instrument to monitor cost, latency,
   quality, and errors across all four workloads? Name specific metrics and where you would surface them.

8. **Cost Optimization Roadmap**: If you needed to cut LLM costs by 30% without sacrificing quality,
   what would be your top 5 levers? Rank them by expected impact and implementation difficulty.

Please provide specific, actionable recommendations with concrete numbers wherever possible.\
"""


if __name__ == "__main__":
    estimator = CostEstimator()

    test_cases: list[tuple[str, str, int]] = [
        ("SHORT  (~50 tokens)", _PROMPT_SHORT, 100),
        ("MEDIUM (~500 tokens)", _PROMPT_MEDIUM, 500),
        ("LONG   (~2000 tokens)", _PROMPT_LONG, 1000),
    ]

    print("\n" + "#" * 74)
    print("  COST ESTIMATOR — Comparative Analysis Across 3 Prompt Sizes")
    print("#" * 74)

    for label, prompt, est_output in test_cases:
        print(f"\n>>> {label}")
        estimator.estimate(prompt, estimated_output_tokens=est_output)
