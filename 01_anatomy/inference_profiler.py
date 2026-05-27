#!/usr/bin/env python3
"""Inference profiler for Anthropic API calls."""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent / ".env")

_INPUT_PRICE_PER_TOKEN: float = 0.80 / 1_000_000
_OUTPUT_PRICE_PER_TOKEN: float = 4.00 / 1_000_000


@dataclass
class InferenceProfile:
    """Complete timing, token, and cost profile of a single API call."""

    model: str
    prompt: str
    response_text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_latency_ms: float
    inter_token_times_ms: list[float]
    avg_inter_token_ms: float
    throughput_tokens_per_sec: float
    cost_input_usd: float
    cost_output_usd: float
    cost_total_usd: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class InferenceProfiler:
    """
    Profiles Anthropic API calls for timing, token usage, and cost.

    Uses streaming to measure real TTFT and inter-token latencies.
    Appends each profile to a JSON file after every call.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        output_file: str = "profile_results.json",
        input_price_per_token: float = _INPUT_PRICE_PER_TOKEN,
        output_price_per_token: float = _OUTPUT_PRICE_PER_TOKEN,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.output_file = Path(output_file)
        self.input_price_per_token = input_price_per_token
        self.output_price_per_token = output_price_per_token
        self.client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def profile(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> InferenceProfile:
        """
        Run a streamed API call and return its full inference profile.

        TTFT = time from request dispatch to first text token.
        Inter-token times = gaps between consecutive text delta events.

        Args:
            prompt: User message to send.
            max_tokens: Maximum tokens to generate.
            system: Optional system prompt.

        Returns:
            InferenceProfile with timing, token, and cost data.

        Raises:
            anthropic.APIConnectionError: If the network request fails.
            anthropic.APIStatusError: If the API returns an error status.
        """
        messages: list[dict] = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response_parts: list[str] = []
        inter_token_times: list[float] = []
        ttft_ms: Optional[float] = None
        last_token_time: float = 0.0
        request_start = time.perf_counter()

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    now = time.perf_counter()
                    if ttft_ms is None:
                        ttft_ms = (now - request_start) * 1000
                        last_token_time = now
                    else:
                        inter_token_times.append((now - last_token_time) * 1000)
                        last_token_time = now
                    response_parts.append(event.delta.text)
            final_message = stream.get_final_message()

        total_latency_ms = (time.perf_counter() - request_start) * 1000
        input_tokens: int = final_message.usage.input_tokens
        output_tokens: int = final_message.usage.output_tokens
        cost_input = input_tokens * self.input_price_per_token
        cost_output = output_tokens * self.output_price_per_token
        avg_inter = (
            sum(inter_token_times) / len(inter_token_times)
            if inter_token_times
            else 0.0
        )
        throughput = (
            output_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0.0
        )

        profile = InferenceProfile(
            model=self.model,
            prompt=prompt,
            response_text="".join(response_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms if ttft_ms is not None else 0.0,
            total_latency_ms=total_latency_ms,
            inter_token_times_ms=inter_token_times,
            avg_inter_token_ms=avg_inter,
            throughput_tokens_per_sec=throughput,
            cost_input_usd=cost_input,
            cost_output_usd=cost_output,
            cost_total_usd=cost_input + cost_output,
        )

        self._save(profile)
        self._print_summary(profile)
        return profile

    def _save(self, profile: InferenceProfile) -> None:
        """Append the profile to the JSON results file."""
        results: list[dict] = []
        if self.output_file.exists():
            try:
                with open(self.output_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results = data
            except (json.JSONDecodeError, OSError):
                results = []
        results.append(asdict(profile))
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _print_summary(self, profile: InferenceProfile) -> None:
        """Print a formatted summary of the profile to stdout."""
        sep = "=" * 56
        print()
        print(sep)
        print(f"  INFERENCE PROFILE — {profile.model}")
        print(sep)
        print(f"  Timestamp:         {profile.timestamp}")
        print(f"  TTFT:              {profile.ttft_ms:>10.1f} ms")
        print(f"  Total latency:     {profile.total_latency_ms:>10.1f} ms")
        print(f"  Avg inter-token:   {profile.avg_inter_token_ms:>10.2f} ms")
        print(f"  Throughput:        {profile.throughput_tokens_per_sec:>10.1f} tok/s")
        print(f"  Input tokens:      {profile.input_tokens:>10,}")
        print(f"  Output tokens:     {profile.output_tokens:>10,}")
        print(f"  Input cost:        ${profile.cost_input_usd:.6f}")
        print(f"  Output cost:       ${profile.cost_output_usd:.6f}")
        print(f"  Total cost:        ${profile.cost_total_usd:.6f}")
        print(sep)


if __name__ == "__main__":
    profiler = InferenceProfiler(
        model="claude-haiku-4-5",
        output_file="profile_results.json",
        input_price_per_token=0.80 / 1_000_000,
        output_price_per_token=4.00 / 1_000_000,
    )
    print("Running inference profile...")
    result = profiler.profile(
        prompt="Explain in 3-5 sentences how transformer attention works.",
        max_tokens=256,
        system="You are a concise technical writer.",
    )
    print(f"\nResponse preview: {result.response_text[:120]}...")
    print(f"\nResults saved to: profile_results.json")
