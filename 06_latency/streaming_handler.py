"""
streaming_handler.py - Real-time streaming with latency measurement.

Wraps the Anthropic streaming API to provide per-token callbacks,
TTFT (Time to First Token) measurement, perceived-speed scoring,
and a side-by-side streaming vs blocking comparison.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    """
    Aggregated metrics for a completed streaming request.

    Attributes
    ----------
    full_response:
        The complete text returned by the model.
    ttft_ms:
        Time to first token in milliseconds.
    total_ms:
        Wall-clock time from request start to stream close.
    tokens_per_second:
        Output throughput (output_tokens / total_seconds).
    perceived_speed_score:
        1-10 score derived from TTFT; 10 = instant, 1 = very slow.
    input_tokens:
        Prompt tokens billed by the API.
    output_tokens:
        Generated tokens billed by the API.
    model:
        Model ID used for this request.
    """

    full_response: str
    ttft_ms: float
    total_ms: float
    tokens_per_second: float
    perceived_speed_score: int
    input_tokens: int
    output_tokens: int
    model: str


@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of streaming vs blocking for one prompt.

    In blocking mode the user perceives the full round-trip as latency
    because nothing appears until the response is complete.  In streaming
    mode the perceived latency is TTFT.

    Attributes
    ----------
    prompt:
        The prompt used for both calls.
    streaming:
        Full StreamResult from the streaming call.
    blocking_ttft_ms:
        Perceived latency in blocking mode (== blocking_total_ms).
    blocking_total_ms:
        Wall-clock time for the blocking call.
    perceived_speed_improvement_pct:
        ``(blocking_ttft - streaming_ttft) / blocking_ttft * 100``.
    """

    prompt: str
    streaming: StreamResult
    blocking_ttft_ms: float
    blocking_total_ms: float
    perceived_speed_improvement_pct: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class StreamingHandler:
    """
    Wraps Anthropic streaming calls with timing callbacks and metrics.

    Parameters
    ----------
    model:
        Anthropic model ID to use for all calls (default: claude-haiku-4-5).
    max_tokens:
        Upper bound on generated tokens per request.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perceived_speed_score(self, ttft_ms: float) -> int:
        """
        Map TTFT to a 1-10 perceived-speed score.

        Thresholds are based on human reaction-time research:
        <200 ms feels instant; >4 s feels broken.

        Parameters
        ----------
        ttft_ms:
            Time to first token in milliseconds.

        Returns
        -------
        int
            Score from 1 (very slow) to 10 (instant).
        """
        thresholds = [
            (200, 10),
            (400, 9),
            (700, 8),
            (1_000, 7),
            (1_500, 6),
            (2_500, 5),
            (4_000, 4),
            (6_000, 3),
            (9_000, 2),
        ]
        for ceiling, score in thresholds:
            if ttft_ms < ceiling:
                return score
        return 1

    def _score_bar(self, score: int, width: int = 10) -> str:
        """Return an ASCII progress bar for a 1-10 score."""
        filled = round(score * width / 10)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _safe(self, text: str) -> str:
        """Replace non-ASCII bytes so output never raises UnicodeEncodeError."""
        return text.encode("ascii", errors="replace").decode("ascii")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        on_first_token: Optional[Callable[[float], None]] = None,
        on_token: Optional[Callable[[str, float], None]] = None,
        on_complete: Optional[Callable[[str, float, int], None]] = None,
    ) -> StreamResult:
        """
        Stream a prompt and invoke callbacks as tokens arrive.

        Parameters
        ----------
        prompt:
            User message to send to the model.
        on_first_token:
            Called exactly once with ``ttft_ms`` when the first text delta
            arrives.
        on_token:
            Called for every token with ``(text, elapsed_ms)`` where
            ``elapsed_ms`` is measured from the start of the request.
        on_complete:
            Called after the stream closes with
            ``(full_response, total_ms, output_tokens)``.

        Returns
        -------
        StreamResult
            Aggregated timing and usage metrics.
        """
        chunks: List[str] = []
        ttft_ms: Optional[float] = None
        t_start = time.perf_counter()

        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                if ttft_ms is None:
                    ttft_ms = elapsed_ms
                    if on_first_token is not None:
                        on_first_token(ttft_ms)

                chunks.append(text)

                if on_token is not None:
                    on_token(text, elapsed_ms)

            final = stream.get_final_message()

        total_ms = (time.perf_counter() - t_start) * 1000

        # Guard against empty responses (e.g. max_tokens=0)
        if ttft_ms is None:
            ttft_ms = total_ms

        full_response = "".join(chunks)
        output_tokens: int = final.usage.output_tokens
        input_tokens: int = final.usage.input_tokens
        tps = output_tokens / (total_ms / 1000) if total_ms > 0 else 0.0

        result = StreamResult(
            full_response=full_response,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_second=tps,
            perceived_speed_score=self._perceived_speed_score(ttft_ms),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

        if on_complete is not None:
            on_complete(full_response, total_ms, output_tokens)

        return result

    def compare_streaming_vs_blocking(self, prompt: str) -> ComparisonResult:
        """
        Run the same prompt in streaming then blocking mode and compare.

        The key insight is that streaming dramatically lowers *perceived*
        latency (TTFT) even when total wall-clock time is similar, because
        users see content immediately rather than staring at a blank screen.

        Parameters
        ----------
        prompt:
            User message to evaluate in both modes.

        Returns
        -------
        ComparisonResult
            Metrics for both modes and the perceived-speed improvement
            expressed as a percentage.
        """
        streaming_result = self.stream(prompt)

        t_start = time.perf_counter()
        self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        blocking_total_ms = (time.perf_counter() - t_start) * 1000

        improvement_pct = (
            (blocking_total_ms - streaming_result.ttft_ms) / blocking_total_ms * 100
            if blocking_total_ms > 0
            else 0.0
        )

        return ComparisonResult(
            prompt=prompt,
            streaming=streaming_result,
            blocking_ttft_ms=blocking_total_ms,
            blocking_total_ms=blocking_total_ms,
            perceived_speed_improvement_pct=improvement_pct,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    handler = StreamingHandler(model="claude-haiku-4-5", max_tokens=300)

    prompts: List[str] = [
        "In one sentence, what is a neural network?",
        "List three benefits of streaming APIs for user experience.",
        "Write a haiku about network latency.",
        "What is the difference between p50 and p99 latency? Answer in 2 sentences.",
        "Give a one-paragraph summary of how transformer attention works.",
    ]

    results: List[StreamResult] = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 60}")
        label = prompt[:54] + "..." if len(prompt) > 57 else prompt
        print(f"  Request {i}/{len(prompts)}: {label}")
        print(f"{'=' * 60}")

        def _on_first_token(ttft: float) -> None:
            sys.stdout.write(f"  [TTFT {ttft:.0f}ms] ")
            sys.stdout.flush()

        def _on_token(text: str, elapsed: float) -> None:
            safe = text.encode("ascii", errors="replace").decode("ascii")
            sys.stdout.write(safe)
            sys.stdout.flush()

        r = handler.stream(prompt, on_first_token=_on_first_token, on_token=_on_token)
        results.append(r)

        # Newline after streamed content
        print()
        print(
            f"  total={r.total_ms:.0f}ms | "
            f"{r.tokens_per_second:.1f} tok/s | "
            f"score={r.perceived_speed_score}/10"
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    sep = "-" * 62
    print(f"\n{'=' * 62}")
    print("  STREAMING SUMMARY")
    print(f"{'=' * 62}")
    print(f"  {'#':<3}  {'TTFT':>8}  {'Total':>8}  {'Tok/s':>6}  {'Score':>5}  Bar")
    print(f"  {sep}")
    for idx, r in enumerate(results, 1):
        bar = handler._score_bar(r.perceived_speed_score)
        print(
            f"  {idx:<3}  {r.ttft_ms:>7.0f}ms  "
            f"{r.total_ms:>7.0f}ms  "
            f"{r.tokens_per_second:>6.1f}  "
            f"{r.perceived_speed_score:>4}/10  {bar}"
        )
    print(f"  {sep}")
    avg_ttft = sum(r.ttft_ms for r in results) / len(results)
    avg_tps = sum(r.tokens_per_second for r in results) / len(results)
    avg_score = sum(r.perceived_speed_score for r in results) / len(results)
    print(
        f"  {'avg':<3}  {avg_ttft:>7.0f}ms  "
        f"{'':>8}  {avg_tps:>6.1f}  {avg_score:>4.1f}/10"
    )

    # ------------------------------------------------------------------
    # Streaming vs blocking comparison
    # ------------------------------------------------------------------
    cmp_prompt = (
        "Explain why TTFT matters for user experience in 2-3 sentences."
    )
    print(f"\n{'=' * 62}")
    print("  STREAMING vs BLOCKING COMPARISON")
    print(f"{'=' * 62}")
    print(f"  Prompt : {cmp_prompt[:55]}...")
    print("  Running both modes... ", end="", flush=True)

    cmp = handler.compare_streaming_vs_blocking(cmp_prompt)
    print("done.")
    print()
    print(f"  {'Mode':<12}  {'Perceived TTFT':>16}  {'Total':>8}")
    print(f"  {'-' * 12}  {'-' * 16}  {'-' * 8}")
    print(
        f"  {'Streaming':<12}  {cmp.streaming.ttft_ms:>13.0f}ms  "
        f"{cmp.streaming.total_ms:>7.0f}ms"
    )
    print(
        f"  {'Blocking':<12}  {cmp.blocking_ttft_ms:>13.0f}ms  "
        f"{cmp.blocking_total_ms:>7.0f}ms"
    )
    print()
    print(
        f"  Perceived speed improvement: "
        f"{cmp.perceived_speed_improvement_pct:.1f}%  "
        f"(streaming shows first token {cmp.perceived_speed_improvement_pct:.0f}% sooner)"
    )
    print(
        f"  Streaming score : {cmp.streaming.perceived_speed_score}/10  "
        + handler._score_bar(cmp.streaming.perceived_speed_score)
    )
    blocking_score = handler._perceived_speed_score(cmp.blocking_ttft_ms)
    print(
        f"  Blocking score  : {blocking_score}/10  "
        + handler._score_bar(blocking_score)
    )
    print()
