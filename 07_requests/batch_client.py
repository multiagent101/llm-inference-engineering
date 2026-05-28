"""
batch_client.py - Anthropic Batch API client for non-urgent, cost-sensitive workloads.

The Message Batches API processes requests asynchronously (up to 24 hours) at
50 % of the standard per-token price.  This module wraps the SDK's batch
primitives with polling, result download, cost-savings reporting, and JSON
persistence.

Workflow
--------
1. ``create_batch``     — submit a list of prompts; receive a ``batch_id``.
2. ``poll_batch``       — single non-blocking status check.
3. ``get_results``      — download and decode results once the batch is "ended".
4. ``process_batch_sync`` — convenience wrapper that chains 1-3 with a text
                            progress bar and optional timeout.

Cost savings
------------
Standard-API cost  = input_tokens * input_price + output_tokens * output_price
Batch-API cost     = Standard-API cost * 0.50   (50 % discount)
Savings            = Standard cost - Batch cost
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
from anthropic.types.messages.message_batch_succeeded_result import (
    MessageBatchSucceededResult,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Pricing (USD per token)
# ---------------------------------------------------------------------------

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":          {"input": 0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    "claude-sonnet-4-5":         {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-6":         {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-opus-4-7":           {"input": 5.00 / 1_000_000, "output": 25.00 / 1_000_000},
}
_DEFAULT_INPUT_PRICE:  float = 3.00 / 1_000_000
_DEFAULT_OUTPUT_PRICE: float = 15.00 / 1_000_000

_BATCH_DISCOUNT: float = 0.50  # Batch API is 50 % cheaper than standard


def _standard_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _MODEL_PRICING.get(
        model,
        {"input": _DEFAULT_INPUT_PRICE, "output": _DEFAULT_OUTPUT_PRICE},
    )
    return pricing["input"] * input_tokens + pricing["output"] * output_tokens


def _batch_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return _standard_cost(model, input_tokens, output_tokens) * (1 - _BATCH_DISCOUNT)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BatchStatus:
    """
    Snapshot of a batch's processing state, returned by :meth:`BatchAPIClient.poll_batch`.

    Attributes
    ----------
    batch_id:
        The ``msgbatch_…`` identifier returned by the API.
    processing_status:
        ``"in_progress"`` | ``"canceling"`` | ``"ended"``.
    succeeded:
        Requests that completed successfully.
    errored:
        Requests that failed with an API error.
    canceled:
        Requests canceled before processing.
    expired:
        Requests not processed within the 24-hour window.
    processing:
        Requests still being processed.
    total:
        Total number of requests in the batch.
    created_at:
        UTC datetime when the batch was submitted.
    ended_at:
        UTC datetime when processing finished, or ``None`` if still running.
    expires_at:
        UTC datetime after which unprocessed requests expire (created_at + 24 h).
    results_url:
        HTTPS URL of the ``.jsonl`` results file; ``None`` until the batch ends.
    is_complete:
        ``True`` when ``processing_status == "ended"``.
    """

    batch_id:          str
    processing_status: str
    succeeded:         int
    errored:           int
    canceled:          int
    expired:           int
    processing:        int
    total:             int
    created_at:        datetime
    ended_at:          Optional[datetime]
    expires_at:        datetime
    results_url:       Optional[str]
    is_complete:       bool


@dataclass
class BatchResult:
    """
    Outcome for one request within a batch.

    Attributes
    ----------
    custom_id:
        The ``custom_id`` supplied when the batch was created (``req-0000``, …).
    prompt_index:
        Zero-based index of this request in the original ``prompts`` list.
    success:
        ``True`` if the request succeeded; ``False`` for errors / cancellations.
    text:
        Model-generated text, or ``None`` on failure.
    input_tokens:
        Prompt tokens billed (0 on failure).
    output_tokens:
        Generated tokens billed (0 on failure).
    standard_cost_usd:
        What this call would have cost at standard (non-batch) pricing.
    batch_cost_usd:
        Actual cost at 50 %-discounted batch pricing.
    savings_usd:
        ``standard_cost_usd - batch_cost_usd``.
    result_type:
        Raw result type from the API: ``"succeeded"`` | ``"errored"`` |
        ``"canceled"`` | ``"expired"``.
    error:
        Error description when ``success`` is ``False``.
    """

    custom_id:          str
    prompt_index:       int
    success:            bool
    text:               Optional[str]
    input_tokens:       int
    output_tokens:      int
    standard_cost_usd:  float
    batch_cost_usd:     float
    savings_usd:        float
    result_type:        str
    error:              Optional[str] = None


@dataclass
class SavingsReport:
    """
    Aggregate cost and savings summary for one batch run.

    Attributes
    ----------
    batch_id:
        Identifies the batch this report belongs to.
    model:
        Model used for all requests in the batch.
    total_requests:
        Total requests submitted.
    successful_requests:
        Requests that returned a model response.
    total_input_tokens:
        Sum of input tokens across all successful requests.
    total_output_tokens:
        Sum of output tokens across all successful requests.
    standard_cost_usd:
        Total cost had standard (real-time) API pricing been used.
    batch_cost_usd:
        Actual cost at batch pricing (50 % discount).
    savings_usd:
        Absolute dollar saving.
    savings_pct:
        Percentage saving (always 50 % when all requests succeed).
    elapsed_seconds:
        Wall-clock time from batch creation to result download.
    """

    batch_id:             str
    model:                str
    total_requests:       int
    successful_requests:  int
    total_input_tokens:   int
    total_output_tokens:  int
    standard_cost_usd:    float
    batch_cost_usd:       float
    savings_usd:          float
    savings_pct:          float
    elapsed_seconds:      float


# ---------------------------------------------------------------------------
# Helper: text progress bar
# ---------------------------------------------------------------------------


def _bar(filled: int, total: int, width: int = 24) -> str:
    """Return an ASCII progress bar, e.g. ``[################--------]  67%``."""
    if total == 0:
        return "[" + "-" * width + "]   0%"
    ratio     = filled / total
    filled_w  = round(ratio * width)
    bar_str   = "[" + "#" * filled_w + "-" * (width - filled_w) + "]"
    return f"{bar_str} {ratio * 100:5.1f}%"


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as ``Xm Ys`` or ``Ys``."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# BatchAPIClient
# ---------------------------------------------------------------------------


class BatchAPIClient:
    """
    Client for the Anthropic Message Batches API.

    Wraps ``anthropic.Anthropic().messages.batches`` with structured result
    types, cost-savings calculations, and file persistence.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var.
    output_dir:
        Directory where ``batch_results_{batch_id}.json`` files are written.
        Defaults to the same directory as this module file.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        resolved_key     = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client     = anthropic.Anthropic(api_key=resolved_key)
        self._output_dir = output_dir or Path(__file__).parent

        # In-memory registry: batch_id -> model (needed for cost calc)
        self._batch_models: dict[str, str] = {}

    # ------------------------------------------------------------------
    # create_batch
    # ------------------------------------------------------------------

    def create_batch(
        self,
        prompts:    list[str],
        model:      str = "claude-haiku-4-5",
        max_tokens: int = 1024,
    ) -> str:
        """
        Submit a list of prompts as a single batch request.

        Each prompt is assigned a deterministic ``custom_id`` of the form
        ``req-NNNN`` (zero-padded to four digits) so results can be matched
        back to the original prompt list.

        Parameters
        ----------
        prompts:
            Ordered list of user messages to send.
        model:
            Anthropic model ID to use for every request in the batch.
        max_tokens:
            Upper bound on generated tokens per request.

        Returns
        -------
        str
            The batch ID (``msgbatch_…``) assigned by the API.

        Raises
        ------
        anthropic.APIError
            On any HTTP-level failure from the Anthropic API.
        """
        if not prompts:
            raise ValueError("prompts must not be empty")

        requests = [
            {
                "custom_id": f"req-{i:04d}",
                "params": {
                    "model":    model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for i, prompt in enumerate(prompts)
        ]

        batch = self._client.messages.batches.create(requests=requests)
        self._batch_models[batch.id] = model
        return batch.id

    # ------------------------------------------------------------------
    # poll_batch
    # ------------------------------------------------------------------

    def poll_batch(
        self,
        batch_id:             str,
        poll_interval_seconds: int = 30,   # stored but not used here; callers use it
    ) -> BatchStatus:
        """
        Perform a single, non-blocking status check on *batch_id*.

        The ``poll_interval_seconds`` parameter is not used internally (this
        method does not sleep); it is accepted as a documentation hint for
        callers that implement their own polling loops.

        Parameters
        ----------
        batch_id:
            The ``msgbatch_…`` identifier returned by :meth:`create_batch`.
        poll_interval_seconds:
            Ignored here; passed through for caller-loop use.

        Returns
        -------
        BatchStatus
            Current state of the batch.

        Raises
        ------
        anthropic.NotFoundError
            If *batch_id* does not exist or has been deleted.
        """
        mb = self._client.messages.batches.retrieve(batch_id)
        rc = mb.request_counts

        return BatchStatus(
            batch_id          = mb.id,
            processing_status = mb.processing_status,
            succeeded         = rc.succeeded,
            errored           = rc.errored,
            canceled          = rc.canceled,
            expired           = rc.expired,
            processing        = rc.processing,
            total             = rc.succeeded + rc.errored + rc.canceled
                                + rc.expired + rc.processing,
            created_at        = mb.created_at,
            ended_at          = mb.ended_at,
            expires_at        = mb.expires_at,
            results_url       = mb.results_url,
            is_complete       = mb.processing_status == "ended",
        )

    # ------------------------------------------------------------------
    # get_results
    # ------------------------------------------------------------------

    def get_results(self, batch_id: str) -> list[BatchResult]:
        """
        Download and decode results for a completed batch.

        Results from the API are not ordered; this method re-orders them by
        ``prompt_index`` (derived from the ``custom_id`` ``req-NNNN`` format)
        so the returned list mirrors the original ``prompts`` ordering.

        Parameters
        ----------
        batch_id:
            Completed batch to fetch results for.

        Returns
        -------
        list[BatchResult]
            One ``BatchResult`` per request, sorted by ``prompt_index``.

        Raises
        ------
        RuntimeError
            If the batch has not yet ended.
        anthropic.NotFoundError
            If *batch_id* does not exist.
        """
        # Guard: verify the batch is actually done
        status = self.poll_batch(batch_id)
        if not status.is_complete:
            raise RuntimeError(
                f"Batch {batch_id} has not ended yet "
                f"(status={status.processing_status}).  "
                "Call get_results only after poll_batch returns is_complete=True."
            )

        model = self._batch_models.get(batch_id, "claude-haiku-4-5")
        results: list[BatchResult] = []

        for individual in self._client.messages.batches.results(batch_id):
            custom_id   = individual.custom_id
            result      = individual.result
            result_type = result.type

            # Derive the original list index from custom_id "req-NNNN"
            try:
                prompt_index = int(custom_id.split("-")[1])
            except (IndexError, ValueError):
                prompt_index = 0

            if result_type == "succeeded" and isinstance(result, MessageBatchSucceededResult):
                msg          = result.message
                text         = msg.content[0].text if msg.content else ""
                in_tok       = msg.usage.input_tokens
                out_tok      = msg.usage.output_tokens
                std_cost     = _standard_cost(model, in_tok, out_tok)
                bat_cost     = _batch_cost(model, in_tok, out_tok)

                results.append(BatchResult(
                    custom_id         = custom_id,
                    prompt_index      = prompt_index,
                    success           = True,
                    text              = text,
                    input_tokens      = in_tok,
                    output_tokens     = out_tok,
                    standard_cost_usd = std_cost,
                    batch_cost_usd    = bat_cost,
                    savings_usd       = std_cost - bat_cost,
                    result_type       = result_type,
                ))
            else:
                error_msg = (
                    str(result.error) if hasattr(result, "error") else result_type
                )
                results.append(BatchResult(
                    custom_id         = custom_id,
                    prompt_index      = prompt_index,
                    success           = False,
                    text              = None,
                    input_tokens      = 0,
                    output_tokens     = 0,
                    standard_cost_usd = 0.0,
                    batch_cost_usd    = 0.0,
                    savings_usd       = 0.0,
                    result_type       = result_type,
                    error             = error_msg,
                ))

        results.sort(key=lambda r: r.prompt_index)
        return results

    # ------------------------------------------------------------------
    # save_results
    # ------------------------------------------------------------------

    def save_results(
        self,
        batch_id:     str,
        results:      list[BatchResult],
        savings:      SavingsReport,
    ) -> Path:
        """
        Persist results and savings report to ``batch_results_{batch_id}.json``.

        Parameters
        ----------
        batch_id:
            Used to construct the output filename.
        results:
            List of :class:`BatchResult` objects from :meth:`get_results`.
        savings:
            :class:`SavingsReport` from :meth:`process_batch_sync` (or built
            manually via :meth:`build_savings_report`).

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        output_path = self._output_dir / f"batch_results_{batch_id}.json"
        payload = {
            "savings_report": asdict(savings),
            "results": [asdict(r) for r in results],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # build_savings_report
    # ------------------------------------------------------------------

    def build_savings_report(
        self,
        batch_id:        str,
        model:           str,
        results:         list[BatchResult],
        elapsed_seconds: float,
    ) -> SavingsReport:
        """
        Compute aggregate cost savings from a list of :class:`BatchResult`.

        Parameters
        ----------
        batch_id:
            Batch identifier for the report header.
        model:
            Model used in the batch.
        results:
            Individual results as returned by :meth:`get_results`.
        elapsed_seconds:
            Wall-clock time from batch creation to result retrieval.

        Returns
        -------
        SavingsReport
        """
        successful   = [r for r in results if r.success]
        in_tok_total = sum(r.input_tokens  for r in successful)
        out_tok_total= sum(r.output_tokens for r in successful)
        std_total    = sum(r.standard_cost_usd for r in successful)
        bat_total    = sum(r.batch_cost_usd    for r in successful)
        savings      = std_total - bat_total
        savings_pct  = (savings / std_total * 100) if std_total > 0 else 50.0

        return SavingsReport(
            batch_id             = batch_id,
            model                = model,
            total_requests       = len(results),
            successful_requests  = len(successful),
            total_input_tokens   = in_tok_total,
            total_output_tokens  = out_tok_total,
            standard_cost_usd    = std_total,
            batch_cost_usd       = bat_total,
            savings_usd          = savings,
            savings_pct          = savings_pct,
            elapsed_seconds      = elapsed_seconds,
        )

    # ------------------------------------------------------------------
    # process_batch_sync
    # ------------------------------------------------------------------

    def process_batch_sync(
        self,
        prompts:          list[str],
        model:            str  = "claude-haiku-4-5",
        max_tokens:       int  = 1024,
        poll_interval_seconds: int  = 30,
        timeout_minutes:  int  = 60,
    ) -> tuple[list[BatchResult], SavingsReport]:
        """
        End-to-end synchronous batch workflow with a text progress bar.

        Creates the batch, polls at ``poll_interval_seconds`` intervals until
        the batch is complete or ``timeout_minutes`` is reached, downloads
        results, computes savings, and persists everything to a JSON file.

        Parameters
        ----------
        prompts:
            List of user messages to process.
        model:
            Anthropic model ID.
        max_tokens:
            Upper bound on generated tokens per request.
        poll_interval_seconds:
            Seconds to sleep between status polls.
        timeout_minutes:
            Maximum total wait time before raising ``TimeoutError``.

        Returns
        -------
        tuple[list[BatchResult], SavingsReport]
            Results ordered by prompt index and an aggregate savings summary.

        Raises
        ------
        TimeoutError
            If the batch does not complete within ``timeout_minutes``.
        anthropic.APIError
            On any unrecoverable API failure.
        """
        sep = "=" * 66

        # --- Create ---
        print(f"\n{sep}")
        print(f"  Submitting {len(prompts)} prompts to Batch API  (model={model})")
        print(sep)

        t_start  = time.monotonic()
        batch_id = self.create_batch(prompts, model, max_tokens)

        print(f"  Batch created: {batch_id}")
        print(f"  Poll interval: {poll_interval_seconds}s  |  Timeout: {timeout_minutes}m\n")

        timeout_seconds = timeout_minutes * 60
        poll_n          = 0

        # --- Poll loop ---
        while True:
            elapsed = time.monotonic() - t_start

            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within "
                    f"{timeout_minutes} minutes."
                )

            poll_n += 1
            status  = self.poll_batch(batch_id, poll_interval_seconds)

            # Build progress line
            done_count = status.succeeded + status.errored + status.canceled + status.expired
            bar_str    = _bar(done_count, status.total)
            ts         = datetime.now(timezone.utc).strftime("%H:%M:%S")

            print(f"  [{ts} UTC]  Poll #{poll_n:<3}  elapsed={_fmt_elapsed(elapsed)}")
            print(f"    Status   : {status.processing_status}")
            print(f"    Progress : {bar_str}  ({done_count}/{status.total} done)")
            print(
                f"    Counts   : "
                f"{status.succeeded} succeeded  "
                f"{status.errored} errored  "
                f"{status.processing} processing  "
                f"{status.expired} expired"
            )

            if status.is_complete:
                print(f"    Batch ended after {_fmt_elapsed(elapsed)}.\n")
                break

            next_poll = min(poll_interval_seconds, max(1, int(timeout_seconds - elapsed)))
            print(f"    Next poll: {next_poll}s\n")
            time.sleep(next_poll)

        # --- Download results ---
        print(f"  Downloading results for {batch_id} ...")
        results = self.get_results(batch_id)

        elapsed_total = time.monotonic() - t_start

        # --- Savings ---
        savings = self.build_savings_report(batch_id, model, results, elapsed_total)

        # --- Persist ---
        out_path = self.save_results(batch_id, results, savings)
        print(f"  Results saved to: {out_path}\n")

        return results, savings


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PROMPTS = [
        "In one sentence, explain what the Anthropic Batch API is.",
        "What are three advantages of processing LLM requests in batches?",
        "Write a haiku about asynchronous computing.",
        "What is the typical use case for batch inference vs real-time inference?",
        "Summarise the trade-off between cost and latency in LLM APIs in two sentences.",
    ]

    _MODEL      = "claude-haiku-4-5"
    _MAX_TOKENS = 256
    _POLL_SEC   = 30

    sep = "=" * 66

    client = BatchAPIClient()

    # --- Run the full workflow ---
    results, savings = client.process_batch_sync(
        prompts               = _PROMPTS,
        model                 = _MODEL,
        max_tokens            = _MAX_TOKENS,
        poll_interval_seconds = _POLL_SEC,
        timeout_minutes       = 60,
    )

    # --- Per-request results ---
    print(sep)
    print("  PER-REQUEST RESULTS")
    print(sep)
    print(f"  {'#':<3}  {'ID':<10}  {'OK':>4}  {'IN tok':>8}  {'OUT tok':>8}  {'std $':>10}  {'batch $':>10}  {'save $':>10}")
    print(f"  {'-' * 3}  {'-' * 10}  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for r in results:
        ok = "ok" if r.success else "FAIL"
        print(
            f"  {r.prompt_index + 1:<3}  {r.custom_id:<10}  {ok:>4}  "
            f"{r.input_tokens:>8}  {r.output_tokens:>8}  "
            f"${r.standard_cost_usd:>9.6f}  ${r.batch_cost_usd:>9.6f}  ${r.savings_usd:>9.6f}"
        )

        if r.success and r.text:
            preview = r.text.replace("\n", " ")[:70]
            print(f"       Response: {preview}")
        elif not r.success:
            print(f"       Error   : {r.error}")

    # --- Savings summary ---
    print(f"\n{sep}")
    print("  COST SAVINGS REPORT  (Batch API vs Standard API)")
    print(sep)
    thin = "-" * 38

    print(f"  {'Batch ID':<32}  {savings.batch_id}")
    print(f"  {'Model':<32}  {savings.model}")
    print(f"  {'Total requests':<32}  {savings.total_requests}")
    print(f"  {'Successful requests':<32}  {savings.successful_requests}")
    print(f"  {thin}")
    print(f"  {'Total input tokens':<32}  {savings.total_input_tokens:>10,}")
    print(f"  {'Total output tokens':<32}  {savings.total_output_tokens:>10,}")
    print(f"  {thin}")
    print(f"  {'Standard API cost':<32}  ${savings.standard_cost_usd:>12.6f}")
    print(f"  {'Batch API cost':<32}  ${savings.batch_cost_usd:>12.6f}")
    print(f"  {'Savings (absolute)':<32}  ${savings.savings_usd:>12.6f}")
    print(f"  {'Savings (percent)':<32}  {savings.savings_pct:>11.1f}%")
    print(f"  {thin}")
    print(f"  {'Elapsed time':<32}  {_fmt_elapsed(savings.elapsed_seconds):>12}")
    print(f"{sep}\n")
