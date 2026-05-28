"""
11_observability/structured_logger.py

Structured JSON Lines (JSONL) logger for LLM API calls.

Features
--------
- PII scrubbing (email, phone, credit card) before storing the prompt hash
- Configurable sampling: 100% in dev, 10% in prod (reduces storage cost)
- Daily log rotation with configurable retention (default 30 days)
- Cross-file search with field-value filters and time window
- Error aggregation by type, model, and feature
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_REPORT_WIDTH = 72

# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?"                  # optional US country code
    r"(?:\([0-9]{3}\)|[0-9]{3})"           # area code
    r"[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"   # local number
)
# Generic 16-digit card: groups of 4 separated by space, dash, or nothing
_CC_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """One structured log record for an LLM API call."""

    timestamp: str               # ISO-8601 UTC with ms precision
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    prompt_hash: str             # SHA-256 of the original prompt (not stored)
    response_length: int         # characters in the response
    feature_name: str            # product feature that triggered the call
    user_id: str
    environment: str             # "dev" | "prod"
    error: Optional[str] = None  # exception type + message, or None


@dataclass
class ScrubResult:
    """Result of a PII-scrubbing pass."""

    scrubbed_text: str
    pii_types_found: list[str]   # e.g. ["email", "credit_card"]
    replacement_count: int


@dataclass
class ErrorSummary:
    """Aggregated error report returned by :meth:`get_error_summary`."""

    period_hours: int
    total_calls: int
    total_errors: int
    error_rate: float
    by_error_type: dict[str, int]
    by_model: dict[str, int]
    by_feature: dict[str, int]


# ---------------------------------------------------------------------------
# LLMStructuredLogger
# ---------------------------------------------------------------------------

class LLMStructuredLogger:
    """
    Structured JSONL logger for LLM API calls with PII protection and rotation.

    Log files are written to *log_dir* as ``llm_YYYY-MM-DD.jsonl``.  A new
    file is opened automatically when the UTC date changes.  Files older than
    *rotation_days* are deleted on each rotation.

    Args:
        log_dir:         Directory for JSONL files (created if absent).
        environment:     ``"dev"`` or ``"prod"``.  Controls the default
                         sample rate (1.0 vs 0.1) when *sample_rate* is
                         not given explicitly.
        sample_rate:     Fraction of calls to log (0.0 – 1.0).  *None*
                         means auto-select from *environment*.
        rotation_days:   Number of daily log files to retain.

    Example::

        logger = LLMStructuredLogger(environment="prod")
        logger.log_call(
            request_id="abc123", model="claude-haiku-4-5",
            prompt="Help me ...", input_tokens=120, output_tokens=80,
            cost_usd=0.000416, latency_ms=342.0, response_length=310,
            feature_name="search", user_id="u_001",
        )
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/llm",
        environment: str = "dev",
        sample_rate: Optional[float] = None,
        rotation_days: int = 30,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._env = environment
        self._sample_rate = (
            sample_rate
            if sample_rate is not None
            else (1.0 if environment == "dev" else 0.1)
        )
        self._rotation_days = rotation_days
        self._lock = threading.Lock()
        self._current_date: Optional[str] = None
        self._current_file: Optional[Path] = None
        # Session counters
        self._total_logged = 0
        self._total_sampled_out = 0
        self._total_pii_scrubs = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_call(
        self,
        request_id: str,
        model: str,
        prompt: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        response_length: int,
        feature_name: str,
        user_id: str,
        error: Optional[str] = None,
    ) -> Optional[LogEntry]:
        """
        Log one LLM API call.

        The *prompt* is PII-scrubbed before its SHA-256 hash is stored;
        the raw prompt text is never written to disk.

        Returns the :class:`LogEntry` that was persisted, or *None* if the
        call was dropped by the sampling filter.
        """
        # Sampling gate
        if random.random() > self._sample_rate:
            self._total_sampled_out += 1
            return None

        # PII scrub (hash always computed from the ORIGINAL prompt)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        scrub = self.scrub_pii(prompt)
        if scrub.replacement_count > 0:
            self._total_pii_scrubs += 1

        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            request_id=request_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 8),
            latency_ms=round(latency_ms, 2),
            prompt_hash=prompt_hash,
            response_length=response_length,
            feature_name=feature_name,
            user_id=user_id,
            environment=self._env,
            error=error,
        )
        self._write(entry)
        return entry

    @staticmethod
    def scrub_pii(text: str) -> ScrubResult:
        """
        Replace PII patterns in *text* with safe placeholders.

        Detects and replaces: email addresses, US phone numbers,
        16-digit payment card numbers.

        Returns:
            :class:`ScrubResult` with the sanitised text and a summary of
            what was found.
        """
        pii_types: list[str] = []
        count = 0

        # Credit cards first (16-digit sequences, before phones steal digits)
        result, n = _CC_RE.subn("[CREDIT_CARD]", text)
        if n:
            pii_types.append("credit_card")
            count += n
        text = result

        result, n = _EMAIL_RE.subn("[EMAIL]", text)
        if n:
            pii_types.append("email")
            count += n
        text = result

        result, n = _PHONE_RE.subn("[PHONE]", text)
        if n:
            pii_types.append("phone")
            count += n
        text = result

        return ScrubResult(
            scrubbed_text=text,
            pii_types_found=pii_types,
            replacement_count=count,
        )

    def search_logs(
        self,
        filters: dict[str, object],
        hours: int = 24,
    ) -> list[dict]:
        """
        Search log files for entries matching ALL supplied filters.

        Supported filter keys
        ---------------------
        Any top-level field of :class:`LogEntry` (e.g. ``model``,
        ``feature_name``, ``user_id``, ``environment``).

        Special keys:
        - ``"has_error"`` (bool) — True to match entries with an error.
        - ``"error_type"`` (str) — substring match against the error field.

        Args:
            filters: Key-value pairs that must all match.
            hours:   Look back this many hours from now.

        Returns:
            List of matching log entries as dicts (as stored in JSONL).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_date = cutoff.date()
        results: list[dict] = []

        for log_file in sorted(self._log_dir.glob("llm_*.jsonl")):
            # Skip files that predate the window by filename
            try:
                file_date_str = log_file.stem[4:]   # "llm_2026-05-28" -> date part
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
                if file_date < cutoff_date:
                    continue
            except ValueError:
                pass

            try:
                with open(log_file, encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        # Timestamp filter
                        ts_str = entry.get("timestamp", "")
                        if ts_str.endswith("Z"):
                            ts_str = ts_str[:-1] + "+00:00"
                        try:
                            ts = datetime.fromisoformat(ts_str)
                        except ValueError:
                            continue
                        if ts < cutoff:
                            continue

                        # Field filters
                        if not self._matches(entry, filters):
                            continue

                        results.append(entry)
            except OSError:
                pass

        return results

    def get_error_summary(self, hours: int = 24) -> ErrorSummary:
        """
        Aggregate all errors logged in the last *hours* hours.

        Returns:
            :class:`ErrorSummary` with counts by type, model, and feature.
        """
        all_entries = self.search_logs({}, hours=hours)
        error_entries = [e for e in all_entries if e.get("error") is not None]

        by_type: dict[str, int] = {}
        by_model: dict[str, int] = {}
        by_feature: dict[str, int] = {}

        for e in error_entries:
            raw_error = e.get("error", "unknown") or "unknown"
            # Use the part before the first colon as the error type
            err_type = raw_error.split(":")[0].strip()
            by_type[err_type] = by_type.get(err_type, 0) + 1
            by_model[e.get("model", "?")] = by_model.get(e.get("model", "?"), 0) + 1
            by_feature[e.get("feature_name", "?")] = (
                by_feature.get(e.get("feature_name", "?"), 0) + 1
            )

        total = len(all_entries)
        return ErrorSummary(
            period_hours=hours,
            total_calls=total,
            total_errors=len(error_entries),
            error_rate=len(error_entries) / total if total else 0.0,
            by_error_type=dict(
                sorted(by_type.items(), key=lambda x: x[1], reverse=True)
            ),
            by_model=dict(
                sorted(by_model.items(), key=lambda x: x[1], reverse=True)
            ),
            by_feature=dict(
                sorted(by_feature.items(), key=lambda x: x[1], reverse=True)
            ),
        )

    def session_stats(self) -> dict[str, object]:
        """Return counters accumulated during this logger session."""
        return {
            "total_logged": self._total_logged,
            "total_sampled_out": self._total_sampled_out,
            "total_pii_scrubs": self._total_pii_scrubs,
            "sample_rate": self._sample_rate,
            "environment": self._env,
            "log_dir": str(self._log_dir),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, entry: LogEntry) -> None:
        """Serialize *entry* to the current daily JSONL file."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._current_date:
                self._current_date = today
                self._current_file = self._log_dir / f"llm_{today}.jsonl"
                self._rotate()
            assert self._current_file is not None
            with open(self._current_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")
            self._total_logged += 1

    def _rotate(self) -> None:
        """Delete JSONL files older than *rotation_days*."""
        cutoff = (
            datetime.now(timezone.utc).date() - timedelta(days=self._rotation_days)
        )
        for f in self._log_dir.glob("llm_*.jsonl"):
            try:
                file_date = datetime.strptime(f.stem[4:], "%Y-%m-%d").date()
                if file_date < cutoff:
                    f.unlink()
            except (ValueError, OSError):
                pass

    @staticmethod
    def _matches(entry: dict, filters: dict[str, object]) -> bool:
        """Return True when *entry* satisfies ALL filters."""
        for key, value in filters.items():
            if key == "has_error":
                has = entry.get("error") is not None
                if bool(value) != has:
                    return False
            elif key == "error_type":
                err = entry.get("error") or ""
                if str(value) not in err:
                    return False
            else:
                if entry.get(key) != value:
                    return False
        return True


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6"]
_FEATURES = ["search", "chat", "summarize", "code_review", "translation"]
_USERS = [f"u_{i:03d}" for i in range(1, 11)]

# Errors: None appears more often to simulate a realistic error rate
_ERRORS: list[Optional[str]] = [
    None, None, None, None, None, None, None,
    "RateLimitError: 429 Too Many Requests",
    "TimeoutError: request timed out after 30s",
    "ValidationError: max_tokens exceeds model limit",
]

# Prompts with PII (mixed in among normal ones)
_PROMPTS = [
    "What is exponential backoff and why does it matter?",
    "Contact support@helpdesk.example.com about the billing issue",
    "Summarize this report on neural scaling laws",
    "Call our customer at +1-555-867-5309 to confirm the order",
    "Review this Python function for off-by-one errors",
    "Charge subscription to card 4532 1234 5678 9010",
    "Translate 'good morning' to Japanese, French, and Arabic",
    "User jane.smith@corp.org (card 5412-3456-7890-1234) needs support",
    "Explain the difference between precision and recall",
    "Send invoice to bob@invoices.net, phone 555.234.5678",
    "Write unit tests for a binary search implementation",
    "Escalate to manager at alice@mgmt.io, mobile +1 (800) 555-0199",
    "What are the main causes of LLM hallucinations?",
    "Help me debug this SQL query joining three tables",
    "Describe the CAP theorem in distributed systems",
]


def _sep(char: str = "=", w: int = _REPORT_WIDTH) -> str:
    return char * w


def _simulate_call(idx: int, rng: random.Random) -> dict:
    """Return a dict of kwargs for one simulated log_call."""
    model = rng.choice(_MODELS)
    in_tok = rng.randint(50, 500)
    out_tok = rng.randint(20, 400)
    price_in = 0.80 if "haiku" in model else 3.00
    price_out = 4.00 if "haiku" in model else 15.00
    cost = (in_tok * price_in + out_tok * price_out) / 1_000_000
    return {
        "request_id": f"req-{idx:04d}",
        "model": model,
        "prompt": rng.choice(_PROMPTS),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "latency_ms": rng.uniform(150.0, 2500.0),
        "response_length": rng.randint(50, 800),
        "feature_name": rng.choice(_FEATURES),
        "user_id": rng.choice(_USERS),
        "error": rng.choice(_ERRORS),
    }


if __name__ == "__main__":
    import shutil

    LOG_DIR = Path("logs/llm_demo")
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)   # clean slate for demo

    # ------------------------------------------------------------------
    # Section 0: PII scrubbing demo
    # ------------------------------------------------------------------
    pii_examples = [
        "Contact user at john.doe@example.com for the follow-up",
        "Callback number is +1-800-555-0100, ask for Sarah",
        "Charge account 4532 1234 5678 9010 for the annual plan",
        "Send to alice@company.org, card: 5412-3456-7890-1234, tel 555.867.5309",
        "No sensitive information in this prompt whatsoever",
    ]

    print(_sep("="))
    print("  PII SCRUBBING DEMO")
    print(_sep("="))
    print(
        f"  {'#':>2}  {'PII types found':<22}  {'Replacements':>12}  "
        f"Result"
    )
    print(_sep("-"))
    for i, raw in enumerate(pii_examples, 1):
        r = LLMStructuredLogger.scrub_pii(raw)
        types_str = ", ".join(r.pii_types_found) if r.pii_types_found else "none"
        print(f"  {i:>2}  {types_str:<22}  {r.replacement_count:>12}  {r.scrubbed_text}")
    print()

    # ------------------------------------------------------------------
    # Section 1: Log 50 simulated calls
    # ------------------------------------------------------------------
    logger = LLMStructuredLogger(
        log_dir=LOG_DIR,
        environment="dev",
        sample_rate=1.0,
    )

    rng = random.Random(42)
    print(_sep("="))
    print("  LOGGING 50 SIMULATED CALLS  (dev mode, 100% sample rate)")
    print(_sep("="))
    print(
        f"  {'#':>4}  {'Model':<22}  {'Feature':<12}  "
        f"{'Tok in':>6}  {'Tok out':>7}  {'ms':>7}  {'Error'}"
    )
    print(_sep("-"))

    for i in range(1, 51):
        kwargs = _simulate_call(i, rng)
        entry = logger.log_call(**kwargs)
        err_str = kwargs["error"].split(":")[0] if kwargs["error"] else "-"
        print(
            f"  {i:>4}  {kwargs['model']:<22}  {kwargs['feature_name']:<12}  "
            f"{kwargs['input_tokens']:>6}  {kwargs['output_tokens']:>7}  "
            f"{kwargs['latency_ms']:>6.0f}  {err_str}"
        )
    print()

    # ------------------------------------------------------------------
    # Section 2: Search — filter by feature
    # ------------------------------------------------------------------
    print(_sep("="))
    print("  SEARCH: feature_name='code_review'")
    print(_sep("="))
    code_calls = logger.search_logs({"feature_name": "code_review"}, hours=1)
    print(f"  Found {len(code_calls)} entries")
    for e in code_calls[:5]:
        err = e.get("error") or "-"
        print(
            f"    {e['request_id']}  model={e['model']:<22}"
            f"  latency={e['latency_ms']:.0f}ms  error={err}"
        )
    print()

    # ------------------------------------------------------------------
    # Section 3: Search — all errors
    # ------------------------------------------------------------------
    print(_sep("="))
    print("  SEARCH: has_error=True")
    print(_sep("="))
    error_calls = logger.search_logs({"has_error": True}, hours=1)
    print(f"  Found {len(error_calls)} error entries")
    for e in error_calls:
        print(
            f"    {e['request_id']}  feature={e['feature_name']:<12}"
            f"  error={e.get('error', '')}"
        )
    print()

    # ------------------------------------------------------------------
    # Section 4: Search — specific error type
    # ------------------------------------------------------------------
    print(_sep("="))
    print("  SEARCH: error_type='RateLimitError'")
    print(_sep("="))
    rl_calls = logger.search_logs({"error_type": "RateLimitError"}, hours=1)
    print(f"  Found {len(rl_calls)} RateLimitError entries")
    for e in rl_calls:
        print(
            f"    {e['request_id']}  model={e['model']:<22}"
            f"  user={e['user_id']}"
        )
    print()

    # ------------------------------------------------------------------
    # Section 5: Error summary
    # ------------------------------------------------------------------
    summary = logger.get_error_summary(hours=1)
    print(_sep("="))
    print("  ERROR SUMMARY  (last 1 hour)")
    print(_sep("="))
    w = 32
    print(f"  {'Total calls logged':<{w}} {summary.total_calls}")
    print(f"  {'Total errors':<{w}} {summary.total_errors}")
    print(f"  {'Error rate':<{w}} {summary.error_rate * 100:.1f}%")
    print()
    print(f"  By error type:")
    for k, v in summary.by_error_type.items():
        bar = "#" * v
        print(f"    {k:<35} {v:>3}  {bar}")
    print()
    print(f"  By model:")
    for k, v in summary.by_model.items():
        print(f"    {k:<35} {v:>3}")
    print()
    print(f"  By feature:")
    for k, v in summary.by_feature.items():
        print(f"    {k:<35} {v:>3}")
    print()

    # ------------------------------------------------------------------
    # Section 6: Session stats
    # ------------------------------------------------------------------
    stats = logger.session_stats()
    print(_sep("="))
    print("  SESSION STATS")
    print(_sep("="))
    for k, v in stats.items():
        print(f"  {k:<{w}} {v}")

    print()
    print(
        f"  Log file: "
        f"{next(LOG_DIR.glob('llm_*.jsonl'), 'none')}"
    )
    print(_sep("="))
