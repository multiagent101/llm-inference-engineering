from __future__ import annotations

"""
regression_detector.py - Automated quality regression detection for LLM models.

Maintains a fixed test set of 20 questions with expected-answer keywords.
On first run, establishes a baseline and persists it to disk. Subsequent runs
compare against the baseline using a chi-squared test and alert when the pass
rate drops below an acceptable threshold or regresses from baseline by more than
a configurable magnitude.

Key classes:
    TestItem            -- one immutable item in the fixed test set
    TestResult          -- per-item evaluation outcome
    RegressionReport    -- full report for one test run
    RegressionDetector  -- orchestrates all regression logic
"""

import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEN_MODEL: str = "claude-haiku-4-5"
_GEN_MAX_TOKENS: int = 256
_DEFAULT_PASS_THRESHOLD: float = 0.85   # run-level minimum pass rate
_DEFAULT_REGRESSION_DELTA: float = 0.10  # drop from baseline that flags regression
_SIGNIFICANCE_LEVEL: float = 0.05
_DEFAULT_BASELINE_PATH: Path = Path(__file__).parent / "regression_baseline.json"


# ---------------------------------------------------------------------------
# Fixed test set (20 items, never changes between runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestItem:
    """One immutable item in the fixed regression test set.

    Attributes:
        id:                Stable numeric identifier (used in overrides dict).
        question:          Prompt sent to the model under test.
        expected_keywords: Tuple of strings; response passes if any one is found
                           (case-insensitive substring match).
        category:          High-level group: factual | coding | math | reasoning.
    """

    id: int
    question: str
    expected_keywords: tuple[str, ...]
    category: str


_TEST_SET: tuple[TestItem, ...] = (
    # --- factual (6) ---
    TestItem(1,  "What is the chemical symbol for gold?",
             ("Au",), "factual"),
    TestItem(2,  "In what year did the first Moon landing occur?",
             ("1969",), "factual"),
    TestItem(3,  "What is the largest planet in the solar system?",
             ("Jupiter",), "factual"),
    TestItem(4,  "What is the chemical formula for water?",
             ("H2O",), "factual"),
    TestItem(5,  "Who developed the theory of general relativity?",
             ("Einstein",), "factual"),
    TestItem(6,  "Approximately how fast does light travel in kilometres per second?",
             ("300,000", "299,792", "300000"), "factual"),
    # --- coding (6) ---
    TestItem(7,  "How do you reverse a string in Python?",
             ("[::-1]", "reversed(", "slicing"), "coding"),
    TestItem(8,  "What is the time complexity of binary search?",
             ("O(log n)", "log n", "logarithmic"), "coding"),
    TestItem(9,  "How do you check whether a key exists in a Python dictionary?",
             (" in ", "get(", ".keys()"), "coding"),
    TestItem(10, "What does the built-in len() function return in Python?",
             ("length", "number of", "size", "count"), "coding"),
    TestItem(11, "How do you open and read a text file in Python?",
             ("open(", "with open", ".read("), "coding"),
    TestItem(12, "What is a Python list comprehension?",
             ("[", "for ", "iterable", "expression"), "coding"),
    # --- math (4) ---
    TestItem(13, "What is 17 multiplied by 13?",
             ("221",), "math"),
    TestItem(14, "What is the square root of 256?",
             ("16",), "math"),
    TestItem(15, "What is 30 percent of 150?",
             ("45",), "math"),
    TestItem(16, "What is 2 raised to the power of 10?",
             ("1024", "1,024"), "math"),
    # --- reasoning (4) ---
    TestItem(17, "A car travels at 60 mph for 2.5 hours. How far does it travel?",
             ("150",), "reasoning"),
    TestItem(18, "What is the next number in the sequence: 1, 1, 2, 3, 5, 8?",
             ("13",), "reasoning"),
    TestItem(19, "If today is Wednesday, what day will it be in exactly 10 days?",
             ("Saturday",), "reasoning"),
    TestItem(20, "A store sells 3 apples for $1. How many apples can you buy for $5?",
             ("15",), "reasoning"),
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Evaluation outcome for a single test item.

    Attributes:
        item_id:         Corresponds to TestItem.id.
        question:        The prompt that was evaluated.
        category:        Category copied from the test item.
        response:        Raw model output (may be a dry-run override in tests).
        passed:          True when any expected keyword was found in the response.
        matched_keyword: The keyword that triggered pass, or None on failure.
    """

    item_id: int
    question: str
    category: str
    response: str
    passed: bool
    matched_keyword: Optional[str]


@dataclass
class RegressionReport:
    """Complete report for one regression test run.

    Attributes:
        model:                     Model identifier used for generation.
        run_timestamp:             ISO-8601 UTC timestamp of the run.
        pass_rate:                 Fraction of items that passed (0.0-1.0).
        n_passed:                  Absolute count of passing items.
        n_total:                   Total items evaluated (always 20).
        results:                   Per-item TestResult list.
        failed_tests:              Subset of results where passed=False.
        regression_detected:       True when pass_rate or delta triggers an alert.
        is_statistically_significant: True when chi-squared p-value < 0.05.
        chi2_stat:                 Chi-squared statistic (Yates corrected, df=1).
        p_value:                   Two-tailed p-value from chi-squared test.
        comparison_vs_baseline:    Dict with keys: baseline_pass_rate,
                                   current_pass_rate, delta, baseline_timestamp,
                                   baseline_model.
        most_affected_category:    Category with the largest drop from baseline.
        alert_message:             Human-readable alert string, empty if no regression.
        by_category:               Dict mapping category -> {n_passed, n_total, pass_rate}.
    """

    model: str
    run_timestamp: str
    pass_rate: float
    n_passed: int
    n_total: int
    results: list[TestResult]
    failed_tests: list[TestResult]
    regression_detected: bool
    is_statistically_significant: bool
    chi2_stat: float
    p_value: float
    comparison_vs_baseline: dict
    most_affected_category: str
    alert_message: str
    by_category: dict[str, dict]


# ---------------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------------


def _chi2_two_proportions(
    n1_pass: int,
    n1_total: int,
    n2_pass: int,
    n2_total: int,
) -> tuple[float, float]:
    """Chi-squared test with Yates continuity correction for two proportions.

    Compares the pass rate from a baseline run (n1) with a current run (n2)
    using a 2x2 contingency table and df=1.

    Args:
        n1_pass:  Passing items in the baseline run.
        n1_total: Total items in the baseline run.
        n2_pass:  Passing items in the current run.
        n2_total: Total items in the current run.

    Returns:
        Tuple (chi2_statistic, p_value). Returns (0.0, 1.0) when the test
        cannot be computed (zero marginal totals).
    """
    a = n1_pass
    b = n1_total - n1_pass
    c = n2_pass
    d = n2_total - n2_pass
    n = a + b + c + d

    if n == 0:
        return 0.0, 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0

    ad_bc = abs(a * d - b * c)
    correction = n / 2.0
    if ad_bc <= correction:
        return 0.0, 1.0

    numerator = (ad_bc - correction) ** 2
    denominator = row1 * row2 * col1 * col2
    chi2 = n * numerator / denominator

    # p-value for chi-squared with df=1 via complementary error function
    # P(chi2_1 > x) = erfc(sqrt(x/2))
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return round(chi2, 4), round(p_value, 4)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RegressionDetector:
    """Detects LLM quality regressions against a fixed 20-item test set.

    On the first call to run_regression_test(), the results are persisted as
    the baseline. Every subsequent call compares against that baseline and
    flags a regression when:

        - pass_rate < pass_threshold (absolute floor), OR
        - pass_rate drops more than regression_delta from the baseline.

    Statistical significance is assessed via a chi-squared test (Yates
    correction, df=1, alpha=0.05). Note that with only 20 items the test has
    limited power; practical thresholds are therefore applied independently.

    Args:
        api_key:           Anthropic API key; falls back to ANTHROPIC_API_KEY.
        baseline_path:     File path for the persisted baseline JSON.
        pass_threshold:    Minimum acceptable pass rate (default 0.85).
        regression_delta:  Drop from baseline that triggers an alert (default 0.10).
        gen_model:         Model used to generate responses under test.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        baseline_path: Path = _DEFAULT_BASELINE_PATH,
        pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
        regression_delta: float = _DEFAULT_REGRESSION_DELTA,
        gen_model: str = _GEN_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._baseline_path = baseline_path
        self._pass_threshold = pass_threshold
        self._regression_delta = regression_delta
        self._gen_model = gen_model
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_regression_test(
        self,
        model: str,
        _dry_run_responses: Optional[dict[int, str]] = None,
    ) -> RegressionReport:
        """Evaluate the model on all 20 test items and return a regression report.

        If no baseline exists, this run is saved as the new baseline (pass_rate
        is recorded but regression_detected will be False).

        Args:
            model:                Model identifier to call for response generation.
            _dry_run_responses:   Optional dict mapping TestItem.id to a pre-written
                                  response string. Items present in this dict skip the
                                  API call and use the supplied string instead.
                                  Intended for simulation and offline testing only.

        Returns:
            RegressionReport with per-item results, chi-squared test outcome,
            and a human-readable alert_message if regression is detected.
        """
        overrides = _dry_run_responses or {}
        results: list[TestResult] = []

        for item in _TEST_SET:
            response = (
                overrides[item.id]
                if item.id in overrides
                else self._generate_response(model, item.question)
            )
            matched = self._find_keyword(response, item.expected_keywords)
            results.append(
                TestResult(
                    item_id=item.id,
                    question=item.question,
                    category=item.category,
                    response=response,
                    passed=matched is not None,
                    matched_keyword=matched,
                )
            )

        by_category = self._category_stats(results)
        n_passed = sum(1 for r in results if r.passed)
        n_total = len(results)
        pass_rate = n_passed / n_total
        failed = [r for r in results if not r.passed]
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        baseline = self._load_baseline()

        if baseline is None:
            self._save_baseline(model, timestamp, pass_rate, n_passed, n_total,
                                results, by_category)
            return RegressionReport(
                model=model,
                run_timestamp=timestamp,
                pass_rate=round(pass_rate, 4),
                n_passed=n_passed,
                n_total=n_total,
                results=results,
                failed_tests=failed,
                regression_detected=False,
                is_statistically_significant=False,
                chi2_stat=0.0,
                p_value=1.0,
                comparison_vs_baseline={
                    "baseline_pass_rate": pass_rate,
                    "current_pass_rate": pass_rate,
                    "delta": 0.0,
                    "baseline_timestamp": timestamp,
                    "baseline_model": model,
                },
                most_affected_category="-",
                alert_message="",
                by_category=by_category,
            )

        bl_pass_rate = baseline["pass_rate"]
        bl_n_passed = baseline["n_passed"]
        bl_n_total = baseline["n_total"]
        delta = pass_rate - bl_pass_rate

        chi2, p_val = _chi2_two_proportions(bl_n_passed, bl_n_total,
                                             n_passed, n_total)
        significant = p_val < _SIGNIFICANCE_LEVEL

        regression_detected = (
            pass_rate < self._pass_threshold
            or delta < -self._regression_delta
        )

        most_affected = self._most_affected_category(
            baseline.get("by_category", {}), by_category
        )

        comparison = {
            "baseline_pass_rate": bl_pass_rate,
            "current_pass_rate": round(pass_rate, 4),
            "delta": round(delta, 4),
            "baseline_timestamp": baseline["timestamp"],
            "baseline_model": baseline["model"],
        }

        if regression_detected:
            pct_drop = abs(delta) * 100
            sig_note = " (statistically significant)" if significant else ""
            alert = (
                f"REGRESSION ALERT: quality dropped {pct_drop:.0f}pp across "
                f"{len(failed)} test cases. "
                f"Most affected category: {most_affected}.{sig_note}"
            )
        else:
            alert = ""

        return RegressionReport(
            model=model,
            run_timestamp=timestamp,
            pass_rate=round(pass_rate, 4),
            n_passed=n_passed,
            n_total=n_total,
            results=results,
            failed_tests=failed,
            regression_detected=regression_detected,
            is_statistically_significant=significant,
            chi2_stat=chi2,
            p_value=p_val,
            comparison_vs_baseline=comparison,
            most_affected_category=most_affected,
            alert_message=alert,
            by_category=by_category,
        )

    def schedule_check(
        self,
        model: str,
        interval_hours: float = 6.0,
        on_regression: Optional[Callable[[RegressionReport], None]] = None,
    ) -> threading.Event:
        """Run regression tests periodically in a background thread.

        The first test run is deferred until the first interval elapses, so
        calling schedule_check() followed immediately by stop_event.set()
        makes no API calls.

        Args:
            model:          Model to test on each cycle.
            interval_hours: Time between test runs. Defaults to 6 hours.
            on_regression:  Optional callback invoked whenever a regression is
                            detected. Receives the full RegressionReport. If
                            None, regressions are silently recorded in the
                            baseline history.

        Returns:
            A threading.Event. Call stop_event.set() to stop the scheduler
            cleanly. The background thread is a daemon and will not prevent
            the process from exiting.
        """
        self._stop_event.clear()
        stop = self._stop_event

        def _worker() -> None:
            while True:
                # Sleep first so the caller can stop before the first run.
                if stop.wait(interval_hours * 3600):
                    break
                report = self.run_regression_test(model)
                if report.regression_detected and on_regression is not None:
                    on_regression(report)

        thread = threading.Thread(
            target=_worker, daemon=True, name="RegressionDetector"
        )
        thread.start()
        return stop

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_response(self, model: str, question: str) -> str:
        """Call the model and return the raw response text."""
        message = self._client.messages.create(
            model=model,
            max_tokens=_GEN_MAX_TOKENS,
            messages=[{"role": "user", "content": question}],
        )
        return message.content[0].text

    @staticmethod
    def _find_keyword(
        response: str, keywords: tuple[str, ...]
    ) -> Optional[str]:
        """Return the first keyword found in response (case-insensitive), or None."""
        rl = response.lower()
        for kw in keywords:
            if kw.lower() in rl:
                return kw
        return None

    @staticmethod
    def _category_stats(results: list[TestResult]) -> dict[str, dict]:
        """Compute per-category pass/fail counts and pass_rate."""
        cats: dict[str, dict] = {}
        for r in results:
            entry = cats.setdefault(r.category, {"n_passed": 0, "n_total": 0})
            entry["n_total"] += 1
            if r.passed:
                entry["n_passed"] += 1
        for entry in cats.values():
            entry["pass_rate"] = round(entry["n_passed"] / entry["n_total"], 4)
        return cats

    @staticmethod
    def _most_affected_category(
        baseline_cats: dict[str, dict],
        current_cats: dict[str, dict],
    ) -> str:
        """Return the category name with the largest drop in pass_rate from baseline."""
        worst_cat = "-"
        worst_drop = 0.0
        for cat, cur in current_cats.items():
            bl_rate = baseline_cats.get(cat, {}).get("pass_rate", cur["pass_rate"])
            drop = bl_rate - cur["pass_rate"]
            if drop > worst_drop:
                worst_drop = drop
                worst_cat = cat
        return worst_cat

    def _load_baseline(self) -> Optional[dict]:
        """Load the persisted baseline; return None if no file exists."""
        if not self._baseline_path.exists():
            return None
        return json.loads(self._baseline_path.read_text(encoding="utf-8"))

    def _save_baseline(
        self,
        model: str,
        timestamp: str,
        pass_rate: float,
        n_passed: int,
        n_total: int,
        results: list[TestResult],
        by_category: dict[str, dict],
    ) -> None:
        """Serialise the baseline to disk as JSON."""
        data = {
            "model": model,
            "timestamp": timestamp,
            "pass_rate": pass_rate,
            "n_passed": n_passed,
            "n_total": n_total,
            "by_category": by_category,
            "results": [
                {
                    "id": r.item_id,
                    "question": r.question,
                    "category": r.category,
                    "response": r.response,
                    "passed": r.passed,
                }
                for r in results
            ],
        }
        self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SEP = "=" * 70

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env or environment.")
        sys.exit(1)

    # Use a dedicated demo baseline so repeated runs start fresh.
    demo_baseline = Path(__file__).parent / "regression_baseline_demo.json"
    if demo_baseline.exists():
        demo_baseline.unlink()

    detector = RegressionDetector(
        api_key=api_key,
        baseline_path=demo_baseline,
        gen_model=_GEN_MODEL,
    )

    # ------------------------------------------------------------------
    # Phase 1: establish baseline
    # ------------------------------------------------------------------
    print(_SEP)
    print(f"  PHASE 1: ESTABLISHING BASELINE  (model: {_GEN_MODEL})")
    print(_SEP)
    print(f"  Running {len(_TEST_SET)} test items via real API calls...")
    print()

    baseline_report = detector.run_regression_test(_GEN_MODEL)

    print(f"  Overall  : {baseline_report.n_passed}/{baseline_report.n_total} passed"
          f"  ({baseline_report.pass_rate:.0%})")
    print()
    print("  By category:")
    for cat in sorted(baseline_report.by_category):
        s = baseline_report.by_category[cat]
        bar = "#" * s["n_passed"]
        print(f"    {cat:<12} {s['n_passed']}/{s['n_total']}  {bar}")
    print()

    if baseline_report.failed_tests:
        print("  Items that failed at baseline:")
        for r in baseline_report.failed_tests:
            print(f"    [{r.item_id:02d}] ({r.category}) {r.question[:58]}")
        print()

    print(f"  Baseline saved: {demo_baseline.name}")

    # ------------------------------------------------------------------
    # Phase 2: simulate regression on 5 items
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  PHASE 2: SIMULATING REGRESSION  (5 injected failures)")
    print(_SEP)

    # Items chosen to cover all four categories.
    # IDs: 3=factual, 8=coding, 13=math, 17=reasoning, 20=reasoning
    regressed_ids = {3, 8, 13, 17, 20}
    _BAD_RESPONSE = "I'm sorry, I don't have enough information to answer that."

    print(f"  Degrading test IDs: {sorted(regressed_ids)}")
    categories_hit = {
        item.category
        for item in _TEST_SET
        if item.id in regressed_ids
    }
    print(f"  Affected categories: {', '.join(sorted(categories_hit))}")
    print()

    # Build the full dry-run dict: real baseline responses + bad overrides.
    # This avoids additional API calls for the 15 non-regressed items.
    baseline_data = json.loads(demo_baseline.read_text(encoding="utf-8"))
    dry_run: dict[int, str] = {
        r["id"]: r["response"]
        for r in baseline_data["results"]
        if r["id"] not in regressed_ids
    }
    for rid in regressed_ids:
        dry_run[rid] = _BAD_RESPONSE

    regression_report = detector.run_regression_test(
        _GEN_MODEL, _dry_run_responses=dry_run
    )

    bl_rate = regression_report.comparison_vs_baseline["baseline_pass_rate"]
    cur_rate = regression_report.pass_rate
    delta_pp = (cur_rate - bl_rate) * 100

    print(f"  Baseline pass rate  : {bl_rate:.0%}  ({baseline_report.n_passed}/{baseline_report.n_total})")
    print(f"  Current pass rate   : {cur_rate:.0%}  ({regression_report.n_passed}/{regression_report.n_total})")
    print(f"  Delta               : {delta_pp:+.0f} percentage points")
    print(f"  Chi-squared         : {regression_report.chi2_stat:.3f}"
          f"  (p={regression_report.p_value:.4f})")
    sig_label = "yes" if regression_report.is_statistically_significant else (
        "no (small N limits power -- practical thresholds still apply)"
    )
    print(f"  Significant at 0.05 : {sig_label}")
    print()

    print("  By category (baseline -> current):")
    bl_cats = baseline_data.get("by_category", {})
    for cat in sorted(regression_report.by_category):
        cur = regression_report.by_category[cat]
        bl_r = bl_cats.get(cat, {}).get("pass_rate", cur["pass_rate"])
        drop = bl_r - cur["pass_rate"]
        marker = "  <-- REGRESSED" if drop > 0 else ""
        print(
            f"    {cat:<12} {bl_r:.0%} -> {cur['pass_rate']:.0%}"
            f"  ({cur['n_passed']}/{cur['n_total']}){marker}"
        )

    print()
    print("  Failed tests:")
    for r in regression_report.failed_tests:
        injected = " [injected]" if r.item_id in regressed_ids else " [pre-existing]"
        print(f"    [{r.item_id:02d}] ({r.category}){injected}  {r.question[:50]}")

    print()
    print(f"  Regression detected : {regression_report.regression_detected}")
    print(f"  Most affected       : {regression_report.most_affected_category}")
    print()
    if regression_report.regression_detected:
        print(f"  *** {regression_report.alert_message} ***")
    else:
        print("  No regression detected.")

    # ------------------------------------------------------------------
    # Phase 3: schedule_check demonstration
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  PHASE 3: SCHEDULE_CHECK  (interval_hours=6)")
    print(_SEP)
    print("  Configuration:")
    print(f"    Model            : {_GEN_MODEL}")
    print(f"    Interval         : 6 hours")
    print(f"    Pass threshold   : {detector._pass_threshold:.0%}")
    print(f"    Regression delta : {detector._regression_delta:.0%}")
    print()

    alerts_received: list[str] = []

    def _handle_regression(report: RegressionReport) -> None:
        alerts_received.append(report.alert_message)

    stop_event = detector.schedule_check(
        _GEN_MODEL,
        interval_hours=6.0,
        on_regression=_handle_regression,
    )
    print("  Scheduler started (daemon thread, sleeps 6h before first run).")
    print("  Stopping immediately to avoid API calls in demo...")
    stop_event.set()
    time.sleep(0.1)  # let the thread exit cleanly
    print("  Scheduler stopped.")
    print()
    print("  In production, schedule_check() would:")
    print("    - Run regression tests every 6 hours in the background")
    print("    - Invoke on_regression() callback if quality drops")
    print("    - Send alerts to PagerDuty / Slack via the callback")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  DEMO SUMMARY")
    print(_SEP)
    print(f"  Test set size         : {len(_TEST_SET)} items")
    print(f"  Baseline pass rate    : {bl_rate:.0%}")
    print(f"  Post-regression rate  : {cur_rate:.0%}  ({abs(delta_pp):.0f}pp drop)")
    print(f"  Regression detected   : {regression_report.regression_detected}")
    print(f"  Most affected cat.    : {regression_report.most_affected_category}")
    print(f"  Alert message         : {regression_report.alert_message or '(none)'}")
    print()
