"""
sla_monitor.py - SLA compliance monitoring for LLM inference latency.

Provides two operating modes:
- Real-time: ``check(latency_ms)`` evaluates a rolling window of recent
  measurements and fires an alert when any percentile breaches its threshold.
- Historical: ``get_compliance_report(hours=N)`` reads latency_data.json
  (written by LatencyProfiler), buckets measurements by hour, and reports
  the fraction of hours each SLA tier was met.

All violations are persisted atomically to sla_violations.json.

Standard-library only: json, math, random, datetime, dataclasses.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEFAULT_DATA_FILE = Path(__file__).parent / "latency_data.json"
_DEFAULT_VIOLATIONS_FILE = Path(__file__).parent / "sla_violations.json"

# Minimum rolling-window samples before each percentile tier is evaluated.
# Prevents false positives during warm-up.
_MIN_SAMPLES: Dict[str, int] = {"p50": 5, "p90": 10, "p99": 50}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SLAStatus:
    """
    Result of a single SLA compliance check against the rolling window.

    Attributes
    ----------
    is_compliant:
        True when no percentile threshold was breached.
    violated_percentile:
        The worst-violated tier ("p50", "p90", "p99") or None.
    action_required:
        Escalation level derived from the violated tier:
        "none" | "investigate" | "escalate" | "page_oncall".
    message:
        Human-readable description of the check result.
    latency_ms:
        The raw latency value that triggered this check.
    timestamp:
        ISO-8601 UTC timestamp of the check.
    window_p50_ms / window_p90_ms / window_p99_ms:
        Rolling-window percentile values at check time (None if too few samples).
    """

    is_compliant: bool
    violated_percentile: Optional[str]
    action_required: str
    message: str
    latency_ms: float
    timestamp: str
    window_p50_ms: Optional[float]
    window_p90_ms: Optional[float]
    window_p99_ms: Optional[float]


@dataclass
class BucketSummary:
    """
    Percentile metrics for one hourly compliance bucket.

    Attributes
    ----------
    bucket_start:
        Human-readable bucket label (e.g. "2026-05-27 06:00 UTC").
    count:
        Number of measurements in this bucket.
    p50_ms / p90_ms / p99_ms:
        Computed percentiles for the bucket.
    p50_ok / p90_ok / p99_ok:
        True when the corresponding percentile is within threshold.
    """

    bucket_start: str
    count: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p50_ok: bool
    p90_ok: bool
    p99_ok: bool

    @property
    def all_ok(self) -> bool:
        """True when every SLA tier is within threshold for this bucket."""
        return self.p50_ok and self.p90_ok and self.p99_ok


@dataclass
class ComplianceReport:
    """
    SLA compliance summary over a configurable look-back window.

    Attributes
    ----------
    period_hours:
        The look-back window used to generate this report.
    total_measurements:
        Measurements found within the look-back window.
    total_violations:
        Count of violations persisted to sla_violations.json.
    p50_compliance_pct / p90_compliance_pct / p99_compliance_pct:
        Percentage of hourly buckets where each tier was within threshold.
    overall_compliance_pct:
        Percentage of hourly buckets where ALL tiers were within threshold.
    worst_p50_ms / worst_p90_ms / worst_p99_ms:
        Peak per-bucket percentile values observed.
    bucket_details:
        Per-hour breakdown (ordered chronologically).
    """

    period_hours: int
    total_measurements: int
    total_violations: int
    p50_compliance_pct: float
    p90_compliance_pct: float
    p99_compliance_pct: float
    overall_compliance_pct: float
    worst_p50_ms: float
    worst_p90_ms: float
    worst_p99_ms: float
    bucket_details: List[BucketSummary]


# ---------------------------------------------------------------------------
# SLAMonitor
# ---------------------------------------------------------------------------


class SLAMonitor:
    """
    Monitors LLM inference latency against configurable SLA thresholds.

    Parameters
    ----------
    p50_ms:
        Median latency SLA threshold in milliseconds (default 1 200).
    p90_ms:
        90th-percentile SLA threshold in milliseconds (default 3 000).
    p99_ms:
        99th-percentile SLA threshold in milliseconds (default 5 000).
    window_minutes:
        Rolling window size for real-time ``check()`` evaluations (default 60).
    data_file:
        Path to the JSON file written by ``LatencyProfiler``.
        Defaults to ``06_latency/latency_data.json``.
    violations_file:
        Path where SLA violations are persisted.
        Defaults to ``06_latency/sla_violations.json``.
    """

    def __init__(
        self,
        p50_ms: float = 1_200.0,
        p90_ms: float = 3_000.0,
        p99_ms: float = 5_000.0,
        window_minutes: int = 60,
        data_file: Optional[Path] = None,
        violations_file: Optional[Path] = None,
    ) -> None:
        self.p50_ms = p50_ms
        self.p90_ms = p90_ms
        self.p99_ms = p99_ms
        self.window_minutes = window_minutes
        self._data_file = data_file or _DEFAULT_DATA_FILE
        self._violations_file = violations_file or _DEFAULT_VIOLATIONS_FILE

        # Rolling window: list of (utc_datetime, latency_ms)
        self._window: List[Tuple[datetime, float]] = []
        self._violations: List[dict] = []
        self._load_violations()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_data: List[float], p: float) -> float:
        """
        Linear-interpolation percentile (mirrors LatencyProfiler method).

        Parameters
        ----------
        sorted_data:
            Pre-sorted list of float values.
        p:
            Percentile to compute (0–100).

        Returns
        -------
        float
            Interpolated percentile value.
        """
        n = len(sorted_data)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_data[0]
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac

    def _prune_window(self, now: datetime) -> None:
        """Drop measurements older than window_minutes from the rolling buffer."""
        cutoff = now - timedelta(minutes=self.window_minutes)
        self._window = [(ts, v) for ts, v in self._window if ts >= cutoff]

    def _window_percentiles(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute (p50, p90, p99) from the current rolling window.

        Returns ``None`` for any tier that lacks the minimum sample count.
        """
        values = sorted(v for _, v in self._window)
        n = len(values)
        p50 = self._percentile(values, 50) if n >= _MIN_SAMPLES["p50"] else None
        p90 = self._percentile(values, 90) if n >= _MIN_SAMPLES["p90"] else None
        p99 = self._percentile(values, 99) if n >= _MIN_SAMPLES["p99"] else None
        return p50, p90, p99

    @staticmethod
    def _action_for(violated: Optional[str]) -> str:
        """Map violated tier to the appropriate on-call action."""
        return {
            None: "none",
            "p50": "investigate",
            "p90": "escalate",
            "p99": "page_oncall",
        }[violated]

    def _load_violations(self) -> None:
        """Load existing violations from disk (called once at init)."""
        if not self._violations_file.exists():
            return
        try:
            data = json.loads(self._violations_file.read_text(encoding="utf-8"))
            self._violations = data.get("violations", [])
        except (json.JSONDecodeError, OSError):
            self._violations = []

    def _save_violations(self) -> None:
        """Persist violations atomically via a .tmp swap."""
        tmp = self._violations_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"violations": self._violations}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._violations_file)

    def _load_latency_data(self) -> List[dict]:
        """
        Load raw measurement dicts from latency_data.json.

        Returns an empty list if the file is absent or malformed.
        """
        if not self._data_file.exists():
            return []
        try:
            data = json.loads(self._data_file.read_text(encoding="utf-8"))
            return data.get("measurements", [])
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _compliance_bar(pct: float, width: int = 12) -> str:
        """Return an ASCII progress bar representing a compliance percentage."""
        filled = round(pct * width / 100.0)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, latency_ms: float) -> SLAStatus:
        """
        Record a measurement and evaluate rolling-window SLA compliance.

        Adds ``latency_ms`` to the in-memory rolling window, prunes entries
        older than ``window_minutes``, then computes current percentiles and
        compares them against thresholds.  The worst-violated tier determines
        the action level (p99 > p90 > p50).

        Violations are persisted to ``sla_violations.json`` automatically.
        Insufficient data (below ``_MIN_SAMPLES``) is treated as compliant
        to avoid false positives at startup.

        Parameters
        ----------
        latency_ms:
            Total response latency of the incoming measurement in milliseconds.

        Returns
        -------
        SLAStatus
            Compliance result with rolling-window percentiles and action level.
        """
        now = datetime.now(timezone.utc)
        self._window.append((now, latency_ms))
        self._prune_window(now)

        p50, p90, p99 = self._window_percentiles()
        n = len(self._window)

        # Evaluate worst-violated tier (p99 takes precedence over p90 over p50)
        violated: Optional[str] = None
        if p99 is not None and p99 > self.p99_ms:
            violated = "p99"
        elif p90 is not None and p90 > self.p90_ms:
            violated = "p90"
        elif p50 is not None and p50 > self.p50_ms:
            violated = "p50"

        action = self._action_for(violated)

        if violated is None:
            p50_label = f"p50={p50:.0f}ms" if p50 is not None else "p50=n/a"
            message = f"SLA OK ({p50_label}, window n={n})"
        else:
            actual: float = {"p50": p50, "p90": p90, "p99": p99}[violated]  # type: ignore[index]
            threshold: float = {
                "p50": self.p50_ms,
                "p90": self.p90_ms,
                "p99": self.p99_ms,
            }[violated]
            message = (
                f"SLA VIOLATION: {violated}={actual:.0f}ms"
                f" exceeds {threshold:.0f}ms (window n={n})"
            )

        status = SLAStatus(
            is_compliant=violated is None,
            violated_percentile=violated,
            action_required=action,
            message=message,
            latency_ms=round(latency_ms, 2),
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            window_p50_ms=round(p50, 2) if p50 is not None else None,
            window_p90_ms=round(p90, 2) if p90 is not None else None,
            window_p99_ms=round(p99, 2) if p99 is not None else None,
        )

        if not status.is_compliant:
            self._violations.append(asdict(status))
            self._save_violations()

        return status

    def get_compliance_report(self, hours: int = 24) -> ComplianceReport:
        """
        Generate a compliance report for the last ``hours`` hours.

        Reads historical measurements from ``latency_data.json``, groups
        them into hourly buckets, computes p50/p90/p99 per bucket via
        linear interpolation, and summarises the fraction of hours each
        SLA tier was met.

        Parameters
        ----------
        hours:
            Look-back window in hours (default: 24).

        Returns
        -------
        ComplianceReport
            Per-bucket breakdown and aggregate compliance percentages.
        """
        raw = self._load_latency_data()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)

        entries: List[Tuple[datetime, float]] = []
        for m in raw:
            try:
                ts_str: str = m.get("timestamp", "")
                # Accept both "Z" suffix and "+00:00" offset
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    entries.append((ts, float(m["total_ms"])))
            except (KeyError, ValueError, TypeError):
                continue

        if not entries:
            return ComplianceReport(
                period_hours=hours,
                total_measurements=0,
                total_violations=len(self._violations),
                p50_compliance_pct=0.0,
                p90_compliance_pct=0.0,
                p99_compliance_pct=0.0,
                overall_compliance_pct=0.0,
                worst_p50_ms=0.0,
                worst_p90_ms=0.0,
                worst_p99_ms=0.0,
                bucket_details=[],
            )

        # Group into hourly buckets keyed by truncated datetime
        buckets: Dict[datetime, List[float]] = {}
        for ts, v in entries:
            key = ts.replace(minute=0, second=0, microsecond=0)
            buckets.setdefault(key, []).append(v)

        bucket_details: List[BucketSummary] = []
        worst_p50 = worst_p90 = worst_p99 = 0.0

        for key in sorted(buckets):
            vals = sorted(buckets[key])
            p50 = self._percentile(vals, 50)
            p90 = self._percentile(vals, 90)
            p99 = self._percentile(vals, 99)
            worst_p50 = max(worst_p50, p50)
            worst_p90 = max(worst_p90, p90)
            worst_p99 = max(worst_p99, p99)
            bucket_details.append(
                BucketSummary(
                    bucket_start=key.strftime("%Y-%m-%d %H:%M UTC"),
                    count=len(vals),
                    p50_ms=round(p50, 1),
                    p90_ms=round(p90, 1),
                    p99_ms=round(p99, 1),
                    p50_ok=p50 <= self.p50_ms,
                    p90_ok=p90 <= self.p90_ms,
                    p99_ok=p99 <= self.p99_ms,
                )
            )

        nb = len(bucket_details)
        p50_pct = sum(1 for b in bucket_details if b.p50_ok) / nb * 100
        p90_pct = sum(1 for b in bucket_details if b.p90_ok) / nb * 100
        p99_pct = sum(1 for b in bucket_details if b.p99_ok) / nb * 100
        overall_pct = sum(1 for b in bucket_details if b.all_ok) / nb * 100

        return ComplianceReport(
            period_hours=hours,
            total_measurements=len(entries),
            total_violations=len(self._violations),
            p50_compliance_pct=p50_pct,
            p90_compliance_pct=p90_pct,
            p99_compliance_pct=p99_pct,
            overall_compliance_pct=overall_pct,
            worst_p50_ms=round(worst_p50, 1),
            worst_p90_ms=round(worst_p90, 1),
            worst_p99_ms=round(worst_p99, 1),
            bucket_details=bucket_details,
        )

    def print_report(self, report: ComplianceReport) -> None:
        """
        Print a formatted compliance report to stdout.

        Parameters
        ----------
        report:
            The ``ComplianceReport`` returned by ``get_compliance_report()``.
        """
        sep = "=" * 72
        thin = "-" * 72

        print(sep)
        print("  SLA COMPLIANCE REPORT")
        print(
            f"  Period    : last {report.period_hours} hours"
            f"   ({report.total_measurements} measurements,"
            f" {len(report.bucket_details)} hourly buckets)"
        )
        print(
            f"  Thresholds: p50<={self.p50_ms:.0f}ms  "
            f"p90<={self.p90_ms:.0f}ms  "
            f"p99<={self.p99_ms:.0f}ms"
        )
        print(sep)
        print(f"  Violations logged : {report.total_violations}")
        print()

        def _row(tier: str, worst: float, threshold: float, pct: float) -> str:
            flag = "OK  " if worst <= threshold else "FAIL"
            bar = self._compliance_bar(pct)
            return (
                f"  {flag}  {tier:<4}  worst={worst:>8.0f}ms"
                f"  limit={threshold:>6.0f}ms  {pct:>6.1f}%  {bar}"
            )

        print(f"  {'':4}  {'Tier':<4}  {'Worst observed':>16}  {'Limit':>12}  {'Compli':>6}  Bar")
        print(f"  {thin}")
        print(_row("p50", report.worst_p50_ms, self.p50_ms, report.p50_compliance_pct))
        print(_row("p90", report.worst_p90_ms, self.p90_ms, report.p90_compliance_pct))
        print(_row("p99", report.worst_p99_ms, self.p99_ms, report.p99_compliance_pct))
        print(f"  {thin}")
        bar = self._compliance_bar(report.overall_compliance_pct)
        print(
            f"  Overall (all tiers): {report.overall_compliance_pct:.1f}%  {bar}"
        )
        print()

        if report.bucket_details:
            print(f"  HOURLY BREAKDOWN  ({len(report.bucket_details)} buckets)")
            print(f"  {thin}")
            print(
                f"  {'Bucket':<22}  {'n':>4}  "
                f"{'p50':>8}  {'p90':>8}  {'p99':>8}  Status"
            )
            print(f"  {thin}")
            for b in report.bucket_details:
                flags = ""
                if not b.p50_ok:
                    flags += " p50!"
                if not b.p90_ok:
                    flags += " p90!"
                if not b.p99_ok:
                    flags += " p99!"
                status = "OK" if b.all_ok else "VIOLATION:" + flags
                print(
                    f"  {b.bucket_start:<22}  {b.count:>4}  "
                    f"{b.p50_ms:>7.0f}ms  {b.p90_ms:>7.0f}ms  {b.p99_ms:>7.0f}ms  "
                    f"{status}"
                )

        print(sep)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    rng = random.Random(42)
    now_utc = datetime.now(timezone.utc)

    data_file = Path(__file__).parent / "latency_data.json"
    violations_file = Path(__file__).parent / "sla_violations.json"

    # Clear stale violations from previous runs
    if violations_file.exists():
        violations_file.unlink()

    # ------------------------------------------------------------------
    # Phase 1 – Generate 24h of simulated measurements
    # ------------------------------------------------------------------
    # Regime windows (hour-of-day as float, exclusive upper bound):
    #   06:00-07:00  degraded  -> p90 violations expected
    #   12:00-12:30  incident  -> p99 violations expected
    #   18:00-19:00  degraded  -> p90 violations expected
    _DEGRADED: List[Tuple[float, float]] = [(6.0, 7.0), (18.0, 19.0)]
    _INCIDENT: List[Tuple[float, float]] = [(12.0, 12.5)]

    print("Generating 24h simulated latency data (1 440 measurements, 1/min)...")
    measurements: List[dict] = []
    models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]
    endpoints = ["chat", "summarise", "classify", "extract"]

    for minute in range(1440):
        ts = now_utc - timedelta(minutes=1440 - minute)
        h = ts.hour + ts.minute / 60.0

        in_incident = any(lo <= h < hi for lo, hi in _INCIDENT)
        in_degraded = any(lo <= h < hi for lo, hi in _DEGRADED)

        if in_incident:
            mu, sigma = math.log(8_000), 0.35
        elif in_degraded:
            mu, sigma = math.log(3_200), 0.50
        else:
            mu, sigma = math.log(800), 0.45

        total_ms = rng.lognormvariate(mu, sigma)
        ttft_ms = total_ms * rng.uniform(0.25, 0.45)
        tokens = max(1, int(rng.lognormvariate(math.log(350), 0.45)))
        tps = tokens / (total_ms / 1_000)

        measurements.append(
            {
                "ttft_ms": round(ttft_ms, 2),
                "total_ms": round(total_ms, 2),
                "tokens_output": tokens,
                "model": rng.choice(models),
                "endpoint": rng.choice(endpoints),
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tokens_per_second": round(tps, 2),
            }
        )

    tmp = data_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"measurements": measurements}, indent=2), encoding="utf-8"
    )
    tmp.replace(data_file)
    print(f"  Written {len(measurements)} records to {data_file.name}")
    print("  Degraded windows : hours 06-07 and 18-19  (mean ~3200ms, p90 violations)")
    print("  Incident window  : hour  12:00-12:30       (mean ~8000ms, p99 violations)")

    # ------------------------------------------------------------------
    # Phase 2 – Real-time check() demo (200 sequential calls)
    # ------------------------------------------------------------------
    monitor = SLAMonitor(
        p50_ms=1_200,
        p90_ms=3_000,
        p99_ms=5_000,
        window_minutes=60,
        data_file=data_file,
        violations_file=violations_file,
    )

    print()
    print("=" * 72)
    print("  REAL-TIME check() DEMO  (200 calls, rolling 60-min window)")
    print("=" * 72)
    print("  Thresholds: p50<=1200ms  p90<=3000ms  p99<=5000ms")
    print()

    phases: List[Tuple[str, int, float, float]] = [
        ("Normal   (n=100, mean~800ms) ", 100, math.log(800),   0.45),
        ("Degraded (n= 50, mean~3200ms)", 50,  math.log(3_200), 0.50),
        ("Incident (n= 50, mean~8000ms)", 50,  math.log(8_000), 0.35),
    ]

    total_calls = 0
    for phase_label, n, mu, sigma in phases:
        print(f"  Phase: {phase_label}")
        first_violation: Optional[SLAStatus] = None
        for _ in range(n):
            total_calls += 1
            ms = rng.lognormvariate(mu, sigma)
            status = monitor.check(ms)
            if not status.is_compliant and first_violation is None:
                first_violation = status

        if first_violation is not None:
            tier = first_violation.violated_percentile
            actual_val = {
                "p50": first_violation.window_p50_ms,
                "p90": first_violation.window_p90_ms,
                "p99": first_violation.window_p99_ms,
            }[tier]
            print(f"    [ALERT] {first_violation.message}")
            print(f"    Action : {first_violation.action_required}")
        else:
            wp50, _, _ = monitor._window_percentiles()
            p50_str = f"{wp50:.0f}ms" if wp50 is not None else "n/a"
            print(f"    All {n} checks passed  (window p50={p50_str})")
        print()

    print(f"  Total check() calls : {total_calls}")
    print(f"  Total violations    : {len(monitor._violations)}")
    print(f"  Violations file     : {violations_file.name}")

    # ------------------------------------------------------------------
    # Phase 3 – Historical compliance report (reads latency_data.json)
    # ------------------------------------------------------------------
    print()
    report = monitor.get_compliance_report(hours=24)
    monitor.print_report(report)
