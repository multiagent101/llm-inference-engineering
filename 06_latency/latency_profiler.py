#!/usr/bin/env python3
"""
latency_profiler.py -- Production latency measurement with percentile statistics.

Records per-request latency (TTFT + total response time), computes percentile
reports (p50–p99.9), draws text histograms, identifies outliers via IQR fencing,
and fires configurable alerts when p99 exceeds a threshold. All measurements are
persisted to JSON for cross-session aggregation.

Standard-library only: statistics, json, datetime, math, random.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── persistence ────────────────────────────────────────────────────────────
_DEFAULT_DATA_FILE = Path(__file__).parent / "latency_data.json"

# ── histogram bucket definitions: (lower_ms, upper_ms, label) ─────────────
_BUCKETS: List[Tuple[float, float, str]] = [
    (0,      500,          "     0-  500ms"),
    (500,   1_000,         "   500- 1000ms"),
    (1_000,  2_000,        "  1000- 2000ms"),
    (2_000,  3_000,        "  2000- 3000ms"),
    (3_000,  5_000,        "  3000- 5000ms"),
    (5_000,  8_000,        "  5000- 8000ms"),
    (8_000,  float("inf"), "       8000ms+"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """One recorded LLM request latency entry."""

    ttft_ms: float           # time-to-first-token
    total_ms: float          # end-to-end response time
    tokens_output: int       # output tokens generated
    model: str
    endpoint: str            # logical feature / route name
    timestamp: str           # ISO-8601 UTC
    tokens_per_second: float  # tokens_output / (total_ms / 1000)


@dataclass
class PercentileReport:
    """Percentile statistics computed from a measurement set."""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float
    # Time-to-first-token
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    # Total latency
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float            # p99.9
    # Throughput
    avg_tokens_per_second: float
    # Alert
    alert_triggered: bool
    alert_threshold_ms: float
    alert_message: str


@dataclass
class OutlierSummary:
    """IQR-fence outlier detection result."""

    count: int
    total_requests: int
    threshold_ms: float       # Q3 + multiplier * IQR
    threshold_multiplier: float
    pct_of_total: float
    outliers: List[Measurement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: List[float], p: float) -> float:
    """
    Compute percentile ``p`` using linear interpolation on sorted data.

    Parameters
    ----------
    sorted_data : list[float]
        Values sorted in ascending order.  Must be non-empty.
    p : float
        Percentile in the range [0, 100].

    Returns
    -------
    float
    """
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    idx  = (p / 100.0) * (n - 1)
    lo   = int(idx)
    hi   = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def _bucket_index(value_ms: float) -> int:
    for i, (lo, hi, _) in enumerate(_BUCKETS):
        if lo <= value_ms < hi:
            return i
    return len(_BUCKETS) - 1


# ---------------------------------------------------------------------------
# LatencyProfiler
# ---------------------------------------------------------------------------

class LatencyProfiler:
    """
    Production-grade latency profiler with percentile statistics and outlier
    detection.

    Measurements are stored in memory and, when ``auto_save=True``, written to
    a JSON file after every ``record()`` call so multiple process runs are
    aggregated automatically.  On initialisation the profiler loads any
    existing data from that file.

    Parameters
    ----------
    alert_threshold_ms : float
        Fire an alert when the computed p99 exceeds this value.
        Default 5000 ms.
    data_file : Path or str, optional
        Path for JSON persistence.  Defaults to ``latency_data.json`` in the
        same directory as this script.
    auto_save : bool
        Persist every measurement to ``data_file`` immediately.  Default True.
    on_alert : callable, optional
        Callback receiving the ``PercentileReport`` when an alert fires.
        Prints to stdout when ``None``.
    """

    def __init__(
        self,
        alert_threshold_ms: float = 5_000.0,
        data_file: Optional[Path] = None,
        auto_save: bool = True,
        on_alert: Optional[Callable[[PercentileReport], None]] = None,
    ) -> None:
        self.alert_threshold_ms = alert_threshold_ms
        self._data_file         = Path(data_file) if data_file else _DEFAULT_DATA_FILE
        self.auto_save          = auto_save
        self._on_alert          = on_alert
        self._alert_active      = False
        self._measurements: List[Measurement] = []

        if auto_save and self._data_file.exists():
            self._measurements = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> List[Measurement]:
        """Load measurements from the JSON file; silently returns [] on error."""
        try:
            with open(self._data_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [Measurement(**m) for m in raw.get("measurements", [])]
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return []

    def _save(self) -> None:
        """Atomically write all measurements to the JSON file."""
        payload = {
            "alert_threshold_ms": self.alert_threshold_ms,
            "total_measurements": len(self._measurements),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "measurements": [asdict(m) for m in self._measurements],
        }
        tmp = self._data_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(self._data_file)

    # ------------------------------------------------------------------
    # Alert
    # ------------------------------------------------------------------

    def _fire_alert(self, report: PercentileReport) -> None:
        if self._on_alert:
            self._on_alert(report)
        else:
            print(
                f"\n  [LATENCY ALERT]  p99={report.p99_ms:.0f}ms exceeds "
                f"threshold={self.alert_threshold_ms:.0f}ms  "
                f"(n={report.count})\n"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        ttft_ms: float,
        total_ms: float,
        tokens_output: int,
        model: str,
        endpoint: str,
    ) -> None:
        """
        Record one request's latency measurements.

        Parameters
        ----------
        ttft_ms : float
            Milliseconds from request dispatch to first output token.
        total_ms : float
            Milliseconds from request dispatch to full response.
        tokens_output : int
            Number of output tokens generated.
        model : str
            Model identifier (e.g. ``"claude-haiku-4-5"``).
        endpoint : str
            Logical feature or route name (e.g. ``"chat"``, ``"summarise"``).
        """
        tps = tokens_output / (total_ms / 1000.0) if total_ms > 0 else 0.0
        entry = Measurement(
            ttft_ms=round(ttft_ms, 2),
            total_ms=round(total_ms, 2),
            tokens_output=tokens_output,
            model=model,
            endpoint=endpoint,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tokens_per_second=round(tps, 2),
        )
        self._measurements.append(entry)

        if self.auto_save:
            self._save()

        # Alert check — only meaningful once we have enough data points
        if len(self._measurements) >= 20:
            report = self.get_percentiles()
            if report.alert_triggered and not self._alert_active:
                self._alert_active = True
                self._fire_alert(report)
            elif not report.alert_triggered:
                self._alert_active = False   # reset so alert can re-fire if p99 spikes again

    def get_percentiles(self) -> PercentileReport:
        """
        Compute percentile statistics for all recorded measurements.

        Returns
        -------
        PercentileReport
            p50, p75, p90, p95, p99, p99.9 for total latency plus TTFT p50/p95/p99.
        """
        n = len(self._measurements)
        if n == 0:
            return PercentileReport(
                count=0, min_ms=0, max_ms=0, mean_ms=0, stddev_ms=0,
                ttft_p50_ms=0, ttft_p95_ms=0, ttft_p99_ms=0,
                p50_ms=0, p75_ms=0, p90_ms=0, p95_ms=0, p99_ms=0, p999_ms=0,
                avg_tokens_per_second=0,
                alert_triggered=False,
                alert_threshold_ms=self.alert_threshold_ms,
                alert_message="",
            )

        totals = sorted(m.total_ms for m in self._measurements)
        ttfts  = sorted(m.ttft_ms  for m in self._measurements)

        mean_v   = statistics.mean(totals)
        stddev_v = statistics.stdev(totals) if n > 1 else 0.0
        p99      = _percentile(totals, 99)
        alert    = p99 > self.alert_threshold_ms

        return PercentileReport(
            count=n,
            min_ms=totals[0],
            max_ms=totals[-1],
            mean_ms=mean_v,
            stddev_ms=stddev_v,
            ttft_p50_ms=_percentile(ttfts, 50),
            ttft_p95_ms=_percentile(ttfts, 95),
            ttft_p99_ms=_percentile(ttfts, 99),
            p50_ms=_percentile(totals, 50),
            p75_ms=_percentile(totals, 75),
            p90_ms=_percentile(totals, 90),
            p95_ms=_percentile(totals, 95),
            p99_ms=p99,
            p999_ms=_percentile(totals, 99.9),
            avg_tokens_per_second=statistics.mean(
                m.tokens_per_second for m in self._measurements
            ),
            alert_triggered=alert,
            alert_threshold_ms=self.alert_threshold_ms,
            alert_message=(
                f"p99={p99:.0f}ms > threshold={self.alert_threshold_ms:.0f}ms"
                if alert else ""
            ),
        )

    def get_distribution(self, bar_width: int = 38) -> str:
        """
        Build a text histogram of total latency distribution.

        Parameters
        ----------
        bar_width : int
            Maximum character width for histogram bars.  Default 38.

        Returns
        -------
        str
            Multi-line histogram string ready for printing.
        """
        n = len(self._measurements)
        if n == 0:
            return "  (no measurements)"

        totals = [m.total_ms for m in self._measurements]
        counts = [0] * len(_BUCKETS)
        for v in totals:
            counts[_bucket_index(v)] += 1

        max_cnt = max(counts) if any(counts) else 1
        sep     = "  " + "-" * 64
        lines   = [
            f"\n  Latency Distribution  (total response time,  n={n})",
            sep,
            f"  {'Bucket':<16} {'Count':>6}  {'Pct':>5}   Bar",
            sep,
        ]
        for (_lo, _hi, label), cnt in zip(_BUCKETS, counts):
            pct     = cnt / n * 100
            bar_len = round(cnt / max_cnt * bar_width)
            bar     = "#" * bar_len
            lines.append(f"  {label}  {cnt:>5}  {pct:>4.1f}%   {bar}")
        lines.append(sep)
        return "\n".join(lines)

    def identify_outliers(
        self,
        threshold_multiplier: float = 2.5,
    ) -> OutlierSummary:
        """
        Identify anomalous requests using IQR fencing.

        A request is flagged when ``total_ms > Q3 + threshold_multiplier * IQR``.
        Standard box-plot uses 1.5; 2.5 gives a more conservative fence that
        catches only the most extreme values.

        Parameters
        ----------
        threshold_multiplier : float
            Fence multiplier applied to the inter-quartile range.  Default 2.5.

        Returns
        -------
        OutlierSummary
        """
        n = len(self._measurements)
        if n < 4:
            return OutlierSummary(
                count=0, total_requests=n,
                threshold_ms=0.0, threshold_multiplier=threshold_multiplier,
                pct_of_total=0.0,
            )

        totals = sorted(m.total_ms for m in self._measurements)
        q1     = _percentile(totals, 25)
        q3     = _percentile(totals, 75)
        fence  = q3 + threshold_multiplier * (q3 - q1)

        flagged = sorted(
            (m for m in self._measurements if m.total_ms > fence),
            key=lambda m: m.total_ms,
            reverse=True,
        )
        return OutlierSummary(
            count=len(flagged),
            total_requests=n,
            threshold_ms=fence,
            threshold_multiplier=threshold_multiplier,
            pct_of_total=len(flagged) / n * 100,
            outliers=flagged,
        )

    def print_report(self) -> None:
        """Print the full profiler report: percentiles, histogram, outliers."""
        if not self._measurements:
            print("  LatencyProfiler: no measurements recorded.")
            return

        report  = self.get_percentiles()
        outlier = self.identify_outliers()
        n       = report.count
        W       = 68
        SEP     = "=" * W
        SEP2    = "-" * W

        # ── header ────────────────────────────────────────────────────
        print()
        print(SEP)
        print("  LATENCY PROFILER REPORT")
        print(SEP2)
        print(f"  Measurements:      {n}")
        ts0 = self._measurements[0].timestamp[:19].replace("T", " ")
        ts1 = self._measurements[-1].timestamp[:19].replace("T", " ")
        print(f"  Period:            {ts0}  to  {ts1} UTC")
        print(f"  Alert threshold:   {self.alert_threshold_ms:.0f} ms")
        if self.auto_save:
            print(f"  Data file:         {self._data_file}")

        # ── model mix ─────────────────────────────────────────────────
        model_counts: Dict[str, int] = {}
        for m in self._measurements:
            model_counts[m.model] = model_counts.get(m.model, 0) + 1
        print()
        print(SEP2)
        print("  MODEL MIX")
        print(SEP2)
        for mdl, cnt in sorted(model_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {mdl:<26}  {cnt:>4}  ({cnt / n * 100:.0f}%)")

        # ── TTFT ──────────────────────────────────────────────────────
        print()
        print(SEP2)
        print("  TIME TO FIRST TOKEN (TTFT)")
        print(SEP2)
        print(f"  p50   {report.ttft_p50_ms:>8.0f} ms")
        print(f"  p95   {report.ttft_p95_ms:>8.0f} ms")
        print(f"  p99   {report.ttft_p99_ms:>8.0f} ms")

        # ── total latency ─────────────────────────────────────────────
        print()
        print(SEP2)
        print("  TOTAL LATENCY PERCENTILES")
        print(SEP2)
        print(f"  min   {report.min_ms:>8.0f} ms")
        print(f"  mean  {report.mean_ms:>8.0f} ms   stddev={report.stddev_ms:.0f} ms")
        print(f"  p50   {report.p50_ms:>8.0f} ms")
        print(f"  p75   {report.p75_ms:>8.0f} ms")
        print(f"  p90   {report.p90_ms:>8.0f} ms")
        print(f"  p95   {report.p95_ms:>8.0f} ms")
        alert_tag = f"   <-- ALERT: {report.alert_message}" if report.alert_triggered else ""
        print(f"  p99   {report.p99_ms:>8.0f} ms{alert_tag}")
        print(f"  p99.9 {report.p999_ms:>8.0f} ms")
        print(f"  max   {report.max_ms:>8.0f} ms")
        print(f"\n  Avg throughput:    {report.avg_tokens_per_second:.1f} tok/s")

        # ── histogram ─────────────────────────────────────────────────
        print(self.get_distribution())

        # ── outliers ──────────────────────────────────────────────────
        print()
        print(SEP2)
        print(
            f"  OUTLIERS  "
            f"(IQR fence x{outlier.threshold_multiplier}  >  "
            f"{outlier.threshold_ms:.0f} ms)"
        )
        print(SEP2)
        if outlier.count == 0:
            print("  No outliers detected.")
        else:
            print(
                f"  {outlier.count} outliers  "
                f"({outlier.pct_of_total:.1f}% of requests):"
            )
            print(
                f"  {'#':>4}  {'total_ms':>9}  {'ttft_ms':>8}  "
                f"{'tokens':>6}  {'model':<22}  endpoint"
            )
            print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*22}  {'-'*10}")
            for i, m in enumerate(outlier.outliers[:12], 1):
                print(
                    f"  {i:>4}  {m.total_ms:>8.0f}ms"
                    f"  {m.ttft_ms:>7.0f}ms"
                    f"  {m.tokens_output:>6}"
                    f"  {m.model:<22}"
                    f"  {m.endpoint}"
                )
            if outlier.count > 12:
                print(f"         ... and {outlier.count - 12} more")

        if report.alert_triggered:
            print()
            print(f"  *** ALERT ACTIVE: {report.alert_message} ***")

        print(SEP)
        print()


# ---------------------------------------------------------------------------
# Simulation helpers for __main__
# ---------------------------------------------------------------------------

def _sim_total_ms(rng: random.Random) -> float:
    """
    Log-normal latency with 1% extreme outliers.

    Distribution targets:
      p50  ~1 200 ms  (median of the log-normal)
      p99  ~8 700 ms  (theoretical 99th pctile without outliers)
      2 outliers at 15 000 – 25 000 ms in 200 samples
    """
    mu    = math.log(1_200)
    sigma = 0.85
    val   = rng.lognormvariate(mu, sigma)
    if rng.random() < 0.01:           # ~1% extreme outliers
        val = rng.uniform(15_000, 25_000)
    return val


def _sim_ttft_ms(total_ms: float, rng: random.Random) -> float:
    """TTFT is 30–50% of total response time with small jitter."""
    ratio = rng.uniform(0.30, 0.50)
    jitter = rng.gauss(0, total_ms * 0.04)
    return max(50.0, total_ms * ratio + jitter)


def _sim_tokens(rng: random.Random) -> int:
    """Output token count: log-normal around 350 tokens."""
    return max(1, round(rng.lognormvariate(math.log(350), 0.45)))


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _MODELS    = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]
    _WEIGHTS   = [0.50, 0.35, 0.15]
    _ENDPOINTS = ["chat", "summarise", "classify", "extract", "generate"]
    _EP_W      = [0.40,  0.20,        0.20,       0.10,      0.10]

    rng = random.Random(42)   # fixed seed for reproducibility

    # auto_save=False: demo runs without writing to disk.
    # Set auto_save=True in production to persist across restarts.
    profiler = LatencyProfiler(alert_threshold_ms=5_000, auto_save=False)

    N = 200
    print()
    print(f"  Simulating {N} LLM requests")
    print(f"  Target distribution: p50~1200ms, p99~8700ms, ~2 extreme outliers")
    print()

    for i in range(N):
        total  = _sim_total_ms(rng)
        ttft   = _sim_ttft_ms(total, rng)
        tokens = _sim_tokens(rng)
        model  = rng.choices(_MODELS, weights=_WEIGHTS)[0]
        ep     = rng.choices(_ENDPOINTS, weights=_EP_W)[0]
        profiler.record(ttft, total, tokens, model, ep)

    print(f"  Simulation complete ({N} records).")

    profiler.print_report()
