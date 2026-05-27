#!/usr/bin/env python3
"""Real-time cost anomaly detection for Anthropic API spend."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np


# Severity thresholds — ratio of actual to expected cost
SEVERITY_THRESHOLDS: dict[str, float] = {
    "EMERGENCY": 3.0,
    "CRITICAL":  2.0,
    "WARNING":   1.5,
}

# Ordered from most to least severe (used in comparisons)
_SEVERITY_RANK: dict[str, int] = {"WARNING": 1, "CRITICAL": 2, "EMERGENCY": 3}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DailyCost:
    """Aggregated cost data for a single calendar day."""

    date: str
    total_cost_usd: float
    calls: int


@dataclass
class Alert:
    """A detected cost anomaly produced by one detection method."""

    timestamp: str
    date: str
    method: str          # "zscore" | "ema"
    severity: str        # "WARNING" | "CRITICAL" | "EMERGENCY"
    expected_usd: float
    actual_usd: float
    deviation_pct: float
    zscore: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_severity(actual: float, expected: float) -> Optional[str]:
    """Return the highest applicable severity, or None if not anomalous."""
    if expected <= 0:
        return None
    ratio = actual / expected
    for severity in ("EMERGENCY", "CRITICAL", "WARNING"):
        if ratio >= SEVERITY_THRESHOLDS[severity]:
            return severity
    return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Detects cost anomalies in Anthropic API spend using two independent methods:

    * **Rolling Z-score** — flags a day when its cost exceeds the rolling
      mean by more than ``zscore_threshold`` standard deviations over the
      preceding ``window`` days.

    * **Exponential Moving Average (EMA)** — uses the previous day's EMA as
      the expected cost; flags when the actual cost overshoots the EMA by the
      WARNING / CRITICAL / EMERGENCY severity thresholds.

    Both methods read daily aggregates from the SQLite database produced by
    ``CostTracker`` and write alerts to a JSON file for downstream consumers.
    """

    def __init__(
        self,
        db_path: str = "cost_tracking.db",
        alerts_file: str = "anomaly_alerts.json",
    ) -> None:
        self.db_path = Path(db_path)
        self.alerts_file = Path(alerts_file)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_daily_costs(self, days: int = 60) -> list[DailyCost]:
        """
        Aggregate daily cost totals from the SQLite database.

        Args:
            days: Number of past days to include.

        Returns:
            DailyCost records sorted by date ascending.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT DATE(timestamp)  AS date,
                       COUNT(*)         AS calls,
                       SUM(cost_usd)    AS total_cost_usd
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
                """,
                (since,),
            ).fetchall()
        return [
            DailyCost(
                date=r["date"],
                total_cost_usd=float(r["total_cost_usd"]),
                calls=int(r["calls"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Detection: rolling Z-score
    # ------------------------------------------------------------------

    def detect_zscore(
        self,
        daily_costs: list[DailyCost],
        window: int = 14,
        zscore_threshold: float = 2.0,
    ) -> list[Alert]:
        """
        Rolling Z-score anomaly detection.

        For each day ``i``, compute the mean and standard deviation of the
        ``window`` days immediately preceding it.  A day is flagged when::

            (cost_i - mean) / std  >=  zscore_threshold

        Severity is then determined by the ratio of actual to expected cost.

        Args:
            daily_costs: Daily cost records sorted ascending by date.
            window: Look-back window in days.  Requires ``len(daily_costs) > window``.
            zscore_threshold: Minimum Z-score to trigger an alert (default 2.0 = 2σ).

        Returns:
            List of Alert objects for each flagged day.
        """
        alerts: list[Alert] = []
        costs = np.array([d.total_cost_usd for d in daily_costs], dtype=float)

        for i in range(window, len(costs)):
            baseline = costs[i - window : i]
            mean = float(np.mean(baseline))
            std = float(np.std(baseline, ddof=1))

            if std < 1e-9:
                continue

            z = (costs[i] - mean) / std
            if z < zscore_threshold:
                continue

            actual = float(costs[i])
            severity = _classify_severity(actual, mean)
            if severity is None:
                continue

            alerts.append(
                Alert(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    date=daily_costs[i].date,
                    method="zscore",
                    severity=severity,
                    expected_usd=round(mean, 6),
                    actual_usd=round(actual, 6),
                    deviation_pct=round((actual - mean) / mean * 100, 2),
                    zscore=round(float(z), 3),
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Detection: EMA
    # ------------------------------------------------------------------

    def detect_ema(
        self,
        daily_costs: list[DailyCost],
        alpha: float = 0.3,
        warmup: int = 7,
    ) -> list[Alert]:
        """
        Exponential Moving Average (EMA) anomaly detection.

        The EMA of the previous day acts as the expected cost.  The EMA is
        updated *after* the anomaly check so a spike does not corrupt the
        baseline::

            expected  = EMA_{t-1}
            EMA_t     = alpha * actual_t  +  (1 - alpha) * EMA_{t-1}

        Days within the first ``warmup`` days are skipped to let the EMA
        stabilise before raising alerts.

        Args:
            daily_costs: Daily cost records sorted ascending by date.
            alpha: Smoothing factor ∈ (0, 1).  Higher = more reactive.
            warmup: Days to skip before raising alerts.

        Returns:
            List of Alert objects for each flagged day.
        """
        alerts: list[Alert] = []
        if not daily_costs:
            return alerts

        ema = daily_costs[0].total_cost_usd

        for i, day in enumerate(daily_costs[1:], start=1):
            prev_ema = ema
            actual = day.total_cost_usd

            # Always update EMA (use actual value, not clipped)
            ema = alpha * actual + (1.0 - alpha) * ema

            if i < warmup or prev_ema <= 0:
                continue

            severity = _classify_severity(actual, prev_ema)
            if severity is None:
                continue

            alerts.append(
                Alert(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    date=day.date,
                    method="ema",
                    severity=severity,
                    expected_usd=round(prev_ema, 6),
                    actual_usd=round(actual, 6),
                    deviation_pct=round((actual - prev_ema) / prev_ema * 100, 2),
                    zscore=None,
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Combined run
    # ------------------------------------------------------------------

    def detect_all(
        self,
        days: int = 60,
        window: int = 14,
        alpha: float = 0.3,
        zscore_threshold: float = 2.0,
    ) -> list[Alert]:
        """
        Run both detection methods and return combined alerts sorted by date.

        Args:
            days: History window to load from the database.
            window: Z-score rolling window size.
            alpha: EMA smoothing factor.
            zscore_threshold: Minimum Z-score to flag (default 2σ).

        Returns:
            All alerts from both methods, sorted by (date, method).
        """
        daily = self.load_daily_costs(days=days)
        if len(daily) < 2:
            return []
        z_alerts = self.detect_zscore(daily, window=window, zscore_threshold=zscore_threshold)
        e_alerts = self.detect_ema(daily, alpha=alpha, warmup=max(window // 2, 3))
        return sorted(z_alerts + e_alerts, key=lambda a: (a.date, a.method))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_alerts(self, alerts: list[Alert]) -> None:
        """
        Append alerts to the JSON file, preserving existing entries.

        Args:
            alerts: Alerts from the current detection run.
        """
        existing: list[dict] = []
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        existing = data
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.extend(asdict(a) for a in alerts)
        with open(self.alerts_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(
        self,
        alerts: list[Alert],
        daily_costs: list[DailyCost],
    ) -> None:
        """Print a formatted anomaly detection report to stdout."""
        SEP  = "=" * 76
        THIN = "-" * 76

        BADGE = {"WARNING": "[W]", "CRITICAL": "[!!]", "EMERGENCY": "[!!!]"}

        total_cost = sum(d.total_cost_usd for d in daily_costs)
        spike_dates: set[str] = {a.date for a in alerts}

        print()
        print(SEP)
        print("  ANOMALY DETECTION REPORT")
        print(SEP)
        print(f"  Days analysed:    {len(daily_costs):>6}")
        print(f"  Total spend:      ${total_cost:>10.4f}")
        print(f"  Anomalous days:   {len(spike_dates):>6}")
        print(f"  Total alerts:     {len(alerts):>6}  (zscore + ema combined)")

        if not alerts:
            print()
            print("  No anomalies detected.")
            print(SEP)
            return

        for method, label in (("zscore", "Z-SCORE (rolling 14-day window)"),
                               ("ema",    "EMA (alpha=0.3)")):
            subset = [a for a in alerts if a.method == method]
            if not subset:
                continue
            print()
            print(f"  {label}  —  {len(subset)} alert(s)")
            print(THIN)
            header = f"  {'Date':<12} {'Sev':<10} {'Expected':>10} {'Actual':>10} {'Dev %':>8}"
            if method == "zscore":
                header += f"  {'Z-score':>8}"
            print(header)
            divider = f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}"
            if method == "zscore":
                divider += f"  {'-'*8}"
            print(divider)
            for a in subset:
                badge = BADGE.get(a.severity, "")
                line = (
                    f"  {a.date:<12} {badge+' '+a.severity:<10} "
                    f"${a.expected_usd:>9.4f} ${a.actual_usd:>9.4f} "
                    f"{a.deviation_pct:>7.1f}%"
                )
                if method == "zscore" and a.zscore is not None:
                    line += f"  {a.zscore:>8.3f}"
                print(line)

        # Timeline
        print()
        print(f"  DAILY COST TIMELINE  (last {min(len(daily_costs), 45)} days)")
        print(THIN)
        print(f"  {'Date':<12} {'Cost':>10} {'Calls':>6}  Status")
        print(f"  {'-'*12} {'-'*10} {'-'*6}  {'-'*24}")

        # Highest severity per date (across both methods)
        best_sev: dict[str, str] = {}
        for a in alerts:
            prev = best_sev.get(a.date)
            if prev is None or _SEVERITY_RANK[a.severity] > _SEVERITY_RANK[prev]:
                best_sev[a.date] = a.severity

        for d in daily_costs[-45:]:
            sev = best_sev.get(d.date, "")
            flag = f"  {BADGE.get(sev, '')} {sev}" if sev else ""
            print(f"  {d.date:<12} ${d.total_cost_usd:>9.4f} {d.calls:>6}{flag}")

        print(SEP)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    DB_PATH     = "anomaly_demo.db"
    ALERTS_FILE = "anomaly_alerts_demo.json"

    for f in [DB_PATH, ALERTS_FILE]:
        if Path(f).exists():
            os.remove(f)

    # -- Build synthetic 45-day history in SQLite -------------------------
    schema = """
        CREATE TABLE api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT, user_id TEXT, team_name TEXT, model TEXT,
            input_tokens INTEGER, output_tokens INTEGER,
            cost_usd REAL, latency_ms REAL, timestamp TEXT
        )
    """

    rng   = np.random.default_rng(42)
    today = datetime.now(timezone.utc).date()

    # day_offset: how many days ago (45 = earliest, 1 = yesterday)
    spike_offsets: dict[int, float] = {
        25: 3.8,   # EMERGENCY  ~25 days ago
        12: 2.3,   # CRITICAL   ~12 days ago
         5: 4.2,   # EMERGENCY    5 days ago
        18: 1.6,   # WARNING    ~18 days ago
    }

    print("Generating 45 days of synthetic cost data ...")
    print(f"  Spike days (offset -> multiplier): {spike_offsets}\n")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(schema)
        for offset in range(45, 0, -1):
            day = today - timedelta(days=offset)
            base = float(rng.normal(2.0, 0.30))
            base = max(0.40, base)
            multiplier = spike_offsets.get(offset, 1.0)
            daily_cost = base * multiplier

            # Spread the daily cost across 3–8 simulated calls
            n_calls = int(rng.integers(3, 9))
            splits = rng.dirichlet(np.ones(n_calls)) * daily_cost
            for cost_slice in splits:
                ts = datetime(
                    day.year, day.month, day.day,
                    int(rng.integers(0, 24)),
                    int(rng.integers(0, 60)),
                    tzinfo=timezone.utc,
                ).isoformat()
                conn.execute(
                    "INSERT INTO api_calls "
                    "(feature_name, user_id, team_name, model, "
                    " input_tokens, output_tokens, cost_usd, latency_ms, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "simulated", "demo-user", "demo-team",
                        "claude-sonnet-4-5",
                        int(rng.integers(500, 3000)),
                        int(rng.integers(100, 600)),
                        float(cost_slice),
                        float(rng.uniform(300, 3000)),
                        ts,
                    ),
                )

    # -- Run both detectors -----------------------------------------------
    detector = AnomalyDetector(db_path=DB_PATH, alerts_file=ALERTS_FILE)
    alerts   = detector.detect_all(days=50, window=14, alpha=0.3, zscore_threshold=2.0)
    daily    = detector.load_daily_costs(days=50)

    detector.save_alerts(alerts)
    detector.print_report(alerts, daily)

    print(f"\n{len(alerts)} alert(s) saved to: {ALERTS_FILE}")
