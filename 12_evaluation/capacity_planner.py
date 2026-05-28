from __future__ import annotations

"""
capacity_planner.py - LLM cost capacity planner with trend extrapolation.

Fits an ordinary-least-squares linear trend to daily usage records, then
projects costs 1-24 months ahead under three growth scenarios
(conservative / base / optimistic) with 90% prediction intervals.

Reports cost-optimisation opportunities (caching, model tiering, batch API,
prompt compression) and per-month recommendations once costs cross key
thresholds.

Historical data can be supplied as a Python list of DailyRecord objects or
loaded from a SQLite database with a daily_usage table.
"""

import calendar
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  Install with:  pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_Z_90: float = 1.645          # z-score for 90% prediction interval (normal, valid n > 30)
_DEFAULT_COST_PER_QUERY: float = 0.00083
_DEFAULT_MODEL: str = "claude-haiku-4-5"
_MIN_RECORDS: int = 7

# (monthly_cost_lo, monthly_cost_hi, recommendation_text)
_COST_BANDS: tuple[tuple, ...] = (
    (0,        200,          "No optimisation required -- costs are minimal."),
    (200,      800,          "Apply prompt compression (10-15% token reduction)."),
    (800,      2_000,        "Implement semantic caching to cut repeated-query costs."),
    (2_000,    8_000,        "Add model tiering: route simple queries to Haiku."),
    (8_000,    float("inf"), "Evaluate self-hosting ROI (see breakeven_calculator.py)."),
)

# (strategy_name, cost_reduction_fraction)
_OPTIMISATIONS: tuple[tuple[str, float], ...] = (
    ("Semantic caching (50% hit rate)",          0.50),
    ("Model tiering (40% of queries to Haiku)",  0.40),
    ("Async Batch API (50% cost discount)",      0.50),
    ("Prompt compression (15% token reduction)", 0.15),
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DailyRecord:
    """One day of production LLM usage.

    Attributes:
        date:             Calendar date for this record.
        query_count:      Total API calls made on this day.
        cost_per_query:   Average cost per call in USD (varies by model mix).
        model:            Dominant model used on this day (informational).
    """

    date: date
    query_count: int
    cost_per_query: float
    model: str

    @property
    def daily_cost(self) -> float:
        """Total spend for this day in USD."""
        return self.query_count * self.cost_per_query


@dataclass
class ScenarioMonth:
    """Cost projection for a single calendar month under one growth scenario.

    Attributes:
        year_month:          YYYY-MM string for the projected month.
        days_in_month:       Number of calendar days in this month.
        scenario:            "conservative", "base", or "optimistic".
        projected_queries:   Estimated total API calls for the month.
        projected_cost_usd:  Point estimate of total cost in USD.
        ci_lower_usd:        Lower bound of the 90% prediction interval.
        ci_upper_usd:        Upper bound of the 90% prediction interval.
        recommendation:      Action guidance based on the projected cost level.
    """

    year_month: str
    days_in_month: int
    scenario: str
    projected_queries: int
    projected_cost_usd: float
    ci_lower_usd: float
    ci_upper_usd: float
    recommendation: str


@dataclass
class OptimisationSuggestion:
    """One potential cost-reduction strategy and its projected impact.

    Attributes:
        name:                        Human-readable strategy description.
        savings_fraction:            Fraction of total cost removed if applied alone.
        six_month_savings_usd:       USD saved over the projection horizon.
        optimised_six_month_cost:    Remaining cost after applying this strategy.
    """

    name: str
    savings_fraction: float
    six_month_savings_usd: float
    optimised_six_month_cost: float


@dataclass
class ProjectionReport:
    """Complete capacity plan produced by CapacityPlanner.project().

    Attributes:
        generated_at:                ISO-8601 UTC timestamp of the report.
        model:                       Dominant model label from the historical data.
        historical_days:             Number of days included in trend fitting.
        history_start:               ISO date of the earliest record.
        history_end:                 ISO date of the most recent record.
        daily_cost_slope_usd:        Fitted USD/day trend (positive = growing cost).
        daily_query_slope:           Fitted queries/day trend.
        r_squared:                   R^2 of the cost-trend regression (0-1).
        last_30d_avg_daily_queries:  30-day trailing mean daily query count.
        last_30d_avg_daily_cost_usd: 30-day trailing mean daily cost in USD.
        last_30d_monthly_run_rate:   Implied monthly cost from the last 30 days.
        conservative:                Month projections at 80% of fitted slope.
        base:                        Month projections at the fitted slope.
        optimistic:                  Month projections at 120% of fitted slope.
        optimisation_suggestions:    Strategies with per-suggestion cost impact.
    """

    generated_at: str
    model: str
    historical_days: int
    history_start: str
    history_end: str
    daily_cost_slope_usd: float
    daily_query_slope: float
    r_squared: float
    last_30d_avg_daily_queries: float
    last_30d_avg_daily_cost_usd: float
    last_30d_monthly_run_rate: float
    conservative: list[ScenarioMonth]
    base: list[ScenarioMonth]
    optimistic: list[ScenarioMonth]
    optimisation_suggestions: list[OptimisationSuggestion]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CapacityPlanner:
    """Projects LLM API costs from historical daily usage data.

    Fits a least-squares linear trend to daily total cost and extrapolates
    forward under three growth scenarios with 90% prediction intervals based
    on regression residuals. All scenarios are anchored at the last historical
    fitted value so they diverge only going forward.

    Args:
        records:               Historical daily records in chronological order.
                               Required unless db_path points to a valid file.
        db_path:               Path to a SQLite database containing a table:
                                 daily_usage(date TEXT, query_count INTEGER,
                                             cost_per_query REAL, model TEXT)
                               Takes priority over records when the file exists.
        default_model:         Model label used when records carry no model field.
        default_cost_per_query: Fallback cost per query for query-count estimates.
    """

    def __init__(
        self,
        records: Optional[list[DailyRecord]] = None,
        db_path: Optional[str] = None,
        default_model: str = _DEFAULT_MODEL,
        default_cost_per_query: float = _DEFAULT_COST_PER_QUERY,
    ) -> None:
        if db_path and Path(db_path).exists():
            loaded = self._load_from_db(db_path)
        elif records:
            loaded = sorted(records, key=lambda r: r.date)
        else:
            raise ValueError(
                "Provide either a non-empty records list or a valid db_path."
            )
        if len(loaded) < _MIN_RECORDS:
            raise ValueError(
                f"At least {_MIN_RECORDS} daily records are required; "
                f"got {len(loaded)}."
            )
        self._records: list[DailyRecord] = loaded
        self._default_model: str = default_model
        self._avg_cpq: float = (
            sum(r.cost_per_query for r in loaded) / len(loaded)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, months_ahead: int = 6) -> ProjectionReport:
        """Fit a linear trend and project costs into future calendar months.

        Fits OLS regression on daily cost, computes prediction intervals, then
        generates three scenario projections (conservative / base / optimistic)
        and a list of optimisation opportunities ranked by 6-month savings.

        Args:
            months_ahead: Calendar months to project. Must be in [1, 24].

        Returns:
            ProjectionReport with per-month ScenarioMonth lists, 90% CI bounds,
            and OptimisationSuggestion entries ranked by savings fraction.

        Raises:
            ValueError: When months_ahead is outside [1, 24].
        """
        if not 1 <= months_ahead <= 24:
            raise ValueError(f"months_ahead must be 1-24, got {months_ahead}")

        costs = np.array([r.daily_cost for r in self._records], dtype=float)
        queries = np.array([r.query_count for r in self._records], dtype=float)

        c_slope, c_intercept, r2, c_se, c_xm, c_Sxx, n = self._fit(costs)
        q_slope, *_ = self._fit(queries)

        window = min(30, len(self._records))
        last30 = self._records[-window:]
        avg_dq = sum(r.query_count for r in last30) / window
        avg_dc = sum(r.daily_cost for r in last30) / window

        last_date = self._records[-1].date

        conservative = self._project_scenario(
            "conservative", 0.8,
            c_slope, c_intercept, c_se, n, c_xm, c_Sxx,
            months_ahead, last_date,
        )
        base = self._project_scenario(
            "base", 1.0,
            c_slope, c_intercept, c_se, n, c_xm, c_Sxx,
            months_ahead, last_date,
        )
        optimistic = self._project_scenario(
            "optimistic", 1.2,
            c_slope, c_intercept, c_se, n, c_xm, c_Sxx,
            months_ahead, last_date,
        )

        base_total = sum(m.projected_cost_usd for m in base)
        suggestions = [
            OptimisationSuggestion(
                name=name,
                savings_fraction=frac,
                six_month_savings_usd=round(base_total * frac, 2),
                optimised_six_month_cost=round(base_total * (1 - frac), 2),
            )
            for name, frac in _OPTIMISATIONS
        ]

        return ProjectionReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            model=self._records[-1].model or self._default_model,
            historical_days=n,
            history_start=str(self._records[0].date),
            history_end=str(last_date),
            daily_cost_slope_usd=round(c_slope, 6),
            daily_query_slope=round(float(q_slope), 2),
            r_squared=round(r2, 4),
            last_30d_avg_daily_queries=round(avg_dq, 1),
            last_30d_avg_daily_cost_usd=round(avg_dc, 4),
            last_30d_monthly_run_rate=round(avg_dc * 30, 2),
            conservative=conservative,
            base=base,
            optimistic=optimistic,
            optimisation_suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit(
        y: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, int]:
        """Fit y = slope * x + intercept using ordinary least squares.

        Args:
            y: 1-D array of observations in chronological order.

        Returns:
            Tuple (slope, intercept, r_squared, std_err, x_mean, Sxx, n)
            where Sxx = sum((xi - x_mean)^2).
        """
        n = len(y)
        x = np.arange(n, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        y_hat = np.polyval(coeffs, x)
        residuals = y - y_hat
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        std_err = float(np.sqrt(ss_res / (n - 2))) if n > 2 else 0.0
        x_mean = float(np.mean(x))
        Sxx = float(np.sum((x - x_mean) ** 2))
        return slope, intercept, r2, std_err, x_mean, Sxx, n

    @staticmethod
    def _predict(
        x_new: float,
        slope: float,
        intercept: float,
        std_err: float,
        n: int,
        x_mean: float,
        Sxx: float,
    ) -> tuple[float, float, float]:
        """Return (point_estimate, ci_lower, ci_upper) at x_new.

        Applies the standard OLS prediction interval:
            y_hat +/- Z_90 * s * sqrt(1 + 1/n + (x_new - x_mean)^2 / Sxx)

        The interval widens as x_new moves further from the historical centre,
        correctly penalising distant extrapolations.

        Args:
            x_new:     Day index of the point to predict.
            slope:     Regression slope (may be scenario-adjusted).
            intercept: Regression intercept (scenario-adjusted to match anchor).
            std_err:   Residual standard deviation from the base regression.
            n:         Number of historical observations.
            x_mean:    Mean of the historical x indices.
            Sxx:       Sum of squared deviations of x indices from x_mean.

        Returns:
            Tuple (point_estimate, lower_bound, upper_bound) in the same units as y.
        """
        y_hat = slope * x_new + intercept
        leverage = float(np.sqrt(1.0 + 1.0 / n + (x_new - x_mean) ** 2 / Sxx))
        margin = _Z_90 * std_err * leverage
        return y_hat, y_hat - margin, y_hat + margin

    def _project_scenario(
        self,
        scenario: str,
        slope_mult: float,
        base_slope: float,
        base_intercept: float,
        std_err: float,
        n: int,
        x_mean: float,
        Sxx: float,
        months_ahead: int,
        last_date: date,
    ) -> list[ScenarioMonth]:
        """Generate per-month ScenarioMonth projections for one growth scenario.

        The effective slope is base_slope * slope_mult. The effective intercept
        is recalculated so that all scenarios agree at the last historical fitted
        value (x = n-1), diverging only into the future.

        Args:
            scenario:       Human-readable scenario label.
            slope_mult:     Multiplier applied to the base slope (0.8 / 1.0 / 1.2).
            base_slope:     Fitted slope from the base regression.
            base_intercept: Fitted intercept from the base regression.
            std_err:        Residual standard deviation (same for all scenarios).
            n:              Number of historical observations.
            x_mean:         Mean of historical x indices.
            Sxx:            Sum of squared deviations of historical x indices.
            months_ahead:   Number of future months to project.
            last_date:      Date of the last historical record.

        Returns:
            List of ScenarioMonth, one per future calendar month.
        """
        eff_slope = base_slope * slope_mult
        # Anchor: all scenarios share the same fitted value at the last history day.
        y_anchor = base_slope * (n - 1) + base_intercept
        eff_intercept = y_anchor - eff_slope * (n - 1)

        months: list[ScenarioMonth] = []
        for m in range(1, months_ahead + 1):
            month_start = self._add_months(last_date, m)
            next_start = self._add_months(last_date, m + 1)
            days = (next_start - month_start).days

            days_to_mid = (month_start - last_date).days + days // 2
            x_new = float(n - 1 + days_to_mid)

            y_hat, ci_lo, ci_hi = self._predict(
                x_new, eff_slope, eff_intercept, std_err, n, x_mean, Sxx
            )

            daily_cost = max(0.0, y_hat)
            monthly_cost = daily_cost * days
            monthly_ci_lo = max(0.0, ci_lo) * days
            monthly_ci_hi = max(0.0, ci_hi) * days

            proj_q = (
                int(monthly_cost / self._avg_cpq) if self._avg_cpq > 0 else 0
            )

            months.append(
                ScenarioMonth(
                    year_month=month_start.strftime("%Y-%m"),
                    days_in_month=days,
                    scenario=scenario,
                    projected_queries=proj_q,
                    projected_cost_usd=round(monthly_cost, 2),
                    ci_lower_usd=round(monthly_ci_lo, 2),
                    ci_upper_usd=round(monthly_ci_hi, 2),
                    recommendation=self._recommendation(monthly_cost),
                )
            )
        return months

    @staticmethod
    def _recommendation(monthly_cost: float) -> str:
        """Return the threshold-based action string for a given monthly cost."""
        for lo, hi, msg in _COST_BANDS:
            if lo <= monthly_cost < hi:
                return msg
        return _COST_BANDS[-1][2]

    @staticmethod
    def _load_from_db(db_path: str) -> list[DailyRecord]:
        """Load DailyRecord list from a SQLite daily_usage table.

        Args:
            db_path: Path to the SQLite file.

        Returns:
            Records sorted ascending by date.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT date, query_count, cost_per_query, model "
            "FROM daily_usage ORDER BY date"
        )
        records = [
            DailyRecord(
                date=date.fromisoformat(row[0]),
                query_count=int(row[1]),
                cost_per_query=float(row[2]),
                model=str(row[3]),
            )
            for row in cur.fetchall()
        ]
        conn.close()
        return records

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        """Return the first day of the calendar month `months` ahead of `d`."""
        m = d.month + months
        y = d.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        return date(y, m, 1)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SEP = "=" * 72

    # ------------------------------------------------------------------
    # Generate 90 days of synthetic history
    #   - Blended model mix: 70% Haiku + 30% Sonnet  =>  $0.0028/query avg
    #   - Linear growth: 10,000 to ~19,000 queries/day  (+100 qpd)
    #   - Weekly seasonality: weekdays 100%, weekends 60%
    #   - Gaussian noise: sigma=10%
    # ------------------------------------------------------------------
    rng = random.Random(42)
    hist_end = date.today() - timedelta(days=1)
    hist_start = hist_end - timedelta(days=89)

    records: list[DailyRecord] = []
    for i in range(90):
        d = hist_start + timedelta(days=i)
        base_q = 10_000 + 100 * i
        weekday_mult = 0.60 if d.weekday() >= 5 else 1.0
        noise = rng.gauss(1.0, 0.10)
        query_count = max(1, int(base_q * weekday_mult * noise))
        records.append(
            DailyRecord(
                date=d,
                query_count=query_count,
                cost_per_query=0.0028,
                model="claude-haiku-4-5",
            )
        )

    planner = CapacityPlanner(records=records)
    report = planner.project(months_ahead=6)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print(_SEP)
    print("  CAPACITY PLANNER  --  LLM Cost Projection")
    print(_SEP)
    print(f"  Model          : {report.model}")
    print(f"  History        : {report.history_start} to {report.history_end}"
          f"  ({report.historical_days} days)")
    print(f"  Trend (cost)   : {report.daily_cost_slope_usd:+.4f} USD/day"
          f"   ({report.daily_query_slope:+.1f} queries/day)")
    print(f"  R-squared      : {report.r_squared:.3f}"
          f"  ({'strong' if report.r_squared >= 0.8 else 'moderate'} fit)")
    print(f"  Last-30d avg   : {report.last_30d_avg_daily_queries:,.0f} queries/day"
          f"  |  ${report.last_30d_avg_daily_cost_usd:.2f}/day"
          f"  |  ~${report.last_30d_monthly_run_rate:,.0f}/month")

    # ------------------------------------------------------------------
    # Historical mini-chart (weekly averages)
    # ------------------------------------------------------------------
    print()
    print("  Historical weekly cost averages:")
    BAR_MAX = 30
    week_costs = []
    for w in range(0, 90, 7):
        chunk = records[w : w + 7]
        avg_c = sum(r.daily_cost for r in chunk) / len(chunk)
        week_costs.append((chunk[0].date, avg_c))
    max_c = max(c for _, c in week_costs)
    for d, avg_c in week_costs:
        bar_len = max(1, int(avg_c / max_c * BAR_MAX))
        bar = "#" * bar_len
        print(f"    {d}  ${avg_c:6.2f}/day  {bar}")

    # ------------------------------------------------------------------
    # Scenario projection table
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  SCENARIO PROJECTIONS  (90% prediction interval for Base scenario)")
    print(_SEP)
    print(
        f"  {'Month':<9}  {'Conservative':>14}  {'Base':>14}  {'Optimistic':>14}"
        f"  {'90% CI (Base)':>24}"
    )
    print("  " + "-" * 80)
    for con, bas, opt in zip(report.conservative, report.base, report.optimistic):
        ci = f"[${bas.ci_lower_usd:,.0f} - ${bas.ci_upper_usd:,.0f}]"
        print(
            f"  {bas.year_month:<9}"
            f"  ${con.projected_cost_usd:>13,.2f}"
            f"  ${bas.projected_cost_usd:>13,.2f}"
            f"  ${opt.projected_cost_usd:>13,.2f}"
            f"  {ci:>24}"
        )
    print("  " + "-" * 80)
    con_tot = sum(m.projected_cost_usd for m in report.conservative)
    bas_tot = sum(m.projected_cost_usd for m in report.base)
    opt_tot = sum(m.projected_cost_usd for m in report.optimistic)
    print(
        f"  {'6-month total':<9}"
        f"  ${con_tot:>13,.2f}"
        f"  ${bas_tot:>13,.2f}"
        f"  ${opt_tot:>13,.2f}"
    )

    # ------------------------------------------------------------------
    # Optimisation opportunities
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print(f"  OPTIMISATION OPPORTUNITIES  (6-month Base total: ${bas_tot:,.2f})")
    print(_SEP)
    print(
        f"  {'Strategy':<46}  {'Savings %':>9}  {'6m Savings':>11}  {'6m Total':>10}"
    )
    print("  " + "-" * 80)
    for sug in report.optimisation_suggestions:
        print(
            f"  {sug.name:<46}"
            f"  {sug.savings_fraction:>8.0%}"
            f"  ${sug.six_month_savings_usd:>10,.2f}"
            f"  ${sug.optimised_six_month_cost:>9,.2f}"
        )

    # ------------------------------------------------------------------
    # Caching deep-dive (the most commonly recommended optimisation)
    # ------------------------------------------------------------------
    caching = next(
        (s for s in report.optimisation_suggestions if "caching" in s.name.lower()),
        None,
    )
    if caching:
        print()
        print(_SEP)
        print(
            f"  CACHING IMPACT  (50% hit rate, applied to Base scenario)"
        )
        print(_SEP)
        print(
            f"  {'Month':<9}  {'Without Cache':>15}  {'With Cache':>13}"
            f"  {'Monthly Saving':>15}  {'Queries Saved':>14}"
        )
        print("  " + "-" * 72)
        for bas in report.base:
            cached = bas.projected_cost_usd * (1 - caching.savings_fraction)
            saving = bas.projected_cost_usd - cached
            q_saved = int(bas.projected_queries * caching.savings_fraction)
            print(
                f"  {bas.year_month:<9}"
                f"  ${bas.projected_cost_usd:>14,.2f}"
                f"  ${cached:>12,.2f}"
                f"  ${saving:>14,.2f}"
                f"  {q_saved:>13,}"
            )
        print("  " + "-" * 72)
        total_saved = caching.six_month_savings_usd
        total_cached = caching.optimised_six_month_cost
        print(
            f"  {'Total':<9}"
            f"  ${bas_tot:>14,.2f}"
            f"  ${total_cached:>12,.2f}"
            f"  ${total_saved:>14,.2f}"
        )
        print()
        print(
            f"  Implementing semantic caching at 50% hit rate reduces the"
        )
        print(
            f"  projected 6-month cost from ${bas_tot:,.2f} to"
            f" ${total_cached:,.2f}  (-{caching.savings_fraction:.0%})."
        )

    # ------------------------------------------------------------------
    # Per-month recommendations (Base scenario, deduplicated)
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  MONTHLY RECOMMENDATIONS  (Base scenario)")
    print(_SEP)
    prev_rec = ""
    for m in report.base:
        if m.recommendation != prev_rec:
            print(
                f"  From {m.year_month}  ${m.projected_cost_usd:>8,.0f}/month"
                f"  --  {m.recommendation}"
            )
            prev_rec = m.recommendation

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  PROJECTION SUMMARY")
    print(_SEP)
    month1 = report.base[0]
    month6 = report.base[-1]
    growth_pct = (month6.projected_cost_usd - month1.projected_cost_usd) / month1.projected_cost_usd * 100
    print(f"  Month 1 (base)  : ${month1.projected_cost_usd:,.2f}")
    print(f"  Month 6 (base)  : ${month6.projected_cost_usd:,.2f}")
    print(f"  6-month growth  : {growth_pct:+.1f}%")
    print(f"  6-month totals  : conservative ${con_tot:,.2f}"
          f"  |  base ${bas_tot:,.2f}  |  optimistic ${opt_tot:,.2f}")
    print(f"  Best single opt.: {report.optimisation_suggestions[0].name}")
    print(f"                    saves ${report.optimisation_suggestions[0].six_month_savings_usd:,.2f}"
          f" over 6 months  (-{report.optimisation_suggestions[0].savings_fraction:.0%})")
    print()
