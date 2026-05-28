"""
09_selfhost/breakeven_calculator.py

Break-even analysis between managed API and self-hosted LLM inference.
Accounts for all hidden costs: GPU rental, ops labor, setup amortization,
networking, and monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

_DAYS_PER_MONTH: float = 30.44


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """Parameters that describe managed-API usage and pricing."""

    queries_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    model: str
    cost_per_1m_input: float   # USD per 1 million input tokens
    cost_per_1m_output: float  # USD per 1 million output tokens


@dataclass
class SelfHostConfig:
    """Parameters that describe self-hosted inference infrastructure."""

    gpu_cost_per_hour: float            # USD/hour for one GPU
    gpus_needed: int                    # GPUs required for the workload
    ops_engineer_hours_per_month: float # SRE / ML-ops hours each month
    ops_engineer_hourly_rate: float     # USD/hour for ops labor
    server_setup_cost_usd: float        # one-time capex or migration cost
    server_setup_amortized_months: int  # spread setup cost over this many months
    networking_cost_monthly: float      # egress, load-balancers, CDN (USD/month)
    monitoring_cost_monthly: float      # Prometheus/Grafana/alerting (USD/month)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Monthly self-hosting cost split by category."""

    gpu: float
    ops_labor: float
    setup_amortized: float
    networking: float
    monitoring: float

    @property
    def total(self) -> float:
        """Sum of all monthly self-hosting categories."""
        return (
            self.gpu + self.ops_labor + self.setup_amortized
            + self.networking + self.monitoring
        )


@dataclass
class SensitivityRow:
    """One row of the volume-sensitivity table."""

    multiplier: float
    queries_per_day: int
    monthly_api_usd: float
    monthly_selfhost_usd: float
    monthly_savings_usd: float
    cheaper: str   # "api" or "self-host"


@dataclass
class BreakevenResult:
    """Complete output of a break-even analysis run."""

    scenario_name: str
    api_cfg: APIConfig
    sh_cfg: SelfHostConfig

    # Volume & token counts
    monthly_queries: int
    monthly_input_tokens: int
    monthly_output_tokens: int

    # Managed API monthly spend
    monthly_api_cost: float

    # Self-host monthly spend (broken down)
    costs: CostBreakdown

    # Analytics
    breakeven_queries_per_day: float
    months_to_roi: Optional[float]  # None when self-host never pays off
    cheaper_option: str             # "api" or "selfhost"
    monthly_savings: float          # positive: savings vs. the costlier option
    sensitivity: list[SensitivityRow]


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class BreakevenCalculator:
    """
    Calculates the break-even point between managed API and self-hosted LLM.

    All monetary values are USD.  Time base is a 30.44-day month.

    The self-host cost model is fixed-capacity: GPU, ops, setup, networking,
    and monitoring costs do not change with query volume.  The managed-API
    cost is purely variable (cost-per-query * monthly volume).

    Break-even derivation::

        api_cost_per_query * QPD * days_per_month = sh_monthly_fixed
        QPD_breakeven = sh_monthly_fixed / (api_cost_per_query * days_per_month)

    Usage::

        calc = BreakevenCalculator(api_cfg, sh_cfg, "My Scenario")
        result = calc.calculate()
        print(generate_report(result))
    """

    def __init__(
        self,
        api_cfg: APIConfig,
        sh_cfg: SelfHostConfig,
        scenario_name: str = "",
    ) -> None:
        self.api_cfg = api_cfg
        self.sh_cfg = sh_cfg
        self.scenario_name = scenario_name

    # ------------------------------------------------------------------
    # Core cost functions
    # ------------------------------------------------------------------

    def _api_cost_per_query(self) -> float:
        """Return the USD cost of one managed-API query."""
        cfg = self.api_cfg
        return (
            cfg.avg_input_tokens  * cfg.cost_per_1m_input  / 1_000_000
            + cfg.avg_output_tokens * cfg.cost_per_1m_output / 1_000_000
        )

    def monthly_api_cost(self, queries_per_day: Optional[int] = None) -> float:
        """
        Monthly managed-API spend for a given query volume.

        If *queries_per_day* is omitted, uses ``api_cfg.queries_per_day``.
        """
        qpd = (
            queries_per_day
            if queries_per_day is not None
            else self.api_cfg.queries_per_day
        )
        return self._api_cost_per_query() * qpd * _DAYS_PER_MONTH

    def monthly_selfhost_cost(self) -> CostBreakdown:
        """
        Full monthly self-hosting cost broken down by category.

        Cost does not vary with query volume (fixed-capacity model).
        """
        sh = self.sh_cfg
        return CostBreakdown(
            gpu=sh.gpu_cost_per_hour * sh.gpus_needed * 24.0 * _DAYS_PER_MONTH,
            ops_labor=sh.ops_engineer_hours_per_month * sh.ops_engineer_hourly_rate,
            setup_amortized=sh.server_setup_cost_usd / sh.server_setup_amortized_months,
            networking=sh.networking_cost_monthly,
            monitoring=sh.monitoring_cost_monthly,
        )

    # ------------------------------------------------------------------
    # Break-even & ROI
    # ------------------------------------------------------------------

    def breakeven_queries_per_day(self, sh_costs: CostBreakdown) -> float:
        """
        Minimum queries/day at which self-hosting matches managed-API cost.

        Derived from the equation:
            api_cost_per_query * QPD * _DAYS_PER_MONTH = sh_monthly_total
        Solved for QPD.
        """
        cost_per_q = self._api_cost_per_query()
        if cost_per_q == 0.0:
            return math.inf
        return sh_costs.total / (cost_per_q * _DAYS_PER_MONTH)

    def months_to_roi(self, monthly_savings: float) -> Optional[float]:
        """
        Months until the one-time setup cost is fully recovered from savings.

        Returns *None* when self-hosting is more expensive (no positive ROI).
        """
        if monthly_savings <= 0.0:
            return None
        return self.sh_cfg.server_setup_cost_usd / monthly_savings

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(self, sh_costs: CostBreakdown) -> list[SensitivityRow]:
        """
        Compare costs at six query volumes relative to the configured base.

        Tested multipliers: 0.5x, 0.8x, 1.0x (base), 1.2x, 1.5x, 2.0x.
        """
        base_qpd = self.api_cfg.queries_per_day
        rows: list[SensitivityRow] = []
        for m in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
            qpd = max(1, int(base_qpd * m))
            api_cost = self.monthly_api_cost(qpd)
            savings = api_cost - sh_costs.total  # positive = api more expensive
            rows.append(
                SensitivityRow(
                    multiplier=m,
                    queries_per_day=qpd,
                    monthly_api_usd=api_cost,
                    monthly_selfhost_usd=sh_costs.total,
                    monthly_savings_usd=savings,
                    cheaper="self-host" if savings > 0 else "api",
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def calculate(self) -> BreakevenResult:
        """Run the full break-even analysis and return a :class:`BreakevenResult`."""
        cfg = self.api_cfg
        sh_costs = self.monthly_selfhost_cost()
        monthly_queries = int(cfg.queries_per_day * _DAYS_PER_MONTH)
        monthly_api = self.monthly_api_cost()
        be_qpd = self.breakeven_queries_per_day(sh_costs)
        monthly_diff = monthly_api - sh_costs.total  # positive = api is costlier
        roi = self.months_to_roi(monthly_diff)
        cheaper = "selfhost" if monthly_diff > 0 else "api"

        return BreakevenResult(
            scenario_name=self.scenario_name,
            api_cfg=cfg,
            sh_cfg=self.sh_cfg,
            monthly_queries=monthly_queries,
            monthly_input_tokens=monthly_queries * cfg.avg_input_tokens,
            monthly_output_tokens=monthly_queries * cfg.avg_output_tokens,
            monthly_api_cost=monthly_api,
            costs=sh_costs,
            breakeven_queries_per_day=be_qpd,
            months_to_roi=roi,
            cheaper_option=cheaper,
            monthly_savings=abs(monthly_diff),
            sensitivity=self.sensitivity_analysis(sh_costs),
        )


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

_REPORT_WIDTH = 72


def _sep(char: str = "=") -> str:
    return char * _REPORT_WIDTH


def _kv(label: str, value: str, width: int = 42) -> str:
    return f"  {label:<{width}} {value}"


def _pct(a: float, b: float) -> str:
    """Return (a/b)*100 formatted as a percentage string, or 'N/A' if b==0."""
    if b == 0.0:
        return "N/A"
    return f"{a / b * 100:.1f}%"


def generate_report(r: BreakevenResult) -> str:
    """
    Build a detailed plain-text break-even report.

    Shows all intermediate calculations so the reader can verify each number.
    Returns the report as a single string (no side effects).
    """
    lines: list[str] = []

    def h1(title: str) -> None:
        lines.append(_sep("="))
        lines.append(f"  {title}")
        lines.append(_sep("="))

    def h2(title: str) -> None:
        lines.append("")
        lines.append(f"  {title}")
        lines.append(_sep("-"))

    def kv(label: str, value: str) -> None:
        lines.append(_kv(label, value))

    cfg = r.api_cfg
    sh = r.sh_cfg
    c = r.costs

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    h1(f"BREAK-EVEN ANALYSIS -- {r.scenario_name.upper()}")
    kv("Model", cfg.model)
    kv("Query volume", f"{cfg.queries_per_day:,} queries/day")
    kv("Avg tokens per query", f"{cfg.avg_input_tokens} input / {cfg.avg_output_tokens} output")
    kv("API input price", f"${cfg.cost_per_1m_input:.2f} per 1M tokens")
    kv("API output price", f"${cfg.cost_per_1m_output:.2f} per 1M tokens")

    # ------------------------------------------------------------------
    # Monthly token volume
    # ------------------------------------------------------------------
    h2("MONTHLY TOKEN VOLUME")
    kv("Days per month (avg)", f"{_DAYS_PER_MONTH}")
    kv("Monthly queries", f"{r.monthly_queries:,}")
    kv("Monthly input tokens", f"{r.monthly_input_tokens:,}")
    kv("Monthly output tokens", f"{r.monthly_output_tokens:,}")

    # ------------------------------------------------------------------
    # Managed API cost (with formula)
    # ------------------------------------------------------------------
    h2("MANAGED API COSTS")
    cost_per_q = (
        cfg.avg_input_tokens  * cfg.cost_per_1m_input  / 1_000_000
        + cfg.avg_output_tokens * cfg.cost_per_1m_output / 1_000_000
    )
    kv("Formula",
       f"({cfg.avg_input_tokens} * ${cfg.cost_per_1m_input}/1M)"
       f" + ({cfg.avg_output_tokens} * ${cfg.cost_per_1m_output}/1M)")
    kv("Cost per query", f"${cost_per_q:.7f}")
    kv("Monthly API cost",
       f"${cost_per_q:.7f} x {cfg.queries_per_day:,} QPD x {_DAYS_PER_MONTH} days"
       f"  =  ${r.monthly_api_cost:,.2f}")
    kv("Annual API cost (est.)", f"${r.monthly_api_cost * 12:,.2f}")

    # ------------------------------------------------------------------
    # Self-host costs (with formula per line)
    # ------------------------------------------------------------------
    h2("SELF-HOSTING COSTS (monthly breakdown)")
    gpu_formula = (
        f"${sh.gpu_cost_per_hour}/hr x {sh.gpus_needed} GPU"
        f" x 24h x {_DAYS_PER_MONTH}d"
    )
    kv(f"GPU cost  ({gpu_formula})", f"${c.gpu:>10,.2f}")

    ops_formula = (
        f"{sh.ops_engineer_hours_per_month}h x ${sh.ops_engineer_hourly_rate}/hr"
    )
    kv(f"Ops labor  ({ops_formula})", f"${c.ops_labor:>10,.2f}")

    setup_formula = (
        f"${sh.server_setup_cost_usd:,.0f} / {sh.server_setup_amortized_months} months"
    )
    kv(f"Setup amortized  ({setup_formula})", f"${c.setup_amortized:>10,.2f}")

    kv("Networking (egress/LB/CDN)", f"${c.networking:>10,.2f}")
    kv("Monitoring (Prometheus/Grafana)", f"${c.monitoring:>10,.2f}")
    lines.append("  " + _sep("-")[2:])

    gpu_pct = _pct(c.gpu, c.total)
    ops_pct = _pct(c.ops_labor, c.total)
    setup_pct = _pct(c.setup_amortized, c.total)
    net_pct = _pct(c.networking, c.total)
    mon_pct = _pct(c.monitoring, c.total)
    kv("  GPU share of total", gpu_pct)
    kv("  Ops labor share", ops_pct)
    kv("  Setup share", setup_pct)
    kv("  Networking share", net_pct)
    kv("  Monitoring share", mon_pct)
    lines.append("  " + _sep("-")[2:])
    kv("TOTAL monthly self-host", f"${c.total:,.2f}")
    kv("Annual self-host (est.)", f"${c.total * 12:,.2f}")

    # ------------------------------------------------------------------
    # Head-to-head comparison
    # ------------------------------------------------------------------
    h2("HEAD-TO-HEAD COMPARISON")
    kv("Monthly managed API", f"${r.monthly_api_cost:>12,.2f}")
    kv("Monthly self-hosted", f"${r.costs.total:>12,.2f}")
    diff = r.monthly_api_cost - r.costs.total
    diff_sign = "+" if diff >= 0 else "-"
    kv("Difference (API - self-host)", f"{diff_sign}${abs(diff):>11,.2f}")
    lines.append("  " + _sep("-")[2:])
    winner_label = "SELF-HOST WINS" if r.cheaper_option == "selfhost" else "MANAGED API WINS"
    kv("Cheaper option", winner_label)
    kv("Monthly savings", f"${r.monthly_savings:,.2f}")
    kv("Annual savings", f"${r.monthly_savings * 12:,.2f}")
    if r.months_to_roi is not None:
        kv("Setup payback period",
           f"{r.months_to_roi:.1f} months  ({r.months_to_roi / 12:.1f} years)")
    else:
        kv("Setup payback period", "N/A -- self-host is costlier at this volume")

    # ------------------------------------------------------------------
    # Break-even analysis
    # ------------------------------------------------------------------
    h2("BREAK-EVEN ANALYSIS")
    be = r.breakeven_queries_per_day
    kv("Formula",
       "sh_monthly_total / (api_cost_per_query x days_per_month)")
    kv("Calculation",
       f"${c.total:,.2f} / (${cost_per_q:.7f} x {_DAYS_PER_MONTH})")
    kv("Break-even volume", f"{be:,.0f} queries/day")
    kv("Current volume", f"{cfg.queries_per_day:,} queries/day")
    margin = cfg.queries_per_day - be
    if margin >= 0:
        kv("Volume above break-even",
           f"+{margin:,.0f} queries/day  -> self-host justified")
    else:
        kv("Volume below break-even",
           f"{margin:,.0f} queries/day  -> use managed API")

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------
    h2("SENSITIVITY ANALYSIS -- varying query volume")
    col_w = [16, 10, 14, 14, 14, 11]
    header = (
        f"  {'Queries/day':>{col_w[0]}}"
        f"  {'Scale':>{col_w[1]}}"
        f"  {'API/month':>{col_w[2]}}"
        f"  {'SH/month':>{col_w[3]}}"
        f"  {'Savings':>{col_w[4]}}"
        f"  {'Winner':>{col_w[5]}}"
    )
    lines.append(header)
    lines.append(
        f"  {'-'*col_w[0]}  {'-'*col_w[1]}  {'-'*col_w[2]}"
        f"  {'-'*col_w[3]}  {'-'*col_w[4]}  {'-'*col_w[5]}"
    )
    for row in r.sensitivity:
        sav = row.monthly_savings_usd
        sav_str = f"+${sav:,.0f}" if sav >= 0 else f"-${abs(sav):,.0f}"
        base_marker = " <-- base" if row.multiplier == 1.0 else ""
        lines.append(
            f"  {row.queries_per_day:>{col_w[0]},}"
            f"  {'x'+str(row.multiplier):>{col_w[1]}}"
            f"  ${row.monthly_api_usd:>{col_w[2]-1},.0f}"
            f"  ${row.monthly_selfhost_usd:>{col_w[3]-1},.0f}"
            f"  {sav_str:>{col_w[4]}}"
            f"  {row.cheaper:>{col_w[5]}}"
            f"{base_marker}"
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    h2("RECOMMENDATION")
    if r.cheaper_option == "selfhost":
        lines.append(
            f"  Self-hosting is CHEAPER at {cfg.queries_per_day:,} queries/day."
        )
        lines.append(
            f"  Break-even threshold: {be:,.0f} queries/day."
        )
        above = cfg.queries_per_day - be
        lines.append(
            f"  Current volume is {above:,.0f} queries/day ABOVE the threshold."
        )
        lines.append("")
        lines.append(
            f"  Use managed API for fewer than {be:,.0f} queries/day."
        )
        lines.append(
            f"  Above that threshold, self-hosting saves ${r.monthly_savings:,.2f}/month"
            f"  (${r.monthly_savings * 12:,.2f}/year)."
        )
        if r.months_to_roi is not None:
            lines.append(
                f"  One-time setup cost of ${sh.server_setup_cost_usd:,.0f}"
                f" paid back in {r.months_to_roi:.1f} months."
            )
    else:
        lines.append(
            f"  Managed API is CHEAPER at {cfg.queries_per_day:,} queries/day."
        )
        lines.append(
            f"  Break-even threshold: {be:,.0f} queries/day."
        )
        gap = be - cfg.queries_per_day
        lines.append(
            f"  You need {gap:,.0f} more queries/day to justify self-hosting."
        )
        lines.append("")
        lines.append(
            f"  Use managed API until you reach {be:,.0f} queries/day."
        )
        lines.append(
            f"  At that volume, self-hosting breaks even"
            f" at ${c.total:,.2f}/month vs ${c.total:,.2f}/month."
        )

    lines.append("")
    lines.append(_sep("="))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Scenario 1: Startup -- 50K queries/day, haiku-class model
    # ------------------------------------------------------------------
    # Infra: 1x A10G GPU on AWS (~$1.50/hr), minimal ops, small setup
    # ------------------------------------------------------------------
    startup = BreakevenCalculator(
        api_cfg=APIConfig(
            queries_per_day=50_000,
            avg_input_tokens=500,
            avg_output_tokens=200,
            model="claude-haiku-4-5 (equivalent)",
            cost_per_1m_input=0.80,
            cost_per_1m_output=4.00,
        ),
        sh_cfg=SelfHostConfig(
            gpu_cost_per_hour=1.50,
            gpus_needed=1,
            ops_engineer_hours_per_month=20.0,
            ops_engineer_hourly_rate=80.0,
            server_setup_cost_usd=5_000.0,
            server_setup_amortized_months=24,
            networking_cost_monthly=200.0,
            monitoring_cost_monthly=100.0,
        ),
        scenario_name="Startup",
    )

    # ------------------------------------------------------------------
    # Scenario 2: Scale-up -- 500K queries/day, sonnet-class model
    # ------------------------------------------------------------------
    # Infra: 4x A100 GPUs on GCP ($3.00/hr each), part-time SRE
    # ------------------------------------------------------------------
    scaleup = BreakevenCalculator(
        api_cfg=APIConfig(
            queries_per_day=500_000,
            avg_input_tokens=800,
            avg_output_tokens=400,
            model="claude-sonnet-4-6 (equivalent)",
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
        ),
        sh_cfg=SelfHostConfig(
            gpu_cost_per_hour=3.00,
            gpus_needed=4,
            ops_engineer_hours_per_month=40.0,
            ops_engineer_hourly_rate=100.0,
            server_setup_cost_usd=20_000.0,
            server_setup_amortized_months=24,
            networking_cost_monthly=500.0,
            monitoring_cost_monthly=200.0,
        ),
        scenario_name="Scale-up",
    )

    # ------------------------------------------------------------------
    # Scenario 3: Enterprise -- 5M queries/day, mixed model fleet
    # ------------------------------------------------------------------
    # Mix: 70% haiku ($0.80/$4.00) + 30% sonnet ($3.00/$15.00)
    #   Blended input:  0.70*0.80 + 0.30*3.00  = 1.46/1M
    #   Blended output: 0.70*4.00 + 0.30*15.00 = 7.30/1M
    # Infra: 16x H100 on cloud ($4.50/hr each), dedicated SRE team,
    #        on-prem cluster ($500K setup amortized over 36 months)
    # ------------------------------------------------------------------
    enterprise = BreakevenCalculator(
        api_cfg=APIConfig(
            queries_per_day=5_000_000,
            avg_input_tokens=600,
            avg_output_tokens=300,
            model="Mixed fleet: 70% haiku / 30% sonnet (blended pricing)",
            cost_per_1m_input=1.46,
            cost_per_1m_output=7.30,
        ),
        sh_cfg=SelfHostConfig(
            gpu_cost_per_hour=4.50,
            gpus_needed=16,
            ops_engineer_hours_per_month=80.0,
            ops_engineer_hourly_rate=120.0,
            server_setup_cost_usd=500_000.0,
            server_setup_amortized_months=36,
            networking_cost_monthly=2_000.0,
            monitoring_cost_monthly=500.0,
        ),
        scenario_name="Enterprise",
    )

    for calc in (startup, scaleup, enterprise):
        result = calc.calculate()
        print(generate_report(result))
        print()
