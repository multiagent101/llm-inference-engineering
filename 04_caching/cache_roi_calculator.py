#!/usr/bin/env python3
"""
cache_roi_calculator.py -- ROI analysis for LLM application-level caching.

Compares monthly API spend with and without a cache layer, accounts for
infrastructure costs, and produces a recommendation with 3-scenario analysis
(pessimistic / realistic / optimistic).

Requires only Python standard-library modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1M tokens, 2025-Q4)
# ---------------------------------------------------------------------------
_PRICING: Dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":  (0.80,   4.00),
    "claude-sonnet-4-6": (3.00,  15.00),
    "claude-opus-4-7":   (15.00, 75.00),
    # short aliases
    "haiku":  (0.80,   4.00),
    "sonnet": (3.00,  15.00),
    "opus":   (15.00, 75.00),
}

# A single model name *or* a {model_name: weight} dict (weights must sum to 1)
ModelSpec = Union[str, Dict[str, float]]

_DAYS_PER_MONTH: int = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ROIMetrics:
    """ROI metrics for a single cache hit-rate scenario."""

    hit_rate: float                       # fraction of queries served from cache
    cost_per_query_usd: float             # weighted average API cost per query
    monthly_api_cost_no_cache: float      # what we pay without any cache
    monthly_api_cost_with_cache: float    # API spend after cache deflects hits
    monthly_api_savings: float            # gross API saving (hit_rate * baseline)
    monthly_infrastructure_cost: float    # cache infra bill
    monthly_net_savings: float            # api_savings - infra_cost
    annual_net_savings: float             # monthly_net_savings * 12
    roi_percentage: float                 # monthly_net_savings / infra_cost * 100
    break_even_hit_rate: float            # minimum hit rate to cover infra cost
    payback_days: float                   # days to recover one-time setup cost


@dataclass
class ROIReport:
    """Full 3-scenario analysis produced by :meth:`CacheROICalculator.analyse`."""

    pessimistic: ROIMetrics   # hit_rate - scenario_delta
    realistic: ROIMetrics     # hit_rate as configured
    optimistic: ROIMetrics    # hit_rate + scenario_delta
    recommendation: str       # plain-text go / no-go verdict
    config_advice: str        # actionable configuration hints


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class CacheROICalculator:
    """
    Calculate the economic ROI of an LLM caching layer.

    Parameters
    ----------
    queries_per_day : int
        Average number of LLM API calls per day (before caching).
    avg_input_tokens : int
        Average prompt + context token count per query.
    avg_output_tokens : int
        Average completion token count per query.
    model : ModelSpec
        Either a model-name string (e.g. ``"claude-haiku-4-5"``) or a
        ``{model_name: weight}`` dict for mixed-model deployments.
        Weights must sum to 1.0.
    cache_hit_rate : float
        Expected fraction of queries served from cache (0.0 to 1.0).
    monthly_infrastructure_cost_usd : float
        Monthly running cost of the cache infrastructure (Redis, vector DB,
        embedding service, extra RAM …).
    setup_cost_usd : float, optional
        One-time implementation cost (engineering time, tooling licences …).
        Used only for payback-period calculation.  Default 0.
    scenario_delta : float, optional
        Hit-rate shift applied for pessimistic / optimistic scenarios.
        Default 0.20 (plus or minus 20 percentage points).
    """

    def __init__(
        self,
        queries_per_day: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        model: ModelSpec,
        cache_hit_rate: float,
        monthly_infrastructure_cost_usd: float,
        setup_cost_usd: float = 0.0,
        scenario_delta: float = 0.20,
    ) -> None:
        if not 0.0 <= cache_hit_rate <= 1.0:
            raise ValueError("cache_hit_rate must be in [0.0, 1.0]")
        if monthly_infrastructure_cost_usd < 0:
            raise ValueError("monthly_infrastructure_cost_usd must be >= 0")
        if setup_cost_usd < 0:
            raise ValueError("setup_cost_usd must be >= 0")
        if not 0.0 < scenario_delta < 1.0:
            raise ValueError("scenario_delta must be in (0, 1)")

        self.queries_per_day = queries_per_day
        self.avg_input_tokens = avg_input_tokens
        self.avg_output_tokens = avg_output_tokens
        self.model = model
        self.cache_hit_rate = cache_hit_rate
        self.monthly_infrastructure_cost_usd = monthly_infrastructure_cost_usd
        self.setup_cost_usd = setup_cost_usd
        self.scenario_delta = scenario_delta

        self._cost_per_query: float = self._resolve_cost_per_query(model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_cost_per_query(self, model: ModelSpec) -> float:
        """Return the weighted-average API cost (USD) for one query."""
        if isinstance(model, str):
            key = model.lower()
            if key not in _PRICING:
                raise ValueError(
                    f"Unknown model '{model}'.  Known models: {sorted(_PRICING)}"
                )
            inp_usd, out_usd = _PRICING[key]
            return (
                self.avg_input_tokens  * inp_usd / 1_000_000
                + self.avg_output_tokens * out_usd / 1_000_000
            )

        # dict: weighted average
        total_weight = sum(model.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Model weights must sum to 1.0, got {total_weight:.6f}"
            )
        cost = 0.0
        for model_name, weight in model.items():
            key = model_name.lower()
            if key not in _PRICING:
                raise ValueError(f"Unknown model '{model_name}'")
            inp_usd, out_usd = _PRICING[key]
            cost += weight * (
                self.avg_input_tokens  * inp_usd / 1_000_000
                + self.avg_output_tokens * out_usd / 1_000_000
            )
        return cost

    def _compute_metrics(self, hit_rate: float) -> ROIMetrics:
        """Core calculation for an arbitrary hit rate."""
        hit_rate = max(0.0, min(1.0, hit_rate))

        monthly_no_cache  = self.queries_per_day * self._cost_per_query * _DAYS_PER_MONTH
        monthly_with_cache = monthly_no_cache * (1.0 - hit_rate)
        api_savings        = monthly_no_cache * hit_rate
        infra              = self.monthly_infrastructure_cost_usd
        net                = api_savings - infra
        annual             = net * 12

        roi_pct     = (net / infra * 100) if infra > 0 else math.inf
        break_even  = (infra / monthly_no_cache) if monthly_no_cache > 0 else 0.0

        daily_net = net / _DAYS_PER_MONTH
        if self.setup_cost_usd <= 0.0:
            payback = 0.0
        elif daily_net <= 0.0:
            payback = math.inf
        else:
            payback = self.setup_cost_usd / daily_net

        return ROIMetrics(
            hit_rate=hit_rate,
            cost_per_query_usd=self._cost_per_query,
            monthly_api_cost_no_cache=monthly_no_cache,
            monthly_api_cost_with_cache=monthly_with_cache,
            monthly_api_savings=api_savings,
            monthly_infrastructure_cost=infra,
            monthly_net_savings=net,
            annual_net_savings=annual,
            roi_percentage=roi_pct,
            break_even_hit_rate=break_even,
            payback_days=payback,
        )

    # ------------------------------------------------------------------
    # Recommendation generators
    # ------------------------------------------------------------------

    def _build_recommendation(
        self,
        pess: ROIMetrics,
        real: ROIMetrics,
    ) -> str:
        be = real.break_even_hit_rate
        hr = self.cache_hit_rate

        if real.monthly_net_savings <= 0:
            return (
                f"NOT RECOMMENDED at current settings.  The cache infrastructure "
                f"costs more than it saves.  Break-even hit rate is {be:.1%}; "
                f"current estimate is only {hr:.0%}.  Either reduce infra costs or "
                f"improve cache design to reach a higher hit rate."
            )
        if pess.monthly_net_savings > 0:
            return (
                f"STRONGLY RECOMMENDED.  Even the pessimistic scenario "
                f"(hit rate {pess.hit_rate:.0%}) yields net savings of "
                f"${pess.monthly_net_savings:,.0f}/month.  Expected annual "
                f"gain: ${real.annual_net_savings:,.0f}.  Break-even is at "
                f"{be:.1%}, well below your worst-case estimate."
            )
        return (
            f"RECOMMENDED WITH CAVEATS.  Profitable at the realistic hit "
            f"rate ({hr:.0%}: ${real.monthly_net_savings:,.0f}/month) but "
            f"the pessimistic case is marginally negative.  Set a hit-rate "
            f"alert at {be:.1%} and review if performance degrades."
        )

    def _build_config_advice(self, real: ROIMetrics) -> list[str]:
        tips: list[str] = []
        tips.append(
            f"Break-even hit rate: {real.break_even_hit_rate:.1%}  "
            f"(${self.monthly_infrastructure_cost_usd:,.0f}/mo infra)."
        )
        if real.monthly_api_cost_no_cache > 0:
            gain_per_pct = real.monthly_api_cost_no_cache / 100
            tips.append(
                f"Every +1 pp hit rate saves ~${gain_per_pct:,.0f}/month."
            )
        if real.payback_days == 0.0:
            tips.append("No setup cost -- savings start on day 1.")
        elif math.isfinite(real.payback_days):
            tips.append(
                f"Setup cost (${self.setup_cost_usd:,.0f}) pays back in "
                f"{real.payback_days:.0f} days."
            )
        else:
            tips.append(
                "Setup cost never recovers at current performance.  "
                "Reduce infra cost or improve hit rate first."
            )
        tips.append(
            "Layer exact cache (SHA-256) over semantic cache to maximise "
            "hit rate at minimal compute overhead."
        )
        return tips

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def what_if(self, hit_rate: float) -> ROIMetrics:
        """
        Simulate a custom hit-rate scenario.

        Parameters
        ----------
        hit_rate : float
            Cache hit rate to evaluate (0.0 to 1.0).

        Returns
        -------
        ROIMetrics
            Full economics for the given hit rate.

        Examples
        --------
        >>> calc = CacheROICalculator(...)
        >>> m = calc.what_if(0.55)
        >>> print(f"Net savings at 55%: ${m.monthly_net_savings:,.0f}/month")
        """
        if not 0.0 <= hit_rate <= 1.0:
            raise ValueError("hit_rate must be in [0.0, 1.0]")
        return self._compute_metrics(hit_rate)

    def analyse(self) -> ROIReport:
        """
        Run the full 3-scenario analysis and produce a recommendation.

        Returns
        -------
        ROIReport
            Pessimistic, realistic, and optimistic metrics plus
            recommendation and configuration advice.
        """
        pess = self._compute_metrics(self.cache_hit_rate - self.scenario_delta)
        real = self._compute_metrics(self.cache_hit_rate)
        opti = self._compute_metrics(self.cache_hit_rate + self.scenario_delta)

        return ROIReport(
            pessimistic=pess,
            realistic=real,
            optimistic=opti,
            recommendation=self._build_recommendation(pess, real),
            config_advice="\n".join(
                f"  * {tip}" for tip in self._build_config_advice(real)
            ),
        )

    def print_report(self, report: Optional[ROIReport] = None) -> None:
        """
        Print a formatted text report to stdout.

        Parameters
        ----------
        report : ROIReport, optional
            Pre-computed report.  If ``None``, :meth:`analyse` is called.
        """
        if report is None:
            report = self.analyse()

        W   = 72
        SEP = "=" * W
        SEP2 = "-" * W

        # ---- header ----
        if isinstance(self.model, str):
            model_label = self.model
        else:
            parts = ", ".join(
                f"{m}:{w:.0%}" for m, w in self.model.items()
            )
            model_label = f"mix ({parts})"

        print()
        print(SEP)
        print("  CACHE ROI ANALYSIS")
        print(SEP2)
        print(f"  Model:           {model_label}")
        print(f"  Queries/day:     {self.queries_per_day:,}")
        print(
            f"  Avg tokens:      {self.avg_input_tokens:,} input / "
            f"{self.avg_output_tokens:,} output"
        )
        print(
            f"  Cost per query:  ${report.realistic.cost_per_query_usd:.6f}"
        )
        print(f"  Cache hit rate:  {self.cache_hit_rate:.0%}  (base estimate)")
        print(
            f"  Infra cost:      ${self.monthly_infrastructure_cost_usd:,.0f}/month"
        )
        if self.setup_cost_usd > 0:
            print(f"  Setup cost:      ${self.setup_cost_usd:,.0f}  (one-time)")
        print(SEP)

        # ---- scenario comparison table ----
        scenarios = [
            (f"Pessim. ({report.pessimistic.hit_rate:.0%})", report.pessimistic),
            (f"Realistic ({report.realistic.hit_rate:.0%})",   report.realistic),
            (f"Optimist. ({report.optimistic.hit_rate:.0%})", report.optimistic),
        ]

        COL = 16
        H0  = 32

        def _hdr(s: str) -> str:
            return f"{s:>{COL}}"

        def _money(v: float) -> str:
            return f"{'N/A':>{COL}}" if math.isinf(v) else f"${v:>{COL-1},.0f}"

        def _pct(v: float) -> str:
            return f"{'inf%':>{COL}}" if math.isinf(v) else f"{v:>{COL-1}.1f}%"

        def _days(v: float) -> str:
            if v == 0.0:
                return f"{'day 1':>{COL}}"
            if math.isinf(v):
                return f"{'never':>{COL}}"
            return f"{v:>{COL-1}.0f}d"

        header_cols = "  " + f"{'Metric':<{H0}}" + "".join(_hdr(n) for n, _ in scenarios)
        print(header_cols)
        print(SEP2)

        rows: list[tuple[str, object]] = [
            ("API cost / month (no cache)",  lambda m: _money(m.monthly_api_cost_no_cache)),
            ("API cost / month (w/ cache)",  lambda m: _money(m.monthly_api_cost_with_cache)),
            ("API savings / month",          lambda m: _money(m.monthly_api_savings)),
            ("Infra cost / month",           lambda m: _money(m.monthly_infrastructure_cost)),
            ("Net savings / month",          lambda m: _money(m.monthly_net_savings)),
            ("Annual net savings",           lambda m: _money(m.annual_net_savings)),
            ("ROI %",                        lambda m: _pct(m.roi_percentage)),
            ("Break-even hit rate",          lambda m: f"{m.break_even_hit_rate:>{COL-1}.1%} "),
            ("Setup payback",                lambda m: _days(m.payback_days)),
        ]

        for label, fmt in rows:
            vals = "".join(fmt(m) for _, m in scenarios)  # type: ignore[operator]
            print(f"  {label:<{H0}}{vals}")

        # ---- recommendation ----
        print()
        print(SEP2)
        print("  RECOMMENDATION")
        print(SEP2)
        _wrap_print(report.recommendation, width=W, indent=2)

        print()
        print("  Configuration advice:")
        print(report.config_advice)
        print(SEP)
        print()

    def print_what_if_table(
        self,
        hit_rates: Optional[list[float]] = None,
    ) -> None:
        """
        Print a what-if sensitivity table for a range of hit rates.

        Parameters
        ----------
        hit_rates : list[float], optional
            Hit rates to evaluate.  Defaults to 0.10, 0.20, ... 0.90.
        """
        if hit_rates is None:
            hit_rates = [round(r / 100, 2) for r in range(10, 100, 10)]

        W   = 72
        SEP = "-" * W
        print()
        print("  WHAT-IF SENSITIVITY TABLE")
        print(SEP)
        print(
            f"  {'Hit rate':>10}  {'API savings/mo':>16}  "
            f"{'Net savings/mo':>16}  {'ROI %':>10}  {'Payback':>10}"
        )
        print(SEP)
        for hr in hit_rates:
            m = self.what_if(hr)
            roi_s   = "inf%" if math.isinf(m.roi_percentage) else f"{m.roi_percentage:.0f}%"
            pb_s    = "day 1" if m.payback_days == 0.0 else (
                      "never" if math.isinf(m.payback_days) else
                      f"{m.payback_days:.0f}d"
            )
            marker = " <--" if abs(hr - self.cache_hit_rate) < 1e-9 else ""
            print(
                f"  {hr:>9.0%}  ${m.monthly_api_savings:>15,.0f}  "
                f"${m.monthly_net_savings:>15,.0f}  {roi_s:>9}  {pb_s:>10}{marker}"
            )
        print(SEP)
        print()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap_print(text: str, width: int = 72, indent: int = 0) -> None:
    """Word-wrap ``text`` to ``width`` columns and print it."""
    prefix = " " * indent
    available = width - indent
    words = text.split()
    line  = ""
    for word in words:
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= available:
            line += " " + word
        else:
            print(prefix + line)
            line = word
    if line:
        print(prefix + line)


# ---------------------------------------------------------------------------
# __main__ -- three realistic scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SCENARIOS = [
        {
            "label":            "STARTUP",
            "description":      "10k queries/day, Claude Haiku, 45% hit rate",
            "queries_per_day":  10_000,
            "avg_input_tokens": 1_200,
            "avg_output_tokens": 300,
            "model":            "claude-haiku-4-5",
            "cache_hit_rate":   0.45,
            # Simple in-memory semantic cache on the app server -- minimal extra cost
            "monthly_infrastructure_cost_usd": 50.0,
            # ~3 days of one engineer's time
            "setup_cost_usd":   3_000.0,
        },
        {
            "label":            "SCALE-UP",
            "description":      "100k queries/day, Claude Sonnet, 60% hit rate",
            "queries_per_day":  100_000,
            "avg_input_tokens": 2_000,
            "avg_output_tokens": 500,
            "model":            "claude-sonnet-4-6",
            "cache_hit_rate":   0.60,
            # Managed Redis + small vector-DB instance
            "monthly_infrastructure_cost_usd": 400.0,
            # ~1 week team effort + managed services setup
            "setup_cost_usd":   15_000.0,
        },
        {
            "label":            "ENTERPRISE",
            "description":      "1M queries/day, model mix, 70% hit rate",
            "queries_per_day":  1_000_000,
            "avg_input_tokens": 3_000,
            "avg_output_tokens": 800,
            # 60% Haiku (high-volume simple tasks), 30% Sonnet, 10% Opus
            "model": {
                "claude-haiku-4-5":  0.60,
                "claude-sonnet-4-6": 0.30,
                "claude-opus-4-7":   0.10,
            },
            "cache_hit_rate":   0.70,
            # Enterprise vector-DB cluster + Redis cluster + embedding microservice
            "monthly_infrastructure_cost_usd": 3_500.0,
            # ~1 month platform team effort
            "setup_cost_usd":   80_000.0,
        },
    ]

    for cfg in SCENARIOS:
        print()
        print("#" * 72)
        print(f"#  {cfg['label']}: {cfg['description']}")
        print("#" * 72)

        calc = CacheROICalculator(
            queries_per_day=cfg["queries_per_day"],
            avg_input_tokens=cfg["avg_input_tokens"],
            avg_output_tokens=cfg["avg_output_tokens"],
            model=cfg["model"],
            cache_hit_rate=cfg["cache_hit_rate"],
            monthly_infrastructure_cost_usd=cfg["monthly_infrastructure_cost_usd"],
            setup_cost_usd=cfg["setup_cost_usd"],
            scenario_delta=0.20,
        )

        report = calc.analyse()
        calc.print_report(report)

        # what-if sensitivity table for this scenario
        custom_rates = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        calc.print_what_if_table(custom_rates)

        # Single what_if call example
        target = min(cfg["cache_hit_rate"] + 0.15, 1.0)
        m = calc.what_if(target)
        print(
            f"  what_if({target:.0%}):  "
            f"net ${m.monthly_net_savings:,.0f}/month, "
            f"ROI {'inf' if math.isinf(m.roi_percentage) else f'{m.roi_percentage:.0f}'}%"
        )
        print()
