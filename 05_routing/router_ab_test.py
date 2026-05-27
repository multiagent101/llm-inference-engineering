#!/usr/bin/env python3
"""
router_ab_test.py -- A/B testing framework for LLM routing configurations.

Runs two routing strategies side-by-side on the same prompt set, collects
per-request metrics (cost, quality, latency, fallbacks), and determines the
winning configuration using paired t-tests with statistical significance.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
except ImportError:                          # pragma: no cover
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    from scipy import stats as scipy_stats

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from task_classifier import TaskClassifier, TaskComplexity, _PRICING
from model_router import _FALLBACK_CHAIN, _MAX_TOKENS

load_dotenv(Path(__file__).parent.parent / ".env")

_ALPHA = 0.05           # significance level for all t-tests


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConfiguredResponse:
    """Result of routing one prompt under a specific configuration."""

    prompt: str
    complexity: str
    initial_model: str          # model chosen by the config before quality gate
    model_used: str             # final model after any fallback
    input_tokens: int
    output_tokens: int
    cost_usd: float             # total incl. quality-judge calls and fallback
    quality_score: Optional[float]    # post-gate quality (None if gate skipped)
    initial_quality_score: Optional[float]  # score that triggered fallback
    latency_ms: float
    fallback_triggered: bool
    response_text: str


@dataclass
class ConfigStats:
    """Aggregated metrics for one routing configuration over a prompt set."""

    name: str
    total_requests: int
    total_cost_usd: float
    total_cost_opus_equivalent: float
    savings_vs_opus_pct: float
    avg_cost_per_request: float
    avg_quality_score: Optional[float]
    avg_latency_ms: float
    fallback_count: int
    fallback_rate_pct: float
    model_distribution: Dict[str, int]
    complexity_distribution: Dict[str, int]
    # Raw series kept for paired t-tests (not shown in repr)
    costs: List[float] = field(default_factory=list, repr=False)
    quality_scores: List[float] = field(default_factory=list, repr=False)
    latencies: List[float] = field(default_factory=list, repr=False)


@dataclass
class MetricTest:
    """Result of a paired t-test on one metric across the two configurations."""

    metric: str
    mean_a: float
    mean_b: float
    t_statistic: float
    p_value: float
    significant: bool           # p < _ALPHA
    winner: str                 # "A", "B", or "equal" (if not significant)
    effect_pct: float           # |mean_a - mean_b| / max(mean_a, mean_b) * 100


@dataclass
class ABTestResult:
    """Full comparison result between two routing configurations."""

    winner: str                     # "A", "B", or "TIE"
    config_a: ConfigStats
    config_b: ConfigStats
    cost_test: MetricTest
    quality_test: Optional[MetricTest]
    latency_test: MetricTest
    recommendation: str


# ---------------------------------------------------------------------------
# ConfiguredRouter  (lightweight router using a custom model map)
# ---------------------------------------------------------------------------

class ConfiguredRouter:
    """
    A routing engine that maps TaskComplexity levels to custom model choices.

    Unlike ModelRouter (which uses a fixed model map from task_classifier),
    ConfiguredRouter accepts a ``model_map`` dict so arbitrary routing
    strategies can be evaluated side-by-side in A/B tests.

    Parameters
    ----------
    name : str
        Human-readable label used in reports.
    model_map : Dict[str, str]
        Mapping from complexity level string ("SIMPLE", "STANDARD", "COMPLEX",
        "EXPERT") to Anthropic model ID.
    quality_threshold : float
        Minimum acceptable LLM-as-judge score (1-10).  Responses at or below
        this value trigger a fallback.  Default 7.0.
    enable_quality_gate : bool
        When True, non-ceiling-model responses are judged before acceptance.
    llm_confidence_threshold : float
        Forwarded to TaskClassifier.  Default 0.70.
    """

    def __init__(
        self,
        name: str,
        model_map: Dict[str, str],
        quality_threshold: float = 7.0,
        enable_quality_gate: bool = True,
        llm_confidence_threshold: float = 0.70,
    ) -> None:
        self.name                  = name
        self.model_map             = model_map
        self.quality_threshold     = quality_threshold
        self.enable_quality_gate   = enable_quality_gate
        self.classifier            = TaskClassifier(llm_confidence_threshold)
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        return self._client

    # ------------------------------------------------------------------
    # Helpers (mirrors ModelRouter internals)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cost(model: str, in_tok: int, out_tok: int) -> float:
        inp_p, out_p = _PRICING.get(model, (15.00, 75.00))
        return (in_tok * inp_p + out_tok * out_p) / 1_000_000

    def _call_model(
        self,
        model: str,
        prompt: str,
        complexity: TaskComplexity,
    ) -> Tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        response = self.client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS[complexity],
            messages=[{"role": "user", "content": prompt}],
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _judge_quality(self, prompt: str, response_text: str) -> float:
        """LLM-as-judge quality score 1-10; returns 8.0 on failure."""
        judge_msg = (
            f"User request:\n{prompt[:400]}\n\n"
            f"Model response (first 2000 chars):\n{response_text[:2000]}\n\n"
            "Rate the QUALITY of what you can see on a 1-10 scale.\n"
            "  10 = accurate, complete  |  8 = good, minor gaps\n"
            "  6 = partially answers   |  4 = errors or off-topic\n"
            "Do NOT penalise for truncation if visible content is good.\n"
            'Reply ONLY with JSON: {"score": N}'
        )
        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=60,
                system="You are an objective LLM response quality evaluator.",
                messages=[{"role": "user", "content": judge_msg}],
            )
            raw  = resp.content[0].text.strip()
            data = _parse_json(raw)
            return min(10.0, max(1.0, float(data.get("score", 8.0))))
        except Exception:
            return 8.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, prompt: str) -> ConfiguredResponse:
        """
        Route ``prompt`` under this configuration and return full metrics.

        Parameters
        ----------
        prompt : str
            The user prompt to answer.

        Returns
        -------
        ConfiguredResponse
        """
        wall_t0 = time.perf_counter()

        # ── classify ──────────────────────────────────────────────────
        cls   = self.classifier.classify(prompt)
        cplx  = cls.complexity
        model = self.model_map.get(cplx.value, "claude-sonnet-4-6")

        # ── first call ────────────────────────────────────────────────
        text, in_tok, out_tok = self._call_model(model, prompt, cplx)
        total_cost = self._compute_cost(model, in_tok, out_tok)

        final_model      = model
        quality_score: Optional[float]   = None
        init_quality: Optional[float]    = None
        fallback         = False

        # ── quality gate (skip if model has no fallback) ──────────────
        has_fallback = _FALLBACK_CHAIN.get(model) is not None
        if self.enable_quality_gate and has_fallback:
            quality_score = self._judge_quality(prompt, text)
            total_cost   += self._compute_cost("claude-haiku-4-5", 350, 60)

            if quality_score <= self.quality_threshold:
                fb_model = _FALLBACK_CHAIN[model]
                if fb_model:
                    init_quality = quality_score
                    fallback     = True

                    fb_text, fb_in, fb_out = self._call_model(fb_model, prompt, cplx)
                    total_cost += self._compute_cost(fb_model, fb_in, fb_out)

                    fb_q = self._judge_quality(prompt, fb_text)
                    total_cost  += self._compute_cost("claude-haiku-4-5", 350, 60)

                    final_model   = fb_model
                    text          = fb_text
                    in_tok        = fb_in
                    out_tok       = fb_out
                    quality_score = fb_q

        latency_ms = (time.perf_counter() - wall_t0) * 1000

        return ConfiguredResponse(
            prompt=prompt,
            complexity=cplx.value,
            initial_model=model,
            model_used=final_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=total_cost,
            quality_score=quality_score,
            initial_quality_score=init_quality,
            latency_ms=latency_ms,
            fallback_triggered=fallback,
            response_text=text,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _short(text: str, n: int = 58) -> str:
    flat = text.replace("\n", " ")
    safe = flat.encode("ascii", errors="replace").decode("ascii")
    return safe[:n] + ("..." if len(safe) > n else "")


def _opus_cost(model: str, in_tok: int, out_tok: int) -> float:
    return ConfiguredRouter._compute_cost("claude-opus-4-7", in_tok, out_tok)


# ---------------------------------------------------------------------------
# RouterABTest
# ---------------------------------------------------------------------------

class RouterABTest:
    """
    A/B test harness that evaluates two routing configurations on identical
    prompt sets and reports statistical significance for each metric.

    Paired t-tests (scipy.stats.ttest_rel) are used because both configs
    answer the same prompts, making the samples naturally paired.

    Parameters
    ----------
    alpha : float
        Significance level for all t-tests.  Default 0.05.
    """

    def __init__(self, alpha: float = _ALPHA) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_router(cfg: dict) -> ConfiguredRouter:
        return ConfiguredRouter(
            name=cfg.get("name", "Unnamed"),
            model_map=cfg["model_map"],
            quality_threshold=cfg.get("quality_threshold", 7.0),
            enable_quality_gate=cfg.get("enable_quality_gate", True),
            llm_confidence_threshold=cfg.get("llm_confidence_threshold", 0.70),
        )

    @staticmethod
    def _aggregate(name: str, responses: List[ConfiguredResponse]) -> ConfigStats:
        """Compute ConfigStats from a list of responses."""
        n          = len(responses)
        costs      = [r.cost_usd for r in responses]
        latencies  = [r.latency_ms for r in responses]
        q_scores   = [r.quality_score for r in responses if r.quality_score is not None]
        fallbacks  = sum(1 for r in responses if r.fallback_triggered)

        total_cost  = sum(costs)
        opus_costs  = [
            ConfiguredRouter._compute_cost("claude-opus-4-7", r.input_tokens, r.output_tokens)
            for r in responses
        ]
        total_opus  = sum(opus_costs)
        savings_pct = (total_opus - total_cost) / total_opus * 100 if total_opus > 0 else 0.0

        model_dist: Dict[str, int] = {}
        for r in responses:
            model_dist[r.model_used] = model_dist.get(r.model_used, 0) + 1

        cmplx_dist: Dict[str, int] = {}
        for r in responses:
            cmplx_dist[r.complexity] = cmplx_dist.get(r.complexity, 0) + 1

        return ConfigStats(
            name=name,
            total_requests=n,
            total_cost_usd=total_cost,
            total_cost_opus_equivalent=total_opus,
            savings_vs_opus_pct=savings_pct,
            avg_cost_per_request=total_cost / n,
            avg_quality_score=sum(q_scores) / len(q_scores) if q_scores else None,
            avg_latency_ms=sum(latencies) / n,
            fallback_count=fallbacks,
            fallback_rate_pct=fallbacks / n * 100,
            model_distribution=model_dist,
            complexity_distribution=cmplx_dist,
            costs=costs,
            quality_scores=q_scores,
            latencies=latencies,
        )

    def _paired_t(
        self,
        metric: str,
        a_vals: List[float],
        b_vals: List[float],
        higher_is_better: bool = False,
    ) -> MetricTest:
        """
        Paired t-test for one metric.

        Parameters
        ----------
        metric : str
            Human-readable metric name.
        a_vals, b_vals : list[float]
            Per-request values for config A and B (same length, same order).
        higher_is_better : bool
            True for quality (prefer higher); False for cost/latency (prefer lower).
        """
        if len(a_vals) < 2 or len(a_vals) != len(b_vals):
            mean_a = sum(a_vals) / len(a_vals) if a_vals else 0.0
            mean_b = sum(b_vals) / len(b_vals) if b_vals else 0.0
            return MetricTest(
                metric=metric, mean_a=mean_a, mean_b=mean_b,
                t_statistic=0.0, p_value=1.0, significant=False,
                winner="equal", effect_pct=0.0,
            )

        t_stat, p_value = scipy_stats.ttest_rel(a_vals, b_vals)
        t_stat   = float(t_stat)
        p_value  = float(p_value)
        sig      = p_value < self.alpha
        mean_a   = sum(a_vals) / len(a_vals)
        mean_b   = sum(b_vals) / len(b_vals)
        denom    = max(abs(mean_a), abs(mean_b), 1e-10)
        effect   = abs(mean_a - mean_b) / denom * 100

        if not sig:
            winner = "equal"
        elif higher_is_better:
            winner = "A" if mean_a > mean_b else "B"
        else:
            winner = "A" if mean_a < mean_b else "B"

        return MetricTest(
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            t_statistic=t_stat,
            p_value=p_value,
            significant=sig,
            winner=winner,
            effect_pct=effect,
        )

    @staticmethod
    def _score_winner(
        cost_t: MetricTest,
        quality_t: Optional[MetricTest],
        latency_t: MetricTest,
    ) -> str:
        """Return "A", "B", or "TIE" from weighted metric scores."""
        a, b = 0.0, 0.0

        def _tally(test: Optional[MetricTest], weight: float) -> None:
            nonlocal a, b
            if test is None:
                a += weight / 2; b += weight / 2
            elif test.winner == "A":
                a += weight
            elif test.winner == "B":
                b += weight
            else:
                a += weight / 2; b += weight / 2

        _tally(cost_t,    2.0)   # cost: high weight
        _tally(quality_t, 2.0)   # quality: high weight
        _tally(latency_t, 1.0)   # latency: lower weight

        if a > b:   return "A"
        if b > a:   return "B"
        return "TIE"

    @staticmethod
    def _recommendation(
        winner: str,
        sa: ConfigStats,
        sb: ConfigStats,
        cost_t: MetricTest,
        quality_t: Optional[MetricTest],
        latency_t: MetricTest,
    ) -> str:
        """Build a plain-text recommendation paragraph."""
        lines: List[str] = []

        if winner == "TIE":
            lines.append("RESULT: TIE -- both configurations perform comparably.")
            lines.append(
                f"Use {sa.name} as the default (it is the established baseline)."
            )
            return "\n".join(lines)

        ws  = sa if winner == "A" else sb
        ls  = sb if winner == "A" else sa
        wlb = winner          # "A" or "B"
        llb = "B" if winner == "A" else "A"

        lines.append(f"USE {ws.name.upper()}")
        lines.append("")

        # -- cost -------------------------------------------------------
        cost_diff = (ls.avg_cost_per_request - ws.avg_cost_per_request) / max(ls.avg_cost_per_request, 1e-10) * 100
        if cost_t.winner == wlb and cost_t.significant:
            lines.append(
                f"Cost: {ws.name} is {cost_diff:.1f}% cheaper "
                f"(${ws.avg_cost_per_request:.5f} vs "
                f"${ls.avg_cost_per_request:.5f}/request, "
                f"p={cost_t.p_value:.3f} -- SIGNIFICANT)."
            )
        elif cost_t.winner == llb and cost_t.significant:
            lines.append(
                f"Cost: {ls.name} is cheaper ({cost_t.effect_pct:.1f}%, "
                f"p={cost_t.p_value:.3f}), but quality/latency advantages "
                f"favour {ws.name}."
            )
        else:
            lines.append(
                f"Cost: similar (${sa.avg_cost_per_request:.5f} vs "
                f"${sb.avg_cost_per_request:.5f}/request, "
                f"p={cost_t.p_value:.3f} -- not significant)."
            )

        # -- quality ----------------------------------------------------
        if quality_t is not None:
            qa = sa.avg_quality_score or 0.0
            qb = sb.avg_quality_score or 0.0
            if quality_t.significant:
                qw = qa if winner == "A" else qb
                ql = qb if winner == "A" else qa
                lines.append(
                    f"Quality: {ws.name} scores higher ({qw:.1f} vs {ql:.1f}/10, "
                    f"p={quality_t.p_value:.3f} -- SIGNIFICANT)."
                )
            else:
                lines.append(
                    f"Quality: comparable ({qa:.1f} vs {qb:.1f}/10, "
                    f"p={quality_t.p_value:.3f} -- difference may be chance)."
                )

        # -- latency ----------------------------------------------------
        if latency_t.significant and latency_t.winner == wlb:
            lat_w = sa.avg_latency_ms if winner == "A" else sb.avg_latency_ms
            lat_l = sb.avg_latency_ms if winner == "A" else sa.avg_latency_ms
            lines.append(
                f"Latency: {ws.name} is {latency_t.effect_pct:.0f}% faster "
                f"({lat_w:.0f} vs {lat_l:.0f} ms, "
                f"p={latency_t.p_value:.3f} -- SIGNIFICANT)."
            )
        elif not latency_t.significant:
            lines.append(
                f"Latency: similar ({sa.avg_latency_ms:.0f} vs "
                f"{sb.avg_latency_ms:.0f} ms, p={latency_t.p_value:.3f})."
            )

        # -- fallbacks --------------------------------------------------
        if sa.fallback_rate_pct != sb.fallback_rate_pct:
            lines.append(
                f"Fallbacks: {sa.name} {sa.fallback_rate_pct:.0f}% vs "
                f"{sb.name} {sb.fallback_rate_pct:.0f}%."
            )

        # -- savings vs opus -------------------------------------------
        lines.append(
            f"Savings vs always-opus: {ws.name} saves "
            f"{ws.savings_vs_opus_pct:.1f}% "
            f"({ls.name} saves {ls.savings_vs_opus_pct:.1f}%)."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_test(
        self,
        prompts: List[str],
        config_a: dict,
        config_b: dict,
        verbose: bool = True,
    ) -> ABTestResult:
        """
        Run both configurations on every prompt and return an ABTestResult.

        Prompts are processed in pairs (A then B per prompt) so wall-clock
        API conditions are matched and the paired t-test is valid.

        Parameters
        ----------
        prompts : list[str]
            Shared prompt set.  Both configurations answer every prompt.
        config_a, config_b : dict
            Configuration dicts with keys:
              - ``name`` (str)
              - ``model_map`` (Dict[str, str])
              - ``quality_threshold`` (float, optional, default 7.0)
              - ``enable_quality_gate`` (bool, optional, default True)
        verbose : bool
            Print per-prompt progress.  Default True.

        Returns
        -------
        ABTestResult
        """
        router_a = self._build_router(config_a)
        router_b = self._build_router(config_b)
        name_a   = config_a.get("name", "Config A")
        name_b   = config_b.get("name", "Config B")

        responses_a: List[ConfiguredResponse] = []
        responses_b: List[ConfiguredResponse] = []

        if verbose:
            W = 72
            print()
            print("=" * W)
            print(f"  A/B TEST: {name_a}  vs  {name_b}  --  {len(prompts)} prompts")
            print("=" * W)
            print(
                f"  {'#':>3}  {'Prompt':<46}  "
                f"{'A model':<8}  {'B model':<8}  {'A cost':>8}  {'B cost':>8}"
            )
            print(f"  {'-'*3}  {'-'*46}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        for i, prompt in enumerate(prompts, 1):
            ra = router_a.route(prompt)
            rb = router_b.route(prompt)
            responses_a.append(ra)
            responses_b.append(rb)

            if verbose:
                def _mshort(m: str) -> str:
                    return m.replace("claude-", "").replace("-4-5", "-h").replace("-4-6", "-s").replace("-4-7", "-o")

                a_m = _mshort(ra.model_used) + ("!" if ra.fallback_triggered else "")
                b_m = _mshort(rb.model_used) + ("!" if rb.fallback_triggered else "")
                print(
                    f"  {i:>3}  {_short(prompt, 46):<46}  "
                    f"{a_m:<9} {b_m:<9} "
                    f"${ra.cost_usd:>7.5f}  ${rb.cost_usd:>7.5f}"
                )

        # ── aggregate ─────────────────────────────────────────────────
        stats_a = self._aggregate(name_a, responses_a)
        stats_b = self._aggregate(name_b, responses_b)

        # ── statistical tests ─────────────────────────────────────────
        cost_t = self._paired_t(
            "cost", stats_a.costs, stats_b.costs, higher_is_better=False
        )
        q_t: Optional[MetricTest] = None
        if stats_a.quality_scores and stats_b.quality_scores and \
                len(stats_a.quality_scores) == len(stats_b.quality_scores):
            q_t = self._paired_t(
                "quality", stats_a.quality_scores, stats_b.quality_scores,
                higher_is_better=True,
            )
        latency_t = self._paired_t(
            "latency", stats_a.latencies, stats_b.latencies, higher_is_better=False
        )

        winner = self._score_winner(cost_t, q_t, latency_t)
        rec    = self._recommendation(winner, stats_a, stats_b, cost_t, q_t, latency_t)

        return ABTestResult(
            winner=winner,
            config_a=stats_a,
            config_b=stats_b,
            cost_test=cost_t,
            quality_test=q_t,
            latency_test=latency_t,
            recommendation=rec,
        )

    def print_result(self, result: ABTestResult) -> None:
        """Print a formatted A/B test report to stdout."""
        W    = 72
        SEP  = "=" * W
        SEP2 = "-" * W
        sa   = result.config_a
        sb   = result.config_b

        print()
        print(SEP)
        print("  METRIC COMPARISON")
        print(SEP2)

        def _row(label: str, a_val: str, b_val: str, note: str = "") -> None:
            print(f"  {label:<32}  {a_val:>14}  {b_val:>14}  {note}")

        _row("", sa.name, sb.name)
        _row("-" * 32, "-" * 14, "-" * 14)
        _row("Avg cost / request",
             f"${sa.avg_cost_per_request:.5f}",
             f"${sb.avg_cost_per_request:.5f}")
        _row("Total cost",
             f"${sa.total_cost_usd:.4f}",
             f"${sb.total_cost_usd:.4f}")
        _row("Savings vs always-opus",
             f"{sa.savings_vs_opus_pct:.1f}%",
             f"{sb.savings_vs_opus_pct:.1f}%")

        qa_s = f"{sa.avg_quality_score:.2f}/10" if sa.avg_quality_score else "n/a"
        qb_s = f"{sb.avg_quality_score:.2f}/10" if sb.avg_quality_score else "n/a"
        _row("Avg quality score", qa_s, qb_s)
        _row("Avg latency (ms)",
             f"{sa.avg_latency_ms:.0f}",
             f"{sb.avg_latency_ms:.0f}")
        _row("Fallback rate",
             f"{sa.fallback_rate_pct:.0f}% ({sa.fallback_count}/{sa.total_requests})",
             f"{sb.fallback_rate_pct:.0f}% ({sb.fallback_count}/{sb.total_requests})")

        print()
        print(SEP2)
        print("  MODEL DISTRIBUTION")
        print(SEP2)
        all_models = sorted(
            set(sa.model_distribution) | set(sb.model_distribution)
        )
        for m in all_models:
            short_m = m.replace("claude-", "")
            ca = sa.model_distribution.get(m, 0)
            cb = sb.model_distribution.get(m, 0)
            print(f"  {short_m:<22}  A:{ca:>2}   B:{cb:>2}")

        print()
        print(SEP2)
        print(f"  STATISTICAL SIGNIFICANCE  (paired t-test, alpha={self.alpha})")
        print(SEP2)
        print(
            f"  {'Metric':<10}  {'Mean A':>10}  {'Mean B':>10}  "
            f"{'t-stat':>8}  {'p-value':>8}  {'Sig?':>5}  {'Winner':>6}"
        )
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*6}")

        def _trow(t: MetricTest, fmt: str) -> None:
            sig_s = "YES" if t.significant else "no"
            print(
                f"  {t.metric:<10}  "
                f"{format(t.mean_a, fmt):>10}  "
                f"{format(t.mean_b, fmt):>10}  "
                f"{t.t_statistic:>+8.3f}  "
                f"{t.p_value:>8.4f}  "
                f"{sig_s:>5}  "
                f"{t.winner:>6}"
            )

        _trow(result.cost_test, ".5f")
        if result.quality_test:
            _trow(result.quality_test, ".2f")
        _trow(result.latency_test, ".0f")

        print()
        print(SEP2)
        winner_label = (
            f"WINNER: {result.winner} ({(sa if result.winner == 'A' else sb).name})"
            if result.winner != "TIE" else "RESULT: TIE"
        )
        print(f"  {winner_label}")
        print(SEP2)
        for line in result.recommendation.split("\n"):
            print(f"  {line}")
        print(SEP)
        print()


# ---------------------------------------------------------------------------
# __main__ -- Config A (standard) vs Config B (haiku-aggressive)
# ---------------------------------------------------------------------------

_CONFIG_A: dict = {
    "name": "Config-A (standard)",
    "model_map": {
        "SIMPLE":   "claude-haiku-4-5",
        "STANDARD": "claude-sonnet-4-6",
        "COMPLEX":  "claude-sonnet-4-6",
        "EXPERT":   "claude-opus-4-7",
    },
    "quality_threshold": 7.0,
    "enable_quality_gate": True,
}

_CONFIG_B: dict = {
    "name": "Config-B (haiku-first)",
    "model_map": {
        "SIMPLE":   "claude-haiku-4-5",
        "STANDARD": "claude-haiku-4-5",   # key difference: STANDARD -> haiku
        "COMPLEX":  "claude-sonnet-4-6",
        "EXPERT":   "claude-opus-4-7",
    },
    "quality_threshold": 7.0,
    "enable_quality_gate": True,
}

_PROMPTS: List[str] = [
    # SIMPLE (3)
    "What is the capital of Brazil?",
    "How many days are in a leap year?",
    "Translate 'good evening' to Japanese.",

    # STANDARD (5)
    "Explain how DNS resolution works step by step.",
    "What are the ACID properties in database transactions?",
    "Describe the differences between supervised and unsupervised learning.",
    "How does TLS/SSL encryption protect web traffic?",
    "What are the main advantages of TypeScript over plain JavaScript?",

    # COMPLEX (5)
    "Write a Python function that implements merge sort.",
    "Implement a simple HTTP server in Python using only the standard library.",
    "Design a database schema for a Twitter-like social network.",
    "Write a Python decorator that retries a function on failure with exponential backoff.",
    "Explain and implement the observer design pattern in Python.",

    # EXPERT (2)
    (
        "Design a fault-tolerant distributed consensus algorithm for a multi-region "
        "database cluster, addressing split-brain scenarios and network partitions. "
        "Include trade-offs for CAP theorem constraints."
    ),
    (
        "Propose a novel approach to reduce hallucinations in large language models, "
        "including theoretical foundations, evaluation methodology, and expected "
        "impact on benchmark performance."
    ),
]


if __name__ == "__main__":
    tester = RouterABTest(alpha=_ALPHA)
    result = tester.run_test(_PROMPTS, _CONFIG_A, _CONFIG_B, verbose=True)
    tester.print_result(result)
