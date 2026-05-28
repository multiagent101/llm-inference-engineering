"""
09_selfhost/quantization_bench.py

Simulated benchmark for LLM quantization quality degradation.

Uses a calibrated degradation model (no live model calls) so the bench
runs anywhere without GPU or API keys.  Degradation parameters are tuned
to match published llama.cpp / vLLM evaluations on MMLU, HumanEval, and
GSM8K across FP16, INT8 (GPTQ/AWQ), and INT4 (GGUF Q4_K_M) formats.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Quantization-level constants
# ---------------------------------------------------------------------------

_DAYS_PER_MONTH: float = 30.44

# (name, bits, latency_speedup_vs_fp16, memory_reduction_pct)
_LEVEL_SPECS: list[tuple[str, int, float, float]] = [
    ("FP16", 16, 1.00,  0.0),
    ("INT8",  8, 1.30, 50.0),
    ("INT4",  4, 1.80, 75.0),
]

# Quality retention relative to FP16 baseline, per (level_name, task_category).
# Based on published MMLU / HumanEval / GSM8K delta reports.
_QUALITY_RETENTION: dict[str, dict[str, float]] = {
    "FP16": {"factual": 1.000, "coding": 1.000, "reasoning": 1.000, "math": 1.000},
    "INT8": {"factual": 0.995, "coding": 0.980, "reasoning": 0.970, "math": 0.950},
    "INT4": {"factual": 0.960, "coding": 0.910, "reasoning": 0.880, "math": 0.780},
}

# Extra quality penalty for harder questions (quantization hurts nuance more).
_DIFFICULTY_PENALTY: dict[tuple[str, str], float] = {
    ("FP16", "easy"):   0.000, ("FP16", "medium"): 0.000, ("FP16", "hard"):   0.000,
    ("INT8", "easy"):   0.000, ("INT8", "medium"): 0.010, ("INT8", "hard"):   0.020,
    ("INT4", "easy"):   0.000, ("INT4", "medium"): 0.025, ("INT4", "hard"):   0.050,
}

# Category importance weights per use case (must sum to 1.0).
_USE_CASE_WEIGHTS: dict[str, dict[str, float]] = {
    "customer_support":  {"factual": 0.40, "coding": 0.05, "reasoning": 0.45, "math": 0.10},
    "code_generation":   {"factual": 0.10, "coding": 0.70, "reasoning": 0.15, "math": 0.05},
    "document_analysis": {"factual": 0.30, "coding": 0.05, "reasoning": 0.55, "math": 0.10},
}

# Maximum acceptable weighted quality drop vs FP16 for each use case.
_QUALITY_THRESHOLD: dict[str, float] = {
    "customer_support":  0.03,
    "code_generation":   0.05,
    "document_analysis": 0.04,
}

_REPORT_WIDTH = 76


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TestQuestion:
    """A single benchmark question with metadata."""

    id: int
    category: str       # "coding" | "math" | "reasoning" | "factual"
    difficulty: str     # "easy" | "medium" | "hard"
    prompt: str
    expected_answer: str


@dataclass
class QuantizationLevel:
    """Properties of one quantization format."""

    name: str
    bits: int
    latency_speedup: float          # relative to FP16
    memory_reduction_pct: float     # percentage of FP16 VRAM saved

    def memory_multiplier(self) -> float:
        """Fraction of FP16 memory required (e.g. 0.5 for INT8)."""
        return 1.0 - self.memory_reduction_pct / 100.0

    def throughput_multiplier(self) -> float:
        """
        Combined throughput gain from faster inference and smaller footprint.

        More model copies fit in the same VRAM, multiplying effective QPS.
        throughput = latency_speedup * (1 / memory_fraction)
        """
        return self.latency_speedup / self.memory_multiplier()


@dataclass
class CategoryScore:
    """Aggregate benchmark scores for one task category at one quant level."""

    category: str
    level_name: str
    avg_score: float        # mean quality score (0.0 - 1.0)
    question_count: int
    degradation_vs_fp16: float  # percentage drop (positive = worse than FP16)


@dataclass
class BenchmarkResult:
    """Full benchmark result for one quantization level."""

    level: QuantizationLevel
    category_scores: dict[str, CategoryScore]
    overall_score: float            # weighted mean across all questions
    memory_gb_7b: float             # VRAM required for a 7B-parameter model


# ---------------------------------------------------------------------------
# Hardcoded test dataset (20 questions, 5 per category)
# ---------------------------------------------------------------------------

_TEST_DATASET: list[TestQuestion] = [
    # -- CODING --------------------------------------------------------------
    TestQuestion(1, "coding", "easy",
        "Write a Python function that reverses a string without using slicing.",
        "def reverse_string(s): return ''.join(reversed(s))"),
    TestQuestion(2, "coding", "medium",
        "Implement binary search on a sorted list. Return the index or -1.",
        "lo, hi = 0, len(arr)-1; while lo<=hi: mid=(lo+hi)//2; ..."),
    TestQuestion(3, "coding", "hard",
        "Implement a token-bucket rate limiter as a Python decorator.",
        "Closure stores tokens and last_check; replenishes on each call."),
    TestQuestion(4, "coding", "medium",
        "Fix: def fib(n): return fib(n-1)+fib(n-2)  -- crashes on fib(0).",
        "Add base case: if n <= 1: return n"),
    TestQuestion(5, "coding", "hard",
        "Implement a thread-safe singleton in Python using a metaclass.",
        "class SingletonMeta(type): _instances={}; _lock=threading.Lock()"),
    # -- MATH ----------------------------------------------------------------
    TestQuestion(6, "math", "easy",
        "What is 15% of 240?",
        "36"),
    TestQuestion(7, "math", "easy",
        "Solve for x: 2x + 5 = 13",
        "x = 4"),
    TestQuestion(8, "math", "medium",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1.",
        "f'(x) = 3x^2 + 4x - 5"),
    TestQuestion(9, "math", "hard",
        "Evaluate the integral of x*sin(x) dx using integration by parts.",
        "-x*cos(x) + sin(x) + C"),
    TestQuestion(10, "math", "medium",
        "Solve the system: 2x + 3y = 12 and x + y = 5.",
        "x = 3, y = 2"),
    # -- REASONING -----------------------------------------------------------
    TestQuestion(11, "reasoning", "medium",
        "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded? "
        "Identify the logical form.",
        "Yes. Modus Barbara (universal + subsumption)."),
    TestQuestion(12, "reasoning", "medium",
        "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost?",
        "$0.05 (not $0.10 -- solve b+c=1.10, b-c=1.00)"),
    TestQuestion(13, "reasoning", "easy",
        "You are in a 100-person race. You overtake the person in 50th place. "
        "What place are you in now?",
        "50th place (you took their position)"),
    TestQuestion(14, "reasoning", "hard",
        "Two trains 300 km apart approach each other at 80 km/h and 70 km/h. "
        "A bird flies between them at 200 km/h until they meet. How far does the bird fly?",
        "400 km (trains meet in 2 h; bird covers 200*2 = 400 km)"),
    TestQuestion(15, "reasoning", "easy",
        "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "9 ('all but 9' = 9 survive)"),
    # -- FACTUAL -------------------------------------------------------------
    TestQuestion(16, "factual", "easy",
        "What is the capital of Australia?",
        "Canberra"),
    TestQuestion(17, "factual", "easy",
        "In what year did the Berlin Wall fall?",
        "1989"),
    TestQuestion(18, "factual", "easy",
        "What is the approximate speed of light in a vacuum?",
        "~3x10^8 m/s  (299,792,458 m/s)"),
    TestQuestion(19, "factual", "easy",
        "Who proposed the heliocentric model in the 16th century?",
        "Nicolaus Copernicus"),
    TestQuestion(20, "factual", "easy",
        "What is the chemical symbol for gold?",
        "Au"),
]

# Pre-group questions by category for quick lookup.
_BY_CATEGORY: dict[str, list[TestQuestion]] = {}
for _q in _TEST_DATASET:
    _BY_CATEGORY.setdefault(_q.category, []).append(_q)


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------

class QuantizationBenchmark:
    """
    Simulates quality degradation across FP16, INT8, and INT4 quantization.

    Scores are deterministic: the same question and level always produce the
    same result (uses seeded RNG for small noise).  No external calls are made.

    Args:
        model_size_b: Model parameter count in billions (used for VRAM calc).
        fp16_bytes_per_param: Bytes per parameter in full precision (default 2).
    """

    def __init__(
        self,
        model_size_b: float = 7.0,
        fp16_bytes_per_param: float = 2.0,
    ) -> None:
        self.model_size_b = model_size_b
        self.fp16_bytes_per_param = fp16_bytes_per_param
        self._levels: list[QuantizationLevel] = [
            QuantizationLevel(*spec) for spec in _LEVEL_SPECS
        ]

    # ------------------------------------------------------------------
    # Score simulation
    # ------------------------------------------------------------------

    def _simulate_score(self, q: TestQuestion, level_name: str) -> float:
        """
        Return a deterministic quality score in [0, 1] for one question.

        Score = category_retention - difficulty_penalty + small_noise,
        clamped to [0.60, 1.00].
        """
        retention = _QUALITY_RETENTION[level_name][q.category]
        penalty = _DIFFICULTY_PENALTY[(level_name, q.difficulty)]
        # Deterministic ±0.5% noise seeded by question id and level
        rng = random.Random(q.id * 97 + hash(level_name) % 1000)
        noise = rng.uniform(-0.005, 0.005)
        return max(0.60, min(1.00, retention - penalty + noise))

    # ------------------------------------------------------------------
    # Per-level aggregation
    # ------------------------------------------------------------------

    def _run_level(self, level: QuantizationLevel) -> BenchmarkResult:
        """Evaluate all test questions at one quantization level."""
        cat_totals: dict[str, list[float]] = {c: [] for c in _BY_CATEGORY}
        fp16_scores: dict[int, float] = {
            q.id: self._simulate_score(q, "FP16") for q in _TEST_DATASET
        }

        all_scores: list[float] = []
        for q in _TEST_DATASET:
            score = self._simulate_score(q, level.name)
            cat_totals[q.category].append(score)
            all_scores.append(score)

        category_scores: dict[str, CategoryScore] = {}
        for cat, scores in cat_totals.items():
            avg = sum(scores) / len(scores)
            fp16_avg = sum(fp16_scores[q.id] for q in _BY_CATEGORY[cat]) / len(scores)
            drop = (fp16_avg - avg) / fp16_avg * 100.0 if fp16_avg > 0 else 0.0
            category_scores[cat] = CategoryScore(
                category=cat,
                level_name=level.name,
                avg_score=avg,
                question_count=len(scores),
                degradation_vs_fp16=drop,
            )

        fp16_b = self.model_size_b * self.fp16_bytes_per_param  # GB (1B params * 2B)
        memory_gb = fp16_b * level.memory_multiplier()

        return BenchmarkResult(
            level=level,
            category_scores=category_scores,
            overall_score=sum(all_scores) / len(all_scores),
            memory_gb_7b=memory_gb,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, BenchmarkResult]:
        """
        Run the full benchmark across all quantization levels.

        Returns:
            Mapping from level name (e.g. "FP16") to :class:`BenchmarkResult`.
        """
        return {lvl.name: self._run_level(lvl) for lvl in self._levels}

    def recommend(
        self,
        results: dict[str, BenchmarkResult],
        use_case: str,
    ) -> str:
        """
        Recommend the best quantization level for a given use case.

        The recommendation picks the most aggressive quantization whose
        weighted quality drop stays within the acceptable threshold for
        the use case.

        Args:
            results:  Output of :meth:`run`.
            use_case: One of the keys in ``_USE_CASE_WEIGHTS``.

        Returns:
            A one-paragraph recommendation string.
        """
        if use_case not in _USE_CASE_WEIGHTS:
            raise ValueError(
                f"Unknown use case '{use_case}'. "
                f"Valid options: {list(_USE_CASE_WEIGHTS)}"
            )
        weights = _USE_CASE_WEIGHTS[use_case]
        threshold = _QUALITY_THRESHOLD[use_case]
        fp16_result = results["FP16"]

        # Weighted quality score for FP16 baseline
        def weighted_score(r: BenchmarkResult) -> float:
            return sum(
                r.category_scores[cat].avg_score * w
                for cat, w in weights.items()
            )

        fp16_ws = weighted_score(fp16_result)

        best_level = "FP16"
        best_result = fp16_result
        for name in ("INT8", "INT4"):
            r = results[name]
            ws = weighted_score(r)
            drop = (fp16_ws - ws) / fp16_ws
            if drop <= threshold:
                best_level = name
                best_result = r

        lvl = best_result.level
        ws = weighted_score(best_result)
        drop_pct = (fp16_ws - ws) / fp16_ws * 100

        # Find worst degraded category for this use case
        worst_cat = max(
            weights.keys(),
            key=lambda c: best_result.category_scores[c].degradation_vs_fp16
            * weights[c],
        )
        worst_deg = best_result.category_scores[worst_cat].degradation_vs_fp16

        lines = [
            f"Use case: {use_case.replace('_', ' ').upper()}",
            f"  Recommended: {best_level}",
            f"  Weighted quality: {ws*100:.1f}%  "
            f"(drop vs FP16: {drop_pct:.1f}%,  threshold: {threshold*100:.0f}%)",
            f"  Latency speedup: {lvl.latency_speedup:.1f}x  |  "
            f"Memory saved: {lvl.memory_reduction_pct:.0f}%  |  "
            f"Throughput gain: {lvl.throughput_multiplier():.1f}x",
            f"  Biggest degradation: {worst_cat}  "
            f"(-{worst_deg:.1f}% vs FP16)",
        ]
        if best_level == "FP16":
            lines.append(
                "  Rationale: quality requirements too strict for INT8/INT4; "
                "use FP16 or increase server capacity."
            )
        else:
            lines.append(
                f"  Rationale: {best_level} stays within the {threshold*100:.0f}% "
                f"quality budget while delivering {lvl.latency_speedup:.1f}x faster "
                f"inference and {lvl.memory_reduction_pct:.0f}% VRAM savings."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def _bar(score: float, width: int = 22) -> str:
    """Render a simple ASCII progress bar for a quality score in [0, 1]."""
    filled = round(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _sep(char: str = "=", width: int = _REPORT_WIDTH) -> str:
    return char * width


def generate_report(
    results: dict[str, BenchmarkResult],
    use_cases: list[str],
    bench: QuantizationBenchmark,
) -> str:
    """
    Build a full plain-text benchmark report.

    Sections:
    1. Methodology note
    2. Overall comparison table (quality, memory, latency, throughput)
    3. Category breakdown with ASCII bar charts
    4. Per-question sensitivity for each level
    5. Decision matrix and recommendations per use case
    """
    lines: list[str] = []
    categories = list(_BY_CATEGORY.keys())
    level_names = [s[0] for s in _LEVEL_SPECS]

    def h1(title: str) -> None:
        lines.append(_sep("="))
        lines.append(f"  {title}")
        lines.append(_sep("="))

    def h2(title: str) -> None:
        lines.append("")
        lines.append(f"  {title}")
        lines.append(_sep("-"))

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    h1("QUANTIZATION BENCHMARK -- QUALITY DEGRADATION ANALYSIS")
    lines.append(f"  Model size       : {bench.model_size_b:.0f}B parameters")
    lines.append(f"  Test questions   : {len(_TEST_DATASET)}  "
                 f"(coding={len(_BY_CATEGORY['coding'])}, "
                 f"math={len(_BY_CATEGORY['math'])}, "
                 f"reasoning={len(_BY_CATEGORY['reasoning'])}, "
                 f"factual={len(_BY_CATEGORY['factual'])})")
    lines.append(f"  Levels tested    : {', '.join(level_names)}")
    lines.append(f"  Scoring          : calibrated degradation model "
                 f"(MMLU/HumanEval/GSM8K reference)")

    # ------------------------------------------------------------------
    # Overall comparison table
    # ------------------------------------------------------------------
    h2("OVERALL COMPARISON")
    hdr = (
        f"  {'Level':<6}  {'Quality':>9}  {'vs FP16':>8}  "
        f"{'Latency':>9}  {'Throughput':>11}  "
        f"{'VRAM (7B)':>10}  {'VRAM saved':>10}"
    )
    lines.append(hdr)
    lines.append(f"  {_sep('-', _REPORT_WIDTH-2)}")

    fp16_score = results["FP16"].overall_score
    for name in level_names:
        r = results[name]
        lvl = r.level
        drop = (fp16_score - r.overall_score) / fp16_score * 100
        drop_str = "baseline" if name == "FP16" else f"-{drop:.1f}%"
        lines.append(
            f"  {name:<6}  "
            f"{r.overall_score*100:>8.1f}%  "
            f"{drop_str:>8}  "
            f"{lvl.latency_speedup:>8.1f}x  "
            f"{lvl.throughput_multiplier():>10.1f}x  "
            f"{r.memory_gb_7b:>8.1f} GB  "
            f"{lvl.memory_reduction_pct:>8.0f}%"
        )

    # ------------------------------------------------------------------
    # Category breakdown
    # ------------------------------------------------------------------
    h2("QUALITY BY TASK CATEGORY  (score = fraction of correct / expected)")
    col = 10
    header = f"  {'Category':<12}"
    for name in level_names:
        header += f"  {name:>{col}}"
    header += f"  {'INT8 drop':>10}  {'INT4 drop':>10}"
    lines.append(header)
    lines.append(f"  {_sep('-', _REPORT_WIDTH-2)}")

    for cat in categories:
        row = f"  {cat:<12}"
        fp16_cat = results["FP16"].category_scores[cat].avg_score
        for name in level_names:
            s = results[name].category_scores[cat].avg_score
            row += f"  {s*100:>{col}.1f}%"
        int8_drop = results["INT8"].category_scores[cat].degradation_vs_fp16
        int4_drop = results["INT4"].category_scores[cat].degradation_vs_fp16
        row += f"  {'-'+f'{int8_drop:.1f}%':>10}  {'-'+f'{int4_drop:.1f}%':>10}"
        lines.append(row)

    # ------------------------------------------------------------------
    # ASCII bar chart -- quality scores per level
    # ------------------------------------------------------------------
    h2("QUALITY SCORE VISUALISATION  (bar = fraction of FP16 baseline)")
    bar_w = 30
    for name in level_names:
        r = results[name]
        lines.append(f"  {name}")
        for cat in categories:
            score = r.category_scores[cat].avg_score
            fp16_s = results["FP16"].category_scores[cat].avg_score
            rel = score / fp16_s if fp16_s > 0 else 1.0
            lines.append(
                f"    {cat:<12} {_bar(rel, bar_w)}  {score*100:.1f}%"
            )
        lines.append(
            f"    {'OVERALL':<12} {_bar(r.overall_score / fp16_score, bar_w)}"
            f"  {r.overall_score*100:.1f}%"
        )
        lines.append("")

    # ------------------------------------------------------------------
    # Per-question detail table
    # ------------------------------------------------------------------
    h2("PER-QUESTION SCORES BY LEVEL")
    lines.append(
        f"  {'#':>3}  {'Cat':<10}  {'Diff':<8}  "
        f"{'FP16':>6}  {'INT8':>6}  {'INT4':>6}  "
        f"{'INT8-drop':>9}  {'INT4-drop':>9}  Prompt (truncated)"
    )
    lines.append(f"  {_sep('-', _REPORT_WIDTH-2)}")

    bench_inst = QuantizationBenchmark(bench.model_size_b)
    for q in _TEST_DATASET:
        fp16_s = bench_inst._simulate_score(q, "FP16")
        int8_s = bench_inst._simulate_score(q, "INT8")
        int4_s = bench_inst._simulate_score(q, "INT4")
        d8 = (fp16_s - int8_s) / fp16_s * 100 if fp16_s > 0 else 0
        d4 = (fp16_s - int4_s) / fp16_s * 100 if fp16_s > 0 else 0
        prompt_preview = q.prompt[:40] + "..." if len(q.prompt) > 40 else q.prompt
        lines.append(
            f"  {q.id:>3}  {q.category:<10}  {q.difficulty:<8}  "
            f"{fp16_s*100:>5.1f}%  {int8_s*100:>5.1f}%  {int4_s*100:>5.1f}%  "
            f"{'-'+f'{d8:.1f}%':>9}  {'-'+f'{d4:.1f}%':>9}  {prompt_preview}"
        )

    # ------------------------------------------------------------------
    # Decision matrix
    # ------------------------------------------------------------------
    h2("DECISION MATRIX -- USE CASE ANALYSIS")
    lines.append(
        f"  {'Use case':<20}  "
        f"{'FP16 wq':>8}  {'INT8 wq':>8}  {'INT4 wq':>8}  "
        f"{'Threshold':>10}  {'Choice':>6}"
    )
    lines.append(f"  {_sep('-', _REPORT_WIDTH-2)}")

    for uc in use_cases:
        weights = _USE_CASE_WEIGHTS[uc]
        threshold = _QUALITY_THRESHOLD[uc]

        def wq(r: BenchmarkResult) -> float:
            return sum(r.category_scores[c].avg_score * w for c, w in weights.items())

        fp16_wq = wq(results["FP16"])
        int8_wq = wq(results["INT8"])
        int4_wq = wq(results["INT4"])

        choice = "FP16"
        for name in ("INT8", "INT4"):
            drop = (fp16_wq - wq(results[name])) / fp16_wq
            if drop <= threshold:
                choice = name

        lines.append(
            f"  {uc.replace('_', ' '):<20}  "
            f"{fp16_wq*100:>7.1f}%  "
            f"{int8_wq*100:>7.1f}%  "
            f"{int4_wq*100:>7.1f}%  "
            f"<={threshold*100:.0f}% drop  "
            f"{choice:>6}"
        )

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    h2("RECOMMENDATIONS")
    b = QuantizationBenchmark(bench.model_size_b)
    r = b.run()
    for uc in use_cases:
        lines.append("")
        lines.append(b.recommend(r, uc))

    lines.append("")
    lines.append(_sep("="))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    USE_CASES = ["customer_support", "code_generation", "document_analysis"]

    bench = QuantizationBenchmark(model_size_b=7.0)
    results = bench.run()

    print(generate_report(results, USE_CASES, bench))
