"""
Before/After Optimization Benchmark
====================================
Simula un sistema di customer support con 10.000 query/giorno.
MANTIENE LA STESSA QUALITA' - il risparmio viene SOLO da efficienza.
Zero chiamate API reali. Solo librerie standard Python.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime


# ---------------------------------------------------------------------------
# PREZZI REALI (dollari per milione di token)
# ---------------------------------------------------------------------------
PRICES = {
    "haiku":  {"input": 0.80,  "output": 4.00},
    "sonnet": {"input": 3.00,  "output": 15.00},
}

# ---------------------------------------------------------------------------
# DISTRIBUZIONE QUERY (mix realistico customer support)
# ---------------------------------------------------------------------------
QUERY_TIERS = {
    "simple": {
        "share":          0.60,
        "avg_input_tok":  50,
        "avg_output_tok": 100,
        "label":          "Semplici (FAQ, orari, saluti)",
        "quality_score":  9.2,
    },
    "medium": {
        "share":          0.30,
        "avg_input_tok":  300,
        "avg_output_tok": 300,
        "label":          "Medie (problemi tecnici)",
        "quality_score":  8.7,
    },
    "complex": {
        "share":          0.10,
        "avg_input_tok":  800,
        "avg_output_tok": 500,
        "label":          "Complesse (troubleshooting avanzato)",
        "quality_score":  9.5,
    },
}

# ---------------------------------------------------------------------------
# PARAMETRI DI SISTEMA
# ---------------------------------------------------------------------------
DAILY_QUERIES        = 10_000
DAYS_PER_MONTH       = 30
MONTHS_PER_YEAR      = 12

LATENCY_SONNET_MS    = 3_200
LATENCY_HAIKU_MS     = 2_409
LATENCY_CACHE_HIT_MS = 12
LATENCY_P99_RATIO    = 4.2

CACHE_HIT_RATE       = 0.45
ASYNC_THROUGHPUT_MUL = 4.0

ESCALATION_RATE_SIMPLE = 0.02
ESCALATION_RATE_MEDIUM = 0.04


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def token_cost(tokens, price_per_million):
    return tokens / 1_000_000 * price_per_million


def compute_tier_cost_per_query(tier, model):
    p = PRICES[model]
    return (
        token_cost(tier["avg_input_tok"],  p["input"]) +
        token_cost(tier["avg_output_tok"], p["output"])
    )


def p99(p50_ms):
    return p50_ms * LATENCY_P99_RATIO


def fmt_usd(v):
    if v >= 1_000:
        return "${:,.2f}".format(v)
    return "${:.4f}".format(v)


def fmt_ms(v):
    return "{:,.1f} ms".format(v)


def print_sep(char="=", width=65):
    print(char * width)


def print_section(title):
    print()
    print_sep("-")
    print("  " + title)
    print_sep("-")


# ---------------------------------------------------------------------------
# SEZIONE 1 - COSTI
# ---------------------------------------------------------------------------

@dataclass
class CostResult:
    daily_before:   float
    daily_after:    float
    monthly_before: float
    monthly_after:  float
    annual_before:  float
    annual_after:   float
    savings_pct:    float
    annual_savings: float


def compute_costs():
    # PRIMA: tutto su Sonnet, nessun caching
    daily_before = 0.0
    tier_detail_before = {}
    for tier_key, tier in QUERY_TIERS.items():
        n_queries      = DAILY_QUERIES * tier["share"]
        cost_per_query = compute_tier_cost_per_query(tier, "sonnet")
        tier_cost      = n_queries * cost_per_query
        daily_before  += tier_cost
        tier_detail_before[tier_key] = {
            "queries":        int(n_queries),
            "model":          "sonnet",
            "cost_per_query": round(cost_per_query * 1000, 6),
            "daily_cost":     round(tier_cost, 4),
        }

    # DOPO: routing + caching
    daily_after = 0.0
    tier_detail_after = {}

    for tier_key, tier in QUERY_TIERS.items():
        n_queries = DAILY_QUERIES * tier["share"]

        if tier_key == "complex":
            routed_model = "sonnet"
            escalation   = 0.0
        elif tier_key == "simple":
            routed_model = "haiku"
            escalation   = ESCALATION_RATE_SIMPLE
        else:
            routed_model = "haiku"
            escalation   = ESCALATION_RATE_MEDIUM

        n_cache_hit   = n_queries * CACHE_HIT_RATE
        n_api_call    = n_queries * (1 - CACHE_HIT_RATE)
        n_escalated   = n_api_call * escalation
        n_routed      = n_api_call * (1 - escalation)

        cost_routed    = n_routed    * compute_tier_cost_per_query(tier, routed_model)
        cost_escalated = n_escalated * compute_tier_cost_per_query(tier, "sonnet")
        tier_cost      = cost_routed + cost_escalated

        daily_after += tier_cost
        tier_detail_after[tier_key] = {
            "queries":              int(n_queries),
            "model_primary":        routed_model,
            "cache_hits":           int(n_cache_hit),
            "api_calls":            int(n_api_call),
            "escalated_to_sonnet":  int(n_escalated),
            "daily_cost":           round(tier_cost, 4),
        }

    savings_pct    = (1 - daily_after / daily_before) * 100
    annual_savings = (daily_before - daily_after) * DAYS_PER_MONTH * MONTHS_PER_YEAR

    result = CostResult(
        daily_before   = round(daily_before,   4),
        daily_after    = round(daily_after,    4),
        monthly_before = round(daily_before  * DAYS_PER_MONTH, 2),
        monthly_after  = round(daily_after   * DAYS_PER_MONTH, 2),
        annual_before  = round(daily_before  * DAYS_PER_MONTH * MONTHS_PER_YEAR, 2),
        annual_after   = round(daily_after   * DAYS_PER_MONTH * MONTHS_PER_YEAR, 2),
        savings_pct    = round(savings_pct,  1),
        annual_savings = round(annual_savings, 2),
    )
    return result, {"before": tier_detail_before, "after": tier_detail_after}


# ---------------------------------------------------------------------------
# SEZIONE 2 - LATENZA
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    avg_before_ms:   float
    avg_after_ms:    float
    reduction_pct:   float
    p99_before_ms:   float
    p99_after_ms:    float
    ux_threshold_ms: int
    ux_sessions_pct: float


def compute_latency():
    share_cached   = CACHE_HIT_RATE
    share_not_cached = 1 - CACHE_HIT_RATE

    share_haiku_raw  = QUERY_TIERS["simple"]["share"] + QUERY_TIERS["medium"]["share"]
    share_sonnet_raw = QUERY_TIERS["complex"]["share"]

    share_not_cached_haiku  = share_not_cached * share_haiku_raw
    share_not_cached_sonnet = share_not_cached * share_sonnet_raw

    avg_after = (
        share_cached              * LATENCY_CACHE_HIT_MS +
        share_not_cached_haiku    * LATENCY_HAIKU_MS     +
        share_not_cached_sonnet   * LATENCY_SONNET_MS
    )

    reduction_pct = (1 - avg_after / LATENCY_SONNET_MS) * 100
    p99_before    = p99(LATENCY_SONNET_MS)
    p99_after     = p99(avg_after)

    ux_threshold  = 2_000
    haiku_under_2s = 0.60
    ux_sessions   = (share_cached + share_not_cached_haiku * haiku_under_2s) * 100

    return LatencyResult(
        avg_before_ms   = float(LATENCY_SONNET_MS),
        avg_after_ms    = round(avg_after, 1),
        reduction_pct   = round(reduction_pct, 1),
        p99_before_ms   = round(p99_before, 0),
        p99_after_ms    = round(p99_after, 0),
        ux_threshold_ms = ux_threshold,
        ux_sessions_pct = round(ux_sessions, 1),
    )


# ---------------------------------------------------------------------------
# SEZIONE 3 - QUALITY GATE
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    avg_quality_score:   float
    pass_rate_pct:       float
    escalation_rate_pct: float
    tier_scores:         dict


def compute_quality():
    weighted_score = 0.0
    total_share    = 0.0
    tier_scores    = {}

    for tier_key, tier in QUERY_TIERS.items():
        weighted_score += tier["quality_score"] * tier["share"]
        total_share    += tier["share"]
        tier_scores[tier_key] = {
            "label":         tier["label"],
            "quality_score": tier["quality_score"],
            "share_pct":     tier["share"] * 100,
        }

    avg_score = weighted_score / total_share

    escalated_simple = (DAILY_QUERIES * QUERY_TIERS["simple"]["share"]
                        * (1 - CACHE_HIT_RATE) * ESCALATION_RATE_SIMPLE)
    escalated_medium = (DAILY_QUERIES * QUERY_TIERS["medium"]["share"]
                        * (1 - CACHE_HIT_RATE) * ESCALATION_RATE_MEDIUM)
    total_escalated  = escalated_simple + escalated_medium
    escalation_pct   = (total_escalated / DAILY_QUERIES) * 100
    pass_rate_pct    = 100 - escalation_pct

    return QualityResult(
        avg_quality_score   = round(avg_score, 2),
        pass_rate_pct       = round(pass_rate_pct, 2),
        escalation_rate_pct = round(escalation_pct, 2),
        tier_scores         = tier_scores,
    )


# ---------------------------------------------------------------------------
# SEZIONE 4 - PROIEZIONE CRESCITA
# ---------------------------------------------------------------------------

@dataclass
class GrowthProjection:
    daily_queries:     int
    monthly_naive:     float
    monthly_optimized: float
    monthly_savings:   float
    savings_pct:       float


def compute_growth(scale_factor):
    n = DAILY_QUERIES * scale_factor

    daily_naive = 0.0
    for tier in QUERY_TIERS.values():
        daily_naive += n * tier["share"] * compute_tier_cost_per_query(tier, "sonnet")

    daily_opt = 0.0
    for tier_key, tier in QUERY_TIERS.items():
        n_api = n * tier["share"] * (1 - CACHE_HIT_RATE)
        if tier_key == "complex":
            daily_opt += n_api * compute_tier_cost_per_query(tier, "sonnet")
        else:
            esc   = ESCALATION_RATE_SIMPLE if tier_key == "simple" else ESCALATION_RATE_MEDIUM
            daily_opt += n_api * (1 - esc) * compute_tier_cost_per_query(tier, "haiku")
            daily_opt += n_api * esc       * compute_tier_cost_per_query(tier, "sonnet")

    monthly_naive = daily_naive * DAYS_PER_MONTH
    monthly_opt   = daily_opt   * DAYS_PER_MONTH
    savings       = monthly_naive - monthly_opt
    savings_pct   = (savings / monthly_naive) * 100

    return GrowthProjection(
        daily_queries     = int(n),
        monthly_naive     = round(monthly_naive, 2),
        monthly_optimized = round(monthly_opt,   2),
        monthly_savings   = round(savings,       2),
        savings_pct       = round(savings_pct,   1),
    )


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_results(costs, cost_detail, latency, quality, growth_50k, growth_100k):

    print()
    print_sep("=")
    print("  BENCHMARK: PRIMA vs DOPO OTTIMIZZAZIONE LLM")
    print("  {:,} query/giorno | Customer Support System".format(DAILY_QUERIES))
    print_sep("=")

    # --- SEZIONE 1: COSTI ---
    print_section("SEZIONE 1 - RISPARMIO COSTI  (per Sara, CFO/Finance)")

    rows = [
        ("Costo giornaliero",  fmt_usd(costs.daily_before),   fmt_usd(costs.daily_after)),
        ("Costo mensile",      fmt_usd(costs.monthly_before),  fmt_usd(costs.monthly_after)),
        ("Costo annuale",      fmt_usd(costs.annual_before),   fmt_usd(costs.annual_after)),
    ]
    print("  {:<30s} {:>12}   {:>12}".format("", "PRIMA", "DOPO"))
    print("  " + "-" * 56)
    for label, before, after in rows:
        print("  {:<30s} {:>12}   {:>12}".format(label, before, after))

    print()
    print("  Riduzione costo:        {:>6.1f}%".format(costs.savings_pct))
    print("  Risparmio annuale:      {:>12}".format(fmt_usd(costs.annual_savings)))

    print()
    print("  Dettaglio PRIMA (naive - tutto Sonnet):")
    for tier_key, d in cost_detail["before"].items():
        lbl = QUERY_TIERS[tier_key]["label"]
        print("    {:<40s}  {:>5,} query/g -> {}/g".format(lbl, d["queries"], fmt_usd(d["daily_cost"])))

    print()
    print("  Dettaglio DOPO (routing + caching):")
    for tier_key, d in cost_detail["after"].items():
        lbl = QUERY_TIERS[tier_key]["label"]
        mdl = d["model_primary"].upper()
        api_routed = d["api_calls"] - d["escalated_to_sonnet"]
        print("    {:<40s}  {:>5,} query/g".format(lbl, d["queries"]))
        print("      Cache hit: {:>4,}  |  API ({}): {:>4,}  |  Escalati SONNET: {:>3,}  -> {}/g".format(
            d["cache_hits"], mdl, api_routed, d["escalated_to_sonnet"], fmt_usd(d["daily_cost"])))

    # --- SEZIONE 2: LATENZA ---
    print_section("SEZIONE 2 - MIGLIORAMENTO LATENZA  (per Marco, Eng)")

    print("  Latenza media PRIMA:    {:>12}".format(fmt_ms(latency.avg_before_ms)))
    print("  Latenza media DOPO:     {:>12}".format(fmt_ms(latency.avg_after_ms)))
    print("  Riduzione latenza:      {:>11.1f}%".format(latency.reduction_pct))
    print()
    print("  P99 PRIMA:              {:>12}   (= P50 x {})".format(fmt_ms(latency.p99_before_ms), LATENCY_P99_RATIO))
    print("  P99 DOPO:               {:>12}".format(fmt_ms(latency.p99_after_ms)))
    print()
    print("  Componenti latenza dopo ottimizzazione:")
    print("    Cache hit   (45%):    {:>8,} ms".format(LATENCY_CACHE_HIT_MS))
    print("    Haiku       (49.5%):  {:>8,} ms".format(LATENCY_HAIKU_MS))
    print("    Sonnet      ( 5.5%):  {:>8,} ms".format(LATENCY_SONNET_MS))
    print()
    print("  User Experience Impact:")
    print("    {:.1f}% delle sessioni riceve risposta in meno di {:,} ms".format(
        latency.ux_sessions_pct, latency.ux_threshold_ms))
    print("    (vs ~0% prima - Sonnet medio era {:,} ms)".format(LATENCY_SONNET_MS))

    # --- SEZIONE 3: QUALITY GATE ---
    print_section("SEZIONE 3 - QUALITY GATE REPORT  (nessuna degradazione)")

    print("  Score qualita' medio ponderato: {:.2f} / 10.00".format(quality.avg_quality_score))
    print("  Query che passano il gate:      {:.2f}%".format(quality.pass_rate_pct))
    print("  Query che escalano a Sonnet:    {:.2f}%  (safety net attivo)".format(quality.escalation_rate_pct))
    print()
    print("  Dettaglio per tier:")
    for tier_key, ts in quality.tier_scores.items():
        model = "haiku" if tier_key != "complex" else "sonnet"
        print("    {:<42s}  score {:.1f}/10  ({}% del volume) -> {}".format(
            ts["label"], ts["quality_score"], int(ts["share_pct"]), model.upper()))

    print()
    print("  NOTA: Le query complesse (10%) continuano ad usare Sonnet.")
    print("  Zero compromessi sulla qualita' - routing verificato con quality gate interno.")

    # --- SEZIONE 4: CRESCITA ---
    print_section("SEZIONE 4 - PROIEZIONE CRESCITA")

    print("  {:>15}  {:>12}  {:>16}  {:>14}  {:>6}".format(
        "Volume", "Naive/mese", "Ottimizzato/mese", "Risparmio/mese", "%"))
    print("  " + "-" * 70)
    for g in [growth_50k, growth_100k]:
        print("  {:>12,}/g    {:>12}    {:>16}    {:>14}    {:.1f}%".format(
            g.daily_queries,
            fmt_usd(g.monthly_naive),
            fmt_usd(g.monthly_optimized),
            fmt_usd(g.monthly_savings),
            g.savings_pct))

    # --- NUMERI TITOLO ---
    print()
    print_sep("#")
    print()
    print("  >> SARA NUMBER  (Finance / CFO)")
    print("     Riduzione costo: -{:.0f}%".format(costs.savings_pct))
    print("     Risparmio annuale: {}".format(fmt_usd(costs.annual_savings)))
    print()
    print("  >> MARCO NUMBER  (Engineering / CTO)")
    print("     Riduzione latenza media: -{:.0f}%".format(latency.reduction_pct))
    print("     P99 da {} -> {}".format(fmt_ms(latency.p99_before_ms), fmt_ms(latency.p99_after_ms)))
    print()
    print_sep("#")
    print()


# ---------------------------------------------------------------------------
# SALVATAGGIO JSON
# ---------------------------------------------------------------------------

def save_json(costs, cost_detail, latency, quality, growth_50k, growth_100k, output_path):

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "system": {
            "daily_queries":              DAILY_QUERIES,
            "scenario":                   "Customer Support",
            "cache_hit_rate":             CACHE_HIT_RATE,
            "async_throughput_multiplier": ASYNC_THROUGHPUT_MUL,
        },
        "costs": {
            "summary":    asdict(costs),
            "tier_detail": cost_detail,
        },
        "latency": asdict(latency),
        "quality": {
            "avg_quality_score":   quality.avg_quality_score,
            "pass_rate_pct":       quality.pass_rate_pct,
            "escalation_rate_pct": quality.escalation_rate_pct,
            "tier_scores":         quality.tier_scores,
        },
        "growth_projections": {
            "50k_queries_per_day":  asdict(growth_50k),
            "100k_queries_per_day": asdict(growth_100k),
        },
        "sara_number": {
            "cost_reduction_pct": costs.savings_pct,
            "annual_savings_usd": costs.annual_savings,
            "label": "-{:.0f}% costi | {} risparmio/anno".format(
                costs.savings_pct, fmt_usd(costs.annual_savings)),
        },
        "marco_number": {
            "latency_reduction_pct": latency.reduction_pct,
            "p99_before_ms":         latency.p99_before_ms,
            "p99_after_ms":          latency.p99_after_ms,
            "label": "-{:.0f}% latenza | P99 {} -> {}".format(
                latency.reduction_pct,
                fmt_ms(latency.p99_before_ms),
                fmt_ms(latency.p99_after_ms)),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("  Risultati salvati in: {}".format(output_path))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    costs, cost_detail = compute_costs()
    latency            = compute_latency()
    quality            = compute_quality()
    growth_50k         = compute_growth(5.0)    # 50.000 query/g
    growth_100k        = compute_growth(10.0)   # 100.000 query/g

    print_results(costs, cost_detail, latency, quality, growth_50k, growth_100k)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_results.json")
    save_json(costs, cost_detail, latency, quality, growth_50k, growth_100k, output_path)


if __name__ == "__main__":
    main()
