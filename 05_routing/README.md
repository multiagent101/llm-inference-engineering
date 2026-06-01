# Chapter 5 — Model Routing: The Right Model for Every Task

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/05_routing

Not every query needs the most powerful model. A routing system that sends simple queries to fast, cheap models reduces costs by 50-60% without quality degradation.

- **task_classifier.py** — Classifies every request by complexity: SIMPLE, STANDARD, COMPLEX, EXPERT. Uses rule-based and LLM-based approaches with configurable thresholds.
- **model_router.py** — Routes each request to the appropriate model tier with quality gate verification and automatic fallback to a more powerful model if quality is insufficient.
- **router_ab_test.py** — A/B tests two routing configurations with statistical significance (t-test). Produces a recommendation with confidence level.

## Real benchmarks from this chapter
- 58% of queries routed to Haiku (cheapest tier)
- Fallback rate: under 8% in production workloads
- Cost reduction vs always-using-Opus: 55-60%

## Quick start
```
pip install anthropic python-dotenv scipy
python task_classifier.py
python model_router.py
python router_ab_test.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
