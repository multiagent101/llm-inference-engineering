# Chapter 4 — Caching Strategies: From Exact Match to Semantic Cache

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/04_caching

A well-designed caching system eliminates 40-70% of API calls. This chapter implements three layers of cache.

- **exact_cache.py** — SHA256-keyed exact match cache with TTL, persistence and hit rate statistics.
- **semantic_cache.py** — Embedding-based cache that finds semantically similar queries above a configurable similarity threshold. Returns cached responses for paraphrased versions of the same question.
- **cache_roi_calculator.py** — Calculates real monthly savings from your cache hit rate across three scenarios: startup, scale-up, enterprise.

## Real benchmarks from this chapter
- Semantic cache hit rate: 60-75% on paraphrased query sets
- Optimal similarity threshold: 0.85 for most production workloads
- ROI: positive from day 1 at 10,000+ queries/day

## Quick start
```
pip install anthropic python-dotenv sentence-transformers numpy
python exact_cache.py
python semantic_cache.py
python cache_roi_calculator.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
