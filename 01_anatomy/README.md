# Chapter 1 — Anatomy of an LLM Inference Request

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/01_anatomy

Before you optimize anything, you need to know exactly where your time and money go.

This chapter provides two production-ready modules:

- **inference_profiler.py** — Measures real TTFT, inter-token latency, total latency, token count and cost for every API call using streaming. Results are saved to `profile_results.json`.
- **cost_estimator.py** — Estimates the cost of a prompt before sending it, across all configured models. Shows you which model is cheapest for your specific input.

## Real benchmarks from this chapter
- TTFT measured: 2,409ms
- Total latency: 3,714ms
- Cost per call: $0.000563 (claude-haiku-4-5)
- Opus costs 18-19x more than Haiku for the same prompt

## Quick start
```
pip install anthropic python-dotenv
python inference_profiler.py
python cost_estimator.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
