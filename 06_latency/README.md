# Chapter 6 — Latency Anatomy: TTFT, Throughput and Perceived Speed

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/06_latency

Your p50 latency is acceptable. Your p99 is what users complain about. This chapter gives you the tools to measure, monitor and improve both.

- **latency_profiler.py** — Collects latency measurements and computes p50, p75, p90, p95, p99, p99.9 percentiles with histogram visualization.
- **streaming_handler.py** — Manages streaming responses with TTFT measurement, per-token callbacks and perceived speed scoring.
- **sla_monitor.py** — Monitors SLA compliance in real time. Alerts when p99 exceeds configured thresholds. Saves violation history.

## Real benchmarks from this chapter
- TTFT measured: 2,409ms (claude-haiku-4-5, streaming)
- P99 estimated: 4.2x P50 on realistic traffic distributions
- SLA compliance: 94.2% at 5,000ms p99 threshold

## Quick start
```
pip install anthropic python-dotenv
python latency_profiler.py
python streaming_handler.py
python sla_monitor.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
