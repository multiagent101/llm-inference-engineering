# Benchmarks — Before and After Optimization

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/benchmarks

This directory contains the benchmark script that measures real cost and latency improvements from applying the techniques in the book.

## What the benchmark measures

**Scenario**: customer support system running 10,000 queries/day with realistic traffic mix:
- 60% simple queries (FAQ, greetings) — avg 50 tokens input, 100 output
- 30% medium queries (technical issues) — avg 300 tokens input, 300 output
- 10% complex queries (advanced troubleshooting) — avg 800 tokens input, 500 output

**Before optimization**: claude-sonnet-4-5 for all queries, no caching, serial requests

**After optimization**: model routing with quality gate + 45% semantic cache hit rate

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Daily API cost | $35.70 | $9.64 | -73% |
| Annual cost | $13,031 | $3,518 | -$9,498 saved |
| Average latency | 3,200ms | 1,373ms | -57% |
| P99 latency | 13,440ms | 5,770ms | -57% |

Quality is maintained: model routing uses a quality gate that verifies every response before returning it. Complex queries (10%) always use the full Sonnet model.

## Run the benchmark
```
python before_after_optimization.py
```

Results are saved to `optimization_results.json`.

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
