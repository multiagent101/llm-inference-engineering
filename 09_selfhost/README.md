# Chapter 9 — When and How to Self-Host: The Real Break-Even Analysis

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/09_selfhost

A team estimated $800/month for self-hosting. The real cost was $2,100/month. This chapter shows you the analysis they missed.

- **breakeven_calculator.py** — Calculates real self-hosting break-even including all hidden costs: GPU, ops engineer time, networking, monitoring, amortized setup. Includes sensitivity analysis.
- **quantization_bench.py** — Benchmarks quality degradation per quantization level (FP16, INT8, INT4) across task categories: coding, math, reasoning, factual.

## Real benchmarks from this chapter
- Break-even point: approximately 500K-800K queries/day for most configurations
- INT8 quality loss: ~3% vs FP16 baseline
- INT4 quality loss: ~12% vs FP16 baseline (significant on math/reasoning)

## Quick start
```
pip install anthropic python-dotenv
python breakeven_calculator.py
python quantization_bench.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
