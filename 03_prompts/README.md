# Chapter 3 — Prompt Optimization and Token Compression

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/03_prompts

The average production prompt has 34% redundant tokens. This chapter gives you the tools to find and remove them without losing quality.

- **token_counter.py** — Counts tokens before sending, across multiple models and providers. Includes cost estimation and model comparison.
- **prompt_compressor.py** — Compresses prompts using 4 strategies: redundancy removal, instruction compression, context pruning, few-shot abbreviation. Includes quality verification via LLM-as-judge.
- **prompt_optimizer.py** — A/B tests prompt variants automatically and ranks them by quality/cost ratio.

## Real benchmarks from this chapter
- Average token reduction: 30-40% per compression strategy
- Quality score maintained above 8.5/10 after compression
- Cost savings: proportional to token reduction

## Quick start
```
pip install anthropic python-dotenv sentence-transformers scikit-learn
python token_counter.py
python prompt_compressor.py
python prompt_optimizer.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
