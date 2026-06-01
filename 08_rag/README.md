# Chapter 8 — Context Management and RAG Cost Optimization

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/08_rag

A naive RAG system adding 10 chunks of 500 tokens each costs $750/day at 10,000 queries. This chapter cuts that by 60%.

- **rag_cost_analyzer.py** — Profiles cost by component: embedding, retrieval, context injection, LLM call. Recommends optimal chunk size for your corpus.
- **context_budget_manager.py** — Selects the most relevant chunks that fit within a fixed token budget, using relevance scoring and position penalty.
- **embedding_cache.py** — SHA256-keyed embedding cache with batch computation, hit rate statistics and monthly savings estimation.

## Real benchmarks from this chapter
- Optimal chunk size: 256-512 tokens for most technical corpora
- Context reduction: 55-65% with budget manager vs naive top-k
- Embedding cache hit rate: 60%+ on production query patterns

## Quick start
```
pip install anthropic python-dotenv sentence-transformers numpy tiktoken
python rag_cost_analyzer.py
python context_budget_manager.py
python embedding_cache.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
