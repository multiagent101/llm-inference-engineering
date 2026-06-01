# Chapter 7 — Request Optimization: Batching, Async and Parallelism

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/07_requests

Serial requests, one at a time, work until they don't. This chapter shows you how to structure requests for maximum throughput.

- **async_client.py** — Production-ready async client with semaphore-based concurrency control, rate limiting and exponential backoff retry.
- **request_queue.py** — Priority queue with HIGH/NORMAL/LOW tiers, configurable timeouts and dead letter handling for failed requests.
- **batch_client.py** — Wrapper for the Anthropic Batch API with polling, result download and cost savings calculation (50% vs standard API).

## Real benchmarks from this chapter
- Async vs serial on 10 requests: 4x throughput improvement
- Batch API cost saving: 50% vs standard API calls
- Request overhead reduction: significant at 1,000+ requests/day

## Quick start
```
pip install anthropic python-dotenv httpx
python async_client.py
python request_queue.py
python batch_client.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
