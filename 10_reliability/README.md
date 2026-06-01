# Chapter 10 — Reliability Engineering: Retries, Fallbacks and Circuit Breakers

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/10_reliability

A retry loop bug caused 4.2 million requests during a 20-minute outage. The API recovered. The bill did not. This chapter prevents that.

- **retry_client.py** — Retry with exponential backoff, jitter and retry budget tracking. Only retries on retriable errors (429, 500, 502, 503, 529).
- **circuit_breaker.py** — Three-state circuit breaker (CLOSED/OPEN/HALF_OPEN) that stops cascading failures before they cascade.
- **provider_failover.py** — Automatic failover between providers with health checking and failover logging.

## Real benchmarks from this chapter
- Unmanaged retry cost during 20-min outage: 4.2M requests
- Circuit breaker recovery time: configurable, default 60 seconds
- Failover latency overhead: under 200ms additional

## Quick start
```
pip install anthropic python-dotenv
python retry_client.py
python circuit_breaker.py
python provider_failover.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
