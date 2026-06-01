# Chapter 11 — Observability: Monitoring a System You Can't See Inside

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/11_observability

You cannot open the model and look inside. But you can observe everything around it. This chapter builds the complete observability stack.

- **structured_logger.py** — Structured JSON Lines logger with PII scrubbing, configurable sampling rate and log rotation. Never logs raw prompts.
- **metrics_exporter.py** — Prometheus-compatible metrics exporter with HTTP server on port 8000. Ready to scrape with Prometheus and visualize with Grafana.

## Real benchmarks from this chapter
- PII scrubbing: email, phone, credit card patterns detected and removed
- Metrics overhead: under 2ms per request
- Prometheus scrape endpoint: /metrics on port 8000

## Quick start
```
pip install anthropic python-dotenv
python structured_logger.py
python metrics_exporter.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
