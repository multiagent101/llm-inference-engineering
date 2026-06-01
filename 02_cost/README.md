# Chapter 2 — Cost Anatomy and the Measurement Foundation

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/02_cost

You cannot reduce what you cannot measure. This chapter builds the complete cost measurement stack.

- **cost_tracker.py** — Decorator-based cost tracker. Wraps any function that calls the API and logs cost, tokens, latency, feature name and team to a local SQLite database.
- **anomaly_detector.py** — Detects cost spikes automatically using Z-score and EMA. Generates alerts with severity levels: WARNING, CRITICAL, EMERGENCY.
- **cost_dashboard.py** — Generates a standalone HTML dashboard with 30-day cost history, top features, top teams and recent calls. No server required.

## Real benchmarks from this chapter
- code-review on claude-opus-4 = 53% of total budget
- engineering and analytics cost 10x more than product per call
- Spike detected at 3.8x average → EMERGENCY (Z-score 20.2)

## Quick start
```
pip install anthropic python-dotenv numpy
python cost_tracker.py
python anomaly_detector.py
python cost_dashboard.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
