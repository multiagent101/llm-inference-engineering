# Chapter 12 — Evaluation on a Budget and Production Playbooks

**GitHub**: https://github.com/multiagent101/llm-inference-engineering/tree/main/12_evaluation

Evaluating your LLM system properly costs as much as running it — unless you have a strategy. This chapter gives you both: cheap evaluation and crisis playbooks.

- **cheap_judge.py** — LLM-as-judge using claude-haiku-4-5 (cheapest model) with configurable criteria and 10% sampling. Reduces evaluation cost by 90%.
- **regression_detector.py** — Detects quality regressions against a fixed baseline using statistical significance testing. Alerts before users notice.
- **capacity_planner.py** — Projects future costs using linear regression on historical data. Outputs month-by-month estimates with confidence intervals for three growth scenarios.

## Real benchmarks from this chapter
- Evaluation cost reduction: 90% with 10% sampling vs full evaluation
- Regression detection: statistically significant at p<0.05
- Capacity planning accuracy: within 15% on 90-day historical data

## Quick start
```
pip install anthropic python-dotenv numpy
python cheap_judge.py
python regression_detector.py
python capacity_planner.py
```

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
