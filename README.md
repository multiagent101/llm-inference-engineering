# LLM Inference Engineering — Production Code Repository

**GitHub**: https://github.com/multiagent101/llm-inference-engineering

Companion code repository for **LLM Inference Engineering Handbook** by Kylan C. Holt.

## What this repository contains

Production-ready Python modules for every chapter of the book. Each module is tested, typed and documented. The code in the book shows simplified versions — this repository contains the full implementations.

## Results you can replicate

| Optimization | Technique | Impact |
|---|---|---|
| API cost reduction | Model routing + caching | -73% |
| Latency reduction | Streaming + async + routing | -57% |
| Evaluation cost | 10% sampling + cheap judge | -90% |
| RAG context | Budget manager + reranking | -60% |

All numbers were measured on real API calls, not estimated.

## Repository structure

| Chapter | Topic | Modules |
|---|---|---|
| [01_anatomy](https://github.com/multiagent101/llm-inference-engineering/tree/main/01_anatomy) | Inference profiling and cost estimation | inference_profiler.py, cost_estimator.py |
| [02_cost](https://github.com/multiagent101/llm-inference-engineering/tree/main/02_cost) | Cost tracking, anomaly detection, dashboard | cost_tracker.py, anomaly_detector.py, cost_dashboard.py |
| [03_prompts](https://github.com/multiagent101/llm-inference-engineering/tree/main/03_prompts) | Token counting and prompt compression | token_counter.py, prompt_compressor.py, prompt_optimizer.py |
| [04_caching](https://github.com/multiagent101/llm-inference-engineering/tree/main/04_caching) | Exact match and semantic caching | exact_cache.py, semantic_cache.py, cache_roi_calculator.py |
| [05_routing](https://github.com/multiagent101/llm-inference-engineering/tree/main/05_routing) | Model routing and A/B testing | task_classifier.py, model_router.py, router_ab_test.py |
| [06_latency](https://github.com/multiagent101/llm-inference-engineering/tree/main/06_latency) | Latency profiling and SLA monitoring | latency_profiler.py, streaming_handler.py, sla_monitor.py |
| [07_requests](https://github.com/multiagent101/llm-inference-engineering/tree/main/07_requests) | Async client, queuing, batch API | async_client.py, request_queue.py, batch_client.py |
| [08_rag](https://github.com/multiagent101/llm-inference-engineering/tree/main/08_rag) | RAG cost optimization | rag_cost_analyzer.py, context_budget_manager.py, embedding_cache.py |
| [09_selfhost](https://github.com/multiagent101/llm-inference-engineering/tree/main/09_selfhost) | Self-hosting break-even analysis | breakeven_calculator.py, quantization_bench.py |
| [10_reliability](https://github.com/multiagent101/llm-inference-engineering/tree/main/10_reliability) | Retry, circuit breaker, failover | retry_client.py, circuit_breaker.py, provider_failover.py |
| [11_observability](https://github.com/multiagent101/llm-inference-engineering/tree/main/11_observability) | Logging and metrics | structured_logger.py, metrics_exporter.py |
| [12_evaluation](https://github.com/multiagent101/llm-inference-engineering/tree/main/12_evaluation) | Evaluation and capacity planning | cheap_judge.py, regression_detector.py, capacity_planner.py |
| [benchmarks](https://github.com/multiagent101/llm-inference-engineering/tree/main/benchmarks) | Before/after optimization benchmark | before_after_optimization.py |

## Setup

```
git clone https://github.com/multiagent101/llm-inference-engineering
cd llm-inference-engineering
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

Each chapter directory has its own README with specific installation instructions.

## Full book
LLM Inference Engineering Handbook by Kylan C. Holt
