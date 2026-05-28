from __future__ import annotations

"""LLM metrics exporter -- Prometheus text format (exposition version 0.0.4)."""

import bisect
import random
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional


_DURATION_BUCKETS: tuple[float, ...] = (
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
)
_METRICS_PORT: int = 8000
_CONTENT_TYPE: str = "text/plain; version=0.0.4; charset=utf-8"


class LLMMetricsExporter:
    """Collects LLM call metrics and exports them in Prometheus text format.

    Metrics exported:
        llm_requests_total         counter   labels: model, status
        llm_request_duration_ms    histogram
        llm_tokens_total           counter   labels: type (input/output)
        llm_cost_usd_total         counter
        llm_cache_hits_total       counter
        llm_errors_total           counter   labels: error_type

    Thread-safe: all state is protected by a single lock so record_request()
    and export_prometheus() can be called concurrently from any thread.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()

        # (model, status) -> call count
        self._requests: dict[tuple[str, str], int] = {}

        # Histogram stored as non-cumulative bucket counts.
        # _dur_nc[i] = observations in (prev_boundary, _DURATION_BUCKETS[i]].
        # _dur_nc[-1] = observations above the last boundary (overflow / +Inf).
        self._dur_nc: list[int] = [0] * (len(_DURATION_BUCKETS) + 1)
        self._dur_sum: float = 0.0
        self._dur_count: int = 0

        self._tokens_input: int = 0
        self._tokens_output: int = 0
        self._cost_usd: float = 0.0
        self._cache_hits: int = 0

        # error_type -> count
        self._errors: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(
        self,
        model: str,
        status: str,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cached: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Record a single LLM API call.

        Args:
            model: Model identifier, e.g. "claude-haiku-4-5".
            status: "success" or "error".
            duration_ms: End-to-end latency in milliseconds.
            input_tokens: Prompt tokens consumed.
            output_tokens: Completion tokens generated.
            cost_usd: Monetary cost of the call in USD.
            cached: True when the response was served from a semantic cache.
            error: Exception class name if status == "error", otherwise None.
        """
        with self._lock:
            key = (model, status)
            self._requests[key] = self._requests.get(key, 0) + 1

            idx = bisect.bisect_left(_DURATION_BUCKETS, duration_ms)
            self._dur_nc[idx] += 1
            self._dur_sum += duration_ms
            self._dur_count += 1

            self._tokens_input += input_tokens
            self._tokens_output += output_tokens
            self._cost_usd += cost_usd

            if cached:
                self._cache_hits += 1
            if error is not None:
                self._errors[error] = self._errors.get(error, 0) + 1

    def export_prometheus(self) -> str:
        """Serialise current metrics as a Prometheus text exposition string.

        Returns:
            A UTF-8-safe string conforming to Prometheus exposition format 0.0.4.
            Each metric family is preceded by HELP and TYPE comment lines.
            The histogram uses cumulative bucket counts as required by the spec.
        """
        with self._lock:
            lines: list[str] = []

            # --- llm_requests_total ---
            lines.append("# HELP llm_requests_total Total LLM API requests")
            lines.append("# TYPE llm_requests_total counter")
            for (model, status), count in sorted(self._requests.items()):
                lines.append(
                    f'llm_requests_total{{model="{model}",status="{status}"}} {count}'
                )

            # --- llm_request_duration_ms (histogram) ---
            lines.append("")
            lines.append(
                "# HELP llm_request_duration_ms Request duration in milliseconds"
            )
            lines.append("# TYPE llm_request_duration_ms histogram")
            cumulative = 0
            for i, boundary in enumerate(_DURATION_BUCKETS):
                cumulative += self._dur_nc[i]
                lines.append(
                    f'llm_request_duration_ms_bucket{{le="{boundary}"}} {cumulative}'
                )
            lines.append(
                f'llm_request_duration_ms_bucket{{le="+Inf"}} {self._dur_count}'
            )
            lines.append(f"llm_request_duration_ms_sum {self._dur_sum:.1f}")
            lines.append(f"llm_request_duration_ms_count {self._dur_count}")

            # --- llm_tokens_total ---
            lines.append("")
            lines.append("# HELP llm_tokens_total Total tokens processed")
            lines.append("# TYPE llm_tokens_total counter")
            lines.append(f'llm_tokens_total{{type="input"}} {self._tokens_input}')
            lines.append(f'llm_tokens_total{{type="output"}} {self._tokens_output}')

            # --- llm_cost_usd_total ---
            lines.append("")
            lines.append("# HELP llm_cost_usd_total Total cost in USD")
            lines.append("# TYPE llm_cost_usd_total counter")
            lines.append(f"llm_cost_usd_total {self._cost_usd:.6f}")

            # --- llm_cache_hits_total ---
            lines.append("")
            lines.append("# HELP llm_cache_hits_total Total cache hits")
            lines.append("# TYPE llm_cache_hits_total counter")
            lines.append(f"llm_cache_hits_total {self._cache_hits}")

            # --- llm_errors_total ---
            lines.append("")
            lines.append("# HELP llm_errors_total Total errors by type")
            lines.append("# TYPE llm_errors_total counter")
            for error_type, count in sorted(self._errors.items()):
                lines.append(
                    f'llm_errors_total{{error_type="{error_type}"}} {count}'
                )

            lines.append("")  # trailing newline required by Prometheus spec
            return "\n".join(lines)

    def save_prometheus_file(self, path: str) -> None:
        """Write metrics to a .prom file for the node_exporter textfile collector.

        Args:
            path: Destination file path. Parent directories are created if needed.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.export_prometheus(), encoding="utf-8")

    def make_http_server(self, port: int = _METRICS_PORT) -> HTTPServer:
        """Create an HTTP server that exposes GET /metrics for Prometheus scraping.

        The caller is responsible for starting and stopping the server::

            server = exporter.make_http_server()
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            # ... application runs ...
            server.shutdown()

        Args:
            port: TCP port to bind. Defaults to 8000.

        Returns:
            A configured but not yet started HTTPServer instance.
        """
        exporter = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    body = exporter.export_prometheus().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", _CONTENT_TYPE)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt: str, *args: object) -> None:
                pass  # suppress default stderr access log

        return HTTPServer(("", port), _Handler)


# ======================================================================
# Demo
# ======================================================================

if __name__ == "__main__":
    _MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6"]
    _HAIKU_COSTS = (0.80, 4.00)    # $/1M tokens: input, output
    _SONNET_COSTS = (3.00, 15.00)
    _ERROR_TYPES = ["TimeoutError", "RateLimitError", "ValidationError"]
    _SEP = "=" * 68

    rng = random.Random(42)
    exporter = LLMMetricsExporter()

    # ------------------------------------------------------------------
    # Phase 1: record 100 simulated requests
    # ------------------------------------------------------------------
    print(_SEP)
    print("  RECORDING 100 SIMULATED REQUESTS")
    print(_SEP)

    total_success = 0
    total_error = 0
    total_cached = 0

    for _i in range(100):
        model = rng.choice(_MODELS)
        is_error = rng.random() < 0.12
        is_cached = rng.random() < 0.18
        duration_ms = rng.uniform(80.0, 4_800.0)
        input_tokens = rng.randint(50, 500)
        output_tokens = 0 if is_error else rng.randint(20, 400)
        costs = _HAIKU_COSTS if model == "claude-haiku-4-5" else _SONNET_COSTS
        cost_usd = (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000
        if is_cached:
            cost_usd *= 0.1   # semantic cache reduces compute cost
        error_type = rng.choice(_ERROR_TYPES) if is_error else None

        exporter.record_request(
            model=model,
            status="error" if is_error else "success",
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached=is_cached,
            error=error_type,
        )

        if is_error:
            total_error += 1
        else:
            total_success += 1
        if is_cached:
            total_cached += 1

    print()
    print(f"  Success   : {total_success}")
    print(f"  Error     : {total_error}")
    print(f"  Cached    : {total_cached}")
    print()
    print("  Requests by (model, status):")
    for (model, status), count in sorted(exporter._requests.items()):
        print(f"    {model:<28} {status:<10} {count}")
    print()
    print(f"  Tokens    : {exporter._tokens_input} input, {exporter._tokens_output} output")
    print(f"  Cost      : ${exporter._cost_usd:.4f} USD")
    print(
        f"  Errors    : "
        + ", ".join(f"{k}={v}" for k, v in sorted(exporter._errors.items()))
    )

    # Duration histogram (non-cumulative bar chart)
    print()
    print("  Duration histogram (ms, non-cumulative):")
    _bucket_labels = [f"le={int(b):<6}" for b in _DURATION_BUCKETS] + ["overflow"]
    for label, count in zip(_bucket_labels, exporter._dur_nc):
        bar = "#" * count
        print(f"    {label}  {count:3d}  {bar}")

    # ------------------------------------------------------------------
    # Phase 2: Prometheus text output
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print("  PROMETHEUS TEXT EXPORT")
    print(_SEP)
    prom_text = exporter.export_prometheus()
    print(prom_text)

    # ------------------------------------------------------------------
    # Phase 3: save to file
    # ------------------------------------------------------------------
    prom_path = "metrics/llm_metrics.prom"
    exporter.save_prometheus_file(prom_path)
    print(f"  Saved: {prom_path}")

    # ------------------------------------------------------------------
    # Phase 4: HTTP server -- serve for 10 seconds
    # ------------------------------------------------------------------
    print()
    print(_SEP)
    print(f"  HTTP SERVER  port={_METRICS_PORT}")
    print(_SEP)

    server = exporter.make_http_server(port=_METRICS_PORT)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"  Listening: http://localhost:{_METRICS_PORT}/metrics")
    time.sleep(0.2)  # let the server complete its bind

    # Verify with a local GET /metrics
    try:
        url = f"http://localhost:{_METRICS_PORT}/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            raw = resp.read().decode("utf-8")
        line_count = raw.count("\n")
        size_bytes = len(raw)
        print(f"  GET /metrics -> HTTP 200  {line_count} lines  {size_bytes} bytes")
        print()
        visible = [ln for ln in raw.splitlines() if ln][:12]
        for ln in visible:
            print(f"    {ln}")
        print("    [...]")
    except Exception as exc:
        print(f"  GET /metrics failed: {exc}")

    # 10-second countdown
    print()
    for remaining in range(10, 0, -1):
        sys.stdout.write(f"\r  Running... {remaining:2d}s remaining ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r  Shutting down...                  \n")
    sys.stdout.flush()

    server.shutdown()
    print("  Done.")
