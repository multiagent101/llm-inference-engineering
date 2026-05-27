#!/usr/bin/env python3
"""Cost tracking system for Anthropic API calls."""

import functools
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# USD per token for supported models
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    "claude-sonnet-4-5": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-opus-4":     {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-opus-4-7":   {"input":  5.00 / 1_000_000, "output": 25.00 / 1_000_000},
}

# Fallback pricing for unrecognised model IDs
_DEFAULT_INPUT_PRICE: float = 3.00 / 1_000_000
_DEFAULT_OUTPUT_PRICE: float = 15.00 / 1_000_000


# ---------------------------------------------------------------------------
# Mock types (mirror Anthropic SDK response shape for testing)
# ---------------------------------------------------------------------------

@dataclass
class _MockUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class MockMessage:
    """Lightweight stand-in for anthropic.types.Message used in unit tests."""

    model: str
    usage: _MockUsage


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Records and analyses Anthropic API call costs in a local SQLite database.

    Usage
    -----
    Instantiate once (e.g. at module level), then apply the ``track_cost``
    decorator to any function that returns an Anthropic ``Message`` object::

        tracker = CostTracker()

        @tracker.track_cost(feature_name="chat", user_id="u1", team_name="product")
        def ask(prompt: str) -> anthropic.types.Message:
            return client.messages.create(...)

    The decorated function's return value is passed through unchanged.
    """

    def __init__(self, db_path: str = "cost_tracking.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create the schema if the database does not yet exist."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name    TEXT    NOT NULL,
                    user_id         TEXT    NOT NULL,
                    team_name       TEXT    NOT NULL,
                    model           TEXT    NOT NULL,
                    input_tokens    INTEGER NOT NULL,
                    output_tokens   INTEGER NOT NULL,
                    cost_usd        REAL    NOT NULL,
                    latency_ms      REAL    NOT NULL,
                    timestamp       TEXT    NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts      ON api_calls(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feature ON api_calls(feature_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_team    ON api_calls(team_name)")

    def _since(self, days: int) -> str:
        """ISO-8601 timestamp for N days ago (UTC)."""
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    @staticmethod
    def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        prices = MODEL_PRICING.get(
            model,
            {"input": _DEFAULT_INPUT_PRICE, "output": _DEFAULT_OUTPUT_PRICE},
        )
        return input_tokens * prices["input"] + output_tokens * prices["output"]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        feature_name: str,
        user_id: str,
        team_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Insert one API call record directly (bypassing the decorator)."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO api_calls
                    (feature_name, user_id, team_name, model,
                     input_tokens, output_tokens, cost_usd, latency_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (feature_name, user_id, team_name, model,
                 input_tokens, output_tokens, cost_usd, latency_ms, ts),
            )

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def track_cost(
        self,
        feature_name: str,
        user_id: str,
        team_name: str,
    ) -> Callable[[F], F]:
        """
        Decorator factory that auto-records cost for any function returning
        an Anthropic ``Message`` (or any object exposing ``.model`` and
        ``.usage.input_tokens`` / ``.usage.output_tokens``).

        Args:
            feature_name: Logical feature label (e.g. ``"summarization"``).
            user_id:      End-user identifier.
            team_name:    Team that owns the feature.

        Example::

            @tracker.track_cost("rag-search", "u99", "search-team")
            def search(query: str) -> anthropic.types.Message:
                return client.messages.create(model="claude-haiku-4-5", ...)
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000

                model: str = getattr(result, "model", "unknown")
                usage = getattr(result, "usage", None)
                input_tokens: int = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens: int = getattr(usage, "output_tokens", 0) if usage else 0
                cost_usd = self._compute_cost(model, input_tokens, output_tokens)

                self.record(
                    feature_name=feature_name,
                    user_id=user_id,
                    team_name=team_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                )
                return result

            return wrapper  # type: ignore[return-value]
        return decorator

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_cost_by_feature(self, days: int = 30) -> list[dict]:
        """
        Total cost and call count grouped by feature name.

        Args:
            days: Look-back window in days.

        Returns:
            List of dicts ordered by descending total cost, each containing
            ``feature_name``, ``calls``, ``total_input_tokens``,
            ``total_output_tokens``, ``total_cost_usd``, ``avg_latency_ms``.
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT feature_name,
                       COUNT(*)              AS calls,
                       SUM(input_tokens)     AS total_input_tokens,
                       SUM(output_tokens)    AS total_output_tokens,
                       SUM(cost_usd)         AS total_cost_usd,
                       AVG(latency_ms)       AS avg_latency_ms
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY feature_name
                ORDER BY total_cost_usd DESC
                """,
                (self._since(days),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cost_by_team(self, days: int = 30) -> list[dict]:
        """
        Total cost and call count grouped by team name.

        Returns:
            List of dicts ordered by descending total cost, each containing
            ``team_name``, ``calls``, ``total_cost_usd``, ``avg_cost_per_call``.
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT team_name,
                       COUNT(*)          AS calls,
                       SUM(cost_usd)     AS total_cost_usd,
                       AVG(cost_usd)     AS avg_cost_per_call
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY team_name
                ORDER BY total_cost_usd DESC
                """,
                (self._since(days),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_totals(self, days: int = 30) -> list[dict]:
        """
        Cost and call count aggregated per calendar day (UTC).

        Returns:
            List of dicts ordered by descending date, each containing
            ``date``, ``calls``, ``total_cost_usd``, ``total_input_tokens``,
            ``total_output_tokens``.
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT DATE(timestamp)       AS date,
                       COUNT(*)              AS calls,
                       SUM(cost_usd)         AS total_cost_usd,
                       SUM(input_tokens)     AS total_input_tokens,
                       SUM(output_tokens)    AS total_output_tokens
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                """,
                (self._since(days),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_top_expensive_calls(self, limit: int = 10) -> list[dict]:
        """
        The N most expensive individual API calls in the database.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of full row dicts ordered by descending ``cost_usd``.
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, feature_name, user_id, team_name, model,
                       input_tokens, output_tokens, cost_usd, latency_ms, timestamp
                FROM api_calls
                ORDER BY cost_usd DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, days: int = 30) -> None:
        """Print a full cost breakdown report to stdout."""
        SEP  = "=" * 72
        THIN = "-" * 72

        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), SUM(cost_usd), SUM(input_tokens), SUM(output_tokens)
                FROM api_calls WHERE timestamp >= ?
                """,
                (self._since(days),),
            ).fetchone()
        total_calls, total_cost, total_in, total_out = row
        total_cost = total_cost or 0.0

        print()
        print(SEP)
        print(f"  COST TRACKING REPORT  (last {days} days)")
        print(SEP)
        print(f"  Total calls:         {total_calls or 0:>10,}")
        print(f"  Total input tokens:  {total_in or 0:>10,}")
        print(f"  Total output tokens: {total_out or 0:>10,}")
        print(f"  Total cost (USD):    ${total_cost:>11.6f}")

        # By feature
        print()
        print("  BY FEATURE")
        print(THIN)
        print(f"  {'Feature':<24} {'Calls':>6} {'Input tok':>10} {'Output tok':>11} {'Cost (USD)':>12}")
        print(f"  {'-'*24} {'-'*6} {'-'*10} {'-'*11} {'-'*12}")
        for r in self.get_cost_by_feature(days):
            print(
                f"  {r['feature_name']:<24} {r['calls']:>6,} "
                f"{r['total_input_tokens']:>10,} {r['total_output_tokens']:>11,} "
                f"${r['total_cost_usd']:>11.6f}"
            )

        # By team
        print()
        print("  BY TEAM")
        print(THIN)
        print(f"  {'Team':<20} {'Calls':>6} {'Total cost':>13} {'Avg / call':>13}")
        print(f"  {'-'*20} {'-'*6} {'-'*13} {'-'*13}")
        for r in self.get_cost_by_team(days):
            print(
                f"  {r['team_name']:<20} {r['calls']:>6,} "
                f"${r['total_cost_usd']:>12.6f} ${r['avg_cost_per_call']:>12.6f}"
            )

        # Daily totals
        print()
        print("  DAILY TOTALS")
        print(THIN)
        print(f"  {'Date':<12} {'Calls':>6} {'Total cost':>13}")
        print(f"  {'-'*12} {'-'*6} {'-'*13}")
        for r in self.get_daily_totals(days):
            print(f"  {r['date']:<12} {r['calls']:>6,} ${r['total_cost_usd']:>12.6f}")

        # Top 5 most expensive calls
        print()
        print("  TOP 5 MOST EXPENSIVE CALLS")
        print(THIN)
        print(f"  {'Feature':<22} {'Team':<14} {'Model':<22} {'Cost':>10}")
        print(f"  {'-'*22} {'-'*14} {'-'*22} {'-'*10}")
        for r in self.get_top_expensive_calls(5):
            print(
                f"  {r['feature_name']:<22} {r['team_name']:<14} "
                f"{r['model']:<22} ${r['cost_usd']:>9.6f}"
            )

        print(SEP)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    DB_PATH = "cost_tracking_demo.db"

    # Start fresh for a clean demo run
    if Path(DB_PATH).exists():
        os.remove(DB_PATH)

    tracker = CostTracker(db_path=DB_PATH)

    # --- catalogue of simulated workloads ---
    SCENARIOS: list[dict] = [
        {"feature": "doc-summarization",  "team": "data-science",  "model": "claude-opus-4",     "in_range": (2000, 8000), "out_range": (300, 800)},
        {"feature": "doc-summarization",  "team": "data-science",  "model": "claude-sonnet-4-5", "in_range": (2000, 8000), "out_range": (300, 800)},
        {"feature": "customer-chat",       "team": "product",       "model": "claude-haiku-4-5",  "in_range": (200, 800),   "out_range": (100, 300)},
        {"feature": "customer-chat",       "team": "product",       "model": "claude-sonnet-4-5", "in_range": (200, 800),   "out_range": (100, 300)},
        {"feature": "code-review",         "team": "engineering",   "model": "claude-opus-4",     "in_range": (1000, 4000), "out_range": (400, 1200)},
        {"feature": "code-review",         "team": "engineering",   "model": "claude-sonnet-4-5", "in_range": (1000, 4000), "out_range": (400, 1200)},
        {"feature": "data-extraction",     "team": "data-science",  "model": "claude-haiku-4-5",  "in_range": (500, 2000),  "out_range": (100, 400)},
        {"feature": "report-generation",   "team": "analytics",     "model": "claude-sonnet-4-5", "in_range": (800, 3000),  "out_range": (500, 1500)},
        {"feature": "report-generation",   "team": "analytics",     "model": "claude-opus-4",     "in_range": (800, 3000),  "out_range": (500, 1500)},
        {"feature": "semantic-search",     "team": "search",        "model": "claude-haiku-4-5",  "in_range": (100, 500),   "out_range": (50, 200)},
    ]

    USERS: list[str] = [f"user_{i:03d}" for i in range(1, 9)]

    rng = random.Random(42)  # reproducible output

    print(f"Simulating 20 mock API calls into '{DB_PATH}' ...\n")

    for i in range(20):
        scenario = rng.choice(SCENARIOS)
        user_id  = rng.choice(USERS)
        in_tok   = rng.randint(*scenario["in_range"])
        out_tok  = rng.randint(*scenario["out_range"])
        cost     = CostTracker._compute_cost(scenario["model"], in_tok, out_tok)
        latency  = rng.uniform(300, 4000)
        # Spread timestamps across the last 7 days for realistic daily totals
        ts = (
            datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 7))
        ).isoformat()

        # Use the decorator on a mock function to exercise the full code path
        mock_response = MockMessage(
            model=scenario["model"],
            usage=_MockUsage(input_tokens=in_tok, output_tokens=out_tok),
        )

        @tracker.track_cost(
            feature_name=scenario["feature"],
            user_id=user_id,
            team_name=scenario["team"],
        )
        def _mock_call(resp: MockMessage = mock_response) -> MockMessage:
            return resp

        _mock_call()

        print(
            f"  [{i+1:>2}/20] {scenario['feature']:<24} "
            f"{scenario['model']:<22} "
            f"in={in_tok:>5} out={out_tok:>4} cost=${cost:.6f}"
        )

    tracker.print_report(days=30)
