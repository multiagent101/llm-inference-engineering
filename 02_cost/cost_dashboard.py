#!/usr/bin/env python3
"""Static HTML cost dashboard for Anthropic API spend."""

import html as _html
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DashboardThresholds:
    """Configurable cost thresholds that trigger colour coding."""

    daily_warning_usd: float = 50.0
    daily_critical_usd: float = 100.0
    call_warning_usd: float = 0.05
    call_critical_usd: float = 0.10


# ---------------------------------------------------------------------------
# Dashboard generator
# ---------------------------------------------------------------------------

class CostDashboard:
    """
    Reads Anthropic API cost data from a SQLite database and produces a
    fully self-contained HTML file (no CDN, no server required).

    Sections
    --------
    * Summary cards — total spend for today / week / month.
    * ASCII bar chart — daily costs for the last 30 days rendered with
      block characters inside a ``<pre>`` block.
    * Top-5 features and top-5 teams by cost (last 30 days).
    * Recent 20 API calls with cost and latency.

    Values that exceed the configured thresholds are highlighted in
    yellow (warning) or red (critical).
    """

    _BAR_CHAR   = "█"   # █
    _EMPTY_CHAR = "░"   # ░
    _BAR_WIDTH  = 40         # characters

    def __init__(
        self,
        db_path: str = "cost_tracking.db",
        output_file: str = "cost_dashboard.html",
        thresholds: Optional[DashboardThresholds] = None,
    ) -> None:
        self.db_path     = Path(db_path)
        self.output_file = Path(output_file)
        self.t           = thresholds or DashboardThresholds()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def _scalar(self, sql: str, params: tuple = ()) -> float:
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
            return float(row[0] or 0.0)

    def _since(self, days: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # ------------------------------------------------------------------
    # Data queries
    # ------------------------------------------------------------------

    def _cost_period(self, days: int) -> float:
        """Total cost for the last N days."""
        return self._scalar(
            "SELECT SUM(cost_usd) FROM api_calls WHERE timestamp >= ?",
            (self._since(days),),
        )

    def _calls_period(self, days: int) -> int:
        return int(self._scalar(
            "SELECT COUNT(*) FROM api_calls WHERE timestamp >= ?",
            (self._since(days),),
        ))

    def _daily_costs(self, days: int = 30) -> list[dict]:
        """Daily cost aggregates, sorted ascending."""
        return self._query(
            """
            SELECT DATE(timestamp) AS date,
                   COUNT(*)        AS calls,
                   SUM(cost_usd)   AS total_cost_usd
            FROM api_calls
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
            """,
            (self._since(days),),
        )

    def _top_features(self, n: int = 5) -> list[dict]:
        return self._query(
            """
            SELECT feature_name,
                   COUNT(*)        AS calls,
                   SUM(cost_usd)   AS total_cost_usd,
                   AVG(latency_ms) AS avg_latency_ms
            FROM api_calls
            WHERE timestamp >= ?
            GROUP BY feature_name
            ORDER BY total_cost_usd DESC
            LIMIT ?
            """,
            (self._since(30), n),
        )

    def _top_teams(self, n: int = 5) -> list[dict]:
        return self._query(
            """
            SELECT team_name,
                   COUNT(*)      AS calls,
                   SUM(cost_usd) AS total_cost_usd,
                   AVG(cost_usd) AS avg_cost_per_call
            FROM api_calls
            WHERE timestamp >= ?
            GROUP BY team_name
            ORDER BY total_cost_usd DESC
            LIMIT ?
            """,
            (self._since(30), n),
        )

    def _recent_calls(self, n: int = 20) -> list[dict]:
        return self._query(
            """
            SELECT timestamp, feature_name, user_id, team_name, model,
                   input_tokens, output_tokens, cost_usd, latency_ms
            FROM api_calls
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (n,),
        )

    # ------------------------------------------------------------------
    # ASCII bar chart
    # ------------------------------------------------------------------

    def _ascii_chart(self, daily: list[dict]) -> str:
        """
        Render daily costs as an HTML-embedded ASCII bar chart.

        Each row shows: date │ bar █░ │ $cost [flag]

        Bars are coloured green / yellow / red via CSS spans based on the
        configured thresholds.

        Args:
            daily: List of dicts with keys ``date`` and ``total_cost_usd``.

        Returns:
            HTML string suitable for embedding inside a ``<pre>`` element.
        """
        if not daily:
            return "<em>No data available.</em>"

        max_cost = max(d["total_cost_usd"] for d in daily) or 1.0
        lines: list[str] = []

        for d in daily:
            cost   = d["total_cost_usd"]
            filled = round((cost / max_cost) * self._BAR_WIDTH)
            empty  = self._BAR_WIDTH - filled

            if cost >= self.t.daily_critical_usd:
                bar_cls, flag = "bar-crit", " [!!]"
            elif cost >= self.t.daily_warning_usd:
                bar_cls, flag = "bar-warn", " [!] "
            else:
                bar_cls, flag = "bar-ok",   ""

            bar_html = (
                f'<span class="{bar_cls}">{self._BAR_CHAR * filled}</span>'
                f'<span class="bar-empty">{self._EMPTY_CHAR * empty}</span>'
            )
            flag_html = f'<span class="{bar_cls}">{flag}</span>'
            lines.append(
                f'<span class="ch-date">{d["date"]}</span> '
                f'&#x2502; {bar_html} &#x2502; '
                f'<span class="{bar_cls}">${cost:>8.4f}</span>'
                f'{flag_html}'
            )

        # Legend
        lines.append("")
        lines.append(
            f'<span class="ch-date">Legend: </span>'
            f'<span class="bar-ok">{self._BAR_CHAR} normal</span>  '
            f'<span class="bar-warn">{self._BAR_CHAR} warning &gt;${self.t.daily_warning_usd:.0f}</span>  '
            f'<span class="bar-crit">{self._BAR_CHAR} critical &gt;${self.t.daily_critical_usd:.0f}</span>'
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    @staticmethod
    def _css() -> str:
        return """
        :root {
            --bg:      #0d1117;
            --surface: #161b22;
            --border:  #30363d;
            --text:    #c9d1d9;
            --muted:   #8b949e;
            --accent:  #58a6ff;
            --ok:      #3fb950;
            --warn:    #d29922;
            --crit:    #f85149;
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            font-size: 15px;
            line-height: 1.5;
        }

        /* ── Header ──────────────────────────────────────────────── */
        header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 1.1rem 2rem;
        }
        header h1 { font-size: 1.25rem; color: var(--accent); }
        .subtitle { color: var(--muted); font-size: .82rem; margin-top: .15rem; }

        /* ── Layout ──────────────────────────────────────────────── */
        main { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
        section { margin-bottom: 2.5rem; }
        h2 {
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
            padding-bottom: .5rem;
            margin-bottom: 1.2rem;
        }

        /* ── Cards ───────────────────────────────────────────────── */
        .cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }
        @media (max-width: 640px) { .cards { grid-template-columns: 1fr; } }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
        }
        .card-label {
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .6px;
            color: var(--muted);
        }
        .card-value {
            font-size: 2rem;
            font-weight: 700;
            margin: .35rem 0 .2rem;
        }
        .card-sub { font-size: .8rem; color: var(--muted); }

        /* ── Chart ───────────────────────────────────────────────── */
        .chart-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.2rem 1.6rem;
            overflow-x: auto;
        }
        pre {
            font-family: 'Cascadia Code', 'Fira Code', Consolas, 'Courier New', monospace;
            font-size: .81rem;
            line-height: 1.75;
            white-space: pre;
        }
        .ch-date   { color: var(--muted); }
        .bar-ok    { color: var(--ok); }
        .bar-warn  { color: var(--warn); font-weight: 600; }
        .bar-crit  { color: var(--crit); font-weight: 700; }
        .bar-empty { color: #21262d; }

        /* ── Two-column panels ───────────────────────────────────── */
        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }
        @media (max-width: 780px) { .two-col { grid-template-columns: 1fr; } }
        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .panel h2 {
            padding: 1rem 1.2rem .5rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0;
        }

        /* ── Tables ──────────────────────────────────────────────── */
        .tbl-wrap { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: .87rem; }
        thead th {
            background: var(--surface);
            color: var(--muted);
            text-align: left;
            padding: .6rem .9rem;
            font-weight: 600;
            font-size: .73rem;
            text-transform: uppercase;
            letter-spacing: .4px;
            border-bottom: 1px solid var(--border);
        }
        tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
        tbody tr:last-child { border-bottom: none; }
        tbody tr:hover { background: #1c2128; }
        td { padding: .55rem .9rem; vertical-align: middle; }
        td.num { text-align: right; font-family: Consolas, monospace; }

        /* ── Pills ───────────────────────────────────────────────── */
        .pill {
            display: inline-block;
            padding: .15rem .55rem;
            border-radius: 4px;
            font-size: .75rem;
            font-weight: 600;
        }
        .pill-ok   { background: #122117; color: var(--ok); }
        .pill-warn { background: #2d1f00; color: var(--warn); }
        .pill-crit { background: #2d0f0f; color: var(--crit); }

        /* ── Cost colour helpers ─────────────────────────────────── */
        .c-ok   { color: var(--ok); }
        .c-warn { color: var(--warn); font-weight: 600; }
        .c-crit { color: var(--crit); font-weight: 700; }
        .c-accent { color: var(--accent); }

        /* ── Footer ──────────────────────────────────────────────── */
        footer {
            text-align: center;
            color: var(--muted);
            font-size: .78rem;
            padding: 2rem 0 1rem;
            border-top: 1px solid var(--border);
        }
        """

    # ------------------------------------------------------------------
    # HTML generation helpers
    # ------------------------------------------------------------------

    def _cost_css(self, cost: float, warn: float, crit: float) -> str:
        if cost >= crit:
            return "c-crit"
        if cost >= warn:
            return "c-warn"
        return "c-ok"

    def _pill_css(self, cost: float, warn: float, crit: float) -> str:
        if cost >= crit:
            return "pill pill-crit"
        if cost >= warn:
            return "pill pill-warn"
        return "pill pill-ok"

    def _card(
        self, label: str, cost: float, calls: int, warn: float, crit: float
    ) -> str:
        css = self._cost_css(cost, warn, crit)
        return (
            f'<div class="card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value {css}">${cost:.4f}</div>'
            f'<div class="card-sub">{calls:,} API calls</div>'
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """
        Query the database, build all HTML sections, and return the
        complete standalone HTML document as a string.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        today_cost  = self._cost_period(1)
        week_cost   = self._cost_period(7)
        month_cost  = self._cost_period(30)
        today_calls = self._calls_period(1)
        week_calls  = self._calls_period(7)
        month_calls = self._calls_period(30)

        daily    = self._daily_costs(30)
        features = self._top_features(5)
        teams    = self._top_teams(5)
        recent   = self._recent_calls(20)

        # ── Cards ──────────────────────────────────────────────────
        dw, dc = self.t.daily_warning_usd, self.t.daily_critical_usd
        cards_html = (
            self._card("Today",      today_cost,  today_calls,  dw,      dc)
            + self._card("This Week", week_cost,  week_calls,   dw * 7,  dc * 7)
            + self._card("This Month", month_cost, month_calls, dw * 30, dc * 30)
        )

        # ── ASCII chart ────────────────────────────────────────────
        chart_html = self._ascii_chart(daily)

        # ── Top features ───────────────────────────────────────────
        feat_rows = ""
        for r in features:
            cost = r["total_cost_usd"]
            pill = self._pill_css(cost, dw, dc)
            feat_rows += (
                f"<tr>"
                f"<td>{_html.escape(r['feature_name'])}</td>"
                f'<td class="num">{r["calls"]:,}</td>'
                f'<td class="num"><span class="{pill}">${cost:.4f}</span></td>'
                f'<td class="num">{r["avg_latency_ms"]:.0f} ms</td>'
                f"</tr>"
            )

        # ── Top teams ──────────────────────────────────────────────
        team_rows = ""
        for r in teams:
            cost = r["total_cost_usd"]
            pill = self._pill_css(cost, dw, dc)
            team_rows += (
                f"<tr>"
                f"<td>{_html.escape(r['team_name'])}</td>"
                f'<td class="num">{r["calls"]:,}</td>'
                f'<td class="num"><span class="{pill}">${cost:.4f}</span></td>'
                f'<td class="num">${r["avg_cost_per_call"]:.5f}</td>'
                f"</tr>"
            )

        # ── Recent calls ───────────────────────────────────────────
        call_rows = ""
        cw, cc = self.t.call_warning_usd, self.t.call_critical_usd
        for r in recent:
            cost     = r["cost_usd"]
            ts       = r["timestamp"][:19].replace("T", " ")
            cost_css = self._cost_css(cost, cw, cc)
            call_rows += (
                f"<tr>"
                f'<td style="font-size:.76rem;color:var(--muted)">{_html.escape(ts)}</td>'
                f"<td>{_html.escape(r['feature_name'])}</td>"
                f"<td>{_html.escape(r['team_name'])}</td>"
                f'<td style="font-size:.76rem">{_html.escape(r["model"])}</td>'
                f'<td class="num">{r["input_tokens"]:,}</td>'
                f'<td class="num">{r["output_tokens"]:,}</td>'
                f'<td class="num {cost_css}">${cost:.5f}</td>'
                f'<td class="num">{r["latency_ms"]:.0f} ms</td>'
                f"</tr>"
            )

        # ── Assemble HTML ──────────────────────────────────────────
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Anthropic API Cost Dashboard</title>
  <style>{self._css()}</style>
</head>
<body>
<header>
  <h1>Anthropic API &mdash; Cost Dashboard</h1>
  <div class="subtitle">
    Generated {_html.escape(now)}
    &nbsp;&middot;&nbsp;
    Warning threshold: &gt;${self.t.daily_warning_usd:.2f}/day
    &nbsp;&middot;&nbsp;
    Critical threshold: &gt;${self.t.daily_critical_usd:.2f}/day
  </div>
</header>

<main>

  <section>
    <h2>Summary</h2>
    <div class="cards">{cards_html}</div>
  </section>

  <section>
    <h2>Daily Cost &mdash; Last 30 Days</h2>
    <div class="chart-box"><pre>{chart_html}</pre></div>
  </section>

  <div class="two-col">
    <div class="panel">
      <h2>Top 5 Features &mdash; Last 30 Days</h2>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>Feature</th><th>Calls</th><th>Total Cost</th><th>Avg Latency</th></tr>
          </thead>
          <tbody>{feat_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <h2>Top 5 Teams &mdash; Last 30 Days</h2>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>Team</th><th>Calls</th><th>Total Cost</th><th>Avg / Call</th></tr>
          </thead>
          <tbody>{team_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <section>
    <h2>Recent API Calls</h2>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th><th>Feature</th><th>Team</th><th>Model</th>
            <th>In Tokens</th><th>Out Tokens</th><th>Cost</th><th>Latency</th>
          </tr>
        </thead>
        <tbody>{call_rows}</tbody>
      </table>
    </div>
  </section>

</main>

<footer>
  Generated by cost_dashboard.py &nbsp;&middot;&nbsp; Anthropic API Cost Tracking
</footer>
</body>
</html>"""

    def save(self) -> Path:
        """
        Generate the HTML dashboard and write it to ``output_file``.

        Returns:
            The path to the written file.
        """
        content = self.generate()
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(content)
        return self.output_file


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _create_demo_db(db_path: Path) -> None:
    """Populate a fresh SQLite database with 35 days of synthetic API calls."""
    import random

    FEATURES = [
        ("doc-summarization",  "data-science",  "claude-opus-4",     2000, 8000,  300,  800),
        ("customer-chat",       "product",       "claude-haiku-4-5",   200,  800,  100,  300),
        ("code-review",         "engineering",   "claude-opus-4",     1000, 4000,  400, 1200),
        ("data-extraction",     "data-science",  "claude-haiku-4-5",   500, 2000,  100,  400),
        ("report-generation",   "analytics",     "claude-sonnet-4-5",  800, 3000,  500, 1500),
        ("semantic-search",     "search",        "claude-haiku-4-5",   100,  500,   50,  200),
        ("translation",         "product",       "claude-haiku-4-5",   300,  900,  100,  350),
        ("content-moderation",  "trust-safety",  "claude-sonnet-4-5",  400, 1200,  100,  300),
    ]

    MODEL_PRICING = {
        "claude-haiku-4-5":  (0.80e-6, 4.00e-6),
        "claude-sonnet-4-5": (3.00e-6, 15.00e-6),
        "claude-opus-4":     (15.00e-6, 75.00e-6),
    }

    # Spike days: offset from today (positive = days ago)
    SPIKES = {3: 3.5, 10: 2.3, 22: 4.1, 28: 1.8}

    rng   = random.Random(42)
    today = datetime.now(timezone.utc).date()

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name  TEXT    NOT NULL,
                user_id       TEXT    NOT NULL,
                team_name     TEXT    NOT NULL,
                model         TEXT    NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd      REAL    NOT NULL,
                latency_ms    REAL    NOT NULL,
                timestamp     TEXT    NOT NULL
            )
        """)

        rows: list[tuple] = []
        for offset in range(35, 0, -1):
            day        = today - timedelta(days=offset)
            multiplier = SPIKES.get(offset, 1.0)
            n_calls    = rng.randint(15, 40)

            for _ in range(n_calls):
                feat, team, model, in_lo, in_hi, out_lo, out_hi = rng.choice(FEATURES)
                in_tok  = int(rng.uniform(in_lo, in_hi) * multiplier)
                out_tok = int(rng.uniform(out_lo, out_hi) * multiplier)
                in_p, out_p = MODEL_PRICING[model]
                cost    = in_tok * in_p + out_tok * out_p
                lat_ms  = rng.uniform(400, 5000) * (0.5 if model == "claude-haiku-4-5" else 1.0)
                ts = datetime(
                    day.year, day.month, day.day,
                    rng.randint(0, 23), rng.randint(0, 59),
                    tzinfo=timezone.utc,
                ).isoformat()
                rows.append((
                    feat, f"user_{rng.randint(1, 20):03d}", team,
                    model, in_tok, out_tok, cost, lat_ms, ts,
                ))

        conn.executemany(
            "INSERT INTO api_calls "
            "(feature_name, user_id, team_name, model, input_tokens, output_tokens, "
            " cost_usd, latency_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        print(f"  Inserted {len(rows)} synthetic API call rows into {db_path}")


if __name__ == "__main__":
    DB_PATH  = Path("cost_tracking.db")
    HTML_OUT = Path("cost_dashboard.html")

    if not DB_PATH.exists():
        print(f"'{DB_PATH}' not found — generating demo data ...")
        _create_demo_db(DB_PATH)
    else:
        print(f"Using existing database: {DB_PATH}")

    thresholds = DashboardThresholds(
        daily_warning_usd  = 4.0,
        daily_critical_usd = 8.0,
        call_warning_usd   = 0.05,
        call_critical_usd  = 0.15,
    )

    dashboard = CostDashboard(
        db_path     = str(DB_PATH),
        output_file = str(HTML_OUT),
        thresholds  = thresholds,
    )

    out = dashboard.save()
    print(f"Dashboard written to: {out.resolve()}")
    print("Open it in any browser — no server needed.")
