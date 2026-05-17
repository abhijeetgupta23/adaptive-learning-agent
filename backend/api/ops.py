"""/ops/* — self-hosted, dependency-free telemetry dashboard.

`GET /ops/dashboard`      → JSON: live count, recent sessions, rolling 24h
                            cost + p50/p99 latency per agent, recent errors.
`GET /ops/dashboard.html` → tiny single-file dark-theme HTML view that polls
                            the JSON endpoint every 30s.

The dashboard reads from the `session_runs` SQLite table (written by
`backend/api/session.py` at session start/end/error) and the in-process
`live_session_registry` updated by the same orchestrator.

# TODO: protect /ops in deploy — currently unauthenticated, local-first.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from backend.persistence import (
    list_finished_within,
    list_recent_completed,
    list_recent_errors,
    parse_cost_by_agent,
    parse_latency_by_agent,
)

router = APIRouter(prefix="/ops", tags=["ops"])


@dataclass
class _LiveRegistry:
    """In-process registry of sessions currently mid-stream. Module-level
    lifetime — entries are added at SSE-stream start and removed in its
    `finally` block. Survives across the test suite via singleton pattern."""
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


live_session_registry = _LiveRegistry()


async def register_live_session(
    session_id: str, *, user_id: str, started_at: datetime | None = None,
) -> None:
    async with live_session_registry.lock:
        live_session_registry.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "started_at": (started_at or datetime.now(UTC)).isoformat(),
        }


async def unregister_live_session(session_id: str) -> None:
    async with live_session_registry.lock:
        live_session_registry.sessions.pop(session_id, None)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, 0 <= pct <= 100. Empty → 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _row_to_recent(row: Any) -> dict[str, Any]:
    return {
        "session_id": row.session_id,
        "user_id": row.user_id,
        "domain": row.domain,
        "sub_goal": row.sub_goal,
        "unit_count": row.unit_count,
        "units_passed": row.units_passed,
        "total_cost_usd": row.total_cost_usd,
        "total_latency_s": row.total_latency_s,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "error": row.error,
    }


def _error_row_to_brief(row: Any) -> dict[str, Any]:
    # The orchestrator stores errors as "ExcType: message" — split so the
    # dashboard column reads cleanly; cap message length for table sanity.
    raw = (row.error or "").strip()
    if ":" in raw:
        exc_type, _, msg = raw.partition(":")
        agent = exc_type.strip() or "error"
        message = msg.strip()
    else:
        agent = "error"
        message = raw
    if len(message) > 200:
        message = message[:197] + "..."
    return {
        "session_id": row.session_id,
        "timestamp": (row.finished_at or row.started_at).isoformat(),
        "agent": agent,
        "message": message,
    }


async def build_dashboard_payload(session_factory) -> dict[str, Any]:
    """Pure function — gathers all dashboard data into a JSON-ready dict."""
    async with live_session_registry.lock:
        live = list(live_session_registry.sessions.values())

    async with session_factory() as db:
        recent = await list_recent_completed(session=db, limit=10)
        errors = await list_recent_errors(session=db, limit=10)
        last_24h = await list_finished_within(session=db, hours=24.0)

    # Aggregate cost-by-agent + per-call latencies across the 24h window.
    cost_by_agent: dict[str, float] = {}
    latencies_by_agent: dict[str, list[float]] = {}
    for row in last_24h:
        for agent, usd in parse_cost_by_agent(row).items():
            cost_by_agent[agent] = cost_by_agent.get(agent, 0.0) + float(usd)
        for agent, lats in parse_latency_by_agent(row).items():
            latencies_by_agent.setdefault(agent, []).extend(lats)

    latency_p50_p99 = {
        agent: {
            "p50_ms": round(_percentile(lats, 50.0), 1),
            "p99_ms": round(_percentile(lats, 99.0), 1),
            "count": len(lats),
        }
        for agent, lats in latencies_by_agent.items()
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "live_session_count": len(live),
        "live_sessions": live,
        "recent_sessions": [_row_to_recent(r) for r in recent],
        "cost_by_agent_24h": {
            k: round(v, 6) for k, v in cost_by_agent.items()
        },
        "latency_p50_p99_by_agent_24h": latency_p50_p99,
        "recent_errors": [_error_row_to_brief(r) for r in errors if r.error],
    }


@router.get("/dashboard")
async def get_dashboard(request: Request) -> dict[str, Any]:
    session_factory = request.app.state.session_factory
    return await build_dashboard_payload(session_factory)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>adaptive-learning-agent — ops</title>
  <style>
    body { background:#0d1117; color:#c9d1d9;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      margin: 0; padding: 24px; }
    h1 { color:#58a6ff; margin: 0 0 4px; font-size: 1.3em; }
    .sub { color:#8b949e; font-size:.85em; margin-bottom: 18px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px; }
    .card { background:#161b22; border:1px solid #30363d; border-radius:8px;
      padding:14px; }
    .card h2 { margin:0 0 10px; font-size:.78em; text-transform:uppercase;
      letter-spacing:1px; color:#8b949e; font-weight:600; }
    .big { font-size:2.4em; color:#58a6ff; line-height:1; }
    table { width:100%; border-collapse:collapse; font-size:.82em; }
    th, td { text-align:left; padding:6px 8px; border-bottom:1px solid #21262d;
      vertical-align: top; }
    th { color:#8b949e; font-weight:600; font-size:.72em;
      text-transform:uppercase; letter-spacing:.5px; }
    td.num { text-align:right; font-variant-numeric: tabular-nums; }
    .err { color:#f85149; }
    .pill { display:inline-block; padding:1px 6px; border-radius:10px;
      font-size:.72em; background:#21262d; color:#8b949e;
      border:1px solid #30363d; }
    .pill.live { color:#3fb950; border-color:#238636; }
    .meta { color:#8b949e; font-size:.72em; margin-top:6px; }
    code { color:#a5d6ff; }
  </style>
</head>
<body>
  <h1>adaptive-learning-agent — ops dashboard</h1>
  <div class="sub">
    self-hosted telemetry · auto-refresh 30s ·
    <span id="updated">loading…</span>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Live sessions</h2>
      <div class="big" id="live-count">—</div>
      <div class="meta" id="live-list"></div>
    </div>
    <div class="card">
      <h2>Cost by agent (last 24h)</h2>
      <table id="cost-table">
        <thead><tr><th>Agent</th><th class="num">USD</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Latency p50 / p99 by agent (24h)</h2>
      <table id="latency-table">
        <thead><tr><th>Agent</th><th class="num">p50 ms</th>
          <th class="num">p99 ms</th><th class="num">n</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card" style="grid-column:1/-1">
      <h2>Recent sessions</h2>
      <table id="recent-table">
        <thead><tr><th>Session</th><th>User</th><th>Domain</th>
          <th>Sub-goal</th><th class="num">Units</th>
          <th class="num">USD</th><th class="num">Latency s</th>
          <th>Finished</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card" style="grid-column:1/-1">
      <h2>Recent errors</h2>
      <table id="errors-table">
        <thead><tr><th>Timestamp</th><th>Session</th>
          <th>Agent</th><th>Message</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <script>
    const esc = s => String(s ?? "").replace(/[&<>]/g, c =>
      ({"&":"&amp;","<":"&lt;",">":"&gt;"})[c]);
    const fmt = (n, d=4) => (n==null) ? "—" : Number(n).toFixed(d);
    async function refresh() {
      try {
        const r = await fetch("/ops/dashboard");
        const d = await r.json();
        document.getElementById("updated").textContent =
          "updated " + d.generated_at;
        document.getElementById("live-count").textContent = d.live_session_count;
        document.getElementById("live-list").innerHTML =
          (d.live_sessions || []).map(s =>
            `<span class="pill live">${esc(s.session_id)}</span>`).join(" ")
          || '<span class="pill">none</span>';
        const ct = document.querySelector("#cost-table tbody");
        ct.innerHTML = Object.entries(d.cost_by_agent_24h || {})
          .sort((a,b) => b[1]-a[1])
          .map(([a,v]) =>
            `<tr><td>${esc(a)}</td><td class="num">$${fmt(v,4)}</td></tr>`)
          .join("") || '<tr><td colspan="2" class="meta">no calls</td></tr>';
        const lt = document.querySelector("#latency-table tbody");
        lt.innerHTML = Object.entries(d.latency_p50_p99_by_agent_24h || {})
          .map(([a,v]) =>
            `<tr><td>${esc(a)}</td><td class="num">${fmt(v.p50_ms,1)}</td>
             <td class="num">${fmt(v.p99_ms,1)}</td>
             <td class="num">${v.count}</td></tr>`)
          .join("") || '<tr><td colspan="4" class="meta">no data</td></tr>';
        const rt = document.querySelector("#recent-table tbody");
        rt.innerHTML = (d.recent_sessions || []).map(s =>
          `<tr><td><code>${esc(s.session_id)}</code></td>
           <td>${esc(s.user_id)}</td><td>${esc(s.domain)}</td>
           <td>${esc(s.sub_goal)}</td>
           <td class="num">${s.units_passed ?? "—"}/${s.unit_count ?? "—"}</td>
           <td class="num">$${fmt(s.total_cost_usd,4)}</td>
           <td class="num">${fmt(s.total_latency_s,2)}</td>
           <td class="meta">${esc(s.finished_at)}</td></tr>`)
          .join("") || '<tr><td colspan="8" class="meta">no sessions yet</td></tr>';
        const et = document.querySelector("#errors-table tbody");
        et.innerHTML = (d.recent_errors || []).map(e =>
          `<tr><td class="meta">${esc(e.timestamp)}</td>
           <td><code>${esc(e.session_id)}</code></td>
           <td>${esc(e.agent)}</td>
           <td class="err">${esc(e.message)}</td></tr>`)
          .join("") || '<tr><td colspan="4" class="meta">clean</td></tr>';
      } catch (e) {
        document.getElementById("updated").textContent =
          "refresh failed: " + e.message;
      }
    }
    refresh();
    setInterval(refresh, 30000);
  </script>
</body>
</html>
"""


@router.get("/dashboard.html", response_class=HTMLResponse)
async def get_dashboard_html() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)


# Re-export for tests that want to seed/clear the live registry directly.
__all__ = [
    "build_dashboard_payload",
    "live_session_registry",
    "register_live_session",
    "router",
    "unregister_live_session",
]
