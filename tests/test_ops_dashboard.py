"""Tests for the self-hosted /ops/dashboard endpoint.

We don't drive a full session here — the dashboard reads from the
`session_runs` SQLite table, so we seed rows directly and call the
endpoint against an app wired to an in-memory DB.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import ops_router
from backend.api.ops import (
    build_dashboard_payload,
    live_session_registry,
    register_live_session,
    unregister_live_session,
)
from backend.persistence import (
    IN_MEMORY_DB_URL,
    SessionRunRow,
    create_sqlite_engine,
    init_db,
    make_session_factory,
)


@pytest_asyncio.fixture
async def app_with_inmem_db():
    """Bare FastAPI app wired with only the ops router + in-memory DB."""
    engine = create_sqlite_engine(IN_MEMORY_DB_URL)
    await init_db(engine)
    session_factory = make_session_factory(engine)

    app = FastAPI()
    app.state.session_factory = session_factory
    app.include_router(ops_router)

    # Clear any leftover live-session state from a prior test.
    async with live_session_registry.lock:
        live_session_registry.sessions.clear()

    try:
        yield app, session_factory
    finally:
        async with live_session_registry.lock:
            live_session_registry.sessions.clear()
        await engine.dispose()


async def _seed_run(
    session_factory,
    *,
    session_id: str,
    user_id: str = "alice",
    domain: str = "databases",
    sub_goal: str = "understand SQL joins",
    unit_count: int = 3,
    units_passed: int = 3,
    cost_by_agent: dict[str, float] | None = None,
    latency_by_agent: dict[str, list[float]] | None = None,
    finished_offset_minutes: float = 1.0,
    error: str | None = None,
) -> None:
    cost_by_agent = cost_by_agent or {
        "intent": 0.001, "curriculum": 0.04, "game": 0.12,
    }
    latency_by_agent = latency_by_agent or {
        "intent": [120.0],
        "curriculum": [350.0, 410.0],
        "game": [800.0, 1100.0, 950.0, 5000.0],
    }
    now = datetime.now(UTC)
    started = now - timedelta(minutes=finished_offset_minutes + 1)
    finished = now - timedelta(minutes=finished_offset_minutes)
    row = SessionRunRow(
        session_id=session_id,
        user_id=user_id,
        domain=domain,
        sub_goal=sub_goal,
        unit_count=unit_count,
        units_passed=units_passed,
        total_cost_usd=sum(cost_by_agent.values()),
        total_latency_s=sum(sum(v) for v in latency_by_agent.values()) / 1000.0,
        cost_by_agent_json=json.dumps(cost_by_agent),
        latency_by_agent_json=json.dumps(latency_by_agent),
        started_at=started,
        finished_at=finished,
        error=error,
    )
    async with session_factory() as session:
        session.add(row)
        await session.commit()


@pytest.mark.asyncio
async def test_dashboard_empty(app_with_inmem_db) -> None:
    app, _ = app_with_inmem_db
    client = TestClient(app)
    resp = client.get("/ops/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["live_session_count"] == 0
    assert data["live_sessions"] == []
    assert data["recent_sessions"] == []
    assert data["cost_by_agent_24h"] == {}
    assert data["latency_p50_p99_by_agent_24h"] == {}
    assert data["recent_errors"] == []
    # generated_at is a valid ISO timestamp
    datetime.fromisoformat(data["generated_at"])


@pytest.mark.asyncio
async def test_dashboard_aggregates_recent_sessions(app_with_inmem_db) -> None:
    app, session_factory = app_with_inmem_db
    await _seed_run(
        session_factory, session_id="s1",
        cost_by_agent={"intent": 0.002, "game": 0.10},
        latency_by_agent={
            "intent": [100.0, 200.0],
            "game": [500.0, 900.0, 1500.0, 2500.0],
        },
        finished_offset_minutes=10,
    )
    await _seed_run(
        session_factory, session_id="s2",
        cost_by_agent={"intent": 0.001, "game": 0.05},
        latency_by_agent={
            "intent": [150.0],
            "game": [600.0, 1000.0],
        },
        finished_offset_minutes=5,
    )

    client = TestClient(app)
    data = client.get("/ops/dashboard").json()

    # Newest-first ordering on finished_at.
    assert [s["session_id"] for s in data["recent_sessions"]] == ["s2", "s1"]
    # 24h cost rollup sums per-agent across both runs.
    assert data["cost_by_agent_24h"]["intent"] == pytest.approx(0.003, abs=1e-6)
    assert data["cost_by_agent_24h"]["game"] == pytest.approx(0.15, abs=1e-6)
    # Latency aggregates: intent has 3 samples (100, 200, 150) → sorted
    # [100, 150, 200] → p50 = 150.
    intent_lat = data["latency_p50_p99_by_agent_24h"]["intent"]
    assert intent_lat["count"] == 3
    assert intent_lat["p50_ms"] == pytest.approx(150.0, abs=0.1)
    # game has 6 samples; p99 ≈ max (2500).
    game_lat = data["latency_p50_p99_by_agent_24h"]["game"]
    assert game_lat["count"] == 6
    assert game_lat["p99_ms"] >= 2000.0


@pytest.mark.asyncio
async def test_dashboard_caps_recent_at_10(app_with_inmem_db) -> None:
    app, session_factory = app_with_inmem_db
    for i in range(15):
        await _seed_run(
            session_factory, session_id=f"s{i:02d}",
            finished_offset_minutes=15 - i * 0.5,
        )
    client = TestClient(app)
    data = client.get("/ops/dashboard").json()
    assert len(data["recent_sessions"]) == 10


@pytest.mark.asyncio
async def test_dashboard_excludes_runs_older_than_24h(app_with_inmem_db) -> None:
    app, session_factory = app_with_inmem_db
    # Recent run inside the 24h window.
    await _seed_run(
        session_factory, session_id="recent",
        cost_by_agent={"intent": 0.01},
        finished_offset_minutes=60,
    )
    # Old run, finished 48h ago — should not contribute to 24h aggregates.
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(SessionRunRow(
            session_id="old",
            user_id="alice",
            domain="db",
            sub_goal="joins",
            unit_count=1, units_passed=1,
            total_cost_usd=999.0,
            total_latency_s=999.0,
            cost_by_agent_json=json.dumps({"intent": 999.0}),
            latency_by_agent_json=json.dumps({"intent": [99999.0]}),
            started_at=now - timedelta(hours=49),
            finished_at=now - timedelta(hours=48),
        ))
        await session.commit()

    client = TestClient(app)
    data = client.get("/ops/dashboard").json()
    # The old run still appears in recent_sessions (we show last 10 ever),
    # but its huge cost must NOT be in the 24h rollup.
    assert data["cost_by_agent_24h"]["intent"] == pytest.approx(0.01, abs=1e-6)


@pytest.mark.asyncio
async def test_dashboard_surfaces_errors(app_with_inmem_db) -> None:
    app, session_factory = app_with_inmem_db
    await _seed_run(
        session_factory, session_id="bad1",
        error="GameAgentError: PTC loop never converged",
        finished_offset_minutes=2,
    )
    await _seed_run(
        session_factory, session_id="ok1",
        finished_offset_minutes=1,
    )
    client = TestClient(app)
    data = client.get("/ops/dashboard").json()
    assert len(data["recent_errors"]) == 1
    err = data["recent_errors"][0]
    assert err["session_id"] == "bad1"
    assert err["agent"] == "GameAgentError"
    assert "PTC loop never converged" in err["message"]


@pytest.mark.asyncio
async def test_live_session_registry_round_trip(app_with_inmem_db) -> None:
    app, _ = app_with_inmem_db
    await register_live_session("live-1", user_id="alice")
    await register_live_session("live-2", user_id="bob")
    client = TestClient(app)
    data = client.get("/ops/dashboard").json()
    assert data["live_session_count"] == 2
    ids = {s["session_id"] for s in data["live_sessions"]}
    assert ids == {"live-1", "live-2"}

    await unregister_live_session("live-1")
    data = client.get("/ops/dashboard").json()
    assert data["live_session_count"] == 1


@pytest.mark.asyncio
async def test_dashboard_html_renders(app_with_inmem_db) -> None:
    app, _ = app_with_inmem_db
    client = TestClient(app)
    resp = client.get("/ops/dashboard.html")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # Sanity: contains the dark-theme palette and polls the JSON endpoint.
    assert "#0d1117" in body
    assert "#58a6ff" in body
    assert "/ops/dashboard" in body
    assert "setInterval" in body
    # Section headers for each card.
    for header in [
        "Live sessions",
        "Cost by agent",
        "Latency p50",
        "Recent sessions",
        "Recent errors",
    ]:
        assert header in body


@pytest.mark.asyncio
async def test_build_dashboard_payload_pure(app_with_inmem_db) -> None:
    """The pure builder works without an HTTP layer — useful for callers
    that want to embed the payload elsewhere (eg the writeup script)."""
    _, session_factory = app_with_inmem_db
    await _seed_run(
        session_factory, session_id="p1",
        cost_by_agent={"curriculum": 0.03},
        latency_by_agent={"curriculum": [400.0]},
    )
    payload = await build_dashboard_payload(session_factory)
    assert payload["cost_by_agent_24h"] == {"curriculum": 0.03}
    assert payload["latency_p50_p99_by_agent_24h"]["curriculum"]["count"] == 1
