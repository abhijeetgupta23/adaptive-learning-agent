from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # must run before backend.* imports so os.getenv sees .env values

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.agents.teaching.registry import build_teaching_registry  # noqa: E402
from backend.api import ops_router, session_router  # noqa: E402
from backend.orchestration.anthropic_client import get_anthropic_client  # noqa: E402
from backend.orchestration.logging import configure_logging  # noqa: E402
from backend.orchestration.middleware import MiddlewareClient  # noqa: E402
from backend.persistence import (  # noqa: E402
    DEFAULT_DB_URL,
    create_sqlite_engine,
    init_db,
    make_session_factory,
)
from backend.retrieval import build_wikipedia_search_tool  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    db_url = os.getenv("ADAPTIVE_LEARNING_DB_URL", DEFAULT_DB_URL)
    engine = create_sqlite_engine(db_url)
    await init_db(engine)
    app.state.db_engine = engine
    app.state.session_factory = make_session_factory(engine)

    if os.getenv("ANTHROPIC_API_KEY"):
        # Wrap the SDK client in retry + circuit-breaker middleware. Call sites
        # see the same `.messages.create(...)` API; transient failures recover
        # transparently and a sustained outage trips the breaker.
        app.state.anthropic = MiddlewareClient(get_anthropic_client())
    else:
        app.state.anthropic = None

    # Curriculum agent uses Anthropic's server-side web_search by default
    # (no separate vendor key needed). The `search_tool` slot is kept on
    # app.state for tests that inject a local fake.
    app.state.search_tool = None

    # Wikipedia stays as a second grounding source — no API key required.
    app.state.wikipedia_tool = build_wikipedia_search_tool()

    # Teaching skill registry: built once, holds frontmatter for all methods.
    # Real generators bind to the wrapped client; stubs are frontmatter-only.
    app.state.teaching_registry = build_teaching_registry(
        client=app.state.anthropic,
    )

    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(title="adaptive-learning-agent", version="0.1.0", lifespan=lifespan)

# Default CORS to wildcard so a fresh deploy works before the frontend URL is
# known. Set ADAPTIVE_LEARNING_FRONTEND_ORIGIN once the Vercel URL exists to
# narrow this down to that exact origin (the production-grade setup).
_explicit_origin = os.getenv("ADAPTIVE_LEARNING_FRONTEND_ORIGIN")
_cors_origins = (
    [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        _explicit_origin,
    ]
    if _explicit_origin
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Wildcard origin can't combine with credentials per the CORS spec, so we
    # only enable credentials when an explicit origin is set.
    allow_credentials=bool(_explicit_origin),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)
# Self-hosted ops dashboard (cost + latency + recent runs). Local-first;
# see ops.py for the # TODO: protect /ops in deploy note.
app.include_router(ops_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
