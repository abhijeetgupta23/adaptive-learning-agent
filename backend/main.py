from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # must run before backend.* imports so os.getenv sees .env values

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.agents.teaching.registry import build_teaching_registry  # noqa: E402
from backend.api import session_router  # noqa: E402
from backend.orchestration.anthropic_client import get_anthropic_client  # noqa: E402
from backend.orchestration.logging import configure_logging  # noqa: E402
from backend.orchestration.middleware import MiddlewareClient  # noqa: E402
from backend.persistence import (  # noqa: E402
    DEFAULT_DB_URL,
    create_sqlite_engine,
    init_db,
    make_session_factory,
)
from backend.retrieval import (  # noqa: E402
    build_web_search_tool,
    build_wikipedia_search_tool,
)


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

    if os.getenv("TAVILY_API_KEY"):
        app.state.search_tool = build_web_search_tool()
    else:
        app.state.search_tool = None

    # Wikipedia is a second grounding source. No API key required.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
