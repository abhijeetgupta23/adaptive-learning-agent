from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from backend.persistence.models import Base

DEFAULT_DB_URL = "sqlite+aiosqlite:///./adaptive_learning.db"
IN_MEMORY_DB_URL = "sqlite+aiosqlite:///:memory:"


def create_sqlite_engine(url: str = DEFAULT_DB_URL) -> AsyncEngine:
    kwargs: dict = {"echo": False}
    if ":memory:" in url:
        # StaticPool keeps one connection so the in-memory DB survives across sessions.
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_async_engine(url, **kwargs)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
