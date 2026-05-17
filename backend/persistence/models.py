from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LearnerStateRow(Base):
    __tablename__ = "learner_states"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class SessionRunRow(Base):
    """One row per session run — the audit/telemetry trail behind /ops/dashboard.

    Inserted with started_at + nulls at session start, updated with totals at
    session end (or with `error` filled if the run blew up). The /ops endpoint
    aggregates over this table for cost / latency / recent-error views.
    """
    __tablename__ = "session_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    sub_goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    units_passed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-agent JSON blobs so the dashboard can aggregate across the last 24h
    # without joining a separate call-level table. Small and append-only.
    cost_by_agent_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_by_agent_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
