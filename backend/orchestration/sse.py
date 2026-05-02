from __future__ import annotations

from collections.abc import AsyncIterator

from backend.schemas.events import SSEEvent


def encode_sse_event(event: SSEEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def encode_sse_stream(events: AsyncIterator[SSEEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield encode_sse_event(event)
