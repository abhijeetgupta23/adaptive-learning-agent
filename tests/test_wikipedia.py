"""Tests for backend.retrieval.wikipedia + wikipedia_tool."""
from __future__ import annotations

from typing import Any

import pytest

from backend.retrieval.wikipedia import search_wikipedia
from backend.retrieval.wikipedia_tool import build_wikipedia_search_tool


class _FakeResp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeWikiClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.last_params: dict[str, Any] | None = None

    async def get(self, url: str, **kwargs: Any) -> _FakeResp:
        self.last_params = kwargs.get("params")
        return _FakeResp(self._payload)


@pytest.mark.asyncio
async def test_search_wikipedia_returns_structured_rows():
    payload = {
        "pages": [
            {"key": "Asyncio", "title": "Asyncio", "excerpt": "Python async I/O library"},
            {"key": "Event_loop", "title": "Event loop", "description": "scheduling primitive"},
        ],
    }
    client = _FakeWikiClient(payload)
    rows = await search_wikipedia("asyncio", client=client, limit=2)
    assert len(rows) == 2
    assert rows[0]["url"] == "https://en.wikipedia.org/wiki/Asyncio"
    assert rows[0]["title"] == "Asyncio"
    assert "Python async" in rows[0]["snippet"]
    assert rows[1]["snippet"] == "scheduling primitive"  # falls back to description


@pytest.mark.asyncio
async def test_wikipedia_tool_handler_uses_query_input():
    client = _FakeWikiClient({"pages": [
        {"key": "Sql", "title": "SQL", "excerpt": "Structured Query Language"},
    ]})
    tool = build_wikipedia_search_tool(client=client, max_results=3)
    result = await tool.handler({"query": "sql joins"})
    assert tool.name == "wikipedia_search"
    assert client.last_params == {"q": "sql joins", "limit": 3}
    assert result[0]["title"] == "SQL"


@pytest.mark.asyncio
async def test_search_wikipedia_handles_empty_pages():
    client = _FakeWikiClient({"pages": []})
    rows = await search_wikipedia("ghhhgh nonsense", client=client)
    assert rows == []
