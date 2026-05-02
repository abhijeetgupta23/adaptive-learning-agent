from __future__ import annotations

import logging
import os
from typing import Any, Protocol

_logger = logging.getLogger(__name__)


class TavilyLike(Protocol):
    async def search(self, **kwargs: Any) -> dict[str, Any]: ...


def get_tavily_client(api_key: str | None = None) -> Any:
    from tavily import AsyncTavilyClient

    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set")
    return AsyncTavilyClient(api_key=key)


async def search_web(
    query: str,
    *,
    client: TavilyLike | None = None,
    max_results: int = 5,
    snippet_chars: int = 500,
) -> list[dict[str, str]]:
    """Search the web. Returns list of {url, title, snippet}."""
    client = client or get_tavily_client()
    response = await client.search(query=query, max_results=max_results)
    results = response.get("results", []) if isinstance(response, dict) else []
    out: list[dict[str, str]] = []
    for r in results:
        content = (r.get("content") or "")[:snippet_chars]
        out.append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "snippet": content,
        })
    _logger.info("tavily_search", extra={"query": query, "result_count": len(out)})
    return out
