"""Wikipedia REST API client and Tool wrapper.

A second grounding source alongside Tavily. No API key required. Useful for
canonical/definitional content; less so for tutorials and recent posts.
"""
from __future__ import annotations

from typing import Any

import httpx

WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"


class WikipediaClientLike:
    async def get(self, url: str, **kwargs: Any) -> Any: ...  # pragma: no cover


async def search_wikipedia(
    query: str,
    *,
    client: WikipediaClientLike | None = None,
    limit: int = 5,
) -> list[dict[str, str]]:
    """Search English Wikipedia. Returns up to `limit` `{url, title, snippet}` rows."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=10.0)
    try:
        resp = await client.get(
            WIKIPEDIA_SEARCH_URL, params={"q": query, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()
        out: list[dict[str, str]] = []
        for page in data.get("pages", []):
            key = page.get("key") or page.get("title", "").replace(" ", "_")
            out.append({
                "url": f"https://en.wikipedia.org/wiki/{key}",
                "title": page.get("title", ""),
                "snippet": page.get("excerpt", "") or page.get("description", ""),
            })
        return out
    finally:
        if own_client:
            await client.aclose()
