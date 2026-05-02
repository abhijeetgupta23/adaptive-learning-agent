"""Tool wrapper around Wikipedia search."""
from __future__ import annotations

from typing import Any

from backend.orchestration.tools import Tool
from backend.retrieval.wikipedia import WikipediaClientLike, search_wikipedia

WIKIPEDIA_TOOL_DESCRIPTION = (
    "Search English Wikipedia for canonical concept definitions and "
    "encyclopedic context. Best for foundational concepts, named entities, "
    "history, and definitions. Less useful for tutorials, blog posts, or "
    "recent changes — prefer `web_search` for those. "
    "Returns a list of results, each with url, title, and snippet."
)

WIKIPEDIA_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Concept or entity to look up on Wikipedia.",
        }
    },
    "required": ["query"],
}


def build_wikipedia_search_tool(
    *,
    client: WikipediaClientLike | None = None,
    max_results: int = 5,
) -> Tool:
    async def handler(inp: dict[str, Any]) -> list[dict[str, str]]:
        return await search_wikipedia(
            inp["query"], client=client, limit=max_results,
        )

    return Tool(
        name="wikipedia_search",
        description=WIKIPEDIA_TOOL_DESCRIPTION,
        input_schema=WIKIPEDIA_INPUT_SCHEMA,
        handler=handler,
    )
