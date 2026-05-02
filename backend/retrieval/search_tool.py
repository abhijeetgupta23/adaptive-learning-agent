from __future__ import annotations

from typing import Any

from backend.orchestration.tools import Tool
from backend.retrieval.tavily import TavilyLike, search_web

WEB_SEARCH_TOOL_DESCRIPTION = (
    "Search the web for current information on a topic. "
    "Returns a list of results, each with url, title, and snippet. "
    "Use this to ground claims in retrievable sources."
)

WEB_SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query. Be specific — include the topic and context.",
        }
    },
    "required": ["query"],
}


def build_web_search_tool(
    *,
    client: TavilyLike | None = None,
    max_results: int = 5,
) -> Tool:
    async def handler(inp: dict[str, Any]) -> list[dict[str, str]]:
        return await search_web(inp["query"], client=client, max_results=max_results)

    return Tool(
        name="web_search",
        description=WEB_SEARCH_TOOL_DESCRIPTION,
        input_schema=WEB_SEARCH_INPUT_SCHEMA,
        handler=handler,
    )
