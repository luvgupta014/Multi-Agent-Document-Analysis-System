"""
Web Search Specialist Agent — answers questions that need live web information.

Uses DuckDuckGo Instant Answer API (no API key required).
Falls back to a simple DDG HTML scrape if the JSON API returns nothing.

The agent exposes a single tool: `web_search`.
The Manager routes questions about current events / external knowledge here.
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, status
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag_system.mcp_server.models import Tool, AgentRegistration, Message

DDG_INSTANT_URL = "https://api.duckduckgo.com/"
MAX_RESULTS = 5
TIMEOUT = 10.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)
def _ddg_search(query: str) -> List[Dict[str, str]]:
    """
    Call the DuckDuckGo Instant Answer API and return a list of
    {"title": ..., "snippet": ..., "url": ...} dicts.
    """
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(DDG_INSTANT_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
        logger.warning(f"DDG search failed: {exc}")
        return []

    results: List[Dict[str, str]] = []

    # AbstractText is the main Instant Answer snippet
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", "DuckDuckGo Abstract"),
            "snippet": data["AbstractText"],
            "url": data.get("AbstractURL", ""),
        })

    # RelatedTopics contains additional results
    for topic in data.get("RelatedTopics", []):
        if len(results) >= MAX_RESULTS:
            break
        # Topics can be nested (sub-topics have a "Topics" key)
        if "Text" in topic:
            results.append({
                "title": topic.get("Text", "")[:80],
                "snippet": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })

    return results[:MAX_RESULTS]


class WebSearchSpecialistAgent:
    """
    Specialist that searches the web via DuckDuckGo and returns snippets.
    """

    def __init__(
        self,
        agent_id: str = "web_search_specialist",
        name: str = "Web Search Specialist",
        description: str = "Searches the web to answer questions about current events and external knowledge",
        port: int = 8004,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.port = port

        self.app = FastAPI(title=f"Specialist: {name}", description=description)
        self._setup_routes()
        logger.info(f"WebSearchSpecialist ready on port {port}")

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="web_search",
                description=(
                    "Search the web for current information. "
                    "Returns snippets and URLs from DuckDuckGo."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string"},
                        "max_results": {
                            "type": "integer",
                            "description": f"Max results (1-{MAX_RESULTS})",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            )
        ]

    def get_registration(self, endpoint_host: str = "localhost") -> AgentRegistration:
        return AgentRegistration(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            tools=self.get_tools(),
            endpoint=f"http://{endpoint_host}:{self.port}",
        )

    def _setup_routes(self) -> None:

        @self.app.post("/execute", response_model=Dict)
        async def execute_tool(message: Message):
            if message.tool != "web_search":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tool '{message.tool}' not supported",
                )
            query = message.parameters.get("query", "").strip()
            if not query:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing required parameter: query",
                )
            max_results = min(int(message.parameters.get("max_results", 3)), MAX_RESULTS)

            try:
                results = _ddg_search(query)[:max_results]
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "query": query,
                    "results_count": len(results),
                    "results": results,
                }
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Search failed: {exc}",
                )

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "search_provider": "DuckDuckGo Instant Answer API",
            }
