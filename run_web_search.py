"""
Entry point for the Web Search Specialist Agent.

Usage:
    python run_web_search.py
"""
from contextlib import asynccontextmanager
import asyncio
import os
import sys

import uvicorn
from dotenv import load_dotenv
from loguru import logger

from rag_system.agents.web_search import WebSearchSpecialistAgent
from rag_system.mcp_server.client import register_with_mcp
from rag_system.utils.config import load_config
from rag_system.utils.logging_config import setup_logging

load_dotenv()


def main():
    setup_logging("web_search_specialist")
    cfg = load_config()
    agents_cfg = cfg.get("agents", {})

    if not agents_cfg.get("web_search_enabled", True):
        logger.info("Web search specialist disabled in config.yaml (agents.web_search_enabled=false)")
        sys.exit(0)

    port = int(
        os.getenv(
            "PORT",
            os.getenv("WEB_SEARCH_PORT", str(agents_cfg.get("web_search_port", 8004))),
        )
    )

    mcp_host = os.getenv("MCP_SERVER_HOST", "localhost")
    if mcp_host == "0.0.0.0":
        mcp_host = "localhost"
    mcp_port = os.getenv("MCP_SERVER_PORT", str(cfg.get("mcp_server", {}).get("port", 8000)))
    mcp_url = os.getenv("MCP_SERVER_URL", f"http://{mcp_host}:{mcp_port}")
    endpoint_host = os.getenv("WEB_SEARCH_HOST", "localhost")

    agent = WebSearchSpecialistAgent(port=port)

    @asynccontextmanager
    async def lifespan(application):
        await asyncio.sleep(2)
        await register_with_mcp(agent, mcp_url=mcp_url, endpoint_host=endpoint_host)
        yield

    agent.app.router.lifespan_context = lifespan

    uvicorn.run(
        agent.app,
        host="0.0.0.0",
        port=agent.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
