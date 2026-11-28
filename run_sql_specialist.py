"""
Entry point for the SQL Specialist Agent.

Usage:
    python run_sql_specialist.py
"""
from contextlib import asynccontextmanager
import asyncio
import os

import uvicorn
from dotenv import load_dotenv

from rag_system.agents.sql_specialist import SQLSpecialistAgent
from rag_system.mcp_server.client import register_with_mcp
from rag_system.utils.config import load_config
from rag_system.utils.logging_config import setup_logging

load_dotenv()


def main():
    setup_logging("sql_specialist")
    cfg = load_config()
    agents_cfg = cfg.get("agents", {})

    db_path = os.getenv("SQL_DB_PATH", str(agents_cfg.get("sql_db_path", "data/sample.db")))
    port = int(os.getenv("PORT", os.getenv("SQL_SPECIALIST_PORT", str(agents_cfg.get("sql_port", 8003)))))

    mcp_host = os.getenv("MCP_SERVER_HOST", "localhost")
    if mcp_host == "0.0.0.0":
        mcp_host = "localhost"
    mcp_port = os.getenv("MCP_SERVER_PORT", str(cfg.get("mcp_server", {}).get("port", 8000)))
    mcp_url = os.getenv("MCP_SERVER_URL", f"http://{mcp_host}:{mcp_port}")
    endpoint_host = os.getenv("SQL_SPECIALIST_HOST", "localhost")

    agent = SQLSpecialistAgent(db_path=db_path, port=port)

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
