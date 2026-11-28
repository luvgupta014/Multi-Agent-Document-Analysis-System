from contextlib import asynccontextmanager
import asyncio
import os
import sys

import uvicorn
from dotenv import load_dotenv
from loguru import logger

from rag_system.agents.specialist import SpecialistAgent
from rag_system.mcp_server.client import register_with_mcp
from rag_system.utils.config import load_config
from rag_system.utils.logging_config import setup_logging

load_dotenv()

_cfg = load_config()
_spec_cfg = _cfg.get("specialist", {})
_kb_cfg = _cfg.get("knowledge_base", {})

SPECIALIST_CONFIG = {
    "agent_id": os.getenv("SPECIALIST_ID", _spec_cfg.get("agent_id", "specialist_climate")),
    "name": _spec_cfg.get("name", "Climate Document Specialist"),
    "description": _spec_cfg.get(
        "description",
        "Expert in climate change, environmental science, renewable energy, data pipelines, and ML model performance",
    ),
    "collection_name": os.getenv("SPECIALIST_COLLECTION", _spec_cfg.get("collection_name", "climate_docs")),
    "port": int(os.getenv("PORT", os.getenv("SPECIALIST_PORT", str(_spec_cfg.get("port", 8001))))),
    "persist_dir": os.getenv("KB_PERSIST_DIR", _kb_cfg.get("persist_dir", "./chroma_db")),
}

MCP_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
if MCP_HOST == "0.0.0.0":
    MCP_HOST = "localhost"
MCP_PORT = os.getenv("MCP_SERVER_PORT", str(_cfg.get("mcp_server", {}).get("port", 8000)))
MCP_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_HOST}:{MCP_PORT}")

# hostname the MCP server should use to call back to this specialist
SPECIALIST_HOST = os.getenv("SPECIALIST_HOST", "localhost")


def main():
    setup_logging(f"specialist_{SPECIALIST_CONFIG['agent_id']}")
    os.makedirs(SPECIALIST_CONFIG["persist_dir"], exist_ok=True)

    try:
        agent = SpecialistAgent(**SPECIALIST_CONFIG)
    except Exception as e:
        logger.error(f"Failed to create specialist: {e}")
        sys.exit(1)

    @asynccontextmanager
    async def lifespan(application):
        await asyncio.sleep(2)
        await register_with_mcp(agent, mcp_url=MCP_URL, endpoint_host=SPECIALIST_HOST)
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
