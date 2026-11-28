import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from loguru import logger
import os
import sys

from rag_system.agents.specialist import SpecialistAgent
from rag_system.utils.logging_config import setup_logging

load_dotenv()

SPECIALIST_CONFIG = {
    "agent_id": os.getenv("SPECIALIST_ID", "specialist_climate"),
    "name": "Climate Document Specialist",
    "description": "Expert in climate change, environmental science, renewable energy, data pipelines, and ML model performance",
    "collection_name": os.getenv("SPECIALIST_COLLECTION", "climate_docs"),
    "port": int(os.getenv("PORT", os.getenv("SPECIALIST_PORT", "8001"))),
    "persist_dir": os.getenv("KB_PERSIST_DIR", "./chroma_db")
}

MCP_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
if MCP_HOST == "0.0.0.0":
    MCP_HOST = "localhost"
MCP_PORT = os.getenv("MCP_SERVER_PORT", "8000")
MCP_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_HOST}:{MCP_PORT}")

# hostname the MCP server should use to call back to this specialist
# in Docker this should be the container/service name (e.g. "specialist")
SPECIALIST_HOST = os.getenv("SPECIALIST_HOST", "localhost")


async def register_with_mcp(agent: SpecialistAgent, max_retries: int = 5):
    registration = agent.get_registration(endpoint_host=SPECIALIST_HOST)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{MCP_URL}/register",
                    json=registration.dict()
                )
                response.raise_for_status()
                logger.info("Registered with MCP server")
                return True

        except httpx.ConnectError:
            if attempt < max_retries - 1:
                logger.warning(f"MCP server not reachable. Retrying in 3s... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(3)
            else:
                logger.error("Failed to connect to MCP server")
                return False

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    return False


def main():
    setup_logging(f"specialist_{SPECIALIST_CONFIG['agent_id']}")
    os.makedirs("chroma_db", exist_ok=True)

    try:
        agent = SpecialistAgent(**SPECIALIST_CONFIG)
    except Exception as e:
        logger.error(f"Failed to create specialist: {e}")
        sys.exit(1)

    @asynccontextmanager
    async def lifespan(application):
        await asyncio.sleep(2)
        await register_with_mcp(agent)
        yield

    agent.app.router.lifespan_context = lifespan  # register after server is up

    uvicorn.run(
        agent.app,
        host="0.0.0.0",
        port=agent.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
