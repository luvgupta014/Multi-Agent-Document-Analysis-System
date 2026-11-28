import uvicorn
from dotenv import load_dotenv
import os

from rag_system.utils.logging_config import setup_logging

load_dotenv()

if __name__ == "__main__":
    setup_logging("mcp_server")

    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("MCP_SERVER_PORT", "8000")))

    uvicorn.run(
        "rag_system.mcp_server.server:app",
        host=host,
        port=port,
        log_level="info"
    )
