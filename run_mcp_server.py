import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))

    uvicorn.run(
        "syfe_assignment.mcp_server.server:app",
        host=host,
        port=port,
        log_level="info"
    )
