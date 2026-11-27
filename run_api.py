"""
REST API entry point.

Usage:
    python run_api.py

The Manager REST API will be available at http://localhost:8002
Interactive docs at http://localhost:8002/docs
"""
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("MANAGER_API_PORT", "8002"))
    uvicorn.run(
        "rag_system.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
