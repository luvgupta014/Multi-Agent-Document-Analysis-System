"""
Shared MCP registration helper used by specialist entry-point scripts.
"""
from __future__ import annotations

import asyncio
import httpx
from loguru import logger


async def register_with_mcp(
    agent,
    mcp_url: str,
    endpoint_host: str = "localhost",
    max_retries: int = 5,
    retry_delay: float = 3.0,
) -> bool:
    """Register *agent* with the MCP server, retrying on transient connection errors."""
    registration = agent.get_registration(endpoint_host=endpoint_host)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{mcp_url.rstrip('/')}/register",
                    json=registration.model_dump(),
                )
                response.raise_for_status()
                logger.info(f"Registered '{agent.agent_id}' with MCP server at {mcp_url}")
                return True
        except httpx.ConnectError:
            if attempt < max_retries - 1:
                logger.warning(
                    f"MCP server not reachable. Retrying in {retry_delay:.0f}s... "
                    f"({attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to MCP server at {mcp_url}")
                return False
        except Exception as exc:
            logger.error(f"Registration error for '{agent.agent_id}': {exc}")
            return False

    return False
