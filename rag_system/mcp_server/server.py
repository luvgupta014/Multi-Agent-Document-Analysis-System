from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict
import httpx
from loguru import logger

from .models import AgentRegistration, Message, MessageResponse
from .registry import AgentRegistry

# single shared registry instance - agents register on startup
registry = AgentRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MCP Server starting")
    yield
    logger.info("MCP Server shutting down")


app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol server for agent-to-agent communication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/register", response_model=Dict, status_code=status.HTTP_201_CREATED)
async def register_agent(registration: AgentRegistration):
    success = registry.register_agent(registration)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{registration.agent_id}' is already registered"
        )

    return {
        "status": "registered",
        "message": f"Agent '{registration.name}' successfully registered",
        "agent_id": registration.agent_id,
        "tools": [t.name for t in registration.tools]
    }


@app.delete("/unregister/{agent_id}", response_model=Dict)
async def unregister_agent(agent_id: str):
    success = registry.unregister_agent(agent_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found"
        )

    return {"status": "unregistered", "agent_id": agent_id}


@app.get("/agents", response_model=List[AgentRegistration])
async def list_agents():
    return registry.list_agents()


@app.get("/agents/{agent_id}", response_model=AgentRegistration)
async def get_agent(agent_id: str):
    agent = registry.get_agent(agent_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found"
        )

    return agent


@app.post("/send", response_model=MessageResponse)
async def send_message(message: Message):
    logger.info(f"Message: {message.sender} -> {message.recipient} [{message.tool}]")

    recipient_agent = registry.get_agent(message.recipient)

    if not recipient_agent:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=f"Recipient '{message.recipient}' is not registered"
        )

    has_tool = any(t.name == message.tool for t in recipient_agent.tools)

    if not has_tool:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=f"Agent '{message.recipient}' does not support tool '{message.tool}'"
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{recipient_agent.endpoint}/execute",
                json=message.dict()
            )
            response.raise_for_status()
            result = response.json()

            return MessageResponse(
                message_id=message.message_id,
                success=True,
                result=result
            )

    except httpx.TimeoutException:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=f"Timeout: {message.recipient} did not respond"
        )

    except httpx.HTTPError as e:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=f"Communication error: {str(e)}"
        )

    except Exception as e:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=f"Unexpected error: {str(e)}"
        )


@app.get("/mcp/v1/tools", response_model=List[Dict])
async def list_tools():
    agents = registry.list_agents()
    tools = []
    for agent in agents:
        for tool in agent.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "agent_id": agent.agent_id,
                "agent_endpoint": agent.endpoint
            })
    return tools


@app.post("/mcp/v1/tools/{tool_name}/invoke", response_model=MessageResponse)
async def invoke_tool(tool_name: str, message: Message):
    agents = registry.find_agents_with_tool(tool_name)
    if not agents:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=f"No agent provides tool '{tool_name}'"
        )

    target = agents[0]
    message.recipient = target.agent_id
    message.tool = tool_name

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{target.endpoint}/execute",
                json=message.dict()
            )
            response.raise_for_status()
            return MessageResponse(
                message_id=message.message_id,
                success=True,
                result=response.json()
            )
    except Exception as e:
        return MessageResponse(
            message_id=message.message_id,
            success=False,
            error=str(e)
        )


@app.get("/health", response_model=Dict)
async def health_check():
    stats = registry.get_stats()
    return {
        "status": "healthy",
        "service": "MCP Server",
        "version": "1.0.0",
        "registered_agents": stats["total_agents"],
        "agents": stats["agents"]
    }
