from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentRegistration(BaseModel):
    agent_id: str
    name: str
    description: str
    tools: List[Tool] = Field(default_factory=list)
    endpoint: str


class Message(BaseModel):
    sender: str
    recipient: str
    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MessageResponse(BaseModel):
    message_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
