from typing import Dict, List, Optional
from loguru import logger

from .models import AgentRegistration


class AgentRegistry:

    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        logger.info("Agent registry initialized")

    def register_agent(self, registration: AgentRegistration) -> bool:
        if registration.agent_id in self._agents:
            logger.warning(f"Agent {registration.agent_id} already registered")
            return False

        self._agents[registration.agent_id] = registration
        logger.info(f"Registered: {registration.name} ({registration.agent_id})")
        return True

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentRegistration]:
        return list(self._agents.values())

    def find_agents_with_tool(self, tool_name: str) -> List[AgentRegistration]:
        return [
            agent for agent in self._agents.values()
            if any(tool.name == tool_name for tool in agent.tools)
        ]

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"Unregistered: {agent_id}")
            return True
        return False

    def get_stats(self) -> Dict:
        return {
            "total_agents": len(self._agents),
            "agents": [
                {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "tools": [t.name for t in agent.tools]
                }
                for agent in self._agents.values()
            ]
        }
