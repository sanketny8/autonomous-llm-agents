"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AgentResult(BaseModel):
    """Agent execution result."""
    answer: str
    steps: List[Dict[str, Any]]
    sources: List[str] = []
    cost: float = 0.0
    duration: float = 0.0


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, llm_model: str = "gpt-4-turbo"):
        self.name = name
        self.llm_model = llm_model
        self.tools = []
        self.memory = None
    
    @abstractmethod
    async def run(self, task: str, **kwargs) -> AgentResult:
        """Execute the agent on a task."""
        pass
    
    def register_tool(self, tool: Any) -> None:
        """Register a tool with the agent."""
        self.tools.append(tool)
    
    async def plan(self, task: str) -> List[str]:
        """Plan the steps to accomplish a task."""
        return [task]  # Simple default implementation
    
    async def execute_step(self, step: str) -> Dict[str, Any]:
        """Execute a single step."""
        return {"step": step, "result": "executed"}
    
    async def reflect(self, result: Any) -> bool:
        """Reflect on whether the result is satisfactory."""
        return True

