"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class AgentStep:
    """Single step in agent execution."""
    iteration: int
    thought: str
    action: str
    action_input: str
    observation: str
    timestamp: float = 0.0


class AgentResult(BaseModel):
    """Agent execution result."""
    answer: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    cost: float = 0.0
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Provides common functionality for LLM-based agents including:
    - Tool registration and management
    - Memory management
    - Execution tracking
    """
    
    def __init__(
        self,
        name: str,
        llm_model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_iterations: int = 10
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            llm_model: LLM model to use
            temperature: Sampling temperature
            max_iterations: Maximum iterations for agent loop
        """
        self.name = name
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.tools = []
        self.memory = None
    
    @abstractmethod
    async def run(self, task: str, **kwargs) -> AgentResult:
        """
        Execute the agent on a task.
        
        Args:
            task: Task description
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with answer and execution trace
        """
        pass
    
    def register_tool(self, tool: Any) -> None:
        """
        Register a tool with the agent.
        
        Args:
            tool: Tool instance (must have name, description, and run method)
        """
        self.tools.append(tool)
    
    def register_tools(self, tools: List[Any]) -> None:
        """Register multiple tools."""
        self.tools.extend(tools)
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get tool by name."""
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                return tool
        return None
    
    def set_memory(self, memory: Any) -> None:
        """Set agent memory."""
        self.memory = memory
    
    async def plan(self, task: str) -> List[str]:
        """
        Plan the steps to accomplish a task.
        
        Default implementation returns task as single step.
        Override in subclasses for more sophisticated planning.
        
        Args:
            task: Task to plan for
            
        Returns:
            List of planned steps
        """
        return [task]
    
    async def execute_step(self, step: str) -> Dict[str, Any]:
        """
        Execute a single step.
        
        Args:
            step: Step to execute
            
        Returns:
            Execution result
        """
        return {"step": step, "result": "executed"}
    
    async def reflect(self, result: Any) -> bool:
        """
        Reflect on whether the result is satisfactory.
        
        Args:
            result: Result to reflect on
            
        Returns:
            True if satisfactory, False otherwise
        """
        return True

