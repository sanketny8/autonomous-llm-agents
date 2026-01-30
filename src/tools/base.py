"""Base tool class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base class for agent tools.
    
    All tools must implement:
    - name: Tool name for agent to reference
    - description: What the tool does
    - run: Execute the tool
    """
    
    name: str = "base_tool"
    description: str = "Base tool description"
    
    @abstractmethod
    async def run(self, input_text: str) -> Any:
        """
        Execute the tool.
        
        Args:
            input_text: Input for the tool
            
        Returns:
            Tool output (any type)
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

