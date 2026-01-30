"""ReAct (Reasoning + Acting) agent implementation."""

import logging
import time
import json
import re
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import os

from src.agents.base import BaseAgent, AgentResult, AgentStep

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) agent.
    
    Uses a reasoning-action loop:
    1. Thought: Agent reasons about what to do
    2. Action: Agent decides on an action (tool call or final answer)
    3. Observation: Result of the action
    
    Based on "ReAct: Synergizing Reasoning and Acting in Language Models"
    (Yao et al., 2022)
    """
    
    def __init__(
        self,
        name: str = "ReActAgent",
        llm_model: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        max_iterations: int = 10,
        api_key: Optional[str] = None
    ):
        """
        Initialize ReAct agent.
        
        Args:
            name: Agent name
            llm_model: LLM model to use
            temperature: Sampling temperature (0.0 for deterministic)
            max_iterations: Maximum reasoning-action iterations
            api_key: OpenAI API key (uses env var if None)
        """
        super().__init__(name, llm_model, temperature, max_iterations)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.history: List[str] = []
    
    async def run(self, task: str, **kwargs) -> AgentResult:
        """
        Execute ReAct loop on a task.
        
        Args:
            task: Task to accomplish
            **kwargs: Additional arguments (verbose, etc.)
            
        Returns:
            AgentResult with answer and execution trace
        """
        start_time = time.time()
        steps: List[Dict[str, Any]] = []
        self.history = []
        
        verbose = kwargs.get("verbose", False)
        
        try:
            logger.info(f"Starting ReAct agent on task: {task}")
            
            for iteration in range(self.max_iterations):
                if verbose:
                    print(f"\n{'='*60}\nIteration {iteration + 1}/{self.max_iterations}\n{'='*60}")
                
                # Generate thought
                thought = await self._think(task, steps)
                
                if verbose:
                    print(f"\n💭 Thought: {thought}")
                
                # Parse action from thought
                action_dict = self._parse_action(thought)
                
                if action_dict["type"] == "final_answer":
                    # Agent is done
                    duration = time.time() - start_time
                    
                    if verbose:
                        print(f"\n✅ Final Answer: {action_dict['content']}")
                    
                    return AgentResult(
                        answer=action_dict["content"],
                        steps=steps,
                        sources=self._extract_sources(steps),
                        duration=duration,
                        success=True
                    )
                
                # Execute action (tool call)
                observation = await self._execute_action(
                    action_dict["tool_name"],
                    action_dict["tool_input"]
                )
                
                if verbose:
                    print(f"\n🔧 Action: {action_dict['tool_name']}({action_dict['tool_input'][:100]}...)")
                    print(f"\n👁️ Observation: {observation[:200]}...")
                
                # Record step
                step = {
                    "iteration": iteration + 1,
                    "thought": thought,
                    "action": action_dict["tool_name"],
                    "action_input": action_dict["tool_input"],
                    "observation": observation,
                    "timestamp": time.time() - start_time
                }
                steps.append(step)
                
                # Update history
                self.history.append(
                    f"Thought: {thought}\n"
                    f"Action: {action_dict['tool_name']}({action_dict['tool_input']})\n"
                    f"Observation: {observation}"
                )
            
            # Max iterations reached
            duration = time.time() - start_time
            logger.warning(f"Reached max iterations ({self.max_iterations})")
            
            return AgentResult(
                answer="I couldn't complete the task within the iteration limit. Here's what I found: " + 
                       (steps[-1]["observation"] if steps else "No progress made."),
                steps=steps,
                sources=self._extract_sources(steps),
                duration=duration,
                success=False,
                error="Max iterations reached"
            )
            
        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}")
            duration = time.time() - start_time
            
            return AgentResult(
                answer=f"An error occurred: {str(e)}",
                steps=steps,
                sources=[],
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def _think(self, task: str, steps: List[Dict[str, Any]]) -> str:
        """
        Generate reasoning step using LLM.
        
        Args:
            task: Original task
            steps: Previous steps
            
        Returns:
            Thought/reasoning from LLM
        """
        prompt = self._build_react_prompt(task, steps)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return f"Error thinking: {str(e)}"
    
    def _build_react_prompt(self, task: str, steps: List[Dict[str, Any]]) -> str:
        """Build the ReAct prompt with task and history."""
        # Tool descriptions
        tool_descriptions = self._format_tools()
        
        # Previous steps
        history_text = ""
        if steps:
            history_text = "\n\n".join([
                f"Thought: {step['thought']}\n"
                f"Action: {step['action']}[{step['action_input']}]\n"
                f"Observation: {step['observation']}"
                for step in steps
            ])
        
        prompt = f"""You are a helpful assistant that uses a reasoning and acting approach to solve tasks.

Available tools:
{tool_descriptions}

Task: {task}

Previous steps:
{history_text if history_text else "None yet - this is the first step."}

Think about what to do next. You can either:
1. Use a tool: Action: tool_name[tool_input]
2. Give final answer: Final Answer: your answer here

What should we do next?"""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for ReAct agent."""
        return """You are a ReAct agent. You think step-by-step and use tools to accomplish tasks.

Your responses should follow this format:

Thought: I need to [reasoning about what to do]
Action: tool_name[tool_input]

OR if you have enough information:

Thought: I now have enough information to answer
Final Answer: [your complete answer]

Be concise and specific. Always format actions as: tool_name[input]"""
    
    def _format_tools(self) -> str:
        """Format tool descriptions for prompt."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools:
            desc = f"- {tool.name}: {tool.description}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def _parse_action(self, thought: str) -> Dict[str, Any]:
        """
        Parse action from thought.
        
        Expected formats:
        - Action: tool_name[input]
        - Final Answer: answer
        
        Returns:
            Dict with type, tool_name, tool_input, or content
        """
        # Check for final answer
        if "Final Answer:" in thought:
            match = re.search(r"Final Answer:\s*(.+)", thought, re.DOTALL | re.IGNORECASE)
            if match:
                return {
                    "type": "final_answer",
                    "content": match.group(1).strip()
                }
        
        # Check for action
        match = re.search(r"Action:\s*(\w+)\[(.+?)\]", thought, re.DOTALL)
        if match:
            tool_name = match.group(1).strip()
            tool_input = match.group(2).strip()
            return {
                "type": "action",
                "tool_name": tool_name,
                "tool_input": tool_input
            }
        
        # No clear action - assume final answer
        return {
            "type": "final_answer",
            "content": thought
        }
    
    async def _execute_action(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool action.
        
        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool
            
        Returns:
            Observation/result from tool
        """
        # Find tool
        tool = self.get_tool(tool_name)
        
        if tool is None:
            available_tools = [t.name for t in self.tools]
            return f"Error: Tool '{tool_name}' not found. Available tools: {available_tools}"
        
        try:
            # Execute tool
            result = await tool.run(tool_input)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def _extract_sources(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Extract sources from steps."""
        sources = []
        for step in steps:
            if "observation" in step:
                # Simple heuristic: if observation looks like a URL, add it
                obs = step["observation"]
                urls = re.findall(r'https?://[^\s]+', obs)
                sources.extend(urls)
        return list(set(sources))  # Deduplicate

