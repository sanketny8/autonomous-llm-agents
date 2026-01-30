"""Planner agent for breaking down complex tasks."""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import os

from src.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Planner agent that breaks down complex tasks into steps.
    
    Uses LLM to create a plan, then can coordinate execution.
    """
    
    def __init__(
        self,
        name: str = "PlannerAgent",
        llm_model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        api_key: Optional[str] = None
    ):
        """
        Initialize planner agent.
        
        Args:
            name: Agent name
            llm_model: LLM model to use
            temperature: Sampling temperature (lower for more focused planning)
            api_key: OpenAI API key
        """
        super().__init__(name, llm_model, temperature, max_iterations=1)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def run(self, task: str, **kwargs) -> AgentResult:
        """
        Create a plan for the task.
        
        Args:
            task: Task to plan for
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with the plan
        """
        start_time = time.time()
        
        try:
            logger.info(f"Planning task: {task}")
            
            # Generate plan
            plan = await self.plan(task)
            
            # Format as answer
            answer = "Plan:\n" + "\n".join([
                f"{i+1}. {step}" for i, step in enumerate(plan)
            ])
            
            duration = time.time() - start_time
            
            return AgentResult(
                answer=answer,
                steps=[{
                    "iteration": 1,
                    "thought": "Created plan for task",
                    "action": "plan",
                    "plan": plan,
                    "timestamp": duration
                }],
                sources=[],
                duration=duration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in planner: {e}")
            duration = time.time() - start_time
            
            return AgentResult(
                answer=f"Error creating plan: {str(e)}",
                steps=[],
                sources=[],
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def plan(self, task: str) -> List[str]:
        """
        Create a detailed plan for the task.
        
        Args:
            task: Task to plan for
            
        Returns:
            List of steps
        """
        prompt = f"""You are a helpful planning assistant. Break down the following task into clear, actionable steps.

Task: {task}

Create a step-by-step plan. Each step should be:
1. Specific and actionable
2. Ordered logically
3. Self-contained

Format your response as a numbered list:
1. First step
2. Second step
3. Third step
...

Plan:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful planning assistant that breaks down complex tasks into clear steps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Parse numbered steps
            steps = []
            for line in plan_text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove number and dot
                    step = line.split('.', 1)[-1].strip()
                    if step:
                        steps.append(step)
            
            logger.info(f"Created plan with {len(steps)} steps")
            return steps if steps else [task]
            
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return [task]  # Fallback to original task

