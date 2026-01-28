"""Demo script for Research Agent."""

import asyncio
from src.agents.base import BaseAgent, AgentResult
from typing import Dict, Any, List


class ResearchAgent(BaseAgent):
    """Agent that researches topics using web search and summarization."""
    
    def __init__(self):
        super().__init__(name="ResearchAgent", llm_model="gpt-4-turbo")
    
    async def run(self, task: str, **kwargs) -> AgentResult:
        """Execute research task."""
        print(f"🔍 Starting research on: {task}")
        
        # Step 1: Plan the research
        steps = await self.plan(task)
        print(f"📋 Research plan created: {len(steps)} steps")
        
        # Step 2: Execute each step
        results = []
        for i, step in enumerate(steps, 1):
            print(f"⚙️  Executing step {i}/{len(steps)}: {step}")
            result = await self.execute_step(step)
            results.append(result)
        
        # Step 3: Synthesize findings
        print("📝 Synthesizing research findings...")
        answer = self._synthesize_results(results)
        
        return AgentResult(
            answer=answer,
            steps=results,
            sources=["https://example.com/source1", "https://example.com/source2"],
            cost=0.15,
            duration=45.2
        )
    
    def _synthesize_results(self, results: List[Dict[str, Any]]) -> str:
        """Combine research results into a coherent summary."""
        return """
        Based on extensive research, here are the key findings:
        
        1. LLM agents are autonomous systems that can plan and execute tasks
        2. They use tools to interact with external systems
        3. Memory systems help agents maintain context across interactions
        4. Multi-agent collaboration enables complex problem solving
        
        Recent developments include improved reasoning capabilities and 
        more sophisticated tool use patterns.
        """


async def main():
    """Run research agent demo."""
    agent = ResearchAgent()
    
    # Example research task
    task = "Research the latest developments in LLM agents and summarize key findings"
    
    result = await agent.run(task)
    
    print("\n" + "="*60)
    print("RESEARCH RESULTS")
    print("="*60)
    print(result.answer)
    print(f"\nTotal cost: ${result.cost:.2f}")
    print(f"Duration: {result.duration:.1f}s")
    print(f"Sources: {len(result.sources)}")


if __name__ == "__main__":
    asyncio.run(main())

