"""Demo: Research agent using ReAct + tools."""

import asyncio
import os
from src.agents.react_agent import ReActAgent
from src.agents.planner_agent import PlannerAgent
from src.tools.web_search import WebSearchTool
from src.tools.calculator import CalculatorTool
from src.tools.python_repl import PythonREPLTool


async def demo_react_agent():
    """Demo ReAct agent with tools."""
    print("\n" + "="*60)
    print("ReAct Agent Demo")
    print("="*60)
    
    # Create agent
    agent = ReActAgent(
        name="ResearchAgent",
        temperature=0.0,
        max_iterations=10
    )
    
    # Register tools
    agent.register_tools([
        WebSearchTool(max_results=3),
        CalculatorTool(),
        PythonREPLTool()
    ])
    
    # Example tasks
    tasks = [
        "What is the current population of Tokyo? Calculate how many people that is per square kilometer if Tokyo is 2,194 km².",
        "Search for the latest news about GPT-4 and summarize the top 3 results.",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Task {i}: {task}")
        print("="*60)
        
        result = await agent.run(task, verbose=True)
        
        print(f"\n{'='*60}")
        print(f"Final Answer:")
        print(f"{'='*60}")
        print(result.answer)
        print(f"\nSteps taken: {len(result.steps)}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Success: {result.success}")


async def demo_planner_agent():
    """Demo planner agent."""
    print("\n" + "="*60)
    print("Planner Agent Demo")
    print("="*60)
    
    planner = PlannerAgent(name="Planner", temperature=0.2)
    
    task = "Build a machine learning model to predict house prices"
    
    print(f"\nTask: {task}\n")
    
    result = await planner.run(task)
    
    print(result.answer)
    print(f"\nDuration: {result.duration:.2f}s")


async def main():
    """Run all demos."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "AUTONOMOUS AGENT DEMOS" + " "*21 + "║")
    print("╚" + "="*58 + "╝")
    
    # Demo 1: Planner
    await demo_planner_agent()
    
    # Demo 2: ReAct agent
    await demo_react_agent()
    
    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
