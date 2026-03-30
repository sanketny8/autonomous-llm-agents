"""Tests for agents."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import BaseAgent, AgentResult, AgentStep
from src.agents.react_agent import ReActAgent
from src.tools.calculator import CalculatorTool


# -- AgentStep and AgentResult --

class TestAgentStep:
    def test_creation(self):
        step = AgentStep(
            iteration=1,
            thought="I should calculate",
            action="calculator",
            action_input="2+2",
            observation="4",
        )
        assert step.iteration == 1
        assert step.thought == "I should calculate"
        assert step.action == "calculator"
        assert step.observation == "4"
        assert step.timestamp == 0.0

    def test_custom_timestamp(self):
        step = AgentStep(
            iteration=1, thought="t", action="a",
            action_input="i", observation="o", timestamp=1.5
        )
        assert step.timestamp == 1.5


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(answer="done")
        assert result.answer == "done"
        assert result.steps == []
        assert result.sources == []
        assert result.cost == 0.0
        assert result.duration == 0.0
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        result = AgentResult(answer="failed", success=False, error="timeout")
        assert result.success is False
        assert result.error == "timeout"


# -- BaseAgent --

class TestBaseAgent:
    """Test BaseAgent via a concrete subclass."""

    def _make_agent(self, **kwargs):
        class ConcreteAgent(BaseAgent):
            async def run(self, task, **kw):
                return AgentResult(answer=task)
        return ConcreteAgent(**kwargs)

    def test_init_defaults(self):
        agent = self._make_agent(name="test")
        assert agent.name == "test"
        assert agent.llm_model == "gpt-4-turbo-preview"
        assert agent.temperature == 0.7
        assert agent.max_iterations == 10
        assert agent.tools == []
        assert agent.memory is None

    def test_register_tool(self):
        agent = self._make_agent(name="test")
        tool = CalculatorTool()
        agent.register_tool(tool)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "calculator"

    def test_register_tools(self):
        agent = self._make_agent(name="test")
        agent.register_tools([CalculatorTool(), CalculatorTool()])
        assert len(agent.tools) == 2

    def test_get_tool_found(self):
        agent = self._make_agent(name="test")
        agent.register_tool(CalculatorTool())
        assert agent.get_tool("calculator") is not None

    def test_get_tool_not_found(self):
        agent = self._make_agent(name="test")
        assert agent.get_tool("nonexistent") is None

    def test_set_memory(self):
        agent = self._make_agent(name="test")
        agent.set_memory({"key": "value"})
        assert agent.memory == {"key": "value"}

    @pytest.mark.asyncio
    async def test_plan_default(self):
        agent = self._make_agent(name="test")
        plan = await agent.plan("do something")
        assert plan == ["do something"]

    @pytest.mark.asyncio
    async def test_execute_step_default(self):
        agent = self._make_agent(name="test")
        result = await agent.execute_step("step1")
        assert result == {"step": "step1", "result": "executed"}

    @pytest.mark.asyncio
    async def test_reflect_default(self):
        agent = self._make_agent(name="test")
        assert await agent.reflect("anything") is True


# -- ReActAgent --

class TestReActAgent:
    """Test ReActAgent with mocked LLM calls."""

    def test_init(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(name="test", api_key="fake-key")
            assert agent.name == "test"
            assert agent.temperature == 0.0
            assert agent.max_iterations == 10

    def test_format_tools_empty(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            assert agent._format_tools() == "No tools available."

    def test_format_tools_with_tools(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            agent.register_tool(CalculatorTool())
            formatted = agent._format_tools()
            assert "calculator" in formatted

    def test_parse_action_final_answer(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            result = agent._parse_action(
                "Thought: I know the answer\nFinal Answer: 42"
            )
            assert result["type"] == "final_answer"
            assert result["content"] == "42"

    def test_parse_action_tool_call(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            result = agent._parse_action(
                "Thought: I need to calculate\nAction: calculator[2 + 2]"
            )
            assert result["type"] == "action"
            assert result["tool_name"] == "calculator"
            assert result["tool_input"] == "2 + 2"

    def test_parse_action_no_match_defaults_to_final(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            result = agent._parse_action("Just some random text")
            assert result["type"] == "final_answer"
            assert result["content"] == "Just some random text"

    def test_build_react_prompt_no_steps(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            prompt = agent._build_react_prompt("solve 2+2", [])
            assert "solve 2+2" in prompt
            assert "None yet" in prompt

    def test_build_react_prompt_with_steps(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            steps = [{
                "thought": "need calc",
                "action": "calculator",
                "action_input": "2+2",
                "observation": "4"
            }]
            prompt = agent._build_react_prompt("solve 2+2", steps)
            assert "calculator" in prompt
            assert "4" in prompt

    def test_extract_sources_with_urls(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            steps = [
                {"observation": "Found at https://example.com/page"},
                {"observation": "Also https://other.com and https://example.com/page"},
            ]
            sources = agent._extract_sources(steps)
            assert "https://example.com/page" in sources
            assert "https://other.com" in sources
            # Deduplication
            assert len([s for s in sources if s == "https://example.com/page"]) == 1

    def test_extract_sources_no_urls(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            steps = [{"observation": "The result is 42"}]
            sources = agent._extract_sources(steps)
            assert sources == []

    @pytest.mark.asyncio
    async def test_execute_action_tool_not_found(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            result = await agent._execute_action("nonexistent", "input")
            assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_action_with_tool(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            agent.register_tool(CalculatorTool())
            result = await agent._execute_action("calculator", "3 + 4")
            assert result == "7"

    @pytest.mark.asyncio
    async def test_run_immediate_final_answer(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")

            # Mock _think to return a final answer immediately
            agent._think = AsyncMock(
                return_value="Thought: I know this\nFinal Answer: The answer is 42"
            )

            result = await agent.run("What is the meaning of life?")
            assert result.success is True
            assert "42" in result.answer

    @pytest.mark.asyncio
    async def test_run_with_tool_then_answer(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key")
            agent.register_tool(CalculatorTool())

            # First call: use calculator, second call: final answer
            agent._think = AsyncMock(side_effect=[
                "Thought: I need to calculate\nAction: calculator[5 * 6]",
                "Thought: I have the result\nFinal Answer: 30",
            ])

            result = await agent.run("What is 5 times 6?")
            assert result.success is True
            assert "30" in result.answer
            assert len(result.steps) == 1
            assert result.steps[0]["action"] == "calculator"

    @pytest.mark.asyncio
    async def test_run_max_iterations(self):
        with patch("src.agents.react_agent.AsyncOpenAI"):
            agent = ReActAgent(api_key="fake-key", max_iterations=2)
            agent.register_tool(CalculatorTool())

            # Always use a tool, never give final answer
            agent._think = AsyncMock(
                return_value="Thought: keep going\nAction: calculator[1+1]"
            )

            result = await agent.run("loop forever")
            assert result.success is False
            assert result.error == "Max iterations reached"
