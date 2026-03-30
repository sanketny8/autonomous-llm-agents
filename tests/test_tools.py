"""Tests for tools."""

import pytest
import asyncio
from src.tools.calculator import CalculatorTool
from src.tools.python_repl import PythonREPLTool
from src.tools.base import BaseTool


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    @pytest.fixture
    def calculator(self):
        return CalculatorTool()

    def test_name_and_description(self, calculator):
        assert calculator.name == "calculator"
        assert "math" in calculator.description.lower() or "calculate" in calculator.description.lower()

    @pytest.mark.asyncio
    async def test_addition(self, calculator):
        result = await calculator.run("2 + 2")
        assert result == "4"

    @pytest.mark.asyncio
    async def test_subtraction(self, calculator):
        result = await calculator.run("10 - 3")
        assert result == "7"

    @pytest.mark.asyncio
    async def test_multiplication(self, calculator):
        result = await calculator.run("6 * 7")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_division(self, calculator):
        result = await calculator.run("10 / 4")
        assert result == "2.5"

    @pytest.mark.asyncio
    async def test_exponentiation(self, calculator):
        result = await calculator.run("2 ** 10")
        assert result == "1024"

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        result = await calculator.run("(10 + 5) * 2")
        assert result == "30"

    @pytest.mark.asyncio
    async def test_negative_numbers(self, calculator):
        result = await calculator.run("-5 + 3")
        assert result == "-2"

    @pytest.mark.asyncio
    async def test_invalid_expression(self, calculator):
        result = await calculator.run("not_math")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator):
        result = await calculator.run("1 / 0")
        assert "Error" in result

    def test_str_representation(self, calculator):
        s = str(calculator)
        assert "calculator" in s


class TestPythonREPLTool:
    """Tests for PythonREPLTool."""

    @pytest.fixture
    def repl(self):
        return PythonREPLTool()

    def test_name_and_description(self, repl):
        assert repl.name == "python_repl"
        assert "python" in repl.description.lower()

    @pytest.mark.asyncio
    async def test_print_output(self, repl):
        result = await repl.run("print('hello world')")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_math_computation(self, repl):
        result = await repl.run("print(sum(range(1, 11)))")
        assert result == "55"

    @pytest.mark.asyncio
    async def test_no_output(self, repl):
        result = await repl.run("x = 42")
        assert "no output" in result.lower() or "success" in result.lower()

    @pytest.mark.asyncio
    async def test_persistent_state(self, repl):
        await repl.run("my_var = 123")
        result = await repl.run("print(my_var)")
        assert result == "123"

    @pytest.mark.asyncio
    async def test_syntax_error(self, repl):
        result = await repl.run("def bad(")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_runtime_error(self, repl):
        result = await repl.run("print(1/0)")
        assert "Error" in result

    def test_str_representation(self, repl):
        s = str(repl)
        assert "python_repl" in s


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTool()

    def test_subclass_must_implement_run(self):
        class IncompleteTool(BaseTool):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteTool()

    def test_subclass_with_run(self):
        class ConcreteTool(BaseTool):
            name = "concrete"
            description = "A concrete tool"

            async def run(self, input_text: str):
                return input_text

        tool = ConcreteTool()
        assert tool.name == "concrete"
        assert str(tool) == "concrete: A concrete tool"
