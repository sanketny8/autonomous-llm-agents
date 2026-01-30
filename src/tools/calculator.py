"""Simple calculator tool."""

import logging
import ast
import operator
from typing import Union

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """
    Simple calculator tool.
    
    Evaluates mathematical expressions safely.
    """
    
    name = "calculator"
    description = (
        "Calculate mathematical expressions. "
        "Input should be a valid math expression like '2 + 2' or '(10 * 5) / 2'. "
        "Supports +, -, *, /, **, (), and numbers."
    )
    
    # Allowed operations
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    async def run(self, input_text: str) -> str:
        """
        Evaluate mathematical expression.
        
        Args:
            input_text: Math expression
            
        Returns:
            Result as string
        """
        try:
            logger.debug(f"Calculating: {input_text}")
            
            # Parse expression
            node = ast.parse(input_text, mode='eval')
            
            # Evaluate
            result = self._eval_node(node.body)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error in calculator: {e}")
            return f"Error: Could not calculate '{input_text}'. {str(e)}"
    
    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """
        Recursively evaluate AST node.
        
        Only allows safe operations (no function calls, etc.)
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS[type(node.op)]
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS[type(node.op)]
            return op(operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")

