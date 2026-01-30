"""Python REPL tool for code execution."""

import logging
import sys
import io
import contextlib
from typing import Any

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)


class PythonREPLTool(BaseTool):
    """
    Python REPL (Read-Eval-Print Loop) tool.
    
    Executes Python code in a sandboxed environment.
    WARNING: This is not fully sandboxed - use with caution in production!
    """
    
    name = "python_repl"
    description = (
        "Execute Python code. "
        "Input should be valid Python code. "
        "Returns the output or error message. "
        "Use for calculations, data processing, etc."
    )
    
    def __init__(self):
        """Initialize Python REPL with persistent environment."""
        self.globals = {"__builtins__": __builtins__}
        self.locals = {}
    
    async def run(self, input_text: str) -> str:
        """
        Execute Python code.
        
        Args:
            input_text: Python code to execute
            
        Returns:
            Output from code execution or error message
        """
        try:
            logger.debug(f"Executing Python code: {input_text[:100]}...")
            
            # Capture stdout
            stdout_capture = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_capture):
                # Execute code
                exec(input_text, self.globals, self.locals)
            
            # Get output
            output = stdout_capture.getvalue()
            
            if output:
                return output.strip()
            else:
                return "Code executed successfully (no output)"
                
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return f"Error: {type(e).__name__}: {str(e)}"

