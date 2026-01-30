"""Web search tool using DuckDuckGo."""

import logging
from typing import List, Dict, Any
from duckduckgo_search import DDGS

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo.
    
    No API key required!
    """
    
    name = "web_search"
    description = (
        "Search the web for information. "
        "Input should be a search query string. "
        "Returns search results with titles, URLs, and snippets."
    )
    
    def __init__(self, max_results: int = 5):
        """
        Initialize web search tool.
        
        Args:
            max_results: Maximum number of results to return
        """
        self.max_results = max_results
        self.ddgs = DDGS()
    
    async def run(self, input_text: str) -> str:
        """
        Execute web search.
        
        Args:
            input_text: Search query
            
        Returns:
            Formatted search results
        """
        try:
            logger.debug(f"Searching web for: {input_text}")
            
            # Perform search
            results = list(self.ddgs.text(
                input_text,
                max_results=self.max_results
            ))
            
            if not results:
                return "No search results found."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted = (
                    f"[{i}] {result['title']}\n"
                    f"URL: {result['link']}\n"
                    f"Snippet: {result['body']}\n"
                )
                formatted_results.append(formatted)
            
            output = "\n".join(formatted_results)
            logger.debug(f"Found {len(results)} results")
            
            return output
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error searching web: {str(e)}"

