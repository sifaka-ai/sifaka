"""Web search tools for Sifaka.

This module provides web search capabilities using various providers
like DuckDuckGo and Tavily, integrated with PydanticAI's common tools.
"""

import os
from typing import List, Optional, Dict, Any, Literal

from pydantic_ai.tools import Tool
from sifaka.tools.base import BaseSifakaTool, ToolConfigurationError, register_tool
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Import PydanticAI common tools with fallback
try:
    from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    logger.warning("DuckDuckGo search tool not available. Install with: pip install 'pydantic-ai-slim[duckduckgo]'")

try:
    from pydantic_ai.common_tools.tavily import tavily_search_tool
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily search tool not available. Install with: pip install 'pydantic-ai-slim[tavily]'")


@register_tool("web_search")
class WebSearchTool(BaseSifakaTool):
    """Web search tool supporting multiple providers."""
    
    def __init__(
        self,
        providers: List[Literal["duckduckgo", "tavily"]] = None,
        tavily_api_key: Optional[str] = None,
        max_results: int = 5,
        **kwargs
    ):
        super().__init__(
            name="web_search",
            description="Search the web for information using various providers",
            category="retrieval",
            provider="multiple",
            requires_auth=True,  # Tavily requires auth
            rate_limited=True,
            **kwargs
        )
        
        self.providers = providers or ["duckduckgo"]
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        
        # Validate provider availability
        for provider in self.providers:
            if provider == "duckduckgo" and not DUCKDUCKGO_AVAILABLE:
                raise ToolConfigurationError(
                    f"DuckDuckGo provider requested but not available. "
                    f"Install with: pip install 'pydantic-ai-slim[duckduckgo]'"
                )
            elif provider == "tavily" and not TAVILY_AVAILABLE:
                raise ToolConfigurationError(
                    f"Tavily provider requested but not available. "
                    f"Install with: pip install 'pydantic-ai-slim[tavily]'"
                )
    
    def validate_configuration(self) -> None:
        """Validate web search configuration."""
        if "tavily" in self.providers and not self.tavily_api_key:
            raise ToolConfigurationError(
                "Tavily provider requires API key. Set TAVILY_API_KEY environment variable "
                "or pass tavily_api_key parameter."
            )
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create PydanticAI web search tools."""
        tools = []
        
        for provider in self.providers:
            if provider == "duckduckgo" and DUCKDUCKGO_AVAILABLE:
                tool = duckduckgo_search_tool(max_results=self.max_results)
                tools.append(tool)
                logger.debug("Added DuckDuckGo search tool")
            
            elif provider == "tavily" and TAVILY_AVAILABLE:
                tool = tavily_search_tool(
                    api_key=self.tavily_api_key,
                    max_results=self.max_results
                )
                tools.append(tool)
                logger.debug("Added Tavily search tool")
        
        return tools


@register_tool("duckduckgo_search")
class DuckDuckGoSearchTool(BaseSifakaTool):
    """DuckDuckGo-specific search tool."""
    
    def __init__(self, max_results: int = 5, **kwargs):
        super().__init__(
            name="duckduckgo_search",
            description="Search the web using DuckDuckGo",
            category="retrieval",
            provider="duckduckgo",
            requires_auth=False,
            rate_limited=True,
            **kwargs
        )
        self.max_results = max_results
    
    def validate_configuration(self) -> None:
        """Validate DuckDuckGo configuration."""
        if not DUCKDUCKGO_AVAILABLE:
            raise ToolConfigurationError(
                "DuckDuckGo search not available. "
                "Install with: pip install 'pydantic-ai-slim[duckduckgo]'"
            )
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create DuckDuckGo search tool."""
        if not DUCKDUCKGO_AVAILABLE:
            return []
        
        tool = duckduckgo_search_tool(max_results=self.max_results)
        return [tool]


@register_tool("tavily_search")
class TavilySearchTool(BaseSifakaTool):
    """Tavily-specific search tool."""
    
    def __init__(self, api_key: Optional[str] = None, max_results: int = 5, **kwargs):
        super().__init__(
            name="tavily_search",
            description="Search the web using Tavily",
            category="retrieval",
            provider="tavily",
            requires_auth=True,
            rate_limited=True,
            **kwargs
        )
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
    
    def validate_configuration(self) -> None:
        """Validate Tavily configuration."""
        if not TAVILY_AVAILABLE:
            raise ToolConfigurationError(
                "Tavily search not available. "
                "Install with: pip install 'pydantic-ai-slim[tavily]'"
            )
        
        if not self.api_key:
            raise ToolConfigurationError(
                "Tavily API key required. Set TAVILY_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create Tavily search tool."""
        if not TAVILY_AVAILABLE or not self.api_key:
            return []
        
        tool = tavily_search_tool(api_key=self.api_key, max_results=self.max_results)
        return [tool]


def create_web_search_tools(
    providers: List[Literal["duckduckgo", "tavily"]] = None,
    tavily_api_key: Optional[str] = None,
    max_results: int = 5,
    **kwargs
) -> List[Tool]:
    """Create web search tools for the specified providers.
    
    Args:
        providers: List of search providers to use
        tavily_api_key: API key for Tavily (optional, can use env var)
        max_results: Maximum number of search results per query
        **kwargs: Additional configuration options
        
    Returns:
        List of PydanticAI Tool instances
        
    Example:
        ```python
        # Use both providers
        tools = create_web_search_tools(["duckduckgo", "tavily"])
        
        # Use only DuckDuckGo
        tools = create_web_search_tools(["duckduckgo"])
        
        # Use Tavily with custom API key
        tools = create_web_search_tools(["tavily"], tavily_api_key="your-key")
        ```
    """
    tool = WebSearchTool(
        providers=providers,
        tavily_api_key=tavily_api_key,
        max_results=max_results,
        **kwargs
    )
    return tool.create_pydantic_tools()
