"""Sifaka Tools - External retrieval tools for PydanticAI agents.

This module provides external retrieval capabilities that complement Sifaka's
built-in graph-based state management and storage system.

Key Features:
- Web search tools (DuckDuckGo, Tavily)
- External data source integration
- PydanticAI common tools wrapper
- Extensible tool registry system

Note: For internal thought/state retrieval, use Sifaka's built-in graph persistence
and storage.tools module instead of these external tools.

Example Usage:
    ```python
    from pydantic_ai import Agent
    from sifaka.tools import create_web_search_tools

    # Create agent with web search capabilities
    web_tools = create_web_search_tools(providers=["duckduckgo", "tavily"])

    # Combine with agent
    agent = Agent(
        "openai:gpt-4",
        tools=web_tools,
        system_prompt="You can search the web for current information."
    )
    ```
"""

# Import with error handling for development
try:
    from sifaka.tools.base import (
        SifakaToolError,
        ToolRegistry,
        discover_all_tools,
        discover_retrieval_tools,
        tool_registry,
    )
except ImportError as e:
    print(f"Warning: Could not import base tools: {e}")
    SifakaToolError = Exception
    ToolRegistry = None
    discover_all_tools = None
    discover_retrieval_tools = None
    tool_registry = None

try:
    from sifaka.tools.retrieval import (
        create_web_search_tools,
    )
except ImportError as e:
    print(f"Warning: Could not import retrieval tools: {e}")
    create_web_search_tools = None

try:
    from sifaka.tools.common import get_pydantic_ai_common_tools
except ImportError as e:
    print(f"Warning: Could not import common tools: {e}")
    get_pydantic_ai_common_tools = None

__all__ = [
    # Core
    "SifakaToolError",
    "ToolRegistry",
    "tool_registry",
    # Tool discovery
    "discover_all_tools",
    "discover_retrieval_tools",
    # External retrieval tools
    "create_web_search_tools",
    # Common tools
    "get_pydantic_ai_common_tools",
]
