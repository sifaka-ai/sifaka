"""Tool infrastructure for extending Sifaka critics with external capabilities.

This package provides the foundation for integrating external tools and services
into Sifaka's critic system. Tools enable critics to access real-time information,
perform fact-checking, and enhance their analysis with external data sources.

## Package Overview:

- **base.py**: Core interfaces (ToolInterface, StorageInterface) that all tools must implement
- **registry.py**: Global registry for tool discovery and management
- **External tools**: Additional packages can provide specialized tools (web search, databases, APIs)

## Tool System Architecture:

    Critics → ToolRegistry → Tool Implementations → External Services

    1. Critics check if tools are enabled and available
    2. ToolRegistry provides access to registered tool classes
    3. Tool implementations handle external service integration
    4. Results are integrated into critic feedback

## Usage:

    >>> from sifaka.tools import ToolRegistry, ToolInterface
    >>>
    >>> # Check available tools
    >>> print("Available tools:", ToolRegistry.list_available())
    >>>
    >>> # Get a tool for use
    >>> search_tool_class = ToolRegistry.get("web_search")
    >>> if search_tool_class:
    ...     tool = search_tool_class()
    ...     results = await tool("search query")

## Creating Custom Tools:

Implement the ToolInterface protocol and register with ToolRegistry:

    >>> class MyCustomTool:
    ...     async def __call__(self, query: str) -> List[Dict[str, Any]]:
    ...         # Implementation here
    ...         return [{"result": "data"}]
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_tool"
    ...
    ...     @property
    ...     def description(self) -> str:
    ...         return "My custom tool description"
    >>>
    >>> # Register for use by critics
    >>> ToolRegistry.register("my_tool", MyCustomTool)

## Tool Integration in Critics:

Critics that support tools typically:
1. Check configuration for tool enablement
2. Retrieve tools from registry by name
3. Use tools to enhance analysis (fact-checking, context gathering)
4. Integrate tool results into critique feedback
5. Track tool usage in CritiqueResult metadata

See individual tool implementations for specific capabilities and usage patterns.
"""

# Import built-in tools to auto-register them
from . import arxiv, web_search, wikipedia
from .base import StorageInterface, ToolInterface
from .registry import ToolRegistry

__all__ = [
    "ToolInterface",
    "StorageInterface",
    "ToolRegistry",
    "web_search",
    "wikipedia",
    "arxiv",
]
