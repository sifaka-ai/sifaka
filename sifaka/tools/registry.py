"""Tool registry for Sifaka tool management and discovery.

This module provides a simple, global registry for tool implementations.
Tools are registered by name and can be retrieved by critics that support
tool calling functionality.

## Design:

The registry uses a class-based singleton pattern with class methods,
making it easy to register and retrieve tools from anywhere in the codebase.

## Usage:

    >>> from sifaka.tools.registry import ToolRegistry
    >>> from sifaka.tools.duckduckgo import DuckDuckGoTool
    >>>
    >>> # Register a tool (typically done at import time)
    >>> ToolRegistry.register("duckduckgo", DuckDuckGoTool)
    >>>
    >>> # Retrieve and use a tool
    >>> tool_class = ToolRegistry.get("duckduckgo")
    >>> if tool_class:
    ...     tool = tool_class()
    ...     results = await tool("search query")

## Tool Discovery:

Tools are typically registered when their modules are imported:

    # In tools/my_tool.py
    from .registry import ToolRegistry

    class MyTool:
        # implementation
        pass

    # Auto-register on import
    ToolRegistry.register("my_tool", MyTool)

This allows critics to discover tools by importing the tools package.
"""

from typing import Dict, List, Optional, Type

from .base import ToolInterface


class ToolRegistry:
    """Global registry for tool implementations.

    Provides a centralized location for registering and discovering tools.
    Uses class methods to maintain a single global registry that can be
    accessed from anywhere in the application.

    Key features:
    - Simple registration by name
    - Type-safe tool retrieval
    - List available tools for debugging
    - No instance required (all class methods)

    Example:
        >>> # Register during module import
        >>> ToolRegistry.register("search", WebSearchTool)
        >>>
        >>> # Use in critics
        >>> tool_class = ToolRegistry.get("search")
        >>> if tool_class:
        ...     tool = tool_class()
        ...     results = await tool("query")
        >>>
        >>> # Debug available tools
        >>> print("Available:", ToolRegistry.list_available())

    Thread safety:
        Registry operations are not thread-safe. Registration should
        happen during module import time before concurrent access.
    """

    _tools: Dict[str, Type[ToolInterface]] = {}
    """Internal storage for registered tool classes.

    Maps tool names to their implementation classes.
    Tools are stored as classes, not instances, allowing
    critics to create fresh instances with custom parameters.
    """

    @classmethod
    def register(cls, name: str, tool_class: Type[ToolInterface]) -> None:
        """Register a tool implementation.

        Stores a tool class in the registry for later retrieval.
        Tool names should be unique across the application.

        Args:
            name: Unique identifier for the tool (e.g., "duckduckgo").
                Should match the tool's name property for consistency.
            tool_class: Class implementing ToolInterface protocol.
                Must have async __call__, name, and description properties.

        Example:
            >>> class MySearchTool:
            ...     async def __call__(self, query: str) -> List[Dict]:
            ...         return [{"result": "found"}]
            ...
            ...     @property
            ...     def name(self) -> str:
            ...         return "my_search"
            ...
            ...     @property
            ...     def description(self) -> str:
            ...         return "Custom search tool"
            >>>
            >>> ToolRegistry.register("my_search", MySearchTool)

        Note:
            Registration typically happens during module import.
            Duplicate names will overwrite previous registrations.
        """
        cls._tools[name] = tool_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[ToolInterface]]:
        """Get a tool class by name.

        Retrieves a previously registered tool class. Returns None if
        the tool is not found, allowing safe tool availability checking.

        Args:
            name: Tool name as registered

        Returns:
            Tool class if found, None if not registered

        Example:
            >>> # Safe tool retrieval
            >>> tool_class = ToolRegistry.get("duckduckgo")
            >>> if tool_class:
            ...     tool = tool_class()
            ...     results = await tool("search query")
            ... else:
            ...     print("DuckDuckGo tool not available")

        Note:
            Returns the class, not an instance. This allows critics
            to create instances with custom configuration.
        """
        return cls._tools.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered tool names.

        Returns a list of tool names that have been registered.
        Useful for debugging, configuration validation, and
        dynamic tool discovery.

        Returns:
            List of tool names in registration order

        Example:
            >>> available_tools = ToolRegistry.list_available()
            >>> print(f"Tools: {', '.join(available_tools)}")
            >>>
            >>> # Check if specific tool is available
            >>> if "duckduckgo" in available_tools:
            ...     # Use the tool
            ...     pass

        Note:
            Empty list if no tools have been registered.
            Tool names reflect the keys used during registration.
        """
        return list(cls._tools.keys())
