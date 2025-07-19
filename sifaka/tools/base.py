"""Base interfaces for Sifaka tools and tool storage.

This module defines the protocols that all tools must implement to be compatible
with Sifaka's critic system. It provides a clean interface between critics and
external tools while maintaining type safety.

## Tool System Overview:

Tools extend critics' capabilities by providing access to external resources:
- Web search for fact-checking
- Database queries for context
- API calls for real-time data
- File system access for documentation

## Design Principles:

1. **Protocol-Based**: Uses Python protocols for duck typing
2. **Async-First**: All operations are async for non-blocking execution
3. **Storage Agnostic**: Tools can cache results using any storage backend
4. **LLM-Compatible**: Results are structured for easy LLM consumption

## Usage:

    >>> from sifaka.tools.base import ToolInterface
    >>>
    >>> class MyTool:
    ...     async def __call__(self, query: str) -> List[Dict[str, Any]]:
    ...         return [{"result": "data"}]
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_tool"
    ...
    >>> # Tool automatically satisfies ToolInterface protocol

## Tool Integration:

Tools are discovered and used by critics that support tool calling:
- Critics check if tools are enabled in configuration
- Tools are called with search queries or context
- Results are integrated into critic feedback
- Tool usage is tracked in CritiqueResult metadata
"""

from typing import Any, Dict, List, Optional, Protocol, TypeVar

T = TypeVar("T", bound=Dict[str, Any])


class ToolInterface(Protocol):
    """Protocol for PydanticAI-compatible tools.

    Defines the interface that all Sifaka tools must implement. Tools are
    external services or resources that critics can use to enhance their
    analysis with real-time data, fact-checking, or context gathering.

    Key requirements:
    - Async callable interface for non-blocking execution
    - Structured return format for LLM compatibility
    - Descriptive metadata for tool selection

    Example implementation:
        >>> class WebSearchTool:
        ...     async def __call__(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        ...         results = await search_web(query)
        ...         return [{
        ...             "title": r.title,
        ...             "snippet": r.snippet,
        ...             "url": r.url
        ...         } for r in results]
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "web_search"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Search the web for factual information"

    Protocol compliance:
        Any class implementing these methods will automatically satisfy
        the ToolInterface protocol via Python's structural subtyping.
    """

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute tool query and return structured results.

        Args:
            query: The search query or input for the tool.
                Should be a clear, specific request.
            **kwargs: Tool-specific parameters. See ToolCallParams in
                core.type_defs for expected fields:
                - max_results: Maximum number of results to return
                - filters: Additional filtering criteria
                - timeout: Operation timeout in seconds
                - cache: Whether to cache results
                - metadata: Additional metadata

        Returns:
            List of dictionaries containing structured results.
            Each dict should have consistent keys across calls.
            Common patterns:
            - {"title": str, "content": str, "url": str}
            - {"result": str, "confidence": float, "source": str}

        Example:
            >>> tool = WebSearchTool()
            >>> results = await tool("Eiffel Tower height", max_results=3)
            >>> print(results[0]["title"])  # "Eiffel Tower - Wikipedia"
        """
        ...

    @property
    def name(self) -> str:
        """Tool name for identification and registration.

        Returns:
            Unique identifier for this tool. Used in:
            - Tool registry for lookup
            - Configuration settings
            - Logging and debugging
            - CritiqueResult metadata

        Should be:
        - Lowercase with underscores (e.g., "web_search")
        - Descriptive but concise
        - Unique across all tools
        """
        ...

    @property
    def description(self) -> str:
        """Tool description for LLM context and user interfaces.

        Returns:
            Human-readable description of what this tool does.
            Used by:
            - Critics to decide when to use this tool
            - Configuration UIs for tool selection
            - Documentation generation
            - Error messages and logs

        Should be:
        - Clear and concise (one sentence preferred)
        - Focused on the tool's primary purpose
        - Include any important limitations

        Examples:
        - "Search the web for current factual information"
        - "Query internal documentation database"
        - "Analyze code repositories for context"
        """
        ...


class StorageInterface(Protocol):
    """Protocol for tool storage backends.

    Defines the interface for caching and persisting tool results.
    Tools can use storage backends to cache expensive operations,
    implement rate limiting, or maintain session state.

    Key features:
    - Async operations for non-blocking access
    - Optional TTL support for automatic expiration
    - Simple key-value interface
    - Backend-agnostic (memory, Redis, database, etc.)

    Example usage:
        >>> storage = RedisStorage()
        >>>
        >>> # Cache search results
        >>> cache_key = f"search:{hash(query)}"
        >>> if await storage.exists(cache_key):
        ...     results = await storage.get(cache_key)
        ... else:
        ...     results = await expensive_search(query)
        ...     await storage.set(cache_key, results, ttl=3600)

    Common implementations:
    - In-memory storage for development
    - Redis for production caching
    - Database storage for persistence
    - File storage for debugging
    """

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key.

        Args:
            key: Storage key to look up

        Returns:
            Stored value if key exists, None if not found

        Note:
            Should handle serialization/deserialization automatically.
            Complex objects should be preserved across get/set cycles.
        """
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value with optional time-to-live.

        Args:
            key: Storage key (should be unique)
            value: Value to store (any serializable type)
            ttl: Time-to-live in seconds. None means no expiration.

        Example:
            >>> # Store with 1 hour expiration
            >>> await storage.set("temp_data", {"result": "value"}, ttl=3600)
            >>>
            >>> # Store permanently
            >>> await storage.set("permanent", "data")
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete value by key.

        Args:
            key: Storage key to delete

        Note:
            Should not raise error if key doesn't exist.
            Idempotent operation.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: Storage key to check

        Returns:
            True if key exists (even if value is None), False otherwise

        Note:
            Useful for cache hit/miss logic without fetching the value.
        """
        ...
