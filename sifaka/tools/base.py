"""Base interfaces for Sifaka tools."""

from typing import Protocol, List, Dict, Any, Optional


class ToolInterface(Protocol):
    """Protocol for PydanticAI-compatible tools."""

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute tool query and return results."""
        ...

    @property
    def name(self) -> str:
        """Tool name for identification."""
        ...

    @property
    def description(self) -> str:
        """Tool description for LLM context."""
        ...


class StorageInterface(Protocol):
    """Protocol for tool storage backends."""

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete value by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
