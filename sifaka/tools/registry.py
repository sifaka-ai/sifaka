"""Tool registry for Sifaka."""

from typing import Dict, Type, Optional, List
from .base import ToolInterface


class ToolRegistry:
    """Registry for tool implementations."""
    
    _tools: Dict[str, Type[ToolInterface]] = {}
    
    @classmethod
    def register(cls, name: str, tool_class: Type[ToolInterface]) -> None:
        """Register a tool implementation."""
        cls._tools[name] = tool_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[ToolInterface]]:
        """Get a tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered tools."""
        return list(cls._tools.keys())