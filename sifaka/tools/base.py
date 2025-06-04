"""Base classes and utilities for Sifaka tools.

This module provides the foundation for all Sifaka tools, including
error handling, tool registration, and common utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from pydantic_ai.tools import Tool
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SifakaToolError(Exception):
    """Base exception for all Sifaka tool errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.context = context or {}


class ToolConfigurationError(SifakaToolError):
    """Raised when a tool is misconfigured."""

    pass


class ToolExecutionError(SifakaToolError):
    """Raised when a tool fails during execution."""

    pass


@dataclass
class ToolMetadata:
    """Metadata for a Sifaka tool."""

    name: str
    description: str
    category: str
    provider: Optional[str] = None
    requires_auth: bool = False
    rate_limited: bool = False
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class BaseSifakaTool(ABC):
    """Base class for all Sifaka tools.

    This provides a common interface for tool creation, configuration,
    and metadata management.
    """

    def __init__(self, name: str, description: str, category: str, **kwargs):
        self.metadata = ToolMetadata(
            name=name, description=description, category=category, **kwargs
        )
        logger.debug(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    def create_pydantic_tools(self) -> List[Tool]:
        """Create PydanticAI Tool instances.

        Returns:
            List of Tool instances that can be used with PydanticAI agents
        """
        pass

    def validate_configuration(self) -> None:
        """Validate tool configuration.

        Raises:
            ToolConfigurationError: If configuration is invalid
        """
        pass


class ToolRegistry:
    """Registry for managing Sifaka tools.

    This provides a centralized way to register, discover, and create tools.
    """

    def __init__(self):
        self._tools: Dict[str, Type[BaseSifakaTool]] = {}
        self._instances: Dict[str, BaseSifakaTool] = {}
        logger.debug("Initialized ToolRegistry")

    def register(self, tool_class: Type[BaseSifakaTool], name: Optional[str] = None) -> None:
        """Register a tool class.

        Args:
            tool_class: The tool class to register
            name: Optional name override (defaults to class name)
        """
        tool_name = name or tool_class.__name__
        self._tools[tool_name] = tool_class
        logger.debug(f"Registered tool: {tool_name}")

    def create_tool(self, name: str, **kwargs) -> BaseSifakaTool:
        """Create a tool instance.

        Args:
            name: Name of the registered tool
            **kwargs: Configuration arguments for the tool

        Returns:
            Configured tool instance

        Raises:
            ToolConfigurationError: If tool is not registered or configuration fails
        """
        if name not in self._tools:
            raise ToolConfigurationError(f"Tool '{name}' not registered")

        tool_class = self._tools[name]
        try:
            instance = tool_class(**kwargs)
            instance.validate_configuration()
            self._instances[name] = instance
            return instance
        except Exception as e:
            raise ToolConfigurationError(f"Failed to create tool '{name}': {e}") from e

    def get_tool(self, name: str) -> Optional[BaseSifakaTool]:
        """Get an existing tool instance.

        Args:
            name: Name of the tool

        Returns:
            Tool instance if exists, None otherwise
        """
        return self._instances.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """List all tool categories."""
        categories = set()
        for instance in self._instances.values():
            categories.add(instance.metadata.category)
        return list(categories)

    def get_all_tools(self, categories: Optional[List[str]] = None) -> List[Tool]:
        """Get all available tools as PydanticAI Tool instances.

        Args:
            categories: Optional list of categories to filter by

        Returns:
            List of all available PydanticAI Tool instances
        """
        tools = []
        for instance in self._instances.values():
            if categories is None or instance.metadata.category in categories:
                try:
                    tools.extend(instance.create_pydantic_tools())
                except Exception as e:
                    logger.warning(f"Failed to create tools for {instance.metadata.name}: {e}")
        return tools

    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a specific category.

        Args:
            category: Tool category to filter by

        Returns:
            List of PydanticAI Tool instances in the category
        """
        return self.get_all_tools(categories=[category])


# Global tool registry instance
tool_registry = ToolRegistry()


def register_tool(name: Optional[str] = None):
    """Decorator to register a tool class.

    Args:
        name: Optional name override
    """

    def decorator(tool_class: Type[BaseSifakaTool]):
        tool_registry.register(tool_class, name)
        return tool_class

    return decorator


def create_tools_from_config(config: Dict[str, Dict[str, Any]]) -> List[Tool]:
    """Create tools from configuration dictionary.

    Args:
        config: Dictionary mapping tool names to their configuration

    Returns:
        List of PydanticAI Tool instances

    Example:
        ```python
        config = {
            "web_search": {"provider": "duckduckgo"},
            "redis_retrieval": {"connection_string": "redis://localhost"}
        }
        tools = create_tools_from_config(config)
        ```
    """
    tools = []
    for tool_name, tool_config in config.items():
        try:
            tool_instance = tool_registry.create_tool(tool_name, **tool_config)
            tools.extend(tool_instance.create_pydantic_tools())
        except Exception as e:
            logger.error(f"Failed to create tool '{tool_name}': {e}")
            # Continue with other tools rather than failing completely

    return tools


def discover_all_tools(categories: Optional[List[str]] = None) -> List[Tool]:
    """Discover and return all available tools from the global registry.

    This function provides automatic tool discovery for critics and agents
    that want access to all available tools without manual configuration.

    Args:
        categories: Optional list of tool categories to include (e.g., ["retrieval", "web_search"])
                   If None, includes all categories

    Returns:
        List of all available PydanticAI Tool instances

    Example:
        ```python
        # Get all available tools
        all_tools = discover_all_tools()

        # Get only retrieval tools
        retrieval_tools = discover_all_tools(categories=["retrieval"])

        # Use with a critic
        critic = SelfRAGCritic(retrieval_tools=discover_all_tools())
        ```
    """
    return tool_registry.get_all_tools(categories=categories)


def discover_retrieval_tools() -> List[Tool]:
    """Convenience function to discover all retrieval tools.

    Returns:
        List of all retrieval-category tools
    """
    return discover_all_tools(categories=["retrieval"])
