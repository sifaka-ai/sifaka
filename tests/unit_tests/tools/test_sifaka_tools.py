"""Comprehensive unit tests for Sifaka tool infrastructure.

This module tests the base tool infrastructure:
- BaseSifakaTool: Abstract base class for all tools
- ToolMetadata: Tool metadata container
- ToolRegistry: Tool registration and management
- Error handling and configuration

Tests cover:
- Tool creation and configuration
- Tool registration and discovery
- Error scenarios and recovery
- Mock-based testing without external dependencies
"""

from typing import List

import pytest
from pydantic_ai.tools import Tool

from sifaka.tools.base import (
    BaseSifakaTool,
    SifakaToolError,
    ToolConfigurationError,
    ToolExecutionError,
    ToolMetadata,
    ToolRegistry,
)


class TestToolMetadata:
    """Test suite for ToolMetadata class."""

    def test_tool_metadata_creation_minimal(self):
        """Test creating ToolMetadata with minimal parameters."""
        metadata = ToolMetadata(name="TestTool", description="A test tool", category="testing")

        assert metadata.name == "TestTool"
        assert metadata.description == "A test tool"
        assert metadata.category == "testing"
        assert metadata.provider is None
        assert metadata.requires_auth is False
        assert metadata.rate_limited is False
        assert metadata.dependencies == []

    def test_tool_metadata_creation_full(self):
        """Test creating ToolMetadata with all parameters."""
        dependencies = ["dep1", "dep2"]

        metadata = ToolMetadata(
            name="FullTool",
            description="A full-featured tool",
            category="advanced",
            provider="TestProvider",
            requires_auth=True,
            rate_limited=True,
            dependencies=dependencies,
        )

        assert metadata.name == "FullTool"
        assert metadata.description == "A full-featured tool"
        assert metadata.category == "advanced"
        assert metadata.provider == "TestProvider"
        assert metadata.requires_auth is True
        assert metadata.rate_limited is True
        assert metadata.dependencies == dependencies

    def test_tool_metadata_post_init(self):
        """Test ToolMetadata post-initialization behavior."""
        # Test with None dependencies
        metadata = ToolMetadata(
            name="TestTool", description="Test", category="test", dependencies=None
        )

        assert metadata.dependencies == []


class TestSifakaToolError:
    """Test suite for SifakaToolError exception classes."""

    def test_sifaka_tool_error_creation_minimal(self):
        """Test creating SifakaToolError with minimal parameters."""
        error = SifakaToolError("Tool failed")

        assert str(error) == "Tool failed"
        assert error.tool_name is None
        assert error.context == {}

    def test_sifaka_tool_error_creation_full(self):
        """Test creating SifakaToolError with all parameters."""
        context = {"retry_count": 2, "timeout": 30}

        error = SifakaToolError(
            message="Connection failed", tool_name="WebSearchTool", context=context
        )

        assert str(error) == "Connection failed"
        assert error.tool_name == "WebSearchTool"
        assert error.context == context

    def test_tool_configuration_error(self):
        """Test ToolConfigurationError."""
        error = ToolConfigurationError("Invalid config")

        assert str(error) == "Invalid config"
        assert isinstance(error, SifakaToolError)

    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        error = ToolExecutionError("Execution failed")

        assert str(error) == "Execution failed"
        assert isinstance(error, SifakaToolError)


class MockSifakaTool(BaseSifakaTool):
    """Mock tool implementation for testing."""

    def __init__(self, name: str = "MockTool", should_fail: bool = False, **kwargs):
        super().__init__(
            name=name, description="A mock tool for testing", category="testing", **kwargs
        )
        self.should_fail = should_fail
        self.call_count = 0

    def create_pydantic_tools(self) -> List[Tool]:
        """Create mock PydanticAI tools."""

        def mock_function(query: str) -> str:
            """Mock tool function."""
            self.call_count += 1
            if self.should_fail:
                raise RuntimeError("Mock tool failure")
            return f"Mock result for: {query}"

        return [Tool(mock_function, name=self.metadata.name)]


class TestBaseSifakaTool:
    """Test suite for BaseSifakaTool abstract base class."""

    def test_base_sifaka_tool_creation(self):
        """Test creating a concrete BaseSifakaTool implementation."""
        tool = MockSifakaTool(name="TestTool")

        assert tool.metadata.name == "TestTool"
        assert tool.metadata.description == "A mock tool for testing"
        assert tool.metadata.category == "testing"
        assert tool.call_count == 0

    def test_tool_create_pydantic_tools_success(self):
        """Test successful creation of PydanticAI tools."""
        tool = MockSifakaTool(name="SuccessTool", should_fail=False)

        pydantic_tools = tool.create_pydantic_tools()

        assert len(pydantic_tools) == 1
        assert isinstance(pydantic_tools[0], Tool)
        assert pydantic_tools[0].name == "SuccessTool"

    def test_tool_create_pydantic_tools_execution(self):
        """Test execution of created PydanticAI tools."""
        tool = MockSifakaTool(name="ExecuteTool", should_fail=False)

        pydantic_tools = tool.create_pydantic_tools()
        tool_func = pydantic_tools[0].function

        result = tool_func("test query")

        assert result == "Mock result for: test query"
        assert tool.call_count == 1

    def test_tool_create_pydantic_tools_failure(self):
        """Test PydanticAI tool execution failure."""
        tool = MockSifakaTool(name="FailTool", should_fail=True)

        pydantic_tools = tool.create_pydantic_tools()
        tool_func = pydantic_tools[0].function

        with pytest.raises(RuntimeError, match="Mock tool failure"):
            tool_func("test query")

    def test_tool_validate_configuration_default(self):
        """Test default configuration validation."""
        tool = MockSifakaTool(name="ValidTool")

        # Should not raise any exception
        tool.validate_configuration()

    def test_tool_metadata_access(self):
        """Test accessing tool metadata."""
        tool = MockSifakaTool(
            name="MetadataTool", provider="TestProvider", requires_auth=True, rate_limited=True
        )

        assert tool.metadata.name == "MetadataTool"
        assert tool.metadata.provider == "TestProvider"
        assert tool.metadata.requires_auth is True
        assert tool.metadata.rate_limited is True


class TestToolRegistry:
    """Test suite for ToolRegistry class."""

    def test_tool_registry_creation(self):
        """Test creating a new ToolRegistry."""
        registry = ToolRegistry()

        assert registry._tools == {}
        assert registry._instances == {}

    def test_tool_registry_register(self):
        """Test registering a tool class."""
        registry = ToolRegistry()

        registry.register(MockSifakaTool, "TestMockTool")

        assert "TestMockTool" in registry._tools
        assert registry._tools["TestMockTool"] == MockSifakaTool

    def test_tool_registry_register_default_name(self):
        """Test registering a tool class with default name."""
        registry = ToolRegistry()

        registry.register(MockSifakaTool)

        assert "MockSifakaTool" in registry._tools
        assert registry._tools["MockSifakaTool"] == MockSifakaTool

    def test_tool_registry_create_tool_success(self):
        """Test successful tool creation."""
        registry = ToolRegistry()
        registry.register(MockSifakaTool, "TestTool")

        tool = registry.create_tool("TestTool", name="CreatedTool")

        assert isinstance(tool, MockSifakaTool)
        assert tool.metadata.name == "CreatedTool"
        assert "TestTool" in registry._instances

    def test_tool_registry_create_tool_not_registered(self):
        """Test creating a tool that's not registered."""
        registry = ToolRegistry()

        with pytest.raises(ToolConfigurationError, match="Tool 'NonExistent' not registered"):
            registry.create_tool("NonExistent")

    def test_tool_registry_get_tool(self):
        """Test getting an existing tool instance."""
        registry = ToolRegistry()
        registry.register(MockSifakaTool, "GetTool")

        # Create tool first
        created_tool = registry.create_tool("GetTool", name="GetToolInstance")

        # Get the same tool
        retrieved_tool = registry.get_tool("GetTool")

        assert retrieved_tool is created_tool
        assert retrieved_tool.metadata.name == "GetToolInstance"

    def test_tool_registry_get_nonexistent_tool(self):
        """Test getting a non-existent tool."""
        registry = ToolRegistry()

        result = registry.get_tool("NonExistent")

        assert result is None

    def test_tool_registry_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()
        registry.register(MockSifakaTool, "Tool1")
        registry.register(MockSifakaTool, "Tool2")

        tools = registry.list_tools()

        assert "Tool1" in tools
        assert "Tool2" in tools
        assert len(tools) == 2

    def test_tool_registry_list_categories(self):
        """Test listing tool categories."""
        registry = ToolRegistry()
        registry.register(MockSifakaTool, "Tool1")
        registry.create_tool("Tool1", name="Instance1", category="cat1")

        categories = registry.list_categories()

        assert "cat1" in categories

    def test_tool_registry_get_all_tools(self):
        """Test getting all tools as PydanticAI tools."""
        registry = ToolRegistry()
        registry.register(MockSifakaTool, "AllTool")
        registry.create_tool("AllTool", name="AllToolInstance")

        all_tools = registry.get_all_tools()

        assert len(all_tools) == 1
        assert isinstance(all_tools[0], Tool)

    def test_tool_registry_get_tools_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        registry.register(MockSifakaTool, "CatTool")
        registry.create_tool("CatTool", name="CatToolInstance", category="test_category")

        category_tools = registry.get_tools_by_category("test_category")

        assert len(category_tools) == 1
        assert isinstance(category_tools[0], Tool)
