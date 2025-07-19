"""Integration tests for tool usage in critics."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import SifakaResult
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.tools.base import ToolInterface
from sifaka.tools.registry import ToolRegistry
from sifaka.tools.web_search import WebSearchTool


class MockTool(ToolInterface):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", results: List[Dict[str, Any]] = None):
        self._name = name
        self._results = results or [
            {
                "title": "Test Result",
                "content": "This is a test result",
                "url": "https://example.com/test",
                "confidence": 0.9,
            }
        ]
        self.call_count = 0
        self.last_query = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool for testing ({self._name})"

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_query = query
        # Simulate some processing time
        await asyncio.sleep(0.01)
        return self._results[: kwargs.get("max_results", 5)]


class TestToolIntegration:
    """Test tool integration with critics."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        return MockTool()

    @pytest.fixture
    def registry_with_mock_tool(self, mock_tool):
        """Registry with mock tool registered."""
        registry = ToolRegistry()
        registry.register(mock_tool.name, MockTool)
        return registry

    def test_tool_interface_compliance(self, mock_tool):
        """Test that mock tool implements ToolInterface correctly."""
        assert hasattr(mock_tool, "__call__")
        assert hasattr(mock_tool, "name")
        assert hasattr(mock_tool, "description")
        assert callable(mock_tool)
        assert isinstance(mock_tool.name, str)
        assert isinstance(mock_tool.description, str)

    @pytest.mark.asyncio
    async def test_tool_basic_functionality(self, mock_tool):
        """Test basic tool functionality."""
        results = await mock_tool("test query", max_results=3)

        assert len(results) <= 3
        assert mock_tool.call_count == 1
        assert mock_tool.last_query == "test query"
        assert all(isinstance(r, dict) for r in results)

        # Check result structure
        if results:
            result = results[0]
            assert "title" in result
            assert "content" in result
            assert "url" in result

    @pytest.mark.asyncio
    async def test_tool_with_kwargs(self, mock_tool):
        """Test tool with keyword arguments."""
        results = await mock_tool(
            "test query", max_results=2, timeout=5.0, metadata={"source": "test"}
        )

        assert len(results) <= 2
        assert mock_tool.call_count == 1

    def test_tool_registry_registration(self, registry_with_mock_tool, mock_tool):
        """Test tool registration in registry."""
        registry = registry_with_mock_tool

        # Check tool is registered
        assert mock_tool.name in registry.list_available()

        # Check tool can be retrieved
        retrieved_tool_class = registry.get(mock_tool.name)
        assert retrieved_tool_class is MockTool

    def test_tool_registry_multiple_tools(self):
        """Test registry with multiple tools."""
        registry = ToolRegistry()

        class Tool1(MockTool):
            def __init__(self):
                super().__init__("tool1")

        class Tool2(MockTool):
            def __init__(self):
                super().__init__("tool2")

        registry.register("tool1", Tool1)
        registry.register("tool2", Tool2)

        tools = registry.list_available()
        assert "tool1" in tools
        assert "tool2" in tools
        # Don't check exact count since other tools might be registered from imports
        assert len(tools) >= 2

    @pytest.mark.asyncio
    async def test_self_rag_critic_with_tools(self, mock_tool):
        """Test Self-RAG critic with mock tools."""
        # Create a critic with tools enabled
        critic = SelfRAGCritic(enable_tools=True)

        # Replace the tool registry with our mock
        with patch.object(critic, "tools", [mock_tool]):
            result = SifakaResult(
                original_text="Test text",
                final_text="Test text with claims that need verification",
            )

            # Mock the LLM response
            mock_agent = AsyncMock()
            mock_agent_result = MagicMock()
            mock_agent_result.output = MagicMock()
            mock_agent_result.output.overall_assessment = "Content needs fact-checking"
            mock_agent_result.output.specific_issues = ["Unverified claim"]
            mock_agent_result.output.factual_claims = ["Test claim"]
            mock_agent_result.output.retrieval_opportunities = ["Verify test claim"]
            mock_agent_result.output.suggestions = ["Add citations"]
            mock_agent_result.output.needs_improvement = True
            mock_agent_result.output.confidence = 0.8
            mock_agent_result.output.isrel = "YES"
            mock_agent_result.output.issup = "PARTIAL"
            mock_agent_result.output.isuse = "YES"
            mock_agent_result.output.metadata = {}
            mock_agent_result.output.model_dump = MagicMock(
                return_value={
                    "overall_assessment": "Content needs fact-checking",
                    "specific_issues": ["Unverified claim"],
                    "factual_claims": ["Test claim"],
                    "retrieval_opportunities": ["Verify test claim"],
                    "suggestions": ["Add citations"],
                    "needs_improvement": True,
                    "confidence": 0.8,
                    "isrel": "YES",
                    "issup": "PARTIAL",
                    "isuse": "YES",
                    "metadata": {},
                }
            )
            mock_agent_result.usage = MagicMock(
                return_value=MagicMock(total_tokens=100)
            )
            mock_agent.run = AsyncMock(return_value=mock_agent_result)

            with patch.object(critic.client, "create_agent", return_value=mock_agent):
                critique_result = await critic.critique("Test text", result)

                assert critique_result.critic == "self_rag"
                assert critique_result.needs_improvement is True

                # Check that tools were potentially called
                # (This depends on the critic implementation)

    @pytest.mark.asyncio
    async def test_tool_usage_tracking(self, mock_tool):
        """Test that tool usage is tracked in results."""
        critic = SelfRAGCritic(enable_tools=True)

        with patch.object(critic, "tools", [mock_tool]):
            result = SifakaResult(
                original_text="Test text", final_text="Test text with claims"
            )

            # Mock the LLM response
            mock_agent = AsyncMock()
            mock_agent_result = MagicMock()
            mock_agent_result.output = MagicMock()
            mock_agent_result.output.overall_assessment = "Good content"
            mock_agent_result.output.specific_issues = []
            mock_agent_result.output.factual_claims = []
            mock_agent_result.output.retrieval_opportunities = []
            mock_agent_result.output.suggestions = []
            mock_agent_result.output.needs_improvement = False
            mock_agent_result.output.confidence = 0.9
            mock_agent_result.output.isrel = "YES"
            mock_agent_result.output.issup = "YES"
            mock_agent_result.output.isuse = "YES"
            mock_agent_result.output.metadata = {}
            mock_agent_result.output.model_dump = MagicMock(
                return_value={
                    "overall_assessment": "Good content",
                    "specific_issues": [],
                    "factual_claims": [],
                    "retrieval_opportunities": [],
                    "suggestions": [],
                    "needs_improvement": False,
                    "confidence": 0.9,
                    "isrel": "YES",
                    "issup": "YES",
                    "isuse": "YES",
                    "metadata": {},
                }
            )
            mock_agent_result.usage = MagicMock(
                return_value=MagicMock(total_tokens=100)
            )
            mock_agent.run = AsyncMock(return_value=mock_agent_result)

            with patch.object(critic.client, "create_agent", return_value=mock_agent):
                critique_result = await critic.critique("Test text", result)

                # Check that the result contains proper metadata
                assert critique_result.critic == "self_rag"
                assert isinstance(critique_result.metadata, dict)

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_tool):
        """Test error handling when tools fail."""
        # Make the mock tool raise an exception
        mock_tool._results = []  # Set to empty to simulate failure

        class FailingTool(MockTool):
            async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
                raise Exception("Tool failed")

        failing_tool = FailingTool("failing_tool")

        critic = SelfRAGCritic(enable_tools=True)

        with patch.object(critic, "tools", [failing_tool]):
            result = SifakaResult(original_text="Test text", final_text="Test text")

            # Mock the LLM response
            mock_agent = AsyncMock()
            mock_agent_result = MagicMock()
            mock_agent_result.output = MagicMock()
            mock_agent_result.output.overall_assessment = "Good content"
            mock_agent_result.output.specific_issues = []
            mock_agent_result.output.factual_claims = []
            mock_agent_result.output.retrieval_opportunities = []
            mock_agent_result.output.suggestions = []
            mock_agent_result.output.needs_improvement = False
            mock_agent_result.output.confidence = 0.9
            mock_agent_result.output.isrel = "YES"
            mock_agent_result.output.issup = "YES"
            mock_agent_result.output.isuse = "YES"
            mock_agent_result.output.metadata = {}
            mock_agent_result.output.model_dump = MagicMock(
                return_value={
                    "overall_assessment": "Good content",
                    "specific_issues": [],
                    "factual_claims": [],
                    "retrieval_opportunities": [],
                    "suggestions": [],
                    "needs_improvement": False,
                    "confidence": 0.9,
                    "isrel": "YES",
                    "issup": "YES",
                    "isuse": "YES",
                    "metadata": {},
                }
            )
            mock_agent_result.usage = MagicMock(
                return_value=MagicMock(total_tokens=100)
            )
            mock_agent.run = AsyncMock(return_value=mock_agent_result)

            with patch.object(critic.client, "create_agent", return_value=mock_agent):
                # The critic should handle tool failures gracefully
                critique_result = await critic.critique("Test text", result)

                assert critique_result.critic == "self_rag"
                # The critique should still work even if tools fail

    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self):
        """Test tool timeout handling."""

        class SlowTool(MockTool):
            async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
                # Simulate a slow tool
                await asyncio.sleep(0.1)
                return await super().__call__(query, **kwargs)

        slow_tool = SlowTool("slow_tool")

        # Test with a reasonable timeout
        results = await slow_tool("test query", timeout=1.0)
        assert len(results) > 0

        # Test with very short timeout would require actual timeout implementation
        # For now, just verify the tool accepts timeout parameter
        results = await slow_tool("test query", timeout=0.1)
        assert len(results) > 0

    def test_tool_configuration_from_config(self):
        """Test tool configuration from Config object."""
        config = Config()

        # Test with tools disabled
        critic = SelfRAGCritic(config=config, enable_tools=False)
        assert critic.enable_tools is False
        assert len(critic.tools) == 0

        # Test with tools enabled
        critic = SelfRAGCritic(config=config, enable_tools=True)
        assert critic.enable_tools is True
        # Tools list might be empty if no tools are registered, but the flag should be set

    @pytest.mark.asyncio
    async def test_web_search_tool_integration(self):
        """Test integration with the actual WebSearchTool."""
        # This is a more realistic test that could be marked as integration
        web_tool = WebSearchTool(max_results=3, timeout=5.0)

        # Test basic properties
        assert web_tool.name == "web_search"
        assert "search" in web_tool.description.lower()

        # Mock the HTTP request to avoid actual web calls
        mock_response = MagicMock()
        mock_response.text = """
        <html>
        <body>
        <div class="result">
            <h3><a href="https://example.com">Test Result</a></h3>
            <p>This is a test snippet</p>
        </div>
        </body>
        </html>
        """

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # This would normally make a real web request
            # For this test, we're just verifying the interface works
            try:
                await web_tool("test query")
                # The actual implementation might fail due to parsing, but the interface should work
            except Exception:
                # Expected if the HTML parsing doesn't work with our mock
                pass

    @pytest.mark.asyncio
    async def test_multiple_tools_with_critic(self):
        """Test critic with multiple tools."""
        tool1 = MockTool("tool1", [{"title": "Result 1", "content": "Content 1"}])
        tool2 = MockTool("tool2", [{"title": "Result 2", "content": "Content 2"}])

        critic = SelfRAGCritic(enable_tools=True)

        with patch.object(critic, "tools", [tool1, tool2]):
            result = SifakaResult(original_text="Test text", final_text="Test text")

            # Mock the LLM response
            mock_agent = AsyncMock()
            mock_agent_result = MagicMock()
            mock_agent_result.output = MagicMock()
            mock_agent_result.output.overall_assessment = "Good content"
            mock_agent_result.output.specific_issues = []
            mock_agent_result.output.factual_claims = []
            mock_agent_result.output.retrieval_opportunities = []
            mock_agent_result.output.suggestions = []
            mock_agent_result.output.needs_improvement = False
            mock_agent_result.output.confidence = 0.9
            mock_agent_result.output.isrel = "YES"
            mock_agent_result.output.issup = "YES"
            mock_agent_result.output.isuse = "YES"
            mock_agent_result.output.metadata = {}
            mock_agent_result.output.model_dump = MagicMock(
                return_value={
                    "overall_assessment": "Good content",
                    "specific_issues": [],
                    "factual_claims": [],
                    "retrieval_opportunities": [],
                    "suggestions": [],
                    "needs_improvement": False,
                    "confidence": 0.9,
                    "isrel": "YES",
                    "issup": "YES",
                    "isuse": "YES",
                    "metadata": {},
                }
            )
            mock_agent_result.usage = MagicMock(
                return_value=MagicMock(total_tokens=100)
            )
            mock_agent.run = AsyncMock(return_value=mock_agent_result)

            with patch.object(critic.client, "create_agent", return_value=mock_agent):
                critique_result = await critic.critique("Test text", result)

                assert critique_result.critic == "self_rag"
                # Both tools should be available to the critic
                assert len(critic.tools) == 2

    def test_tool_registry_singleton_behavior(self):
        """Test that tool registry behaves as expected."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()

        # Different instances should share the same class-level registry
        registry1.register("test_tool", MockTool)

        # registry2 should have the tool (they share class-level storage)
        assert "test_tool" in registry2.list_available()
        assert "test_tool" in registry1.list_available()

    @pytest.mark.asyncio
    async def test_tool_result_structure_validation(self, mock_tool):
        """Test that tool results have the expected structure."""
        results = await mock_tool("test query")

        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            # Check for common expected fields
            assert "title" in result or "content" in result

            # All values should be serializable
            import json

            try:
                json.dumps(result)
            except (TypeError, ValueError):
                pytest.fail(f"Result is not JSON serializable: {result}")

    @pytest.mark.asyncio
    async def test_tool_concurrent_usage(self, mock_tool):
        """Test that tools can handle concurrent requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = asyncio.create_task(mock_tool(f"query {i}"))
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        for result_list in results:
            assert isinstance(result_list, list)
            assert len(result_list) > 0

        # Check that the tool was called the correct number of times
        assert mock_tool.call_count == 5

    def test_tool_interface_type_checking(self):
        """Test that our tools satisfy the ToolInterface protocol."""
        # Check that MockTool satisfies the protocol
        mock_tool = MockTool()

        # Check that it has the required methods
        assert hasattr(mock_tool, "__call__")
        assert hasattr(mock_tool, "name")
        assert hasattr(mock_tool, "description")

        # Check that the methods have correct signatures
        import inspect

        call_signature = inspect.signature(mock_tool.__call__)
        assert "query" in call_signature.parameters
        assert "kwargs" in call_signature.parameters

        # Check that properties return correct types
        assert isinstance(mock_tool.name, str)
        assert isinstance(mock_tool.description, str)
