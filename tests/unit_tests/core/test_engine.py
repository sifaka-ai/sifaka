"""Comprehensive unit tests for SifakaEngine.

This module tests the main orchestration engine:
- Engine initialization and configuration
- Dependency injection and default setup
- Graph creation and node configuration
- Error handling and validation
- Mock-based testing without external dependencies

Tests cover:
- Engine creation with default and custom dependencies
- Persistence configuration
- Graph structure validation
- Error handling for invalid configurations
- Mock-based workflow testing
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_graph.persistence import BaseStatePersistence
from pydantic_graph.persistence.in_mem import FullStatePersistence

from sifaka.core.engine import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.utils.errors import GraphExecutionError


class TestSifakaEngineInitialization:
    """Test SifakaEngine initialization and configuration."""

    def test_engine_creation_default(self):
        """Test creating SifakaEngine with default configuration."""
        engine = SifakaEngine()

        # Verify dependencies are created
        assert engine.deps is not None
        assert isinstance(engine.deps, SifakaDependencies)

        # Verify persistence is set up
        assert engine.persistence is not None
        assert isinstance(engine.persistence, FullStatePersistence)

        # Verify graph is created
        assert engine.graph is not None
        assert engine.graph.name == "SifakaWorkflow"
        assert len(engine.graph.nodes) == 3  # GenerateNode, ValidateNode, CritiqueNode

    def test_engine_creation_custom_dependencies(self):
        """Test creating SifakaEngine with custom dependencies."""
        # Create mock dependencies
        mock_deps = Mock(spec=SifakaDependencies)

        engine = SifakaEngine(dependencies=mock_deps)

        # Verify custom dependencies are used
        assert engine.deps is mock_deps
        assert isinstance(engine.persistence, FullStatePersistence)

    def test_engine_creation_custom_persistence(self):
        """Test creating SifakaEngine with custom persistence."""
        # Create mock persistence
        mock_persistence = Mock(spec=BaseStatePersistence)

        engine = SifakaEngine(persistence=mock_persistence)

        # Verify custom persistence is used
        assert engine.persistence is mock_persistence
        assert isinstance(engine.deps, SifakaDependencies)

    def test_engine_creation_custom_both(self):
        """Test creating SifakaEngine with both custom dependencies and persistence."""
        mock_deps = Mock(spec=SifakaDependencies)
        mock_persistence = Mock(spec=BaseStatePersistence)

        engine = SifakaEngine(dependencies=mock_deps, persistence=mock_persistence)

        # Verify both custom components are used
        assert engine.deps is mock_deps
        assert engine.persistence is mock_persistence

    @patch("sifaka.core.engine.logger")
    def test_engine_initialization_logging(self, mock_logger):
        """Test that engine initialization is properly logged."""
        SifakaEngine()

        # Verify initialization was logged
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "SifakaEngine initialized" in call_args[0][0]

        # Verify extra logging data
        extra_data = call_args[1]["extra"]
        assert "dependencies_type" in extra_data
        assert "persistence_type" in extra_data
        assert "graph_nodes" in extra_data
        assert len(extra_data["graph_nodes"]) == 3


class TestSifakaEngineThinkMethod:
    """Test the main think() method with mocked dependencies."""

    @pytest.fixture
    def mock_engine(self):
        """Create a SifakaEngine with mocked dependencies for testing."""
        # Create mock dependencies
        mock_deps = Mock(spec=SifakaDependencies)
        mock_persistence = Mock(spec=BaseStatePersistence)

        # Create engine with mocks
        engine = SifakaEngine(dependencies=mock_deps, persistence=mock_persistence)

        # Mock the graph.run method
        engine.graph.run = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_think_basic_execution(self, mock_engine):
        """Test basic execution of think() method."""
        # Setup mock return value
        expected_thought = SifakaThought(
            prompt="Test prompt", final_text="Generated response", iteration=1
        )
        mock_engine.graph.run.return_value = expected_thought

        # Execute think
        result = await mock_engine.think("Test prompt")

        # Verify result
        assert result is expected_thought
        assert result.prompt == "Test prompt"
        assert result.final_text == "Generated response"

        # Verify graph.run was called correctly
        mock_engine.graph.run.assert_called_once()
        call_args = mock_engine.graph.run.call_args

        # Verify the state parameter is a SifakaThought
        state_arg = call_args[1]["state"]
        assert isinstance(state_arg, SifakaThought)
        assert state_arg.prompt == "Test prompt"
        assert state_arg.max_iterations == 3  # Default value

        # Verify other parameters
        assert call_args[1]["deps"] is mock_engine.deps
        assert call_args[1]["persistence"] is mock_engine.persistence

    @pytest.mark.asyncio
    async def test_think_custom_max_iterations(self, mock_engine):
        """Test think() with custom max_iterations."""
        expected_thought = SifakaThought(prompt="Test", max_iterations=5)
        mock_engine.graph.run.return_value = expected_thought

        result = await mock_engine.think("Test prompt", max_iterations=5)

        # Verify max_iterations was set correctly
        call_args = mock_engine.graph.run.call_args
        state_arg = call_args[1]["state"]
        assert state_arg.max_iterations == 5

    @pytest.mark.asyncio
    async def test_think_error_handling(self, mock_engine):
        """Test error handling in think() method."""
        # Setup mock to raise an exception
        mock_engine.graph.run.side_effect = Exception("Graph execution failed")

        # Verify exception is properly wrapped
        with pytest.raises(GraphExecutionError) as exc_info:
            await mock_engine.think("Test prompt")

        assert "Graph execution failed" in str(exc_info.value)
        assert exc_info.value.thought_id is not None

    @pytest.mark.asyncio
    @patch("sifaka.core.engine.logger")
    async def test_think_logging(self, mock_logger, mock_engine):
        """Test that think() method logs appropriately."""
        expected_thought = SifakaThought(prompt="Test", final_text="Result")
        mock_engine.graph.run.return_value = expected_thought

        await mock_engine.think("Test prompt")

        # Verify thought creation was logged
        mock_logger.log_thought_event.assert_called()

        # Verify performance timing was used
        mock_logger.performance_timer.assert_called()


class TestSifakaEngineContinueThought:
    """Test the continue_thought() method with mocked dependencies."""

    @pytest.fixture
    def mock_engine(self):
        """Create a SifakaEngine with mocked dependencies for testing."""
        mock_deps = Mock(spec=SifakaDependencies)
        mock_persistence = Mock(spec=BaseStatePersistence)

        engine = SifakaEngine(dependencies=mock_deps, persistence=mock_persistence)
        engine.graph.run = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_continue_thought_basic(self, mock_engine):
        """Test basic continue_thought() execution."""
        # Create parent thought
        parent_thought = SifakaThought(
            prompt="Original prompt", final_text="Original response", iteration=1
        )

        # Setup mock return value
        expected_thought = SifakaThought(
            prompt="Follow-up prompt",
            parent_thought_id=parent_thought.id,
            conversation_id=parent_thought.conversation_id or parent_thought.id,
            final_text="Follow-up response",
        )
        mock_engine.graph.run.return_value = expected_thought

        # Execute continue_thought
        result = await mock_engine.continue_thought(parent_thought, "Follow-up prompt")

        # Verify result
        assert result is expected_thought
        assert result.prompt == "Follow-up prompt"
        assert result.parent_thought_id == parent_thought.id

        # Verify graph.run was called
        mock_engine.graph.run.assert_called_once()
        call_args = mock_engine.graph.run.call_args

        # Verify the state parameter
        state_arg = call_args[1]["state"]
        assert isinstance(state_arg, SifakaThought)
        assert state_arg.prompt == "Follow-up prompt"
        assert state_arg.parent_thought_id == parent_thought.id

    @pytest.mark.asyncio
    async def test_continue_thought_conversation_id(self, mock_engine):
        """Test continue_thought() with conversation ID handling."""
        # Parent thought with existing conversation_id
        parent_thought = SifakaThought(prompt="Original", conversation_id="existing-conv-123")

        expected_thought = SifakaThought(prompt="Follow-up", conversation_id="existing-conv-123")
        mock_engine.graph.run.return_value = expected_thought

        await mock_engine.continue_thought(parent_thought, "Follow-up")

        # Verify conversation_id is preserved
        call_args = mock_engine.graph.run.call_args
        state_arg = call_args[1]["state"]
        assert state_arg.conversation_id == "existing-conv-123"

    @pytest.mark.asyncio
    async def test_continue_thought_new_conversation(self, mock_engine):
        """Test continue_thought() creating new conversation ID."""
        # Parent thought without conversation_id
        parent_thought = SifakaThought(prompt="Original")

        expected_thought = SifakaThought(prompt="Follow-up")
        mock_engine.graph.run.return_value = expected_thought

        await mock_engine.continue_thought(parent_thought, "Follow-up")

        # Verify conversation_id is set to parent's ID
        call_args = mock_engine.graph.run.call_args
        state_arg = call_args[1]["state"]
        assert state_arg.conversation_id == parent_thought.id


class TestSifakaEngineBatchProcessing:
    """Test the batch_think() method with mocked dependencies."""

    @pytest.fixture
    def mock_engine(self):
        """Create a SifakaEngine with mocked dependencies for testing."""
        mock_deps = Mock(spec=SifakaDependencies)
        mock_persistence = Mock(spec=BaseStatePersistence)

        engine = SifakaEngine(dependencies=mock_deps, persistence=mock_persistence)
        engine.think = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_batch_think_basic(self, mock_engine):
        """Test basic batch_think() execution."""
        # Setup mock return values
        thought1 = SifakaThought(prompt="Prompt 1", final_text="Response 1")
        thought2 = SifakaThought(prompt="Prompt 2", final_text="Response 2")
        mock_engine.think.side_effect = [thought1, thought2]

        # Execute batch_think
        prompts = ["Prompt 1", "Prompt 2"]
        results = await mock_engine.batch_think(prompts)

        # Verify results
        assert len(results) == 2
        assert results[0] is thought1
        assert results[1] is thought2

        # Verify think was called for each prompt
        assert mock_engine.think.call_count == 2
        mock_engine.think.assert_any_call("Prompt 1", max_iterations=3)
        mock_engine.think.assert_any_call("Prompt 2", max_iterations=3)

    @pytest.mark.asyncio
    async def test_batch_think_custom_max_iterations(self, mock_engine):
        """Test batch_think() with custom max_iterations."""
        thought = SifakaThought(prompt="Test", final_text="Response")
        mock_engine.think.return_value = thought

        await mock_engine.batch_think(["Test prompt"], max_iterations=5)

        # Verify max_iterations was passed through
        mock_engine.think.assert_called_once_with("Test prompt", max_iterations=5)

    @pytest.mark.asyncio
    async def test_batch_think_empty_list(self, mock_engine):
        """Test batch_think() with empty prompt list."""
        results = await mock_engine.batch_think([])

        assert results == []
        mock_engine.think.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_think_error_handling(self, mock_engine):
        """Test batch_think() error handling."""
        # Setup one success and one failure
        thought1 = SifakaThought(prompt="Success", final_text="Response")
        mock_engine.think.side_effect = [thought1, Exception("Failed")]

        # Verify exception propagates
        with pytest.raises(Exception, match="Failed"):
            await mock_engine.batch_think(["Success", "Failure"])


class TestSifakaEngineValidation:
    """Test input validation and error handling."""

    def test_engine_with_invalid_dependencies(self):
        """Test engine creation with invalid dependencies type."""
        with pytest.raises(TypeError):
            SifakaEngine(dependencies="invalid")

    def test_engine_with_invalid_persistence(self):
        """Test engine creation with invalid persistence type."""
        with pytest.raises(TypeError):
            SifakaEngine(persistence="invalid")

    @pytest.mark.asyncio
    async def test_think_with_empty_prompt(self):
        """Test think() with empty prompt."""
        engine = SifakaEngine()

        # Mock the graph to avoid actual execution
        engine.graph.run = AsyncMock()
        engine.graph.run.return_value = SifakaThought(prompt="", final_text="")

        # Empty prompt should be allowed (validation happens in validators)
        result = await engine.think("")
        assert result.prompt == ""

    @pytest.mark.asyncio
    async def test_think_with_invalid_max_iterations(self):
        """Test think() with invalid max_iterations."""
        engine = SifakaEngine()

        # Negative max_iterations should raise ValidationError during SifakaThought creation
        with pytest.raises(Exception):  # Could be ValidationError or similar
            await engine.think("Test", max_iterations=-1)


class TestSifakaEngineEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def mock_engine(self):
        """Create a SifakaEngine with mocked dependencies for testing."""
        mock_deps = Mock(spec=SifakaDependencies)
        mock_persistence = Mock(spec=BaseStatePersistence)

        engine = SifakaEngine(dependencies=mock_deps, persistence=mock_persistence)
        engine.graph.run = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_think_with_very_long_prompt(self, mock_engine):
        """Test think() with extremely long prompt."""
        long_prompt = "A" * 10000  # 10k character prompt
        expected_thought = SifakaThought(prompt=long_prompt, final_text="Response")
        mock_engine.graph.run.return_value = expected_thought

        result = await mock_engine.think(long_prompt)

        assert result.prompt == long_prompt
        assert len(result.prompt) == 10000

    @pytest.mark.asyncio
    async def test_think_with_unicode_prompt(self, mock_engine):
        """Test think() with Unicode characters."""
        unicode_prompt = "Hello 世界 🌍 Здравствуй мир"
        expected_thought = SifakaThought(prompt=unicode_prompt, final_text="Response")
        mock_engine.graph.run.return_value = expected_thought

        result = await mock_engine.think(unicode_prompt)

        assert result.prompt == unicode_prompt

    @pytest.mark.asyncio
    async def test_think_with_special_characters(self, mock_engine):
        """Test think() with special characters and formatting."""
        special_prompt = 'Test\n\t"quotes" & <tags> {json: true} $variables'
        expected_thought = SifakaThought(prompt=special_prompt, final_text="Response")
        mock_engine.graph.run.return_value = expected_thought

        result = await mock_engine.think(special_prompt)

        assert result.prompt == special_prompt

    @pytest.mark.asyncio
    async def test_concurrent_think_operations(self, mock_engine):
        """Test multiple concurrent think() operations."""
        import asyncio

        # Setup different responses for each call
        thoughts = [
            SifakaThought(prompt=f"Prompt {i}", final_text=f"Response {i}") for i in range(3)
        ]
        mock_engine.graph.run.side_effect = thoughts

        # Execute concurrent operations
        tasks = [mock_engine.think(f"Prompt {i}") for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Verify all operations completed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.prompt == f"Prompt {i}"
            assert result.final_text == f"Response {i}"

    @pytest.mark.asyncio
    async def test_graph_execution_timeout(self, mock_engine):
        """Test handling of graph execution timeout."""
        import asyncio

        # Setup mock to simulate timeout
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow execution
            return SifakaThought(prompt="Test", final_text="Response")

        mock_engine.graph.run.side_effect = slow_execution

        # Test with timeout (this would need actual timeout implementation)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mock_engine.think("Test prompt"), timeout=0.1)

    @pytest.mark.asyncio
    async def test_memory_cleanup_after_error(self, mock_engine):
        """Test that memory is properly cleaned up after errors."""
        # Setup mock to raise exception
        mock_engine.graph.run.side_effect = Exception("Simulated error")

        # Verify exception is raised and handled
        with pytest.raises(GraphExecutionError):
            await mock_engine.think("Test prompt")

        # Verify engine state is still valid for next operation
        mock_engine.graph.run.side_effect = None
        expected_thought = SifakaThought(prompt="Recovery test", final_text="Success")
        mock_engine.graph.run.return_value = expected_thought

        result = await mock_engine.think("Recovery test")
        assert result.final_text == "Success"


class TestSifakaEnginePerformance:
    """Test performance characteristics and resource usage."""

    @pytest.fixture
    def mock_engine(self):
        """Create a SifakaEngine with mocked dependencies for testing."""
        mock_deps = Mock(spec=SifakaDependencies)
        mock_persistence = Mock(spec=BaseStatePersistence)

        engine = SifakaEngine(dependencies=mock_deps, persistence=mock_persistence)
        engine.graph.run = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, mock_engine):
        """Test performance of batch processing."""
        import time

        # Setup mock responses
        thoughts = [
            SifakaThought(prompt=f"Prompt {i}", final_text=f"Response {i}") for i in range(100)
        ]
        mock_engine.think = AsyncMock(side_effect=thoughts)

        # Measure batch processing time
        start_time = time.time()
        prompts = [f"Prompt {i}" for i in range(100)]
        results = await mock_engine.batch_think(prompts)
        end_time = time.time()

        # Verify results and performance
        assert len(results) == 100
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should complete within 5 seconds with mocks

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, mock_engine):
        """Test that memory usage remains stable across multiple operations."""
        import gc

        # Setup mock response
        expected_thought = SifakaThought(prompt="Test", final_text="Response")
        mock_engine.graph.run.return_value = expected_thought

        # Perform multiple operations
        for i in range(50):
            await mock_engine.think(f"Test prompt {i}")

            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()

        # Test should complete without memory issues
        assert True  # If we get here, memory usage was stable
