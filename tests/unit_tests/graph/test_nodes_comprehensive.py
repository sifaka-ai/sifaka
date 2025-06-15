"""Comprehensive unit tests for Sifaka graph nodes.

This module provides extensive testing of the graph nodes that orchestrate
the Sifaka workflow:
- GenerateNode: Text generation using PydanticAI agents
- ValidateNode: Parallel validator execution
- CritiqueNode: Parallel critic execution
- Node transitions and decision logic
- Error handling and recovery
- Performance characteristics

Tests cover:
- Node execution with mocked dependencies
- State transitions and flow control
- Parallel execution behavior
- Error scenarios and recovery
- Performance and resource usage
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.graph.nodes import CritiqueNode, GenerateNode, ValidateNode


class MockGraphRunContext:
    """Mock GraphRunContext for testing nodes."""

    def __init__(self, state: SifakaThought, deps: SifakaDependencies):
        self.state = state
        self.deps = deps


class TestGenerateNode:
    """Test GenerateNode functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        mock_deps = Mock(spec=SifakaDependencies)
        mock_deps.generator_agent = Mock()
        mock_deps.generator_agent.model = "openai:gpt-4"
        return mock_deps

    @pytest.fixture
    def generate_node(self):
        """Create GenerateNode instance for testing."""
        return GenerateNode()

    @pytest.mark.asyncio
    async def test_generate_node_basic_execution(self, generate_node, mock_dependencies):
        """Test basic GenerateNode execution."""
        # Setup thought
        thought = SifakaThought(prompt="Test prompt")
        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Mock agent response
        mock_result = Mock()
        mock_result.data = "Generated response text"
        mock_result.all_messages = [
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Generated response text"},
        ]
        mock_dependencies.generator_agent.run_async = AsyncMock(return_value=mock_result)

        # Execute node
        result = await generate_node.run(ctx)

        # Verify result is ValidateNode
        assert isinstance(result, ValidateNode)

        # Verify thought was updated
        assert len(thought.generations) == 1
        generation = thought.generations[0]
        assert generation.text == "Generated response text"
        assert generation.model == "openai:gpt-4"
        assert generation.iteration == 0
        assert len(generation.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_generate_node_with_context(self, generate_node, mock_dependencies):
        """Test GenerateNode with existing context and feedback."""
        # Setup thought with previous generation and feedback
        thought = SifakaThought(prompt="Test prompt", iteration=1)
        thought.add_generation("First draft", "gpt-4", 0)
        thought.add_validation("length", False, {"reason": "too short"})
        thought.add_critique("constitutional", "Needs more detail", ["Add examples"], True)

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Mock agent response
        mock_result = Mock()
        mock_result.data = "Improved response with examples"
        mock_result.all_messages = [
            {"role": "user", "content": "Improved prompt"},
            {"role": "assistant", "content": "Improved response with examples"},
        ]
        mock_dependencies.generator_agent.run_async = AsyncMock(return_value=mock_result)

        # Execute node
        await generate_node.run(ctx)

        # Verify new generation was added
        assert len(thought.generations) == 2
        new_generation = thought.generations[1]
        assert new_generation.text == "Improved response with examples"
        assert new_generation.iteration == 1

    @pytest.mark.asyncio
    async def test_generate_node_error_handling(self, generate_node, mock_dependencies):
        """Test GenerateNode error handling."""
        thought = SifakaThought(prompt="Test prompt")
        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Mock agent to raise exception
        mock_dependencies.generator_agent.run_async = AsyncMock(
            side_effect=Exception("Generation failed")
        )

        # Verify exception is properly handled
        with pytest.raises(Exception, match="Generation failed"):
            await generate_node.run(ctx)

    @pytest.mark.asyncio
    async def test_generate_node_with_tools(self, generate_node, mock_dependencies):
        """Test GenerateNode with tool usage."""
        thought = SifakaThought(prompt="Search for renewable energy info")
        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Mock agent response with tool usage
        mock_result = Mock()
        mock_result.data = "Based on search results, renewable energy..."
        mock_result.all_messages = [
            {"role": "user", "content": "Search for renewable energy info"},
            {"role": "assistant", "content": "Based on search results, renewable energy..."},
        ]
        # Simulate tool calls in the result
        mock_result.tool_calls = [
            {
                "tool": "web_search",
                "input": {"query": "renewable energy"},
                "output": {"results": []},
            }
        ]
        mock_dependencies.generator_agent.run_async = AsyncMock(return_value=mock_result)

        # Execute node
        await generate_node.run(ctx)

        # Verify generation includes tool usage
        assert len(thought.generations) == 1
        generation = thought.generations[0]
        assert "renewable energy" in generation.text


class TestValidateNode:
    """Test ValidateNode functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies with validators."""
        mock_deps = Mock(spec=SifakaDependencies)

        # Create mock validators
        mock_validator1 = Mock(spec=Validator)
        mock_validator1.name = "length"
        mock_validator1.validate_async = AsyncMock(
            return_value={"passed": True, "details": {"word_count": 150}}
        )

        mock_validator2 = Mock(spec=Validator)
        mock_validator2.name = "coherence"
        mock_validator2.validate_async = AsyncMock(
            return_value={"passed": False, "details": {"score": 0.6, "threshold": 0.8}}
        )

        mock_deps.validators = [mock_validator1, mock_validator2]
        return mock_deps

    @pytest.fixture
    def validate_node(self):
        """Create ValidateNode instance for testing."""
        return ValidateNode()

    @pytest.mark.asyncio
    async def test_validate_node_basic_execution(self, validate_node, mock_dependencies):
        """Test basic ValidateNode execution."""
        # Setup thought with generation
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Generated text for validation", "gpt-4", 0)

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Execute node
        result = await validate_node.run(ctx)

        # Verify result is CritiqueNode
        assert isinstance(result, CritiqueNode)

        # Verify validations were added
        assert len(thought.validations) == 2

        # Check individual validation results
        length_validation = next(v for v in thought.validations if v.validator_name == "length")
        assert length_validation.passed is True
        assert length_validation.details["word_count"] == 150

        coherence_validation = next(
            v for v in thought.validations if v.validator_name == "coherence"
        )
        assert coherence_validation.passed is False
        assert coherence_validation.details["score"] == 0.6

    @pytest.mark.asyncio
    async def test_validate_node_parallel_execution(self, validate_node, mock_dependencies):
        """Test that validators run in parallel."""
        import time

        # Setup thought
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Test text", "gpt-4", 0)

        # Create slow validators to test parallelism
        async def slow_validator1(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"passed": True, "details": {}}

        async def slow_validator2(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"passed": True, "details": {}}

        mock_dependencies.validators[0].validate_async = slow_validator1
        mock_dependencies.validators[1].validate_async = slow_validator2

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Measure execution time
        start_time = time.time()
        await validate_node.run(ctx)
        end_time = time.time()

        # Should complete in ~0.1 seconds (parallel) not ~0.2 seconds (sequential)
        execution_time = end_time - start_time
        assert execution_time < 0.15  # Allow some overhead

    @pytest.mark.asyncio
    async def test_validate_node_error_handling(self, validate_node, mock_dependencies):
        """Test ValidateNode error handling."""
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Test text", "gpt-4", 0)

        # Make one validator fail
        mock_dependencies.validators[0].validate_async = AsyncMock(
            side_effect=Exception("Validator failed")
        )

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Execute node - should handle errors gracefully
        result = await validate_node.run(ctx)

        # Should still proceed to CritiqueNode
        assert isinstance(result, CritiqueNode)

        # Should have validation results (including error)
        assert len(thought.validations) >= 1

        # Failed validator should have error details
        failed_validation = next(
            v for v in thought.validations if v.validator_name == "length" and not v.passed
        )
        assert "error" in failed_validation.details

    @pytest.mark.asyncio
    async def test_validate_node_no_validators(self, validate_node):
        """Test ValidateNode with no validators configured."""
        # Setup dependencies with no validators
        mock_deps = Mock(spec=SifakaDependencies)
        mock_deps.validators = []

        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Test text", "gpt-4", 0)

        ctx = MockGraphRunContext(thought, mock_deps)

        # Execute node
        result = await validate_node.run(ctx)

        # Should proceed to CritiqueNode
        assert isinstance(result, CritiqueNode)

        # Should have no validations
        assert len(thought.validations) == 0

    @pytest.mark.asyncio
    async def test_validate_node_no_current_text(self, validate_node, mock_dependencies):
        """Test ValidateNode when thought has no current text."""
        thought = SifakaThought(prompt="Test prompt")
        # No generation added

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Execute node
        result = await validate_node.run(ctx)

        # Should still proceed to CritiqueNode
        assert isinstance(result, CritiqueNode)

        # Validators should still run (might validate empty text)
        assert len(thought.validations) == 2


class TestCritiqueNode:
    """Test CritiqueNode functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies with critics."""
        mock_deps = Mock(spec=SifakaDependencies)

        # Create mock critics
        mock_critic1 = Mock(spec=Critic)
        mock_critic1.name = "constitutional"
        mock_critic1.critique_async = AsyncMock(
            return_value={
                "feedback": "Good structure but needs examples",
                "suggestions": ["Add examples", "Improve conclusion"],
                "needs_improvement": True,
            }
        )

        mock_critic2 = Mock(spec=Critic)
        mock_critic2.name = "reflexion"
        mock_critic2.critique_async = AsyncMock(
            return_value={"feedback": "Well written", "suggestions": [], "needs_improvement": False}
        )

        mock_deps.critics = {"constitutional": mock_critic1, "reflexion": mock_critic2}
        return mock_deps

    @pytest.fixture
    def critique_node(self):
        """Create CritiqueNode instance for testing."""
        return CritiqueNode()

    @pytest.mark.asyncio
    async def test_critique_node_continue_iteration(self, critique_node, mock_dependencies):
        """Test CritiqueNode when iteration should continue."""
        # Setup thought that should continue (has improvement suggestions)
        thought = SifakaThought(prompt="Test prompt", iteration=1, max_iterations=3)
        thought.add_generation("First draft", "gpt-4", 0)
        thought.add_validation("length", False, {"reason": "too short"})

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Execute node
        result = await critique_node.run(ctx)

        # Should return GenerateNode for next iteration
        assert isinstance(result, GenerateNode)

        # Verify critiques were added
        assert len(thought.critiques) == 2

        # Verify iteration was incremented
        assert thought.iteration == 2

        # Check critique details
        constitutional_critique = next(
            c for c in thought.critiques if c.critic_name == "constitutional"
        )
        assert constitutional_critique.needs_improvement is True
        assert len(constitutional_critique.suggestions) == 2

    @pytest.mark.asyncio
    async def test_critique_node_end_workflow(self, critique_node, mock_dependencies):
        """Test CritiqueNode when workflow should end."""
        # Setup thought that should end (max iterations reached)
        thought = SifakaThought(prompt="Test prompt", iteration=3, max_iterations=3)
        thought.add_generation("Final draft", "gpt-4", 2)
        thought.add_validation("length", True, {"word_count": 200})

        # Make critics suggest no improvement
        mock_dependencies.critics["constitutional"].critique_async = AsyncMock(
            return_value={
                "feedback": "Excellent work",
                "suggestions": [],
                "needs_improvement": False,
            }
        )

        ctx = MockGraphRunContext(thought, mock_dependencies)

        # Execute node
        result = await critique_node.run(ctx)

        # Should end workflow
        from pydantic_graph import End

        assert isinstance(result, End)

        # Verify final_text was set
        assert thought.final_text == "Final draft"
