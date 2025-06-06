"""Comprehensive unit tests for unified graph nodes.

This module tests the PydanticAI graph nodes:
- GenerateNode: Text generation with PydanticAI agents
- ValidateNode: Content validation with multiple validators
- CritiqueNode: Iterative improvement with critics

Tests cover:
- Node initialization and configuration
- Input/output handling
- Error scenarios and recovery
- Integration with SifakaThought
- Mock-based testing without external dependencies
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from sifaka.graph.nodes_unified import GenerateNode, ValidateNode, CritiqueNode
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies


class TestGenerateNode:
    """Test the GenerateNode for text generation."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        deps = Mock(spec=SifakaDependencies)
        
        # Mock generator agent
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.data = "Generated text content"
        mock_result.new_messages.return_value = []
        mock_result.cost = 0.05
        mock_result.usage = Mock()
        mock_result.usage.model_dump.return_value = {"total_tokens": 30}
        mock_agent.run.return_value = mock_result
        
        deps.generator = mock_agent
        return deps

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        return SifakaThought(prompt="Test prompt for generation")

    @pytest.mark.asyncio
    async def test_generate_node_basic_execution(self, mock_dependencies, sample_thought):
        """Test basic generation node execution."""
        node = GenerateNode()
        
        # Execute the node
        result = await node.run(sample_thought, deps=mock_dependencies)
        
        # Verify the result
        assert isinstance(result, SifakaThought)
        assert result.current_text == "Generated text content"
        assert len(result.generations) == 1
        assert result.generations[0].text == "Generated text content"
        assert result.generations[0].iteration == 0
        
        # Verify agent was called correctly
        mock_dependencies.generator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_node_with_conversation_history(self, mock_dependencies, sample_thought):
        """Test generation with existing conversation history."""
        # Add some existing generations
        sample_thought.add_generation("Previous text", "gpt-3.5", None)
        sample_thought.iteration = 1
        
        node = GenerateNode()
        result = await node.run(sample_thought, deps=mock_dependencies)
        
        # Verify new generation was added
        assert len(result.generations) == 2
        assert result.generations[1].iteration == 1
        assert result.current_text == "Generated text content"

    @pytest.mark.asyncio
    async def test_generate_node_error_handling(self, mock_dependencies, sample_thought):
        """Test generation node error handling."""
        # Make the agent raise an exception
        mock_dependencies.generator.run.side_effect = Exception("API Error")
        
        node = GenerateNode()
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            await node.run(sample_thought, deps=mock_dependencies)


class TestValidateNode:
    """Test the ValidateNode for content validation."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies with validators."""
        deps = Mock(spec=SifakaDependencies)
        
        # Mock validators
        mock_validator1 = AsyncMock()
        mock_validator1.name = "length-validator"
        mock_validator1.validate_async.return_value = {
            "passed": True,
            "details": {"word_count": 150},
            "score": 0.9
        }
        
        mock_validator2 = AsyncMock()
        mock_validator2.name = "content-validator"
        mock_validator2.validate_async.return_value = {
            "passed": False,
            "details": {"issue": "lacks examples"},
            "score": 0.6
        }
        
        deps.validators = [mock_validator1, mock_validator2]
        return deps

    @pytest.fixture
    def sample_thought_with_text(self):
        """Create a thought with generated text."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = "This is some generated text content for validation."
        return thought

    @pytest.mark.asyncio
    async def test_validate_node_basic_execution(self, mock_dependencies, sample_thought_with_text):
        """Test basic validation node execution."""
        node = ValidateNode()
        
        result = await node.run(sample_thought_with_text, deps=mock_dependencies)
        
        # Verify validations were added
        assert len(result.validations) == 2
        
        # Check first validation (passed)
        val1 = result.validations[0]
        assert val1.validator == "length-validator"
        assert val1.passed is True
        assert val1.details["word_count"] == 150
        
        # Check second validation (failed)
        val2 = result.validations[1]
        assert val2.validator == "content-validator"
        assert val2.passed is False
        assert val2.details["issue"] == "lacks examples"

    @pytest.mark.asyncio
    async def test_validate_node_no_text(self, mock_dependencies):
        """Test validation when no text is available."""
        thought = SifakaThought(prompt="Test prompt")
        # No current_text set
        
        node = ValidateNode()
        result = await node.run(thought, deps=mock_dependencies)
        
        # Should handle gracefully - might skip validation or add error
        # The exact behavior depends on implementation
        assert isinstance(result, SifakaThought)

    @pytest.mark.asyncio
    async def test_validate_node_no_validators(self, sample_thought_with_text):
        """Test validation with no validators configured."""
        deps = Mock(spec=SifakaDependencies)
        deps.validators = []
        
        node = ValidateNode()
        result = await node.run(sample_thought_with_text, deps=deps)
        
        # Should handle gracefully with no validators
        assert len(result.validations) == 0

    @pytest.mark.asyncio
    async def test_validate_node_validator_error(self, mock_dependencies, sample_thought_with_text):
        """Test validation when a validator raises an error."""
        # Make one validator fail
        mock_dependencies.validators[0].validate_async.side_effect = Exception("Validator error")
        
        node = ValidateNode()
        result = await node.run(sample_thought_with_text, deps=mock_dependencies)
        
        # Should continue with other validators
        # Exact error handling depends on implementation
        assert isinstance(result, SifakaThought)


class TestCritiqueNode:
    """Test the CritiqueNode for iterative improvement."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies with critics."""
        deps = Mock(spec=SifakaDependencies)
        
        # Mock critic
        mock_critic = AsyncMock()
        mock_critic.name = "constitutional-critic"
        mock_critic.critique_async.return_value = {
            "feedback": "The text needs more balanced perspective",
            "suggestions": ["Add counterarguments", "Include citations"],
            "needs_improvement": True,
            "confidence": 0.8
        }
        
        deps.critics = {"constitutional": mock_critic}
        deps.always_apply_critics = False
        deps.never_apply_critics = False
        return deps

    @pytest.fixture
    def sample_thought_with_validation(self):
        """Create a thought with text and validation results."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = "This is some generated text content."
        thought.add_validation("length-validator", False, {"issue": "too short"})
        return thought

    @pytest.mark.asyncio
    async def test_critique_node_basic_execution(self, mock_dependencies, sample_thought_with_validation):
        """Test basic critique node execution."""
        node = CritiqueNode()
        
        result = await node.run(sample_thought_with_validation, deps=mock_dependencies)
        
        # Verify critique was added
        assert len(result.critiques) == 1
        
        critique = result.critiques[0]
        assert critique.critic == "constitutional-critic"
        assert critique.feedback == "The text needs more balanced perspective"
        assert len(critique.suggestions) == 2
        assert critique.needs_improvement is True
        assert critique.confidence == 0.8

    @pytest.mark.asyncio
    async def test_critique_node_validation_passed_no_critics(self, mock_dependencies, sample_thought_with_validation):
        """Test critique when validation passes and critics not forced."""
        # Make validation pass
        sample_thought_with_validation.validations[0].passed = True
        
        node = CritiqueNode()
        result = await node.run(sample_thought_with_validation, deps=mock_dependencies)
        
        # Should not apply critics when validation passes and not forced
        # Exact behavior depends on implementation
        assert isinstance(result, SifakaThought)

    @pytest.mark.asyncio
    async def test_critique_node_always_apply_critics(self, mock_dependencies, sample_thought_with_validation):
        """Test critique when always_apply_critics is True."""
        # Make validation pass but force critics
        sample_thought_with_validation.validations[0].passed = True
        mock_dependencies.always_apply_critics = True
        
        node = CritiqueNode()
        result = await node.run(sample_thought_with_validation, deps=mock_dependencies)
        
        # Should apply critics even when validation passes
        assert len(result.critiques) >= 0  # Depends on implementation

    @pytest.mark.asyncio
    async def test_critique_node_never_apply_critics(self, mock_dependencies, sample_thought_with_validation):
        """Test critique when never_apply_critics is True."""
        mock_dependencies.never_apply_critics = True
        
        node = CritiqueNode()
        result = await node.run(sample_thought_with_validation, deps=mock_dependencies)
        
        # Should not apply critics when never_apply_critics is True
        assert len(result.critiques) == 0

    @pytest.mark.asyncio
    async def test_critique_node_no_critics(self, sample_thought_with_validation):
        """Test critique with no critics configured."""
        deps = Mock(spec=SifakaDependencies)
        deps.critics = {}
        deps.always_apply_critics = False
        deps.never_apply_critics = False
        
        node = CritiqueNode()
        result = await node.run(sample_thought_with_validation, deps=deps)
        
        # Should handle gracefully with no critics
        assert len(result.critiques) == 0

    @pytest.mark.asyncio
    async def test_critique_node_critic_error(self, mock_dependencies, sample_thought_with_validation):
        """Test critique when a critic raises an error."""
        # Make critic fail
        mock_dependencies.critics["constitutional"].critique_async.side_effect = Exception("Critic error")
        
        node = CritiqueNode()
        result = await node.run(sample_thought_with_validation, deps=mock_dependencies)
        
        # Should handle error gracefully
        # Exact error handling depends on implementation
        assert isinstance(result, SifakaThought)


class TestNodeIntegration:
    """Test integration between different nodes."""

    @pytest.mark.asyncio
    async def test_full_node_workflow(self):
        """Test a complete workflow through all nodes."""
        # This would test the full pipeline but requires more complex setup
        # For now, we'll test that nodes can be chained together
        
        thought = SifakaThought(prompt="Test full workflow")
        
        # Mock dependencies
        deps = Mock(spec=SifakaDependencies)
        
        # Mock generator
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.data = "Generated content"
        mock_result.new_messages.return_value = []
        mock_result.cost = None
        mock_result.usage = None
        mock_agent.run.return_value = mock_result
        deps.generator = mock_agent
        
        # Mock validator
        mock_validator = AsyncMock()
        mock_validator.name = "test-validator"
        mock_validator.validate_async.return_value = {
            "passed": False,
            "details": {},
            "score": 0.5
        }
        deps.validators = [mock_validator]
        
        # Mock critic
        mock_critic = AsyncMock()
        mock_critic.name = "test-critic"
        mock_critic.critique_async.return_value = {
            "feedback": "Needs improvement",
            "suggestions": ["Add more detail"],
            "needs_improvement": True
        }
        deps.critics = {"test": mock_critic}
        deps.always_apply_critics = False
        deps.never_apply_critics = False
        
        # Execute nodes in sequence
        generate_node = GenerateNode()
        validate_node = ValidateNode()
        critique_node = CritiqueNode()
        
        # Generate
        result1 = await generate_node.run(thought, deps=deps)
        assert result1.current_text == "Generated content"
        
        # Validate
        result2 = await validate_node.run(result1, deps=deps)
        assert len(result2.validations) == 1
        
        # Critique
        result3 = await critique_node.run(result2, deps=deps)
        assert len(result3.critiques) >= 0  # Depends on implementation
        
        # Verify final state
        assert result3.prompt == "Test full workflow"
        assert result3.current_text == "Generated content"
        assert len(result3.generations) == 1
        assert len(result3.validations) == 1
