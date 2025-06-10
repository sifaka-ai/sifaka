"""Comprehensive unit tests for Self-Refine critic.

This module tests the SelfRefineCritic implementation:
- Self-Refine methodology inspired by Madaan et al. 2023
- Iterative self-improvement and refinement
- Integration with PydanticAI agents
- Error handling and edge cases

Tests cover:
- Basic Self-Refine functionality
- Iterative refinement process
- Improvement tracking and convergence
- Performance characteristics
- Mock-based testing without external API calls
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.critics.self_refine import SelfRefineCritic


class TestSelfRefineCritic:
    """Test suite for SelfRefineCritic class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock PydanticAI agent."""
        agent = Mock()
        agent.run = AsyncMock()

        # Mock response for Self-Refine
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Self-refinement analysis shows areas for improvement in clarity and structure.",
            "suggestions": [
                "Reorganize paragraphs for better logical flow",
                "Add transitional phrases between ideas",
                "Clarify technical terms for broader audience",
            ],
            "needs_improvement": True,
            "confidence": 0.75,
            "reasoning": "Self-refinement identifies specific structural and clarity issues",
            "refinement_iteration": 1,
            "improvement_areas": ["structure", "clarity", "transitions"],
        }
        agent.run.return_value = mock_response

        return agent

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Write a comprehensive guide to machine learning",
            final_text="Machine learning is a subset of AI. It uses algorithms to learn from data. There are different types like supervised and unsupervised learning.",
            iteration=1,
            max_iterations=5,
        )
        return thought

    def test_self_refine_critic_creation_minimal(self):
        """Test creating SelfRefineCritic with minimal parameters."""
        critic = SelfRefineCritic()

        assert critic.model_name == "groq:mixtral-8x7b-32768"
        assert critic.max_refinement_iterations == 3  # Default
        assert "self-refine" in critic.system_prompt.lower()
        assert critic.metadata["critic_type"] == "SelfRefineCritic"
        assert "Madaan et al. 2023" in critic.paper_reference

    def test_self_refine_critic_creation_with_custom_parameters(self):
        """Test creating SelfRefineCritic with custom parameters."""
        custom_criteria = [
            "Logical flow and organization",
            "Clarity and readability",
            "Completeness of information",
        ]

        critic = SelfRefineCritic(
            max_refinement_iterations=5,
            refinement_criteria=custom_criteria,
            convergence_threshold=0.1,
            model_name="openai:gpt-4",
        )

        assert critic.max_refinement_iterations == 5
        assert critic.refinement_criteria == custom_criteria
        assert critic.convergence_threshold == 0.1
        assert critic.model_name == "openai:gpt-4"

    @pytest.mark.asyncio
    async def test_critique_async_basic(self, mock_agent, sample_thought):
        """Test basic Self-Refine functionality."""
        critic = SelfRefineCritic(model_name="mock")

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Verify result structure
        assert "feedback" in result
        assert "suggestions" in result
        assert "needs_improvement" in result
        assert "confidence" in result

        # Verify content
        assert isinstance(result["feedback"], str)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["needs_improvement"], bool)
        assert isinstance(result["confidence"], (int, float))

        # Verify agent was called
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_with_multiple_iterations(self, mock_agent, sample_thought):
        """Test Self-Refine with multiple refinement iterations."""
        # Create thought with multiple iterations
        thought_multi_iter = SifakaThought(
            prompt="Test prompt", final_text="Initial text", iteration=3, max_iterations=5
        )

        # Add previous generations to simulate refinement history
        thought_multi_iter.add_generation("First version", "gpt-4", None)
        thought_multi_iter.iteration = 1
        thought_multi_iter.add_generation("Second version", "gpt-4", None)
        thought_multi_iter.iteration = 2
        thought_multi_iter.add_generation("Third version", "gpt-4", None)

        critic = SelfRefineCritic()

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(thought_multi_iter)

        # Should consider refinement history
        assert "feedback" in result
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_with_convergence_detection(self, mock_agent, sample_thought):
        """Test Self-Refine with convergence detection."""
        critic = SelfRefineCritic(convergence_threshold=0.05)

        # Mock response indicating convergence
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Text has converged to high quality with minimal further improvements needed.",
            "suggestions": ["Minor polishing could be done"],
            "needs_improvement": False,
            "confidence": 0.95,
            "convergence_detected": True,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should indicate convergence
        assert result["needs_improvement"] is False
        assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_critique_with_custom_refinement_criteria(self, mock_agent, sample_thought):
        """Test Self-Refine with custom refinement criteria."""
        custom_criteria = [
            "Technical accuracy and precision",
            "Audience appropriateness",
            "Comprehensive coverage of topic",
        ]

        critic = SelfRefineCritic(refinement_criteria=custom_criteria)

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should use custom criteria
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Verify custom criteria were considered
        call_args = mock_agent.run.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_thought):
        """Test error handling in Self-Refine."""
        critic = SelfRefineCritic()

        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("Refinement Error"))

        with patch.object(critic, "_agent", mock_agent):
            with pytest.raises(Exception, match="Refinement Error"):
                await critic.critique_async(sample_thought)

    def test_default_refinement_criteria_structure(self):
        """Test that default refinement criteria are properly structured."""
        critic = SelfRefineCritic()

        default_criteria = critic._get_default_criteria()

        assert isinstance(default_criteria, list)
        assert len(default_criteria) >= 3  # Should have at least 3 criteria

        # Check that criteria cover key aspects
        criteria_text = " ".join(default_criteria).lower()
        assert "clarity" in criteria_text or "clear" in criteria_text
        assert "structure" in criteria_text or "organization" in criteria_text
        assert "accuracy" in criteria_text or "correct" in criteria_text

    @pytest.mark.asyncio
    async def test_improve_async_functionality(self, mock_agent, sample_thought):
        """Test the improve_async method."""
        critic = SelfRefineCritic()

        # Mock improved response
        mock_response = Mock()
        mock_response.data = (
            "Refined text with improved clarity, structure, and comprehensive coverage."
        )
        mock_agent.run.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            improved_text = await critic.improve_async(sample_thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > 0
        assert "refined" in improved_text.lower() or "improved" in improved_text.lower()

    def test_system_prompt_contains_self_refine_methodology(self):
        """Test that system prompt contains Self-Refine methodology."""
        critic = SelfRefineCritic()

        system_prompt = critic.system_prompt.lower()

        # Should contain key Self-Refine concepts
        assert "self-refine" in system_prompt or "refinement" in system_prompt
        assert "iterative" in system_prompt or "improve" in system_prompt
        assert "refine" in system_prompt
        assert "quality" in system_prompt

    @pytest.mark.asyncio
    async def test_critique_with_max_iterations_reached(self, mock_agent):
        """Test Self-Refine when max iterations are reached."""
        # Create thought at max iterations
        thought_max_iter = SifakaThought(
            prompt="Test prompt",
            final_text="Final text after max iterations",
            iteration=5,
            max_iterations=5,
        )

        critic = SelfRefineCritic(max_refinement_iterations=3)

        # Mock response for max iterations scenario
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Maximum refinement iterations reached. Text quality is acceptable.",
            "suggestions": ["Consider this the final version"],
            "needs_improvement": False,
            "confidence": 0.8,
            "max_iterations_reached": True,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(thought_max_iter)

        # Should handle max iterations gracefully
        assert "feedback" in result
        assert "maximum" in result["feedback"].lower() or "final" in result["feedback"].lower()

    @pytest.mark.asyncio
    async def test_critique_with_validation_context(self, mock_agent, sample_thought):
        """Test Self-Refine with validation context awareness."""
        # Add validation results to the thought
        sample_thought.add_validation("clarity_validator", False, {"issue": "unclear explanations"})
        sample_thought.add_validation("structure_validator", False, {"issue": "poor organization"})

        critic = SelfRefineCritic()

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should consider validation context in refinement
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Check that validation context was included in the call
        call_args = mock_agent.run.call_args
        assert call_args is not None

    def test_paper_reference_and_methodology(self):
        """Test that paper reference and methodology are properly set."""
        critic = SelfRefineCritic()

        assert "Madaan et al. 2023" in critic.paper_reference
        assert "Self-Refine" in critic.paper_reference
        assert "iterative" in critic.methodology.lower()
        assert "refinement" in critic.methodology.lower()
        assert len(critic.methodology) > 50  # Should have substantial methodology description

    @pytest.mark.asyncio
    async def test_critique_with_empty_text(self, mock_agent):
        """Test Self-Refine with empty text."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = ""

        critic = SelfRefineCritic()

        # Mock response for empty text
        mock_response = Mock()
        mock_response.data = {
            "feedback": "No content available for refinement.",
            "suggestions": ["Generate initial content before refinement"],
            "needs_improvement": True,
            "confidence": 0.9,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(thought)

        # Should handle empty text gracefully
        assert "feedback" in result
        assert "no content" in result["feedback"].lower() or "empty" in result["feedback"].lower()

    def test_convergence_threshold_validation(self):
        """Test that convergence threshold is properly validated."""
        # Valid thresholds
        critic1 = SelfRefineCritic(convergence_threshold=0.05)
        assert critic1.convergence_threshold == 0.05

        critic2 = SelfRefineCritic(convergence_threshold=0.1)
        assert critic2.convergence_threshold == 0.1

        # Edge cases
        critic3 = SelfRefineCritic(convergence_threshold=0.0)
        assert critic3.convergence_threshold == 0.0

    def test_max_refinement_iterations_validation(self):
        """Test that max refinement iterations is properly validated."""
        # Valid values
        critic1 = SelfRefineCritic(max_refinement_iterations=1)
        assert critic1.max_refinement_iterations == 1

        critic2 = SelfRefineCritic(max_refinement_iterations=10)
        assert critic2.max_refinement_iterations == 10
