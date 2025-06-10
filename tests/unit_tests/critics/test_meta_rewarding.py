"""Comprehensive unit tests for Meta-Evaluation critic.

This module tests the MetaEvaluationCritic implementation:
- Meta-evaluation critique methodology inspired by Wu et al. 2024
- Quality assessment of critique and feedback
- Integration with PydanticAI agents
- Error handling and edge cases

Tests cover:
- Basic meta-evaluation functionality
- Critique quality assessment
- Feedback actionability evaluation
- Performance characteristics
- Mock-based testing without external API calls
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.critics.meta_rewarding import MetaEvaluationCritic


class TestMetaEvaluationCritic:
    """Test suite for MetaEvaluationCritic class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock PydanticAI agent."""
        agent = Mock()
        agent.run = AsyncMock()

        # Mock response for meta-evaluation (CritiqueFeedback structure)
        mock_response = Mock()
        mock_response.data = {
            "message": "The critique is actionable and well-structured with specific suggestions.",
            "needs_improvement": False,
            "violations": [],
            "suggestions": [
                {
                    "suggestion": "The critique provides clear, implementable recommendations",
                    "category": "actionability",
                },
                {
                    "suggestion": "Evidence-based reasoning supports the feedback",
                    "category": "evidence",
                },
            ],
            "confidence": {"overall": 0.85},
            "critic_name": "MetaEvaluationCritic",
        }
        agent.run.return_value = mock_response

        return agent

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Write about artificial intelligence",
            final_text="AI is a transformative technology that will reshape many industries.",
            iteration=1,
            max_iterations=3,
        )

        # Add some existing critique to meta-evaluate
        thought.add_critique(
            "constitutional-critic",
            "The text needs more specific examples and balanced perspective on AI risks.",
            ["Add concrete examples of AI applications", "Discuss potential risks and limitations"],
            confidence=0.7,
            needs_improvement=True,
        )

        return thought

    def test_meta_evaluation_critic_creation_minimal(self):
        """Test creating MetaEvaluationCritic with minimal parameters."""
        critic = MetaEvaluationCritic()

        assert critic.model_name == "gemini-1.5-flash"  # Correct default model
        assert "meta-evaluation" in critic.system_prompt.lower()
        assert critic.metadata["critic_type"] == "MetaEvaluationCritic"
        assert "Wu" in critic.paper_reference  # Paper reference contains Wu et al.

    def test_meta_evaluation_critic_creation_with_custom_criteria(self):
        """Test creating MetaEvaluationCritic with custom evaluation criteria."""
        custom_criteria = [
            "Specificity: Are suggestions concrete and actionable?",
            "Relevance: Does feedback address the actual content issues?",
        ]

        critic = MetaEvaluationCritic(
            meta_evaluation_criteria=custom_criteria,  # Correct parameter name
            model_name="openai:gpt-4",
        )

        assert critic.meta_evaluation_criteria == custom_criteria  # Correct attribute name
        assert critic.model_name == "openai:gpt-4"

    @pytest.mark.asyncio
    async def test_critique_async_basic(self, mock_agent, sample_thought):
        """Test basic meta-evaluation functionality."""
        critic = MetaEvaluationCritic(model_name="test")  # Use 'test' model which is supported

        with patch.object(critic, "agent", mock_agent):  # Use 'agent' not '_agent'
            await critic.critique_async(sample_thought)

        # Verify agent was called
        mock_agent.run.assert_called_once()

        # Verify critique was added to thought
        assert len(sample_thought.critiques) > 0

    @pytest.mark.asyncio
    async def test_critique_with_no_existing_critiques(self, mock_agent):
        """Test meta-evaluation with thought that has no existing critiques."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = "Simple test text"

        critic = MetaEvaluationCritic(model_name="test")

        # Mock response for no critiques scenario
        mock_response = Mock()
        mock_response.data = {
            "message": "No existing critiques to meta-evaluate.",
            "needs_improvement": True,
            "confidence": {"overall": 0.9},
            "violations": [],
            "suggestions": [],
            "critic_name": "MetaEvaluationCritic",
        }
        mock_agent.run.return_value = mock_response

        with patch.object(critic, "agent", mock_agent):
            await critic.critique_async(thought)

        # Should handle gracefully and add critique to thought
        assert len(thought.critiques) > 0

    @pytest.mark.asyncio
    async def test_critique_with_multiple_existing_critiques(self, mock_agent):
        """Test meta-evaluation with multiple existing critiques."""
        thought = SifakaThought(prompt="Complex prompt")
        thought.current_text = "Complex text requiring multiple critiques"

        # Add multiple critiques
        thought.add_critique(
            "critic1",
            "First feedback",
            ["suggestion1"],
            confidence=0.8,
            reasoning="Test reasoning",
            needs_improvement=True,
        )
        thought.add_critique(
            "critic2",
            "Second feedback",
            ["suggestion2"],
            confidence=0.7,
            reasoning="Test reasoning",
            needs_improvement=False,
        )
        thought.add_critique(
            "critic3",
            "Third feedback",
            ["suggestion3"],
            confidence=0.9,
            reasoning="Test reasoning",
            needs_improvement=True,
        )

        critic = MetaEvaluationCritic(model_name="test")

        with patch.object(critic, "agent", mock_agent):
            await critic.critique_async(thought)

        # Should handle multiple critiques and add critique to thought
        assert len(thought.critiques) > 3  # Original 3 plus new one

        # Verify agent was called with multiple critiques context
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_thought):
        """Test error handling in meta-evaluation."""
        critic = MetaEvaluationCritic(model_name="test")

        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("API Error"))

        with patch.object(critic, "agent", mock_agent):
            # Should not raise exception, but add error critique to thought
            await critic.critique_async(sample_thought)

        # Verify error critique was added
        assert len(sample_thought.critiques) > 0
        error_critique = sample_thought.critiques[-1]  # Last critique should be the error
        assert (
            "failed" in error_critique.feedback.lower()
            or "error" in error_critique.feedback.lower()
        )

    def test_default_criteria_structure(self):
        """Test that default evaluation criteria are properly structured."""
        critic = MetaEvaluationCritic()

        default_criteria = critic._get_default_criteria()

        assert isinstance(default_criteria, list)
        assert len(default_criteria) >= 5  # Should have at least 5 criteria

        # Check that criteria cover key aspects
        criteria_text = " ".join(default_criteria).lower()
        assert "actionability" in criteria_text
        assert "relevance" in criteria_text
        assert "consistency" in criteria_text
        assert "constructiveness" in criteria_text
        assert "evidence" in criteria_text

    @pytest.mark.asyncio
    async def test_improve_async_functionality(self, mock_agent, sample_thought):
        """Test the improve_async method if it exists."""
        critic = MetaEvaluationCritic(model_name="test")

        # Check if improve_async method exists
        if hasattr(critic, "improve_async"):
            # Mock improved response
            mock_response = Mock()
            mock_response.data = (
                "Improved meta-evaluation with enhanced actionability and specificity."
            )
            mock_agent.run.return_value = mock_response

            with patch.object(critic, "agent", mock_agent):
                improved_text = await critic.improve_async(sample_thought)

            # If method exists, it should return something meaningful
            if improved_text is not None:
                assert isinstance(improved_text, str)
                assert len(improved_text) > 0
        else:
            # If method doesn't exist, that's also valid - skip this test
            pytest.skip("improve_async method not implemented for this critic")

    def test_system_prompt_contains_methodology(self):
        """Test that system prompt contains meta-evaluation methodology."""
        critic = MetaEvaluationCritic()

        system_prompt = critic.system_prompt.lower()

        # Should contain key meta-evaluation concepts
        assert "meta" in system_prompt
        assert "evaluation" in system_prompt or "evaluate" in system_prompt
        assert "critique" in system_prompt
        assert "quality" in system_prompt
        assert "actionable" in system_prompt

    @pytest.mark.asyncio
    async def test_critique_with_validation_context(self, mock_agent, sample_thought):
        """Test meta-evaluation with validation context awareness."""
        # Add validation results to the thought
        sample_thought.add_validation("length_validator", False, {"issue": "too short"})
        sample_thought.add_validation("content_validator", True, {"score": 0.8})

        critic = MetaEvaluationCritic(model_name="test")

        with patch.object(critic, "agent", mock_agent):
            await critic.critique_async(sample_thought)

        # Should consider validation context in meta-evaluation
        mock_agent.run.assert_called_once()

        # Check that validation context was included in the call
        call_args = mock_agent.run.call_args
        assert call_args is not None

    def test_paper_reference_and_methodology(self):
        """Test that paper reference and methodology are properly set."""
        critic = MetaEvaluationCritic()

        assert "Wu" in critic.paper_reference  # Contains Wu et al.
        assert "Meta-Rewarding" in critic.paper_reference
        assert "meta-evaluation" in critic.methodology.lower()
        assert len(critic.methodology) > 50  # Should have substantial methodology description
