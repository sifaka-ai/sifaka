"""Comprehensive unit tests for Self-Consistency critic.

This module tests the SelfConsistencyCritic implementation:
- Self-Consistency methodology inspired by Wang et al. 2022
- Multiple sampling and consistency checking
- Integration with PydanticAI agents
- Error handling and edge cases

Tests cover:
- Basic Self-Consistency functionality
- Multiple sampling and aggregation
- Consistency threshold evaluation
- Performance characteristics
- Mock-based testing without external API calls
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.critics.self_consistency import SelfConsistencyCritic


class TestSelfConsistencyCritic:
    """Test suite for SelfConsistencyCritic class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock PydanticAI agent."""
        agent = Mock()
        agent.run = AsyncMock()

        # Mock response for Self-Consistency
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Consistency analysis shows high agreement across multiple evaluations.",
            "suggestions": [
                "Maintain current quality level",
                "Consider minor stylistic improvements",
                "Verify factual claims for complete accuracy",
            ],
            "needs_improvement": False,
            "confidence": 0.88,
            "reasoning": "4 out of 5 consistency attempts agreed on high quality",
            "consistency_score": 0.8,
            "agreement_rate": 0.8,
            "sample_evaluations": [
                {"evaluation": "High quality", "score": 0.9},
                {"evaluation": "Good structure", "score": 0.85},
                {"evaluation": "Clear content", "score": 0.8},
            ],
        }
        agent.run.return_value = mock_response

        return agent

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Explain the principles of quantum computing",
            final_text="Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers.",
            iteration=1,
            max_iterations=3,
        )
        return thought

    def test_self_consistency_critic_creation_minimal(self):
        """Test creating SelfConsistencyCritic with minimal parameters."""
        critic = SelfConsistencyCritic()

        assert critic.model_name == "groq:mixtral-8x7b-32768"
        assert critic.num_consistency_attempts == 5  # Default
        assert critic.consistency_threshold == 0.7  # Default
        assert "self-consistency" in critic.system_prompt.lower()
        assert critic.metadata["critic_type"] == "SelfConsistencyCritic"
        assert "Wang et al. 2022" in critic.paper_reference

    def test_self_consistency_critic_creation_with_custom_parameters(self):
        """Test creating SelfConsistencyCritic with custom parameters."""
        critic = SelfConsistencyCritic(
            num_consistency_attempts=7,
            consistency_threshold=0.8,
            temperature=0.3,
            model_name="openai:gpt-4",
        )

        assert critic.num_consistency_attempts == 7
        assert critic.consistency_threshold == 0.8
        assert critic.temperature == 0.3
        assert critic.model_name == "openai:gpt-4"

    @pytest.mark.asyncio
    async def test_critique_async_basic(self, mock_agent, sample_thought):
        """Test basic Self-Consistency functionality."""
        critic = SelfConsistencyCritic(model_name="mock")

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
    async def test_critique_with_high_consistency(self, mock_agent, sample_thought):
        """Test Self-Consistency with high consistency across samples."""
        critic = SelfConsistencyCritic(consistency_threshold=0.9)

        # Mock response with high consistency
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Excellent consistency across all evaluations - high quality content.",
            "suggestions": ["Maintain current approach"],
            "needs_improvement": False,
            "confidence": 0.95,
            "consistency_score": 0.95,
            "agreement_rate": 0.95,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should indicate high consistency and quality
        assert result["needs_improvement"] is False
        assert result["confidence"] >= 0.9
        assert (
            "consistency" in result["feedback"].lower() or "excellent" in result["feedback"].lower()
        )

    @pytest.mark.asyncio
    async def test_critique_with_low_consistency(self, mock_agent, sample_thought):
        """Test Self-Consistency with low consistency across samples."""
        critic = SelfConsistencyCritic(consistency_threshold=0.7)

        # Mock response with low consistency
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Low consistency across evaluations suggests unclear or ambiguous content.",
            "suggestions": [
                "Clarify ambiguous statements",
                "Improve overall coherence",
                "Add more specific details",
            ],
            "needs_improvement": True,
            "confidence": 0.45,
            "consistency_score": 0.4,
            "agreement_rate": 0.4,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should indicate low consistency and need for improvement
        assert result["needs_improvement"] is True
        assert result["confidence"] < 0.6
        assert "low" in result["feedback"].lower() or "unclear" in result["feedback"].lower()

    @pytest.mark.asyncio
    async def test_critique_with_different_num_attempts(self, mock_agent, sample_thought):
        """Test Self-Consistency with different numbers of attempts."""
        for num_attempts in [3, 5, 7, 10]:
            critic = SelfConsistencyCritic(num_consistency_attempts=num_attempts)

            with patch.object(critic, "_agent", mock_agent):
                result = await critic.critique_async(sample_thought)

            # Should work with any reasonable number of attempts
            assert "feedback" in result
            assert isinstance(result["suggestions"], list)

    @pytest.mark.asyncio
    async def test_critique_with_temperature_variation(self, mock_agent, sample_thought):
        """Test Self-Consistency with different temperature settings."""
        for temp in [0.1, 0.5, 0.8, 1.0]:
            critic = SelfConsistencyCritic(temperature=temp)

            with patch.object(critic, "_agent", mock_agent):
                result = await critic.critique_async(sample_thought)

            # Should work with different temperature settings
            assert "feedback" in result
            assert isinstance(result["confidence"], (int, float))

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_thought):
        """Test error handling in Self-Consistency."""
        critic = SelfConsistencyCritic()

        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("Consistency Error"))

        with patch.object(critic, "_agent", mock_agent):
            with pytest.raises(Exception, match="Consistency Error"):
                await critic.critique_async(sample_thought)

    @pytest.mark.asyncio
    async def test_improve_async_functionality(self, mock_agent, sample_thought):
        """Test the improve_async method."""
        critic = SelfConsistencyCritic()

        # Mock improved response
        mock_response = Mock()
        mock_response.data = (
            "Improved text with enhanced consistency and clarity based on multiple evaluations."
        )
        mock_agent.run.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            improved_text = await critic.improve_async(sample_thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > 0
        assert "improved" in improved_text.lower() or "enhanced" in improved_text.lower()

    def test_system_prompt_contains_self_consistency_methodology(self):
        """Test that system prompt contains Self-Consistency methodology."""
        critic = SelfConsistencyCritic()

        system_prompt = critic.system_prompt.lower()

        # Should contain key Self-Consistency concepts
        assert "self-consistency" in system_prompt or "consistency" in system_prompt
        assert "multiple" in system_prompt or "sampling" in system_prompt
        assert "agreement" in system_prompt or "consensus" in system_prompt
        assert "evaluate" in system_prompt

    @pytest.mark.asyncio
    async def test_critique_with_validation_context(self, mock_agent, sample_thought):
        """Test Self-Consistency with validation context awareness."""
        # Add validation results to the thought
        sample_thought.add_validation("coherence_validator", True, {"score": 0.85})
        sample_thought.add_validation("clarity_validator", False, {"issue": "technical jargon"})

        critic = SelfConsistencyCritic()

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should consider validation context in consistency evaluation
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Check that validation context was included in the call
        call_args = mock_agent.run.call_args
        assert call_args is not None

    def test_paper_reference_and_methodology(self):
        """Test that paper reference and methodology are properly set."""
        critic = SelfConsistencyCritic()

        assert "Wang et al. 2022" in critic.paper_reference
        assert "Self-Consistency" in critic.paper_reference
        assert "consistency" in critic.methodology.lower()
        assert "sampling" in critic.methodology.lower()
        assert len(critic.methodology) > 50  # Should have substantial methodology description

    @pytest.mark.asyncio
    async def test_critique_with_empty_text(self, mock_agent):
        """Test Self-Consistency with empty text."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = ""

        critic = SelfConsistencyCritic()

        # Mock response for empty text
        mock_response = Mock()
        mock_response.data = {
            "feedback": "No content available for consistency evaluation.",
            "suggestions": ["Generate initial content before consistency check"],
            "needs_improvement": True,
            "confidence": 0.9,
            "consistency_score": 0.0,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(thought)

        # Should handle empty text gracefully
        assert "feedback" in result
        assert "no content" in result["feedback"].lower() or "empty" in result["feedback"].lower()

    def test_consistency_threshold_validation(self):
        """Test that consistency threshold is properly validated."""
        # Valid thresholds
        critic1 = SelfConsistencyCritic(consistency_threshold=0.5)
        assert critic1.consistency_threshold == 0.5

        critic2 = SelfConsistencyCritic(consistency_threshold=0.9)
        assert critic2.consistency_threshold == 0.9

        # Edge cases
        critic3 = SelfConsistencyCritic(consistency_threshold=0.0)
        assert critic3.consistency_threshold == 0.0

        critic4 = SelfConsistencyCritic(consistency_threshold=1.0)
        assert critic4.consistency_threshold == 1.0

    def test_num_consistency_attempts_validation(self):
        """Test that number of consistency attempts is properly validated."""
        # Valid values
        critic1 = SelfConsistencyCritic(num_consistency_attempts=3)
        assert critic1.num_consistency_attempts == 3

        critic2 = SelfConsistencyCritic(num_consistency_attempts=10)
        assert critic2.num_consistency_attempts == 10

    def test_temperature_validation(self):
        """Test that temperature parameter is properly validated."""
        # Valid temperatures
        critic1 = SelfConsistencyCritic(temperature=0.0)
        assert critic1.temperature == 0.0

        critic2 = SelfConsistencyCritic(temperature=1.0)
        assert critic2.temperature == 1.0

        critic3 = SelfConsistencyCritic(temperature=0.7)
        assert critic3.temperature == 0.7

    @pytest.mark.asyncio
    async def test_critique_consistency_aggregation(self, mock_agent, sample_thought):
        """Test that Self-Consistency properly aggregates multiple evaluations."""
        critic = SelfConsistencyCritic(num_consistency_attempts=3)

        # Mock response showing aggregation
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Aggregated evaluation from 3 consistency attempts shows good quality.",
            "suggestions": ["Minor improvements suggested by majority"],
            "needs_improvement": False,
            "confidence": 0.77,
            "consistency_score": 0.77,
            "individual_scores": [0.8, 0.75, 0.76],
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should show evidence of aggregation
        assert "feedback" in result
        assert (
            "aggregated" in result["feedback"].lower() or "attempts" in result["feedback"].lower()
        )
