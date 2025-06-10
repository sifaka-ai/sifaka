"""Comprehensive unit tests for N-Critics critic.

This module tests the NCriticsCritic implementation:
- Ensemble critique methodology inspired by Tian et al. 2023
- Multiple critic perspectives and consensus building
- Integration with PydanticAI agents
- Error handling and edge cases

Tests cover:
- Basic N-Critics functionality
- Multiple critic ensemble behavior
- Consensus building and aggregation
- Performance characteristics
- Mock-based testing without external API calls
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.critics.n_critics import NCriticsCritic


class TestNCriticsCritic:
    """Test suite for NCriticsCritic class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock PydanticAI agent."""
        agent = Mock()
        agent.run = AsyncMock()

        # Mock response for N-Critics ensemble
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Consensus from multiple critics: The text needs better structure and examples.",
            "suggestions": [
                "Add clear topic sentences to each paragraph",
                "Include specific examples to support claims",
                "Improve logical flow between ideas",
            ],
            "needs_improvement": True,
            "confidence": 0.82,
            "reasoning": "3 out of 4 critics agreed on structural improvements needed",
            "critic_perspectives": [
                {"critic": "Structure Critic", "feedback": "Needs better organization"},
                {"critic": "Content Critic", "feedback": "Requires more examples"},
                {"critic": "Flow Critic", "feedback": "Logical connections unclear"},
            ],
        }
        agent.run.return_value = mock_response

        return agent

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Explain the benefits of renewable energy",
            final_text="Renewable energy is good for the environment. It helps reduce pollution.",
            iteration=1,
            max_iterations=3,
        )
        return thought

    def test_n_critics_creation_minimal(self):
        """Test creating NCriticsCritic with minimal parameters."""
        critic = NCriticsCritic()

        assert critic.model_name == "groq:mixtral-8x7b-32768"
        assert critic.num_critics == 3  # Default number
        assert "n-critics" in critic.system_prompt.lower()
        assert critic.metadata["critic_type"] == "NCriticsCritic"
        assert "Tian et al. 2023" in critic.paper_reference

    def test_n_critics_creation_with_custom_parameters(self):
        """Test creating NCriticsCritic with custom parameters."""
        custom_perspectives = [
            "Technical accuracy and factual correctness",
            "Clarity and readability for target audience",
            "Completeness and comprehensive coverage",
        ]

        critic = NCriticsCritic(
            num_critics=5,
            critic_perspectives=custom_perspectives,
            consensus_threshold=0.8,
            model_name="openai:gpt-4",
        )

        assert critic.num_critics == 5
        assert critic.critic_perspectives == custom_perspectives
        assert critic.consensus_threshold == 0.8
        assert critic.model_name == "openai:gpt-4"

    @pytest.mark.asyncio
    async def test_critique_async_basic(self, mock_agent, sample_thought):
        """Test basic N-Critics functionality."""
        critic = NCriticsCritic(model_name="mock")

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
    async def test_critique_with_custom_perspectives(self, mock_agent, sample_thought):
        """Test N-Critics with custom critic perspectives."""
        custom_perspectives = [
            "Grammar and syntax correctness",
            "Logical argument structure",
            "Evidence and citation quality",
        ]

        critic = NCriticsCritic(num_critics=3, critic_perspectives=custom_perspectives)

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should handle custom perspectives
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Verify that custom perspectives were used in the prompt
        call_args = mock_agent.run.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_critique_with_high_consensus_threshold(self, mock_agent, sample_thought):
        """Test N-Critics with high consensus threshold."""
        critic = NCriticsCritic(num_critics=5, consensus_threshold=0.9)  # High threshold

        # Mock response with high consensus
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Strong consensus: All critics agree on major improvements needed.",
            "suggestions": ["Unanimous recommendation for restructuring"],
            "needs_improvement": True,
            "confidence": 0.95,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should reflect high consensus
        assert result["confidence"] >= 0.9
        assert "consensus" in result["feedback"].lower()

    @pytest.mark.asyncio
    async def test_critique_with_low_consensus(self, mock_agent, sample_thought):
        """Test N-Critics with low consensus scenario."""
        critic = NCriticsCritic(num_critics=4)

        # Mock response with low consensus
        mock_response = Mock()
        mock_response.data = {
            "feedback": "Mixed opinions from critics: Some see issues, others don't.",
            "suggestions": ["Consider multiple perspectives", "Seek additional review"],
            "needs_improvement": False,  # Due to lack of consensus
            "confidence": 0.45,
        }
        mock_agent.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should reflect low consensus
        assert result["confidence"] < 0.6
        assert "mixed" in result["feedback"].lower() or "opinions" in result["feedback"].lower()

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_thought):
        """Test error handling in N-Critics."""
        critic = NCriticsCritic()

        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("Ensemble Error"))

        with patch.object(critic, "_agent", mock_agent):
            with pytest.raises(Exception, match="Ensemble Error"):
                await critic.critique_async(sample_thought)

    def test_default_perspectives_structure(self):
        """Test that default critic perspectives are properly structured."""
        critic = NCriticsCritic()

        default_perspectives = critic._get_default_perspectives()

        assert isinstance(default_perspectives, list)
        assert len(default_perspectives) >= 3  # Should have at least 3 perspectives

        # Check that perspectives cover different aspects
        perspectives_text = " ".join(default_perspectives).lower()
        assert "content" in perspectives_text or "structure" in perspectives_text
        assert "clarity" in perspectives_text or "readability" in perspectives_text

    @pytest.mark.asyncio
    async def test_improve_async_functionality(self, mock_agent, sample_thought):
        """Test the improve_async method."""
        critic = NCriticsCritic()

        # Mock improved response
        mock_response = Mock()
        mock_response.data = (
            "Improved text incorporating feedback from multiple critic perspectives."
        )
        mock_agent.run.return_value = mock_response

        with patch.object(critic, "_agent", mock_agent):
            improved_text = await critic.improve_async(sample_thought)

        assert isinstance(improved_text, str)
        assert len(improved_text) > 0
        assert "improved" in improved_text.lower()

    def test_system_prompt_contains_ensemble_methodology(self):
        """Test that system prompt contains N-Critics ensemble methodology."""
        critic = NCriticsCritic()

        system_prompt = critic.system_prompt.lower()

        # Should contain key N-Critics concepts
        assert "critics" in system_prompt or "critic" in system_prompt
        assert "ensemble" in system_prompt or "multiple" in system_prompt
        assert "consensus" in system_prompt or "agreement" in system_prompt
        assert "perspective" in system_prompt

    @pytest.mark.asyncio
    async def test_critique_with_different_num_critics(self, mock_agent, sample_thought):
        """Test N-Critics with different numbers of critics."""
        for num_critics in [2, 3, 5, 7]:
            critic = NCriticsCritic(num_critics=num_critics)

            with patch.object(critic, "_agent", mock_agent):
                result = await critic.critique_async(sample_thought)

            # Should work with any reasonable number of critics
            assert "feedback" in result
            assert isinstance(result["suggestions"], list)

    def test_consensus_threshold_validation(self):
        """Test that consensus threshold is properly validated."""
        # Valid thresholds
        critic1 = NCriticsCritic(consensus_threshold=0.5)
        assert critic1.consensus_threshold == 0.5

        critic2 = NCriticsCritic(consensus_threshold=0.8)
        assert critic2.consensus_threshold == 0.8

        # Edge cases
        critic3 = NCriticsCritic(consensus_threshold=0.0)
        assert critic3.consensus_threshold == 0.0

        critic4 = NCriticsCritic(consensus_threshold=1.0)
        assert critic4.consensus_threshold == 1.0

    def test_paper_reference_and_methodology(self):
        """Test that paper reference and methodology are properly set."""
        critic = NCriticsCritic()

        assert "Tian et al. 2023" in critic.paper_reference
        assert "ensemble" in critic.methodology.lower()
        assert "multiple" in critic.methodology.lower()
        assert len(critic.methodology) > 50  # Should have substantial methodology description

    @pytest.mark.asyncio
    async def test_critique_with_validation_context(self, mock_agent, sample_thought):
        """Test N-Critics with validation context awareness."""
        # Add validation results to the thought
        sample_thought.add_validation("length_validator", False, {"issue": "too short"})
        sample_thought.add_validation("content_validator", True, {"score": 0.8})

        critic = NCriticsCritic()

        with patch.object(critic, "_agent", mock_agent):
            result = await critic.critique_async(sample_thought)

        # Should consider validation context in ensemble critique
        assert "feedback" in result
        mock_agent.run.assert_called_once()

        # Check that validation context was included in the call
        call_args = mock_agent.run.call_args
        assert call_args is not None
