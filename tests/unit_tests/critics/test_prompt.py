"""Comprehensive unit tests for PromptCritic.

This module tests the PromptCritic implementation:
- Configurable prompt-based critique
- Custom evaluation criteria
- Integration with PydanticAI agents
- Error handling and edge cases

Tests cover:
- Basic critique functionality
- Custom prompt configuration
- Feedback quality and structure
- Performance characteristics
- Mock-based testing without external API calls
"""

from unittest.mock import Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.critics.prompt import PromptCritic


class TestPromptCritic:
    """Test suite for PromptCritic class."""

    def test_prompt_critic_creation_minimal(self):
        """Test creating PromptCritic with minimal parameters."""
        critic = PromptCritic()

        assert critic.model_name == "mock"
        assert "evaluate" in critic.critique_prompt_template.lower()
        assert critic.metadata["critic_type"] == "PromptCritic"

    def test_prompt_critic_creation_with_custom_prompt(self):
        """Test creating PromptCritic with custom prompt template."""
        custom_prompt = "Please analyze this text for clarity and coherence: {text}"

        critic = PromptCritic(critique_prompt_template=custom_prompt, model_name="test-model")

        assert critic.critique_prompt_template == custom_prompt
        assert critic.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_critique_async_basic(self):
        """Test basic critique functionality."""
        critic = PromptCritic(model_name="mock")

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is a test response that needs evaluation.",
            iteration=1,
            max_iterations=3,
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "The text is clear and well-structured. Score: 8/10"

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        # Check that critique was added to thought
        assert len(thought.critiques) == 1
        critique = thought.critiques[0]
        assert critique.critic_name == "PromptCritic"
        assert "clear and well-structured" in critique.feedback

    @pytest.mark.asyncio
    async def test_critique_async_with_custom_criteria(self):
        """Test critique with custom evaluation criteria."""
        custom_prompt = """
        Evaluate this text for the following criteria:
        1. Technical accuracy
        2. Readability
        3. Completeness

        Text: {text}

        Provide specific feedback for each criterion.
        """

        critic = PromptCritic(critique_prompt_template=custom_prompt, model_name="mock")

        thought = SifakaThought(
            prompt="Explain machine learning",
            final_text="Machine learning is a subset of AI that uses algorithms to learn patterns.",
            iteration=1,
            max_iterations=3,
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = """
        1. Technical accuracy: Good - definition is correct
        2. Readability: Excellent - clear and concise
        3. Completeness: Fair - could use more examples
        """

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert "Technical accuracy" in critique.feedback
        assert "Readability" in critique.feedback
        assert "Completeness" in critique.feedback

    @pytest.mark.asyncio
    async def test_critique_async_empty_text(self):
        """Test critique with empty text."""
        critic = PromptCritic(model_name="mock")

        thought = SifakaThought(prompt="Test prompt", final_text="", iteration=1, max_iterations=3)

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "No text provided for evaluation."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert "No text provided" in critique.feedback

    @pytest.mark.asyncio
    async def test_critique_async_none_text(self):
        """Test critique with None text."""
        critic = PromptCritic(model_name="mock")

        thought = SifakaThought(
            prompt="Test prompt", final_text=None, iteration=1, max_iterations=3
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "No text available for critique."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert critique.feedback is not None

    @pytest.mark.asyncio
    async def test_critique_async_with_scoring(self):
        """Test critique that includes numerical scoring."""
        scoring_prompt = """
        Evaluate this text and provide:
        1. A score from 1-10
        2. Specific feedback

        Text: {text}

        Format: Score: X/10 - Feedback here
        """

        critic = PromptCritic(critique_prompt_template=scoring_prompt, model_name="mock")

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is a well-written and informative response.",
            iteration=1,
            max_iterations=3,
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "Score: 9/10 - Excellent clarity and information density."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert "9/10" in critique.feedback
        assert "Excellent clarity" in critique.feedback

    @pytest.mark.asyncio
    async def test_critique_async_agent_error(self):
        """Test critique when agent raises an error."""
        critic = PromptCritic(model_name="mock")

        thought = SifakaThought(
            prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
        )

        # Mock agent to raise an exception
        with patch.object(critic.agent, "run_sync", side_effect=Exception("Agent error")):
            await critic.critique_async(thought)

        # Should still add a critique with error information
        assert len(thought.critiques) == 1
        critique = thought.critiques[0]
        assert "error" in critique.feedback.lower()

    @pytest.mark.asyncio
    async def test_critique_async_timing(self):
        """Test that critique includes timing information."""
        critic = PromptCritic(model_name="mock")

        thought = SifakaThought(
            prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "Good response quality."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert critique.critique_time > 0

    @pytest.mark.asyncio
    async def test_critique_async_multiple_iterations(self):
        """Test critique across multiple iterations."""
        critic = PromptCritic(model_name="mock")

        # First iteration
        thought = SifakaThought(
            prompt="Test prompt", final_text="Initial response", iteration=1, max_iterations=3
        )

        mock_response1 = Mock()
        mock_response1.data = "Initial response needs improvement."

        with patch.object(critic.agent, "run_sync", return_value=mock_response1):
            await critic.critique_async(thought)

        # Second iteration
        thought.final_text = "Improved response with more detail"
        thought.iteration = 2

        mock_response2 = Mock()
        mock_response2.data = "Much better - good improvement."

        with patch.object(critic.agent, "run_sync", return_value=mock_response2):
            await critic.critique_async(thought)

        # Should have critiques from both iterations
        assert len(thought.critiques) == 2
        assert "needs improvement" in thought.critiques[0].feedback
        assert "Much better" in thought.critiques[1].feedback

    def test_prompt_critic_repr(self):
        """Test PromptCritic string representation."""
        critic = PromptCritic(model_name="test-model")

        repr_str = repr(critic)
        assert "PromptCritic" in repr_str
        assert "test-model" in repr_str

    @pytest.mark.asyncio
    async def test_critique_async_long_text(self):
        """Test critique with very long text."""
        critic = PromptCritic(model_name="mock")

        # Create long text
        long_text = "This is a test sentence. " * 100

        thought = SifakaThought(
            prompt="Test prompt", final_text=long_text, iteration=1, max_iterations=3
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "Text is repetitive but grammatically correct."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert "repetitive" in critique.feedback

    @pytest.mark.asyncio
    async def test_critique_async_special_characters(self):
        """Test critique with special characters and unicode."""
        critic = PromptCritic(model_name="mock")

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="Text with émojis 🚀 and spëcial chàracters!",
            iteration=1,
            max_iterations=3,
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "Text contains special characters and emojis appropriately."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert "special characters" in critique.feedback

    @pytest.mark.asyncio
    async def test_critique_async_with_context(self):
        """Test critique that considers the original prompt context."""
        context_prompt = """
        Evaluate how well this response answers the original question.

        Original question: {prompt}
        Response: {text}

        Consider relevance, completeness, and accuracy.
        """

        critic = PromptCritic(critique_prompt_template=context_prompt, model_name="mock")

        thought = SifakaThought(
            prompt="What are the benefits of renewable energy?",
            final_text="Renewable energy reduces carbon emissions and is sustainable.",
            iteration=1,
            max_iterations=3,
        )

        # Mock the agent response
        mock_response = Mock()
        mock_response.data = "Response directly addresses the question with key benefits mentioned."

        with patch.object(critic.agent, "run_sync", return_value=mock_response):
            await critic.critique_async(thought)

        critique = thought.critiques[0]
        assert "directly addresses" in critique.feedback
        assert "benefits" in critique.feedback
