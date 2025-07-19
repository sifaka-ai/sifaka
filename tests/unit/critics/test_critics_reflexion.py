"""Tests for Reflexion critic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, Generation, SifakaResult
from sifaka.critics.reflexion import ReflexionCritic


class MockAgentResult:
    """Mock result from PydanticAI agent run."""

    def __init__(self, output):
        self.output = output
        self._usage = MagicMock()
        self._usage.total_tokens = 100

    def usage(self):
        """Mock usage data."""
        return self._usage


@pytest.fixture
def mock_pydantic_agent():
    """Create a mock PydanticAI agent."""
    with patch("sifaka.core.llm_client.LLMClient.create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_create.return_value = mock_agent
        yield mock_agent


class TestReflexionCritic:
    """Test the ReflexionCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    @pytest.fixture
    def result_with_history(self):
        """Create a SifakaResult with critique history."""
        result = SifakaResult(
            original_text="Original text", final_text="Current text version"
        )

        # Add some critiques
        result.critiques.append(
            CritiqueResult(
                critic="reflexion",
                feedback="Initial feedback: needs more clarity",
                suggestions=["Simplify opening"],
                needs_improvement=True,
                confidence=0.6,
            )
        )
        result.critiques.append(
            CritiqueResult(
                critic="reflexion",
                feedback="Better, but still needs work on structure",
                suggestions=["Reorganize middle section"],
                needs_improvement=True,
                confidence=0.7,
            )
        )

        # Add some generations
        result.generations.append(
            Generation(text="First version", model="gpt-4o-mini", iteration=1)
        )
        result.generations.append(
            Generation(text="Second version", model="gpt-4o-mini", iteration=2)
        )

        return result

    def test_initialization_default(self):
        """Test default initialization."""
        critic = ReflexionCritic()
        assert critic.name == "reflexion"
        # Model defaults come from config
        assert critic.model is not None
        assert critic.temperature is not None

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.llm.temperature = 0.5
        critic = ReflexionCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = ReflexionCritic(model="gpt-4", temperature=0.8, api_key="test-key")
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.8

    def test_get_response_type(self):
        """Test that critic uses response type."""
        critic = ReflexionCritic()
        # ReflexionCritic has its own response type
        from sifaka.critics.reflexion import ReflexionResponse

        response_type = critic._get_response_type()
        assert response_type == ReflexionResponse

    def test_get_system_prompt(self):
        """Test system prompt generation."""
        critic = ReflexionCritic()
        prompt = critic._get_system_prompt()
        # Should mention reflexion technique
        assert "reflexion" in prompt.lower()
        assert "iterative" in prompt.lower()

    @pytest.mark.asyncio
    async def test_create_messages_first_iteration(self, sample_result):
        """Test message creation for first iteration."""
        critic = ReflexionCritic()
        messages = await critic._create_messages("Test text", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        # System message should mention reflexion
        assert "reflexion" in messages[0]["content"].lower()

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Test text" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_history(self, result_with_history):
        """Test message creation with history."""
        critic = ReflexionCritic()
        messages = await critic._create_messages("Current text", result_with_history)

        user_content = messages[1]["content"]
        assert "Current text" in user_content
        # History is incorporated internally

    @pytest.mark.asyncio
    async def test_critique_success_first_iteration(
        self, sample_result, mock_pydantic_agent
    ):
        """Test successful critique on first iteration."""
        critic = ReflexionCritic()

        # Import the response type
        from sifaka.critics.reflexion import ReflexionResponse

        # Mock the LLM response
        mock_response = ReflexionResponse(
            feedback="The text lacks clarity and structure",
            suggestions=["Add topic sentences", "Improve transitions"],
            needs_improvement=True,
            confidence=0.6,
        )

        # Set up the mock
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Test text", sample_result)

        assert result.critic == "reflexion"
        assert result.feedback == mock_response.feedback
        assert len(result.suggestions) == 2
        assert result.needs_improvement is True
        assert result.confidence == 0.6

    @pytest.mark.asyncio
    async def test_critique_with_improvement_trajectory(
        self, result_with_history, mock_pydantic_agent
    ):
        """Test critique showing improvement over iterations."""
        critic = ReflexionCritic()

        from sifaka.critics.reflexion import ReflexionResponse

        mock_response = ReflexionResponse(
            feedback="Significant progress - structure much improved",
            suggestions=["Minor polish on conclusion"],
            needs_improvement=True,
            confidence=0.9,
        )

        # Set up the mock
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Improved text", result_with_history)

        assert result.confidence == 0.9  # Higher confidence
        assert "Significant progress" in result.feedback

    @pytest.mark.asyncio
    async def test_critique_no_improvement_needed(
        self, result_with_history, mock_pydantic_agent
    ):
        """Test critique when text no longer needs improvement."""
        critic = ReflexionCritic()

        from sifaka.critics.reflexion import ReflexionResponse

        mock_response = ReflexionResponse(
            feedback="Text has reached high quality through iterations",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
        )

        # Set up the mock
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        result = await critic.critique("Final text", result_with_history)

        assert result.needs_improvement is False
        assert len(result.suggestions) == 0
        assert result.confidence == 0.95

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = ReflexionCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = ReflexionCritic(provider="openai")
        assert critic.provider == "openai"

    @pytest.mark.asyncio
    async def test_reflexion_learning_pattern(self, mock_pydantic_agent):
        """Test that reflexion shows learning across iterations."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Original", final_text="Current")

        from sifaka.critics.reflexion import ReflexionResponse

        # Simulate 3 iterations with improving confidence
        confidences = [0.5, 0.7, 0.9]

        # Create all responses
        responses = []
        for i, conf in enumerate(confidences):
            mock_response = ReflexionResponse(
                feedback=f"Iteration {i+1} feedback",
                suggestions=[f"Suggestion {i+1}"] if i < 2 else [],
                needs_improvement=i < 2,
                confidence=conf,
            )
            responses.append(MockAgentResult(mock_response))

        # Set up mock to return different responses
        mock_pydantic_agent.run.side_effect = responses

        for i, conf in enumerate(confidences):
            critique = await critic.critique(f"Text v{i+1}", result)
            # Confidence might be adjusted by internal calculation
            # Check it's reasonably close to the expected value
            assert abs(critique.confidence - conf) < 0.1
            result.critiques.append(critique)

        # Verify confidence improved over iterations
        assert result.critiques[0].confidence < result.critiques[1].confidence
        assert result.critiques[1].confidence < result.critiques[2].confidence
