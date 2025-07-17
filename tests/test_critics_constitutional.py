"""Tests for Constitutional AI critic."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.constitutional import (
    DEFAULT_PRINCIPLES,
    ConstitutionalCritic,
    ConstitutionalResponse,
)


class TestConstitutionalResponse:
    """Test the ConstitutionalResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = ConstitutionalResponse(
            feedback="Text is generally good", suggestions=[], needs_improvement=False
        )
        assert response.feedback == "Text is generally good"
        assert response.needs_improvement is False
        assert response.confidence == 0.8  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        response = ConstitutionalResponse(
            feedback="Detailed feedback",
            suggestions=["Suggestion 1"],
            needs_improvement=True,
            confidence=0.9,
        )

        assert response.feedback == "Detailed feedback"
        assert len(response.suggestions) == 1
        assert response.needs_improvement is True
        assert response.confidence == 0.9


class TestConstitutionalCritic:
    """Test the ConstitutionalCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        critic = ConstitutionalCritic()
        assert critic.name == "constitutional"
        assert critic.principles == DEFAULT_PRINCIPLES
        assert len(critic.principles) == 6

    def test_initialization_custom_principles(self):
        """Test initialization with custom principles."""
        custom_principles = ["Be concise", "Use examples"]
        critic = ConstitutionalCritic(principles=custom_principles)
        assert critic.principles == custom_principles
        assert len(critic.principles) == 2

    def test_initialization_with_config_principles(self):
        """Test initialization with principles from config."""
        config = Config()
        config.constitutional_principles = ["Config principle 1", "Config principle 2"]
        critic = ConstitutionalCritic(config=config)
        assert critic.principles == config.constitutional_principles

    def test_initialization_priority(self):
        """Test priority: custom > config > default."""
        config = Config()
        config.constitutional_principles = ["From config"]
        custom = ["From custom"]

        # Custom takes precedence
        critic = ConstitutionalCritic(config=config, principles=custom)
        assert critic.principles == custom

        # Config takes precedence over default
        critic = ConstitutionalCritic(config=config)
        assert critic.principles == ["From config"]

    def test_get_response_type(self):
        """Test that critic uses ConstitutionalResponse."""
        critic = ConstitutionalCritic()
        assert critic._get_response_type() == ConstitutionalResponse

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation for constitutional evaluation."""
        critic = ConstitutionalCritic(principles=["Principle 1", "Principle 2"])
        messages = await critic._create_messages("Test text", sample_result)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        user_content = messages[0]["content"]
        assert "1. Principle 1" in user_content
        assert "2. Principle 2" in user_content
        assert "Evaluate the following text" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique from the same critic
        sample_result.critiques.append(
            CritiqueResult(
                critic="constitutional",
                feedback="Previous feedback",
                suggestions=["Previous suggestion"],
            )
        )

        critic = ConstitutionalCritic()
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[0]["content"]
        assert "Test text" in user_content

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_result):
        """Test successful critique flow."""
        critic = ConstitutionalCritic()

        # Mock the LLM response
        mock_response = ConstitutionalResponse(
            feedback="Text follows constitutional principles well",
            suggestions=["Minor improvement: Add sources"],
            needs_improvement=True,
            confidence=0.85,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "constitutional"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 1
            assert result.needs_improvement is True
            assert result.confidence == 0.85
            assert "principles_used" in result.metadata

    @pytest.mark.asyncio
    async def test_critique_with_revision_proposals(self, sample_result):
        """Test critique with revision proposals."""
        critic = ConstitutionalCritic()

        mock_response = ConstitutionalResponse(
            feedback="Text needs constitutional revisions",
            suggestions=["Apply proposed revisions"],
            needs_improvement=True,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.needs_improvement is True
            assert "principles_used" in result.metadata

    @pytest.mark.asyncio
    async def test_critique_all_principles_pass(self, sample_result):
        """Test when all principles pass."""
        critic = ConstitutionalCritic(principles=["Be helpful", "Be clear"])

        mock_response = ConstitutionalResponse(
            feedback="Text adheres to all constitutional principles",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        mock_agent_result.data = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence > 0.9

    def test_model_parameters(self):
        """Test model parameter configuration."""
        critic = ConstitutionalCritic(model="gpt-4", temperature=0.5)
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.5

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = ConstitutionalCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = ConstitutionalCritic(provider="openai")
        assert critic.provider == Provider.OPENAI
