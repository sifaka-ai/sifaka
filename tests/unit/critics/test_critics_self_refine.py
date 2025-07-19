"""Tests for Self-Refine critic."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.self_refine import (
    RefinementArea,
    SelfRefineCritic,
    SelfRefineResponse,
)


class TestRefinementArea:
    """Test the RefinementArea model."""

    def test_creation(self):
        """Test creating a refinement area."""
        area = RefinementArea(
            target_state="Simple, clear language instead of complex jargon"
        )
        assert "Simple, clear" in area.target_state

    def test_simple_creation(self):
        """Test simple creation."""
        area = RefinementArea(target_state="Well-structured instead of disorganized")
        assert "Well-structured" in area.target_state


class TestSelfRefineResponse:
    """Test the SelfRefineResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = SelfRefineResponse(
            feedback="Text needs clarity improvements",
            suggestions=["Simplify language"],
            needs_improvement=True,
        )
        assert response.feedback == "Text needs clarity improvements"
        assert response.confidence == 0.75  # default
        assert len(response.refinement_areas) == 0  # default empty list

    def test_creation_full(self):
        """Test creating response with all fields."""
        area1 = RefinementArea(target_state="Compelling opening instead of weak hook")
        area2 = RefinementArea(target_state="Strong closing instead of abrupt ending")

        response = SelfRefineResponse(
            feedback="Multiple areas need refinement",
            suggestions=["Rewrite introduction", "Strengthen conclusion"],
            needs_improvement=True,
            confidence=0.85,
            refinement_areas=[area1, area2],
        )

        assert len(response.refinement_areas) == 2
        assert response.confidence == 0.85

    def test_validation_bounds(self):
        """Test field validation bounds."""
        response = SelfRefineResponse(
            feedback="Test",
            suggestions=[],
            needs_improvement=False,
            confidence=1.0,
        )
        assert response.confidence == 1.0  # max


class TestSelfRefineCritic:
    """Test the SelfRefineCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        critic = SelfRefineCritic()
        assert critic.name == "self_refine"
        assert critic.model == "gpt-3.5-turbo"  # Default from Config
        assert critic.temperature == 0.7

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.llm.temperature = 0.3
        critic = SelfRefineCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = SelfRefineCritic(model="gpt-4", temperature=0.7, api_key="test-key")
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.7

    def test_get_response_type(self):
        """Test that critic uses SelfRefineResponse."""
        critic = SelfRefineCritic()
        assert critic._get_response_type() == SelfRefineResponse

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation for self-refine evaluation."""
        critic = SelfRefineCritic()
        messages = await critic._create_messages("Test text to refine", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Self-Refine critic" in messages[0]["content"]
        assert "text quality" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Test text to refine" in user_content
        assert "Clarity" in user_content
        assert "Completeness" in user_content
        assert "Coherence" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique from the same critic
        sample_result.critiques.append(
            CritiqueResult(
                critic="self_refine",
                feedback="Previous refinement feedback",
                suggestions=["Previous suggestion"],
            )
        )

        critic = SelfRefineCritic()
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous refinement feedback" in user_content

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_result):
        """Test successful critique flow."""
        critic = SelfRefineCritic()

        # Mock the LLM response
        mock_response = SelfRefineResponse(
            feedback="The text needs refinement in clarity and structure",
            suggestions=[
                "Simplify the opening paragraph",
                "Add transitions between sections",
            ],
            needs_improvement=True,
            confidence=0.8,
            refinement_areas=[
                RefinementArea(
                    target_state="Clear and engaging introduction instead of dense and unclear"
                ),
                RefinementArea(
                    target_state="Smooth flow between ideas instead of abrupt topic changes"
                ),
            ],
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        # Mock usage as a callable that returns an object with total_tokens
        mock_usage = Mock()
        mock_usage.total_tokens = 100
        mock_agent_result.usage = Mock(return_value=mock_usage)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "self_refine"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.8
            assert "refinement_areas" in result.metadata
            assert len(result.metadata["refinement_areas"]) == 2
            assert (
                "analysis_depth" not in result.metadata
            )  # metadata was removed from response

    @pytest.mark.asyncio
    async def test_critique_no_improvement_needed(self, sample_result):
        """Test critique when no refinement is needed."""
        critic = SelfRefineCritic()

        mock_response = SelfRefineResponse(
            feedback="The text is well-written and polished",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            refinement_areas=[],
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        # Mock usage as a callable that returns an object with total_tokens
        mock_usage = Mock()
        mock_usage.total_tokens = 100
        mock_agent_result.usage = Mock(return_value=mock_usage)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Excellent text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence > 0.9
            assert len(result.metadata["refinement_areas"]) == 0

    @pytest.mark.asyncio
    async def test_critique_major_refinement(self, sample_result):
        """Test critique requiring major refinement."""
        critic = SelfRefineCritic()

        mock_response = SelfRefineResponse(
            feedback="Significant refinement needed across multiple dimensions",
            suggestions=[
                "Complete restructuring required",
                "Rewrite for clarity",
                "Add missing context",
                "Improve coherence",
            ],
            needs_improvement=True,
            confidence=0.9,
            refinement_areas=[
                RefinementArea(
                    target_state="Logical flow with clear sections instead of disorganized and confusing structure"
                ),
                RefinementArea(
                    target_state="Accessible language instead of dense technical jargon"
                ),
                RefinementArea(
                    target_state="Comprehensive coverage instead of missing key information"
                ),
            ],
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        # Mock usage as a callable that returns an object with total_tokens
        mock_usage = Mock()
        mock_usage.total_tokens = 100
        mock_agent_result.usage = Mock(return_value=mock_usage)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Poor quality text", sample_result)

            assert result.needs_improvement is True
            assert len(result.suggestions) == 4
            # Check that we have high priority areas
            assert len(result.suggestions) == 4
            # RefinementArea only has target_state field
            assert all(
                "target_state" in area for area in result.metadata["refinement_areas"]
            )

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = SelfRefineCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = SelfRefineCritic(provider="openai")
        assert critic.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_critique_with_specific_areas(self, sample_result):
        """Test critique focusing on specific refinement areas."""
        critic = SelfRefineCritic()

        mock_response = SelfRefineResponse(
            feedback="Focus on improving engagement and flow",
            suggestions=["Add compelling examples", "Improve paragraph transitions"],
            needs_improvement=True,
            confidence=0.85,
            refinement_areas=[
                RefinementArea(
                    target_state="Engaging and relatable instead of dry and academic"
                ),
                RefinementArea(
                    target_state="Smooth narrative flow instead of choppy transitions"
                ),
            ],
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        # Mock usage as a callable that returns an object with total_tokens
        mock_usage = Mock()
        mock_usage.total_tokens = 100
        mock_agent_result.usage = Mock(return_value=mock_usage)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Academic text", sample_result)

            assert len(result.metadata["refinement_areas"]) == 2
            # RefinementArea only has target_state field
            areas = result.metadata["refinement_areas"]
            target_states = [area["target_state"] for area in areas]

            # Check that we have the expected refinement areas
            assert any("Engaging and relatable" in state for state in target_states)
            assert any("Smooth narrative flow" in state for state in target_states)
