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
            area="clarity",
            current_state="Uses complex jargon",
            target_state="Simple, clear language",
            priority="high",
        )
        assert area.area == "clarity"
        assert "complex jargon" in area.current_state
        assert "Simple, clear" in area.target_state
        assert area.priority == "high"

    def test_default_priority(self):
        """Test default priority."""
        area = RefinementArea(
            area="structure",
            current_state="Disorganized",
            target_state="Well-structured",
        )
        assert area.priority == "medium"


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
        assert response.quality_score == 0.7  # default
        assert response.refinement_iterations_recommended == 1  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        area1 = RefinementArea(
            area="introduction",
            current_state="Weak hook",
            target_state="Compelling opening",
            priority="high",
        )
        area2 = RefinementArea(
            area="conclusion",
            current_state="Abrupt ending",
            target_state="Strong closing",
            priority="medium",
        )

        response = SelfRefineResponse(
            feedback="Multiple areas need refinement",
            suggestions=["Rewrite introduction", "Strengthen conclusion"],
            needs_improvement=True,
            confidence=0.85,
            refinement_areas=[area1, area2],
            quality_score=0.6,
            refinement_iterations_recommended=2,
            metadata={"word_count": 500},
        )

        assert len(response.refinement_areas) == 2
        assert response.quality_score == 0.6
        assert response.refinement_iterations_recommended == 2
        assert response.metadata["word_count"] == 500

    def test_validation_bounds(self):
        """Test field validation bounds."""
        response = SelfRefineResponse(
            feedback="Test",
            suggestions=[],
            needs_improvement=False,
            confidence=1.0,
            quality_score=0.0,
            refinement_iterations_recommended=5,
        )
        assert response.confidence == 1.0  # max
        assert response.quality_score == 0.0  # min
        assert response.refinement_iterations_recommended == 5  # max


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
        assert critic.model == "gpt-4o-mini"
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
                    area="introduction",
                    current_state="Dense and unclear",
                    target_state="Clear and engaging",
                    priority="high",
                ),
                RefinementArea(
                    area="transitions",
                    current_state="Abrupt topic changes",
                    target_state="Smooth flow between ideas",
                    priority="medium",
                ),
            ],
            quality_score=0.65,
            refinement_iterations_recommended=2,
            metadata={"analysis_depth": "detailed"},
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

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
            assert result.metadata["quality_score"] == 0.65
            assert result.metadata["refinement_iterations_recommended"] == 2

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
            quality_score=0.9,
            refinement_iterations_recommended=0,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Excellent text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence > 0.9
            assert result.metadata["quality_score"] == 0.9
            assert result.metadata["refinement_iterations_recommended"] == 0

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
                    area="overall_structure",
                    current_state="Disorganized and confusing",
                    target_state="Logical flow with clear sections",
                    priority="high",
                ),
                RefinementArea(
                    area="clarity",
                    current_state="Dense technical jargon",
                    target_state="Accessible language",
                    priority="high",
                ),
                RefinementArea(
                    area="completeness",
                    current_state="Missing key information",
                    target_state="Comprehensive coverage",
                    priority="high",
                ),
            ],
            quality_score=0.3,
            refinement_iterations_recommended=4,
            metadata={"severity": "major"},
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Poor quality text", sample_result)

            assert result.needs_improvement is True
            assert len(result.suggestions) == 4
            assert result.metadata["quality_score"] == 0.3
            assert result.metadata["refinement_iterations_recommended"] == 4
            assert all(
                area["priority"] == "high"
                for area in result.metadata["refinement_areas"]
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
                    area="engagement",
                    current_state="Dry and academic",
                    target_state="Engaging and relatable",
                    priority="high",
                ),
                RefinementArea(
                    area="flow",
                    current_state="Choppy transitions",
                    target_state="Smooth narrative flow",
                    priority="medium",
                ),
            ],
            quality_score=0.7,
            refinement_iterations_recommended=1,
            metadata={"focus_areas": ["engagement", "flow"]},
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Academic text", sample_result)

            assert len(result.metadata["refinement_areas"]) == 2
            areas = [area["area"] for area in result.metadata["refinement_areas"]]
            assert "engagement" in areas
            assert "flow" in areas
