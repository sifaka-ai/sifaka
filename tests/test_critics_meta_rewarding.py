"""Tests for Meta-Rewarding critic."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sifaka.critics.meta_rewarding import (
    MetaRewardingCritic,
    MetaRewardingResponse,
    CritiqueEvaluation,
    SuggestionPreference,
)
from sifaka.core.models import SifakaResult, CritiqueResult
from sifaka.core.config import Config


class TestCritiqueEvaluation:
    """Test the CritiqueEvaluation model."""

    def test_creation(self):
        """Test creating a critique evaluation."""
        eval = CritiqueEvaluation(
            aspect="accuracy",
            score=0.85,
            reasoning="The critique correctly identified key issues",
            improvement_needed=False
        )
        assert eval.aspect == "accuracy"
        assert eval.score == 0.85
        assert eval.improvement_needed is False

    def test_with_improvement_needed(self):
        """Test evaluation that needs improvement."""
        eval = CritiqueEvaluation(
            aspect="helpfulness",
            score=0.4,
            reasoning="Suggestions are too vague",
            improvement_needed=True
        )
        assert eval.score == 0.4
        assert eval.improvement_needed is True


class TestSuggestionPreference:
    """Test the SuggestionPreference model."""

    def test_creation(self):
        """Test creating a suggestion preference."""
        pref = SuggestionPreference(
            suggestion="Add specific examples",
            preference_score=0.9,
            rationale="Concrete examples enhance understanding"
        )
        assert pref.suggestion == "Add specific examples"
        assert pref.preference_score == 0.9
        assert "enhance understanding" in pref.rationale


class TestMetaRewardingResponse:
    """Test the MetaRewardingResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = MetaRewardingResponse(
            feedback="Refined feedback",
            initial_feedback="Initial feedback",
            meta_evaluation="The initial critique was good",
            refinement_rationale="Minor adjustments made",
            needs_improvement=True
        )
        assert response.feedback == "Refined feedback"
        assert response.initial_feedback == "Initial feedback"
        assert response.confidence == 0.7  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        eval = CritiqueEvaluation(
            aspect="accuracy",
            score=0.8,
            reasoning="Good accuracy",
            improvement_needed=False
        )
        pref = SuggestionPreference(
            suggestion="Test suggestion",
            preference_score=0.85,
            rationale="Well-reasoned"
        )
        
        response = MetaRewardingResponse(
            feedback="Final refined feedback",
            suggestions=["Suggestion 1", "Suggestion 2"],
            needs_improvement=True,
            confidence=0.9,
            initial_feedback="Initial critique",
            meta_evaluation="Meta-evaluation of critique",
            critique_evaluations=[eval],
            refinement_rationale="Improved based on meta-evaluation",
            suggestion_preferences=[pref],
            initial_quality_score=0.7,
            final_quality_score=0.9,
            improvement_delta=0.2,
            meta_reward=0.85,
            metadata={"iterations": 3}
        )
        
        assert len(response.critique_evaluations) == 1
        assert len(response.suggestion_preferences) == 1
        assert response.improvement_delta == 0.2
        assert response.meta_reward == 0.85
        assert response.metadata["iterations"] == 3


class TestMetaRewardingCritic:
    """Test the MetaRewardingCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text",
            final_text="Final text"
        )

    def test_initialization_default(self):
        """Test default initialization."""
        critic = MetaRewardingCritic()
        assert critic.name == "meta_rewarding"
        assert critic.model == "gpt-4o-mini"
        assert critic.temperature == 0.7

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.temperature = 0.5
        critic = MetaRewardingCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = MetaRewardingCritic(
            model="gpt-4",
            temperature=0.3,
            api_key="test-key"
        )
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.3

    def test_get_response_type(self):
        """Test that critic uses MetaRewardingResponse."""
        critic = MetaRewardingCritic()
        assert critic._get_response_type() == MetaRewardingResponse

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation for meta-rewarding evaluation."""
        critic = MetaRewardingCritic()
        messages = await critic._create_messages("Test text to evaluate", sample_result)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Meta-Rewarding critic" in messages[0]["content"]
        assert "three-stage evaluation" in messages[0]["content"]
        
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Meta-Rewarding framework" in user_content
        assert "STAGE 1 - Initial Critique:" in user_content
        assert "STAGE 2 - Meta-Evaluation:" in user_content
        assert "STAGE 3 - Refined Critique:" in user_content
        assert "improvement delta" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique from the same critic
        sample_result.critiques.append(
            CritiqueResult(
                critic="meta_rewarding",
                feedback="Previous meta-rewarding feedback",
                suggestions=["Previous suggestion"]
            )
        )
        
        critic = MetaRewardingCritic()
        messages = await critic._create_messages("Test text", sample_result)
        
        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous meta-rewarding feedback" in user_content

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_result):
        """Test successful critique flow."""
        critic = MetaRewardingCritic()
        
        # Mock the LLM response
        mock_response = MetaRewardingResponse(
            feedback="Refined: The text is well-structured but needs more examples",
            suggestions=["Add concrete examples", "Include data to support claims"],
            needs_improvement=True,
            confidence=0.85,
            initial_feedback="Initial: The text lacks examples",
            meta_evaluation="The initial critique was accurate but not specific enough",
            critique_evaluations=[
                CritiqueEvaluation(
                    aspect="accuracy",
                    score=0.8,
                    reasoning="Correctly identified missing examples",
                    improvement_needed=False
                ),
                CritiqueEvaluation(
                    aspect="specificity",
                    score=0.6,
                    reasoning="Could be more specific about what examples",
                    improvement_needed=True
                )
            ],
            refinement_rationale="Added specific suggestions for examples and data",
            suggestion_preferences=[
                SuggestionPreference(
                    suggestion="Add concrete examples",
                    preference_score=0.9,
                    rationale="Most impactful improvement"
                )
            ],
            initial_quality_score=0.7,
            final_quality_score=0.85,
            improvement_delta=0.15,
            meta_reward=0.8
        )
        
        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        
        with patch.object(critic.client, 'create_agent', return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)
            
            assert result.critic == "meta_rewarding"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.85
            assert "initial_feedback" in result.metadata
            assert "meta_evaluation" in result.metadata
            assert "improvement_delta" in result.metadata
            assert result.metadata["improvement_delta"] == 0.15
            assert result.metadata["meta_reward"] == 0.8

    @pytest.mark.asyncio
    async def test_critique_no_improvement_needed(self, sample_result):
        """Test critique when no improvement is needed."""
        critic = MetaRewardingCritic()
        
        mock_response = MetaRewardingResponse(
            feedback="The text is excellent as is",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            initial_feedback="Initial: Text is very good",
            meta_evaluation="Initial critique was accurate and complete",
            critique_evaluations=[
                CritiqueEvaluation(
                    aspect="accuracy",
                    score=0.95,
                    reasoning="Accurate assessment",
                    improvement_needed=False
                ),
                CritiqueEvaluation(
                    aspect="completeness",
                    score=0.95,
                    reasoning="Covered all aspects",
                    improvement_needed=False
                )
            ],
            refinement_rationale="No refinement needed",
            suggestion_preferences=[],
            initial_quality_score=0.9,
            final_quality_score=0.95,
            improvement_delta=0.05,
            meta_reward=0.95
        )
        
        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        
        with patch.object(critic.client, 'create_agent', return_value=mock_agent):
            result = await critic.critique("Excellent text", sample_result)
            
            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence > 0.9
            assert result.metadata["meta_reward"] == 0.95

    @pytest.mark.asyncio
    async def test_critique_with_significant_refinement(self, sample_result):
        """Test critique with significant refinement from meta-evaluation."""
        critic = MetaRewardingCritic()
        
        mock_response = MetaRewardingResponse(
            feedback="Refined: Major structural issues need addressing with specific reorganization",
            suggestions=[
                "Move conclusion to strengthen flow",
                "Add transition sentences between sections",
                "Consolidate redundant paragraphs"
            ],
            needs_improvement=True,
            confidence=0.9,
            initial_feedback="Initial: Text has problems",
            meta_evaluation="Initial critique was too vague and unhelpful",
            critique_evaluations=[
                CritiqueEvaluation(
                    aspect="specificity",
                    score=0.3,
                    reasoning="Initial critique lacked actionable details",
                    improvement_needed=True
                )
            ],
            refinement_rationale="Completely rewrote critique with specific, actionable suggestions",
            suggestion_preferences=[
                SuggestionPreference(
                    suggestion="Move conclusion to strengthen flow",
                    preference_score=0.95,
                    rationale="Most critical structural fix"
                )
            ],
            initial_quality_score=0.4,
            final_quality_score=0.9,
            improvement_delta=0.5,
            meta_reward=0.85
        )
        
        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        
        with patch.object(critic.client, 'create_agent', return_value=mock_agent):
            result = await critic.critique("Poorly structured text", sample_result)
            
            assert result.metadata["improvement_delta"] == 0.5
            assert len(result.suggestions) == 3
            assert len(result.metadata["suggestion_preferences"]) == 1

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider
        
        critic = MetaRewardingCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC
        
        critic = MetaRewardingCritic(provider="openai")
        assert critic.provider == Provider.OPENAI