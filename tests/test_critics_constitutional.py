"""Tests for Constitutional AI critic."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sifaka.critics.constitutional import (
    ConstitutionalCritic,
    ConstitutionalResponse,
    PrincipleEvaluation,
    RevisionProposal,
    DEFAULT_PRINCIPLES,
)
from sifaka.core.models import SifakaResult, CritiqueResult
from sifaka.core.config import Config


class TestPrincipleEvaluation:
    """Test the PrincipleEvaluation model."""

    def test_creation(self):
        """Test creating a principle evaluation."""
        eval = PrincipleEvaluation(
            principle="Be helpful",
            category="helpful",
            passed=True,
            severity=0.0,
            violations=[],
            improvements=[],
        )
        assert eval.principle == "Be helpful"
        assert eval.category == "helpful"
        assert eval.passed is True
        assert eval.severity == 0.0

    def test_with_violations(self):
        """Test principle evaluation with violations."""
        eval = PrincipleEvaluation(
            principle="Avoid misinformation",
            category="honest",
            passed=False,
            severity=0.7,
            violations=["Contains unverified claims"],
            improvements=["Add sources", "Qualify statements"],
        )
        assert eval.passed is False
        assert eval.severity == 0.7
        assert len(eval.violations) == 1
        assert len(eval.improvements) == 2


class TestRevisionProposal:
    """Test the RevisionProposal model."""

    def test_creation(self):
        """Test creating a revision proposal."""
        proposal = RevisionProposal(
            original_snippet="This is definitely true",
            revised_snippet="This appears to be true based on current evidence",
            principles_addressed=["Promote accuracy"],
            improvement_rationale="Adds appropriate qualification",
        )
        assert "definitely" in proposal.original_snippet
        assert "appears to be" in proposal.revised_snippet
        assert len(proposal.principles_addressed) == 1


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
        eval = PrincipleEvaluation(
            principle="Test principle", category="helpful", passed=True
        )
        proposal = RevisionProposal(
            original_snippet="old",
            revised_snippet="new",
            principles_addressed=["Test"],
            improvement_rationale="Better",
        )

        response = ConstitutionalResponse(
            feedback="Detailed feedback",
            suggestions=["Suggestion 1"],
            needs_improvement=True,
            confidence=0.9,
            principle_evaluations=[eval],
            revision_proposals=[proposal],
            helpfulness_score=0.8,
            harmlessness_score=0.9,
            honesty_score=0.7,
            requires_major_revision=False,
            self_critique_notes="Good improvements",
            metadata={"key": "value"},
        )

        assert len(response.principle_evaluations) == 1
        assert len(response.revision_proposals) == 1
        assert response.helpfulness_score == 0.8
        assert response.metadata["key"] == "value"


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

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Constitutional AI critic" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Constitutional Principles:" in user_content
        assert "1. Principle 1" in user_content
        assert "2. Principle 2" in user_content
        assert "Stage 1 - Constitutional Evaluation:" in user_content
        assert "Stage 2 - Constitutional Revision:" in user_content

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

        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous feedback" in user_content

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
            principle_evaluations=[
                PrincipleEvaluation(
                    principle="Promote accuracy",
                    category="honest",
                    passed=False,
                    severity=0.3,
                    violations=["Some claims lack sources"],
                    improvements=["Add citations"],
                )
            ],
            helpfulness_score=0.9,
            harmlessness_score=1.0,
            honesty_score=0.7,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "constitutional"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 1
            assert result.needs_improvement is True
            assert result.confidence == 0.85
            assert "principle_evaluations" in result.metadata
            assert result.metadata["helpfulness_score"] == 0.9

    @pytest.mark.asyncio
    async def test_critique_with_revision_proposals(self, sample_result):
        """Test critique with revision proposals."""
        critic = ConstitutionalCritic()

        mock_response = ConstitutionalResponse(
            feedback="Text needs constitutional revisions",
            suggestions=["Apply proposed revisions"],
            needs_improvement=True,
            revision_proposals=[
                RevisionProposal(
                    original_snippet="Everyone knows",
                    revised_snippet="Many experts believe",
                    principles_addressed=["Avoid misinformation"],
                    improvement_rationale="More accurate phrasing",
                )
            ],
            requires_major_revision=True,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text", sample_result)

            assert result.needs_improvement is True
            assert "revision_proposals" in result.metadata
            assert len(result.metadata["revision_proposals"]) == 1
            assert result.metadata["requires_major_revision"] is True

    @pytest.mark.asyncio
    async def test_critique_all_principles_pass(self, sample_result):
        """Test when all principles pass."""
        critic = ConstitutionalCritic(principles=["Be helpful", "Be clear"])

        mock_response = ConstitutionalResponse(
            feedback="Text adheres to all constitutional principles",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            principle_evaluations=[
                PrincipleEvaluation(
                    principle="Be helpful",
                    category="helpful",
                    passed=True,
                    severity=0.0,
                ),
                PrincipleEvaluation(
                    principle="Be clear", category="helpful", passed=True, severity=0.0
                ),
            ],
            helpfulness_score=0.95,
            harmlessness_score=0.95,
            honesty_score=0.95,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

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
