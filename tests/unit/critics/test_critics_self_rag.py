"""Tests for Self-RAG critic."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.self_rag import (
    SelfRAGCritic,
    SelfRAGResponse,
)


class TestSelfRAGResponse:
    """Test the SelfRAGResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = SelfRAGResponse(
            isrel="YES",
            issup="NO",
            isuse="YES",
            overall_assessment="Content needs more evidence",
            improvement_suggestions=["Add citations"],
            needs_improvement=True,
        )
        assert response.overall_assessment == "Content needs more evidence"
        assert response.issup == "NO"
        assert response.confidence_score == 0.7  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        response = SelfRAGResponse(
            isrel="YES",
            issup="PARTIAL",
            isuse="YES",
            overall_assessment="Detailed feedback",
            specific_issues=["Lacks supporting evidence", "Some claims unverified"],
            specific_corrections=["Fact X should be Y"],
            factual_claims=["Test claim", "Another claim"],
            retrieval_opportunities=["Introduction needs more context"],
            improvement_suggestions=["Add sources", "Verify claims"],
            needs_improvement=True,
            confidence_score=0.85,
        )

        assert len(response.factual_claims) == 2
        assert len(response.retrieval_opportunities) == 1
        assert response.confidence_score == 0.85
        assert response.issup == "PARTIAL"


class TestSelfRAGCritic:
    """Test the SelfRAGCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        critic = SelfRAGCritic()
        assert critic.name == "self_rag"
        assert critic.model == "gpt-3.5-turbo"  # Default from Config
        assert critic.temperature == 0.7

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.llm.temperature = 0.5
        critic = SelfRAGCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = SelfRAGCritic(model="gpt-4", temperature=0.3, api_key="test-key")
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.3

    def test_get_response_type(self):
        """Test that critic uses SelfRAGResponse."""
        critic = SelfRAGCritic()
        assert critic._get_response_type() == SelfRAGResponse

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation for Self-RAG evaluation."""
        critic = SelfRAGCritic()
        messages = await critic._create_messages("Test text about AI", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Self-RAG critic" in messages[0]["content"]
        assert "reflection" in messages[0]["content"].lower()

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Self-RAG's reflection framework" in user_content
        assert "ISREL:" in user_content
        assert "ISSUP:" in user_content
        assert "ISUSE:" in user_content
        assert "factual claims" in user_content
        assert "retrieval opportunities" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique from the same critic
        sample_result.critiques.append(
            CritiqueResult(
                critic="self_rag",
                feedback="Previous Self-RAG feedback",
                suggestions=["Add sources"],
            )
        )

        critic = SelfRAGCritic()
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous Self-RAG feedback" in user_content

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_result):
        """Test successful critique flow."""
        critic = SelfRAGCritic()

        # Mock the LLM response
        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="NO",
            isuse="YES",
            overall_assessment="Content has good relevance but lacks supporting evidence",
            specific_issues=[
                "No citations for AI impact claim",
                "Missing specific examples",
            ],
            specific_corrections=[],
            factual_claims=["AI is transforming industries"],
            retrieval_opportunities=["First paragraph - needs specific examples"],
            improvement_suggestions=[
                "Add citations for the claim about AI impact",
                "Include specific examples",
            ],
            needs_improvement=True,
            confidence_score=0.8,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Test text about AI", sample_result)

            assert result.critic == "self_rag"
            # Check that feedback contains reflection tokens and assessment
            assert "ISREL: YES" in result.feedback
            assert "ISSUP: NO" in result.feedback
            assert "ISUSE: YES" in result.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.8  # Maps from confidence_score
            assert "factual_claims" in result.metadata
            assert "retrieval_opportunities" in result.metadata
            assert len(result.metadata["factual_claims"]) == 1
            assert len(result.metadata["retrieval_opportunities"]) == 1

    @pytest.mark.asyncio
    async def test_critique_all_criteria_pass(self, sample_result):
        """Test when all Self-RAG criteria pass."""
        critic = SelfRAGCritic()

        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="YES",
            isuse="YES",
            overall_assessment="Content is relevant, well-supported, and useful",
            specific_issues=[],
            specific_corrections=[],
            factual_claims=["Python is a programming language"],
            retrieval_opportunities=[],
            improvement_suggestions=[],
            needs_improvement=False,
            confidence_score=0.95,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Well-written text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert result.confidence > 0.9
            assert len(result.metadata["retrieval_opportunities"]) == 0

    @pytest.mark.asyncio
    async def test_critique_multiple_claims(self, sample_result):
        """Test critique with multiple factual claims."""
        critic = SelfRAGCritic()

        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="PARTIAL",
            isuse="YES",
            overall_assessment="Multiple claims need verification",
            specific_issues=["Claim 2 lacks support"],
            specific_corrections=[],
            factual_claims=["Claim 1", "Claim 2"],
            retrieval_opportunities=[],
            improvement_suggestions=["Verify statistics", "Add sources"],
            needs_improvement=True,
            confidence_score=0.7,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        mock_agent_result.usage = Mock(return_value=Mock(total_tokens=100))

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch.object(critic.client, "create_agent", return_value=mock_agent):
            result = await critic.critique("Text with multiple claims", sample_result)

            assert len(result.metadata["factual_claims"]) == 2
            # Just check that we have the claims
            assert "Claim 2" in result.metadata["factual_claims"]

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = SelfRAGCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = SelfRAGCritic(provider="openai")
        assert critic.provider == Provider.OPENAI
