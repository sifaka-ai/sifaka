"""Tests for Self-RAG critic."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sifaka.critics.self_rag import (
    SelfRAGCritic,
    SelfRAGResponse,
    FactualClaim,
    RetrievalOpportunity,
)
from sifaka.core.models import SifakaResult, CritiqueResult
from sifaka.core.config import Config


class TestFactualClaim:
    """Test the FactualClaim model."""

    def test_creation(self):
        """Test creating a factual claim."""
        claim = FactualClaim(
            claim="Python was created in 1991",
            isrel=True,
            issup=True,
            isuse=True,
            confidence_level="high",
            retrieval_needed=False,
            suggested_query=""
        )
        assert claim.claim == "Python was created in 1991"
        assert claim.isrel is True
        assert claim.issup is True
        assert claim.isuse is True
        assert claim.confidence_level == "high"
        assert claim.retrieval_needed is False

    def test_with_retrieval_needed(self):
        """Test claim that needs retrieval."""
        claim = FactualClaim(
            claim="The latest Python version has new features",
            isrel=True,
            issup=False,
            isuse=True,
            confidence_level="low",
            retrieval_needed=True,
            suggested_query="Python latest version features 2024"
        )
        assert claim.issup is False
        assert claim.retrieval_needed is True
        assert "Python latest" in claim.suggested_query


class TestRetrievalOpportunity:
    """Test the RetrievalOpportunity model."""

    def test_creation(self):
        """Test creating a retrieval opportunity."""
        opportunity = RetrievalOpportunity(
            location="Third paragraph discussing performance",
            reason="Specific performance metrics needed",
            expected_benefit="Would provide concrete evidence for claims",
            priority="high"
        )
        assert "Third paragraph" in opportunity.location
        assert "performance metrics" in opportunity.reason
        assert opportunity.priority == "high"

    def test_default_priority(self):
        """Test default priority."""
        opportunity = RetrievalOpportunity(
            location="Somewhere",
            reason="Some reason",
            expected_benefit="Some benefit"
        )
        assert opportunity.priority == "medium"


class TestSelfRAGResponse:
    """Test the SelfRAGResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = SelfRAGResponse(
            feedback="Content needs more evidence",
            suggestions=["Add citations"],
            needs_improvement=True,
            overall_relevance=True,
            overall_support=False,
            overall_utility=True
        )
        assert response.feedback == "Content needs more evidence"
        assert response.overall_support is False
        assert response.confidence == 0.7  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        claim = FactualClaim(
            claim="Test claim",
            isrel=True,
            issup=True,
            isuse=True,
            confidence_level="high",
            retrieval_needed=False
        )
        opportunity = RetrievalOpportunity(
            location="Introduction",
            reason="Needs context",
            expected_benefit="Better understanding"
        )
        
        response = SelfRAGResponse(
            feedback="Detailed feedback",
            suggestions=["Add sources", "Verify claims"],
            needs_improvement=True,
            confidence=0.85,
            overall_relevance=True,
            overall_support=False,
            overall_utility=True,
            factual_claims=[claim],
            retrieval_opportunities=[opportunity],
            relevance_score=0.9,
            support_score=0.4,
            utility_score=0.8,
            metadata={"extra": "data"}
        )
        
        assert len(response.factual_claims) == 1
        assert len(response.retrieval_opportunities) == 1
        assert response.relevance_score == 0.9
        assert response.support_score == 0.4


class TestSelfRAGCritic:
    """Test the SelfRAGCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text",
            final_text="Final text"
        )

    def test_initialization_default(self):
        """Test default initialization."""
        critic = SelfRAGCritic()
        assert critic.name == "self_rag"
        assert critic.model == "gpt-4o-mini"
        assert critic.temperature == 0.7

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.temperature = 0.5
        critic = SelfRAGCritic(config=config)
        assert critic.config == config

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = SelfRAGCritic(
            model="gpt-4",
            temperature=0.3,
            api_key="test-key"
        )
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
        assert "ISREL, ISSUP, ISUSE" in messages[0]["content"]
        
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
                suggestions=["Add sources"]
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
            feedback="Content has good relevance but lacks supporting evidence",
            suggestions=["Add citations for the claim about AI impact", "Include specific examples"],
            needs_improvement=True,
            confidence=0.8,
            overall_relevance=True,
            overall_support=False,
            overall_utility=True,
            factual_claims=[
                FactualClaim(
                    claim="AI is transforming industries",
                    isrel=True,
                    issup=False,
                    isuse=True,
                    confidence_level="medium",
                    retrieval_needed=True,
                    suggested_query="AI industry transformation statistics 2024"
                )
            ],
            retrieval_opportunities=[
                RetrievalOpportunity(
                    location="First paragraph",
                    reason="Lacks specific examples",
                    expected_benefit="Concrete evidence would strengthen claims",
                    priority="high"
                )
            ],
            relevance_score=0.9,
            support_score=0.4,
            utility_score=0.8
        )
        
        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        
        with patch.object(critic.client, 'create_agent', return_value=mock_agent):
            result = await critic.critique("Test text about AI", sample_result)
            
            assert result.critic == "self_rag"
            assert result.feedback == mock_response.feedback
            assert len(result.suggestions) == 2
            assert result.needs_improvement is True
            assert result.confidence == 0.8
            assert "factual_claims" in result.metadata
            assert "retrieval_opportunities" in result.metadata
            assert result.metadata["relevance_score"] == 0.9
            assert result.metadata["support_score"] == 0.4

    @pytest.mark.asyncio
    async def test_critique_all_criteria_pass(self, sample_result):
        """Test when all Self-RAG criteria pass."""
        critic = SelfRAGCritic()
        
        mock_response = SelfRAGResponse(
            feedback="Content is relevant, well-supported, and useful",
            suggestions=[],
            needs_improvement=False,
            confidence=0.95,
            overall_relevance=True,
            overall_support=True,
            overall_utility=True,
            factual_claims=[
                FactualClaim(
                    claim="Python is a programming language",
                    isrel=True,
                    issup=True,
                    isuse=True,
                    confidence_level="high",
                    retrieval_needed=False
                )
            ],
            retrieval_opportunities=[],
            relevance_score=0.95,
            support_score=0.95,
            utility_score=0.95
        )
        
        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        
        with patch.object(critic.client, 'create_agent', return_value=mock_agent):
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
            feedback="Multiple claims need verification",
            suggestions=["Verify statistics", "Add sources"],
            needs_improvement=True,
            confidence=0.7,
            overall_relevance=True,
            overall_support=False,
            overall_utility=True,
            factual_claims=[
                FactualClaim(
                    claim="Claim 1",
                    isrel=True,
                    issup=True,
                    isuse=True,
                    confidence_level="high",
                    retrieval_needed=False
                ),
                FactualClaim(
                    claim="Claim 2",
                    isrel=True,
                    issup=False,
                    isuse=True,
                    confidence_level="low",
                    retrieval_needed=True,
                    suggested_query="verify claim 2"
                )
            ],
            retrieval_opportunities=[],
            relevance_score=0.9,
            support_score=0.5,
            utility_score=0.8
        )
        
        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response
        
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        
        with patch.object(critic.client, 'create_agent', return_value=mock_agent):
            result = await critic.critique("Text with multiple claims", sample_result)
            
            assert len(result.metadata["factual_claims"]) == 2
            assert result.metadata["factual_claims"][1]["retrieval_needed"] is True

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider
        
        critic = SelfRAGCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC
        
        critic = SelfRAGCritic(provider="openai")
        assert critic.provider == Provider.OPENAI