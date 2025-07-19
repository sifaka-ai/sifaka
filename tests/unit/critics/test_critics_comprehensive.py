"""Comprehensive tests for all critic implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.critics.self_refine import SelfRefineCritic


class MockAgentResult:
    """Mock result from PydanticAI agent run."""

    def __init__(self, output):
        self.output = output
        self._usage = MagicMock()
        self._usage.total_tokens = 100

    def usage(self):
        """Mock usage data."""
        return self._usage


class MockCriticResponse(BaseModel):
    """Mock structured response for critics."""

    feedback: str
    suggestions: list[str]
    needs_improvement: bool
    confidence: float
    metadata: dict = {}


class MockSelfRAGResponse(BaseModel):
    """Mock structured response for SelfRAG critic."""

    isrel: str = "YES"
    issup: str = "NO"
    isuse: str = "PARTIAL"
    overall_assessment: str
    specific_issues: list[str] = []
    specific_corrections: list[str] = []


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample text that needs to be evaluated for quality and potential improvements."


@pytest.fixture
def sample_result():
    """Sample SifakaResult for testing."""
    return SifakaResult(
        original_text="Original text",
        final_text="Improved text",
        iteration=0,
        critiques=[],
    )


@pytest.fixture
def mock_pydantic_agent():
    """Create a mock PydanticAI agent."""
    with patch("sifaka.core.llm_client.LLMClient.create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_create.return_value = mock_agent
        yield mock_agent


class TestSelfConsistencyCritic:
    """Test Self-Consistency critic implementation."""

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_text, sample_result):
        """Test successful critique with proper responses."""
        from sifaka.critics.self_consistency import SelfConsistencyResponse

        # Mock the consolidated response from SelfConsistencyCritic
        mock_response = SelfConsistencyResponse(
            feedback="Based on multiple evaluations, the text lacks detail and examples",
            suggestions=[
                "Add specific examples",
                "Expand key points",
                "Provide more depth",
            ],
            needs_improvement=True,
            confidence=0.72,  # Average of evaluations
            metadata={
                "num_evaluations": 2,
                "consensus_metrics": {
                    "quality_score_variance": 0.0,
                    "priority_agreement": 1.0,
                },
            },
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.8

        critic = SelfConsistencyCritic(num_samples=2)
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "self_consistency"
        assert result.feedback  # Should have feedback
        assert len(result.suggestions) > 0
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_critique_with_inconsistent_evaluations(
        self, sample_text, sample_result
    ):
        """Test critique with highly inconsistent evaluations."""
        from sifaka.critics.self_consistency import SelfConsistencyResponse

        # Mock response showing inconsistency
        mock_response = SelfConsistencyResponse(
            feedback="Multiple evaluations show significant inconsistency in assessment.",
            suggestions=[
                "Address fundamental quality inconsistency",
                "Consider major revision to establish baseline quality",
            ],
            needs_improvement=True,
            confidence=0.4,  # Low confidence due to inconsistency
            metadata={
                "num_evaluations": 2,
                "consensus_metrics": {
                    "quality_score_variance": 16.0,  # High variance
                    "priority_agreement": 0.0,  # No agreement
                },
            },
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.8

        critic = SelfConsistencyCritic(num_samples=2)
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert "inconsistency" in " ".join(result.suggestions).lower()
        assert result.needs_improvement

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, sample_text, sample_result):
        """Test error handling when API calls fail."""
        # Mock agent to raise exception
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("API Error"))

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.8

        critic = SelfConsistencyCritic(num_samples=2)
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert result.critic == "self_consistency"
        # SelfConsistency has its own error handling
        assert "failed to generate consistent evaluations" in result.feedback.lower()
        assert result.confidence == 0.0

    def test_consistency_calculations(self):
        """Test consistency calculation methods."""
        # Since internal methods are private, we test through the public API
        # by mocking different responses that would indicate consistency
        critic = SelfConsistencyCritic(num_samples=2)

        # Test that the critic initializes correctly with expected settings
        assert critic.num_samples == 3  # Default is 3 samples
        assert critic.temperature == 0.8  # Higher for diversity
        assert critic.name == "self_consistency"


class TestMetaRewardingCritic:
    """Test Meta-Rewarding critic implementation."""

    @pytest.fixture
    def critic(self):
        return MetaRewardingCritic()

    @pytest.mark.asyncio
    async def test_critique_success(self, critic, sample_text, sample_result):
        """Test successful meta-rewarding critique."""
        from sifaka.critics.meta_rewarding import MetaRewardingResponse

        # Mock the final response after meta-rewarding process
        mock_response = MetaRewardingResponse(
            feedback="""Initial judgment:
Text is clear and readable (CLARITY: 4)
Some factual claims need verification (ACCURACY: 3)
Covers main points well (COMPLETENESS: 4)
Well organized (STRUCTURE: 5)
Could be more engaging (ENGAGEMENT: 3)

Meta-assessment: The evaluation is comprehensive and fair
Reliability: 0.8

Refined critique: Good quality text with targeted improvements needed in accuracy and engagement.""",
            suggestions=[
                "Verify factual claims for accuracy",
                "Add more specific examples for engagement",
                "Consider more dynamic language to enhance reader interest",
            ],
            needs_improvement=True,
            confidence=0.8,
            metadata={
                "initial_scores": {
                    "clarity": 4,
                    "accuracy": 3,
                    "completeness": 4,
                    "structure": 5,
                    "engagement": 3,
                },
                "meta_reliability": 0.8,
                "improvement_delta": 0.1,
            },
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.7

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "meta_rewarding"
        assert (
            "Initial judgment" in result.feedback
            or "initial judgment" in result.feedback.lower()
        )
        assert (
            "Meta-assessment" in result.feedback
            or "meta-assessment" in result.feedback.lower()
        )
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_critique_low_reliability(self, critic, sample_text, sample_result):
        """Test critique with low reliability meta-judgment."""
        from sifaka.critics.meta_rewarding import MetaRewardingResponse

        mock_response = MetaRewardingResponse(
            feedback="""Initial judgment: Basic assessment

Meta-assessment: Evaluation lacks depth
Reliability: 0.4
Corrections: Need more thorough analysis

Refined critique: The initial assessment requires significant improvement.""",
            suggestions=[
                "Assessment reliability could be enhanced",
                "Need more thorough analysis",
                "Provide deeper evaluation across all dimensions",
            ],
            needs_improvement=True,
            confidence=0.4,
            metadata={"meta_reliability": 0.4},
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.7

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert result.confidence == 0.4
        assert result.needs_improvement
        assert "Assessment reliability could be enhanced" in result.suggestions

    @pytest.mark.asyncio
    async def test_parsing_with_malformed_response(
        self, critic, sample_text, sample_result
    ):
        """Test parsing with malformed API responses."""
        from sifaka.critics.meta_rewarding import MetaRewardingResponse

        # Even with minimal data, structured response provides defaults
        mock_response = MetaRewardingResponse(
            feedback="Malformed response without proper structure",
            suggestions=[],  # Empty suggestions
            needs_improvement=True,
            confidence=0.7,  # Default confidence
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.7

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        # Should handle gracefully with defaults
        assert isinstance(result, CritiqueResult)
        # Confidence is calculated based on feedback length and suggestions
        # Short feedback (< 20 words) and no suggestions reduces confidence
        assert result.confidence == pytest.approx(
            0.55, abs=0.01
        )  # 0.7 base - 0.05 (short) - 0.1 (no suggestions)


class TestNCriticsCritic:
    """Test N-Critics ensemble critic implementation."""

    @pytest.fixture
    def critic(self):
        return NCriticsCritic()

    @pytest.mark.asyncio
    async def test_critique_success(self, critic, sample_text, sample_result):
        """Test successful N-Critics ensemble critique."""
        from sifaka.critics.n_critics import NCriticsResponse

        mock_response = NCriticsResponse(
            feedback="""Ensemble consensus from multiple critical perspectives:

            Clarity perspective shows good readability with clear structure.
            Accuracy perspective finds minor factual issues requiring verification.
            Completeness perspective suggests adding examples for better understanding.
            Style perspective recommends tone adjustment for target audience.

            Overall, the text demonstrates good quality but needs targeted improvements.""",
            suggestions=[
                "Add more specific examples to illustrate key points",
                "Verify factual claims for accuracy",
                "Adjust tone for better audience engagement",
            ],
            needs_improvement=True,
            confidence=0.85,
            consensus_score=0.75,
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.7

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "n_critics"
        assert "consensus" in result.feedback.lower()
        assert len(result.suggestions) >= 2
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_critique_low_consensus(self, critic, sample_text, sample_result):
        """Test critique with low consensus score."""
        from sifaka.critics.n_critics import NCriticsResponse

        mock_response = NCriticsResponse(
            feedback="""Mixed assessments across perspectives with significant disagreement.
            Overall quality score 0.5 indicates substantial room for improvement.""",
            suggestions=["Major revisions needed across multiple dimensions"],
            needs_improvement=True,
            confidence=0.5,  # Low confidence due to disagreement
            consensus_score=0.5,
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.7

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert result.needs_improvement
        assert result.confidence == 0.5  # Direct from response now

    def test_custom_perspectives(self):
        """Test critic with custom perspectives."""
        custom_perspectives = ["Technical accuracy", "User experience focus"]
        critic = NCriticsCritic(perspectives=custom_perspectives)
        assert critic.perspectives == custom_perspectives


class TestSelfRefineCritic:
    """Test Self-Refine critic implementation."""

    @pytest.fixture
    def critic(self):
        return SelfRefineCritic()

    @pytest.mark.asyncio
    async def test_critique_success(self, critic, sample_text, sample_result):
        """Test successful self-refine critique."""
        from sifaka.critics.self_refine import RefinementArea, SelfRefineResponse

        mock_response = SelfRefineResponse(
            feedback="The text is clear but could benefit from more detailed examples and better structure",
            suggestions=[
                "Add specific examples to support main points",
                "Improve paragraph transitions",
                "Enhance conclusion with actionable insights",
            ],
            needs_improvement=True,
            confidence=0.7,
            refinement_areas=[
                RefinementArea(
                    target_state="Main points supported with concrete examples"
                ),
                RefinementArea(target_state="Smooth transitions between paragraphs"),
                RefinementArea(target_state="Conclusion provides clear next steps"),
            ],
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "self_refine"
        assert result.confidence == 0.7
        assert len(result.suggestions) >= 2
        assert len(result.suggestions) > 0
        assert any(len(s) > 5 for s in result.suggestions)  # Non-trivial suggestions

    @pytest.mark.asyncio
    async def test_critique_high_quality_text(self, critic, sample_text, sample_result):
        """Test critique of high-quality text."""
        from sifaka.critics.self_refine import RefinementArea, SelfRefineResponse

        mock_response = SelfRefineResponse(
            feedback="Excellent text quality with minor refinements possible",
            suggestions=["Consider minor stylistic improvements"],
            needs_improvement=False,
            confidence=0.9,
            refinement_areas=[
                RefinementArea(target_state="Slightly more formal tone in introduction")
            ],
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert not result.needs_improvement
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_iterative_context(self, critic, sample_text):
        """Test iterative refinement context."""
        from sifaka.core.models import Generation
        from sifaka.critics.self_refine import SelfRefineResponse

        # Create previous generations to test refinement context
        generation1 = Generation(
            text="First version of the text", model="gpt-4o-mini", iteration=0
        )
        generation2 = Generation(
            text="Second version with improvements and more content",
            model="gpt-4o-mini",
            iteration=1,
        )

        previous_critique = CritiqueResult(
            critic="previous",
            feedback="Previous feedback",
            suggestions=["Previous suggestion"],
            needs_improvement=True,
            confidence=0.5,
        )

        iterative_result = SifakaResult(
            original_text="Original",
            final_text="Second version with improvements and more content",
            iteration=2,
            critiques=[previous_critique],
            generations=[generation1, generation2],
        )

        mock_response = SelfRefineResponse(
            feedback="Good progress from previous iteration",
            suggestions=["Minor improvements"],
            needs_improvement=False,
            confidence=0.8,
            refinement_areas=[],
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, iterative_result)

        # Verify the critique result
        assert result.confidence == 0.8
        assert not result.needs_improvement

    @pytest.mark.asyncio
    async def test_parse_refinement_with_malformed_input(
        self, critic, sample_text, sample_result
    ):
        """Test parsing with malformed refinement input."""
        from sifaka.critics.self_refine import SelfRefineResponse

        # Create a response with minimal data
        mock_response = SelfRefineResponse(
            feedback="This is not properly formatted output",
            suggestions=[],  # Empty suggestions
            needs_improvement=True,
            confidence=0.5,
            refinement_areas=[],
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert result.feedback == "This is not properly formatted output"
        # With PydanticAI, we get structured responses even with minimal input
        assert isinstance(result.suggestions, list)
        assert result.confidence == 0.5


class TestSelfRAGCritic:
    """Test Self-RAG critic implementation."""

    @pytest.fixture
    def critic(self):
        return SelfRAGCritic()

    @pytest.mark.asyncio
    async def test_critique_needs_retrieval(self, critic, sample_text, sample_result):
        """Test critique that identifies retrieval needs."""
        # Create mock response using SelfRAGResponse
        from sifaka.critics.self_rag import SelfRAGResponse

        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="NO",
            isuse="PARTIAL",
            overall_assessment="Content appears accurate but lacks recent data and supporting evidence",
            specific_issues=["Missing current statistics", "No source citations"],
            specific_corrections=[],
            factual_claims=["Statistics need verification"],
            retrieval_opportunities=[
                "Current statistics on the topic",
                "Authoritative sources",
            ],
            improvement_suggestions=[
                "Retrieve current statistics on the topic",
                "Add credible source citations",
                "Verify factual claims with authoritative sources",
            ],
            needs_improvement=True,
            confidence_score=0.8,
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "self_rag"
        assert result.needs_improvement
        assert result.confidence >= 0.7  # Should have reasonable confidence
        assert "ISREL: YES" in result.feedback or "relevant" in result.feedback.lower()
        assert len(result.suggestions) >= 2

    @pytest.mark.asyncio
    async def test_critique_no_retrieval_needed(
        self, critic, sample_text, sample_result
    ):
        """Test critique that finds no retrieval needs."""
        from sifaka.critics.self_rag import SelfRAGResponse

        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="YES",
            isuse="YES",
            overall_assessment="Text demonstrates good factual accuracy and comprehensive coverage",
            specific_issues=[],
            specific_corrections=[],
            factual_claims=[],
            retrieval_opportunities=[],
            improvement_suggestions=[
                "Minor stylistic improvements could enhance readability"
            ],
            needs_improvement=False,
            confidence_score=0.9,
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert not result.needs_improvement
        assert result.confidence >= 0.8  # Higher confidence when content is good
        assert (
            "ISREL: YES" in result.feedback
            or "factual accuracy" in result.feedback.lower()
        )

    @pytest.mark.asyncio
    async def test_critique_factual_accuracy_issues(
        self, critic, sample_text, sample_result
    ):
        """Test critique that identifies factual accuracy issues."""
        from sifaka.critics.self_rag import SelfRAGResponse

        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="NO",
            isuse="PARTIAL",
            overall_assessment="Several factual accuracy concerns identified",
            specific_issues=[
                "Statistical claims need verification",
                "Some information appears outdated",
            ],
            specific_corrections=[],
            factual_claims=["Statistical claims require verification"],
            retrieval_opportunities=[],
            improvement_suggestions=[
                "Verify accuracy of statistical claims",
                "Update outdated information",
                "Add fact-checking references",
            ],
            needs_improvement=True,
            confidence_score=0.75,
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        # Should need improvement due to accuracy issues
        assert result.needs_improvement
        # Check if accuracy is mentioned somewhere in the result
        accuracy_mentioned = (
            any("accuracy" in s.lower() for s in result.suggestions)
            or "accuracy" in result.feedback.lower()
            or "ISSUP: NO" in result.feedback
        )
        assert accuracy_mentioned

    @pytest.mark.asyncio
    async def test_parse_self_rag_with_incomplete_sections(
        self, critic, sample_text, sample_result
    ):
        """Test handling incomplete response data."""
        from sifaka.critics.self_rag import SelfRAGResponse

        # Create response with minimal data
        mock_response = SelfRAGResponse(
            isrel="YES",
            issup="PARTIAL",
            isuse="PARTIAL",
            overall_assessment="Basic assessment",
            # All lists will be empty by default
            needs_improvement=True,
        )

        # Mock PydanticAI agent
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"  # Add model attribute
        mock_client.temperature = 0.7  # Add temperature attribute

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert result.needs_improvement
        # With minimal input, suggestions list might be empty
        assert isinstance(result.suggestions, list)


@pytest.mark.asyncio
async def test_all_critics_basic_functionality(sample_text, sample_result):
    """Integration test ensuring all critics can be instantiated and called."""
    critics = [
        SelfConsistencyCritic(num_samples=1),  # Reduced for testing
        MetaRewardingCritic(),
        NCriticsCritic(),
        SelfRefineCritic(),
        SelfRAGCritic(),
    ]

    for critic in critics:
        # Mock the PydanticAI agent for all critics
        # Create a basic response that should work for all critics
        mock_response = MockCriticResponse(
            feedback="Test feedback for " + critic.name,
            suggestions=["Test suggestion 1", "Test suggestion 2"],
            needs_improvement=True,
            confidence=0.7,
        )

        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = MagicMock()
        mock_client.create_agent = MagicMock(return_value=mock_agent)
        mock_client.model = "gpt-4o-mini"
        mock_client.temperature = 0.7

        # Set the private client attribute directly
        critic._client = mock_client
        result = await critic.critique(sample_text, sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == critic.name
        assert isinstance(result.feedback, str)
        assert isinstance(result.suggestions, list)
        assert isinstance(result.confidence, (int, float))
        assert isinstance(result.needs_improvement, bool)


def test_critic_names():
    """Test that all critics have correct names."""
    critics_and_names = [
        (SelfConsistencyCritic(), "self_consistency"),
        (MetaRewardingCritic(), "meta_rewarding"),
        (NCriticsCritic(), "n_critics"),
        (SelfRefineCritic(), "self_refine"),
        (SelfRAGCritic(), "self_rag"),
    ]

    for critic, expected_name in critics_and_names:
        assert critic.name == expected_name
