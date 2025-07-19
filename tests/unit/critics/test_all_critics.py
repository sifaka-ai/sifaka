"""Comprehensive tests for all Sifaka critics."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics import (
    ConstitutionalCritic,
    MetaRewardingCritic,
    NCriticsCritic,
    PromptCritic,
    ReflexionCritic,
    SelfConsistencyCritic,
    SelfRAGCritic,
    SelfRefineCritic,
)


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
    factual_claims: list[str] = []
    retrieval_opportunities: list[str] = []
    improvement_suggestions: list[str]
    needs_improvement: bool
    confidence_score: float


@pytest.fixture
def mock_pydantic_agent():
    """Create a mock PydanticAI agent."""
    with patch("sifaka.core.llm_client.LLMClient.create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_create.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def sample_result():
    """Create a sample SifakaResult for testing."""
    return SifakaResult(
        original_text="Write about AI",
        final_text="AI is transforming the world",
        iteration=1,
        generations=[],
        critiques=[],
        validations=[],
        processing_time=1.0,
    )


class TestReflexionCritic:
    """Test ReflexionCritic functionality."""

    @pytest.mark.asyncio
    async def test_reflexion_basic(self, mock_pydantic_agent, sample_result):
        """Test basic reflexion critique."""
        # Create structured response matching the expected format
        mock_response = MockCriticResponse(
            feedback="The text is too brief and lacks detail",
            suggestions=["Add specific examples", "Expand on impact areas"],
            needs_improvement=True,
            confidence=0.85,
        )

        # Mock the agent run to return our response
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = ReflexionCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "reflexion"
        assert result.needs_improvement is True
        assert len(result.suggestions) == 2
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_reflexion_with_history(self, mock_pydantic_agent):
        """Test reflexion with previous critiques."""
        # Create result with history
        prev_critique = CritiqueResult(
            critic="reflexion",
            feedback="Needs more examples",
            suggestions=["Add case studies"],
            needs_improvement=True,
            confidence=0.8,
        )

        result_with_history = SifakaResult(
            original_text="Write about AI",
            final_text="AI is used in healthcare and finance",
            iteration=2,
            generations=[],
            critiques=[prev_critique],
            validations=[],
            processing_time=2.0,
        )

        mock_response = MockCriticResponse(
            feedback="Good improvement with examples",
            suggestions=["Add statistics", "Include future predictions"],
            needs_improvement=True,
            confidence=0.7,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = ReflexionCritic()
        result = await critic.critique(
            "AI is used in healthcare and finance", result_with_history
        )

        # Confidence may be recalculated, so check it's reasonable
        assert 0.6 <= result.confidence <= 0.8
        # Verify agent was called
        mock_pydantic_agent.run.assert_called_once()


class TestConstitutionalCritic:
    """Test ConstitutionalCritic functionality."""

    @pytest.mark.asyncio
    async def test_constitutional_basic(self, mock_pydantic_agent, sample_result):
        """Test basic constitutional critique."""
        mock_response = MockCriticResponse(
            feedback="The text needs to be more helpful and truthful",
            suggestions=["Provide accurate information", "Be more specific"],
            needs_improvement=True,
            confidence=0.9,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = ConstitutionalCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "constitutional"
        assert result.needs_improvement is True
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_constitutional_custom_principles(
        self, mock_pydantic_agent, sample_result
    ):
        """Test constitutional critic with custom principles."""
        custom_principles = [
            "Be technically accurate",
            "Include ethical considerations",
            "Avoid hyperbole",
        ]

        mock_response = MockCriticResponse(
            feedback="Lacks technical accuracy and ethical discussion",
            suggestions=["Add technical details", "Discuss ethical implications"],
            needs_improvement=True,
            confidence=0.95,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = ConstitutionalCritic(principles=custom_principles)
        result = await critic.critique("AI is amazing", sample_result)

        assert result.needs_improvement is True
        # Verify agent was created and called
        mock_pydantic_agent.run.assert_called_once()


class TestSelfRefineCritic:
    """Test SelfRefineCritic functionality."""

    @pytest.mark.asyncio
    async def test_self_refine_basic(self, mock_pydantic_agent, sample_result):
        """Test basic self-refine critique."""
        mock_response = MockCriticResponse(
            feedback="The text could be refined for clarity",
            suggestions=["Simplify language", "Add structure"],
            needs_improvement=True,
            confidence=0.75,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = SelfRefineCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "self_refine"
        assert result.needs_improvement is True
        assert len(result.suggestions) == 2


class TestNCriticsCritic:
    """Test NCriticsCritic functionality."""

    @pytest.mark.asyncio
    async def test_n_critics_ensemble(self, mock_pydantic_agent, sample_result):
        """Test N-critics ensemble evaluation."""
        # NCriticsCritic makes a single call that evaluates from multiple perspectives
        # Mock a single response that synthesizes multiple viewpoints
        mock_response = MockCriticResponse(
            feedback="Based on multi-perspective evaluation: The text lacks clarity and depth. From a clarity perspective, the language is too abstract. From an accuracy perspective, claims lack support. From a completeness perspective, key aspects are missing.",
            suggestions=[
                "Simplify language for better clarity",
                "Add supporting evidence for claims",
                "Include more comprehensive coverage",
                "Add specific examples",
            ],
            needs_improvement=True,
            confidence=0.8,
            metadata={"consensus_score": 0.6},
        )

        # Mock single call for multi-perspective evaluation
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = NCriticsCritic(perspective_count=3)
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "n_critics"
        assert result.needs_improvement is True
        # Should have multiple suggestions from different perspectives
        assert len(result.suggestions) >= 3
        # Check metadata contains consensus score
        assert "consensus_score" in result.metadata
        assert 0.7 <= result.confidence <= 0.9


class TestSelfRAGCritic:
    """Test SelfRAGCritic functionality."""

    @pytest.mark.asyncio
    async def test_self_rag_factual_check(self, mock_pydantic_agent, sample_result):
        """Test self-RAG factual accuracy checking."""
        mock_response = MockSelfRAGResponse(
            overall_assessment="Lacks factual support and citations",
            improvement_suggestions=["Add statistics", "Include sources"],
            needs_improvement=True,
            confidence_score=0.9,
            factual_claims=["AI is transforming the world"],
            retrieval_opportunities=["Statistics on AI impact"],
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = SelfRAGCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "self_rag"
        assert result.needs_improvement is True
        assert any(
            "source" in s.lower() or "statistic" in s.lower()
            for s in result.suggestions
        )


class TestMetaRewardingCritic:
    """Test MetaRewardingCritic functionality."""

    @pytest.mark.asyncio
    async def test_meta_rewarding_two_stage(self, mock_pydantic_agent, sample_result):
        """Test meta-rewarding evaluation (single call with meta-process)."""
        # MetaRewardingCritic does it all in one call with structured instructions
        # Mock response that reflects the meta-rewarding process
        mock_response = MockCriticResponse(
            feedback="After meta-evaluation: The text 'AI is transforming the world' is too vague and lacks supporting evidence. Initial assessment identified surface-level issues, but meta-evaluation revealed deeper problems with substantiation and specificity.",
            suggestions=[
                "Add specific examples of AI transformation in different sectors",
                "Include quantitative data to support claims",
                "Define what 'transforming' means in concrete terms",
                "Provide timeline and scope of transformation",
            ],
            needs_improvement=True,
            confidence=0.85,
            metadata={"meta_evaluation_performed": True},
        )

        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = MetaRewardingCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "meta_rewarding"
        # Should have high confidence after meta-evaluation
        assert result.confidence >= 0.8
        # Should have refined suggestions
        assert len(result.suggestions) >= 2


class TestSelfConsistencyCritic:
    """Test SelfConsistencyCritic functionality."""

    @pytest.mark.asyncio
    async def test_self_consistency_consensus(self, mock_pydantic_agent, sample_result):
        """Test self-consistency consensus building."""
        # Mock multiple evaluations with slight variations
        responses = [
            MockCriticResponse(
                feedback="Needs more detail",
                suggestions=["Add examples"],
                needs_improvement=True,
                confidence=0.8,
            ),
            MockCriticResponse(
                feedback="Too brief",
                suggestions=["Expand content", "Add examples"],
                needs_improvement=True,
                confidence=0.85,
            ),
            MockCriticResponse(
                feedback="Lacks depth",
                suggestions=["Add examples", "Include data"],
                needs_improvement=True,
                confidence=0.75,
            ),
        ]

        # Mock multiple calls for consensus evaluation
        mock_pydantic_agent.run.side_effect = [
            MockAgentResult(resp) for resp in responses
        ]

        critic = SelfConsistencyCritic(num_samples=3)
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "self_consistency"
        assert result.needs_improvement is True
        # Should have consensus feedback
        assert (
            "consensus" in result.feedback.lower()
            or "evaluations" in result.feedback.lower()
        )
        # Should have common suggestions
        assert "Add examples" in result.suggestions


class TestPromptCritic:
    """Test PromptCritic functionality."""

    @pytest.mark.asyncio
    async def test_prompt_critic_custom(self, mock_pydantic_agent, sample_result):
        """Test prompt critic with custom evaluation criteria."""
        custom_prompt = "Evaluate for technical accuracy and innovation"

        mock_response = MockCriticResponse(
            feedback="Lacks technical depth and innovative insights",
            suggestions=["Add technical details", "Include innovative applications"],
            needs_improvement=True,
            confidence=0.88,
        )
        mock_pydantic_agent.run.return_value = MockAgentResult(mock_response)

        critic = PromptCritic(custom_prompt=custom_prompt)
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "prompt"
        assert result.needs_improvement is True
        # Verify agent was called
        mock_pydantic_agent.run.assert_called_once()


class TestCriticErrorHandling:
    """Test error handling across all critics."""

    @pytest.mark.asyncio
    async def test_critic_error_recovery(self, mock_pydantic_agent, sample_result):
        """Test that critics handle errors gracefully."""
        # Mock an API error
        mock_pydantic_agent.run.side_effect = Exception("API Error")

        critics = [
            ReflexionCritic(),
            ConstitutionalCritic(),
            SelfRefineCritic(),
            NCriticsCritic(),
            SelfRAGCritic(),
            MetaRewardingCritic(),
            SelfConsistencyCritic(),
            PromptCritic(custom_prompt="Test prompt"),
        ]

        for critic in critics:
            # SelfRAGCritic and SelfConsistencyCritic have special error handling
            if isinstance(critic, (SelfRAGCritic, SelfConsistencyCritic)):
                result = await critic.critique("Test text", sample_result)
                assert isinstance(result, CritiqueResult)
                assert result.confidence == 0.0
                assert "Error" in result.feedback or "Failed" in result.feedback
            else:
                # Other critics should raise ModelProviderError
                with pytest.raises(Exception) as exc_info:
                    await critic.critique("Test text", sample_result)

                # Verify it's wrapped as ModelProviderError
                assert "failed to process text" in str(exc_info.value)


class TestCriticIntegration:
    """Test critics working together."""

    @pytest.mark.asyncio
    async def test_multiple_critics_sequence(self, mock_pydantic_agent):
        """Test running multiple critics in sequence."""
        result = SifakaResult(
            original_text="Write about AI",
            final_text="AI is transforming the world",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        # Mock different responses for different critics
        responses = [
            MockCriticResponse(
                feedback="Reflexion: needs examples",
                suggestions=["Add examples"],
                needs_improvement=True,
                confidence=0.8,
            ),
            MockCriticResponse(
                feedback="Constitutional: needs ethical discussion",
                suggestions=["Discuss ethics"],
                needs_improvement=True,
                confidence=0.85,
            ),
        ]

        # Mock multiple calls for different critics
        mock_pydantic_agent.run.side_effect = [
            MockAgentResult(resp) for resp in responses
        ]

        # Run multiple critics
        critics = [ReflexionCritic(), ConstitutionalCritic()]

        for critic in critics:
            critique = await critic.critique(result.final_text, result)
            result.add_critique(
                critic=critique.critic,
                feedback=critique.feedback,
                suggestions=critique.suggestions,
                needs_improvement=critique.needs_improvement,
                confidence=critique.confidence,
                metadata=critique.metadata,
            )

        # Verify both critiques were added
        assert len(result.critiques) == 2
        assert result.critiques[0].critic == "reflexion"
        assert result.critiques[1].critic == "constitutional"
        assert all(c.needs_improvement for c in result.critiques)
