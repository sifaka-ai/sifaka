"""Comprehensive tests for all Sifaka critics."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from sifaka.core.models import SifakaResult, CritiqueResult
from sifaka.critics import (
    ReflexionCritic,
    ConstitutionalCritic,
    SelfRefineCritic,
    NCriticsCritic,
    SelfRAGCritic,
    MetaRewardingCritic,
    SelfConsistencyCritic,
    PromptCritic,
)


class MockLLMResponse:
    """Mock response for LLM calls."""

    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    with patch("sifaka.core.llm_client.LLMManager.get_client") as mock_manager:
        mock_client = MagicMock()
        mock_client.complete = AsyncMock()
        mock_manager.return_value = mock_client
        yield mock_client


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
    async def test_reflexion_basic(self, mock_llm_client, sample_result):
        """Test basic reflexion critique."""
        mock_response = MockLLMResponse(
            """{
            "feedback": "The text is too brief and lacks detail",
            "suggestions": ["Add specific examples", "Expand on impact areas"],
            "needs_improvement": true,
            "confidence": 0.85
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

        critic = ReflexionCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert isinstance(result, CritiqueResult)
        assert result.critic == "reflexion"
        assert result.needs_improvement is True
        assert len(result.suggestions) == 2
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_reflexion_with_history(self, mock_llm_client):
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

        mock_response = MockLLMResponse(
            """{
            "feedback": "Good improvement with examples",
            "suggestions": ["Add statistics", "Include future predictions"],
            "needs_improvement": true,
            "confidence": 0.7
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

        critic = ReflexionCritic()
        result = await critic.critique(
            "AI is used in healthcare and finance", result_with_history
        )

        assert result.confidence == 0.7
        # Verify context was included in messages
        mock_llm_client.complete.assert_called_once()
        messages = mock_llm_client.complete.call_args[0][0]
        assert any("Previous feedback" in msg.get("content", "") for msg in messages)


class TestConstitutionalCritic:
    """Test ConstitutionalCritic functionality."""

    @pytest.mark.asyncio
    async def test_constitutional_basic(self, mock_llm_client, sample_result):
        """Test basic constitutional critique."""
        mock_response = MockLLMResponse(
            """{
            "feedback": "The text needs to be more helpful and truthful",
            "suggestions": ["Provide accurate information", "Be more specific"],
            "needs_improvement": true,
            "confidence": 0.9
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

        critic = ConstitutionalCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "constitutional"
        assert result.needs_improvement is True
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_constitutional_custom_principles(
        self, mock_llm_client, sample_result
    ):
        """Test constitutional critic with custom principles."""
        custom_principles = [
            "Be technically accurate",
            "Include ethical considerations",
            "Avoid hyperbole",
        ]

        mock_response = MockLLMResponse(
            """{
            "feedback": "Lacks technical accuracy and ethical discussion",
            "suggestions": ["Add technical details", "Discuss ethical implications"],
            "needs_improvement": true,
            "confidence": 0.95
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

        critic = ConstitutionalCritic(principles=custom_principles)
        result = await critic.critique("AI is amazing", sample_result)

        assert result.needs_improvement is True
        # Verify custom principles were used
        messages = mock_llm_client.complete.call_args[0][0]
        assert any("technically accurate" in msg.get("content", "") for msg in messages)


class TestSelfRefineCritic:
    """Test SelfRefineCritic functionality."""

    @pytest.mark.asyncio
    async def test_self_refine_basic(self, mock_llm_client, sample_result):
        """Test basic self-refine critique."""
        mock_response = MockLLMResponse(
            """{
            "feedback": "The text could be refined for clarity",
            "suggestions": ["Simplify language", "Add structure"],
            "needs_improvement": true,
            "confidence": 0.75
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

        critic = SelfRefineCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "self_refine"
        assert result.needs_improvement is True
        assert len(result.suggestions) == 2


class TestNCriticsCritic:
    """Test NCriticsCritic functionality."""

    @pytest.mark.asyncio
    async def test_n_critics_ensemble(self, mock_llm_client, sample_result):
        """Test N-critics ensemble evaluation."""
        # Mock responses for different perspectives
        responses = [
            MockLLMResponse(
                """{
                "feedback": "Clarity perspective: needs improvement",
                "suggestions": ["Simplify language"],
                "needs_improvement": true,
                "confidence": 0.8
            }"""
            ),
            MockLLMResponse(
                """{
                "feedback": "Depth perspective: too shallow",
                "suggestions": ["Add more detail"],
                "needs_improvement": true,
                "confidence": 0.85
            }"""
            ),
            MockLLMResponse(
                """{
                "feedback": "Engagement perspective: not compelling",
                "suggestions": ["Add examples"],
                "needs_improvement": true,
                "confidence": 0.75
            }"""
            ),
        ]

        mock_llm_client.complete.side_effect = responses

        critic = NCriticsCritic(num_critics=3)
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "n_critics"
        assert result.needs_improvement is True
        # Should have combined suggestions from all perspectives
        assert len(result.suggestions) >= 3
        # Confidence should be average of all
        assert 0.7 <= result.confidence <= 0.9


class TestSelfRAGCritic:
    """Test SelfRAGCritic functionality."""

    @pytest.mark.asyncio
    async def test_self_rag_factual_check(self, mock_llm_client, sample_result):
        """Test self-RAG factual accuracy checking."""
        mock_response = MockLLMResponse(
            """{
            "feedback": "Lacks factual support and citations",
            "suggestions": ["Add statistics", "Include sources"],
            "needs_improvement": true,
            "confidence": 0.9
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

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
    async def test_meta_rewarding_two_stage(self, mock_llm_client, sample_result):
        """Test meta-rewarding two-stage evaluation."""
        # First stage response
        initial_response = MockLLMResponse(
            """{
            "feedback": "Initial assessment: needs improvement",
            "suggestions": ["Add examples", "Improve flow"],
            "needs_improvement": true,
            "confidence": 0.7
        }"""
        )

        # Meta evaluation response
        meta_response = MockLLMResponse(
            """{
            "feedback": "Meta-evaluation: critique is valid but could be more specific",
            "suggestions": ["Add examples", "Improve flow", "Include data"],
            "needs_improvement": true,
            "confidence": 0.85
        }"""
        )

        mock_llm_client.complete.side_effect = [initial_response, meta_response]

        critic = MetaRewardingCritic()
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "meta_rewarding"
        # Should use meta-evaluation confidence
        assert result.confidence == 0.85
        # Should have refined suggestions
        assert len(result.suggestions) >= 2


class TestSelfConsistencyCritic:
    """Test SelfConsistencyCritic functionality."""

    @pytest.mark.asyncio
    async def test_self_consistency_consensus(self, mock_llm_client, sample_result):
        """Test self-consistency consensus building."""
        # Mock multiple evaluations with slight variations
        responses = [
            MockLLMResponse(
                """{
                "feedback": "Needs more detail",
                "suggestions": ["Add examples"],
                "needs_improvement": true,
                "confidence": 0.8
            }"""
            ),
            MockLLMResponse(
                """{
                "feedback": "Too brief",
                "suggestions": ["Expand content", "Add examples"],
                "needs_improvement": true,
                "confidence": 0.85
            }"""
            ),
            MockLLMResponse(
                """{
                "feedback": "Lacks depth",
                "suggestions": ["Add examples", "Include data"],
                "needs_improvement": true,
                "confidence": 0.75
            }"""
            ),
        ]

        mock_llm_client.complete.side_effect = responses

        critic = SelfConsistencyCritic(num_samples=3)
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "self_consistency"
        assert result.needs_improvement is True
        # Should have consensus feedback
        assert "consensus" in result.feedback.lower()
        # Should have common suggestions
        assert "Add examples" in result.suggestions


class TestPromptCritic:
    """Test PromptCritic functionality."""

    @pytest.mark.asyncio
    async def test_prompt_critic_custom(self, mock_llm_client, sample_result):
        """Test prompt critic with custom evaluation criteria."""
        custom_prompt = "Evaluate for technical accuracy and innovation"

        mock_response = MockLLMResponse(
            """{
            "feedback": "Lacks technical depth and innovative insights",
            "suggestions": ["Add technical details", "Include innovative applications"],
            "needs_improvement": true,
            "confidence": 0.88
        }"""
        )
        mock_llm_client.complete.return_value = mock_response

        critic = PromptCritic(custom_prompt=custom_prompt)
        result = await critic.critique("AI is transforming the world", sample_result)

        assert result.critic == "prompt"
        assert result.needs_improvement is True
        # Verify custom prompt was used
        messages = mock_llm_client.complete.call_args[0][0]
        assert any(custom_prompt in msg.get("content", "") for msg in messages)


class TestCriticErrorHandling:
    """Test error handling across all critics."""

    @pytest.mark.asyncio
    async def test_critic_error_recovery(self, mock_llm_client, sample_result):
        """Test that critics handle errors gracefully."""
        # Mock an API error
        mock_llm_client.complete.side_effect = Exception("API Error")

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
            result = await critic.critique("Test text", sample_result)

            # Should return valid CritiqueResult even on error
            assert isinstance(result, CritiqueResult)
            assert result.critic == critic.name
            assert "Error" in result.feedback
            assert result.confidence == 0.0
            assert result.needs_improvement is True
            assert len(result.suggestions) > 0


class TestCriticIntegration:
    """Test critics working together."""

    @pytest.mark.asyncio
    async def test_multiple_critics_sequence(self, mock_llm_client):
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
            MockLLMResponse(
                """{
                "feedback": "Reflexion: needs examples",
                "suggestions": ["Add examples"],
                "needs_improvement": true,
                "confidence": 0.8
            }"""
            ),
            MockLLMResponse(
                """{
                "feedback": "Constitutional: needs ethical discussion",
                "suggestions": ["Discuss ethics"],
                "needs_improvement": true,
                "confidence": 0.85
            }"""
            ),
        ]

        mock_llm_client.complete.side_effect = responses

        # Run multiple critics
        critics = [ReflexionCritic(), ConstitutionalCritic()]

        for critic in critics:
            critique = await critic.critique(result.final_text, result)
            result.add_critique(critique)

        # Verify both critiques were added
        assert len(result.critiques) == 2
        assert result.critiques[0].critic == "reflexion"
        assert result.critiques[1].critic == "constitutional"
        assert all(c.needs_improvement for c in result.critiques)
