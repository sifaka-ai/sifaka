"""Tests for Self-Consistency critic."""

from unittest.mock import AsyncMock, patch

import pytest

from sifaka.core.config import Config
from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.self_consistency import (
    SelfConsistencyCritic,
    SelfConsistencyResponse,
)


class TestSelfConsistencyResponse:
    """Test the SelfConsistencyResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = SelfConsistencyResponse(
            feedback="Consensus: needs clarity improvements",
            suggestions=["Simplify language"],
            needs_improvement=True,
        )
        assert response.feedback == "Consensus: needs clarity improvements"
        assert response.confidence == 0.8  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        eval1 = {
            "feedback": "Needs work",
            "suggestions": ["Fix A"],
            "needs_improvement": True,
            "confidence": 0.7,
        }
        eval2 = {
            "feedback": "Almost there",
            "suggestions": ["Fix A", "Fix B"],
            "needs_improvement": True,
            "confidence": 0.8,
        }

        response = SelfConsistencyResponse(
            feedback="Consensus feedback",
            suggestions=["Fix A", "Fix B"],
            needs_improvement=True,
            confidence=0.85,
            metadata={
                "individual_evaluations": [eval1, eval2],
                "consistency_score": 0.9,
                "common_themes": ["clarity", "structure"],
                "divergent_points": ["tone assessment"],
                "evaluation_variance": 0.1,
                "num_evaluations": 2,
            },
        )

        assert len(response.metadata["individual_evaluations"]) == 2
        assert response.metadata["consistency_score"] == 0.9
        assert "clarity" in response.metadata["common_themes"]
        assert "tone assessment" in response.metadata["divergent_points"]
        assert response.metadata["evaluation_variance"] == 0.1


class TestSelfConsistencyCritic:
    """Test the SelfConsistencyCritic class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        critic = SelfConsistencyCritic()
        assert critic.name == "self_consistency"
        assert critic.model == "gpt-3.5-turbo"  # Default from Config
        assert critic.temperature == 0.8  # Higher for diversity
        assert critic.num_samples == 3

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Config()
        config.critic.self_consistency_num_samples = 5
        critic = SelfConsistencyCritic(config=config)
        assert critic.config == config
        assert critic.num_samples == 5

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        critic = SelfConsistencyCritic(
            model="gpt-4", temperature=0.9, num_samples=5, api_key="test-key"
        )
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.9
        assert critic.num_samples == 3  # Config is None, so default is used

    def test_get_response_type(self):
        """Test that critic uses SelfConsistencyResponse."""
        critic = SelfConsistencyCritic()
        assert critic._get_response_type() == SelfConsistencyResponse

    @pytest.mark.asyncio
    async def test_create_messages(self, sample_result):
        """Test message creation for self-consistency evaluation."""
        critic = SelfConsistencyCritic()
        messages = await critic._create_messages("Test text", sample_result)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "independent text evaluator" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Evaluate this text" in user_content
        assert "Test text" in user_content

    @pytest.mark.asyncio
    async def test_create_messages_with_context(self, sample_result):
        """Test message creation with previous context."""
        # Add a previous critique
        sample_result.critiques.append(
            CritiqueResult(
                critic="self_consistency",
                feedback="Previous feedback",
                suggestions=["Previous suggestion"],
            )
        )

        critic = SelfConsistencyCritic()
        messages = await critic._create_messages("Test text", sample_result)

        user_content = messages[1]["content"]
        assert "Previous feedback:" in user_content
        assert "Previous feedback" in user_content

    @pytest.mark.asyncio
    async def test_get_single_evaluation(self, sample_result):
        """Test _get_single_evaluation method."""
        critic = SelfConsistencyCritic()

        # Mock the CritiqueResult response
        mock_response = CritiqueResult(
            critic="self_consistency_sample_1",
            feedback="Needs improvement",
            suggestions=["Add examples"],
            needs_improvement=True,
            confidence=0.8,
        )

        # Mock the parent's critique method
        with patch.object(
            critic, "critique", new=AsyncMock(return_value=mock_response)
        ):
            with patch("sifaka.critics.self_consistency.super") as mock_super:
                mock_super().critique = AsyncMock(return_value=mock_response)
                evaluation = await critic._get_single_evaluation(
                    "Test text", sample_result, 1
                )

            assert evaluation.feedback == "Needs improvement"
            assert len(evaluation.suggestions) == 1
            assert evaluation.needs_improvement is True

    def test_build_consensus(self, sample_result):
        """Test _build_consensus method."""
        critic = SelfConsistencyCritic()

        eval1 = CritiqueResult(
            critic="self_consistency_sample_1",
            feedback="Needs clarity improvements",
            suggestions=["Simplify sentences", "Add examples"],
            needs_improvement=True,
            confidence=0.7,
        )
        eval2 = CritiqueResult(
            critic="self_consistency_sample_2",
            feedback="Good but verbose, needs clarity",
            suggestions=["Simplify sentences", "Remove redundancy"],
            needs_improvement=True,
            confidence=0.8,
        )
        eval3 = CritiqueResult(
            critic="self_consistency_sample_3",
            feedback="Structure issues, clarity problems",
            suggestions=["Reorganize sections", "Add examples"],
            needs_improvement=True,
            confidence=0.75,
        )

        evaluations = [eval1, eval2, eval3]
        result = critic._build_consensus(evaluations)

        assert result.critic == "self_consistency"
        assert "Based on 3 independent evaluations" in result.feedback
        assert "All evaluations agree that improvement is needed" in result.feedback
        assert len(result.suggestions) <= 5  # Top 5 suggestions
        assert result.needs_improvement is True
        assert result.confidence == 0.75  # Average of 0.7, 0.8, 0.75
        assert result.metadata["num_evaluations"] == 3

    @pytest.mark.asyncio
    async def test_critique_success(self, sample_result):
        """Test successful critique flow with multiple evaluations."""
        critic = SelfConsistencyCritic(num_samples=3)  # Default is 3

        # Mock individual CritiqueResults
        eval1 = CritiqueResult(
            critic="self_consistency_sample_1",
            feedback="Needs work on clarity and structure",
            suggestions=["Fix A"],
            needs_improvement=True,
            confidence=0.7,
        )
        eval2 = CritiqueResult(
            critic="self_consistency_sample_2",
            feedback="Almost good but needs clarity",
            suggestions=["Fix A", "Fix B"],
            needs_improvement=True,
            confidence=0.8,
        )
        eval3 = CritiqueResult(
            critic="self_consistency_sample_3",
            feedback="Needs improvement in multiple areas",
            suggestions=["Fix B", "Fix C"],
            needs_improvement=True,
            confidence=0.75,
        )

        # Mock _get_single_evaluation to return our evaluations
        async def mock_get_single_evaluation(text, result, sample_num):
            if sample_num == 1:
                return eval1
            elif sample_num == 2:
                return eval2
            else:
                return eval3

        with patch.object(
            critic, "_get_single_evaluation", side_effect=mock_get_single_evaluation
        ):
            result = await critic.critique("Test text", sample_result)

            assert result.critic == "self_consistency"
            assert "Based on 3 independent evaluations" in result.feedback
            assert "All evaluations agree that improvement is needed" in result.feedback
            assert "Fix A" in result.suggestions  # Most common suggestion
            assert result.needs_improvement is True
            assert result.confidence == 0.75  # Average of 0.7, 0.8, and 0.75
            assert result.metadata["num_evaluations"] == 3
            # Both evaluations agree on improvement

    @pytest.mark.asyncio
    async def test_critique_high_consistency(self, sample_result):
        """Test critique when evaluations are highly consistent."""
        critic = SelfConsistencyCritic(num_samples=3)

        # Create very similar CritiqueResults
        evals = []
        for i in range(3):
            eval = CritiqueResult(
                critic=f"self_consistency_sample_{i + 1}",
                feedback="Excellent writing with clear structure",
                suggestions=[],
                needs_improvement=False,
                confidence=0.95,
            )
            evals.append(eval)

        # Mock _get_single_evaluation to return our evaluations
        async def mock_get_single_evaluation(text, result, sample_num):
            return evals[sample_num - 1]

        with patch.object(
            critic, "_get_single_evaluation", side_effect=mock_get_single_evaluation
        ):
            result = await critic.critique("Excellent text", sample_result)

            assert result.needs_improvement is False
            assert len(result.suggestions) == 0
            assert (
                pytest.approx(result.confidence) == 0.95
            )  # Handle floating point precision
            assert "All evaluations agree the text is satisfactory" in result.feedback
            assert "Based on 3 independent evaluations:" in result.feedback
            # All 3 evaluations agree no improvement needed

    def test_provider_configuration(self):
        """Test provider configuration."""
        from sifaka.core.llm_client import Provider

        critic = SelfConsistencyCritic(provider=Provider.ANTHROPIC)
        assert critic.provider == Provider.ANTHROPIC

        critic = SelfConsistencyCritic(provider="openai")
        assert critic.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_critique_with_evaluation_failure(self, sample_result):
        """Test handling when one evaluation fails."""
        critic = SelfConsistencyCritic(num_samples=3)

        # Mock evaluations - one will fail
        eval1 = CritiqueResult(
            critic="self_consistency_sample_1",
            feedback="Good",
            suggestions=["Minor fix"],
            needs_improvement=True,
            confidence=0.8,
        )
        eval3 = CritiqueResult(
            critic="self_consistency_sample_3",
            feedback="Needs work",
            suggestions=["Major fix"],
            needs_improvement=True,
            confidence=0.7,
        )

        # Mock _get_single_evaluation to return evaluations with one failure
        async def mock_get_single_evaluation(text, result, sample_num):
            if sample_num == 1:
                return eval1
            elif sample_num == 2:
                raise Exception("API Error")
            else:
                return eval3

        # The gather will handle the exception and filter it out
        with patch.object(
            critic, "_get_single_evaluation", side_effect=mock_get_single_evaluation
        ):
            result = await critic.critique("Test text", sample_result)

            # Should still work with 2 evaluations
            assert result.critic == "self_consistency"
            assert (
                "Based on 2 independent evaluations" in result.feedback
            )  # Only 2 valid
            assert result.needs_improvement is True
            assert result.confidence == 0.75  # Average of 0.8 and 0.7
            assert result.metadata["num_evaluations"] == 2
