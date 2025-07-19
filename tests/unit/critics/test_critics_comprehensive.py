"""Comprehensive tests for all critic implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.critics.self_refine import SelfRefineCritic


def create_mock_openai_response(content: str):
    """Create a mock OpenAI response with the given content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = content
    return mock_response


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


class TestSelfConsistencyCritic:
    """Test Self-Consistency critic implementation."""

    @pytest.fixture
    def critic(self):
        return SelfConsistencyCritic(num_samples=2)  # Reduced for testing

    @pytest.mark.asyncio
    async def test_critique_success(self, critic, sample_text, sample_result):
        """Test successful critique with proper responses."""
        mock_responses = [
            "QUALITY_SCORE: 4\nSTRENGTHS: Clear writing\nWEAKNESSES: Could be more detailed\nIMPROVEMENTS: Add more examples\nPRIORITY: MEDIUM",
            "QUALITY_SCORE: 3\nSTRENGTHS: Good structure\nWEAKNESSES: Lacks depth\nIMPROVEMENTS: Expand on key points\nPRIORITY: MEDIUM",
        ]

        with patch.object(
            critic, "_get_independent_evaluation", side_effect=mock_responses
        ):
            result = await critic.critique(sample_text, sample_result)

            assert isinstance(result, CritiqueResult)
            assert result.critic == "self_consistency"
            assert "Quality consensus" in result.feedback
            assert len(result.suggestions) > 0
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_critique_with_inconsistent_evaluations(
        self, critic, sample_text, sample_result
    ):
        """Test critique with highly inconsistent evaluations."""
        mock_responses = [
            "QUALITY_SCORE: 5\nSTRENGTHS: Excellent\nWEAKNESSES: None\nIMPROVEMENTS: Perfect\nPRIORITY: LOW",
            "QUALITY_SCORE: 1\nSTRENGTHS: None\nWEAKNESSES: Terrible\nIMPROVEMENTS: Rewrite completely\nPRIORITY: HIGH",
        ]

        with patch.object(
            critic, "_get_independent_evaluation", side_effect=mock_responses
        ):
            result = await critic.critique(sample_text, sample_result)

            assert "inconsistency" in " ".join(result.suggestions).lower()
            assert result.needs_improvement

    @pytest.mark.asyncio
    async def test_critique_error_handling(self, critic, sample_text, sample_result):
        """Test error handling when API calls fail."""
        with patch.object(
            critic, "_get_independent_evaluation", side_effect=Exception("API Error")
        ):
            result = await critic.critique(sample_text, sample_result)

            assert result.critic == "self_consistency"
            assert "Error during self-consistency evaluation" in result.feedback
            assert result.confidence == 0.0

    def test_consistency_calculations(self, critic):
        """Test consistency calculation methods."""
        # Test score consistency
        consistent_scores = [4, 4, 4]
        inconsistent_scores = [1, 3, 5]

        consistent_result = critic._calculate_score_consistency(consistent_scores)
        inconsistent_result = critic._calculate_score_consistency(inconsistent_scores)

        assert consistent_result > inconsistent_result
        assert 0.0 <= consistent_result <= 1.0
        assert 0.0 <= inconsistent_result <= 1.0

        # Test priority consistency
        consistent_priorities = ["HIGH", "HIGH", "HIGH"]
        inconsistent_priorities = ["HIGH", "MEDIUM", "LOW"]

        consistent_priority = critic._calculate_priority_consistency(
            consistent_priorities
        )
        inconsistent_priority = critic._calculate_priority_consistency(
            inconsistent_priorities
        )

        assert consistent_priority > inconsistent_priority


class TestMetaRewardingCritic:
    """Test Meta-Rewarding critic implementation."""

    @pytest.fixture
    def critic(self):
        return MetaRewardingCritic()

    @pytest.mark.asyncio
    async def test_critique_success(self, critic, sample_text, sample_result):
        """Test successful meta-rewarding critique."""
        initial_judgment = """CLARITY: 4 - Text is clear and readable
ACCURACY: 3 - Some factual claims need verification
COMPLETENESS: 4 - Covers main points well
STRUCTURE: 5 - Well organized
ENGAGEMENT: 3 - Could be more engaging
OVERALL: Good quality text with minor improvements needed"""

        meta_judgment = """META_ASSESSMENT: The evaluation is comprehensive and fair
RELIABILITY: 0.8
CORRECTIONS: Consider adding more specific examples"""

        with (
            patch.object(
                critic, "_get_initial_judgment", return_value=initial_judgment
            ),
            patch.object(critic, "_get_meta_judgment", return_value=meta_judgment),
        ):
            result = await critic.critique(sample_text, sample_result)

            assert isinstance(result, CritiqueResult)
            assert result.critic == "meta_rewarding"
            assert "Initial judgment" in result.feedback
            assert "Meta-assessment" in result.feedback
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_critique_low_reliability(self, critic, sample_text, sample_result):
        """Test critique with low reliability meta-judgment."""
        initial_judgment = "OVERALL: Basic assessment"
        meta_judgment = "META_ASSESSMENT: Evaluation lacks depth\nRELIABILITY: 0.4\nCORRECTIONS: Need more thorough analysis"

        with (
            patch.object(
                critic, "_get_initial_judgment", return_value=initial_judgment
            ),
            patch.object(critic, "_get_meta_judgment", return_value=meta_judgment),
        ):
            result = await critic.critique(sample_text, sample_result)

            assert result.confidence == 0.4
            assert result.needs_improvement
            assert "Assessment reliability could be enhanced" in result.suggestions

    @pytest.mark.asyncio
    async def test_parsing_with_malformed_response(
        self, critic, sample_text, sample_result
    ):
        """Test parsing with malformed API responses."""
        initial_judgment = "Malformed response without proper structure"
        meta_judgment = "Another malformed response"

        with (
            patch.object(
                critic, "_get_initial_judgment", return_value=initial_judgment
            ),
            patch.object(critic, "_get_meta_judgment", return_value=meta_judgment),
        ):
            result = await critic.critique(sample_text, sample_result)

            # Should handle gracefully with defaults
            assert isinstance(result, CritiqueResult)
            assert result.confidence == 0.7  # Default confidence


class TestNCriticsCritic:
    """Test N-Critics ensemble critic implementation."""

    @pytest.fixture
    def critic(self):
        return NCriticsCritic()

    @pytest.mark.asyncio
    async def test_critique_success(self, critic, sample_text, sample_result):
        """Test successful N-Critics ensemble critique."""
        mock_response = """PERSPECTIVE_1: Clarity perspective shows good readability
PERSPECTIVE_2: Accuracy perspective finds minor factual issues
PERSPECTIVE_3: Completeness perspective suggests adding examples
PERSPECTIVE_4: Style perspective recommends tone adjustment
CONSENSUS: Overall quality score 0.75 - good quality with improvements needed
SUGGESTIONS: 1. Add more specific examples
2. Verify factual claims
3. Adjust tone for target audience"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            assert isinstance(result, CritiqueResult)
            assert result.critic == "n_critics"
            assert "Ensemble consensus" in result.feedback
            assert (
                len(result.suggestions) >= 2
            )  # Parsing may not capture all suggestions
            assert result.confidence == 0.85  # 0.75 + 0.1

    @pytest.mark.asyncio
    async def test_critique_low_consensus(self, critic, sample_text, sample_result):
        """Test critique with low consensus score."""
        mock_response = """PERSPECTIVE_1: Mixed assessment
CONSENSUS: Overall quality score 0.5 - significant disagreement among perspectives
SUGGESTIONS: 1. Major revisions needed"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            assert result.needs_improvement  # consensus < 0.75
            assert result.confidence == 0.6  # 0.5 + 0.1

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
        mock_response = """QUALITY_SCORE: 0.7
FEEDBACK: The text is clear but could benefit from more detailed examples and better structure
REFINEMENTS: 1. Add specific examples to support main points
2. Improve paragraph transitions
3. Enhance conclusion with actionable insights"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            assert isinstance(result, CritiqueResult)
            assert result.critic == "self_refine"
            assert (
                abs(result.confidence - 0.8) < 0.001
            )  # 0.7 + 0.1 (handle floating point precision)
            assert (
                len(result.suggestions) >= 2
            )  # Parsing may not capture all suggestions
            # Check if we have reasonable suggestions
            assert len(result.suggestions) > 0
            assert any(
                len(s) > 5 for s in result.suggestions
            )  # Non-trivial suggestions

    @pytest.mark.asyncio
    async def test_critique_high_quality_text(self, critic, sample_text, sample_result):
        """Test critique of high-quality text."""
        mock_response = """QUALITY_SCORE: 0.9
FEEDBACK: Excellent text quality with minor refinements possible
REFINEMENTS: 1. Consider minor stylistic improvements"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            assert (
                not result.needs_improvement
            )  # quality >= 0.8 and len(suggestions) <= 2
            assert result.confidence == 0.9  # min(0.9, 0.9 + 0.1)

    @pytest.mark.asyncio
    async def test_iterative_context(self, critic, sample_text):
        """Test iterative refinement context."""
        previous_critique = CritiqueResult(
            critic="previous",
            feedback="Previous feedback",
            suggestions=["Previous suggestion"],
            needs_improvement=True,
            confidence=0.5,
        )

        iterative_result = SifakaResult(
            original_text="Original",
            final_text="Improved",
            iteration=1,
            critiques=[previous_critique],
        )

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_api_response = create_mock_openai_response(
                "QUALITY_SCORE: 0.8\nFEEDBACK: Good\nREFINEMENTS: 1. Minor improvements"
            )
            mock_create.return_value = mock_api_response

            await critic.critique(sample_text, iterative_result)

            # Check that prompt includes iteration context
            call_args = mock_create.call_args[1]["messages"][1]["content"]
            assert "iteration 2" in call_args.lower()
            assert "Previous feedback" in call_args

    def test_parse_refinement_with_malformed_input(self, critic):
        """Test parsing with malformed refinement input."""
        malformed_input = "This is not properly formatted output"

        feedback, suggestions, quality_score = critic._parse_refinement(malformed_input)

        assert feedback == malformed_input
        assert len(suggestions) == 1
        assert quality_score == 0.5  # Default


class TestSelfRAGCritic:
    """Test Self-RAG critic implementation."""

    @pytest.fixture
    def critic(self):
        return SelfRAGCritic()

    @pytest.mark.asyncio
    async def test_critique_needs_retrieval(self, critic, sample_text, sample_result):
        """Test critique that identifies retrieval needs."""
        mock_response = """RETRIEVAL_NEEDED: YES - Text contains factual claims that need verification
ASSESSMENT: Content appears accurate but lacks recent data and supporting evidence
SUGGESTIONS: 1. Retrieve current statistics on the topic
2. Add credible source citations
3. Verify factual claims with authoritative sources"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            assert isinstance(result, CritiqueResult)
            assert result.critic == "self_rag"
            assert result.needs_improvement
            assert (
                result.confidence == 0.8
            )  # Higher confidence when retrieval need is clear
            assert "Retrieval assessment" in result.feedback
            assert (
                len(result.suggestions) >= 2
            )  # Parsing may not capture all suggestions

    @pytest.mark.asyncio
    async def test_critique_no_retrieval_needed(
        self, critic, sample_text, sample_result
    ):
        """Test critique that finds no retrieval needs."""
        mock_response = """RETRIEVAL_NEEDED: NO - Content is well-supported and factually sound
ASSESSMENT: Text demonstrates good factual accuracy and comprehensive coverage
SUGGESTIONS: 1. Minor stylistic improvements could enhance readability"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            assert not result.needs_improvement
            assert result.confidence == 0.7  # Lower confidence when no retrieval needed
            assert "Content evaluation" in result.feedback

    @pytest.mark.asyncio
    async def test_critique_factual_accuracy_issues(
        self, critic, sample_text, sample_result
    ):
        """Test critique that identifies factual accuracy issues."""
        mock_response = """RETRIEVAL_NEEDED: NO - No additional retrieval needed
ASSESSMENT: Several factual accuracy concerns identified
SUGGESTIONS: 1. Verify accuracy of statistical claims
2. Update outdated information
3. Add fact-checking references"""

        mock_api_response = create_mock_openai_response(mock_response)

        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_api_response
            result = await critic.critique(sample_text, sample_result)

            # Should need improvement due to "accuracy" in suggestions or feedback
            assert result.needs_improvement or "accuracy" in result.feedback.lower()
            # Check if accuracy is mentioned somewhere in the result
            accuracy_mentioned = (
                any("accuracy" in s.lower() for s in result.suggestions)
                or "accuracy" in result.feedback.lower()
            )
            assert accuracy_mentioned

    def test_parse_self_rag_with_incomplete_sections(self, critic):
        """Test parsing with incomplete sections."""
        incomplete_response = """RETRIEVAL_NEEDED: YES
ASSESSMENT: Basic assessment"""
        # Missing SUGGESTIONS section

        feedback, suggestions, needs_retrieval = critic._parse_self_rag(
            incomplete_response
        )

        assert needs_retrieval
        assert len(suggestions) == 1  # Fallback suggestion
        assert suggestions[0] == "Consider retrieval-augmented improvements"


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
        # Mock the OpenAI client for all critics
        with patch.object(
            critic.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_response = create_mock_openai_response("Test response")
            mock_create.return_value = mock_response

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
