"""Tests for the text generation engine module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.engine.generation import ImprovementResponse, TextGenerator
from sifaka.core.models import (
    CritiqueResult,
    SifakaResult,
    ValidationResult,
)


class TestImprovementResponse:
    """Test the ImprovementResponse model."""

    def test_creation_minimal(self):
        """Test creating response with minimal fields."""
        response = ImprovementResponse(improved_text="Better text")
        assert response.improved_text == "Better text"
        assert response.changes_made == []
        assert response.confidence == 0.8  # default

    def test_creation_full(self):
        """Test creating response with all fields."""
        response = ImprovementResponse(
            improved_text="Much better text",
            changes_made=["Fixed grammar", "Improved clarity"],
            confidence=0.95,
        )
        assert response.improved_text == "Much better text"
        assert len(response.changes_made) == 2
        assert response.confidence == 0.95


class TestTextGenerator:
    """Test the TextGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a text generator instance."""
        return TextGenerator(model="gpt-4o-mini", temperature=0.7)

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Current version")

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.model == "gpt-4o-mini"
        assert generator.temperature == 0.7
        assert generator._client is None

    def test_client_property(self, generator):
        """Test lazy client initialization."""
        with patch("sifaka.core.engine.generation.LLMManager") as mock_manager:
            mock_client = Mock()
            mock_manager.get_client.return_value = mock_client

            # First access creates client
            client = generator.client
            assert client == mock_client
            mock_manager.get_client.assert_called_once_with(
                model="gpt-4o-mini", temperature=0.7
            )

            # Second access reuses client
            client2 = generator.client
            assert client2 == mock_client
            assert mock_manager.get_client.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_improvement_success(self, generator, sample_result):
        """Test successful text improvement generation."""
        # Mock the improvement response
        mock_response = ImprovementResponse(
            improved_text="Significantly improved text",
            changes_made=["Enhanced clarity", "Fixed grammar"],
            confidence=0.9,
        )

        # Mock the PydanticAI agent
        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = Mock()
        mock_client.create_agent = Mock(return_value=mock_agent)

        generator._client = mock_client
        improved_text, prompt = await generator.generate_improvement(
            "Current text", sample_result
        )

        assert improved_text == "Significantly improved text"
        assert prompt is not None
        assert "Current text:" in prompt
        mock_client.create_agent.assert_called_once()
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_improvement_no_change(self, generator, sample_result):
        """Test when improvement returns same text."""
        # Mock response with no change
        mock_response = ImprovementResponse(
            improved_text="Current text",  # Same as input
            changes_made=[],
            confidence=0.5,
        )

        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = Mock()
        mock_client.create_agent = Mock(return_value=mock_agent)

        generator._client = mock_client
        improved_text, prompt = await generator.generate_improvement(
            "Current text", sample_result
        )

        assert improved_text is None
        assert prompt is not None

    @pytest.mark.asyncio
    async def test_generate_improvement_empty_result(self, generator, sample_result):
        """Test when improvement returns empty text."""
        # Mock response with empty text
        mock_response = ImprovementResponse(
            improved_text="", changes_made=[], confidence=0.0
        )

        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = Mock()
        mock_client.create_agent = Mock(return_value=mock_agent)

        generator._client = mock_client
        improved_text, prompt = await generator.generate_improvement(
            "Current text", sample_result
        )

        assert improved_text is None
        assert prompt is not None

    @pytest.mark.asyncio
    async def test_generate_improvement_exception(self, generator, sample_result):
        """Test exception handling during improvement."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("API Error"))

        mock_client = Mock()
        mock_client.create_agent = Mock(return_value=mock_agent)

        generator._client = mock_client
        improved_text, prompt = await generator.generate_improvement(
            "Current text", sample_result
        )

        assert improved_text is None
        assert prompt is not None

    @pytest.mark.asyncio
    async def test_generate_improvement_with_show_prompt(
        self, generator, sample_result, capsys
    ):
        """Test improvement with prompt display."""
        mock_response = ImprovementResponse(improved_text="Better text")

        mock_agent_result = Mock()
        mock_agent_result.output = mock_response

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        mock_client = Mock()
        mock_client.create_agent = Mock(return_value=mock_agent)

        generator._client = mock_client
        improved_text, _ = await generator.generate_improvement(
            "Current text", sample_result, show_prompt=True
        )

        captured = capsys.readouterr()
        assert "IMPROVEMENT PROMPT" in captured.out
        assert "Current text:" in captured.out
        assert "=" * 80 in captured.out

    def test_build_improvement_prompt_basic(self, generator, sample_result):
        """Test basic prompt building."""
        prompt = generator._build_improvement_prompt("Test text", sample_result)

        assert "Please improve the following text" in prompt
        assert "Current text:\nTest text\n" in prompt
        assert "Provide an improved version" in prompt

    def test_build_improvement_prompt_with_validations(self, generator, sample_result):
        """Test prompt building with validation feedback."""
        # Add validation failures
        sample_result.validations.append(
            ValidationResult(
                validator="LengthValidator",
                passed=False,
                details="Text too short (50 < 100 characters)",
            )
        )
        sample_result.validations.append(
            ValidationResult(
                validator="FormatValidator",
                passed=False,
                details="Missing required header",
            )
        )
        sample_result.validations.append(
            ValidationResult(
                validator="ContentValidator", passed=True, details="Content is valid"
            )
        )

        prompt = generator._build_improvement_prompt("Test text", sample_result)

        assert "Validation issues:" in prompt
        assert "LengthValidator: Text too short" in prompt
        assert "FormatValidator: Missing required header" in prompt
        assert "ContentValidator" not in prompt  # Only failures included

    def test_build_improvement_prompt_with_critiques(self, generator, sample_result):
        """Test prompt building with critique feedback."""
        # Add critiques
        sample_result.critiques.append(
            CritiqueResult(
                critic="clarity",
                feedback="Text lacks clarity in the introduction",
                suggestions=["Add topic sentence", "Define key terms", "Use examples"],
                needs_improvement=True,
                confidence=0.8,
            )
        )
        sample_result.critiques.append(
            CritiqueResult(
                critic="accuracy",
                feedback="All facts are accurate",
                suggestions=[],
                needs_improvement=False,
                confidence=0.95,
            )
        )

        prompt = generator._build_improvement_prompt("Test text", sample_result)

        assert "Critic feedback:" in prompt
        assert "clarity:" in prompt
        assert "Text lacks clarity" in prompt
        assert "Suggestions:" in prompt
        assert "Add topic sentence" in prompt
        assert "accuracy:" not in prompt  # Only needs_improvement=True included

    def test_build_improvement_prompt_with_all_feedback(self, generator):
        """Test prompt building with both validations and critiques."""
        result = SifakaResult(original_text="Original", final_text="Current")

        # Add various feedback
        result.validations.append(
            ValidationResult(validator="Length", passed=False, details="Too short")
        )
        result.critiques.append(
            CritiqueResult(
                critic="style",
                feedback="Needs better flow",
                suggestions=["Use transitions"],
                needs_improvement=True,
            )
        )

        prompt = generator._build_improvement_prompt("Test text", result)

        assert "Validation issues:" in prompt
        assert "Critic feedback:" in prompt
        assert "Length: Too short" in prompt
        assert "style:" in prompt
        assert "Needs better flow" in prompt

    def test_format_validation_feedback(self, generator):
        """Test validation feedback formatting."""
        result = SifakaResult(original_text="Original", final_text="Current")

        # Add multiple validations
        for i in range(7):
            result.validations.append(
                ValidationResult(
                    validator=f"Validator{i}",
                    passed=i % 2 == 0,  # Alternate pass/fail
                    details=f"Issue {i}",
                )
            )

        feedback = generator._format_validation_feedback(result)

        # Should only include failures from last 5
        assert "Validator2" not in feedback  # Too old
        assert "Validator3: Issue 3" in feedback
        assert "Validator5: Issue 5" in feedback
        assert "Validator4" not in feedback  # Passed

    def test_format_critique_feedback_basic(self, generator):
        """Test basic critique feedback formatting."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="clarity",
                feedback="Needs clearer structure",
                suggestions=["Add headings", "Use bullet points"],
                needs_improvement=True,
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "clarity:" in feedback
        assert "Needs clearer structure" in feedback
        assert "Suggestions:" in feedback
        assert "Add headings" in feedback
        assert "Use bullet points" in feedback

    def test_format_critique_feedback_with_metadata_self_rag(self, generator):
        """Test critique feedback with SelfRAG metadata."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="self_rag",
                feedback="Needs evidence",
                suggestions=["Add citations"],
                needs_improvement=True,
                metadata={
                    "overall_relevance": False,
                    "overall_support": False,
                    "retrieval_opportunities": [
                        {"location": "Paragraph 2", "reason": "Missing data"},
                        {"location": "Conclusion", "reason": "Need examples"},
                    ],
                },
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "self_rag:" in feedback
        assert "⚠️ Content relevance issues detected (ISREL)" in feedback
        assert "⚠️ Evidence support needed (ISSUP)" in feedback
        assert "Retrieval would help with:" in feedback
        assert "Paragraph 2: Missing data" in feedback

    def test_format_critique_feedback_with_metadata_self_refine(self, generator):
        """Test critique feedback with SelfRefine metadata."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="self_refine",
                feedback="Needs refinement",
                suggestions=["Improve flow"],
                needs_improvement=True,
                metadata={
                    "refinement_areas": [
                        {"area": "Introduction", "target_state": "More engaging"},
                        {"area": "Transitions", "target_state": "Smoother flow"},
                    ]
                },
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "Areas needing refinement:" in feedback
        assert "Introduction: More engaging" in feedback
        assert "Transitions: Smoother flow" in feedback

    def test_format_critique_feedback_with_metadata_n_critics(self, generator):
        """Test critique feedback with NCritics metadata."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="n_critics",
                feedback="Mixed reviews",
                suggestions=["Address concerns"],
                needs_improvement=True,
                metadata={
                    "consensus_score": 0.4,
                    "perspective_assessments": [
                        {"perspective": "Clarity", "score": 0.3},
                        {"perspective": "Accuracy", "score": 0.9},
                    ],
                },
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "Low consensus (0.4) - consider diverse perspectives" in feedback

    def test_format_critique_feedback_with_metadata_constitutional(self, generator):
        """Test critique feedback with Constitutional metadata."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="constitutional",
                feedback="Violates principles",
                suggestions=["Revise content"],
                needs_improvement=True,
                metadata={
                    "requires_major_revision": True,
                    "revision_proposals": [
                        {
                            "original_snippet": "This is problematic text that needs fixing",
                            "revised_snippet": "This is improved text that follows principles",
                        }
                    ],
                },
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "⚠️ Major constitutional revision required" in feedback
        assert "Constitutional revisions available:" in feedback
        assert "This is problematic text that " in feedback
        assert "→" in feedback

    def test_format_critique_feedback_with_metadata_meta_rewarding(self, generator):
        """Test critique feedback with MetaRewarding metadata."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="meta_rewarding",
                feedback="Meta-evaluated",
                suggestions=["Apply top suggestion"],
                needs_improvement=True,
                metadata={
                    "improvement_delta": 0.15,
                    "suggestion_preferences": [
                        {
                            "suggestion": "This is the best suggestion based on meta-evaluation",
                            "score": 0.9,
                        }
                    ],
                },
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "✓ Critique refined with 15.0% improvement" in feedback
        assert (
            "Top suggestion: This is the best suggestion based on meta-evalu"
            in feedback
        )

    def test_format_critique_feedback_limit_suggestions(self, generator):
        """Test that suggestions are limited to 3."""
        result = SifakaResult(original_text="Original", final_text="Current")

        result.critiques.append(
            CritiqueResult(
                critic="test",
                feedback="Many issues",
                suggestions=[f"Fix {i}" for i in range(10)],
                needs_improvement=True,
            )
        )

        feedback = generator._format_critique_feedback(result)

        assert "Fix 0" in feedback
        assert "Fix 1" in feedback
        assert "Fix 2" in feedback
        assert "Fix 3" not in feedback  # Limited to 3

    def test_add_critic_insights_invalid_metadata(self, generator):
        """Test critic insights with invalid metadata types."""
        critique = CritiqueResult(
            critic="self_rag",
            feedback="Test",
            suggestions=[],
            needs_improvement=True,
            metadata={
                "retrieval_opportunities": [
                    "Not a dict",  # Invalid type
                    {"location": "Valid", "reason": "Test"},
                ]
            },
        )

        lines = []
        generator._add_critic_insights(critique, lines)

        # Should handle invalid types gracefully
        assert any("Valid: Test" in line for line in lines)
        assert not any("Not a dict" in line for line in lines)
