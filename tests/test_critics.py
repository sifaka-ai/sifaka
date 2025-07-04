"""Tests for critic implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.prompt import PromptCritic, create_academic_critic
from sifaka.core.models import SifakaResult, CritiqueResult
from sifaka.core.config import Config


class TestReflexionCritic:
    """Test Reflexion critic implementation."""

    def test_initialization(self):
        """Test critic initialization."""
        critic = ReflexionCritic(model="gpt-4", temperature=0.5)
        assert critic.model == "gpt-4"
        assert critic.temperature == 0.5
        assert critic.name == "reflexion"
        assert critic.config is not None
        assert critic.config.base_confidence == 0.65

    def test_build_context_first_iteration(self):
        """Test context building for first iteration."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        context = critic._build_context(result)
        assert "first iteration" in context.lower()

    def test_build_context_with_critiques(self):
        """Test context building with previous critiques."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add some critiques
        result.add_critique(
            critic="other_critic",
            feedback="Previous feedback",
            suggestions=["Previous suggestion"],
            needs_improvement=True,
        )

        context = critic._build_context(result)
        assert "Previous feedback" in context
        assert "Previous suggestion" in context

    def test_generate_critique_format(self):
        """Test critique generation format."""
        critic = ReflexionCritic()
        assert critic.config.response_format == "json"

        # Test that config can be customized
        custom_config = Config(response_format="structured")
        critic2 = ReflexionCritic(config=custom_config)
        assert critic2.config.response_format == "json"  # Reflexion forces JSON

    def test_parse_json_response(self):
        """Test parsing JSON response."""
        critic = ReflexionCritic()
        json_response = """{
    "feedback": "This text shows good structure but lacks depth.",
    "suggestions": ["Add more specific examples", "Improve the conclusion", "Include citations"],
    "needs_improvement": true,
    "confidence": 0.8
}"""

        result = critic._parse_json_response(json_response)
        assert "good structure" in result.feedback
        assert "lacks depth" in result.feedback
        assert len(result.suggestions) == 3
        assert "Add more specific examples" in result.suggestions
        assert result.needs_improvement is True
        assert result.confidence == 0.8

    def test_parse_text_response(self):
        """Test parsing unstructured text response."""
        critic = ReflexionCritic()
        text_response = "This is unstructured feedback without proper sections."

        result = critic._parse_text_response(text_response)
        assert result.feedback == text_response
        assert len(result.suggestions) >= 1  # Should have fallback suggestion
        assert result.confidence >= 0.0

    def test_assess_needs_improvement(self):
        """Test improvement need assessment."""
        critic = ReflexionCritic()

        # Should need improvement
        assert critic._assess_needs_improvement(
            "The text could be improved and needs more clarity",
            ["Fix this", "Change that"],
        )

        # Should not need improvement
        assert not critic._assess_needs_improvement(
            "The text is excellent and well-written", ["Minor suggestion"]
        )

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        critic = ReflexionCritic()

        # Short response
        conf1 = critic._calculate_confidence("Brief feedback", "Brief feedback")
        assert 0.0 <= conf1 <= 1.0

        # Long detailed response
        long_response = "A" * 600
        conf2 = critic._calculate_confidence(long_response, long_response)
        assert conf2 > conf1  # Longer response should have higher confidence

    def test_base_critic_functionality(self):
        """Test that base critic functionality works."""
        critic = ReflexionCritic()

        # Test extract list items
        text = """Here are suggestions:
1. First item
2. Second item
- Third item"""
        items = critic._extract_list_items(text)
        assert len(items) == 3
        assert "First item" in items
        assert "Third item" in items

    @pytest.mark.asyncio
    async def test_critique_success(self):
        """Test successful critique execution."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock OpenAI response with JSON format
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
    "feedback": "Good analysis of the text with clear structure.",
    "suggestions": ["Add more details", "Improve structure"],
    "needs_improvement": true,
    "confidence": 0.75
}"""

        with patch.object(
            critic.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            critique_result = await critic.critique("Test text", result)

            assert isinstance(critique_result, CritiqueResult)
            assert critique_result.critic == "reflexion"
            assert "Good analysis" in critique_result.feedback
            assert len(critique_result.suggestions) == 2
            assert critique_result.confidence == 0.75
            assert critique_result.needs_improvement is True

    @pytest.mark.asyncio
    async def test_critique_error_handling(self):
        """Test critique error handling."""
        critic = ReflexionCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock API error
        with patch.object(
            critic.client.chat.completions, "create", side_effect=Exception("API Error")
        ):
            critique_result = await critic.critique("Test text", result)

            assert isinstance(critique_result, CritiqueResult)
            assert critique_result.critic == "reflexion"
            assert "Error during reflection" in critique_result.feedback
            assert critique_result.confidence == 0.0


class TestPromptCritic:
    """Test configurable prompt critic."""

    def test_initialization_default(self):
        """Test default initialization."""
        critic = PromptCritic()
        assert critic.model == "gpt-4o-mini"
        assert critic.temperature == 0.5
        assert critic.custom_prompt is None
        assert critic.criteria == []
        assert critic.name == "prompt"

    def test_initialization_with_criteria(self):
        """Test initialization with custom criteria."""
        criteria = ["Clear structure", "Proper citations"]
        critic = PromptCritic(criteria=criteria, name_suffix="academic")
        assert critic.criteria == criteria
        assert critic.name == "prompt_academic"

    def test_initialization_with_custom_prompt(self):
        """Test initialization with custom prompt."""
        custom_prompt = "Evaluate this text for academic quality"
        critic = PromptCritic(custom_prompt=custom_prompt)
        assert critic.custom_prompt == custom_prompt

    def test_format_custom_prompt(self):
        """Test custom prompt formatting."""
        custom_prompt = "Evaluate for quality"
        critic = PromptCritic(custom_prompt=custom_prompt)

        formatted = critic._format_custom_prompt("Test text")
        assert custom_prompt in formatted
        assert "Test text" in formatted
        assert "ASSESSMENT:" in formatted

    def test_build_criteria_prompt_default(self):
        """Test building prompt with default criteria."""
        critic = PromptCritic()
        prompt = critic._build_criteria_prompt("Test text")

        assert "Test text" in prompt
        assert "Clarity and readability" in prompt
        assert "ASSESSMENT:" in prompt

    def test_build_criteria_prompt_custom(self):
        """Test building prompt with custom criteria."""
        criteria = ["Academic tone", "Proper methodology"]
        critic = PromptCritic(criteria=criteria)
        prompt = critic._build_criteria_prompt("Test text")

        assert "Academic tone" in prompt
        assert "Proper methodology" in prompt

    def test_parse_evaluation_structured(self):
        """Test parsing structured evaluation."""
        critic = PromptCritic()
        evaluation = """ASSESSMENT: The text meets most criteria well.
CRITERIA_MET: YES
SUGGESTIONS:
1. Add more examples
2. Improve conclusion"""

        feedback, suggestions, meets_criteria = critic._parse_evaluation(evaluation)
        assert "meets most criteria" in feedback
        assert meets_criteria is True
        assert len(suggestions) == 2

    def test_parse_evaluation_criteria_no(self):
        """Test parsing when criteria not met."""
        critic = PromptCritic()
        evaluation = """ASSESSMENT: The text has several issues.
CRITERIA_MET: NO
SUGGESTIONS: Fix the problems"""

        feedback, suggestions, meets_criteria = critic._parse_evaluation(evaluation)
        assert meets_criteria is False

    def test_calculate_criteria_specificity(self):
        """Test criteria specificity calculation."""
        # Custom prompt (most specific)
        custom_critic = PromptCritic(custom_prompt="Custom evaluation")
        assert custom_critic._calculate_criteria_specificity() == 0.2

        # Multiple criteria
        criteria_critic = PromptCritic(criteria=["A", "B", "C", "D", "E"])
        assert criteria_critic._calculate_criteria_specificity() == 0.15

        # No criteria
        default_critic = PromptCritic()
        assert default_critic._calculate_criteria_specificity() == 0.0

    def test_get_domain_indicators(self):
        """Test domain indicator extraction."""
        # Academic critic
        academic_critic = PromptCritic(name_suffix="academic")
        indicators = academic_critic._get_domain_indicators()
        assert "thesis" in indicators
        assert "research" in indicators

        # Business critic
        business_critic = PromptCritic(name_suffix="business")
        indicators = business_critic._get_domain_indicators()
        assert "professional" in indicators
        assert "strategy" in indicators

        # Generic critic
        generic_critic = PromptCritic()
        indicators = generic_critic._get_domain_indicators()
        assert indicators == []

    def test_create_academic_critic(self):
        """Test academic critic factory function."""
        critic = create_academic_critic()
        assert isinstance(critic, PromptCritic)
        assert critic.name == "prompt_academic"
        assert len(critic.criteria) > 0
        assert any("academic" in criterion.lower() for criterion in critic.criteria)

    @pytest.mark.asyncio
    async def test_critique_with_criteria(self):
        """Test critique with custom criteria."""
        criteria = ["Clear thesis", "Strong evidence"]
        critic = PromptCritic(criteria=criteria)
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """ASSESSMENT: Good academic structure.
CRITERIA_MET: YES
SUGGESTIONS:
1. Add more citations"""

        with patch.object(
            critic.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            critique_result = await critic.critique("Academic text", result)

            assert isinstance(critique_result, CritiqueResult)
            assert critique_result.critic == "prompt"
            assert "Good academic structure" in critique_result.feedback
            assert not critique_result.needs_improvement  # Criteria met


class TestConstitutionalCritic:
    """Test Constitutional AI critic."""

    def test_initialization(self):
        """Test constitutional critic initialization."""
        critic = ConstitutionalCritic()
        assert critic.model == "gpt-4o-mini"
        assert critic.temperature == 0.3
        assert len(critic.principles) == 7  # Default principles
        assert critic.name == "constitutional"

    def test_custom_principles(self):
        """Test initialization with custom principles."""
        custom_principles = ["Be helpful", "Be truthful", "Be harmless"]
        critic = ConstitutionalCritic(principles=custom_principles)
        assert critic.principles == custom_principles

    def test_parse_json_with_violations(self):
        """Test parsing JSON response with violations."""

        critic = ConstitutionalCritic()
        json_response = """{
    "feedback": "Poor compliance with principles",
    "suggestions": ["Clarify text", "Improve structure"],
    "needs_improvement": true,
    "confidence": 0.8,
    "metadata": {
        "principle_scores": {"1": 3, "2": 4},
        "violations": [{
            "principle_number": 1,
            "principle_text": "Be clear",
            "violation_description": "Text is confusing",
            "severity": 5
        }]
    }
}"""

        result = critic._parse_json_response(json_response)
        assert "Poor compliance" in result.feedback
        assert result.needs_improvement is True
        assert len(result.suggestions) == 2

    def test_format_principles(self):
        """Test principle formatting."""
        critic = ConstitutionalCritic(
            principles=["Test principle 1", "Test principle 2"]
        )
        formatted = critic._format_principles()

        assert "1. Test principle 1" in formatted
        assert "2. Test principle 2" in formatted

    def test_constitutional_config(self):
        """Test constitutional critic configuration."""
        critic = ConstitutionalCritic()

        assert critic.config.response_format == "json"
        assert critic.config.base_confidence == 0.7
        assert critic.config.domain_weight == 0.15

    @pytest.mark.asyncio
    async def test_critique_with_json_response(self):
        """Test critique with JSON response."""
        critic = ConstitutionalCritic()
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
    "feedback": "Text follows most principles well",
    "suggestions": ["Minor improvements needed"],
    "needs_improvement": false,
    "confidence": 0.85,
    "metadata": {
        "principle_scores": {"1": 5, "2": 4, "3": 5}
    }
}"""

        with patch.object(
            critic.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            critique_result = await critic.critique("Test text", result)

            assert critique_result.critic == "constitutional"
            assert "follows most principles" in critique_result.feedback
            assert critique_result.confidence > 0.8
