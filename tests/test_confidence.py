"""Tests for confidence calculation utilities."""

import pytest
from sifaka.core.confidence import (
    BaseConfidenceCalculator,
    StructuredConfidenceCalculator,
    create_confidence_calculator,
)


class TestBaseConfidenceCalculator:
    """Test base confidence calculator functionality."""

    def test_initialization(self):
        """Test calculator initialization with different base values."""
        calc = BaseConfidenceCalculator(base_confidence=0.5)
        assert calc.base_confidence == 0.5

    def test_initialization_bounds(self):
        """Test initialization respects bounds."""
        # Test upper bound
        calc = BaseConfidenceCalculator(base_confidence=1.5)
        assert calc.base_confidence == 0.9

        # Test lower bound
        calc = BaseConfidenceCalculator(base_confidence=-0.5)
        assert calc.base_confidence == 0.1

    def test_basic_calculation(self):
        """Test basic confidence calculation."""
        calc = BaseConfidenceCalculator(base_confidence=0.6)

        confidence = calc.calculate(
            feedback="This is a clear and specific analysis",
            suggestions=["Add more details", "Improve clarity"],
            evaluation_text="EVALUATION: Good\nSUGGESTIONS: Listed above",
        )

        assert 0.1 <= confidence <= 0.95
        assert confidence >= 0.6  # Should be at least base confidence

    def test_structured_response_bonus(self):
        """Test bonus for structured responses."""
        calc = BaseConfidenceCalculator()

        # Structured response
        structured_confidence = calc.calculate(
            evaluation_text="EVALUATION: Good analysis\nSUGGESTIONS: Add more details",
            structured_response=True,
        )

        # Unstructured response
        unstructured_confidence = calc.calculate(
            evaluation_text="This is just plain text", structured_response=False
        )

        assert structured_confidence > unstructured_confidence

    def test_specificity_factor(self):
        """Test specificity bonus calculation."""
        calc = BaseConfidenceCalculator()

        # Specific feedback
        specific_confidence = calc.calculate(
            feedback="Specifically, the text clearly demonstrates precise understanding"
        )

        # Vague feedback
        vague_confidence = calc.calculate(
            feedback="The text might possibly need some changes perhaps"
        )

        assert specific_confidence > vague_confidence

    def test_suggestion_quality_factor(self):
        """Test suggestion quality assessment."""
        calc = BaseConfidenceCalculator()

        # Actionable suggestions
        actionable_confidence = calc.calculate(
            suggestions=[
                "Add references to studies",
                "Remove redundant paragraph",
                "Clarify methodology",
            ]
        )

        # Vague suggestions
        vague_confidence = calc.calculate(
            suggestions=["Make it better", "Fix issues", "Improve overall quality"]
        )

        assert actionable_confidence > vague_confidence

    def test_domain_expertise_factor(self):
        """Test domain expertise bonus."""
        calc = BaseConfidenceCalculator()

        # Domain-specific feedback
        domain_confidence = calc.calculate(
            feedback="The thesis clearly presents the research methodology",
            domain_indicators=["thesis", "research", "methodology"],
        )

        # Generic feedback
        generic_confidence = calc.calculate(
            feedback="The text is good and well written"
        )

        assert domain_confidence > generic_confidence

    def test_length_factor(self):
        """Test length appropriateness factor."""
        calc = BaseConfidenceCalculator()

        # Appropriate length (sweet spot)
        good_length_confidence = calc.calculate(
            feedback="This provides detailed analysis of the key issues",
            suggestions=["Add conclusion", "Review sources"],
        )

        # Too short
        short_confidence = calc.calculate(feedback="Good", suggestions=["Fix"])

        # Too long (won't test actual long text, but principle is there)
        assert good_length_confidence >= short_confidence

    def test_custom_factors(self):
        """Test custom factor application."""
        calc = BaseConfidenceCalculator()

        confidence = calc.calculate(custom_factor=0.05, another_factor=0.03)

        # Should include custom factors
        assert confidence >= calc.base_confidence

    def test_bounds_enforcement(self):
        """Test confidence stays within bounds."""
        calc = BaseConfidenceCalculator(base_confidence=0.9)

        # Even with all bonuses, should not exceed 0.95
        confidence = calc.calculate(
            feedback="Specifically and clearly demonstrates excellent understanding",
            suggestions=["Add details", "Remove section", "Clarify methodology"],
            evaluation_text="EVALUATION: Excellent\nSUGGESTIONS: Listed",
            structured_response=True,
            domain_indicators=["methodology", "research", "analysis"],
            custom_factor=0.1,
        )

        assert confidence <= 0.95

    def test_minimum_confidence(self):
        """Test minimum confidence enforcement."""
        calc = BaseConfidenceCalculator(base_confidence=0.1)

        # Even with negative factors, should not go below 0.1
        confidence = calc.calculate(custom_factor=-0.5)  # Large negative factor

        assert confidence >= 0.1


class TestStructuredConfidenceCalculator:
    """Test structured confidence calculator for Pydantic models."""

    def test_initialization(self):
        """Test structured calculator initialization."""
        calc = StructuredConfidenceCalculator(base_confidence=0.7)
        assert calc.base_confidence == 0.7

    def test_overall_confidence_usage(self):
        """Test using LLM's self-assessed confidence."""
        calc = StructuredConfidenceCalculator()

        confidence = calc.calculate(overall_confidence=0.8, evaluation_quality=4)

        # Should start from LLM's confidence
        assert confidence >= 0.8

    def test_evaluation_quality_factor(self):
        """Test evaluation quality self-assessment impact."""
        calc = StructuredConfidenceCalculator()

        # High quality assessment
        high_quality = calc.calculate(
            overall_confidence=0.7, evaluation_quality=5  # Excellent
        )

        # Low quality assessment
        low_quality = calc.calculate(
            overall_confidence=0.7, evaluation_quality=1  # Poor
        )

        assert high_quality > low_quality

    def test_violation_confidence_factor(self):
        """Test violation confidence consistency bonus."""
        calc = StructuredConfidenceCalculator()

        # High violation confidences
        high_violation_conf = calc.calculate(
            overall_confidence=0.6, violation_confidences=[0.9, 0.8, 0.85]
        )

        # Low violation confidences
        low_violation_conf = calc.calculate(
            overall_confidence=0.6, violation_confidences=[0.3, 0.4, 0.2]
        )

        assert high_violation_conf > low_violation_conf

    def test_score_distribution_factor(self):
        """Test principle score distribution bonus."""
        calc = StructuredConfidenceCalculator()

        # Varied scores (good discrimination)
        varied_scores = calc.calculate(
            overall_confidence=0.6, principle_scores={1: 5, 2: 2, 3: 4, 4: 1}
        )

        # All middle scores (poor discrimination)
        uniform_scores = calc.calculate(
            overall_confidence=0.6, principle_scores={1: 3, 2: 3, 3: 3, 4: 3}
        )

        assert varied_scores > uniform_scores

    def test_base_factors_integration(self):
        """Test integration with base calculator factors."""
        calc = StructuredConfidenceCalculator()

        confidence = calc.calculate(
            overall_confidence=0.6,
            evaluation_quality=4,
            feedback="Specifically identifies clear issues",
            suggestions=["Add references", "Remove section"],
            structured_response=True,
        )

        # Should benefit from both structured and base factors
        assert confidence > 0.6


class TestConfidenceCalculatorFactory:
    """Test confidence calculator factory function."""

    def test_create_base_calculator(self):
        """Test creating base calculator."""
        calc = create_confidence_calculator("base", base_confidence=0.5)
        assert isinstance(calc, BaseConfidenceCalculator)
        assert calc.base_confidence == 0.5

    def test_create_structured_calculator(self):
        """Test creating structured calculator."""
        calc = create_confidence_calculator("structured", base_confidence=0.7)
        assert isinstance(calc, StructuredConfidenceCalculator)
        assert calc.base_confidence == 0.7

    def test_default_calculator(self):
        """Test default calculator type."""
        calc = create_confidence_calculator()
        assert isinstance(calc, BaseConfidenceCalculator)
