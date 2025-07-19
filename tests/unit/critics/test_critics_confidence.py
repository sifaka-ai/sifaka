"""Tests for the confidence calculator in sifaka.critics.core.confidence."""

import pytest

from sifaka.critics.core.confidence import ConfidenceCalculator


class TestConfidenceCalculator:
    """Test the ConfidenceCalculator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        calc = ConfidenceCalculator()
        assert calc.base_confidence == 0.7

    def test_initialization_custom_base(self):
        """Test initialization with custom base confidence."""
        calc = ConfidenceCalculator(base_confidence=0.5)
        assert calc.base_confidence == 0.5

    def test_basic_calculation(self):
        """Test basic confidence calculation with neutral inputs."""
        calc = ConfidenceCalculator()
        confidence = calc.calculate(
            feedback="This is basic feedback",
            suggestions=["Add more details"],
            response_length=200,
            metadata=None,
        )
        # Base confidence 0.7, short feedback (<20 words: -0.05), 1 suggestion (+0.05), short suggestion (<5 words: -0.05) = 0.65
        assert confidence == pytest.approx(0.65)

    def test_metadata_parameter_ignored(self):
        """Test that metadata parameter doesn't affect calculation."""
        calc = ConfidenceCalculator()

        # Same inputs except metadata
        confidence1 = calc.calculate(
            feedback="Test feedback",
            suggestions=["Test"],
            response_length=200,
            metadata=None,
        )

        confidence2 = calc.calculate(
            feedback="Test feedback",
            suggestions=["Test"],
            response_length=200,
            metadata={"key": "value"},
        )

        assert confidence1 == confidence2

    def test_uncertainty_indicators_penalty(self):
        """Test that feedback is scored based on word count."""
        calc = ConfidenceCalculator(base_confidence=0.7)

        # Short feedback (<20 words)
        confidence = calc.calculate(
            feedback="Short uncertain feedback",
            suggestions=["Improve"],
            response_length=200,
            metadata=None,
        )
        # Base 0.7 - 0.05 (short feedback) + 0.05 (1 suggestion) - 0.05 (short suggestion) = 0.65
        assert confidence == pytest.approx(0.65)
