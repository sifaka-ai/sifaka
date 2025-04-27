"""Tests for the formatting rules."""

import pytest
from typing import Dict, Any, List

from sifaka.rules.formatting import LengthRule, ParagraphRule, StyleRule, FormattingRule
from sifaka.rules.base import RuleResult


class TestLengthRule(LengthRule):
    """Test implementation of LengthRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        length = len(output)

        if length < self.min_length:
            return RuleResult(
                passed=False,
                message=f"Output is below minimum length ({length} characters)",
                metadata={"length": length, "min_length": self.min_length},
            )

        if length > self.max_length:
            return RuleResult(
                passed=False,
                message=f"Output exceeds maximum length ({length} characters)",
                metadata={"length": length, "max_length": self.max_length},
            )

        return RuleResult(
            passed=True,
            message=f"Output length is acceptable ({length} characters)",
            metadata={"length": length},
        )


class TestParagraphRule(ParagraphRule):
    """Test implementation of ParagraphRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        paragraphs = output.split("\n\n")
        issues = []

        for i, paragraph in enumerate(paragraphs):
            sentences = [s.strip() for s in paragraph.split(".") if s.strip()]

            if len(sentences) < self.min_sentences:
                issues.append(
                    f"Paragraph {i+1} has fewer sentences than minimum ({len(sentences)})"
                )

            if len(sentences) > self.max_sentences:
                issues.append(f"Paragraph {i+1} exceeds maximum sentences ({len(sentences)})")

            for j, sentence in enumerate(sentences):
                words = sentence.split()
                if len(words) < self.min_words:
                    issues.append(
                        f"Sentence {j+1} in paragraph {i+1} has fewer words than minimum ({len(words)})"
                    )
                if len(words) > self.max_words:
                    issues.append(
                        f"Sentence {j+1} in paragraph {i+1} exceeds maximum words ({len(words)})"
                    )

        if issues:
            return RuleResult(
                passed=False,
                message="Paragraph formatting issues detected",
                metadata={"issues": issues},
            )

        return RuleResult(passed=True, message="Paragraph formatting is acceptable")


class TestStyleRule(StyleRule):
    """Test implementation of StyleRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        style_scores = {}

        for style, indicators in self.style_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            style_scores[style] = len(found_indicators) / len(indicators)

        dominant_style = max(style_scores.items(), key=lambda x: x[1])

        if dominant_style[1] < self.style_threshold:
            return RuleResult(
                passed=False,
                message="Writing style is inconsistent",
                metadata={"style_scores": style_scores},
            )

        return RuleResult(
            passed=True,
            message=f"Writing style is consistent ({dominant_style[0]})",
            metadata={"style_scores": style_scores},
        )


class TestFormattingRule(FormattingRule):
    """Test implementation of FormattingRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = []
        for pattern_name, pattern in self.formatting_patterns.items():
            matches = list(re.finditer(pattern, output))
            if matches:
                for match in matches:
                    issues.append(f"{pattern_name} at position {match.start()}")

        if issues:
            return RuleResult(
                passed=False,
                message="Formatting issues detected",
                metadata={"issues": issues},
            )

        return RuleResult(
            passed=True,
            message="Text formatting is acceptable",
            metadata={"patterns_checked": list(self.formatting_patterns.keys())},
        )


@pytest.fixture
def length_rule():
    """Create a TestLengthRule instance."""
    return TestLengthRule(
        name="test_length", description="Test length rule", min_length=10, max_length=100
    )


@pytest.fixture
def paragraph_rule():
    """Create a TestParagraphRule instance."""
    return TestParagraphRule(
        name="test_paragraph",
        description="Test paragraph rule",
        min_sentences=2,
        max_sentences=4,
        min_words=3,
        max_words=15,
    )


@pytest.fixture
def style_rule():
    """Create a TestStyleRule instance."""
    return TestStyleRule(name="test_style", description="Test style rule", style_threshold=0.6)


@pytest.fixture
def formatting_rule():
    """Create a TestFormattingRule instance."""
    return TestFormattingRule(name="test_formatting", description="Test formatting rule")


def test_length_rule_initialization():
    """Test LengthRule initialization."""
    rule = TestLengthRule(name="test", description="test", min_length=50, max_length=200)
    assert rule.name == "test"
    assert rule.min_length == 50
    assert rule.max_length == 200


def test_length_rule_validation(length_rule):
    """Test length rule validation."""
    # Test valid length
    result = length_rule.validate("This is a valid length text.")
    assert result.passed
    assert "length" in result.metadata

    # Test too short
    result = length_rule.validate("Too short")
    assert not result.passed
    assert "below minimum" in result.message.lower()

    # Test too long
    result = length_rule.validate("x" * 150)
    assert not result.passed
    assert "exceeds maximum" in result.message.lower()


def test_paragraph_rule_initialization():
    """Test ParagraphRule initialization."""
    rule = TestParagraphRule(
        name="test", description="test", min_sentences=2, max_sentences=5, min_words=5, max_words=20
    )
    assert rule.name == "test"
    assert rule.min_sentences == 2
    assert rule.max_sentences == 5
    assert rule.min_words == 5
    assert rule.max_words == 20


def test_paragraph_rule_validation(paragraph_rule):
    """Test paragraph rule validation."""
    # Test valid paragraph structure
    valid_text = """This is a good first sentence. And here is the second one.

This is another paragraph. It also has two sentences."""
    result = paragraph_rule.validate(valid_text)
    assert result.passed

    # Test too few sentences
    invalid_text = "This is a single sentence paragraph."
    result = paragraph_rule.validate(invalid_text)
    assert not result.passed
    assert "fewer sentences than minimum" in result.metadata["issues"][0]

    # Test too many sentences
    many_sentences = "One. Two. Three. Four. Five. Six."
    result = paragraph_rule.validate(many_sentences)
    assert not result.passed
    assert "exceeds maximum sentences" in result.metadata["issues"][0]


def test_style_rule_initialization():
    """Test StyleRule initialization."""
    custom_indicators = {"technical": ["algorithm", "function"], "casual": ["hey", "thanks"]}
    rule = TestStyleRule(
        name="test", description="test", style_indicators=custom_indicators, style_threshold=0.8
    )
    assert rule.name == "test"
    assert rule.style_indicators == custom_indicators
    assert rule.style_threshold == 0.8


def test_style_rule_validation(style_rule):
    """Test style rule validation."""
    # Test consistent formal style
    formal_text = "Therefore, we must proceed. Furthermore, the analysis shows. Thus, we conclude."
    result = style_rule.validate(formal_text)
    assert result.passed
    assert "formal" in result.message.lower()

    # Test consistent informal style
    informal_text = "Yeah, that's cool! Gonna be awesome! BTW, check this out!"
    result = style_rule.validate(informal_text)
    assert result.passed
    assert "informal" in result.message.lower()

    # Test inconsistent style
    mixed_text = "Therefore, the algorithm is cool. BTW, check out this parameter!"
    result = style_rule.validate(mixed_text)
    assert not result.passed
    assert "inconsistent" in result.message.lower()


def test_formatting_rule_initialization():
    """Test FormattingRule initialization."""
    custom_patterns = {"test_pattern": r"\d+", "another_pattern": r"[A-Z]+"}
    rule = TestFormattingRule(name="test", description="test", formatting_patterns=custom_patterns)
    assert rule.name == "test"
    assert rule.formatting_patterns == custom_patterns


def test_formatting_rule_validation(formatting_rule):
    """Test formatting rule validation."""
    # Test well-formatted text
    valid_text = "This is a properly formatted text. It has correct spacing and punctuation."
    result = formatting_rule.validate(valid_text)
    assert result.passed

    # Test multiple spaces
    invalid_text = "This  has  multiple  spaces."
    result = formatting_rule.validate(invalid_text)
    assert not result.passed
    assert "multiple_spaces" in result.metadata["issues"][0]

    # Test multiple newlines
    invalid_text = "First line.\n\n\nToo many newlines."
    result = formatting_rule.validate(invalid_text)
    assert not result.passed
    assert "multiple_newlines" in result.metadata["issues"][0]

    # Test trailing whitespace
    invalid_text = "Line with trailing space. \nNext line."
    result = formatting_rule.validate(invalid_text)
    assert not result.passed
    assert "trailing_whitespace" in result.metadata["issues"][0]


def test_edge_cases():
    """Test edge cases for all rules."""
    rules = [
        TestLengthRule(name="length", description="test"),
        TestParagraphRule(name="paragraph", description="test"),
        TestStyleRule(name="style", description="test"),
        TestFormattingRule(name="formatting", description="test"),
    ]

    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "numbers_only": "123 456 789",
    }

    for rule in rules:
        for case_name, text in edge_cases.items():
            result = rule.validate(text)
            assert isinstance(result, RuleResult)
            assert isinstance(result.passed, bool)
            assert isinstance(result.message, str)
            assert isinstance(result.metadata, dict)


def test_error_handling():
    """Test error handling for all rules."""
    rules = [
        TestLengthRule(name="length", description="test"),
        TestParagraphRule(name="paragraph", description="test"),
        TestStyleRule(name="style", description="test"),
        TestFormattingRule(name="formatting", description="test"),
    ]

    invalid_inputs = [None, 123, [], {}]

    for rule in rules:
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                rule.validate(invalid_input)


def test_consistent_results():
    """Test consistency of validation results."""
    rules = [
        TestLengthRule(name="length", description="test"),
        TestParagraphRule(name="paragraph", description="test"),
        TestStyleRule(name="style", description="test"),
        TestFormattingRule(name="formatting", description="test"),
    ]

    test_text = """
    This is a test paragraph. It contains multiple sentences.

    Furthermore, we can analyze the style. Therefore, it should be consistent.
    """

    for rule in rules:
        # Run validation multiple times
        results = [rule.validate(test_text) for _ in range(3)]

        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result.passed == first_result.passed
            assert result.message == first_result.message
            assert result.metadata == first_result.metadata
