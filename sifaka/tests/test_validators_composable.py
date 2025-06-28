"""Tests for composable validators."""

import pytest
from sifaka.validators.composable import (
    ValidationRule,
    ComposableValidator,
    ValidatorBuilder,
    Validator,
)
from sifaka.core.models import SifakaResult


class TestValidationRule:
    """Test the ValidationRule dataclass."""

    def test_initialization(self):
        """Test ValidationRule initialization."""
        rule = ValidationRule(
            name="test_rule",
            check=lambda text: len(text) > 5,
            score_func=lambda text: len(text) / 10,
            detail_func=lambda text: f"Length: {len(text)}",
        )

        assert rule.name == "test_rule"
        assert rule.check("hello world") is True
        assert rule.check("hi") is False
        assert rule.score_func("hello") == 0.5
        assert rule.detail_func("test") == "Length: 4"


class TestComposableValidator:
    """Test the ComposableValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original", final_text="Final")

    @pytest.fixture
    def length_rule(self):
        """Create a length validation rule."""
        return ValidationRule(
            name="length",
            check=lambda text: len(text) >= 10,
            score_func=lambda text: min(1.0, len(text) / 20),
            detail_func=lambda text: f"Length: {len(text)} chars",
        )

    @pytest.fixture
    def contains_rule(self):
        """Create a contains validation rule."""
        return ValidationRule(
            name="contains_hello",
            check=lambda text: "hello" in text.lower(),
            score_func=lambda text: 1.0 if "hello" in text.lower() else 0.0,
            detail_func=lambda text: f"Contains 'hello': {'yes' if 'hello' in text.lower() else 'no'}",
        )

    def test_initialization(self):
        """Test ComposableValidator initialization."""
        validator = ComposableValidator("test_validator")
        assert validator.name == "test_validator"
        assert validator.rules == []

    def test_initialization_with_rules(self, length_rule):
        """Test initialization with rules."""
        validator = ComposableValidator("test", [length_rule])
        assert len(validator.rules) == 1
        assert validator.rules[0] == length_rule

    @pytest.mark.asyncio
    async def test_no_rules(self, sample_result):
        """Test validation with no rules."""
        validator = ComposableValidator("empty")
        result = await validator.validate("any text", sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "No validation rules" in result.details

    @pytest.mark.asyncio
    async def test_single_rule_passed(self, sample_result, length_rule):
        """Test validation with single passing rule."""
        validator = ComposableValidator("length_check", [length_rule])
        result = await validator.validate("This is a long enough text", sample_result)

        assert result.passed is True
        assert result.score == 1.0  # 26 chars / 20 = 1.3, capped at 1.0
        assert "Length: 26 chars" in result.details

    @pytest.mark.asyncio
    async def test_single_rule_failed(self, sample_result, length_rule):
        """Test validation with single failing rule."""
        validator = ComposableValidator("length_check", [length_rule])
        result = await validator.validate("Short", sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Length: 5 chars" in result.details

    @pytest.mark.asyncio
    async def test_multiple_rules_all_passed(
        self, sample_result, length_rule, contains_rule
    ):
        """Test validation with multiple passing rules."""
        validator = ComposableValidator("multi", [length_rule, contains_rule])
        result = await validator.validate("Hello world, this is long", sample_result)

        assert result.passed is True
        assert result.score == 1.0  # Average of two 1.0 scores
        assert "Length: 25 chars" in result.details
        assert "Contains 'hello': yes" in result.details

    @pytest.mark.asyncio
    async def test_multiple_rules_partial_passed(
        self, sample_result, length_rule, contains_rule
    ):
        """Test validation with some rules passing."""
        validator = ComposableValidator("multi", [length_rule, contains_rule])
        result = await validator.validate(
            "This is a long text without the word", sample_result
        )

        assert result.passed is False  # All rules must pass
        assert result.score == 0.5  # Average of 1.0 and 0.0
        assert "Length: 36 chars" in result.details
        assert "Contains 'hello': no" in result.details

    @pytest.mark.asyncio
    async def test_rule_exception_handling(self, sample_result):
        """Test handling of rule exceptions."""

        def broken_check(text: str) -> bool:
            # Will raise ZeroDivisionError
            _ = 1 / 0
            return False

        broken_rule = ValidationRule(
            name="broken",
            check=broken_check,
            score_func=lambda text: 0.0,
            detail_func=lambda text: "broken",
        )

        validator = ComposableValidator("error_test", [broken_rule])
        result = await validator.validate("any text", sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "broken: Error - division by zero" in result.details

    def test_and_operator(self, length_rule, contains_rule):
        """Test combining validators with AND operator."""
        validator1 = ComposableValidator("v1", [length_rule])
        validator2 = ComposableValidator("v2", [contains_rule])

        combined = validator1 & validator2

        assert combined.name == "(v1 AND v2)"
        assert len(combined.rules) == 2
        assert combined.rules[0] == length_rule
        assert combined.rules[1] == contains_rule

    @pytest.mark.asyncio
    async def test_or_operator(self, sample_result):
        """Test combining validators with OR operator."""
        # Create two validators
        validator1 = ComposableValidator(
            "length",
            [
                ValidationRule(
                    name="min_length",
                    check=lambda text: len(text) >= 20,
                    score_func=lambda text: 1.0 if len(text) >= 20 else 0.0,
                    detail_func=lambda text: f"Length {len(text)} >= 20",
                )
            ],
        )

        validator2 = ComposableValidator(
            "keyword",
            [
                ValidationRule(
                    name="has_hello",
                    check=lambda text: "hello" in text.lower(),
                    score_func=lambda text: 1.0 if "hello" in text.lower() else 0.0,
                    detail_func=lambda text: f"Has 'hello': {'yes' if 'hello' in text.lower() else 'no'}",
                )
            ],
        )

        combined = validator1 | validator2

        assert "(length OR keyword)" in combined.name

        # Test OR logic: passes if either passes
        result = await combined.validate("Hello", sample_result)  # Only second passes
        assert result.passed is True
        assert result.score == 1.0  # Max of scores

        result = await combined.validate(
            "This is a very long text without keyword", sample_result
        )  # Only first passes
        assert result.passed is True
        assert result.score == 1.0

        result = await combined.validate("Short", sample_result)  # Neither passes
        assert result.passed is False
        assert result.score == 0.0

    def test_not_operator(self, length_rule):
        """Test NOT operator (inversion)."""
        validator = ComposableValidator("length", [length_rule])
        inverted = ~validator

        assert inverted.name == "NOT length"
        assert len(inverted.rules) == 1

        # Test inverted behavior
        inverted_rule = inverted.rules[0]
        assert inverted_rule.name == "NOT length"
        assert inverted_rule.check("Short") is True  # Original would be False
        assert (
            inverted_rule.check("This is long enough") is False
        )  # Original would be True
        # Short text: original score would be 5/20 = 0.25, inverted is 0.75
        assert inverted_rule.score_func("Short") == 0.75  # 1.0 - 0.25
        assert "NOT (" in inverted_rule.detail_func("test")

    @pytest.mark.asyncio
    async def test_complex_combination(self, sample_result):
        """Test complex validator combinations."""
        # (length AND contains) OR pattern
        length_val = Validator.length(min_length=20)
        contains_val = Validator.contains(["python", "code"])
        pattern_val = Validator.matches(r"\b[A-Z]{2,}\b", "uppercase_words")

        combined = (length_val & contains_val) | pattern_val

        # Should pass with pattern match even without length/contains
        result = await combined.validate("Short with CAPS", sample_result)
        assert result.passed is True

        # Should pass with length and contains even without pattern
        result = await combined.validate(
            "This is a long text with python code inside", sample_result
        )
        assert result.passed is True

        # Should fail without any condition met
        result = await combined.validate("Short text", sample_result)
        assert result.passed is False


class TestValidatorBuilder:
    """Test the ValidatorBuilder class."""

    def test_initialization(self):
        """Test ValidatorBuilder initialization."""
        builder = ValidatorBuilder("test_builder")
        assert builder.name == "test_builder"
        assert builder.rules == []

    def test_length_validation(self):
        """Test length validation builder."""
        builder = ValidatorBuilder()
        builder.length(min_length=10, max_length=100)

        assert len(builder.rules) == 1
        rule = builder.rules[0]
        assert rule.name == "length"
        assert rule.check("hello") is False
        assert rule.check("hello world") is True
        assert rule.check("x" * 101) is False

    def test_contains_all_mode(self):
        """Test contains validation with 'all' mode."""
        builder = ValidatorBuilder()
        builder.contains(["python", "code"], mode="all")

        rule = builder.rules[0]
        assert rule.name == "contains_all"
        assert rule.check("Python code is great") is True
        assert rule.check("Only Python here") is False
        assert rule.score_func("Python code") == 1.0
        assert rule.score_func("Just Python") == 0.5

    def test_contains_any_mode(self):
        """Test contains validation with 'any' mode."""
        builder = ValidatorBuilder()
        builder.contains(["python", "java", "rust"], mode="any")

        rule = builder.rules[0]
        assert rule.name == "contains_any"
        assert rule.check("I love Python") is True
        assert rule.check("I love C++") is False  # None of the keywords present
        assert rule.score_func("Python and Rust") == pytest.approx(2 / 3)

    def test_matches_validation(self):
        """Test regex pattern validation."""
        builder = ValidatorBuilder()
        builder.matches(r"^\d{3}-\d{3}-\d{4}$", "phone")

        rule = builder.rules[0]
        assert rule.name == "matches_phone"
        assert rule.check("123-456-7890") is True
        assert rule.check("1234567890") is False
        assert "Pattern 'phone' found" in rule.detail_func("123-456-7890")

    def test_custom_validation(self):
        """Test custom validation rule."""
        builder = ValidatorBuilder()
        builder.custom(
            name="even_length",
            check=lambda text: len(text) % 2 == 0,
            score=lambda text: 1.0 if len(text) % 2 == 0 else 0.5,
            detail=lambda text: f"Length {len(text)} is {'even' if len(text) % 2 == 0 else 'odd'}",
        )

        rule = builder.rules[0]
        assert rule.name == "even_length"
        assert rule.check("even") is True
        assert rule.check("odd") is False
        assert rule.score_func("test") == 1.0

    def test_custom_validation_defaults(self):
        """Test custom validation with default functions."""
        builder = ValidatorBuilder()
        builder.custom(
            name="simple",
            check=lambda text: text.startswith("A"),
        )

        rule = builder.rules[0]
        assert rule.score_func("Any") == 1.0  # Default score
        assert "simple: passed" in rule.detail_func("Apple")
        assert "simple: failed" in rule.detail_func("Banana")

    def test_sentences_validation(self):
        """Test sentence count validation."""
        builder = ValidatorBuilder()
        builder.sentences(min_sentences=2, max_sentences=5)

        rule = builder.rules[0]
        assert rule.name == "sentences"
        assert rule.check("One sentence.") is False
        assert rule.check("First sentence. Second sentence.") is True
        assert rule.check("Too. Many. Sentences. Here. Now. Six.") is False
        assert "3 sentences" in rule.detail_func("One. Two. Three.")

    def test_words_validation(self):
        """Test word count validation."""
        builder = ValidatorBuilder()
        builder.words(min_words=5, max_words=20)

        rule = builder.rules[0]
        assert rule.name == "words"
        assert rule.check("Too few") is False
        assert rule.check("This has exactly five words") is True
        assert "5 words" in rule.detail_func("One two three four five")

    def test_chaining(self):
        """Test method chaining."""
        validator = (
            ValidatorBuilder("chained")
            .length(min_length=10)
            .contains(["test"])
            .matches(r"\d+", "number")
            .sentences(min_sentences=1)
            .words(max_words=100)
            .build()
        )

        assert validator.name == "chained"
        assert len(validator.rules) == 5

    def test_build(self):
        """Test building final validator."""
        builder = ValidatorBuilder("final")
        builder.length(min_length=5)
        validator = builder.build()

        assert isinstance(validator, ComposableValidator)
        assert validator.name == "final"
        assert len(validator.rules) == 1


class TestValidatorFactory:
    """Test the Validator factory class."""

    def test_create(self):
        """Test creating a new builder."""
        builder = Validator.create("custom")
        assert isinstance(builder, ValidatorBuilder)
        assert builder.name == "custom"

    def test_length_factory(self):
        """Test length validator factory."""
        validator = Validator.length(min_length=10, max_length=50)
        assert isinstance(validator, ComposableValidator)
        assert validator.name == "length"
        assert len(validator.rules) == 1

    def test_contains_factory(self):
        """Test contains validator factory."""
        validator = Validator.contains(["python", "code"])
        assert validator.name == "contains"
        assert len(validator.rules) == 1

    def test_matches_factory(self):
        """Test matches validator factory."""
        validator = Validator.matches(r"\w+@\w+\.\w+", "email")
        assert validator.name == "matches"
        assert len(validator.rules) == 1

    def test_sentences_factory(self):
        """Test sentences validator factory."""
        validator = Validator.sentences(min_sentences=2, max_sentences=10)
        assert validator.name == "sentences"
        assert len(validator.rules) == 1

    def test_words_factory(self):
        """Test words validator factory."""
        validator = Validator.words(min_words=50, max_words=500)
        assert validator.name == "words"
        assert len(validator.rules) == 1

    @pytest.mark.asyncio
    async def test_complex_example(self):
        """Test the example usage pattern."""
        # Example from the module: validator = Validator.length(100, 500) & Validator.contains(["AI", "ML"])
        validator = Validator.length(100, 500) & Validator.contains(["AI", "ML"])
        result = SifakaResult(original_text="Test", final_text="Test")

        # Test with text that meets both conditions
        text = "A" * 150 + " AI and ML are fascinating technologies"
        validation = await validator.validate(text, result)
        assert validation.passed is True

        # Test with text that's too short
        validation = await validator.validate("AI and ML", result)
        assert validation.passed is False

        # Test with text that's long enough but missing keywords
        validation = await validator.validate("A" * 200, result)
        assert validation.passed is False
