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

    def test_creation(self):
        """Test creating a validation rule."""
        rule = ValidationRule(
            name="test_rule",
            check=lambda text: len(text) > 0,
            score_func=lambda text: 1.0,
            detail_func=lambda text: "Non-empty"
        )
        assert rule.name == "test_rule"
        assert rule.check("hello") is True
        assert rule.check("") is False
        assert rule.score_func("any") == 1.0
        assert rule.detail_func("any") == "Non-empty"


class TestComposableValidator:
    """Test the ComposableValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text",
            final_text="Final text"
        )

    def test_initialization(self):
        """Test initialization."""
        validator = ComposableValidator("test_validator")
        assert validator.name == "test_validator"
        assert validator.rules == []

    def test_name_property(self):
        """Test name property."""
        validator = ComposableValidator("my_validator")
        assert validator.name == "my_validator"

    @pytest.mark.asyncio
    async def test_no_rules(self, sample_result):
        """Test validation with no rules."""
        validator = ComposableValidator("empty")
        passed, score, details = await validator._perform_validation("text", sample_result)
        assert passed is True
        assert score == 1.0
        assert details == "No validation rules"

    @pytest.mark.asyncio
    async def test_single_rule_pass(self, sample_result):
        """Test with single passing rule."""
        rule = ValidationRule(
            name="length_check",
            check=lambda text: len(text) > 5,
            score_func=lambda text: 1.0,
            detail_func=lambda text: f"Length is {len(text)}"
        )
        validator = ComposableValidator("test", [rule])
        
        passed, score, details = await validator._perform_validation("hello world", sample_result)
        assert passed is True
        assert score == 1.0
        assert "length_check: Length is 11" in details

    @pytest.mark.asyncio
    async def test_single_rule_fail(self, sample_result):
        """Test with single failing rule."""
        rule = ValidationRule(
            name="length_check",
            check=lambda text: len(text) > 10,
            score_func=lambda text: len(text) / 10,
            detail_func=lambda text: f"Too short"
        )
        validator = ComposableValidator("test", [rule])
        
        passed, score, details = await validator._perform_validation("short", sample_result)
        assert passed is False
        assert score == 0.0
        assert "length_check: Too short" in details

    @pytest.mark.asyncio
    async def test_multiple_rules_all_pass(self, sample_result):
        """Test with multiple rules all passing."""
        rules = [
            ValidationRule(
                name="length",
                check=lambda text: len(text) > 5,
                score_func=lambda text: 1.0,
                detail_func=lambda text: "Length OK"
            ),
            ValidationRule(
                name="contains_hello",
                check=lambda text: "hello" in text.lower(),
                score_func=lambda text: 1.0,
                detail_func=lambda text: "Contains hello"
            )
        ]
        validator = ComposableValidator("test", rules)
        
        passed, score, details = await validator._perform_validation("Hello world!", sample_result)
        assert passed is True
        assert score == 1.0
        assert "length: Length OK" in details
        assert "contains_hello: Contains hello" in details

    @pytest.mark.asyncio
    async def test_multiple_rules_partial_pass(self, sample_result):
        """Test with some rules passing."""
        rules = [
            ValidationRule(
                name="rule1",
                check=lambda text: True,
                score_func=lambda text: 0.8,
                detail_func=lambda text: "Rule 1 passed"
            ),
            ValidationRule(
                name="rule2",
                check=lambda text: False,
                score_func=lambda text: 0.5,
                detail_func=lambda text: "Rule 2 failed"
            )
        ]
        validator = ComposableValidator("test", rules)
        
        passed, score, details = await validator._perform_validation("any text", sample_result)
        assert passed is False
        assert score == 0.4  # (0.8 + 0) / 2
        assert "rule1: Rule 1 passed" in details
        assert "rule2: Rule 2 failed" in details

    @pytest.mark.asyncio
    async def test_rule_exception_handling(self, sample_result):
        """Test handling of exceptions in rules."""
        def bad_check(text):
            raise ValueError("Test error")
        
        rule = ValidationRule(
            name="bad_rule",
            check=bad_check,
            score_func=lambda text: 1.0,
            detail_func=lambda text: "Should not reach"
        )
        validator = ComposableValidator("test", [rule])
        
        passed, score, details = await validator._perform_validation("text", sample_result)
        assert "bad_rule: Error - Test error" in details

    def test_and_operator(self):
        """Test combining validators with AND."""
        v1 = ComposableValidator("v1", [])
        v2 = ComposableValidator("v2", [])
        combined = v1 & v2
        
        assert combined.name == "(v1 AND v2)"
        assert len(combined.rules) == 0  # Both had no rules

    def test_and_operator_combines_rules(self):
        """Test AND operator combines rules."""
        rule1 = ValidationRule("r1", lambda x: True, lambda x: 1.0, lambda x: "r1")
        rule2 = ValidationRule("r2", lambda x: True, lambda x: 1.0, lambda x: "r2")
        
        v1 = ComposableValidator("v1", [rule1])
        v2 = ComposableValidator("v2", [rule2])
        combined = v1 & v2
        
        assert combined.name == "(v1 AND v2)"
        assert len(combined.rules) == 2
        assert combined.rules[0].name == "r1"
        assert combined.rules[1].name == "r2"

    @pytest.mark.asyncio
    async def test_or_operator(self, sample_result):
        """Test combining validators with OR."""
        rule1 = ValidationRule("r1", lambda x: True, lambda x: 0.8, lambda x: "Pass")
        rule2 = ValidationRule("r2", lambda x: False, lambda x: 0.5, lambda x: "Fail")
        
        v1 = ComposableValidator("v1", [rule1])
        v2 = ComposableValidator("v2", [rule2])
        combined = v1 | v2
        
        assert "(v1 OR v2)" in combined.name
        
        # Test OR logic - should pass if either passes
        passed, score, details = await combined._perform_validation("text", sample_result)
        assert passed is True  # v1 passes
        assert score == 0.8  # max of scores
        assert "Left:" in details
        assert "Right:" in details

    @pytest.mark.asyncio
    async def test_or_operator_both_fail(self, sample_result):
        """Test OR when both validators fail."""
        rule1 = ValidationRule("r1", lambda x: False, lambda x: 0.3, lambda x: "Fail")
        rule2 = ValidationRule("r2", lambda x: False, lambda x: 0.2, lambda x: "Fail")
        
        v1 = ComposableValidator("v1", [rule1])
        v2 = ComposableValidator("v2", [rule2])
        combined = v1 | v2
        
        passed, score, details = await combined._perform_validation("text", sample_result)
        assert passed is False
        assert score == 0.0  # Both failed, so 0

    def test_invert_operator(self):
        """Test NOT operator."""
        rule = ValidationRule(
            "positive_length",
            lambda x: len(x) > 0,
            lambda x: 1.0,
            lambda x: "Non-empty"
        )
        validator = ComposableValidator("test", [rule])
        inverted = ~validator
        
        assert inverted.name == "NOT test"
        assert len(inverted.rules) == 1
        assert inverted.rules[0].name == "NOT positive_length"
        
        # Test inverted logic
        assert inverted.rules[0].check("hello") is False  # Original passes, inverted fails
        assert inverted.rules[0].check("") is True  # Original fails, inverted passes


class TestValidatorBuilder:
    """Test the ValidatorBuilder class."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = ValidatorBuilder("my_validator")
        assert builder.name == "my_validator"
        assert builder.rules == []

    def test_length_validation(self):
        """Test adding length validation."""
        builder = ValidatorBuilder()
        builder.length(min_length=10, max_length=100)
        
        assert len(builder.rules) == 1
        rule = builder.rules[0]
        assert rule.name == "length"
        assert rule.check("x" * 50) is True
        assert rule.check("short") is False
        assert rule.check("x" * 101) is False

    def test_contains_all_mode(self):
        """Test contains validation with 'all' mode."""
        builder = ValidatorBuilder()
        builder.contains(["hello", "world"], mode="all")
        
        rule = builder.rules[0]
        assert rule.check("Hello World!") is True
        assert rule.check("hello there") is False
        assert rule.score_func("hello world") == 1.0
        assert rule.score_func("just hello") == 0.5

    def test_contains_any_mode(self):
        """Test contains validation with 'any' mode."""
        builder = ValidatorBuilder()
        builder.contains(["python", "java"], mode="any")
        
        rule = builder.rules[0]
        assert rule.check("I love Python") is True
        assert rule.check("Java is good") is True
        assert rule.check("I prefer C++") is False

    def test_matches_pattern(self):
        """Test regex pattern matching."""
        builder = ValidatorBuilder()
        builder.matches(r"\d{3}-\d{3}-\d{4}", "phone")
        
        rule = builder.rules[0]
        assert rule.name == "matches_phone"
        assert rule.check("Call 123-456-7890") is True
        assert rule.check("No phone here") is False
        assert "Pattern 'phone' found" in rule.detail_func("123-456-7890")

    def test_custom_rule(self):
        """Test adding custom rule."""
        builder = ValidatorBuilder()
        builder.custom(
            name="uppercase",
            check=lambda text: any(c.isupper() for c in text),
            score=lambda text: sum(1 for c in text if c.isupper()) / len(text),
            detail=lambda text: f"Has uppercase: {any(c.isupper() for c in text)}"
        )
        
        rule = builder.rules[0]
        assert rule.name == "uppercase"
        assert rule.check("Hello") is True
        assert rule.check("hello") is False

    def test_sentences_validation(self):
        """Test sentence count validation."""
        builder = ValidatorBuilder()
        builder.sentences(min_sentences=2, max_sentences=5)
        
        rule = builder.rules[0]
        assert rule.check("One sentence.") is False
        assert rule.check("First. Second.") is True
        assert rule.check("One. Two. Three. Four. Five. Six.") is False

    def test_words_validation(self):
        """Test word count validation."""
        builder = ValidatorBuilder()
        builder.words(min_words=5, max_words=20)
        
        rule = builder.rules[0]
        assert rule.check("Too few") is False
        assert rule.check("This is just the right length") is True
        assert rule.check(" ".join(["word"] * 25)) is False

    def test_chaining(self):
        """Test method chaining."""
        validator = (ValidatorBuilder("essay")
            .length(100, 1000)
            .sentences(5, 50)
            .contains(["introduction", "conclusion"])
            .build())
        
        assert validator.name == "essay"
        assert len(validator.rules) == 3

    def test_build(self):
        """Test building the validator."""
        builder = ValidatorBuilder("test")
        builder.length(10, 100)
        validator = builder.build()
        
        assert isinstance(validator, ComposableValidator)
        assert validator.name == "test"
        assert len(validator.rules) == 1


class TestValidatorFactory:
    """Test the Validator factory class."""

    def test_create(self):
        """Test creating a builder."""
        builder = Validator.create("custom")
        assert isinstance(builder, ValidatorBuilder)
        assert builder.name == "custom"

    def test_length_factory(self):
        """Test length validator factory."""
        validator = Validator.length(10, 100)
        assert isinstance(validator, ComposableValidator)
        assert validator.name == "length"
        assert len(validator.rules) == 1

    def test_contains_factory(self):
        """Test contains validator factory."""
        validator = Validator.contains(["test", "example"])
        assert validator.name == "contains"
        assert len(validator.rules) == 1

    def test_matches_factory(self):
        """Test matches validator factory."""
        validator = Validator.matches(r"\d+", "number")
        assert validator.name == "matches"
        assert len(validator.rules) == 1

    def test_sentences_factory(self):
        """Test sentences validator factory."""
        validator = Validator.sentences(2, 10)
        assert validator.name == "sentences"
        assert len(validator.rules) == 1

    def test_words_factory(self):
        """Test words validator factory."""
        validator = Validator.words(10, 100)
        assert validator.name == "words"
        assert len(validator.rules) == 1

    @pytest.mark.asyncio
    async def test_composed_validators(self):
        """Test composing validators with operators."""
        # Create composite validator
        length_validator = Validator.length(50, 200)
        keyword_validator = Validator.contains(["AI", "ML"], mode="any")
        
        # Combine with AND
        combined = length_validator & keyword_validator
        
        sample_result = SifakaResult(
            original_text="Original",
            final_text="Final"
        )
        
        # Test text that passes both
        text1 = "x" * 100 + " AI is great"
        passed, score, _ = await combined._perform_validation(text1, sample_result)
        assert passed is True
        
        # Test text that fails length
        text2 = "AI"
        passed, score, _ = await combined._perform_validation(text2, sample_result)
        assert passed is False

    def test_complex_composition(self):
        """Test complex validator composition."""
        # (Length AND Keywords) OR Email
        validator = (
            Validator.length(20, 100) & 
            Validator.contains(["important"])
        ) | Validator.matches(r"\b[\w.-]+@[\w.-]+\.\w+\b", "email")
        
        assert "OR" in validator.name
        assert "AND" in validator.name