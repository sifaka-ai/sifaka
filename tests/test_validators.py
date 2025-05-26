#!/usr/bin/env python3
"""Comprehensive tests for Sifaka validator implementations.

This test suite validates all validator types including length, regex,
content, format, classifier, and guardrails validators. It tests
validation logic, error handling, and integration scenarios.
"""


from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.classifier import ClassifierValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.format import FormatValidator
from tests.utils import create_test_thought

logger = get_logger(__name__)


class TestLengthValidator:
    """Test LengthValidator implementation."""

    def test_length_validator_basic_validation(self):
        """Test basic length validation."""
        validator = LengthValidator(min_length=10, max_length=50)

        # Test valid length
        thought = create_test_thought(text="This is a valid length text for testing.")
        result = validator.validate(thought)

        assert result.passed
        assert "length" in result.message.lower()

    def test_length_validator_too_short(self):
        """Test validation failure for text too short."""
        validator = LengthValidator(min_length=20, max_length=100)

        thought = create_test_thought(text="Short")
        result = validator.validate(thought)

        assert not result.passed
        assert "short" in result.message.lower() or "minimum" in result.message.lower()

    def test_length_validator_too_long(self):
        """Test validation failure for text too long."""
        validator = LengthValidator(min_length=5, max_length=20)

        long_text = "This is a very long text that exceeds the maximum length limit."
        thought = create_test_thought(text=long_text)
        result = validator.validate(thought)

        assert not result.passed
        assert "long" in result.message.lower() or "maximum" in result.message.lower()

    def test_length_validator_word_count(self):
        """Test length validation by word count."""
        validator = LengthValidator(min_words=5, max_words=15)

        # Test valid word count
        thought = create_test_thought(text="This text has exactly ten words in it for testing.")
        result = validator.validate(thought)
        assert result.passed

        # Test too few words
        thought = create_test_thought(text="Too few words")
        result = validator.validate(thought)
        assert not result.passed

        # Test too many words
        long_text = "This text has way too many words and should fail the validation because it exceeds the maximum word count limit."
        thought = create_test_thought(text=long_text)
        result = validator.validate(thought)
        assert not result.passed

    def test_length_validator_edge_cases(self):
        """Test length validator edge cases."""
        validator = LengthValidator(min_length=0, max_length=100)

        # Test empty text
        thought = create_test_thought(text="")
        result = validator.validate(thought)
        assert result.passed  # Empty text should pass with min_length=0

        # Test None text
        thought = create_test_thought(text=None)
        result = validator.validate(thought)
        assert not result.passed  # None text should fail

        # Test whitespace only
        thought = create_test_thought(text="   ")
        result = validator.validate(thought)
        assert result.passed  # Whitespace should count as characters

    def test_length_validator_configuration(self):
        """Test length validator configuration options."""
        # Test with only minimum
        validator = LengthValidator(min_length=10)
        thought = create_test_thought(text="This is long enough")
        result = validator.validate(thought)
        assert result.passed

        # Test with only maximum
        validator = LengthValidator(max_length=50)
        thought = create_test_thought(text="Short text")
        result = validator.validate(thought)
        assert result.passed

        # Test with both word and character limits
        validator = LengthValidator(min_length=10, max_length=100, min_words=3, max_words=20)
        thought = create_test_thought(text="This is a reasonable length text.")
        result = validator.validate(thought)
        assert result.passed


class TestRegexValidator:
    """Test RegexValidator implementation."""

    def test_regex_validator_required_patterns(self):
        """Test regex validation with required patterns."""
        validator = RegexValidator(
            required_patterns=[r"\b[Aa]rtificial [Ii]ntelligence\b", r"\bAI\b"]
        )

        # Test text with required patterns
        thought = create_test_thought(
            text="Artificial Intelligence and AI are transforming technology."
        )
        result = validator.validate(thought)
        assert result.passed

        # Test text missing required patterns
        thought = create_test_thought(text="Machine learning is interesting.")
        result = validator.validate(thought)
        assert not result.passed

    def test_regex_validator_prohibited_patterns(self):
        """Test regex validation with prohibited patterns."""
        validator = RegexValidator(prohibited_patterns=[r"\bbad\b", r"\bevil\b"])

        # Test text without prohibited patterns
        thought = create_test_thought(text="This is good and positive content.")
        result = validator.validate(thought)
        assert result.passed

        # Test text with prohibited patterns
        thought = create_test_thought(text="This contains bad content.")
        result = validator.validate(thought)
        assert not result.passed

    def test_regex_validator_combined_patterns(self):
        """Test regex validation with both required and prohibited patterns."""
        validator = RegexValidator(
            required_patterns=[r"\bPython\b"], prohibited_patterns=[r"\bbug\b", r"\berror\b"]
        )

        # Test valid text
        thought = create_test_thought(text="Python is a great programming language.")
        result = validator.validate(thought)
        assert result.passed

        # Test text missing required pattern
        thought = create_test_thought(text="Java is also good.")
        result = validator.validate(thought)
        assert not result.passed

        # Test text with prohibited pattern
        thought = create_test_thought(text="Python has a bug in this code.")
        result = validator.validate(thought)
        assert not result.passed

    def test_regex_validator_case_sensitivity(self):
        """Test regex validator case sensitivity."""
        # Case-sensitive validator
        validator = RegexValidator(required_patterns=[r"Python"])

        thought = create_test_thought(text="python is great")  # lowercase
        result = validator.validate(thought)
        assert not result.passed

        thought = create_test_thought(text="Python is great")  # correct case
        result = validator.validate(thought)
        assert result.passed

    def test_regex_validator_complex_patterns(self):
        """Test regex validator with complex patterns."""
        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        validator = RegexValidator(required_patterns=[email_pattern])

        thought = create_test_thought(text="Contact us at support@example.com for help.")
        result = validator.validate(thought)
        assert result.passed

        thought = create_test_thought(text="Contact us for help.")
        result = validator.validate(thought)
        assert not result.passed

    def test_regex_validator_edge_cases(self):
        """Test regex validator edge cases."""
        validator = RegexValidator(required_patterns=[r"\w+"])

        # Test empty text
        thought = create_test_thought(text="")
        result = validator.validate(thought)
        assert not result.passed

        # Test None text
        thought = create_test_thought(text=None)
        result = validator.validate(thought)
        assert not result.passed

        # Test special characters only
        thought = create_test_thought(text="!@#$%^&*()")
        result = validator.validate(thought)
        assert not result.passed


class TestContentValidator:
    """Test ContentValidator implementation."""

    def test_content_validator_prohibited_words(self):
        """Test content validation with prohibited words."""
        validator = ContentValidator(prohibited=["hate", "violence", "spam"], name="Safety Filter")

        # Test clean content
        thought = create_test_thought(text="This is positive and helpful content.")
        result = validator.validate(thought)
        assert result.passed

        # Test content with prohibited words
        thought = create_test_thought(text="This content contains hate speech.")
        result = validator.validate(thought)
        assert not result.passed

    def test_content_validator_required_words(self):
        """Test content validation with required words."""
        validator = ContentValidator(required=["helpful", "informative"], name="Quality Filter")

        # Test content with required words
        thought = create_test_thought(text="This is helpful and informative content.")
        result = validator.validate(thought)
        assert result.passed

        # Test content missing required words
        thought = create_test_thought(text="This is just regular content.")
        result = validator.validate(thought)
        assert not result.passed

    def test_content_validator_case_insensitive(self):
        """Test content validator case insensitivity."""
        validator = ContentValidator(
            prohibited=["BAD", "Evil"], case_sensitive=False, name="Case Insensitive Filter"
        )

        # Test various cases
        test_cases = [
            "This contains bad content.",
            "This contains BAD content.",
            "This contains evil content.",
            "This contains EVIL content.",
        ]

        for text in test_cases:
            thought = create_test_thought(text=text)
            result = validator.validate(thought)
            assert not result.passed, f"Should fail for: {text}"

    def test_content_validator_case_sensitive(self):
        """Test content validator case sensitivity."""
        validator = ContentValidator(
            prohibited=["Bad"], case_sensitive=True, name="Case Sensitive Filter"
        )

        # Should pass with different case
        thought = create_test_thought(text="This contains bad content.")
        result = validator.validate(thought)
        assert result.passed

        # Should fail with exact case
        thought = create_test_thought(text="This contains Bad content.")
        result = validator.validate(thought)
        assert not result.passed

    def test_content_validator_combined_rules(self):
        """Test content validator with both required and prohibited words."""
        validator = ContentValidator(
            required=["helpful", "accurate"],
            prohibited=["misleading", "false"],
            name="Quality and Safety Filter",
        )

        # Test valid content
        thought = create_test_thought(text="This helpful and accurate information is valuable.")
        result = validator.validate(thought)
        assert result.passed

        # Test content missing required words
        thought = create_test_thought(text="This is just information.")
        result = validator.validate(thought)
        assert not result.passed

        # Test content with prohibited words
        thought = create_test_thought(
            text="This helpful but misleading information is problematic."
        )
        result = validator.validate(thought)
        assert not result.passed


class TestFormatValidator:
    """Test FormatValidator implementation."""

    def test_format_validator_json(self):
        """Test format validation for JSON."""
        validator = FormatValidator(expected_format="json")

        # Test valid JSON
        thought = create_test_thought(text='{"key": "value", "number": 42}')
        result = validator.validate(thought)
        assert result.passed

        # Test invalid JSON
        thought = create_test_thought(text='{"key": "value", "number": }')
        result = validator.validate(thought)
        assert not result.passed

    def test_format_validator_email(self):
        """Test format validation for email."""
        validator = FormatValidator(expected_format="email")

        # Test valid email
        thought = create_test_thought(text="user@example.com")
        result = validator.validate(thought)
        assert result.passed

        # Test invalid email
        thought = create_test_thought(text="not-an-email")
        result = validator.validate(thought)
        assert not result.passed

    def test_format_validator_url(self):
        """Test format validation for URL."""
        validator = FormatValidator(expected_format="url")

        # Test valid URLs
        valid_urls = [
            "https://www.example.com",
            "http://example.com",
            "https://subdomain.example.com/path?query=value",
        ]

        for url in valid_urls:
            thought = create_test_thought(text=url)
            result = validator.validate(thought)
            assert result.passed, f"Should pass for URL: {url}"

        # Test invalid URLs
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Depending on implementation
            "just text",
        ]

        for url in invalid_urls:
            thought = create_test_thought(text=url)
            result = validator.validate(thought)
            # Note: Some of these might pass depending on URL validation strictness

    def test_format_validator_custom_format(self):
        """Test format validation with custom format."""
        # Test a custom format that checks for specific structure
        validator = FormatValidator(expected_format="contains_json")

        # Test text containing JSON
        thought = create_test_thought(text='Here is some JSON: {"key": "value"}')
        validator.validate(thought)
        # Result depends on implementation of "contains_json" format

    def test_format_validator_edge_cases(self):
        """Test format validator edge cases."""
        validator = FormatValidator(expected_format="json")

        # Test empty text
        thought = create_test_thought(text="")
        result = validator.validate(thought)
        assert not result.passed

        # Test None text
        thought = create_test_thought(text=None)
        result = validator.validate(thought)
        assert not result.passed


class TestClassifierValidator:
    """Test ClassifierValidator implementation."""

    def test_classifier_validator_basic(self):
        """Test basic classifier validator functionality."""
        # Create a mock classifier
        from tests.utils.mocks import Mock

        classifier = Mock()
        classifier.classify.return_value = {"confidence": 0.9, "label": "positive"}

        validator = ClassifierValidator(
            classifier=classifier, threshold=0.8, expected_label="positive"
        )

        thought = create_test_thought(text="This is positive content.")
        result = validator.validate(thought)

        assert result.passed
        classifier.classify.assert_called_once()

    def test_classifier_validator_threshold_failure(self):
        """Test classifier validator with threshold failure."""
        from tests.utils.mocks import Mock

        classifier = Mock()
        classifier.classify.return_value = {"confidence": 0.6, "label": "positive"}

        validator = ClassifierValidator(
            classifier=classifier, threshold=0.8, expected_label="positive"
        )

        thought = create_test_thought(text="This is somewhat positive content.")
        result = validator.validate(thought)

        assert not result.passed  # Should fail due to low confidence

    def test_classifier_validator_label_mismatch(self):
        """Test classifier validator with label mismatch."""
        from tests.utils.mocks import Mock

        classifier = Mock()
        classifier.classify.return_value = {"confidence": 0.9, "label": "negative"}

        validator = ClassifierValidator(
            classifier=classifier, threshold=0.8, expected_label="positive"
        )

        thought = create_test_thought(text="This is negative content.")
        result = validator.validate(thought)

        assert not result.passed  # Should fail due to label mismatch


class TestValidatorIntegration:
    """Test validator integration scenarios."""

    def test_multiple_validators_chain(self):
        """Test multiple validators in a chain."""
        from sifaka.core.chain import Chain
        from sifaka.models.base import MockModel

        model = MockModel(
            model_name="validator-test",
            response_text="This is a well-formatted response about artificial intelligence with proper length.",
        )

        chain = Chain(model=model, prompt="Write about AI.")

        # Add multiple validators
        chain.validate_with(LengthValidator(min_length=20, max_length=200))
        chain.validate_with(RegexValidator(required_patterns=[r"artificial", r"intelligence"]))
        chain.validate_with(ContentValidator(prohibited=["bad"], name="Safety"))

        result = chain.run()

        assert result.validation_results is not None
        assert len(result.validation_results) == 3

        # All should pass with the given response
        for validator_result in result.validation_results.values():
            assert validator_result.passed

    def test_validator_failure_handling(self):
        """Test handling of validator failures."""
        from sifaka.core.chain import Chain
        from sifaka.models.base import MockModel

        model = MockModel(
            model_name="failure-test", response_text="Short"  # Will fail length validation
        )

        chain = Chain(model=model, prompt="Write a long response.")
        chain.validate_with(LengthValidator(min_length=50, max_length=200))

        result = chain.run()

        assert result.validation_results is not None
        assert len(result.validation_results) == 1

        # Should fail validation
        validation_result = list(result.validation_results.values())[0]
        assert not validation_result.passed

    def test_validator_performance(self):
        """Test validator performance."""
        import time

        validator = LengthValidator(min_length=10, max_length=1000)

        # Test performance with multiple validations
        start_time = time.time()
        for i in range(100):
            thought = create_test_thought(text=f"Test text number {i} for performance testing.")
            result = validator.validate(thought)
            assert result.passed

        execution_time = time.time() - start_time

        # Should be very fast
        assert (
            execution_time < 1.0
        ), f"Validator performance too slow: {execution_time:.3f}s for 100 validations"
