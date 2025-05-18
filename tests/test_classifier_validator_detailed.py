"""
Detailed tests for the classifier validator.

This module contains more comprehensive tests for the classifier validator
to improve test coverage.
"""

import pytest
from unittest.mock import MagicMock
from typing import List, Optional, Dict, Any

from sifaka.classifiers import ClassificationResult, Classifier
from sifaka.validators.classifier import ClassifierValidator, ClassifierValidatorConfig
from sifaka.errors import ValidationError
from sifaka.results import ValidationResult


# Create a mock classifier for testing
class MockClassifier:
    """Mock classifier for testing."""

    def __init__(
        self, label: str = "positive", confidence: float = 0.8, name: str = "mock_classifier"
    ):
        self.label = label
        self.confidence = confidence
        self._name = name
        self._description = f"Mock {name} classifier"

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def classify(self, text: str) -> ClassificationResult:
        """Classify text."""
        # Return different results based on the text content for testing
        if not text or not text.strip():
            return ClassificationResult(
                label="neutral",
                confidence=1.0,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        if "negative" in text.lower():
            return ClassificationResult(
                label="negative", confidence=0.9, metadata={"input_length": len(text)}
            )

        if "neutral" in text.lower():
            return ClassificationResult(
                label="neutral", confidence=0.7, metadata={"input_length": len(text)}
            )

        if "low confidence" in text.lower():
            return ClassificationResult(
                label=self.label,
                confidence=0.2,
                metadata={"input_length": len(text), "reason": "low_confidence"},
            )

        # Default case
        return ClassificationResult(
            label=self.label, confidence=self.confidence, metadata={"input_length": len(text)}
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]


class TestClassifierValidatorConfigDetailed:
    """Detailed tests for the ClassifierValidatorConfig class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        # Need to provide at least one label list to avoid validation error
        config = ClassifierValidatorConfig(valid_labels=["positive"])

        assert config.threshold == 0.5
        assert config.valid_labels == ["positive"]
        assert config.invalid_labels is None
        assert config.extraction_function is None

    def test_init_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        valid_labels = ["positive", "neutral"]
        invalid_labels = ["negative"]
        extraction_func = lambda x: x.strip()

        config = ClassifierValidatorConfig(
            threshold=0.7,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_func,
        )

        assert config.threshold == 0.7
        assert config.valid_labels == valid_labels
        assert config.invalid_labels == invalid_labels
        assert config.extraction_function == extraction_func

    def test_init_with_invalid_threshold(self) -> None:
        """Test initialization with an invalid threshold."""
        with pytest.raises(ValidationError) as excinfo:
            ClassifierValidatorConfig(threshold=1.5)

        assert "Threshold must be between 0.0 and 1.0" in str(excinfo.value)
        assert excinfo.value.component == "ClassifierValidatorConfig"
        assert excinfo.value.operation == "initialization"

    def test_init_without_labels(self) -> None:
        """Test initialization without any labels."""
        with pytest.raises(ValidationError) as excinfo:
            ClassifierValidatorConfig(valid_labels=[], invalid_labels=None)

        assert "Either valid_labels or invalid_labels must be provided" in str(excinfo.value)
        assert excinfo.value.component == "ClassifierValidatorConfig"
        assert excinfo.value.operation == "initialization"


class TestClassifierValidatorDetailed:
    """Detailed tests for the ClassifierValidator class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        classifier = MockClassifier()
        validator = ClassifierValidator(classifier=classifier, valid_labels=["positive"])

        assert validator.name == "MockClassifierValidator"
        assert validator.classifier == classifier
        assert validator.config.threshold == 0.5
        assert validator.config.valid_labels == ["positive"]
        assert validator.config.invalid_labels is None
        assert validator.config.extraction_function is None

    def test_init_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        classifier = MockClassifier()
        extraction_func = lambda x: x.strip()

        validator = ClassifierValidator(
            classifier=classifier,
            threshold=0.7,
            valid_labels=["positive", "neutral"],
            invalid_labels=["negative"],
            extraction_function=extraction_func,
            name="custom_validator",
        )

        assert validator.name == "custom_validator"
        assert validator.classifier == classifier
        assert validator.config.threshold == 0.7
        assert validator.config.valid_labels == ["positive", "neutral"]
        assert validator.config.invalid_labels == ["negative"]
        assert validator.config.extraction_function == extraction_func

    def test_init_without_classifier(self) -> None:
        """Test initialization without a classifier."""
        with pytest.raises(ValidationError) as excinfo:
            ClassifierValidator(classifier=None, valid_labels=["positive"])

        assert "Classifier must be provided" in str(excinfo.value)
        assert excinfo.value.component == "ClassifierValidator"
        assert excinfo.value.operation == "initialization"

    def test_validate_empty_text(self) -> None:
        """Test validation of empty text."""
        classifier = MockClassifier()
        validator = ClassifierValidator(classifier=classifier, valid_labels=["positive", "neutral"])

        result = validator._validate("")

        # The mock classifier returns "neutral" for empty text, which is in valid_labels
        assert result.passed is True
        assert "neutral" in result.message
        assert result._details["label"] == "neutral"
        assert result._details["confidence"] == 1.0
        assert result._details["classifier_metadata"]["reason"] == "empty_text"

    def test_validate_valid_label(self) -> None:
        """Test validation with a valid label."""
        classifier = MockClassifier(label="positive", confidence=0.9)
        validator = ClassifierValidator(classifier=classifier, valid_labels=["positive"])

        result = validator._validate("This is a positive text.")

        assert result.passed is True
        assert "positive" in result.message
        assert result._details["label"] == "positive"
        assert result._details["confidence"] == 0.9
        assert result.score >= 0.9  # Score should be at least the confidence

    def test_validate_invalid_label(self) -> None:
        """Test validation with an invalid label."""
        classifier = MockClassifier()
        validator = ClassifierValidator(classifier=classifier, invalid_labels=["negative"])

        result = validator._validate("This is a negative text.")

        assert result.passed is False
        assert "negative" in result.message
        assert result._details["label"] == "negative"
        assert result._details["confidence"] == 0.9
        assert result.score <= 0.5  # Score should be low for invalid labels
        assert len(result.issues) > 0
        assert "not allowed" in result.issues[0]

    def test_validate_not_in_valid_labels(self) -> None:
        """Test validation with a label not in valid_labels."""
        classifier = MockClassifier(label="neutral", confidence=0.8)
        validator = ClassifierValidator(classifier=classifier, valid_labels=["positive"])

        result = validator._validate("This is a neutral text.")

        assert result.passed is False
        assert "not in the list of valid labels" in result.message
        assert result._details["label"] == "neutral"
        assert result._details["confidence"] == 0.7  # From the mock for "neutral" text
        assert result.score == 0.0  # Score should be 0 for non-valid labels
        assert len(result.issues) > 0
        assert "expected one of: positive" in result.issues[0]

    def test_validate_not_in_invalid_labels(self) -> None:
        """Test validation with a label not in invalid_labels."""
        classifier = MockClassifier(label="positive", confidence=0.8)
        validator = ClassifierValidator(classifier=classifier, invalid_labels=["negative"])

        result = validator._validate("This is a positive text.")

        assert result.passed is True
        assert "not in the list of invalid labels" in result.message
        assert result._details["label"] == "positive"
        assert result._details["confidence"] == 0.8
        assert result.score >= 0.5  # Score should be high for non-invalid labels

    def test_validate_below_threshold(self) -> None:
        """Test validation with confidence below threshold."""
        classifier = MockClassifier()
        validator = ClassifierValidator(
            classifier=classifier, threshold=0.5, valid_labels=["positive"]
        )

        result = validator._validate("This is a low confidence text.")

        assert result.passed is False
        assert "below threshold" in result.message
        assert result._details["label"] == "positive"
        assert result._details["confidence"] == 0.2
        assert result.score < 0.5  # Score should be low for below-threshold results
        assert len(result.issues) > 0
        assert "below threshold" in result.issues[0]

    def test_validate_with_extraction_function(self) -> None:
        """Test validation with an extraction function."""
        classifier = MockClassifier()

        # Create an extraction function that extracts text between markers
        def extract_between_markers(text: str) -> str:
            start = text.find("START:")
            end = text.find(":END")
            if start >= 0 and end > start:
                return text[start + 6 : end].strip()
            return text

        validator = ClassifierValidator(
            classifier=classifier,
            valid_labels=["positive"],
            extraction_function=extract_between_markers,
        )

        # The text between markers is negative, but the overall text is not
        result = validator._validate(
            "This is a text with START: This is a negative text. :END embedded in it."
        )

        assert result.passed is False
        assert "negative" in result._details["label"]
        assert result._details["confidence"] == 0.9  # From the mock for "negative" text
