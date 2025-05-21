"""
Tests for the profanity classifier.
"""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.classifiers import ClassificationResult
from sifaka.classifiers.profanity_classifier import ProfanityClassifier


# Mock the better_profanity module to avoid actual profanity detection during tests
@pytest.fixture
def mock_better_profanity():
    """Create a mock better_profanity module for testing."""
    with patch.dict("sys.modules", {"better_profanity": MagicMock()}) as mock_modules:
        # Create a mock profanity module
        mock_profanity = MagicMock()
        mock_modules["better_profanity"].profanity = mock_profanity

        # Configure the mock methods
        mock_profanity.contains_profanity.return_value = False
        mock_profanity.censor.return_value = "This text is clean."

        yield mock_profanity


def test_profanity_classifier_initialization():
    """Test that a profanity classifier can be initialized with various options."""
    with patch.dict("sys.modules", {"better_profanity": MagicMock()}):
        # Test with default options
        classifier = ProfanityClassifier()
        assert classifier.name == "profanity_classifier"
        assert classifier.description == "Detects profanity and inappropriate language in text"
        assert classifier._custom_words == []
        assert classifier._censor_char == "*"

        # Test with custom words
        classifier = ProfanityClassifier(custom_words=["custom", "words"])
        assert classifier._custom_words == ["custom", "words"]

        # Test with custom censor character
        classifier = ProfanityClassifier(censor_char="#")
        assert classifier._censor_char == "#"

        # Test with custom name and description
        classifier = ProfanityClassifier(
            name="custom_profanity_classifier", description="Custom profanity detector"
        )
        assert classifier.name == "custom_profanity_classifier"
        assert classifier.description == "Custom profanity detector"


def test_profanity_classifier_classify_empty_text():
    """Test that classifying empty text returns clean with high confidence."""
    with patch.dict("sys.modules", {"better_profanity": MagicMock()}):
        classifier = ProfanityClassifier()
        result = classifier.classify("")

        assert isinstance(result, ClassificationResult)
        assert result.label == "clean"
        assert result.confidence == 1.0
        assert result.metadata is not None
        assert result.metadata["input_length"] == 0
        assert result.metadata["reason"] == "empty_text"


def test_profanity_classifier_classify_clean_text(mock_better_profanity):
    """Test that classifying clean text returns clean with high confidence."""
    # Configure mock to return clean text
    mock_better_profanity.contains_profanity.return_value = False

    classifier = ProfanityClassifier()
    classifier._initialized = True
    classifier._profanity = mock_better_profanity

    # Mock the censor method to return the same text (no censoring)
    mock_better_profanity.censor.return_value = "This is clean text."

    result = classifier.classify("This is clean text.")

    assert isinstance(result, ClassificationResult)
    assert result.label == "clean"
    assert result.confidence > 0.9
    assert result.metadata is not None
    assert result.metadata["profane_word_count"] == 0
    assert result.metadata["is_censored"] is False


def test_profanity_classifier_classify_profane_text(mock_better_profanity):
    """Test that classifying profane text returns profane with appropriate confidence."""
    # Configure mock to return profane text
    mock_better_profanity.contains_profanity.return_value = True
    mock_better_profanity.censor.return_value = "This text contains ***."

    # Mock the _get_profane_words method to return profane words
    with patch.object(ProfanityClassifier, "_get_profane_words", return_value={"badword"}):
        classifier = ProfanityClassifier()
        classifier._initialized = True
        classifier._profanity = mock_better_profanity

        result = classifier.classify("This text contains badword.")

        assert isinstance(result, ClassificationResult)
        assert result.label == "profane"
        assert result.confidence > 0.5
        assert result.metadata is not None
        assert result.metadata["profane_word_count"] == 1
        assert result.metadata["is_censored"] is True
        assert "badword" in result.metadata["profane_words"]


def test_profanity_classifier_batch_classify(mock_better_profanity):
    """Test that batch_classify correctly processes multiple texts."""

    # Configure mock to return different results for different texts
    def side_effect(text):
        if "profane" in text:
            return True
        return False

    mock_better_profanity.contains_profanity.side_effect = side_effect

    # Mock the _get_profane_words method
    def get_profane_words_side_effect(text):
        if "profane" in text:
            return {"profane"}
        return set()

    with patch.object(
        ProfanityClassifier, "_get_profane_words", side_effect=get_profane_words_side_effect
    ):
        classifier = ProfanityClassifier()
        classifier._initialized = True
        classifier._profanity = mock_better_profanity

        texts = ["This is clean text.", "This text contains profane content."]
        results = classifier.batch_classify(texts)

        assert len(results) == 2
        assert results[0].label == "clean"
        assert results[1].label == "profane"


def test_profanity_classifier_error_handling():
    """Test that the classifier handles errors gracefully."""
    with patch.dict("sys.modules", {"better_profanity": MagicMock()}):
        # Create a classifier
        classifier = ProfanityClassifier()

        # Mock the _initialize method to raise an exception when called
        # but don't raise the exception during the test setup
        classifier._initialize = MagicMock(side_effect=RuntimeError("Test error"))

        # Now when classify is called, it will try to initialize and get the error
        result = classifier.classify("This is test text.")

        assert isinstance(result, ClassificationResult)
        assert result.label == "clean"  # Default to clean on error
        assert result.confidence == 0.5  # Reduced confidence due to error
        assert result.metadata is not None
        assert "error" in result.metadata
        assert "classification_error" in result.metadata["reason"]
