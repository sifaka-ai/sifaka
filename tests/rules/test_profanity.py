"""Tests for the ProfanityClassifier."""

import pytest
from typing import Dict, Any, List, Optional, Set
from unittest.mock import MagicMock

from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.base import ClassificationResult


class MockProfanity:
    """Mock profanity checker for testing."""

    def __init__(self):
        self.censor_char = "*"
        self.custom_words: Set[str] = set()
        self.base_profane_words = {"bad", "inappropriate", "offensive"}

    def contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        text_lower = text.lower()
        return any(word in text_lower for word in self.base_profane_words | self.custom_words)

    def censor(self, text: str) -> str:
        """Censor profane words in text."""
        text_lower = text.lower()
        censored = text
        for word in self.base_profane_words | self.custom_words:
            if word in text_lower:
                # Replace with censor characters matching word length
                censored = censored.replace(word, self.censor_char * len(word))
        return censored

    def add_censor_words(self, words: Set[str]) -> None:
        """Add custom words to censor."""
        self.custom_words.update(words)


@pytest.fixture
def mock_profanity(monkeypatch):
    """Create a mock profanity checker."""
    mock = MockProfanity()
    monkeypatch.setattr("better_profanity.Profanity", lambda: mock)
    return mock


@pytest.fixture
def classifier():
    """Create a ProfanityClassifier instance."""
    return ProfanityClassifier(
        name="test_profanity_classifier",
        description="Test profanity classifier",
        custom_words={"custom_bad_word"},
        censor_char="#",
    )


def test_initialization():
    """Test initialization of ProfanityClassifier."""
    # Test default initialization
    classifier = ProfanityClassifier()
    assert classifier.name == "profanity_classifier"
    assert "Detects profanity" in classifier.description
    assert classifier.custom_words == set()
    assert classifier.censor_char == "*"
    assert classifier.cost == 1

    # Test custom initialization
    custom_words = {"bad_word1", "bad_word2"}
    classifier = ProfanityClassifier(
        name="custom_classifier",
        description="Custom description",
        custom_words=custom_words,
        censor_char="#",
        config={"custom": "config"},
    )
    assert classifier.name == "custom_classifier"
    assert classifier.description == "Custom description"
    assert classifier.custom_words == custom_words
    assert classifier.censor_char == "#"
    assert classifier.config == {"custom": "config"}


def test_profanity_detection(classifier, mock_profanity):
    """Test profanity detection for various inputs."""
    # Test clean text
    result = classifier.classify("Hello, how are you?")
    assert result.label == "clean"
    assert result.confidence > 0.9
    assert not result.metadata["contains_profanity"]
    assert result.metadata["censored_text"] == "Hello, how are you?"

    # Test text with base profanity
    result = classifier.classify("This is bad and inappropriate content")
    assert result.label == "profane"
    assert result.confidence > 0.5
    assert result.metadata["contains_profanity"]
    assert "###" in result.metadata["censored_text"]  # "bad" censored
    assert "############" in result.metadata["censored_text"]  # "inappropriate" censored

    # Test text with custom profanity
    result = classifier.classify("This contains custom_bad_word")
    assert result.label == "profane"
    assert result.confidence > 0.5
    assert result.metadata["contains_profanity"]
    assert "##############" in result.metadata["censored_text"]  # "custom_bad_word" censored


def test_custom_words_handling(classifier, mock_profanity):
    """Test handling of custom profane words."""
    # Add new custom words
    new_words = {"new_bad_word", "another_bad_word"}
    classifier.add_custom_words(new_words)

    # Test detection with new custom words
    result = classifier.classify("This contains new_bad_word")
    assert result.label == "profane"
    assert result.metadata["contains_profanity"]
    assert "###########" in result.metadata["censored_text"]

    # Verify custom words were added to internal set
    assert new_words.issubset(classifier.custom_words)


def test_metadata_handling(classifier, mock_profanity):
    """Test metadata handling in classification results."""
    result = classifier.classify("This is bad content")

    # Check metadata structure
    assert "contains_profanity" in result.metadata
    assert "censored_text" in result.metadata
    assert "censored_word_count" in result.metadata

    # Check metadata values
    assert isinstance(result.metadata["contains_profanity"], bool)
    assert isinstance(result.metadata["censored_text"], str)
    assert isinstance(result.metadata["censored_word_count"], int)


def test_confidence_calculation(classifier, mock_profanity):
    """Test confidence score calculation."""
    # Test with no profanity
    result = classifier.classify("Clean text without any issues")
    assert result.confidence == 1.0  # High confidence for clean text

    # Test with single profane word in short text
    result = classifier.classify("This is bad")
    assert 0.3 <= result.confidence <= 1.0  # Confidence based on proportion

    # Test with multiple profane words
    result = classifier.classify("This is bad and inappropriate text")
    assert 0.3 <= result.confidence <= 1.0  # Higher confidence due to multiple matches


def test_error_handling(classifier, mock_profanity):
    """Test error handling in profanity detection."""
    # Test with None input
    with pytest.raises(ValueError):
        classifier.classify(None)

    # Test with empty string
    result = classifier.classify("")
    assert result.label == "clean"
    assert result.confidence == 1.0

    # Test with whitespace
    result = classifier.classify("   \n\t   ")
    assert result.label == "clean"
    assert result.confidence == 1.0

    # Test with non-string input
    with pytest.raises(ValueError):
        classifier.classify(123)


def test_batch_classification(classifier, mock_profanity):
    """Test batch classification functionality."""
    texts = [
        "Hello, how are you?",
        "This is bad content",
        "Another clean text",
        "This has custom_bad_word in it",
    ]

    results = classifier.batch_classify(texts)

    assert len(results) == len(texts)
    assert all(isinstance(result, ClassificationResult) for result in results)

    # Check individual results
    assert results[0].label == "clean"
    assert results[1].label == "profane"
    assert results[2].label == "clean"
    assert results[3].label == "profane"


def test_censor_character_handling(classifier, mock_profanity):
    """Test custom censor character handling."""
    # Test with default censor char
    default_classifier = ProfanityClassifier()
    result = default_classifier.classify("This is bad")
    assert "***" in result.metadata["censored_text"]

    # Test with custom censor char
    custom_classifier = ProfanityClassifier(censor_char="#")
    result = custom_classifier.classify("This is bad")
    assert "###" in result.metadata["censored_text"]


def test_consistent_results(classifier, mock_profanity):
    """Test consistency of classification results."""
    text = "This is bad and inappropriate content"

    result1 = classifier.classify(text)
    result2 = classifier.classify(text)

    assert result1.label == result2.label
    assert result1.confidence == result2.confidence
    assert result1.metadata == result2.metadata
