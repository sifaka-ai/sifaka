"""Tests for the LanguageClassifier."""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.base import ClassificationResult


class MockLanguage:
    """Mock language detection result."""

    def __init__(self, lang: str, prob: float):
        self.lang = lang
        self.prob = prob


@pytest.fixture
def mock_detect_langs(monkeypatch):
    """Create a mock language detector."""

    def mock_detector(text: str) -> List[MockLanguage]:
        # Default to English for empty or whitespace
        if not text.strip():
            return [MockLanguage("en", 1.0)]

        # Mock different languages based on content
        if "bonjour" in text.lower():
            return [MockLanguage("fr", 0.8), MockLanguage("en", 0.2)]
        elif "hola" in text.lower():
            return [MockLanguage("es", 0.9), MockLanguage("en", 0.1)]
        elif "こんにちは" in text:
            return [MockLanguage("ja", 0.95), MockLanguage("zh-cn", 0.05)]
        else:
            return [MockLanguage("en", 0.7), MockLanguage("fr", 0.2), MockLanguage("de", 0.1)]

    mock = MagicMock(side_effect=mock_detector)
    monkeypatch.setattr("langdetect.detect_langs", mock)
    return mock


@pytest.fixture
def classifier():
    """Create a LanguageClassifier instance."""
    return LanguageClassifier(
        name="test_language_classifier", description="Test language classifier", min_confidence=0.1
    )


def test_initialization():
    """Test initialization of LanguageClassifier."""
    # Test default initialization
    classifier = LanguageClassifier()
    assert classifier.name == "language_classifier"
    assert "Detects text language" in classifier.description
    assert classifier.min_confidence == 0.1
    assert classifier.cost == 1

    # Test custom initialization
    classifier = LanguageClassifier(
        name="custom_classifier",
        description="Custom description",
        min_confidence=0.2,
        config={"custom": "config"},
    )
    assert classifier.name == "custom_classifier"
    assert classifier.description == "Custom description"
    assert classifier.min_confidence == 0.2
    assert classifier.config == {"custom": "config"}


def test_language_detection(classifier, mock_detect_langs):
    """Test language detection for various inputs."""
    # Test English text
    result = classifier.classify("Hello, how are you?")
    assert result.label == "en"
    assert result.confidence == 0.7
    assert "English" in result.metadata["language_name"]

    # Test French text
    result = classifier.classify("Bonjour, comment allez-vous?")
    assert result.label == "fr"
    assert result.confidence == 0.8
    assert "French" in result.metadata["language_name"]

    # Test Spanish text
    result = classifier.classify("Hola, ¿cómo estás?")
    assert result.label == "es"
    assert result.confidence == 0.9
    assert "Spanish" in result.metadata["language_name"]

    # Test Japanese text
    result = classifier.classify("こんにちは、お元気ですか？")
    assert result.label == "ja"
    assert result.confidence == 0.95
    assert "Japanese" in result.metadata["language_name"]


def test_metadata_handling(classifier, mock_detect_langs):
    """Test metadata handling in classification results."""
    result = classifier.classify("Hello, how are you?")

    # Check metadata structure
    assert "language_name" in result.metadata
    assert "all_languages" in result.metadata

    # Check all_languages format
    all_langs = result.metadata["all_languages"]
    assert isinstance(all_langs, dict)
    assert all(isinstance(lang_data, dict) for lang_data in all_langs.values())
    assert all("probability" in lang_data for lang_data in all_langs.values())
    assert all("name" in lang_data for lang_data in all_langs.values())


def test_confidence_threshold(classifier, mock_detect_langs):
    """Test minimum confidence threshold handling."""
    # Set higher minimum confidence
    classifier.min_confidence = 0.3

    result = classifier.classify("Hello, how are you?")
    all_langs = result.metadata["all_languages"]

    # Only languages with confidence >= 0.3 should be included
    assert "en" in all_langs  # 0.7 confidence
    assert "de" not in all_langs  # 0.1 confidence


def test_error_handling(classifier, mock_detect_langs, monkeypatch):
    """Test error handling in language detection."""
    # Test with None input
    with pytest.raises(ValueError):
        classifier.classify(None)

    # Test with empty string
    result = classifier.classify("")
    assert result.label == "en"
    assert result.confidence == 1.0

    # Test with whitespace
    result = classifier.classify("   \n\t   ")
    assert result.label == "en"
    assert result.confidence == 1.0

    # Test detection failure
    def mock_error(*args):
        raise Exception("Detection failed")

    monkeypatch.setattr("langdetect.detect_langs", mock_error)
    result = classifier.classify("Test text")
    assert result.label == "en"  # Default to English
    assert result.confidence == 0.0
    assert "error" in result.metadata


def test_batch_classification(classifier, mock_detect_langs):
    """Test batch classification functionality."""
    texts = [
        "Hello, how are you?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "こんにちは、お元気ですか？",
    ]

    results = classifier.batch_classify(texts)

    assert len(results) == len(texts)
    assert all(isinstance(result, ClassificationResult) for result in results)

    # Check individual results
    assert results[0].label == "en"
    assert results[1].label == "fr"
    assert results[2].label == "es"
    assert results[3].label == "ja"


def test_language_name_lookup(classifier):
    """Test language name lookup functionality."""
    # Test known language codes
    assert classifier.get_language_name("en") == "English"
    assert classifier.get_language_name("fr") == "French"
    assert classifier.get_language_name("ja") == "Japanese"
    assert classifier.get_language_name("zh-cn") == "Chinese (Simplified)"

    # Test unknown language code
    assert classifier.get_language_name("xx") == "Unknown"


def test_consistent_results(classifier, mock_detect_langs):
    """Test consistency of classification results."""
    text = "Hello, this is a test text."

    result1 = classifier.classify(text)
    result2 = classifier.classify(text)

    assert result1.label == result2.label
    assert result1.confidence == result2.confidence
    assert result1.metadata == result2.metadata
