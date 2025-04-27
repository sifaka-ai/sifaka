"""Tests for the language classifier."""

import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.base import ClassificationResult


class MockLanguage:
    """Mock Language class from langdetect."""

    def __init__(self, lang: str, prob: float):
        self.lang = lang
        self.prob = prob


@pytest.fixture
def mock_detect_langs():
    """Create a mock detect_langs function."""

    def detect_langs(text: str) -> List[MockLanguage]:
        # Simple mock implementation that returns English for English-like text
        # and other languages based on specific patterns
        if "こんにちは" in text:
            return [MockLanguage("ja", 0.9), MockLanguage("zh-cn", 0.1)]
        elif "bonjour" in text:
            return [MockLanguage("fr", 0.8), MockLanguage("en", 0.2)]
        elif "hola" in text:
            return [MockLanguage("es", 0.85), MockLanguage("pt", 0.15)]
        elif not text.strip():
            return []
        else:
            return [MockLanguage("en", 0.95), MockLanguage("fr", 0.05)]

    return detect_langs


@pytest.fixture
def language_classifier(mock_detect_langs):
    """Create a LanguageClassifier instance with mocked langdetect."""
    with patch("importlib.import_module") as mock_import:
        mock_langdetect = MagicMock()
        mock_langdetect.detect_langs = mock_detect_langs
        mock_langdetect.DetectorFactory = MagicMock()
        mock_import.return_value = mock_langdetect

        classifier = LanguageClassifier()
        classifier.warm_up()
        return classifier


def test_initialization():
    """Test LanguageClassifier initialization."""
    # Test basic initialization
    classifier = LanguageClassifier()
    assert classifier.name == "language_classifier"
    assert classifier.description == "Detects text language"
    assert classifier.min_confidence == 0.1
    assert classifier.labels == list(LanguageClassifier.LANGUAGE_NAMES.keys())
    assert classifier.cost == 1

    # Test custom initialization
    custom_classifier = LanguageClassifier(
        name="custom",
        description="custom classifier",
        min_confidence=0.2,
        config={"param": "value"},
    )
    assert custom_classifier.name == "custom"
    assert custom_classifier.description == "custom classifier"
    assert custom_classifier.min_confidence == 0.2
    assert custom_classifier.config == {"param": "value"}


def test_warm_up(language_classifier, mock_detect_langs):
    """Test warm_up functionality."""
    assert language_classifier._detect_langs == mock_detect_langs

    # Test error handling
    with patch("importlib.import_module", side_effect=ImportError()):
        classifier = LanguageClassifier()
        with pytest.raises(ImportError):
            classifier.warm_up()

    with patch("importlib.import_module", side_effect=RuntimeError()):
        classifier = LanguageClassifier()
        with pytest.raises(RuntimeError):
            classifier.warm_up()


def test_language_names():
    """Test language name mapping."""
    classifier = LanguageClassifier()

    # Test known language codes
    assert classifier.get_language_name("en") == "English"
    assert classifier.get_language_name("es") == "Spanish"
    assert classifier.get_language_name("fr") == "French"
    assert classifier.get_language_name("ja") == "Japanese"

    # Test unknown language code
    assert classifier.get_language_name("xx") == "Unknown"


def test_classification(language_classifier):
    """Test text classification."""
    # Test English text
    result = language_classifier.classify("This is a test sentence in English.")
    assert isinstance(result, ClassificationResult)
    assert result.label == "en"
    assert result.confidence > 0.9
    assert result.metadata["language_name"] == "English"
    assert isinstance(result.metadata["all_languages"], dict)

    # Test Japanese text
    result = language_classifier.classify("こんにちは")
    assert result.label == "ja"
    assert result.confidence > 0.8
    assert result.metadata["language_name"] == "Japanese"

    # Test French text
    result = language_classifier.classify("bonjour")
    assert result.label == "fr"
    assert result.confidence > 0.7
    assert result.metadata["language_name"] == "French"

    # Test Spanish text
    result = language_classifier.classify("hola")
    assert result.label == "es"
    assert result.confidence > 0.8
    assert result.metadata["language_name"] == "Spanish"

    # Test empty text
    result = language_classifier.classify("")
    assert result.label == "en"  # Default to English
    assert result.confidence == 0.0
    assert "error" in result.metadata

    # Test whitespace text
    result = language_classifier.classify("   \n\t   ")
    assert result.label == "en"  # Default to English
    assert result.confidence == 0.0
    assert "error" in result.metadata


def test_batch_classification(language_classifier):
    """Test batch text classification."""
    texts = [
        "This is English text.",
        "こんにちは",  # Japanese
        "bonjour",  # French
        "hola",  # Spanish
        "",  # Empty text
        "!@#$%^&*()",  # Special characters
    ]

    results = language_classifier.batch_classify(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Verify each result
    assert results[0].label == "en"
    assert results[1].label == "ja"
    assert results[2].label == "fr"
    assert results[3].label == "es"
    assert results[4].label == "en"  # Empty text defaults to English
    assert results[5].label == "en"  # Special chars likely detected as English

    for result in results:
        assert isinstance(result, ClassificationResult)
        assert 0 <= result.confidence <= 1
        assert "language_name" in result.metadata
        assert isinstance(result.metadata["all_languages"], dict)


def test_edge_cases(language_classifier):
    """Test edge cases."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "numbers_only": "123 456 789",
        "mixed_scripts": "Hello こんにちは Bonjour",
        "very_long": "a" * 10000,
        "single_char": "a",
        "repeated_char": "a" * 100,
        "newlines": "Line 1\nLine 2\nLine 3",
    }

    for case_name, text in edge_cases.items():
        result = language_classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert result.label in language_classifier.labels
        assert 0 <= result.confidence <= 1
        assert isinstance(result.metadata, dict)
        assert "language_name" in result.metadata
        assert "all_languages" in result.metadata


def test_error_handling(language_classifier):
    """Test error handling."""
    invalid_inputs = [None, 123, [], {}]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            language_classifier.classify(invalid_input)

        with pytest.raises(Exception):
            language_classifier.batch_classify([invalid_input])


def test_consistent_results(language_classifier):
    """Test consistency of classification results."""
    test_texts = {
        "english": "This is a test text that should give consistent results.",
        "japanese": "こんにちは",
        "french": "bonjour",
        "spanish": "hola",
    }

    for lang, text in test_texts.items():
        # Test single classification consistency
        results = [language_classifier.classify(text) for _ in range(3)]
        first_result = results[0]
        for result in results[1:]:
            assert result.label == first_result.label
            assert result.confidence == first_result.confidence
            assert result.metadata == first_result.metadata

        # Test batch classification consistency
        batch_results = [language_classifier.batch_classify([text]) for _ in range(3)]
        first_batch = batch_results[0]
        for batch in batch_results[1:]:
            assert len(batch) == len(first_batch)
            for r1, r2 in zip(batch, first_batch):
                assert r1.label == r2.label
                assert r1.confidence == r2.confidence
                assert r1.metadata == r2.metadata


def test_confidence_thresholds(language_classifier):
    """Test different confidence thresholds."""
    text = "This is a mixed text avec un peu de français y algo de español"

    # Test with different min_confidence values
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        classifier = LanguageClassifier(min_confidence=threshold)
        classifier._detect_langs = language_classifier._detect_langs

        result = classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert result.label in classifier.labels

        # Check that only languages above threshold are included
        for lang_info in result.metadata["all_languages"].values():
            assert lang_info["probability"] >= threshold
