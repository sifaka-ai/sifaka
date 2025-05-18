"""
Detailed tests for the language classifier.

This module contains more comprehensive tests for the language classifier
to improve test coverage.
"""

from typing import Any, Sequence
from unittest.mock import patch

import pytest

from sifaka.classifiers.language import LanguageClassifier


# Create a mock language detector for testing
class MockLanguageDetector:
    """Mock implementation of LanguageDetector for testing."""

    class MockLangProb:
        """Mock language probability object."""

        def __init__(self, lang: str, prob: float):
            self.lang = lang
            self.prob = prob

    def __init__(self, primary_lang: str = "en", primary_prob: float = 0.9):
        self.primary_lang = primary_lang
        self.primary_prob = primary_prob
        self.langs = [
            self.MockLangProb(primary_lang, primary_prob),
            self.MockLangProb("fr", 0.05),
            self.MockLangProb("de", 0.03),
            self.MockLangProb("es", 0.02),
        ]

    def detect_langs(self, text: str) -> Sequence[Any]:
        """Return mock language probabilities."""
        # Return empty list for empty text
        if not text or not text.strip():
            return []

        # Return different results based on the text content for testing
        if "french" in text.lower():
            return [
                self.MockLangProb("fr", 0.8),
                self.MockLangProb("en", 0.15),
                self.MockLangProb("de", 0.05),
            ]
        elif "german" in text.lower():
            return [
                self.MockLangProb("de", 0.7),
                self.MockLangProb("en", 0.2),
                self.MockLangProb("fr", 0.1),
            ]
        elif "low confidence" in text.lower():
            return [
                self.MockLangProb("en", 0.05),
                self.MockLangProb("fr", 0.04),
                self.MockLangProb("de", 0.03),
            ]

        # Default case
        return self.langs

    def detect(self, text: str) -> str:
        """Return the most likely language."""
        langs = self.detect_langs(text)
        if not langs:
            return "unknown"
        return langs[0].lang


class TestLanguageClassifierDetailed:
    """Detailed tests for the LanguageClassifier."""

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing with custom parameters."""
        detector = MockLanguageDetector()
        classifier = LanguageClassifier(
            min_confidence=0.3,
            fallback_lang="fr",
            fallback_confidence=0.2,
            seed=42,
            detector=detector,
            name="custom_language",
            description="Custom language classifier",
        )

        assert classifier.name == "custom_language"
        assert classifier.description == "Custom language classifier"
        # Access private attributes for testing
        assert classifier._min_confidence == 0.3
        assert classifier._fallback_lang == "fr"
        assert classifier._fallback_confidence == 0.2
        assert classifier._seed == 42
        assert classifier._detector == detector
        # The detector is provided but _initialized is still False until first use
        assert classifier._initialized is False

    def test_load_langdetect_error(self) -> None:
        """Test error handling when langdetect is not available."""
        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'langdetect'"),
        ):
            classifier = LanguageClassifier()

            with pytest.raises(ImportError) as excinfo:
                classifier._load_langdetect()

            assert "langdetect package is required" in str(excinfo.value)
            assert "pip install langdetect" in str(excinfo.value)

    def test_initialize_with_detector(self) -> None:
        """Test initialization with a provided detector."""
        detector = MockLanguageDetector()
        classifier = LanguageClassifier(detector=detector)

        # The detector is provided but _initialized is still False until first use
        assert classifier._initialized is False
        assert classifier._detector == detector

        # Calling _initialize should not change anything
        classifier._initialize()
        assert classifier._detector == detector

    def test_get_language_name(self) -> None:
        """Test getting language names from codes."""
        classifier = LanguageClassifier()

        assert classifier.get_language_name("en") == "English"
        assert classifier.get_language_name("fr") == "French"
        assert classifier.get_language_name("de") == "German"
        assert classifier.get_language_name("unknown") == "Unknown"
        assert (
            classifier.get_language_name("not_in_map") == "not_in_map"
        )  # Returns the code if not found

    def test_classify_empty_text(self) -> None:
        """Test classifying empty text."""
        detector = MockLanguageDetector()
        classifier = LanguageClassifier(detector=detector)

        result = classifier.classify("")

        assert result.label == "en"  # Default fallback
        assert result.confidence == 0.0  # Default fallback confidence
        assert result.metadata["input_length"] == 0
        assert result.metadata["reason"] == "empty_text"
        assert result.metadata["language_name"] == "English"

    def test_classify_english_text(self) -> None:
        """Test classifying English text."""
        detector = MockLanguageDetector(primary_lang="en", primary_prob=0.95)
        classifier = LanguageClassifier(detector=detector)

        result = classifier.classify("This is a sample English text.")

        assert result.label == "en"
        assert result.confidence == 0.95
        assert result.metadata["language_name"] == "English"
        assert result.metadata["input_length"] == len("This is a sample English text.")
        assert "all_langs" in result.metadata
        assert len(result.metadata["all_langs"]) == 4  # Should have 4 languages in the mock

    def test_classify_french_text(self) -> None:
        """Test classifying French text."""
        detector = MockLanguageDetector()
        classifier = LanguageClassifier(detector=detector)

        result = classifier.classify("This is french text for testing.")

        assert result.label == "fr"
        assert result.confidence == 0.8
        assert result.metadata["language_name"] == "French"
        assert "all_langs" in result.metadata
        assert (
            len(result.metadata["all_langs"]) == 3
        )  # Should have 3 languages in the mock for French

    def test_classify_low_confidence(self) -> None:
        """Test classifying text with low confidence."""
        detector = MockLanguageDetector()
        classifier = LanguageClassifier(min_confidence=0.1, detector=detector)

        result = classifier.classify("This is low confidence text.")

        assert result.label == "en"  # Default fallback
        assert result.confidence == 0.0  # Default fallback confidence
        assert result.metadata["reason"] == "low_confidence"
        assert "detected_lang" in result.metadata
        assert "detected_prob" in result.metadata
        assert result.metadata["detected_lang"] == "en"
        assert result.metadata["detected_prob"] == 0.05

    def test_batch_classify(self) -> None:
        """Test batch classification of multiple texts."""
        detector = MockLanguageDetector()
        classifier = LanguageClassifier(detector=detector)

        texts = [
            "This is English text.",
            "This is french text.",
            "",
            "This is german text.",
        ]

        results = classifier.batch_classify(texts)

        assert len(results) == 4
        assert results[0].label == "en"
        assert results[1].label == "fr"
        assert results[2].label == "en"  # Empty text gets fallback
        assert results[2].metadata["reason"] == "empty_text"
        assert results[3].label == "de"

    def test_create_with_custom_detector(self) -> None:
        """Test creating a classifier with a custom detector using the class method."""
        detector = MockLanguageDetector(primary_lang="es", primary_prob=0.85)

        classifier = LanguageClassifier.create_with_custom_detector(
            detector=detector,
            name="spanish_detector",
            description="Spanish language detector",
            min_confidence=0.2,
            fallback_lang="es",
            fallback_confidence=0.1,
        )

        assert classifier.name == "spanish_detector"
        assert classifier.description == "Spanish language detector"
        assert classifier._min_confidence == 0.2
        assert classifier._fallback_lang == "es"
        assert classifier._fallback_confidence == 0.1
        assert classifier._detector == detector

        # Test that it works
        result = classifier.classify("Test text")
        assert result.label == "es"  # From our mock detector
        assert result.confidence == 0.85
