"""Tests for the profanity classifier."""

import pytest
from typing import Dict, Any, List, Set
from unittest.mock import MagicMock, patch

from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.base import ClassificationResult


class MockProfanity:
    """Mock Profanity class from better_profanity."""

    def __init__(self):
        self.censor_char = "*"
        self.custom_words: Set[str] = set()
        self.profane_words = {"bad", "inappropriate", "offensive"}

    def contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        text_lower = text.lower()
        return any(word in text_lower for word in self.profane_words | self.custom_words)

    def censor(self, text: str) -> str:
        """Censor profane words in text."""
        text_lower = text.lower()
        censored = text
        for word in self.profane_words | self.custom_words:
            if word in text_lower:
                start = text_lower.find(word)
                censored = (
                    censored[:start] + self.censor_char * len(word) + censored[start + len(word) :]
                )
        return censored

    def add_censor_words(self, words: Set[str]) -> None:
        """Add custom words to censor."""
        self.custom_words.update(words)


@pytest.fixture
def mock_profanity():
    """Create a mock Profanity instance."""
    return MockProfanity()


@pytest.fixture
def profanity_classifier(mock_profanity):
    """Create a ProfanityClassifier instance with mocked better_profanity."""
    with patch("importlib.import_module") as mock_import:
        mock_profanity_module = MagicMock()
        mock_profanity_module.Profanity = MagicMock(return_value=mock_profanity)
        mock_import.return_value = mock_profanity_module

        classifier = ProfanityClassifier()
        classifier.warm_up()
        return classifier


def test_initialization():
    """Test ProfanityClassifier initialization."""
    # Test basic initialization
    classifier = ProfanityClassifier()
    assert classifier.name == "profanity_classifier"
    assert classifier.description == "Detects profanity and inappropriate language"
    assert classifier.custom_words == set()
    assert classifier.censor_char == "*"
    assert classifier.labels == ["clean", "profane"]
    assert classifier.cost == 1

    # Test custom initialization
    custom_words = {"custom1", "custom2"}
    custom_classifier = ProfanityClassifier(
        name="custom",
        description="custom classifier",
        custom_words=custom_words,
        censor_char="#",
        config={"param": "value"},
    )
    assert custom_classifier.name == "custom"
    assert custom_classifier.description == "custom classifier"
    assert custom_classifier.custom_words == custom_words
    assert custom_classifier.censor_char == "#"
    assert custom_classifier.config == {"param": "value"}


def test_warm_up(profanity_classifier, mock_profanity):
    """Test warm_up functionality."""
    assert profanity_classifier._profanity == mock_profanity

    # Test error handling
    with patch("importlib.import_module", side_effect=ImportError()):
        classifier = ProfanityClassifier()
        with pytest.raises(ImportError):
            classifier.warm_up()

    with patch("importlib.import_module", side_effect=RuntimeError()):
        classifier = ProfanityClassifier()
        with pytest.raises(RuntimeError):
            classifier.warm_up()


def test_classification(profanity_classifier):
    """Test text classification."""
    # Test clean text
    result = profanity_classifier.classify("This is a clean text.")
    assert isinstance(result, ClassificationResult)
    assert result.label == "clean"
    assert result.confidence > 0.9
    assert not result.metadata["contains_profanity"]
    assert result.metadata["censored_text"] == "This is a clean text."
    assert result.metadata["censored_word_count"] == 0

    # Test text with profanity
    result = profanity_classifier.classify("This text is bad and inappropriate.")
    assert result.label == "profane"
    assert result.confidence > 0
    assert result.metadata["contains_profanity"]
    assert "***" in result.metadata["censored_text"]
    assert "************" in result.metadata["censored_text"]
    assert result.metadata["censored_word_count"] > 0

    # Test empty text
    result = profanity_classifier.classify("")
    assert result.label == "clean"
    assert result.confidence == 1.0
    assert not result.metadata["contains_profanity"]
    assert result.metadata["censored_text"] == ""
    assert result.metadata["censored_word_count"] == 0

    # Test whitespace text
    result = profanity_classifier.classify("   \n\t   ")
    assert result.label == "clean"
    assert result.confidence == 1.0
    assert not result.metadata["contains_profanity"]
    assert result.metadata["censored_word_count"] == 0


def test_custom_words(profanity_classifier):
    """Test custom word functionality."""
    # Add custom words
    custom_words = {"custom1", "custom2"}
    profanity_classifier.add_custom_words(custom_words)

    # Test text with custom profanity
    result = profanity_classifier.classify("This text contains custom1.")
    assert result.label == "profane"
    assert result.confidence > 0
    assert result.metadata["contains_profanity"]
    assert "*******" in result.metadata["censored_text"]
    assert result.metadata["censored_word_count"] > 0

    # Test adding more custom words
    more_words = {"custom3", "custom4"}
    profanity_classifier.add_custom_words(more_words)
    assert custom_words | more_words <= profanity_classifier.custom_words


def test_batch_classification(profanity_classifier):
    """Test batch text classification."""
    texts = [
        "This is clean text.",
        "This text is bad.",
        "This is inappropriate content.",
        "",  # Empty text
        "!@#$%^&*()",  # Special characters
        "   \n\t   ",  # Whitespace
    ]

    results = profanity_classifier.batch_classify(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Verify each result
    assert results[0].label == "clean"
    assert results[1].label == "profane"
    assert results[2].label == "profane"
    assert results[3].label == "clean"  # Empty text
    assert results[4].label == "clean"  # Special chars
    assert results[5].label == "clean"  # Whitespace

    for result in results:
        assert isinstance(result, ClassificationResult)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.metadata["contains_profanity"], bool)
        assert isinstance(result.metadata["censored_text"], str)
        assert isinstance(result.metadata["censored_word_count"], int)


def test_edge_cases(profanity_classifier):
    """Test edge cases."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "numbers_only": "123 456 789",
        "mixed_content": "bad123!@#",
        "repeated_word": "bad " * 10,
        "very_long": "a" * 10000,
        "single_char": "a",
        "newlines": "Line 1\nLine 2\nLine 3",
    }

    for case_name, text in edge_cases.items():
        result = profanity_classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert result.label in profanity_classifier.labels
        assert 0 <= result.confidence <= 1
        assert isinstance(result.metadata, dict)
        assert isinstance(result.metadata["contains_profanity"], bool)
        assert isinstance(result.metadata["censored_text"], str)
        assert isinstance(result.metadata["censored_word_count"], int)


def test_error_handling(profanity_classifier):
    """Test error handling."""
    invalid_inputs = [None, 123, [], {}]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            profanity_classifier.classify(invalid_input)

        with pytest.raises(Exception):
            profanity_classifier.batch_classify([invalid_input])


def test_consistent_results(profanity_classifier):
    """Test consistency of classification results."""
    test_texts = {
        "clean": "This is a clean text that should give consistent results.",
        "profane": "This text contains bad and inappropriate content.",
        "mixed": "Some clean text with bad words.",
        "empty": "",
    }

    for case_name, text in test_texts.items():
        # Test single classification consistency
        results = [profanity_classifier.classify(text) for _ in range(3)]
        first_result = results[0]
        for result in results[1:]:
            assert result.label == first_result.label
            assert result.confidence == first_result.confidence
            assert result.metadata == first_result.metadata

        # Test batch classification consistency
        batch_results = [profanity_classifier.batch_classify([text]) for _ in range(3)]
        first_batch = batch_results[0]
        for batch in batch_results[1:]:
            assert len(batch) == len(first_batch)
            for r1, r2 in zip(batch, first_batch):
                assert r1.label == r2.label
                assert r1.confidence == r2.confidence
                assert r1.metadata == r2.metadata


def test_censoring(profanity_classifier):
    """Test text censoring functionality."""
    test_cases = [
        ("This is a bad word.", "This is a *** word.", 1),
        ("Multiple bad and inappropriate words.", "Multiple *** and ************ words.", 2),
        ("Clean text stays unchanged.", "Clean text stays unchanged.", 0),
        ("bad BAD Bad bAd", "*** *** *** ***", 4),
    ]

    for text, expected_censored, expected_count in test_cases:
        result = profanity_classifier.classify(text)
        assert result.metadata["censored_text"] == expected_censored
        assert result.metadata["censored_word_count"] == expected_count
