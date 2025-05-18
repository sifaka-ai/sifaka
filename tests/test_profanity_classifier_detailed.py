"""
Detailed tests for the profanity classifier.

This module contains more comprehensive tests for the profanity classifier
to improve test coverage.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Set, Any

from sifaka.classifiers import ClassificationResult
from sifaka.classifiers.profanity import ProfanityClassifier


# Create a mock better_profanity module for testing
class MockProfanity:
    """Mock for the better_profanity.profanity module."""

    def __init__(self):
        self.custom_words = set()
        self.censor_char = "*"
        self.profane_words = {"badword", "swear", "profane", "offensive"}

    def load_censor_words(self):
        """Mock for load_censor_words method."""
        pass

    def add_censor_words(self, custom_words: List[str]):
        """Mock for add_censor_words method."""
        self.custom_words.update(custom_words)
        self.profane_words.update(custom_words)

    def set_censor_char(self, censor_char: str):
        """Mock for set_censor_char method."""
        self.censor_char = censor_char

    def contains_profanity(self, text: str) -> bool:
        """Mock for contains_profanity method."""
        # Check if any word in text is in profane_words
        text_lower = text.lower()
        # Check if any profane word is in the text (not just exact word matches)
        return any(profane_word in text_lower for profane_word in self.profane_words)

    def censor(self, text: str) -> str:
        """Mock for censor method."""
        # This is a simplified implementation that doesn't match exactly how better_profanity works
        # but is sufficient for testing
        censored_text = text
        for word in self.profane_words:
            if word in text.lower():
                # Find the word in the text (case-insensitive)
                start_idx = text.lower().find(word)
                if start_idx != -1:
                    # Replace the word with censored characters
                    end_idx = start_idx + len(word)
                    censored_text = (
                        censored_text[:start_idx]
                        + (self.censor_char * len(word))
                        + censored_text[end_idx:]
                    )

        return censored_text


# Create a mock better_profanity module
class MockBetterProfanity:
    """Mock for the better_profanity module."""

    def __init__(self):
        self.profanity = MockProfanity()


class TestProfanityClassifierDetailed:
    """Detailed tests for the ProfanityClassifier."""

    def test_init_with_custom_parameters(self) -> None:
        """Test initializing with custom parameters."""
        custom_words = ["custom_bad_word", "another_bad_word"]
        classifier = ProfanityClassifier(
            custom_words=custom_words,
            censor_char="#",
            name="custom_profanity",
            description="Custom profanity classifier",
        )

        assert classifier.name == "custom_profanity"
        assert classifier.description == "Custom profanity classifier"
        # Access private attributes for testing
        assert classifier._custom_words == custom_words
        assert classifier._censor_char == "#"
        assert classifier._profanity is None
        assert classifier._initialized is False

    def test_load_profanity_error(self) -> None:
        """Test error handling when better_profanity is not available."""
        with patch(
            "importlib.import_module", side_effect=ImportError("No module named 'better_profanity'")
        ):
            classifier = ProfanityClassifier()

            with pytest.raises(ImportError) as excinfo:
                classifier._load_profanity()

            assert "better_profanity package is required" in str(excinfo.value)
            assert "pip install better_profanity" in str(excinfo.value)

    def test_load_profanity_runtime_error(self) -> None:
        """Test error handling when better_profanity initialization fails."""
        # Create a mock that raises a runtime error
        mock_profanity = MagicMock()
        mock_profanity.profanity.load_censor_words.side_effect = RuntimeError(
            "Initialization failed"
        )

        with patch("importlib.import_module", return_value=mock_profanity):
            classifier = ProfanityClassifier()

            with pytest.raises(RuntimeError) as excinfo:
                classifier._load_profanity()

            assert "Failed to initialize better_profanity" in str(excinfo.value)
            assert "Initialization failed" in str(excinfo.value)

    def test_initialize(self) -> None:
        """Test initialization of the classifier."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier(custom_words=["custom_bad_word"])

            assert classifier._initialized is False
            assert classifier._profanity is None

            # Initialize the classifier
            classifier._initialize()

            assert classifier._initialized is True
            assert classifier._profanity is not None

            # Check that custom words were added
            assert "custom_bad_word" in classifier._profanity.custom_words

            # Calling initialize again should not change anything
            profanity = classifier._profanity
            classifier._initialize()
            assert classifier._profanity == profanity

    def test_get_profane_words(self) -> None:
        """Test getting profane words from text."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier()

            # Initialize the classifier
            classifier._initialize()

            # Test with text containing profane words
            text = "This text contains a badword and another swear word."
            profane_words = classifier._get_profane_words(text)

            assert len(profane_words) == 2
            assert "badword" in profane_words
            assert "swear" in profane_words

            # Test with clean text
            text = "This text is clean and appropriate."
            profane_words = classifier._get_profane_words(text)

            assert len(profane_words) == 0

    def test_classify_empty_text(self) -> None:
        """Test classifying empty text."""
        classifier = ProfanityClassifier()

        result = classifier.classify("")

        assert result.label == "clean"
        assert result.confidence == 1.0
        assert result.metadata["input_length"] == 0
        assert result.metadata["reason"] == "empty_text"
        assert result.metadata["censored_text"] == ""
        assert result.metadata["is_censored"] is False
        assert result.metadata["profane_word_count"] == 0

    def test_classify_profane_text(self) -> None:
        """Test classifying text with profanity."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier()

            # Test with text containing profane words
            result = classifier.classify("This text contains a badword and another swear word.")

            assert result.label == "profane"
            assert result.confidence > 0.5
            assert result.metadata["input_length"] == len(
                "This text contains a badword and another swear word."
            )
            assert result.metadata["is_censored"] is True
            assert result.metadata["profane_word_count"] == 2
            assert "badword" in result.metadata["profane_words"]
            assert "swear" in result.metadata["profane_words"]
            assert "profanity_score" in result.metadata

            # Check that the censored text has asterisks
            assert (
                "This text contains a ******* and another ***** word."
                == result.metadata["censored_text"]
            )

    def test_classify_clean_text(self) -> None:
        """Test classifying clean text."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier()

            # Test with clean text
            result = classifier.classify("This text is clean and appropriate.")

            assert result.label == "clean"
            assert result.confidence == 1.0
            assert result.metadata["input_length"] == len("This text is clean and appropriate.")
            assert result.metadata["is_censored"] is False
            assert result.metadata["profane_word_count"] == 0
            assert result.metadata["profane_words"] == []
            assert result.metadata["profanity_score"] == 0.0

            # Check that the censored text is the same as the original
            assert result.metadata["censored_text"] == "This text is clean and appropriate."

    def test_classify_with_custom_words(self) -> None:
        """Test classifying text with custom profane words."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier(custom_words=["custom_bad_word"])

            # Test with text containing custom profane words
            result = classifier.classify("This text contains a custom_bad_word.")

            assert result.label == "profane"
            assert result.confidence > 0.5
            assert result.metadata["is_censored"] is True
            assert result.metadata["profane_word_count"] == 1
            # The actual implementation splits the text by words, so it might include punctuation
            assert any("custom_bad_word" in word for word in result.metadata["profane_words"])

            # Check that the censored text has asterisks for the custom word
            assert "This text contains a ***************." == result.metadata["censored_text"]

    def test_classify_with_custom_censor_char(self) -> None:
        """Test classifying text with a custom censor character."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier(censor_char="#")

            # Test with text containing profane words
            result = classifier.classify("This text contains a badword.")

            # Check that the censored text has the custom censor character
            assert "This text contains a #######." == result.metadata["censored_text"]

    def test_classify_error_handling(self) -> None:
        """Test error handling during classification."""
        # Create a mock that raises an error during contains_profanity
        mock_profanity = MagicMock()
        mock_profanity.contains_profanity.side_effect = RuntimeError("Classification failed")

        classifier = ProfanityClassifier()
        classifier._profanity = mock_profanity
        classifier._initialized = True

        # Test with text that will cause an error
        result = classifier.classify("This text will cause an error.")

        assert result.label == "clean"
        assert result.confidence == 0.5
        assert "error" in result.metadata
        assert "Classification failed" in result.metadata["error"]
        assert result.metadata["reason"] == "classification_error"

    def test_batch_classify(self) -> None:
        """Test batch classification of multiple texts."""
        # Create a mock better_profanity module
        mock_better_profanity = MockBetterProfanity()

        with patch("importlib.import_module", return_value=mock_better_profanity):
            classifier = ProfanityClassifier()

            texts = [
                "This text contains a badword.",
                "This text is clean.",
                "",
                "This text has a swear word.",
            ]

            results = classifier.batch_classify(texts)

            assert len(results) == 4
            assert results[0].label == "profane"
            assert results[1].label == "clean"
            assert results[2].label == "clean"  # Empty text
            assert results[2].metadata["reason"] == "empty_text"
            assert results[3].label == "profane"
