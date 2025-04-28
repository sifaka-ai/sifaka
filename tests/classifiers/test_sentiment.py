"""
Tests for the SentimentClassifier class.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.classifiers.base import ClassificationResult, ClassifierConfig
from sifaka.classifiers.sentiment import (
    SentimentClassifier,
    SentimentThresholds,
)


class CustomAnalyzer:
    """A mock implementation of VADER analyzer that works with comparisons."""

    def __init__(self, response_mappings=None):
        """Initialize with optional mappings for specific inputs."""
        self.response_mappings = response_mappings or {}
        self.call_count = 0
        self.last_text = None

    def polarity_scores(self, text):
        """Return sentiment scores for the given text."""
        self.call_count += 1
        self.last_text = text

        # Handle different text inputs with predefined responses
        if text in self.response_mappings:
            return self.response_mappings[text]

        # Default responses based on text content
        if "positive" in text.lower():
            return {"pos": 0.8, "neg": 0.0, "neu": 0.2, "compound": 0.8}
        elif "negative" in text.lower():
            return {"pos": 0.0, "neg": 0.8, "neu": 0.2, "compound": -0.8}
        else:
            return {"pos": 0.1, "neg": 0.1, "neu": 0.8, "compound": 0.0}


class TestSentimentClassifier:
    """Tests for the SentimentClassifier class."""

    def test_initialization(self):
        """Test basic initialization of SentimentClassifier."""
        classifier = SentimentClassifier(
            name="test_sentiment", description="Test sentiment classifier"
        )

        assert classifier.name == "test_sentiment"
        assert classifier.description == "Test sentiment classifier"
        assert classifier.config.labels == ["positive", "neutral", "negative", "unknown"]
        assert classifier.positive_threshold == 0.05
        assert classifier.negative_threshold == -0.05

    def test_initialization_with_thresholds(self):
        """Test initialization with custom thresholds."""
        thresholds = SentimentThresholds(positive=0.2, negative=-0.2)
        classifier = SentimentClassifier(thresholds=thresholds)

        assert classifier.positive_threshold == 0.2
        assert classifier.negative_threshold == -0.2

    def test_initialization_with_config(self):
        """Test initialization with explicit config."""
        config = ClassifierConfig(
            labels=["positive", "neutral", "negative", "unknown"],
            params={"positive_threshold": 0.3, "negative_threshold": -0.3},
        )
        classifier = SentimentClassifier(config=config)

        assert classifier.positive_threshold == 0.3
        assert classifier.negative_threshold == -0.3

    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError):
            SentimentThresholds(positive=-0.1, negative=0.1)  # Invalid order

        with pytest.raises(ValueError):
            SentimentThresholds(positive=1.5, negative=-0.5)  # Out of range

    @patch("importlib.import_module")
    def test_load_vader_import_error(self, mock_import):
        """Test handling of VADER import error."""
        mock_import.side_effect = ImportError("No module named 'vaderSentiment'")

        classifier = SentimentClassifier()
        with pytest.raises(ImportError) as exc_info:
            classifier.warm_up()

        assert "VADER package is required" in str(exc_info.value)

    def test_get_sentiment_label(self):
        """Test that _get_sentiment_label correctly classifies based on score."""
        classifier = SentimentClassifier()

        # Test with default thresholds (positive=0.05, negative=-0.05)
        assert classifier._get_sentiment_label(0.1) == "positive"
        assert classifier._get_sentiment_label(0.05) == "positive"
        assert classifier._get_sentiment_label(0.04) == "neutral"
        assert classifier._get_sentiment_label(0.0) == "neutral"
        assert classifier._get_sentiment_label(-0.04) == "neutral"
        assert classifier._get_sentiment_label(-0.05) == "negative"
        assert classifier._get_sentiment_label(-0.1) == "negative"

        # Test with custom thresholds
        classifier = SentimentClassifier(
            thresholds=SentimentThresholds(positive=0.2, negative=-0.2)
        )

        assert classifier._get_sentiment_label(0.3) == "positive"
        assert classifier._get_sentiment_label(0.2) == "positive"
        assert classifier._get_sentiment_label(0.1) == "neutral"
        assert classifier._get_sentiment_label(0.0) == "neutral"
        assert classifier._get_sentiment_label(-0.1) == "neutral"
        assert classifier._get_sentiment_label(-0.2) == "negative"
        assert classifier._get_sentiment_label(-0.3) == "negative"

    def test_classify_empty_text(self):
        """Test classification of empty text."""
        classifier = SentimentClassifier()

        result = classifier.classify("")
        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "empty_input"

        result = classifier.classify("   ")
        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "empty_input"

    @patch("sifaka.classifiers.sentiment.SentimentClassifier._classify_impl")
    def test_batch_classify(self, mock_classify_impl):
        """Test batch classification delegates to classify for each text."""

        # Set up mock to return predefined results
        def side_effect(text):
            if "positive" in text:
                return ClassificationResult(
                    label="positive", confidence=0.8, metadata={"compound_score": 0.8}
                )
            elif "negative" in text:
                return ClassificationResult(
                    label="negative", confidence=0.8, metadata={"compound_score": -0.8}
                )
            else:
                return ClassificationResult(
                    label="neutral", confidence=0.0, metadata={"compound_score": 0.0}
                )

        mock_classify_impl.side_effect = side_effect

        classifier = SentimentClassifier()

        texts = [
            "This is a positive message",
            "This is a negative message",
            "This is a neutral message",
        ]

        results = classifier.batch_classify(texts)
        assert len(results) == 3

        assert results[0].label == "positive"
        assert results[1].label == "negative"
        assert results[2].label == "neutral"

        # Verify _classify_impl was called for each non-empty text
        assert mock_classify_impl.call_count == 3
