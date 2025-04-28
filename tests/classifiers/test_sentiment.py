"""Tests for the sentiment classifier."""

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from sifaka.classifiers.base import ClassificationResult
from sifaka.classifiers.sentiment import SentimentClassifier


class MockVaderAnalyzer:
    """Mock VADER sentiment analyzer."""

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """Return mock polarity scores."""
        text_lower = text.lower()

        # Default neutral scores
        scores = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

        # Positive indicators
        positive_words = {"great", "excellent", "amazing", "good", "happy"}
        if any(word in text_lower for word in positive_words):
            scores.update({"pos": 0.6, "neg": 0.0, "neu": 0.4, "compound": 0.8})

        # Negative indicators
        negative_words = {"bad", "terrible", "awful", "sad", "angry"}
        if any(word in text_lower for word in negative_words):
            scores.update({"pos": 0.0, "neg": 0.6, "neu": 0.4, "compound": -0.8})

        # Mixed sentiment
        if any(word in text_lower for word in positive_words) and any(
            word in text_lower for word in negative_words
        ):
            scores.update({"pos": 0.4, "neg": 0.4, "neu": 0.2, "compound": 0.0})

        return scores

@pytest.fixture
def mock_vader():
    """Create a mock VADER analyzer instance."""
    return MockVaderAnalyzer()

@pytest.fixture
def sentiment_classifier(mock_vader):
    """Create a SentimentClassifier instance with mocked VADER."""
    with patch("importlib.import_module") as mock_import:
        mock_vader_module = MagicMock()
        mock_vader_module.SentimentIntensityAnalyzer = MagicMock(return_value=mock_vader)
        mock_import.return_value = mock_vader_module

        classifier = SentimentClassifier()
        classifier.warm_up()
        return classifier

def test_initialization():
    """Test SentimentClassifier initialization."""
    # Test basic initialization
    classifier = SentimentClassifier()
    assert classifier.name == "sentiment_classifier"
    assert classifier.description == "Analyzes text sentiment using VADER"
    assert classifier.threshold_pos == 0.05
    assert classifier.threshold_neg == -0.05
    assert classifier.labels == ["positive", "neutral", "negative"]
    assert classifier.cost == 1

    # Test custom initialization
    custom_classifier = SentimentClassifier(
        name="custom",
        description="custom classifier",
        threshold_pos=0.1,
        threshold_neg=-0.1,
        config={"param": "value"},
    )
    assert custom_classifier.name == "custom"
    assert custom_classifier.description == "custom classifier"
    assert custom_classifier.threshold_pos == 0.1
    assert custom_classifier.threshold_neg == -0.1
    assert custom_classifier.config == {"param": "value"}

def test_warm_up(sentiment_classifier, mock_vader):
    """Test warm_up functionality."""
    assert sentiment_classifier._analyzer == mock_vader

    # Test error handling
    with patch("importlib.import_module", side_effect=ImportError()):
        classifier = SentimentClassifier()
        with pytest.raises(ImportError):
            classifier.warm_up()

    with patch("importlib.import_module", side_effect=RuntimeError()):
        classifier = SentimentClassifier()
        with pytest.raises(RuntimeError):
            classifier.warm_up()

def test_sentiment_label_mapping(sentiment_classifier):
    """Test sentiment label mapping."""
    test_cases = [
        (0.1, "positive"),  # Above positive threshold
        (-0.1, "negative"),  # Below negative threshold
        (0.0, "neutral"),  # Between thresholds
        (0.04, "neutral"),  # Just below positive threshold
        (-0.04, "neutral"),  # Just above negative threshold
        (1.0, "positive"),  # Maximum positive
        (-1.0, "negative"),  # Maximum negative
    ]

    for score, expected_label in test_cases:
        assert sentiment_classifier._get_sentiment_label(score) == expected_label

def test_classification(sentiment_classifier):
    """Test text classification."""
    # Test positive text
    result = sentiment_classifier.classify("This is a great and excellent text!")
    assert isinstance(result, ClassificationResult)
    assert result.label == "positive"
    assert result.confidence > 0.5
    assert result.metadata["compound_score"] > 0
    assert result.metadata["pos_score"] > result.metadata["neg_score"]

    # Test negative text
    result = sentiment_classifier.classify("This is a terrible and awful experience.")
    assert result.label == "negative"
    assert result.confidence > 0.5
    assert result.metadata["compound_score"] < 0
    assert result.metadata["neg_score"] > result.metadata["pos_score"]

    # Test neutral text
    result = sentiment_classifier.classify("This is a regular text.")
    assert result.label == "neutral"
    assert result.metadata["compound_score"] == 0
    assert result.metadata["neu_score"] > result.metadata["pos_score"]
    assert result.metadata["neu_score"] > result.metadata["neg_score"]

    # Test mixed sentiment
    result = sentiment_classifier.classify("Good things happened but it was also bad.")
    assert result.label == "neutral"
    assert abs(result.metadata["compound_score"]) < 0.1
    assert result.metadata["pos_score"] > 0
    assert result.metadata["neg_score"] > 0

    # Test empty text
    result = sentiment_classifier.classify("")
    assert result.label == "neutral"
    assert result.confidence == 0.0
    assert result.metadata["compound_score"] == 0.0

    # Test whitespace text
    result = sentiment_classifier.classify("   \n\t   ")
    assert result.label == "neutral"
    assert result.confidence == 0.0
    assert result.metadata["compound_score"] == 0.0

def test_batch_classification(sentiment_classifier):
    """Test batch text classification."""
    texts = [
        "This is great!",  # Positive
        "This is terrible.",  # Negative
        "This is a regular text.",  # Neutral
        "Good and bad things happened.",  # Mixed
        "",  # Empty
        "   \n\t   ",  # Whitespace
    ]

    results = sentiment_classifier.batch_classify(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)

    # Verify each result
    assert results[0].label == "positive"
    assert results[1].label == "negative"
    assert results[2].label == "neutral"
    assert results[3].label == "neutral"  # Mixed sentiment
    assert results[4].label == "neutral"  # Empty text
    assert results[5].label == "neutral"  # Whitespace

    for result in results:
        assert isinstance(result, ClassificationResult)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.metadata["compound_score"], float)
        assert isinstance(result.metadata["pos_score"], float)
        assert isinstance(result.metadata["neg_score"], float)
        assert isinstance(result.metadata["neu_score"], float)

def test_edge_cases(sentiment_classifier):
    """Test edge cases."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "numbers_only": "123 456 789",
        "mixed_content": "great123!@#",
        "repeated_word": "great " * 10,
        "very_long": "a" * 10000,
        "single_char": "a",
        "newlines": "Line 1\nLine 2\nLine 3",
        "emojis": "😀 😃 😄 😁",
    }

    for case_name, text in edge_cases.items():
        result = sentiment_classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert result.label in sentiment_classifier.labels
        assert 0 <= result.confidence <= 1
        assert isinstance(result.metadata, dict)
        assert isinstance(result.metadata["compound_score"], float)
        assert isinstance(result.metadata["pos_score"], float)
        assert isinstance(result.metadata["neg_score"], float)
        assert isinstance(result.metadata["neu_score"], float)

def test_error_handling(sentiment_classifier):
    """Test error handling."""
    invalid_inputs = [None, 123, [], {}]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            sentiment_classifier.classify(invalid_input)

        with pytest.raises(Exception):
            sentiment_classifier.batch_classify([invalid_input])

def test_consistent_results(sentiment_classifier):
    """Test consistency of classification results."""
    test_texts = {
        "positive": "This is a great and excellent text!",
        "negative": "This is a terrible and awful experience.",
        "neutral": "This is a regular text.",
        "mixed": "Good things happened but it was also bad.",
    }

    for sentiment, text in test_texts.items():
        # Test single classification consistency
        results = [sentiment_classifier.classify(text) for _ in range(3)]
        first_result = results[0]
        for result in results[1:]:
            assert result.label == first_result.label
            assert result.confidence == first_result.confidence
            assert result.metadata == first_result.metadata

        # Test batch classification consistency
        batch_results = [sentiment_classifier.batch_classify([text]) for _ in range(3)]
        first_batch = batch_results[0]
        for batch in batch_results[1:]:
            assert len(batch) == len(first_batch)
            for r1, r2 in zip(batch, first_batch):
                assert r1.label == r2.label
                assert r1.confidence == r2.confidence
                assert r1.metadata == r2.metadata

def test_threshold_sensitivity():
    """Test sensitivity to different threshold values."""
    text = "This is a somewhat good text."
    thresholds = [
        (0.0, -0.0),  # Zero thresholds
        (0.3, -0.3),  # Wide thresholds
        (0.01, -0.01),  # Narrow thresholds
        (0.9, -0.9),  # Extreme thresholds
    ]

    for pos, neg in thresholds:
        classifier = SentimentClassifier(threshold_pos=pos, threshold_neg=neg)
        classifier._analyzer = MockVaderAnalyzer()

        result = classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert result.label in classifier.labels

        # Verify threshold logic
        compound_score = result.metadata["compound_score"]
        if compound_score >= pos:
            assert result.label == "positive"
        elif compound_score <= neg:
            assert result.label == "negative"
        else:
            assert result.label == "neutral"
