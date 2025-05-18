"""
Detailed tests for the sentiment classifier.

This module contains more comprehensive tests for the sentiment classifier
to improve test coverage.
"""

import pytest
from typing import List, Dict, Any

from sifaka.classifiers import ClassificationResult
from sifaka.classifiers.sentiment import SentimentClassifier


class TestSentimentClassifierDetailed:
    """Detailed tests for the SentimentClassifier."""

    def test_init_with_custom_word_lists(self) -> None:
        """Test initializing with custom positive and negative word lists."""
        positive_words = ["happy", "joy", "excellent"]
        negative_words = ["sad", "bad", "terrible"]
        
        classifier = SentimentClassifier(
            positive_words=positive_words,
            negative_words=negative_words,
            name="custom_sentiment",
            description="Custom sentiment classifier"
        )
        
        assert classifier.name == "custom_sentiment"
        assert classifier.description == "Custom sentiment classifier"
        # Access private attributes for testing
        assert classifier._positive_words == positive_words
        assert classifier._negative_words == negative_words

    def test_classify_empty_text(self) -> None:
        """Test classifying empty text."""
        classifier = SentimentClassifier()
        result = classifier.classify("")
        
        assert result.label == "neutral"
        assert result.confidence == 1.0
        assert result.metadata["input_length"] == 0
        assert result.metadata["reason"] == "empty_text"

    def test_classify_neutral_text(self) -> None:
        """Test classifying neutral text with no sentiment words."""
        classifier = SentimentClassifier()
        text = "This is a simple text without sentiment words."
        result = classifier.classify(text)
        
        assert result.label == "neutral"
        assert result.confidence >= 0.5  # Confidence for neutral should be moderate
        assert result.metadata["input_length"] == len(text)
        assert result.metadata["positive_count"] == 0
        assert result.metadata["negative_count"] == 0
        assert result.metadata["reason"] == "no_sentiment_words"

    def test_classify_strongly_positive(self) -> None:
        """Test classifying strongly positive text."""
        classifier = SentimentClassifier()
        text = "This is an excellent, amazing, and wonderful product. I love it and am very happy with my purchase."
        result = classifier.classify(text)
        
        assert result.label == "positive"
        assert result.confidence > 0.8  # High confidence for strongly positive
        assert result.metadata["input_length"] == len(text)
        assert result.metadata["positive_count"] > 0
        assert result.metadata["negative_count"] == 0
        assert result.metadata["sentiment_score"] > 0

    def test_classify_strongly_negative(self) -> None:
        """Test classifying strongly negative text."""
        classifier = SentimentClassifier()
        text = "This is a terrible, horrible, and awful product. I hate it and am very disappointed with my purchase."
        result = classifier.classify(text)
        
        assert result.label == "negative"
        assert result.confidence > 0.8  # High confidence for strongly negative
        assert result.metadata["input_length"] == len(text)
        assert result.metadata["positive_count"] == 0
        assert result.metadata["negative_count"] > 0
        assert result.metadata["sentiment_score"] < 0

    def test_classify_mixed_sentiment(self) -> None:
        """Test classifying text with mixed sentiment."""
        classifier = SentimentClassifier()
        text = "The product has some good features but also some bad aspects. It's both impressive and disappointing."
        result = classifier.classify(text)
        
        # The result could be positive, negative, or neutral depending on the exact words
        # We're just checking that the metadata is correct
        assert result.metadata["input_length"] == len(text)
        assert result.metadata["positive_count"] > 0
        assert result.metadata["negative_count"] > 0
        assert "sentiment_score" in result.metadata
        
        # Verify that the sentiment score matches the label
        if result.label == "positive":
            assert result.metadata["sentiment_score"] > 0
        elif result.label == "negative":
            assert result.metadata["sentiment_score"] < 0
        else:  # neutral
            assert abs(result.metadata["sentiment_score"]) <= 0.2

    def test_batch_classify(self) -> None:
        """Test batch classification of multiple texts."""
        classifier = SentimentClassifier()
        texts = [
            "This is excellent.",
            "This is terrible.",
            "",
            "This is a neutral statement."
        ]
        
        results = classifier.batch_classify(texts)
        
        assert len(results) == 4
        assert results[0].label == "positive"
        assert results[1].label == "negative"
        assert results[2].label == "neutral"  # Empty text
        assert results[2].metadata["reason"] == "empty_text"
        # The fourth result should be neutral due to lack of sentiment words
        assert results[3].label == "neutral"
        assert results[3].metadata["reason"] == "no_sentiment_words"

    def test_case_insensitivity(self) -> None:
        """Test that the classifier is case-insensitive."""
        classifier = SentimentClassifier()
        
        # Test with lowercase
        lower_result = classifier.classify("this is good and excellent")
        
        # Test with mixed case
        mixed_result = classifier.classify("This is GOOD and ExCeLlEnT")
        
        # Both should be positive with similar confidence
        assert lower_result.label == "positive"
        assert mixed_result.label == "positive"
        assert abs(lower_result.confidence - mixed_result.confidence) < 0.01
        assert lower_result.metadata["positive_count"] == mixed_result.metadata["positive_count"]

    def test_word_boundary_detection(self) -> None:
        """Test that the classifier correctly detects word boundaries."""
        classifier = SentimentClassifier()
        
        # "good" is a positive word, but "goods" should not match
        result1 = classifier.classify("These are goods.")
        assert result1.metadata["positive_count"] == 0
        
        # "good" as a standalone word should match
        result2 = classifier.classify("This is good.")
        assert result2.metadata["positive_count"] == 1
        
        # "good" with punctuation should match
        result3 = classifier.classify("Good, better, best!")
        assert result3.metadata["positive_count"] >= 1
