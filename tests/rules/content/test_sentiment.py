"""
Tests for sentiment rule and validator.
"""

from unittest.mock import patch, MagicMock

import pytest

from sifaka.classifiers.base import ClassificationResult
from sifaka.rules.content.sentiment import (
    create_sentiment_rule,
    create_sentiment_validator,
    SentimentConfig,
    SentimentRule,
    SentimentValidator,
)


class MockSentimentClassifier:
    """Mock sentiment classifier for testing."""

    def classify(self, text):
        """Return positive for positive text, negative for negative text."""
        if any(term in text.lower() for term in ["good", "great", "excellent"]):
            return ClassificationResult(label="positive", confidence=0.9, metadata={})
        elif any(term in text.lower() for term in ["bad", "terrible", "awful"]):
            return ClassificationResult(label="negative", confidence=0.9, metadata={})
        return ClassificationResult(label="neutral", confidence=0.7, metadata={})


@patch("sifaka.rules.content.sentiment.SimpleSentimentClassifier", MockSentimentClassifier)
def test_sentiment_validator():
    """Test that SentimentValidator properly validates text sentiment."""
    # Create a validator using the factory function
    validator = create_sentiment_validator(threshold=0.6, valid_labels=["positive", "neutral"])

    # Test with positive text
    result = validator.validate("This is a great test.")
    assert result.passed is True
    assert "Sentiment 'positive'" in result.message
    assert "meets criteria" in result.message
    assert result.metadata["sentiment"] == "positive"
    assert result.metadata["confidence"] == 0.9
    assert result.metadata["threshold"] == 0.6
    assert result.metadata["valid_labels"] == ["positive", "neutral"]

    # Test with negative text (should fail)
    result = validator.validate("This is a terrible test.")
    assert result.passed is False
    assert "Sentiment 'negative'" in result.message
    assert "does not meet criteria" in result.message
    assert result.metadata["sentiment"] == "negative"

    # Test with neutral text
    result = validator.validate("This is a test.")
    assert result.passed is True
    assert "Sentiment 'neutral'" in result.message
    assert "meets criteria" in result.message
    assert result.metadata["sentiment"] == "neutral"


@patch("sifaka.rules.content.sentiment.SimpleSentimentClassifier", MockSentimentClassifier)
def test_sentiment_rule_delegation():
    """Test that SentimentRule properly delegates to its validator."""
    # Create a rule using the factory function
    rule = create_sentiment_rule(threshold=0.6, valid_labels=["positive", "neutral"])

    # Test with positive text
    result = rule.validate("This is a great test.")
    assert result.passed is True
    assert "Sentiment 'positive'" in result.message
    assert "meets criteria" in result.message
    assert "rule_id" in result.metadata
    assert result.metadata["rule_id"] == "sentiment_rule"

    # Test with negative text (should fail)
    result = rule.validate("This is a terrible test.")
    assert result.passed is False
    assert "Sentiment 'negative'" in result.message
    assert "does not meet criteria" in result.message
    assert "rule_id" in result.metadata


@patch("sifaka.rules.content.sentiment.SimpleSentimentClassifier", MockSentimentClassifier)
def test_empty_text_handling():
    """Test that empty text is handled properly."""
    # Create a rule
    rule = create_sentiment_rule()

    # Test with empty string
    result = rule.validate("")
    assert result.passed is True
    assert result.message == "Empty text validation skipped"
    assert result.metadata.get("reason") == "empty_input"

    # Test with whitespace-only string
    result = rule.validate("   \n   ")
    assert result.passed is True
    assert result.message == "Empty text validation skipped"
    assert result.metadata.get("reason") == "empty_input"


@patch("sifaka.rules.content.sentiment.SimpleSentimentClassifier", MockSentimentClassifier)
def test_custom_config():
    """Test that custom configuration works properly."""
    # Create a rule with custom configuration
    rule = create_sentiment_rule(
        name="custom_sentiment_rule",
        description="Custom sentiment validation",
        threshold=0.8,
        valid_labels=["positive"],
        priority=2,
        cost=2.0,
        cache_size=200,
    )

    # Test with positive text
    result = rule.validate("This is a great test.")
    assert result.passed is True
    assert "Sentiment 'positive'" in result.message
    assert "meets criteria" in result.message
    assert result.metadata["rule_id"] == "custom_sentiment_rule"
    assert result.metadata["threshold"] == 0.8
    assert result.metadata["valid_labels"] == ["positive"]

    # Test with neutral text (should fail with custom config)
    result = rule.validate("This is a test.")
    assert result.passed is False
    assert "Sentiment 'neutral'" in result.message
    assert "does not meet criteria" in result.message
