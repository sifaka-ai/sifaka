"""
Tests for the sentiment content validation rules and validators.
"""

import pytest
from sifaka.rules.content.sentiment import (
    SentimentConfig,
    SimpleSentimentClassifier,
    SentimentAnalyzer,
    SentimentValidator,
    create_sentiment_validator,
)
from sifaka.rules.base import RuleConfig


def test_sentiment_config():
    """Test that SentimentConfig works correctly."""
    # Test with default parameters
    config = SentimentConfig()
    assert config.threshold == 0.6
    assert config.valid_labels == ["positive", "neutral"]
    assert config.cache_size == 100
    assert config.priority == 1
    assert config.cost == 1.0

    # Test with custom parameters
    custom_config = SentimentConfig(
        threshold=0.8,
        valid_labels=["positive"],
        cache_size=200,
        priority=2,
        cost=0.5,
    )
    assert custom_config.threshold == 0.8
    assert custom_config.valid_labels == ["positive"]
    assert custom_config.cache_size == 200
    assert custom_config.priority == 2
    assert custom_config.cost == 0.5


def test_simple_sentiment_classifier():
    """Test that SimpleSentimentClassifier works correctly."""
    classifier = SimpleSentimentClassifier()

    # Test positive sentiment
    result = classifier.classify("This is good and great and excellent!")
    assert result.label == "positive"
    assert result.confidence == 0.8

    # Test negative sentiment
    result = classifier.classify("This is bad and terrible!")
    assert result.label == "negative"
    assert result.confidence == 0.8

    # Test neutral sentiment
    result = classifier.classify("This is a simple text without sentiment.")
    assert result.label == "neutral"
    assert result.confidence == 0.6


def test_sentiment_analyzer():
    """Test that SentimentAnalyzer works correctly."""
    # Create config
    config = SentimentConfig(
        threshold=0.7,
        valid_labels=["positive", "neutral"],
    )

    # Create analyzer
    analyzer = SentimentAnalyzer(config)

    # Test positive sentiment (should pass)
    result = analyzer.analyze("This is good and great!")
    assert result.passed is True
    assert "sentiment" in result.metadata
    assert result.metadata["sentiment"] == "positive"

    # Test negative sentiment (should fail)
    result = analyzer.analyze("This is bad and terrible!")
    assert result.passed is False
    assert "sentiment" in result.metadata
    assert result.metadata["sentiment"] == "negative"

    # Test neutral sentiment (should fail because confidence is below threshold)
    result = analyzer.analyze("This is a simple text.")
    assert result.passed is False
    assert "sentiment" in result.metadata
    assert result.metadata["sentiment"] == "neutral"
    assert result.metadata["confidence"] == 0.6
    assert result.metadata["threshold"] == 0.7  # Confidence is below threshold


def test_sentiment_validator():
    """Test that SentimentValidator works correctly."""
    # Create config
    config = SentimentConfig(
        threshold=0.7,
        valid_labels=["positive"],
    )

    # Create validator
    validator = SentimentValidator(config)

    # Test positive sentiment (should pass)
    result = validator.validate("This is good and great!")
    assert result.passed is True
    assert "sentiment" in result.metadata
    assert result.metadata["sentiment"] == "positive"

    # Test negative sentiment (should fail)
    result = validator.validate("This is bad and terrible!")
    assert result.passed is False
    assert "sentiment" in result.metadata
    assert result.metadata["sentiment"] == "negative"

    # Test neutral sentiment (should fail)
    result = validator.validate("This is a simple text.")
    assert result.passed is False
    assert "sentiment" in result.metadata
    assert result.metadata["sentiment"] == "neutral"


def test_factory_functions():
    """Test that factory functions work correctly."""
    # Test create_sentiment_validator
    validator = create_sentiment_validator(
        threshold=0.7,
        valid_labels=["positive"],
    )
    assert isinstance(validator, SentimentValidator)
    assert validator.config.threshold == 0.7
    assert validator.config.valid_labels == ["positive"]
