"""Tests for sentiment rules."""

import pytest

from sifaka.rules.sentiment import (
    DEFAULT_EMOTION_CATEGORIES,
    DEFAULT_NEGATIVE_WORDS,
    DEFAULT_POSITIVE_WORDS,
    EmotionalContentConfig,
    EmotionalContentRule,
    SentimentConfig,
    SentimentRule,
    create_emotional_content_rule,
    create_sentiment_rule,
)


@pytest.fixture
def sentiment_config() -> SentimentConfig:
    """Create a test sentiment configuration."""
    return SentimentConfig(
        threshold=0.6,
        positive_words={"good", "great", "excellent"},
        negative_words={"bad", "terrible", "awful"},
        cache_size=100,
        priority=1,
        cost=1.0,
    )

@pytest.fixture
def emotional_content_config() -> EmotionalContentConfig:
    """Create a test emotional content configuration."""
    return EmotionalContentConfig(
        categories={
            "joy": ["happy", "delighted", "excited"],
            "sadness": ["sad", "unhappy", "depressed"],
            "anger": ["angry", "furious", "outraged"],
        },
        min_emotion_score=0.3,
        max_emotion_score=0.8,
        cache_size=100,
        priority=1,
        cost=1.0,
    )

def test_sentiment_config_validation():
    """Test sentiment configuration validation."""
    # Valid configuration
    config = SentimentConfig(
        threshold=0.5,
        positive_words={"good"},
        negative_words={"bad"},
    )
    assert config.threshold == 0.5
    assert config.positive_words == {"good"}
    assert config.negative_words == {"bad"}

    # Invalid threshold
    with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
        SentimentConfig(
            threshold=1.5,
            positive_words={"good"},
            negative_words={"bad"},
        )

    # Empty positive words
    with pytest.raises(ValueError, match="Must provide at least one positive word"):
        SentimentConfig(
            threshold=0.5,
            positive_words=set(),
            negative_words={"bad"},
        )

    # Empty negative words
    with pytest.raises(ValueError, match="Must provide at least one negative word"):
        SentimentConfig(
            threshold=0.5,
            positive_words={"good"},
            negative_words=set(),
        )

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        SentimentConfig(
            threshold=0.5,
            positive_words={"good"},
            negative_words={"bad"},
            cache_size=-1,
        )

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        SentimentConfig(
            threshold=0.5,
            positive_words={"good"},
            negative_words={"bad"},
            priority=-1,
        )

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        SentimentConfig(
            threshold=0.5,
            positive_words={"good"},
            negative_words={"bad"},
            cost=-1,
        )

def test_emotional_content_config_validation():
    """Test emotional content configuration validation."""
    # Valid configuration
    config = EmotionalContentConfig(
        categories={"joy": ["happy", "excited"]},
        min_emotion_score=0.3,
        max_emotion_score=0.8,
    )
    assert config.categories == {"joy": ["happy", "excited"]}
    assert config.min_emotion_score == 0.3
    assert config.max_emotion_score == 0.8

    # Invalid emotion scores
    with pytest.raises(
        ValueError, match="Emotion scores must be between 0.0 and 1.0 and min <= max"
    ):
        EmotionalContentConfig(
            categories={"joy": ["happy"]},
            min_emotion_score=0.8,
            max_emotion_score=0.3,
        )

    # Empty categories
    with pytest.raises(ValueError, match="Must provide at least one emotion category"):
        EmotionalContentConfig(categories={})

    # Empty category indicators
    with pytest.raises(ValueError, match="Category joy must have at least one indicator"):
        EmotionalContentConfig(categories={"joy": []})

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        EmotionalContentConfig(
            categories={"joy": ["happy"]},
            cache_size=-1,
        )

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        EmotionalContentConfig(
            categories={"joy": ["happy"]},
            priority=-1,
        )

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        EmotionalContentConfig(
            categories={"joy": ["happy"]},
            cost=-1,
        )

def test_sentiment_rule_validation(sentiment_config):
    """Test sentiment rule validation."""
    rule = create_sentiment_rule(
        name="test_sentiment",
        description="Test sentiment rule",
        threshold=sentiment_config.threshold,
        positive_words=sentiment_config.positive_words,
        negative_words=sentiment_config.negative_words,
    )

    # Test neutral content
    result = rule._validate_impl("This is a neutral text.")
    assert result.passed
    assert "Neutral sentiment detected" in result.message
    assert result.metadata["sentiment_score"] == 0.5
    assert result.metadata["positive_matches"] == 0
    assert result.metadata["negative_matches"] == 0

    # Test positive content
    result = rule._validate_impl("This is good and great content.")
    assert result.passed
    assert "Positive sentiment detected" in result.message
    assert result.metadata["sentiment_score"] >= sentiment_config.threshold
    assert result.metadata["positive_matches"] > 0
    assert result.metadata["negative_matches"] == 0

    # Test negative content
    result = rule._validate_impl("This is bad and terrible content.")
    assert not result.passed
    assert "Negative sentiment detected" in result.message
    assert result.metadata["sentiment_score"] < sentiment_config.threshold
    assert result.metadata["positive_matches"] == 0
    assert result.metadata["negative_matches"] > 0

    # Test mixed content
    result = rule._validate_impl("This is good but also terrible.")
    assert result.metadata["positive_matches"] > 0
    assert result.metadata["negative_matches"] > 0

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore

def test_emotional_content_rule_validation(emotional_content_config):
    """Test emotional content rule validation."""
    rule = create_emotional_content_rule(
        name="test_emotional",
        description="Test emotional content rule",
        categories=emotional_content_config.categories,
        min_emotion_score=emotional_content_config.min_emotion_score,
        max_emotion_score=emotional_content_config.max_emotion_score,
    )

    # Test neutral content
    result = rule._validate_impl("This is a neutral text without emotions.")
    assert result.passed
    assert "No strong emotions detected" in result.message
    assert not result.metadata["emotion_scores"]
    assert not result.metadata["detected_emotions"]

    # Test balanced emotional content
    result = rule._validate_impl("I am happy but also a bit sad sometimes.")
    assert result.passed
    assert "Balanced emotional content detected" in result.message
    assert "joy" in result.metadata["emotion_scores"]
    assert "sadness" in result.metadata["emotion_scores"]
    assert (
        emotional_content_config.min_emotion_score
        <= result.metadata["average_score"]
        <= emotional_content_config.max_emotion_score
    )

    # Test extreme emotional content
    result = rule._validate_impl("I am absolutely furious, angry, and outraged!")
    assert not result.passed
    assert "Unbalanced emotional content detected" in result.message
    assert "anger" in result.metadata["emotion_scores"]
    assert result.metadata["emotion_scores"]["anger"] > emotional_content_config.max_emotion_score

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore

def test_create_sentiment_rule():
    """Test sentiment rule factory function."""
    # Test with default configuration
    rule = create_sentiment_rule(
        name="test_sentiment",
        description="Test sentiment rule",
    )
    assert isinstance(rule, SentimentRule)
    assert rule.name == "test_sentiment"
    assert rule.description == "Test sentiment rule"

    # Test with custom configuration
    custom_rule = create_sentiment_rule(
        name="custom_sentiment",
        description="Custom sentiment rule",
        threshold=0.7,
        positive_words={"awesome", "fantastic"},
        negative_words={"horrible", "dreadful"},
    )
    assert isinstance(custom_rule, SentimentRule)
    assert custom_rule.name == "custom_sentiment"
    assert custom_rule.description == "Custom sentiment rule"

def test_create_emotional_content_rule():
    """Test emotional content rule factory function."""
    # Test with default configuration
    rule = create_emotional_content_rule(
        name="test_emotional",
        description="Test emotional content rule",
    )
    assert isinstance(rule, EmotionalContentRule)
    assert rule.name == "test_emotional"
    assert rule.description == "Test emotional content rule"

    # Test with custom configuration
    custom_categories = {
        "excitement": ["thrilled", "ecstatic"],
        "calmness": ["peaceful", "serene"],
    }
    custom_rule = create_emotional_content_rule(
        name="custom_emotional",
        description="Custom emotional content rule",
        categories=custom_categories,
        min_emotion_score=0.2,
        max_emotion_score=0.9,
    )
    assert isinstance(custom_rule, EmotionalContentRule)
    assert custom_rule.name == "custom_emotional"
    assert custom_rule.description == "Custom emotional content rule"

def test_default_word_sets():
    """Test default word sets for sentiment analysis."""
    assert isinstance(DEFAULT_POSITIVE_WORDS, frozenset)
    assert len(DEFAULT_POSITIVE_WORDS) > 0
    assert "good" in DEFAULT_POSITIVE_WORDS
    assert "great" in DEFAULT_POSITIVE_WORDS

    assert isinstance(DEFAULT_NEGATIVE_WORDS, frozenset)
    assert len(DEFAULT_NEGATIVE_WORDS) > 0
    assert "bad" in DEFAULT_NEGATIVE_WORDS
    assert "terrible" in DEFAULT_NEGATIVE_WORDS

def test_default_emotion_categories():
    """Test default emotion categories."""
    assert isinstance(DEFAULT_EMOTION_CATEGORIES, dict)
    assert len(DEFAULT_EMOTION_CATEGORIES) > 0
    assert "joy" in DEFAULT_EMOTION_CATEGORIES
    assert "sadness" in DEFAULT_EMOTION_CATEGORIES
    assert "anger" in DEFAULT_EMOTION_CATEGORIES
    assert isinstance(DEFAULT_EMOTION_CATEGORIES["joy"], list)
    assert len(DEFAULT_EMOTION_CATEGORIES["joy"]) > 0
