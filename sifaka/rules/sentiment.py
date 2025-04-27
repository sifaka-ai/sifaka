"""
Sentiment analysis rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.
"""

from typing import Dict, List, Set, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field
from sifaka.rules.base import Rule, RuleResult


@dataclass(frozen=True)
class SentimentConfig:
    """Configuration for sentiment validation."""

    threshold: float
    positive_words: Set[str]
    negative_words: Set[str]
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.positive_words:
            raise ValueError("Must provide at least one positive word")
        if not self.negative_words:
            raise ValueError("Must provide at least one negative word")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")


@dataclass(frozen=True)
class EmotionalContentConfig:
    """Configuration for emotional content validation."""

    categories: Dict[str, List[str]]
    min_emotion_score: float = 0.3
    max_emotion_score: float = 0.8
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.min_emotion_score <= self.max_emotion_score <= 1.0:
            raise ValueError("Emotion scores must be between 0.0 and 1.0 and min <= max")
        if not self.categories:
            raise ValueError("Must provide at least one emotion category")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")
        for category, indicators in self.categories.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")


@runtime_checkable
class SentimentValidator(Protocol):
    """Protocol for sentiment validation."""

    @property
    def config(self) -> SentimentConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate text sentiment."""
        ...


@runtime_checkable
class EmotionalContentValidator(Protocol):
    """Protocol for emotional content validation."""

    @property
    def config(self) -> EmotionalContentConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate emotional content."""
        ...


class DefaultSentimentValidator:
    """Default implementation of sentiment validation."""

    def __init__(self, config: SentimentConfig):
        self._config = config

    @property
    def config(self) -> SentimentConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text sentiment."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        positive_matches = sum(1 for word in self.config.positive_words if word in text_lower)
        negative_matches = sum(1 for word in self.config.negative_words if word in text_lower)

        total_matches = positive_matches + negative_matches
        if total_matches == 0:
            return RuleResult(
                passed=True,
                message="Neutral sentiment detected",
                metadata={
                    "sentiment_score": 0.5,
                    "positive_matches": 0,
                    "negative_matches": 0,
                },
            )

        sentiment_score = positive_matches / total_matches
        is_positive = sentiment_score >= self.config.threshold

        return RuleResult(
            passed=is_positive,
            message=f"{'Positive' if is_positive else 'Negative'} sentiment detected (score: {sentiment_score:.2f})",
            metadata={
                "sentiment_score": sentiment_score,
                "positive_matches": positive_matches,
                "negative_matches": negative_matches,
            },
        )


class DefaultEmotionalContentValidator:
    """Default implementation of emotional content validation."""

    def __init__(self, config: EmotionalContentConfig):
        self._config = config

    @property
    def config(self) -> EmotionalContentConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate emotional content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        emotion_scores: Dict[str, float] = {}
        detected_emotions: Dict[str, List[str]] = {}

        for category, indicators in self.config.categories.items():
            found_indicators = [indicator for indicator in indicators if indicator in text_lower]
            if found_indicators:
                score = len(found_indicators) / len(indicators)
                emotion_scores[category] = score
                detected_emotions[category] = found_indicators

        if not emotion_scores:
            return RuleResult(
                passed=True,
                message="No strong emotions detected",
                metadata={
                    "emotion_scores": {},
                    "detected_emotions": {},
                },
            )

        avg_emotion_score = sum(emotion_scores.values()) / len(emotion_scores)
        is_balanced = (
            self.config.min_emotion_score <= avg_emotion_score <= self.config.max_emotion_score
        )

        return RuleResult(
            passed=is_balanced,
            message=f"{'Balanced' if is_balanced else 'Unbalanced'} emotional content detected (score: {avg_emotion_score:.2f})",
            metadata={
                "emotion_scores": emotion_scores,
                "detected_emotions": detected_emotions,
                "average_score": avg_emotion_score,
            },
        )


class SentimentRule(Rule):
    """Rule for validating text sentiment."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: SentimentValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output sentiment."""
        return self._validator.validate(output)


class EmotionalContentRule(Rule):
    """Rule for validating emotional content."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: EmotionalContentValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output emotional content."""
        return self._validator.validate(output)


# Default word sets for sentiment analysis
DEFAULT_POSITIVE_WORDS: Final[Set[str]] = {
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "happy",
    "joy",
    "love",
    "beautiful",
    "perfect",
    "brilliant",
    "outstanding",
    "superb",
    "delightful",
    "pleasant",
    "impressive",
}

DEFAULT_NEGATIVE_WORDS: Final[Set[str]] = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "poor",
    "disappointing",
    "sad",
    "angry",
    "hate",
    "ugly",
    "wrong",
    "failure",
    "worst",
    "inferior",
    "unpleasant",
    "mediocre",
    "frustrating",
}

# Default emotion categories
DEFAULT_EMOTION_CATEGORIES: Final[Dict[str, List[str]]] = {
    "joy": ["happy", "delighted", "excited", "cheerful", "joyful"],
    "sadness": ["sad", "depressed", "unhappy", "gloomy", "miserable"],
    "anger": ["angry", "furious", "outraged", "irritated", "mad"],
    "fear": ["scared", "afraid", "terrified", "anxious", "worried"],
    "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
}


def create_sentiment_rule(
    name: str,
    description: str,
    threshold: float = 0.6,
    positive_words: Set[str] = None,
    negative_words: Set[str] = None,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> SentimentRule:
    """Create a sentiment rule with default configuration."""
    config = SentimentConfig(
        threshold=threshold,
        positive_words=positive_words or DEFAULT_POSITIVE_WORDS,
        negative_words=negative_words or DEFAULT_NEGATIVE_WORDS,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultSentimentValidator(config)
    return SentimentRule(name=name, description=description, validator=validator)


def create_emotional_content_rule(
    name: str,
    description: str,
    categories: Dict[str, List[str]] = None,
    min_emotion_score: float = 0.3,
    max_emotion_score: float = 0.8,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> EmotionalContentRule:
    """Create an emotional content rule with default configuration."""
    config = EmotionalContentConfig(
        categories=categories or DEFAULT_EMOTION_CATEGORIES,
        min_emotion_score=min_emotion_score,
        max_emotion_score=max_emotion_score,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultEmotionalContentValidator(config)
    return EmotionalContentRule(name=name, description=description, validator=validator)
