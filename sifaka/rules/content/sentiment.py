"""
Sentiment analysis content validation rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The SentimentConfig and EmotionalContentConfig classes extend RuleConfig and provide type-safe access to parameters
    - Factory functions (create_sentiment_rule, create_emotional_content_rule) handle configuration

Usage Example:
    from sifaka.rules.content.sentiment import create_sentiment_rule, create_emotional_content_rule

    # Create a sentiment rule using the factory function
    sentiment_rule = create_sentiment_rule(
        config={
            "threshold": 0.7,
            "positive_words": ["good", "great", "excellent"],
            "negative_words": ["bad", "poor", "terrible"]
        }
    )

    # Create an emotional content rule
    emotional_rule = create_emotional_content_rule(
        config={
            "categories": {
                "joy": ["happy", "delighted", "excited"],
                "sadness": ["sad", "depressed", "unhappy"]
            },
            "min_emotion_score": 0.3,
            "max_emotion_score": 0.8
        }
    )

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import RuleConfig

    rule = SentimentRule(
        config=RuleConfig(
            params={
                "threshold": 0.7,
                "positive_words": ["good", "great", "excellent"],
                "negative_words": ["bad", "poor", "terrible"]
            }
        )
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Set

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import ContentAnalyzer, ContentValidator, DefaultContentAnalyzer


# Default sets of positive and negative words for sentiment analysis
DEFAULT_POSITIVE_WORDS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "awesome",
        "brilliant",
        "outstanding",
        "positive",
        "happy",
        "joy",
        "love",
        "like",
        "best",
        "beautiful",
        "perfect",
        "nice",
        "superb",
    }
)

DEFAULT_NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "poor",
        "terrible",
        "awful",
        "horrible",
        "disappointing",
        "unacceptable",
        "mediocre",
        "negative",
        "sad",
        "hate",
        "dislike",
        "worst",
        "ugly",
        "failure",
        "wrong",
        "problem",
        "difficult",
        "hard",
    }
)

# Default emotion categories for emotional content analysis
DEFAULT_EMOTION_CATEGORIES = {
    "joy": ["happy", "delighted", "excited", "joyful", "cheerful", "pleased", "content"],
    "sadness": ["sad", "depressed", "unhappy", "gloomy", "miserable", "heartbroken", "down"],
    "anger": ["angry", "furious", "outraged", "mad", "irritated", "annoyed", "hostile"],
    "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "frightened"],
    "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "startled"],
    "disgust": ["disgusted", "repulsed", "revolted", "appalled", "nauseated"],
    "trust": ["trusting", "confident", "secure", "faithful", "reliable", "dependable"],
    "anticipation": ["anticipating", "expecting", "looking forward", "hopeful", "eager"],
}


@dataclass(frozen=True)
class SentimentWords:
    """Immutable container for sentiment words configuration."""

    positive_words: frozenset[str]
    negative_words: frozenset[str]
    threshold: float = 0.6

    def __post_init__(self) -> None:
        if not self.positive_words:
            raise ConfigurationError("Positive words list cannot be empty")
        if not self.negative_words:
            raise ConfigurationError("Negative words list cannot be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigurationError("Threshold must be between 0.0 and 1.0")

    def with_positive_words(self, positive_words: Set[str]) -> "SentimentWords":
        """Create new instance with updated positive words."""
        return SentimentWords(
            positive_words=frozenset(positive_words),
            negative_words=self.negative_words,
            threshold=self.threshold,
        )

    def with_negative_words(self, negative_words: Set[str]) -> "SentimentWords":
        """Create new instance with updated negative words."""
        return SentimentWords(
            positive_words=self.positive_words,
            negative_words=frozenset(negative_words),
            threshold=self.threshold,
        )

    def with_threshold(self, threshold: float) -> "SentimentWords":
        """Create new instance with updated threshold."""
        return SentimentWords(
            positive_words=self.positive_words,
            negative_words=self.negative_words,
            threshold=threshold,
        )


@dataclass(frozen=True)
class EmotionCategories:
    """Immutable container for emotion categories configuration."""

    categories: Dict[str, frozenset[str]]
    min_emotion_score: float = 0.3
    max_emotion_score: float = 0.8

    def __post_init__(self) -> None:
        if not self.categories:
            raise ConfigurationError("Emotion categories cannot be empty")
        if not 0.0 <= self.min_emotion_score <= self.max_emotion_score <= 1.0:
            raise ConfigurationError("Emotion scores must be between 0.0 and 1.0 and min <= max")

        # Convert to frozenset if not already
        categories_copy = {}
        for category, indicators in self.categories.items():
            if not indicators:
                raise ConfigurationError(f"Category {category} must have at least one indicator")
            if not isinstance(indicators, frozenset):
                categories_copy[category] = frozenset(indicators)
            else:
                categories_copy[category] = indicators

        if categories_copy:
            object.__setattr__(self, "categories", categories_copy)

    def with_categories(self, categories: Dict[str, List[str]]) -> "EmotionCategories":
        """Create new instance with updated categories."""
        categories_copy = {}
        for category, indicators in categories.items():
            categories_copy[category] = frozenset(indicators)

        return EmotionCategories(
            categories=categories_copy,
            min_emotion_score=self.min_emotion_score,
            max_emotion_score=self.max_emotion_score,
        )

    def with_score_range(self, min_score: float, max_score: float) -> "EmotionCategories":
        """Create new instance with updated score range."""
        return EmotionCategories(
            categories=self.categories,
            min_emotion_score=min_score,
            max_emotion_score=max_score,
        )


@dataclass(frozen=True)
class SentimentConfig(RuleConfig):
    """Configuration for sentiment validation."""

    threshold: float = 0.6
    positive_words: Set[str] = field(
        default_factory=lambda: {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "awesome",
            "brilliant",
            "outstanding",
        }
    )
    negative_words: Set[str] = field(
        default_factory=lambda: {
            "bad",
            "poor",
            "terrible",
            "awful",
            "horrible",
            "disappointing",
            "unacceptable",
            "mediocre",
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.positive_words:
            raise ValueError("Must provide at least one positive word")
        if not self.negative_words:
            raise ValueError("Must provide at least one negative word")

        # For consistency, copy configuration values to params
        if not self.params:
            object.__setattr__(
                self,
                "params",
                {
                    "threshold": self.threshold,
                    "positive_words": self.positive_words,
                    "negative_words": self.negative_words,
                    "cache_size": self.cache_size,
                    "priority": self.priority,
                    "cost": self.cost,
                },
            )


@dataclass(frozen=True)
class EmotionalContentConfig(RuleConfig):
    """Configuration for emotional content validation."""

    categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "joy": ["happy", "delighted", "excited", "joyful", "cheerful"],
            "sadness": ["sad", "depressed", "unhappy", "gloomy", "miserable"],
            "anger": ["angry", "furious", "outraged", "mad", "irritated"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried"],
        }
    )
    min_emotion_score: float = 0.3
    max_emotion_score: float = 0.8
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.min_emotion_score <= self.max_emotion_score <= 1.0:
            raise ValueError("Emotion scores must be between 0.0 and 1.0 and min <= max")
        if not self.categories:
            raise ValueError("Must provide at least one emotion category")
        for category, indicators in self.categories.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")

        # For consistency, copy configuration values to params
        if not self.params:
            object.__setattr__(
                self,
                "params",
                {
                    "categories": self.categories,
                    "min_emotion_score": self.min_emotion_score,
                    "max_emotion_score": self.max_emotion_score,
                    "cache_size": self.cache_size,
                    "priority": self.priority,
                    "cost": self.cost,
                },
            )


class SentimentValidator(ContentValidator):
    """Validator that checks text sentiment."""

    def __init__(
        self,
        sentiment_words: SentimentWords,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultContentAnalyzer())
        self._sentiment_words: Final[SentimentWords] = sentiment_words

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output has the expected sentiment."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            text_lower = output.lower()
            positive_matches = sum(
                1 for word in self._sentiment_words.positive_words if word.lower() in text_lower
            )
            negative_matches = sum(
                1 for word in self._sentiment_words.negative_words if word.lower() in text_lower
            )

            analysis = self._analyzer.analyze(output)

            total_matches = positive_matches + negative_matches
            if total_matches == 0:
                return RuleResult(
                    passed=True,
                    message="Neutral sentiment detected",
                    metadata={
                        "sentiment_score": 0.5,
                        "positive_matches": 0,
                        "negative_matches": 0,
                        "analysis": analysis,
                    },
                )

            sentiment_score = positive_matches / total_matches
            is_positive = sentiment_score >= self._sentiment_words.threshold

            return RuleResult(
                passed=is_positive,
                message=f"{'Positive' if is_positive else 'Negative'} sentiment detected (score: {sentiment_score:.2f})",
                metadata={
                    "sentiment_score": sentiment_score,
                    "positive_matches": positive_matches,
                    "negative_matches": negative_matches,
                    "analysis": analysis,
                },
            )

        except Exception as e:
            raise ValidationError(f"Sentiment validation failed: {str(e)}") from e


class EmotionalContentValidator(ContentValidator):
    """Validator that checks emotional content balance."""

    def __init__(
        self,
        emotion_categories: EmotionCategories,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultContentAnalyzer())
        self._emotion_categories: Final[EmotionCategories] = emotion_categories

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output has balanced emotional content."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            text_lower = output.lower()
            emotion_scores: Dict[str, float] = {}
            detected_emotions: Dict[str, List[str]] = {}

            for category, indicators in self._emotion_categories.categories.items():
                found_indicators = [
                    indicator for indicator in indicators if indicator.lower() in text_lower
                ]
                if found_indicators:
                    score = len(found_indicators) / len(indicators)
                    emotion_scores[category] = score
                    detected_emotions[category] = found_indicators

            analysis = self._analyzer.analyze(output)

            if not emotion_scores:
                return RuleResult(
                    passed=True,
                    message="No strong emotions detected",
                    metadata={
                        "emotion_scores": {},
                        "detected_emotions": {},
                        "analysis": analysis,
                    },
                )

            avg_emotion_score = sum(emotion_scores.values()) / len(emotion_scores)
            is_balanced = (
                self._emotion_categories.min_emotion_score
                <= avg_emotion_score
                <= self._emotion_categories.max_emotion_score
            )

            return RuleResult(
                passed=is_balanced,
                message=f"{'Balanced' if is_balanced else 'Unbalanced'} emotional content detected (score: {avg_emotion_score:.2f})",
                metadata={
                    "emotion_scores": emotion_scores,
                    "detected_emotions": detected_emotions,
                    "average_score": avg_emotion_score,
                    "analysis": analysis,
                },
            )

        except Exception as e:
            raise ValidationError(f"Emotional content validation failed: {str(e)}") from e


class DefaultSentimentValidator(BaseValidator[str]):
    """Default implementation of sentiment validation."""

    def __init__(self, config: SentimentConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> SentimentConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text sentiment."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params for consistency
        threshold = self.config.params.get("threshold", self.config.threshold)
        positive_words = self.config.params.get("positive_words", self.config.positive_words)
        negative_words = self.config.params.get("negative_words", self.config.negative_words)

        text_lower = text.lower()
        positive_matches = sum(1 for word in positive_words if word.lower() in text_lower)
        negative_matches = sum(1 for word in negative_words if word.lower() in text_lower)

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
        is_positive = sentiment_score >= threshold

        return RuleResult(
            passed=is_positive,
            message=f"{'Positive' if is_positive else 'Negative'} sentiment detected (score: {sentiment_score:.2f})",
            metadata={
                "sentiment_score": sentiment_score,
                "positive_matches": positive_matches,
                "negative_matches": negative_matches,
            },
        )


class DefaultEmotionalContentValidator(BaseValidator[str]):
    """Default implementation of emotional content validation."""

    def __init__(self, config: EmotionalContentConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> EmotionalContentConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate emotional content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params for consistency
        categories = self.config.params.get("categories", self.config.categories)
        min_emotion_score = self.config.params.get(
            "min_emotion_score", self.config.min_emotion_score
        )
        max_emotion_score = self.config.params.get(
            "max_emotion_score", self.config.max_emotion_score
        )

        text_lower = text.lower()
        emotion_scores: Dict[str, float] = {}
        detected_emotions: Dict[str, List[str]] = {}

        for category, indicators in categories.items():
            found_indicators = [
                indicator for indicator in indicators if indicator.lower() in text_lower
            ]
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
        is_balanced = min_emotion_score <= avg_emotion_score <= max_emotion_score

        return RuleResult(
            passed=is_balanced,
            message=f"{'Balanced' if is_balanced else 'Unbalanced'} emotional content detected (score: {avg_emotion_score:.2f})",
            metadata={
                "emotion_scores": emotion_scores,
                "detected_emotions": detected_emotions,
                "average_score": avg_emotion_score,
            },
        )


class SentimentRule(
    Rule[str, RuleResult, DefaultSentimentValidator, RuleResultHandler[RuleResult]]
):
    """Rule for validating text sentiment."""

    def __init__(
        self,
        name: str = "sentiment_rule",
        description: str = "Validates text sentiment",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultSentimentValidator] = None,
    ) -> None:
        """
        Initialize the rule with sentiment validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
        )

    def _create_default_validator(self) -> DefaultSentimentValidator:
        """Create a default validator from config."""
        sentiment_config = SentimentConfig(**self._rule_params)
        return DefaultSentimentValidator(sentiment_config)


class EmotionalContentRule(
    Rule[str, RuleResult, DefaultEmotionalContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule for validating emotional content."""

    def __init__(
        self,
        name: str = "emotional_content_rule",
        description: str = "Validates emotional content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultEmotionalContentValidator] = None,
    ) -> None:
        """
        Initialize the rule with emotional content validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
        )

    def _create_default_validator(self) -> DefaultEmotionalContentValidator:
        """Create a default validator from config."""
        emotional_config = EmotionalContentConfig(**self._rule_params)
        return DefaultEmotionalContentValidator(emotional_config)


def create_sentiment_rule(
    name: str = "sentiment_rule",
    description: str = "Validates text sentiment",
    config: Optional[Dict[str, Any]] = None,
) -> SentimentRule:
    """
    Create a sentiment rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured SentimentRule instance
    """
    if config is None:
        config = {
            "threshold": 0.6,
            "positive_words": DEFAULT_POSITIVE_WORDS,
            "negative_words": DEFAULT_NEGATIVE_WORDS,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

    return SentimentRule(
        name=name,
        description=description,
        config=rule_config,
    )


def create_emotional_content_rule(
    name: str = "emotional_content_rule",
    description: str = "Validates emotional content",
    config: Optional[Dict[str, Any]] = None,
) -> EmotionalContentRule:
    """
    Create an emotional content rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured EmotionalContentRule instance
    """
    if config is None:
        config = {
            "categories": DEFAULT_EMOTION_CATEGORIES,
            "min_emotion_score": 0.3,
            "max_emotion_score": 0.8,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

    return EmotionalContentRule(
        name=name,
        description=description,
        config=rule_config,
    )
