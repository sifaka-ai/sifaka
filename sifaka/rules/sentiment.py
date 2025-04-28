"""
Sentiment analysis rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult


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


class SentimentRule(Rule[str, RuleResult, DefaultSentimentValidator, Any]):
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
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultSentimentValidator:
        """Create a default validator from config."""
        sentiment_config = SentimentConfig(**self._rule_params)
        return DefaultSentimentValidator(sentiment_config)


class EmotionalContentRule(Rule[str, RuleResult, DefaultEmotionalContentValidator, Any]):
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
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

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
            "positive_words": {
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "awesome",
                "brilliant",
                "outstanding",
            },
            "negative_words": {
                "bad",
                "poor",
                "terrible",
                "awful",
                "horrible",
                "disappointing",
                "unacceptable",
                "mediocre",
            },
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
            "categories": {
                "joy": ["happy", "delighted", "excited", "joyful", "cheerful"],
                "sadness": ["sad", "depressed", "unhappy", "gloomy", "miserable"],
                "anger": ["angry", "furious", "outraged", "mad", "irritated"],
                "fear": ["afraid", "scared", "terrified", "anxious", "worried"],
            },
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


# Export public classes and functions
__all__ = [
    "SentimentRule",
    "SentimentConfig",
    "DefaultSentimentValidator",
    "EmotionalContentRule",
    "EmotionalContentConfig",
    "DefaultEmotionalContentValidator",
    "create_sentiment_rule",
    "create_emotional_content_rule",
]
