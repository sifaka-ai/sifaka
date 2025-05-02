"""
Sentiment analysis content validation rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.

## Rule and Validator Relationship

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both
- Empty text is handled consistently using BaseValidator.handle_empty_text

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- All rule-specific configuration is stored in RuleConfig.params
- Factory functions handle configuration extraction
- Validator factory functions create standalone validators
- Rule factory functions use validator factory functions internally

## Usage Example

```python
from sifaka.rules.content.sentiment import create_sentiment_rule

# Create a sentiment rule using the factory function
sentiment_rule = create_sentiment_rule(
    threshold=0.7,
    valid_labels=["positive", "neutral"]
)

# Validate text
result = sentiment_rule.validate("This is a great test!")
```
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict

# No need to import SentimentClassifier since we're using our own implementation
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
)


class SentimentConfig(BaseModel):
    """Configuration for sentiment validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for sentiment detection",
    )
    valid_labels: List[str] = Field(
        default=["positive", "neutral"],
        description="List of valid sentiment labels",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )


class SimpleSentimentClassifier:
    """Simple sentiment classifier for testing."""

    def classify(self, text: str):
        """Classify text sentiment.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment label
        """
        from sifaka.classifiers.base import ClassificationResult

        # Simple sentiment detection based on keywords
        positive_words = ["good", "great", "excellent", "happy", "positive"]
        negative_words = ["bad", "terrible", "awful", "sad", "negative"]

        text_lower = text.lower()

        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Determine sentiment
        if positive_count > negative_count:
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"positive_words": positive_count, "negative_words": negative_count},
            )
        elif negative_count > positive_count:
            return ClassificationResult(
                label="negative",
                confidence=0.8,
                metadata={"positive_words": positive_count, "negative_words": negative_count},
            )
        else:
            return ClassificationResult(
                label="neutral",
                confidence=0.6,
                metadata={"positive_words": positive_count, "negative_words": negative_count},
            )


class SentimentAnalyzer:
    """Analyzer for sentiment detection."""

    def __init__(self, config: SentimentConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the analyzer
        """
        self._config = config
        self._threshold = config.threshold
        self._valid_labels = config.valid_labels
        self._classifier = SimpleSentimentClassifier()

    def analyze(self, text: str) -> RuleResult:
        """Analyze text for sentiment.

        Args:
            text: The text to analyze

        Returns:
            RuleResult: The result of the analysis
        """
        # Use the classifier to detect sentiment
        result = self._classifier.classify(text)

        # Determine if the text passes validation
        is_valid = result.label in self._valid_labels and result.confidence >= self._threshold

        return RuleResult(
            passed=is_valid,
            message=(
                f"Sentiment '{result.label}' with confidence {result.confidence:.2f} "
                f"{'meets' if is_valid else 'does not meet'} criteria"
            ),
            metadata={
                "sentiment": result.label,
                "confidence": result.confidence,
                "threshold": self._threshold,
                "valid_labels": self._valid_labels,
                "classifier_metadata": result.metadata,
            },
        )

    def can_analyze(self, text: str) -> bool:
        """Check if this analyzer can analyze the given text."""
        return isinstance(text, str)


class SentimentValidator(BaseValidator[str]):
    """Validator for sentiment detection."""

    def __init__(self, config: SentimentConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__()
        self._config = config
        self._analyzer = SentimentAnalyzer(config)

    @property
    def config(self) -> SentimentConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **_: Any) -> RuleResult:
        """Validate the given text for sentiment.

        Args:
            text: The text to validate
            **_: Additional validation context (unused)

        Returns:
            RuleResult: The result of the validation
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Delegate to analyzer
        return self._analyzer.analyze(text)


class SentimentRule(Rule[str, RuleResult, SentimentValidator, RuleResultHandler[RuleResult]]):
    """Rule for validating sentiment."""

    def __init__(
        self,
        name: str = "sentiment_rule",
        description: str = "Validates text sentiment",
        config: Optional[RuleConfig] = None,
        validator: Optional[SentimentValidator] = None,
    ) -> None:
        """Initialize with configuration.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
        """
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
        )

    def _create_default_validator(self) -> SentimentValidator:
        """Create a default validator from config."""
        # Extract sentiment specific params
        params = self.config.params
        config = SentimentConfig(
            threshold=params.get("threshold", 0.6),
            valid_labels=params.get("valid_labels", ["positive", "neutral"]),
            cache_size=self.config.cache_size,
            priority=self.config.priority,
            cost=self.config.cost,
        )
        return SentimentValidator(config)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate the given text for sentiment.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult: The result of the validation
        """
        # Delegate to validator
        result = self._validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._name)


def create_sentiment_validator(
    threshold: Optional[float] = None,
    valid_labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> SentimentValidator:
    """Create a sentiment validator.

    Args:
        threshold: Threshold for sentiment detection
        valid_labels: List of valid sentiment labels
        **kwargs: Additional keyword arguments for the config

    Returns:
        SentimentValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if threshold is not None:
        config_params["threshold"] = threshold
    if valid_labels is not None:
        config_params["valid_labels"] = valid_labels

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = SentimentConfig(**config_params)

    # Create validator
    return SentimentValidator(config)


def create_sentiment_rule(
    name: str = "sentiment_rule",
    description: str = "Validates text sentiment",
    threshold: Optional[float] = None,
    valid_labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> SentimentRule:
    """Create a sentiment rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for sentiment detection
        valid_labels: List of valid sentiment labels
        **kwargs: Additional keyword arguments for the rule

    Returns:
        SentimentRule: The created rule
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create validator using the validator factory
    validator = create_sentiment_validator(
        threshold=threshold,
        valid_labels=valid_labels,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Create params dictionary for RuleConfig
    params = {}
    if threshold is not None:
        params["threshold"] = threshold
    if valid_labels is not None:
        params["valid_labels"] = valid_labels

    # Create RuleConfig
    config = RuleConfig(params=params, **rule_config_params)

    # Create rule
    return SentimentRule(
        name=name,
        description=description,
        config=config,
        validator=validator,
    )


__all__ = [
    # Config classes
    "SentimentConfig",
    # Analyzer classes
    "SentimentAnalyzer",
    # Validator classes
    "SentimentValidator",
    # Rule classes
    "SentimentRule",
    # Factory functions
    "create_sentiment_validator",
    "create_sentiment_rule",
]
