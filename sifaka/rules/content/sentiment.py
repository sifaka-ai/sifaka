"""
Sentiment analysis content validation rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.

Usage Example:
    ```python
    from sifaka.rules.content.sentiment import create_sentiment_rule

    # Create a sentiment rule using the factory function
    sentiment_rule = create_sentiment_rule(
        threshold=0.7,
        valid_labels=["positive", "neutral"]
    )

    # Validate text
    result = sentiment_rule.validate("This is a great test!")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
"""

import time
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.errors import try_operation
from sifaka.utils.results import (
    create_classification_result,
    create_unknown_result,
    create_rule_result,
    create_error_result,
)
from sifaka.utils.state import create_rule_state

logger = get_logger(__name__)


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
        from sifaka.utils.errors import try_operation
        from sifaka.utils.results import create_classification_result, create_unknown_result
        from sifaka.utils.text import handle_empty_text_for_classifier

        # Handle empty text
        empty_result = handle_empty_text_for_classifier(text)
        if empty_result:
            return empty_result

        # Use try_operation to handle potential errors
        def _classify():
            # Simple sentiment detection based on keywords
            positive_words = ["good", "great", "excellent", "happy", "positive"]
            negative_words = ["bad", "terrible", "awful", "sad", "negative"]

            text_lower = text.lower()

            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            # Determine sentiment
            if positive_count > negative_count:
                return create_classification_result(
                    label="positive",
                    confidence=0.8,
                    component_name="SimpleSentimentClassifier",
                    metadata={"positive_words": positive_count, "negative_words": negative_count},
                )
            elif negative_count > positive_count:
                return create_classification_result(
                    label="negative",
                    confidence=0.8,
                    component_name="SimpleSentimentClassifier",
                    metadata={"positive_words": positive_count, "negative_words": negative_count},
                )
            else:
                return create_classification_result(
                    label="neutral",
                    confidence=0.6,
                    component_name="SimpleSentimentClassifier",
                    metadata={"positive_words": positive_count, "negative_words": negative_count},
                )

        # Execute the classification with error handling
        return try_operation(
            _classify,
            component_name="SimpleSentimentClassifier",
            default_value=create_unknown_result(
                component_name="SimpleSentimentClassifier",
                reason="classification_error",
            ),
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
        from sifaka.utils.errors import try_operation
        from sifaka.utils.results import create_rule_result, create_error_result

        # Use try_operation to handle potential errors
        def _analyze():
            # Use the classifier to detect sentiment
            result = self._classifier.classify(text)

            # Determine if the text passes validation
            is_valid = result.label in self._valid_labels and result.confidence >= self._threshold

            return create_rule_result(
                passed=is_valid,
                message=(
                    f"Sentiment '{result.label}' with confidence {result.confidence:.2f} "
                    f"{'meets' if is_valid else 'does not meet'} criteria"
                ),
                component_name="SentimentAnalyzer",
                metadata={
                    "sentiment": result.label,
                    "confidence": result.confidence,
                    "threshold": self._threshold,
                    "valid_labels": self._valid_labels,
                    "classifier_metadata": result.metadata,
                },
            )

        # Execute the analysis with error handling
        return try_operation(
            _analyze,
            component_name="SentimentAnalyzer",
            default_value=create_error_result(
                message="Error analyzing sentiment",
                component_name="SentimentAnalyzer",
                error_type="ProcessingError",
            ),
        )

    def can_analyze(self, text: str) -> bool:
        """Check if this analyzer can analyze the given text."""
        return isinstance(text, str)


class SentimentValidator(BaseValidator[str]):
    """
    Validator for sentiment detection.

    This validator analyzes text for sentiment, determining if it is positive,
    negative, or neutral based on the configured threshold and valid labels.

    Lifecycle:
        1. Initialization: Set up with sentiment threshold and valid labels
        2. Validation: Analyze text for sentiment
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.sentiment import SentimentValidator, SentimentConfig

        # Create config
        config = SentimentConfig(
            threshold=0.7,
            valid_labels=["positive", "neutral"]
        )

        # Create validator
        validator = SentimentValidator(config)

        # Validate text
        result = validator.validate("This is a great test!")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, config: SentimentConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        self._state_manager.update("config", config)
        self._state_manager.update("analyzer", SentimentAnalyzer(config))

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def config(self) -> SentimentConfig:
        """
        Get the validator configuration.

        Returns:
            The sentiment configuration
        """
        return self._state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate the given text for sentiment.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")

            # Delegate to analyzer
            result = analyzer.analyze(text)

            # Add additional metadata
            result = result.with_metadata(
                validator_type=self.__class__.__name__, processing_time_ms=time.time() - start_time
            )

            # Update statistics
            self.update_statistics(result)

            # Update validation count in metadata
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Cache result if caching is enabled
            if self.config.cache_size > 0:
                cache = self._state_manager.get("cache", {})
                if len(cache) >= self.config.cache_size:
                    # Clear cache if it's full
                    cache = {}
                cache[text] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            self.record_error(e)
            logger.error(f"Sentiment validation failed: {e}")

            error_message = f"Error validating sentiment: {str(e)}"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=time.time() - start_time,
            )

            self.update_statistics(result)
            return result


class SentimentRule(Rule[str]):
    """
    Rule for validating sentiment.

    This rule analyzes text for sentiment, determining if it is positive,
    negative, or neutral based on the configured threshold and valid labels.

    Lifecycle:
        1. Initialization: Set up with sentiment threshold and valid labels
        2. Validation: Delegate to validator to analyze text for sentiment
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.sentiment import SentimentRule, SentimentValidator, SentimentConfig

        # Create config
        config = SentimentConfig(
            threshold=0.7,
            valid_labels=["positive", "neutral"]
        )

        # Create validator
        validator = SentimentValidator(config)

        # Create rule
        rule = SentimentRule(
            name="sentiment_rule",
            description="Validates text sentiment",
            validator=validator
        )

        # Validate text
        result = rule.validate("This is a great test!")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str = "sentiment_rule",
        description: str = "Validates text sentiment",
        config: Optional[RuleConfig] = None,
        validator: Optional[SentimentValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the sentiment rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name, description=description, rule_id=kwargs.pop("rule_id", name), **kwargs
            ),
            validator=validator,
        )

        # Store validator in state
        sentiment_validator = validator or self._create_default_validator()
        self._state_manager.update("sentiment_validator", sentiment_validator)

        # Set additional metadata
        self._state_manager.set_metadata("rule_type", "SentimentRule")
        self._state_manager.set_metadata("creation_time", time.time())

    def _create_default_validator(self) -> SentimentValidator:
        """
        Create a default validator from config.

        Returns:
            A configured SentimentValidator
        """
        # Extract sentiment specific params
        params = self.config.params
        config = SentimentConfig(
            threshold=params.get("threshold", 0.6),
            valid_labels=params.get("valid_labels", ["positive", "neutral"]),
            cache_size=self.config.cache_size,
            priority=self.config.priority,
            cost=self.config.cost,
        )

        # Store config in state for reference
        self._state_manager.update("validator_config", config)

        return SentimentValidator(config)


def create_sentiment_validator(
    threshold: Optional[float] = None,
    valid_labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> SentimentValidator:
    """
    Create a sentiment validator.

    This factory function creates a configured SentimentValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        threshold: Threshold for sentiment detection
        valid_labels: List of valid sentiment labels
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured SentimentValidator

    Examples:
        ```python
        from sifaka.rules.content.sentiment import create_sentiment_validator

        # Create a basic validator
        validator = create_sentiment_validator(threshold=0.7)

        # Create a validator with custom valid labels
        validator = create_sentiment_validator(
            threshold=0.7,
            valid_labels=["positive", "neutral"]
        )
        ```
    """
    try:
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

        # Create and return the validator
        return SentimentValidator(config)

    except Exception as e:
        logger.error(f"Error creating sentiment validator: {e}")
        raise ValueError(f"Error creating sentiment validator: {str(e)}")


def create_sentiment_rule(
    name: str = "sentiment_rule",
    description: str = "Validates text sentiment",
    threshold: Optional[float] = None,
    valid_labels: Optional[List[str]] = None,
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> SentimentRule:
    """
    Create a sentiment rule.

    This factory function creates a configured SentimentRule instance.
    It uses create_sentiment_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for sentiment detection
        valid_labels: List of valid sentiment labels
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured SentimentRule

    Examples:
        ```python
        from sifaka.rules.content.sentiment import create_sentiment_rule

        # Create a basic rule
        rule = create_sentiment_rule(threshold=0.7)

        # Create a rule with custom valid labels and metadata
        rule = create_sentiment_rule(
            threshold=0.7,
            valid_labels=["positive", "neutral"],
            name="custom_sentiment_rule",
            description="Validates text has positive or neutral sentiment",
            rule_id="sentiment_validator",
            severity="warning",
            category="content",
            tags=["sentiment", "content", "validation"]
        )
        ```
    """
    try:
        # Create validator using the validator factory
        validator = create_sentiment_validator(
            threshold=threshold,
            valid_labels=valid_labels,
        )

        # Create params dictionary for RuleConfig
        params = {}
        if threshold is not None:
            params["threshold"] = threshold
        if valid_labels is not None:
            params["valid_labels"] = valid_labels

        # Determine rule name
        rule_name = name or rule_id or "sentiment_rule"

        # Create RuleConfig
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **kwargs,
        )

        # Create and return the rule
        return SentimentRule(
            name=rule_name,
            description=description,
            config=config,
            validator=validator,
        )

    except Exception as e:
        logger.error(f"Error creating sentiment rule: {e}")
        raise ValueError(f"Error creating sentiment rule: {str(e)}")


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
