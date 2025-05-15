"""
Sentiment analysis content validation rules for Sifaka.

This module provides rules for analyzing and validating text sentiment,
including positive/negative sentiment detection and emotional content analysis.

## Overview
The sentiment validation rules help ensure that text meets specific sentiment
requirements, such as being positive, neutral, or having a certain confidence
level. This is useful for content moderation, ensuring appropriate tone in
responses, and analyzing emotional content.

## Components
- **SentimentConfig**: Configuration for sentiment validation
- **SimpleSentimentClassifier**: Basic sentiment classifier implementation
- **SentimentAnalyzer**: Analyzer for sentiment detection
- **SentimentValidator**: Validator for sentiment requirements
- **SentimentRule**: Rule for validating text sentiment
- **Factory Functions**: create_sentiment_validator, create_sentiment_rule

## Usage Examples
```python
from sifaka.rules.content.sentiment import create_sentiment_rule

# Create a sentiment rule using the factory function
sentiment_rule = create_sentiment_rule(
    threshold=0.7,
    valid_labels=["positive", "neutral"]
)

# Validate text
result = sentiment_rule.validate("This is a great test!") if sentiment_rule else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
print(f"Sentiment: {result.metadata['sentiment']}")
print(f"Confidence: {result.metadata['confidence']}")
```

## Error Handling
- Empty text handling through BaseValidator.handle_empty_text
- Classification errors handled through try_operation
- Detailed validation results with metadata for debugging
- Caching for performance optimization
"""

import time
from typing import Any, Dict, List, Optional, Union, Type, cast, TypeVar, Callable
from pydantic import BaseModel, Field, ConfigDict
from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.utils.errors.handling import try_operation
from sifaka.core.results import ClassificationResult as CoreClassificationResult
from sifaka.utils.results import (
    create_classification_result,
    create_unknown_result,
    create_rule_result,
    create_error_result,
)
from sifaka.utils.state import create_rule_state

logger = get_logger(__name__)


class SentimentConfig(BaseModel):
    """
    Configuration for sentiment validation.

    This class defines the configuration options for sentiment validation,
    including threshold, valid labels, and caching parameters.

    ## Architecture
    The class uses Pydantic for validation and immutability, with field
    constraints to ensure valid parameter ranges.

    ## Lifecycle
    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Range validation for threshold (0.0-1.0)
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by SentimentAnalyzer
       - Used by SentimentValidator
       - Used by SentimentRule._create_default_validator

    ## Examples
    ```python
    from sifaka.rules.content.sentiment import SentimentConfig

    # Create with default values
    config = SentimentConfig()

    # Create with custom values
    config = SentimentConfig(
        threshold=0.7,
        valid_labels=["positive", "neutral"],
        cache_size=200
    )
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Threshold for sentiment detection"
    )
    valid_labels: List[str] = Field(
        default=["positive", "neutral"], description="List of valid sentiment labels"
    )
    cache_size: int = Field(default=100, ge=1, description="Size of the validation cache")
    priority: int = Field(default=1, ge=0, description="Priority of the rule")
    cost: float = Field(default=1.0, ge=0.0, description="Cost of running the rule")


class SimpleSentimentClassifier:
    """
    Simple sentiment classifier for testing.

    This class provides a basic sentiment classification implementation
    for demonstration and testing purposes. It analyzes text for positive,
    negative, or neutral sentiment based on keyword matching.

    ## Architecture
    The classifier uses a simple keyword-based approach with error handling
    through the try_operation utility to ensure robust behavior.

    ## Lifecycle
    1. **Classification**: Analyze text for sentiment
       - Check for empty text
       - Count positive and negative keywords
       - Determine sentiment based on keyword counts
       - Return classification result with confidence

    ## Error Handling
    - Empty text handling through handle_empty_text_for_classifier
    - Error handling through try_operation
    - Default to unknown result on classification errors

    ## Examples
    ```python
    from sifaka.rules.content.sentiment import SimpleSentimentClassifier

    # Create classifier
    classifier = SimpleSentimentClassifier()

    # Classify text
    result = classifier.classify("This is a great test!") if classifier else ""
    print(f"Label: {result.label}, Confidence: {result.confidence}")
    ```
    """

    def classify(self, text: str) -> CoreClassificationResult[Any, str]:
        """
        Classify text sentiment.

        This method analyzes the input text and determines its sentiment
        (positive, negative, or neutral) based on keyword matching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment label, confidence, and metadata
        """
        from sifaka.utils.errors.handling import try_operation
        from sifaka.utils.results import create_classification_result, create_unknown_result
        from sifaka.utils.text import handle_empty_text_for_classifier

        empty_result = handle_empty_text_for_classifier(text)
        if empty_result:
            # Type assertion for mypy to understand this is a CoreClassificationResult
            return empty_result  # type: ignore

        def _classify() -> CoreClassificationResult[Any, str]:
            positive_words = ["good", "great", "excellent", "happy", "positive"]
            negative_words = ["bad", "terrible", "awful", "sad", "negative"]
            text_lower = text.lower() if text else ""
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
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

        # Type assertion for mypy
        result: CoreClassificationResult[Any, str] = try_operation(
            _classify,
            component_name="SimpleSentimentClassifier",
            default_value=create_unknown_result(
                component_name="SimpleSentimentClassifier", reason="classification_error"
            ),
        )
        return result


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
        from sifaka.utils.errors.handling import try_operation
        from sifaka.utils.results import create_rule_result, create_error_result

        def _analyze() -> RuleResult:
            classification_result = self._classifier.classify(text)
            if hasattr(classification_result, "label") and hasattr(
                classification_result, "confidence"
            ):
                label = classification_result.label
                confidence = classification_result.confidence
                metadata = (
                    classification_result.metadata
                    if hasattr(classification_result, "metadata")
                    else {}
                )

                is_valid = label in self._valid_labels and confidence >= self._threshold
                return create_rule_result(
                    passed=is_valid,
                    message=f"Sentiment '{label}' with confidence {confidence:.2f} {'meets' if is_valid else 'does not meet'} criteria",
                    component_name="SentimentAnalyzer",
                    metadata={
                        "sentiment": label,
                        "confidence": confidence,
                        "threshold": self._threshold,
                        "valid_labels": self._valid_labels,
                        "classifier_metadata": metadata,
                    },
                )
            else:
                # Handle unexpected result format
                return create_rule_result(
                    passed=False,
                    message="Invalid classification result format",
                    component_name="SentimentAnalyzer",
                    metadata={
                        "threshold": self._threshold,
                        "valid_labels": self._valid_labels,
                    },
                )

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
        result = validator.validate("This is a great test!") if validator else ""
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
        self._state_manager.update("config", config)
        self._state_manager.update("analyzer", SentimentAnalyzer(config))
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def config(self) -> SentimentConfig:
        """
        Get the validator configuration.

        Returns:
            The sentiment configuration
        """
        config = self._state_manager.get("config")
        if isinstance(config, SentimentConfig):
            return config
        # Return a default config if for some reason the stored config is not a SentimentConfig
        return SentimentConfig()

    def validate(self, text: str) -> RuleResult:
        """
        Validate the given text for sentiment.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()
        empty_result = self.handle_empty_text(text)
        if empty_result:
            # Safety check to make sure empty_result is RuleResult
            if isinstance(empty_result, RuleResult):
                return empty_result
            # If not, convert it to RuleResult (though this should not happen)
            return RuleResult(
                passed=False,
                message="Empty text validation failed",
                metadata={"error": "empty_text"},
                score=0.0,
                issues=["Empty text"],
                suggestions=["Provide non-empty text"],
                processing_time_ms=0.0,
            )

        try:
            analyzer = self._state_manager.get("analyzer")
            if analyzer is None:
                # Create error result if analyzer is not found
                return RuleResult(
                    passed=False,
                    message="Sentiment analyzer not found",
                    metadata={"error": "analyzer_not_found"},
                    score=0.0,
                    issues=["Sentiment analyzer not found"],
                    suggestions=["Check validator configuration"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            result = analyzer.analyze(text)

            # Create explicitly typed RuleResult
            typed_result: RuleResult = result.with_metadata(
                validator_type=self.__class__.__name__,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            self.update_statistics(typed_result)
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            if self.config.cache_size > 0:
                cache = self._state_manager.get("cache", {})
                if len(cache) >= self.config.cache_size:
                    cache = {}
                cache[text] = typed_result
                self._state_manager.update("cache", cache)

            # Ensure we return a properly typed RuleResult
            return typed_result

        except Exception as e:
            self.record_error(e)
            if logger:
                logger.error(f"Sentiment validation failed: {e}")

            error_message = f"Error validating sentiment: {str(e)}"

            # Create a properly typed RuleResult for the error case
            error_result: RuleResult = RuleResult(
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
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            self.update_statistics(error_result)
            return error_result


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
        result = rule.validate("This is a great test!") if rule else ""
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str = "sentiment_rule",
        description: str = "Validates text sentiment",
        config: Optional[Optional[RuleConfig]] = None,
        validator: Optional[Optional[SentimentValidator]] = None,
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
        sentiment_validator = validator or self._create_default_validator()
        self._state_manager.update("sentiment_validator", sentiment_validator)
        self._state_manager.set_metadata("rule_type", "SentimentRule")
        self._state_manager.set_metadata("creation_time", time.time())

    def _create_default_validator(self) -> SentimentValidator:
        """
        Create a default validator from config.

        Returns:
            A configured SentimentValidator
        """
        params = self.config.params
        config = SentimentConfig(
            threshold=params.get("threshold", 0.6),
            valid_labels=params.get("valid_labels", ["positive", "neutral"]),
            cache_size=self.config.cache_size,
            priority=self.config.priority,
            cost=self.config.cost,
        )
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
        config_params: Dict[str, Any] = {}
        if threshold is not None:
            config_params["threshold"] = threshold
        if valid_labels is not None:
            config_params["valid_labels"] = valid_labels
        config_params.update(kwargs)
        config = SentimentConfig(**config_params)
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
        validator = create_sentiment_validator(threshold=threshold, valid_labels=valid_labels)
        params: Dict[str, Any] = {}
        if threshold is not None:
            params["threshold"] = threshold
        if valid_labels is not None:
            params["valid_labels"] = valid_labels
        rule_name = name or rule_id or "sentiment_rule"
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **kwargs,
        )
        return SentimentRule(
            name=rule_name, description=description, config=config, validator=validator
        )
    except Exception as e:
        logger.error(f"Error creating sentiment rule: {e}")
        raise ValueError(f"Error creating sentiment rule: {str(e)}")


__all__ = [
    "SentimentConfig",
    "SentimentAnalyzer",
    "SentimentValidator",
    "SentimentRule",
    "create_sentiment_validator",
    "create_sentiment_rule",
]
