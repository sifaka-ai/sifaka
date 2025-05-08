"""
Sentiment validation rules for Sifaka.

This module provides rules for validating the sentiment of text,
ensuring that text has the expected sentiment (positive, negative, or neutral).
"""

import importlib
from typing import List, Optional, Any, Dict, Union, Literal

from pydantic import Field, PrivateAttr

from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RuleResult,
    BaseValidator,
    create_rule,
)
from sifaka.utils.state import StateManager, RuleState, create_rule_state


SentimentType = Literal["positive", "negative", "neutral"]


class SentimentValidator(BaseValidator[str]):
    """
    Validator for text sentiment.

    This validator checks if text has the expected sentiment
    using a sentiment analysis model.

    Attributes:
        expected_sentiment: The expected sentiment of the text
        threshold: Confidence threshold for sentiment detection
    """

    expected_sentiment: SentimentType = Field(
        default="positive",
        description="The expected sentiment of the text",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for sentiment detection",
    )

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self, expected_sentiment: SentimentType = "positive", threshold: float = 0.7, **kwargs
    ):
        """Initialize the validator."""
        super().__init__(**kwargs)
        self.expected_sentiment = expected_sentiment
        self.threshold = threshold

    def warm_up(self) -> None:
        """Initialize the validator if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()
            try:
                # Try to import transformers
                state.transformers = importlib.import_module("transformers")
                # Load sentiment analysis pipeline
                state.pipeline = state.transformers.pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                )
            except ImportError:
                state.transformers = None
                state.pipeline = None
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate that text has the expected sentiment.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Check if transformers is available
        if state.transformers is None or state.pipeline is None:
            return RuleResult(
                passed=False,
                message="transformers package is required for sentiment validation. Install with: pip install transformers",
                metadata={"reason": "missing_dependency"},
            )

        try:
            # Predict sentiment
            result = state.pipeline(text)[0]
            label = result["label"].lower()
            score = result["score"]

            # Map LABEL_0/LABEL_1 to negative/positive if needed
            if label == "label_0":
                label = "negative"
            elif label == "label_1":
                label = "positive"

            # Check if sentiment matches expected sentiment
            matches_expected = label == self.expected_sentiment
            confidence_sufficient = score >= self.threshold

            # Create metadata
            metadata = {
                "sentiment": label,
                "confidence": score,
                "expected_sentiment": self.expected_sentiment,
                "threshold": self.threshold,
            }

            if matches_expected and confidence_sufficient:
                # Passed validation
                return RuleResult(
                    passed=True,
                    message=f"Text has the expected sentiment: {label}",
                    metadata=metadata,
                )
            else:
                # Failed validation
                if not matches_expected:
                    message = f"Text has {label} sentiment, but expected {self.expected_sentiment}"
                else:
                    message = (
                        f"Sentiment confidence ({score:.2f}) is below threshold ({self.threshold})"
                    )

                return RuleResult(
                    passed=False,
                    message=message,
                    metadata=metadata,
                )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Sentiment validation failed: {str(e)}",
                metadata={"error": str(e)},
            )


class SentimentRule(Rule[str, RuleResult, SentimentValidator, Any]):
    """
    Rule that validates text sentiment.

    This rule ensures that text has the expected sentiment.
    It uses a sentiment analysis model to detect the sentiment of the text
    and validates that it matches the expected sentiment.

    Attributes:
        _name: The name of the rule
        _description: Description of the rule
        _config: Rule configuration
        _validator: The validator used by this rule
    """

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()
            state.validator = self._create_default_validator()
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """Validate the text."""
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Check cache
        cache_key = text
        if cache_key in state.cache:
            return state.cache[cache_key]

        # Delegate to validator
        result = state.validator.validate(text, **kwargs)

        # Cache result
        state.cache[cache_key] = result

        return result

    def _create_default_validator(self) -> SentimentValidator:
        """
        Create a default validator.

        Returns:
            A SentimentValidator with default settings
        """
        params = self._config.params
        return SentimentValidator(
            expected_sentiment=params.get("expected_sentiment", "positive"),
            threshold=params.get("threshold", 0.7),
        )


def create_sentiment_validator(
    expected_sentiment: SentimentType = "positive",
    threshold: float = 0.7,
    **kwargs: Any,
) -> SentimentValidator:
    """
    Create a sentiment validator.

    This factory function creates a validator that ensures text has the expected
    sentiment. It uses a sentiment analysis model to detect the sentiment of the
    text and validates that it matches the expected sentiment.

    Args:
        expected_sentiment: The expected sentiment of the text
        threshold: Confidence threshold for sentiment detection
        **kwargs: Additional parameters for the validator

    Returns:
        A validator that validates text sentiment

    Examples:
        ```python
        from sifaka.rules.content.sentiment import create_sentiment_validator

        # Create a validator that expects positive sentiment
        validator = create_sentiment_validator(
            expected_sentiment="positive",
            threshold=0.8
        )

        # Validate text
        result = validator.validate("I love this product!")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'transformers' package to be installed:
    pip install transformers
    """
    validator = SentimentValidator(
        expected_sentiment=expected_sentiment,
        threshold=threshold,
        **kwargs,
    )

    # Initialize the validator
    validator.warm_up()

    return validator


def create_sentiment_rule(
    expected_sentiment: SentimentType = "positive",
    threshold: float = 0.7,
    name: str = "sentiment_rule",
    description: str = "Validates that text has the expected sentiment",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule that validates text sentiment.

    This factory function creates a rule that ensures text has the expected
    sentiment. It uses a sentiment analysis model to detect the sentiment of the
    text and validates that it matches the expected sentiment.

    Args:
        expected_sentiment: The expected sentiment of the text
        threshold: Confidence threshold for sentiment detection
        name: The name of the rule
        description: Description of the rule
        config: Rule configuration
        **kwargs: Additional parameters for the rule

    Returns:
        A rule that validates text sentiment

    Examples:
        ```python
        from sifaka.rules.content.sentiment import create_sentiment_rule

        # Create a rule that expects positive sentiment
        rule = create_sentiment_rule(
            expected_sentiment="positive",
            threshold=0.8,
            name="positive_sentiment_rule"
        )

        # Validate text
        result = rule.validate("I love this product!")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'transformers' package to be installed:
    pip install transformers
    """
    # Create rule configuration
    rule_config = config or RuleConfig()
    rule_config = rule_config.with_params(
        expected_sentiment=expected_sentiment,
        threshold=threshold,
        **{k: v for k, v in kwargs.items() if k not in ["name", "description"]},
    )

    # Create rule
    rule = SentimentRule(
        name=name,
        description=description,
        config=rule_config,
    )

    # Initialize the rule
    rule.warm_up()

    return rule
