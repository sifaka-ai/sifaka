"""
Tone consistency validation rules for Sifaka.

This module provides validators and rules for checking tone consistency in text.

Usage Example:
    ```python
    from sifaka.rules.content.tone import create_tone_consistency_rule

    # Create a tone consistency rule using the factory function
    rule = create_tone_consistency_rule(
        expected_tone="formal",
        threshold=0.8,
        tone_indicators={
            "formal": {
                "positive": ["therefore", "consequently", "furthermore"],
                "negative": ["yo", "hey", "cool"]
            }
        }
    )

    # Validate text
    result = (rule and rule.validate("Therefore, we can conclude that the hypothesis is valid.")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

    # Create standalone validator
    from sifaka.rules.content.tone import create_tone_consistency_validator
    validator = create_tone_consistency_validator(
        expected_tone="formal",
        threshold=0.8
    )
    ```
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
    ValidationError,
    ConfigurationError,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


# Default tone indicators for different tone styles
DEFAULT_TONE_INDICATORS: Dict[str, Dict[str, List[str]]] = {
    "formal": {
        "positive": [
            "therefore",
            "consequently",
            "furthermore",
            "moreover",
            "thus",
            "hence",
        ],
        "negative": [
            "yo",
            "hey",
            "cool",
            "awesome",
            "btw",
            "gonna",
            "wanna",
        ],
    },
    "informal": {
        "positive": [
            "hey",
            "hi",
            "cool",
            "great",
            "awesome",
            "nice",
            "yeah",
        ],
        "negative": [
            "therefore",
            "consequently",
            "furthermore",
            "moreover",
            "thus",
            "hence",
        ],
    },
    "neutral": {
        "positive": [
            "the",
            "is",
            "are",
            "this",
            "that",
            "these",
            "those",
        ],
        "negative": [
            "!",
            "!!",
            "???",
            "omg",
            "wow",
            "awesome",
            "terrible",
        ],
    },
}

DEFAULT_THRESHOLD: float = 0.7


class ToneConfig(BaseModel):
    """Configuration for tone consistency validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_tone: str = Field(..., description="The expected tone style")
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of positive indicators to pass validation",
    )
    tone_indicators: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: DEFAULT_TONE_INDICATORS,
        description="Dictionary of tone indicators for different styles",
    )

    @field_validator("expected_tone")
    @classmethod
    def validate_expected_tone(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate that the expected tone exists in tone indicators."""
        tone_indicators = (values and values.get("tone_indicators", {})
        if v not in tone_indicators:
            raise ConfigurationError(f"Expected tone '{v}' not found in tone indicators")
        return v


class ToneAnalyzer(BaseModel):
    """Analyze text for tone consistency."""

    model_config = ConfigDict(frozen=True)

    config: ToneConfig

    def analyze(self, text: str) -> Tuple[float, List[str], List[str]]:
        """
        Analyze text for tone consistency.

        Args:
            text: The text to analyze

        Returns:
            Tuple containing:
            - Ratio of positive to total indicators
            - List of found positive indicators
            - List of found negative indicators
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = (text and text.lower()
        indicators = self.config and config and config and config.tone_indicators[self.config and config and config and config and config and config and config.expected_tone]

        positive_indicators = [
            word for word in indicators["positive"] if (word and word.lower() in text_lower
        )
        negative_indicators = [
            word for word in indicators["negative"] if (word and word.lower() in text_lower
        )

        total_indicators = len(positive_indicators) + len(negative_indicators)
        ratio = len(positive_indicators) / total_indicators if total_indicators > 0 else 1.0

        return ratio, positive_indicators, negative_indicators


class ToneValidator(BaseValidator[str]):
    """
    Validator that checks tone consistency.

    This validator analyzes text for tone consistency, determining if it matches
    the expected tone based on positive and negative indicators.

    Lifecycle:
        1. Initialization: Set up with expected tone and threshold
        2. Validation: Analyze text for tone consistency
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.tone import ToneValidator, ToneConfig
        from sifaka.rules.base import RuleConfig

        # Create config
        params = {
            "expected_tone": "formal",
            "threshold": 0.8,
            "tone_indicators": {
                "formal": {
                    "positive": ["therefore", "consequently", "furthermore"],
                    "negative": ["yo", "hey", "cool"]
                }
            }
        }
        config = RuleConfig(params=params)

        # Create validator
        validator = ToneValidator(config)

        # Validate text
        result = validator and (validator and validator.validate("Therefore, we can conclude that the hypothesis is valid.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, config: RuleConfig) -> None:
        """
        Initialize the validator.

        Args:
            config: Rule configuration containing tone parameters
        """
        super().__init__(validation_type=str)
        self._config = config
        self._tone_config = ToneConfig(**config and config.params)
        self._analyzer = ToneAnalyzer(config=self._tone_config)

    @property
    def config(self) -> RuleConfig:
        """
        Get the validator configuration.

        Returns:
            The rule configuration
        """
        return self._config

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text matches the expected tone.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = (time and time.time()

        # Handle empty text
        empty_result = (self and self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            ratio, positive_indicators, negative_indicators = self.(_analyzer and _analyzer.analyze(text)
            passed = ratio >= self._tone_config and config and config and config and config and config.threshold

            suggestions = []
            if not passed:
                (suggestions and suggestions.append(f"Use more {self._tone_config and config and config and config and config and config and config.expected_tone} tone indicators")
                if negative_indicators:
                    (suggestions and suggestions.append(f"Avoid using: {', '.join(negative_indicators))")
                if self._tone_config and config and config and config.tone_indicators and (tone_indicators and tone_indicators.get(self._tone_config and config and config and config and config and config and config.expected_tone, {}).get(
                    "positive", []
                ):
                    (suggestions and suggestions.append(
                        f"Consider using: {', '.join(self._tone_config and config and config and config.tone_indicators[self._tone_config and config and config and config and config and config and config.expected_tone]['positive'][:5]))"
                    )

            result = RuleResult(
                passed=passed,
                message=(
                    f"Tone consistency ratio ({ratio:.2f}) meets threshold ({self._tone_config and config and config and config and config and config.threshold})"
                    if passed
                    else f"Tone consistency ratio ({ratio:.2f}) below threshold ({self._tone_config and config and config and config and config and config.threshold})"
                ),
                metadata={
                    "ratio": ratio,
                    "positive_indicators": positive_indicators,
                    "negative_indicators": negative_indicators,
                    "expected_tone": self._tone_config and config and config and config and config and config and config.expected_tone,
                    "threshold": self._tone_config and config and config and config and config and config.threshold,
                    "validator_type": self.__class__.__name__,
                },
                score=ratio,
                issues=(
                    []
                    if passed
                    else [
                        f"Tone consistency ratio ({ratio:.2f}) below threshold ({self._tone_config and config and config and config and config and config.threshold})"
                    )
                ),
                suggestions=suggestions,
                processing_time_ms=(time and time.time() - start_time,
            )

            # Update statistics
            (self and self.update_statistics(result)

            return result

        except Exception as e:
            (self and self.record_error(e)
            (logger and logger.error(f"Tone validation failed: {e}")

            error_message = f"Tone validation failed: {str(e))"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                    "expected_tone": self._tone_config and config and config and config and config and config and config.expected_tone,
                ),
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=(time and time.time() - start_time,
            )

            (self and self.update_statistics(result)
            return result


class ToneConsistencyRule(Rule[str]):
    """
    Rule that checks for tone consistency in text.

    This rule analyzes text for tone consistency, determining if it matches
    the expected tone based on positive and negative indicators.

    Lifecycle:
        1. Initialization: Set up with expected tone and threshold
        2. Validation: Delegate to validator to analyze text for tone consistency
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.tone import ToneConsistencyRule, ToneValidator, ToneConfig
        from sifaka.rules.base import RuleConfig

        # Create config
        params = {
            "expected_tone": "formal",
            "threshold": 0.8,
            "tone_indicators": {
                "formal": {
                    "positive": ["therefore", "consequently", "furthermore"],
                    "negative": ["yo", "hey", "cool"]
                }
            }
        }
        config = RuleConfig(params=params)

        # Create validator
        validator = ToneValidator(config)

        # Create rule
        rule = ToneConsistencyRule(
            name="tone_consistency_rule",
            description="Checks for tone consistency",
            validator=validator
        )

        # Validate text
        result = (rule and rule.validate("Therefore, we can conclude that the hypothesis is valid.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str = "tone_consistency_rule",
        description: str = "Checks for tone consistency",
        config: Optional[Optional[RuleConfig]] = None,
        validator: Optional[Optional[ToneValidator]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the tone consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name, description=description, rule_id=(kwargs and kwargs.pop("rule_id", name), **kwargs
            ),
            validator=validator,
        )

        # Store the validator for reference
        self._tone_validator = validator or (self and self._create_default_validator()

    def _create_default_validator(self) -> ToneValidator:
        """
        Create a default validator from config.

        Returns:
            A configured ToneValidator
        """
        return ToneValidator(self.config)


def create_tone_consistency_validator(
    expected_tone: str,
    threshold: float = DEFAULT_THRESHOLD,
    tone_indicators: Optional[Dict[str, Dict[str, List[str]]]] = None,
    **kwargs,
) -> ToneValidator:
    """
    Create a tone consistency validator with the specified configuration.

    This factory function creates a configured tone consistency validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        expected_tone: The expected tone style
        threshold: Minimum ratio of positive indicators to pass validation
        tone_indicators: Dictionary of tone indicators for different styles
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured tone consistency validator

    Examples:
        ```python
        from sifaka.rules.content.tone import create_tone_consistency_validator

        # Create a basic validator
        validator = create_tone_consistency_validator(
            expected_tone="formal",
            threshold=0.8
        )

        # Create a validator with custom tone indicators
        validator = create_tone_consistency_validator(
            expected_tone="formal",
            threshold=0.8,
            tone_indicators={
                "formal": {
                    "positive": ["therefore", "consequently", "furthermore"],
                    "negative": ["yo", "hey", "cool"]
                }
            }
        )
        ```
    """
    try:
        config_dict = {
            "expected_tone": expected_tone,
            "threshold": threshold,
            "tone_indicators": tone_indicators or DEFAULT_TONE_INDICATORS,
            **kwargs,
        }

        rule_config = RuleConfig(params=config_dict)
        return ToneValidator(rule_config)

    except Exception as e:
        (logger and logger.error(f"Error creating tone consistency validator: {e}")
        raise ValueError(f"Error creating tone consistency validator: {str(e))")


def create_tone_consistency_rule(
    name: str = "tone_consistency_rule",
    description: str = "Validates text tone consistency",
    expected_tone: str = "formal",
    threshold: float = DEFAULT_THRESHOLD,
    tone_indicators: Optional[Dict[str, Dict[str, List[str]]]] = None,
    rule_id: Optional[Optional[str]] = None,
    **kwargs,
) -> ToneConsistencyRule:
    """
    Create a tone consistency rule with configuration.

    This factory function creates a configured ToneConsistencyRule instance.
    It uses create_tone_consistency_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        expected_tone: The expected tone style
        threshold: Minimum ratio of positive indicators to pass validation
        tone_indicators: Dictionary of tone indicators for different styles
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured ToneConsistencyRule instance

    Examples:
        ```python
        from sifaka.rules.content.tone import create_tone_consistency_rule

        # Create a basic rule
        rule = create_tone_consistency_rule(
            expected_tone="formal",
            threshold=0.8
        )

        # Create a rule with custom tone indicators and metadata
        rule = create_tone_consistency_rule(
            expected_tone="formal",
            threshold=0.8,
            tone_indicators={
                "formal": {
                    "positive": ["therefore", "consequently", "furthermore"],
                    "negative": ["yo", "hey", "cool"]
                }
            },
            name="custom_tone_rule",
            description="Validates text has formal tone",
            rule_id="tone_validator",
            severity="warning",
            category="content",
            tags=["tone", "content", "validation"]
        )
        ```
    """
    try:
        # Create validator using the validator factory
        validator = create_tone_consistency_validator(
            expected_tone=expected_tone,
            threshold=threshold,
            tone_indicators=tone_indicators,
        )

        # Create params dictionary for RuleConfig
        params = {
            "expected_tone": expected_tone,
            "threshold": threshold,
            "tone_indicators": tone_indicators or DEFAULT_TONE_INDICATORS,
        }

        # Determine rule name
        rule_name = name or rule_id or "tone_consistency_rule"

        # Create RuleConfig
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **kwargs,
        )

        # Create and return the rule
        return ToneConsistencyRule(
            name=rule_name,
            description=description,
            config=config,
            validator=validator,
        )

    except Exception as e:
        (logger and logger.error(f"Error creating tone consistency rule: {e}")
        raise ValueError(f"Error creating tone consistency rule: {str(e))")


# Export public classes and functions
__all__ = [
    # Helper classes
    "ToneConfig",
    "ToneAnalyzer",
    # Validator classes
    "ToneValidator",
    # Rule classes
    "ToneConsistencyRule",
    # Validator factory functions
    "create_tone_consistency_validator",
    # Rule factory functions
    "create_tone_consistency_rule",
]
