"""
Confidence validation rules for Sifaka.

This module provides rules for validating confidence indicators in text, including:
- Confidence phrase detection
- Confidence scoring
- Confidence level validation

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.factual.confidence import create_confidence_rule

    # Create a confidence rule
    rule = create_confidence_rule(
        confidence_indicators=[
            "highly confident",
            "very certain",
            "definitely",
            "without a doubt",
            "absolutely"
        ],
        threshold=0.8
    )
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.factual.base import BaseFactualValidator


# Default confidence indicators
DEFAULT_CONFIDENCE_INDICATORS: List[str] = [
    "highly confident",
    "very certain",
    "definitely",
    "without a doubt",
    "absolutely",
    "certainly",
    "undoubtedly",
    "clearly",
    "obviously",
    "evidently"
]


class ConfidenceConfig(BaseModel):
    """Configuration for confidence validation."""

    model_config = ConfigDict(frozen=True)

    confidence_indicators: List[str] = Field(
        default_factory=lambda: DEFAULT_CONFIDENCE_INDICATORS,
        description="List of phrases indicating confidence",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required",
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

    @field_validator("confidence_indicators")
    @classmethod
    def validate_confidence_indicators(cls, v: List[str]) -> List[str]:
        """Validate that confidence indicators are not empty."""
        if not v:
            raise ValueError("Confidence indicators cannot be empty")
        return v


class DefaultConfidenceValidator(BaseFactualValidator):
    """Default validator for confidence validation."""

    def __init__(self, config: ConfidenceConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(config)
        self._indicators = config.confidence_indicators
        self._threshold = config.threshold

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for confidence indicators.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        # Count confidence indicators
        indicator_count = sum(1 for indicator in self._indicators if indicator.lower() in text.lower())
        total_indicators = len(self._indicators)

        # Calculate confidence score
        confidence_score = indicator_count / total_indicators if total_indicators > 0 else 0.0
        is_valid = confidence_score >= self._threshold

        return RuleResult(
            is_valid=is_valid,
            score=confidence_score,
            message=f"Confidence score: {confidence_score:.2f} (threshold: {self._threshold})",
        )


class ConfidenceRule(Rule):
    """Rule for validating confidence."""

    def __init__(self, config: ConfidenceConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the rule
        """
        super().__init__(config)
        self._validator = DefaultConfidenceValidator(config)

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for confidence indicators.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        return self._validator.validate(text)


def create_confidence_validator(
    confidence_indicators: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> DefaultConfidenceValidator:
    """Create a confidence validator.

    Args:
        confidence_indicators: List of phrases indicating confidence
        threshold: Minimum confidence score required
        **kwargs: Additional keyword arguments for the config

    Returns:
        DefaultConfidenceValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if confidence_indicators is not None:
        config_params["confidence_indicators"] = confidence_indicators
    if threshold is not None:
        config_params["threshold"] = threshold

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = ConfidenceConfig(**config_params)

    # Create validator
    return DefaultConfidenceValidator(config)


def create_confidence_rule(
    name: str = "confidence_rule",
    description: str = "Validates text for confidence indicators",
    confidence_indicators: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> ConfidenceRule:
    """Create a confidence rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        confidence_indicators: List of phrases indicating confidence
        threshold: Minimum confidence score required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        ConfidenceRule: The created rule
    """
    # Create config dictionary
    config_dict = {
        "confidence_indicators": confidence_indicators or DEFAULT_CONFIDENCE_INDICATORS,
        "threshold": threshold or 0.8,
        **kwargs,
    }

    # Create config
    config = ConfidenceConfig(**config_dict)

    # Create rule
    return ConfidenceRule(config)
