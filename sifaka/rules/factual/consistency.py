"""
Consistency validation rules for Sifaka.

This module provides rules for validating factual consistency in text, including:
- Contradiction detection
- Consistency scoring
- Factual alignment validation

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.factual.consistency import create_consistency_rule

    # Create a consistency rule
    rule = create_consistency_rule(
        contradiction_indicators=[
            "however",
            "but",
            "although",
            "despite",
            "in contrast"
        ],
        threshold=0.7
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


# Default contradiction indicators
DEFAULT_CONTRADICTION_INDICATORS: List[str] = [
    "however",
    "but",
    "although",
    "despite",
    "in contrast",
    "on the other hand",
    "nevertheless",
    "yet",
    "still",
    "nonetheless"
]


class ConsistencyConfig(BaseModel):
    """Configuration for consistency validation."""

    model_config = ConfigDict(frozen=True)

    contradiction_indicators: List[str] = Field(
        default_factory=lambda: DEFAULT_CONTRADICTION_INDICATORS,
        description="List of phrases indicating contradictions",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum consistency score required",
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

    @field_validator("contradiction_indicators")
    @classmethod
    def validate_contradiction_indicators(cls, v: List[str]) -> List[str]:
        """Validate that contradiction indicators are not empty."""
        if not v:
            raise ValueError("Contradiction indicators cannot be empty")
        return v


class DefaultConsistencyValidator(BaseFactualValidator):
    """Default validator for consistency validation."""

    def __init__(self, config: ConsistencyConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(config)
        self._indicators = config.contradiction_indicators
        self._threshold = config.threshold

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for consistency.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        # Count contradiction indicators
        indicator_count = sum(1 for indicator in self._indicators if indicator.lower() in text.lower())
        total_indicators = len(self._indicators)

        # Calculate consistency score
        consistency_score = 1.0 - (indicator_count / total_indicators)
        is_valid = consistency_score >= self._threshold

        return RuleResult(
            is_valid=is_valid,
            score=consistency_score,
            message=f"Consistency score: {consistency_score:.2f} (threshold: {self._threshold})",
        )


class ConsistencyRule(Rule):
    """Rule for validating consistency."""

    def __init__(self, config: ConsistencyConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the rule
        """
        super().__init__(config)
        self._validator = DefaultConsistencyValidator(config)

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for consistency.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        return self._validator.validate(text)


def create_consistency_validator(
    contradiction_indicators: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> DefaultConsistencyValidator:
    """Create a consistency validator.

    Args:
        contradiction_indicators: List of phrases indicating contradictions
        threshold: Minimum consistency score required
        **kwargs: Additional keyword arguments for the config

    Returns:
        DefaultConsistencyValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if contradiction_indicators is not None:
        config_params["contradiction_indicators"] = contradiction_indicators
    if threshold is not None:
        config_params["threshold"] = threshold

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = ConsistencyConfig(**config_params)

    # Create validator
    return DefaultConsistencyValidator(config)


def create_consistency_rule(
    name: str = "consistency_rule",
    description: str = "Validates text for factual consistency",
    contradiction_indicators: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> ConsistencyRule:
    """Create a consistency rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        contradiction_indicators: List of phrases indicating contradictions
        threshold: Minimum consistency score required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        ConsistencyRule: The created rule
    """
    # Create config dictionary
    config_dict = {
        "contradiction_indicators": contradiction_indicators or DEFAULT_CONTRADICTION_INDICATORS,
        "threshold": threshold or 0.7,
        **kwargs,
    }

    # Create config
    config = ConsistencyConfig(**config_dict)

    # Create rule
    return ConsistencyRule(config)
