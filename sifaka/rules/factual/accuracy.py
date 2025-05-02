"""
Accuracy validation rules for Sifaka.

This module provides rules for validating factual accuracy in text, including:
- Fact verification
- Accuracy scoring
- Knowledge base validation

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.factual.accuracy import create_accuracy_rule

    # Create an accuracy rule
    rule = create_accuracy_rule(
        knowledge_base=[
            "The Earth is round",
            "Water boils at 100°C at sea level",
            "The capital of France is Paris"
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


class AccuracyConfig(BaseModel):
    """Configuration for accuracy validation."""

    model_config = ConfigDict(frozen=True)

    knowledge_base: List[str] = Field(
        default_factory=list,
        description="List of known facts for validation",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy score required",
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

    @field_validator("knowledge_base")
    @classmethod
    def validate_knowledge_base(cls, v: List[str]) -> List[str]:
        """Validate that knowledge base is not empty."""
        if not v:
            raise ValueError("Knowledge base cannot be empty")
        return v


class DefaultAccuracyValidator(BaseFactualValidator):
    """Default validator for accuracy validation."""

    def __init__(self, config: AccuracyConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(config)
        self._knowledge_base = config.knowledge_base
        self._threshold = config.threshold

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for accuracy.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        # Count matching facts
        matching_facts = sum(1 for fact in self._knowledge_base if fact.lower() in text.lower())
        total_facts = len(self._knowledge_base)

        # Calculate accuracy score
        accuracy_score = matching_facts / total_facts if total_facts > 0 else 0.0
        is_valid = accuracy_score >= self._threshold

        return RuleResult(
            is_valid=is_valid,
            score=accuracy_score,
            message=f"Accuracy score: {accuracy_score:.2f} (threshold: {self._threshold})",
        )


class AccuracyRule(Rule):
    """Rule for validating accuracy."""

    def __init__(self, config: AccuracyConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the rule
        """
        super().__init__(config)
        self._validator = DefaultAccuracyValidator(config)

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for accuracy.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        return self._validator.validate(text)


def create_accuracy_validator(
    knowledge_base: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> DefaultAccuracyValidator:
    """Create an accuracy validator.

    Args:
        knowledge_base: List of known facts for validation
        threshold: Minimum accuracy score required
        **kwargs: Additional keyword arguments for the config

    Returns:
        DefaultAccuracyValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if knowledge_base is not None:
        config_params["knowledge_base"] = knowledge_base
    if threshold is not None:
        config_params["threshold"] = threshold

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = AccuracyConfig(**config_params)

    # Create validator
    return DefaultAccuracyValidator(config)


def create_accuracy_rule(
    name: str = "accuracy_rule",
    description: str = "Validates text for factual accuracy",
    knowledge_base: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> AccuracyRule:
    """Create an accuracy rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        knowledge_base: List of known facts for validation
        threshold: Minimum accuracy score required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        AccuracyRule: The created rule
    """
    # Create config dictionary
    config_dict = {
        "knowledge_base": knowledge_base or [],
        "threshold": threshold or 0.8,
        **kwargs,
    }

    # Create config
    config = AccuracyConfig(**config_dict)

    # Create rule
    return AccuracyRule(config)
