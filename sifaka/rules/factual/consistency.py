"""
Factual consistency validation rules for Sifaka.

This module provides validators and rules for checking factual consistency in text.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The FactualConsistencyConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_factual_consistency_rule, create_factual_consistency_validator) handle configuration

Usage Example:
    from sifaka.rules.factual.consistency import create_factual_consistency_rule

    # Create a factual consistency rule using the factory function
    rule = create_factual_consistency_rule(
        contradiction_indicators=["but", "however", "although"],
        confidence_threshold=0.8
    )

    # Validate text
    result = rule.validate("This is consistent. However, this contradicts the previous statement.")

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = FactualConsistencyRule(
        config=RuleConfig(
            params={
                "contradiction_indicators": ["but", "however", "although"],
                "confidence_threshold": 0.8
            }
        )
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.factual.base import BaseFactualValidator


__all__ = [
    # Config classes
    "FactualConsistencyConfig",
    # Validator classes
    "DefaultFactualConsistencyValidator",
    # Rule classes
    "FactualConsistencyRule",
    # Factory functions
    "create_factual_consistency_validator",
    "create_factual_consistency_rule",
]


@dataclass(frozen=True)
class FactualConsistencyConfig(RuleConfig):
    """Configuration for factual consistency rules."""

    contradiction_indicators: List[str] = field(
        default_factory=lambda: [
            "but",
            "however",
            "although",
            "nevertheless",
            "on the other hand",
            "in contrast",
            "despite",
            "yet",
            "while",
            "whereas",
        ]
    )
    confidence_threshold: float = 0.7
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if not self.contradiction_indicators:
            raise ValueError("Must provide at least one contradiction indicator")


class DefaultFactualConsistencyValidator(BaseFactualValidator):
    """Default implementation of factual consistency validation."""

    def __init__(self, config: FactualConsistencyConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> FactualConsistencyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for factual consistency."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Check for contradictions using indicators
        found_contradictions = []
        for indicator in self.config.contradiction_indicators:
            if indicator.lower() in text.lower():
                found_contradictions.append(indicator)

        # Calculate confidence score based on contradictions
        confidence_score = 1.0 - (
            len(found_contradictions) / len(self.config.contradiction_indicators)
        )
        meets_threshold = confidence_score >= self.config.confidence_threshold

        if not meets_threshold:
            return RuleResult(
                passed=False,
                message=f"Found potential contradictions: {', '.join(found_contradictions)}",
                metadata={
                    "found_contradictions": found_contradictions,
                    "confidence_score": confidence_score,
                    "threshold": self.config.confidence_threshold,
                },
            )

        return RuleResult(
            passed=True,
            message="No contradictions found",
            metadata={
                "found_contradictions": [],
                "confidence_score": confidence_score,
                "threshold": self.config.confidence_threshold,
            },
        )


class FactualConsistencyRule(Rule[str, RuleResult, DefaultFactualConsistencyValidator, Any]):
    """Rule that checks for factual consistency within the text."""

    def __init__(
        self,
        name: str = "factual_consistency_rule",
        description: str = "Checks for factual consistency",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultFactualConsistencyValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the factual consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
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
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultFactualConsistencyValidator:
        """Create a default validator from config."""
        factual_config = FactualConsistencyConfig(**self._rule_params)
        return DefaultFactualConsistencyValidator(factual_config)


def create_factual_consistency_validator(
    contradiction_indicators: Optional[List[str]] = None,
    confidence_threshold: float = 0.7,
    **kwargs,
) -> DefaultFactualConsistencyValidator:
    """
    Create a factual consistency validator with the specified configuration.

    This factory function creates a configured factual consistency validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        contradiction_indicators: List of phrases that indicate contradictions
        confidence_threshold: Minimum confidence score required (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured factual consistency validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if contradiction_indicators is not None:
        config_params["contradiction_indicators"] = contradiction_indicators
    if confidence_threshold is not None:
        config_params["confidence_threshold"] = confidence_threshold

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = FactualConsistencyConfig(**config_params)

    # Return configured validator
    return DefaultFactualConsistencyValidator(config)


def create_factual_consistency_rule(
    name: str = "factual_consistency_rule",
    description: str = "Validates text for factual consistency",
    contradiction_indicators: Optional[List[str]] = None,
    confidence_threshold: float = 0.7,
    **kwargs,
) -> FactualConsistencyRule:
    """
    Create a factual consistency rule with configuration.

    This factory function creates a configured FactualConsistencyRule instance.
    It uses create_factual_consistency_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        contradiction_indicators: List of phrases that indicate contradictions
        confidence_threshold: Minimum confidence score required (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FactualConsistencyRule instance
    """
    # Create validator using the validator factory
    validator = create_factual_consistency_validator(
        contradiction_indicators=contradiction_indicators,
        confidence_threshold=confidence_threshold,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return FactualConsistencyRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
