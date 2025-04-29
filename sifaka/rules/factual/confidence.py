"""
Confidence validation rules for Sifaka.

This module provides validators and rules for checking confidence indicators in text.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The ConfidenceConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_confidence_rule, create_confidence_validator) handle configuration

Usage Example:
    from sifaka.rules.factual.confidence import create_confidence_rule

    # Create a confidence rule using the factory function
    rule = create_confidence_rule(
        confidence_indicators={
            "high": ["definitely", "certainly", "always"],
            "medium": ["likely", "probably", "usually"],
            "low": ["maybe", "possibly", "sometimes"]
        }
    )

    # Validate text
    result = rule.validate("This is definitely the right approach.")

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = ConfidenceRule(
        config=RuleConfig(
            params={
                "confidence_indicators": {
                    "high": ["definitely", "certainly", "always"],
                    "medium": ["likely", "probably", "usually"],
                    "low": ["maybe", "possibly", "sometimes"]
                }
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
    "ConfidenceConfig",
    # Validator classes
    "DefaultConfidenceValidator",
    # Rule classes
    "ConfidenceRule",
    # Factory functions
    "create_confidence_validator",
    "create_confidence_rule",
]


@dataclass(frozen=True)
class ConfidenceConfig(RuleConfig):
    """Configuration for confidence rules."""

    confidence_indicators: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "high": ["definitely", "certainly", "always", "never", "must", "will"],
            "medium": ["likely", "probably", "usually", "often", "generally"],
            "low": ["maybe", "possibly", "sometimes", "occasionally", "might"],
            "uncertain": ["perhaps", "could", "may", "seems", "appears"],
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.confidence_indicators:
            raise ValueError("Must provide at least one confidence level")
        if not all(indicators for indicators in self.confidence_indicators.values()):
            raise ValueError("Each confidence level must have at least one indicator")


class DefaultConfidenceValidator(BaseFactualValidator):
    """Default implementation of confidence validation."""

    def __init__(self, config: ConfidenceConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> ConfidenceConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for confidence indicators."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        confidence_scores: Dict[str, List[str]] = {}

        # Find indicators for each confidence level
        for level, indicators in self.config.confidence_indicators.items():
            found = [ind for ind in indicators if ind.lower() in text_lower]
            if found:
                confidence_scores[level] = found

        if not confidence_scores:
            return RuleResult(
                passed=False,
                message="No confidence indicators found",
                metadata={"confidence_levels": {}},
            )

        # Determine dominant confidence level
        dominant_level = max(confidence_scores.items(), key=lambda x: len(x[1]))[0]

        return RuleResult(
            passed=True,
            message=f"Found {dominant_level} confidence level",
            metadata={
                "confidence_levels": confidence_scores,
                "dominant_level": dominant_level,
            },
        )


class ConfidenceRule(Rule[str, RuleResult, DefaultConfidenceValidator, Any]):
    """Rule that checks for confidence indicators in the text."""

    def __init__(
        self,
        name: str = "confidence_rule",
        description: str = "Checks for confidence indicators",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultConfidenceValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the confidence rule.

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

    def _create_default_validator(self) -> DefaultConfidenceValidator:
        """Create a default validator from config."""
        confidence_config = ConfidenceConfig(**self._rule_params)
        return DefaultConfidenceValidator(confidence_config)


def create_confidence_validator(
    confidence_indicators: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> DefaultConfidenceValidator:
    """
    Create a confidence validator with the specified configuration.

    This factory function creates a configured confidence validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        confidence_indicators: Dictionary mapping confidence levels to indicator phrases
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured confidence validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if confidence_indicators is not None:
        config_params["confidence_indicators"] = confidence_indicators

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = ConfidenceConfig(**config_params)

    # Return configured validator
    return DefaultConfidenceValidator(config)


def create_confidence_rule(
    name: str = "confidence_rule",
    description: str = "Validates text for confidence indicators",
    confidence_indicators: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> ConfidenceRule:
    """
    Create a confidence rule with configuration.

    This factory function creates a configured ConfidenceRule instance.
    It uses create_confidence_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        confidence_indicators: Dictionary mapping confidence levels to indicator phrases
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured ConfidenceRule instance
    """
    # Create validator using the validator factory
    validator = create_confidence_validator(
        confidence_indicators=confidence_indicators,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return ConfidenceRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
