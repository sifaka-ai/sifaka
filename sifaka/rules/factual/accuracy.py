"""
Factual accuracy validation rules for Sifaka.

This module provides validators and rules for checking factual accuracy in text.

Usage Example:
    from sifaka.rules.factual.accuracy import create_factual_accuracy_rule

    # Create a factual accuracy rule using the factory function
    rule = create_factual_accuracy_rule(
        knowledge_base={
            "earth_shape": {"round", "spherical", "geoid"},
            "gravity": {"9.8 m/s²", "9.8 meters per second squared"}
        }
    )

    # Validate text
    result = rule.validate("The Earth is spherical and gravity is 9.8 m/s².")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.factual.base import BaseFactualValidator


__all__ = [
    # Config classes
    "FactualAccuracyConfig",
    # Validator classes
    "DefaultFactualAccuracyValidator",
    # Rule classes
    "FactualAccuracyRule",
    # Factory functions
    "create_factual_accuracy_validator",
    "create_factual_accuracy_rule",
]


@dataclass(frozen=True)
class FactualAccuracyConfig(RuleConfig):
    """Configuration for factual accuracy rules."""

    knowledge_base: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "earth_shape": {"round", "spherical", "geoid"},
            "gravity": {"9.8 m/s²", "9.8 meters per second squared"},
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.knowledge_base:
            raise ValueError("Must provide at least one knowledge base entry")
        if not all(facts for facts in self.knowledge_base.values()):
            raise ValueError("Each knowledge base entry must have at least one fact")


class DefaultFactualAccuracyValidator(BaseFactualValidator):
    """Default implementation of factual accuracy validation."""

    def __init__(self, config: FactualAccuracyConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> FactualAccuracyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for factual accuracy."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        found_facts: Dict[str, Set[str]] = {}

        # Check each knowledge base entry
        for topic, facts in self.config.knowledge_base.items():
            found = {fact for fact in facts if fact.lower() in text_lower}
            if found:
                found_facts[topic] = found

        if not found_facts:
            return RuleResult(
                passed=True,
                message="No factual claims found to verify",
                metadata={"verified_facts": {}},
            )

        # All found facts are considered accurate since they're from the knowledge base
        return RuleResult(
            passed=True,
            message=f"Verified {len(found_facts)} factual claim(s)",
            metadata={"verified_facts": found_facts},
        )


class FactualAccuracyRule(Rule[str, RuleResult, DefaultFactualAccuracyValidator, Any]):
    """Rule that checks for factual accuracy in the text."""

    def __init__(
        self,
        name: str = "factual_accuracy_rule",
        description: str = "Checks for factual accuracy",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultFactualAccuracyValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the factual accuracy rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultFactualAccuracyValidator:
        """Create a default validator from config."""
        accuracy_config = FactualAccuracyConfig(**self._rule_params)
        return DefaultFactualAccuracyValidator(accuracy_config)


def create_factual_accuracy_validator(
    knowledge_base: Optional[Dict[str, Set[str]]] = None,
    **kwargs,
) -> DefaultFactualAccuracyValidator:
    """
    Create a factual accuracy validator with the specified configuration.

    This factory function creates a configured factual accuracy validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        knowledge_base: Dictionary mapping topics to sets of facts
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured factual accuracy validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if knowledge_base is not None:
        config_params["knowledge_base"] = knowledge_base

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = FactualAccuracyConfig(**config_params)

    # Return configured validator
    return DefaultFactualAccuracyValidator(config)


def create_factual_accuracy_rule(
    name: str = "factual_accuracy_rule",
    description: str = "Validates text for factual accuracy",
    knowledge_base: Optional[Dict[str, Set[str]]] = None,
    **kwargs,
) -> FactualAccuracyRule:
    """
    Create a factual accuracy rule with configuration.

    This factory function creates a configured FactualAccuracyRule instance.
    It uses create_factual_accuracy_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        knowledge_base: Dictionary mapping topics to sets of facts
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FactualAccuracyRule instance
    """
    # Create validator using the validator factory
    validator = create_factual_accuracy_validator(
        knowledge_base=knowledge_base,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return FactualAccuracyRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
