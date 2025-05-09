"""
Factory functions for Sifaka rules.

This module provides factory functions for creating rules and related components.
These functions provide a consistent way to create rules across the framework.
"""

from typing import Any, Dict, Optional, Type

from .base import Rule, BaseValidator
from .config import RuleConfig
from .managers.validation import ValidationManager, ValidationConfig


def create_rule(
    name: str,
    validator: BaseValidator,
    description: Optional[str] = None,
    config: Optional[RuleConfig] = None,
    rule_type: Type[Rule] = Rule,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule with the given validator and configuration.

    This factory function provides a consistent way to create rules
    across the Sifaka framework.

    Args:
        name: Name of the rule
        validator: Validator to use for validation
        description: Description of the rule
        config: Configuration for the rule
        rule_type: Type of rule to create
        **kwargs: Additional arguments for the rule constructor

    Returns:
        A new rule instance
    """
    description = description or f"Rule for {name}"
    config = config or RuleConfig()
    return rule_type(
        name=name, description=description, config=config, validator=validator, **kwargs
    )


def create_validation_manager(
    rules: Optional[list] = None,
    prioritize_by_cost: bool = False,
    **kwargs: Any,
) -> ValidationManager:
    """
    Create a validation manager with the given rules and configuration.

    This factory function provides a consistent way to create validation
    managers across the Sifaka framework.

    Args:
        rules: List of rules to manage
        prioritize_by_cost: Whether to prioritize rules by cost
        **kwargs: Additional configuration parameters

    Returns:
        A new validation manager instance
    """
    config = ValidationConfig(prioritize_by_cost=prioritize_by_cost, params=kwargs)
    return ValidationManager(rules=rules, config=config)
