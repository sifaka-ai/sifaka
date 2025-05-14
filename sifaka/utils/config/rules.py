"""
Rule Configuration Module

This module provides configuration classes and standardization functions for rules.

## Overview
The rule configuration module defines configuration classes for rules in the Sifaka framework.
It provides a consistent approach to configuring rules with standardized parameter handling,
validation, and serialization.

## Components
- **RulePriority**: Enumeration of rule priority levels
- **RuleConfig**: Configuration for rules
- **standardize_rule_config**: Standardization function for rule configurations

## Usage Examples
```python
from sifaka.utils.config.rules import RuleConfig, RulePriority, standardize_rule_config

# Create a rule configuration
config = RuleConfig(
    priority=RulePriority.HIGH,
    params={
        "min_length": 10,
        "max_length": 100
    }
)

# Access configuration values
print(f"Priority: {config.priority}")
print(f"Min length: {config.params.get('min_length') if config.params else None}")

# Use standardization function
config = standardize_rule_config(
    priority="HIGH",
    params={
        "min_length": 10,
        "max_length": 100
    }
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
from pydantic import Field
from .base import BaseConfig

T = TypeVar("T", bound="RuleConfig")


class RulePriority(str, Enum):
    """
    Enumeration of rule priority levels.

    This enumeration defines the possible priority levels for rules in the Sifaka framework.
    Priority levels are used to determine the order in which rules are evaluated.

    ## Usage
    ```python
    from sifaka.utils.config.rules import RulePriority

    # Use as an enum value
    priority = RulePriority.HIGH

    # Use in a rule configuration
    from sifaka.utils.config.rules import RuleConfig
    config = RuleConfig(priority=RulePriority.HIGH)

    # Compare priority levels
    if priority == RulePriority.HIGH:
        print("High priority rule")
    ```

    Attributes:
        LOW: Low priority
        MEDIUM: Medium priority
        HIGH: High priority
        CRITICAL: Critical priority
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleConfig(BaseConfig):
    """
    Configuration for rules.

    This class provides a consistent way to configure rules across the Sifaka framework.
    It handles common configuration options like priority and severity, while
    allowing rule-specific options through the params dictionary.

    ## Architecture
    RuleConfig extends BaseConfig with rule-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during rule initialization and
    remain immutable throughout the rule's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.rules import RuleConfig, RulePriority

    # Create a rule configuration
    config = RuleConfig(
        name="length_rule",
        description="Validates text length",
        priority=RulePriority.HIGH,
        severity="warning",
        category="formatting",
        tags=["length", "formatting"],
        params={
            "min_length": 10,
            "max_length": 100
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Priority: {config.priority}")
    print(f"Min length: {config.params.get('min_length') if config.params else None}")

    # Create a new configuration with updated options
    if config:
        updated_config = config.with_options(priority=RulePriority.MEDIUM)

    # Create a new configuration with updated params
    if config:
        updated_config = config.with_params(min_length=20, max_length=200)
    ```

    Attributes:
        priority: Rule priority level
        severity: Rule severity level
        category: Rule category
        tags: List of tags for the rule
        rule_id: Unique identifier for the rule
        cost: Computational cost of the rule
    """

    priority: RulePriority = Field(default=RulePriority.MEDIUM, description="Rule priority level")
    severity: str = Field(default="info", description="Rule severity level")
    category: str = Field(default="", description="Rule category")
    tags: List[str] = Field(default_factory=list, description="List of tags for the rule")
    rule_id: str = Field(default="", description="Unique identifier for the rule")
    cost: Optional[float] = Field(default=None, description="Computational cost of the rule")


def standardize_rule_config(
    config: Optional[Union[Dict[str, Any], RuleConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = None,  # type: ignore
    **kwargs: Any,
) -> T:
    """
    Standardize rule configuration.

    This utility function ensures that rule configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized RuleConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or RuleConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: RuleConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized RuleConfig object or subclass

    Examples:
        from sifaka.utils.config.rules import standardize_rule_config, RulePriority

        # Create from parameters
        config = standardize_rule_config(
            priority=RulePriority.HIGH,
            severity="warning",
            params={
                "min_length": 10,
                "max_length": 100
            }
        )

        # Create from existing config
        from sifaka.utils.config.rules import RuleConfig
        existing = RuleConfig(priority=RulePriority.MEDIUM)
        updated = standardize_rule_config(
            config=existing,
            params={
                "min_length": 10,
                "max_length": 100
            }
        )

        # Create from dictionary
        dict_config = {
            "priority": "HIGH",
            "severity": "warning",
            "params": {
                "min_length": 10,
                "max_length": 100
            }
        }
        config = standardize_rule_config(config=dict_config)

        # Create specialized config
        # LengthRuleConfig would be defined in a specific rule module
        from sifaka.rules.formatting.length import LengthRuleConfig
        length_config = standardize_rule_config(
            config_class=LengthRuleConfig,
            priority=RulePriority.HIGH,
            params={
                "min_length": 10,
                "max_length": 100
            }
        )
    """
    if config_class is None:
        config_class = RuleConfig  # type: ignore
    final_params: Dict[str, Any] = {}
    if params:
        final_params.update(params)
    if isinstance(config, dict):
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)
        if config and "priority" in config and isinstance(config["priority"], str):
            try:
                config["priority"] = RulePriority(config["priority"])
            except ValueError:
                config["priority"] = RulePriority.MEDIUM
        return cast(
            T, config_class(**{} if config is None else config, params=final_params, **kwargs)
        )
    elif isinstance(config, RuleConfig):
        if config.params:
            final_params.update(config.params)
        config_dict = {**(config.model_dump() if config else {}), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))
    else:
        if "priority" in kwargs and isinstance(kwargs["priority"], str):
            try:
                kwargs["priority"] = RulePriority(kwargs["priority"])
            except ValueError:
                kwargs["priority"] = RulePriority.MEDIUM
        return cast(T, config_class(params=final_params, **kwargs))
