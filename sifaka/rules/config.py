"""
Rule Configuration for Sifaka

This module provides configuration classes and utilities for Sifaka rules,
extending the standardized configuration system from utils/config.py.

## Overview
The rule configuration system provides a consistent way to configure rules
across the Sifaka framework. It defines standard configuration options,
priority levels, and parameter management utilities.

## Components
- RulePriority: Enumeration of standard priority levels for rules
- RuleConfig: Configuration class for rules (imported from utils/config.py)
- get_param: Utility function for retrieving parameters from configuration

## Usage Examples
```python
from sifaka.rules.config import RuleConfig, RulePriority, get_param

# Create a basic rule configuration
config = RuleConfig(
    name="length_rule",
    description="Validates text length",
    priority=RulePriority.HIGH,
    severity="warning",
    category="formatting",
    tags=["length", "formatting"],
    cache_size=100,
    cost=2,
    params={"threshold": 0.8, "max_length": 1000}
)

# Create a new config with updated options
updated_config = config.with_options(priority=RulePriority.CRITICAL)

# Create a new config with updated params
updated_params_config = config.with_params(threshold=0.9, min_length=10)

# Get parameters with default values
threshold = get_param(config, "threshold", 0.5)
max_length = get_param(config, "max_length", 500)
```

## Configuration Options
- name: Name of the rule
- description: Description of the rule
- priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
- severity: Severity level for rule violations (error, warning, info)
- category: Category of the rule (formatting, content, etc.)
- tags: List of tags for categorizing the rule
- cache_size: Size of the validation cache
- cost: Computational cost of validation
- params: Dictionary of rule-specific parameters

## Parameter Management
The params dictionary in RuleConfig allows for storing rule-specific parameters.
The get_param utility function provides a convenient way to retrieve parameters
with default values.
"""

from typing import Any
from enum import Enum

from sifaka.utils.config import RuleConfig


class RulePriority(str, Enum):
    """
    Priority levels for rules.

    This enumeration defines the standard priority levels for rules in the Sifaka framework.
    Rules with higher priority are typically executed before rules with lower priority.

    ## Values
    - LOW: Lowest priority level, used for optional or cosmetic rules
    - MEDIUM: Default priority level, used for standard rules
    - HIGH: High priority level, used for important rules
    - CRITICAL: Highest priority level, used for essential rules

    ## Usage Examples
    ```python
    from sifaka.rules.config import RulePriority, RuleConfig

    # Create a rule configuration with HIGH priority
    config = RuleConfig(
        name="important_rule",
        description="An important validation rule",
        priority=RulePriority.HIGH
    )

    # Priority can also be specified as a string
    config = RuleConfig(
        name="important_rule",
        description="An important validation rule",
        priority="HIGH"
    )

    # Check priority level
    if config.priority == RulePriority.HIGH:
        print("This is a high-priority rule")

    # Use priority for rule ordering
    rules = [rule1, rule2, rule3]
    high_priority_rules = [r for r in rules if r.config.priority == RulePriority.HIGH]
    ```

    ## Implementation Notes
    The RulePriority enum inherits from str to allow for string-based comparison
    and serialization. This makes it easy to use in configuration files and APIs.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def get_param(config: RuleConfig, key: str, default: Any = None) -> Any:
    """
    Get a parameter value from the params dictionary.

    This utility function provides a convenient way to retrieve parameters
    from a rule configuration with default values. It safely handles missing
    keys by returning the specified default value.

    Args:
        config: The rule configuration
        key: The parameter key to look up
        default: Default value to return if key is not found

    Returns:
        The parameter value or default if not found

    Examples:
        ```python
        from sifaka.rules.config import get_param, RuleConfig

        # Create a configuration with parameters
        config = RuleConfig(
            name="length_rule",
            description="Validates text length",
            params={
                "min_length": 10,
                "max_length": 1000,
                "threshold": 0.8
            }
        )

        # Get parameters with default values
        min_length = get_param(config, "min_length", 5)
        max_length = get_param(config, "max_length", 500)
        threshold = get_param(config, "threshold", 0.5)

        # Get a parameter that doesn't exist (will use default)
        format_type = get_param(config, "format_type", "markdown")
        ```
    """
    return config.params.get(key, default)


__all__ = [
    "RuleConfig",
    "RulePriority",
    "get_param",
]
