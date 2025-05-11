"""
Rule configuration for Sifaka.

This module imports standardized configuration classes from utils/config.py and
extends them with rule-specific functionality.

Usage Example:
    ```python
    from sifaka.utils.config import RuleConfig, RulePriority, standardize_rule_config

    # Create a basic config
    config = RuleConfig(
        priority=RulePriority.HIGH,
        cache_size=100,
        cost=2,
        params={"threshold": 0.8, "max_length": 1000}
    )

    # Create a new config with updated options
    updated_config = config.with_options(priority=RulePriority.CRITICAL)

    # Create a new config with updated params
    updated_params_config = config.with_params(threshold=0.9, min_length=10)

    # Standardize configuration
    std_config = standardize_rule_config(
        priority=RulePriority.HIGH,
        params={"threshold": 0.8, "max_length": 1000}
    )
    ```
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
    - LOW: Lowest priority level
    - MEDIUM: Default priority level
    - HIGH: High priority level
    - CRITICAL: Highest priority level

    ## Usage
    ```python
    from sifaka.rules.config import RulePriority, RuleConfig

    # Create a rule configuration with HIGH priority
    config = RuleConfig(
        name="important_rule",
        priority=RulePriority.HIGH
    )

    # Priority can also be specified as a string
    config = RuleConfig(
        name="important_rule",
        priority="HIGH"
    )

    # Check priority level
    if config.priority == RulePriority.HIGH:
        print("This is a high-priority rule")
    ```
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def get_param(config: RuleConfig, key: str, default: Any = None) -> Any:
    """
    Get a parameter value from the params dictionary.

    Args:
        config: The rule configuration
        key: The parameter key to look up
        default: Default value to return if key is not found

    Returns:
        The parameter value or default if not found

    Examples:
        ```python
        # Get a parameter with a default value
        threshold = get_param(config, "threshold", 0.5)
        ```
    """
    return config.params.get(key, default)


__all__ = [
    "RuleConfig",
    "RulePriority",
    "get_param",
]
