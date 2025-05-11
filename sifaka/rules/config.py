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

from sifaka.utils.config import RuleConfig


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
