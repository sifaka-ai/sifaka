"""
Rule configuration for Sifaka.

This module defines the configuration classes for rules in the Sifaka framework.
These classes provide a consistent way to configure rules across the framework.

Usage Example:
    ```python
    from sifaka.rules.config import RuleConfig, RulePriority

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
    ```
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class RulePriority(str, Enum):
    """Priority levels for rule execution."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleConfig(BaseModel):
    """
    Immutable configuration for rules.

    This class provides a consistent way to configure rules across the Sifaka framework.
    It handles common configuration options like priority and caching, while
    allowing rule-specific options through the params dictionary.

    Lifecycle:
        1. Creation: Instantiated with configuration options
        2. Usage: Accessed by rules during setup and validation
        3. Modification: New instances created with updated options (immutable pattern)
        4. Extension: Specialized config classes can extend this base class

    Examples:
        ```python
        from sifaka.rules.config import RuleConfig, RulePriority

        # Create a basic config
        config = RuleConfig(
            name="length_rule",
            description="Validates text length",
            rule_id="length_validator",
            priority=RulePriority.HIGH,
            cache_size=100,
            cost=2,
            params={"min_length": 10, "max_length": 1000}
        )

        # Access configuration
        print(f"Rule: {config.name}, Priority: {config.priority}")
        print(f"Min length: {config.params.get('min_length')}")
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        default="unnamed_rule",
        description="Name of the rule",
    )
    description: str = Field(
        default="",
        description="Description of the rule",
    )
    rule_id: str = Field(
        default="",
        description="Unique identifier for the rule",
    )
    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Priority level for rule execution",
    )
    severity: str = Field(
        default="warning",
        description="Severity level for rule violations",
    )
    category: str = Field(
        default="",
        description="Category of the rule",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorizing the rule",
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="Size of the validation cache",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "RuleConfig":
        """
        Create a new config with updated options.

        This method is useful for updating top-level configuration
        options without modifying the params dictionary.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options

        Examples:
            ```python
            # Create a new config with updated priority
            updated_config = config.with_options(priority=RulePriority.CRITICAL)
            ```
        """
        return RuleConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **params: Any) -> "RuleConfig":
        """
        Create a new config with updated params.

        This method is useful for updating the params dictionary
        without modifying top-level configuration options.

        Args:
            **params: Parameters to update

        Returns:
            New config with updated params

        Examples:
            ```python
            # Create a new config with updated parameters
            updated_config = config.with_params(threshold=0.9, min_length=20)
            ```
        """
        new_params = {**self.params, **params}
        return self.with_options(params=new_params)

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value from the params dictionary.

        Args:
            key: The parameter key to look up
            default: Default value to return if key is not found

        Returns:
            The parameter value or default if not found

        Examples:
            ```python
            # Get a parameter with a default value
            threshold = config.get_param("threshold", 0.5)
            ```
        """
        return self.params.get(key, default)
