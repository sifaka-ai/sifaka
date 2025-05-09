"""
Rule configuration for Sifaka.

This module defines the configuration classes for rules in the Sifaka framework.
These classes provide a consistent way to configure rules across the framework.
"""

from enum import Enum, auto
from typing import Any, Dict

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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Priority level for rule execution",
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="Size of the validation cache",
    )
    cost: int = Field(
        default=1,
        ge=0,
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
        """
        new_params = {**self.params, **params}
        return self.with_options(params=new_params)
