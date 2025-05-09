"""
Rule protocol for Sifaka.

This module defines the protocol for rules in the Sifaka framework.
The RuleProtocol establishes a common interface that all rule implementations must follow.
"""

from typing import Any, Dict, Protocol, runtime_checkable

from ..config import RuleConfig
from ..result import RuleResult


@runtime_checkable
class RuleProtocol(Protocol):
    """
    Protocol defining the interface for rules.

    This protocol is useful for type checking code that works with rules
    without requiring a specific rule implementation. It defines the minimum
    interface that all rule-like objects must implement to be used in contexts
    that expect rules.

    Lifecycle:
        1. Access: Get rule properties (name, description, config)
        2. Validation: Validate input against rule
        3. Result Handling: Process validation results
    """

    @property
    def name(self) -> str:
        """
        Get the rule name.

        Returns:
            The name of the rule
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the rule description.

        Returns:
            The description of the rule
        """
        ...

    @property
    def config(self) -> RuleConfig:
        """
        Get the rule configuration.

        Returns:
            The configuration of the rule
        """
        ...

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate text against the rule.

        Args:
            text: The text to validate
            **kwargs: Additional validation options

        Returns:
            The validation result
        """
        ...
