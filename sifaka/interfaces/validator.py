"""
Validator Protocol Module

This module defines the protocol for validators in the Sifaka framework.
Validators are components that validate text against defined rules or criteria.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .component import ComponentProtocol


@runtime_checkable
class ValidatorProtocol(ComponentProtocol, Protocol):
    """
    Protocol for validators in Sifaka.

    Validators are components that validate text against defined rules or criteria.
    They analyze text and return validation results indicating whether the text
    meets the specified criteria.
    """

    def validate(self, text: str, **kwargs: Any) -> Any:
        """
        Validate the given text against defined rules or criteria.

        Args:
            text: The text to validate
            **kwargs: Additional arguments for validation

        Returns:
            Validation results
        """
        ...
