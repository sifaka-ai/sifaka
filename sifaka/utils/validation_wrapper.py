"""
Validation result wrapper for Sifaka.

This module provides a wrapper for ValidationResult that ensures compatibility
between the Sifaka implementation and the interface protocol.
"""

from typing import Dict, Any, List

from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.interfaces import ValidationResult as ValidationResultProtocol


class ValidationResultWrapper(ValidationResultProtocol):
    """
    Wrapper for ValidationResult that ensures compatibility with the ValidationResult protocol.

    This class wraps a SifakaValidationResult and ensures that it conforms to the
    ValidationResult protocol, handling any differences in property types.
    """

    def __init__(self, result: SifakaValidationResult):
        """
        Initialize the wrapper with a ValidationResult.

        Args:
            result: The ValidationResult to wrap.
        """
        self._result = result

    @property
    def passed(self) -> bool:
        """Whether the validation passed."""
        return self._result.passed if self._result.passed is not None else False

    @property
    def message(self) -> str:
        """Message describing the validation result."""
        return self._result.message if self._result.message is not None else ""

    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the validation result."""
        return self._result.details if self._result.details is not None else {}

    @property
    def score(self) -> float:
        """Validation score (0.0 to 1.0)."""
        return self._result.score if self._result.score is not None else 0.0

    @property
    def issues(self) -> List[str]:
        """List of issues found during validation."""
        return self._result.issues if self._result.issues is not None else []

    @property
    def suggestions(self) -> List[str]:
        """List of suggestions for fixing issues."""
        return self._result.suggestions if self._result.suggestions is not None else []


def wrap_validation_result(result: Any) -> ValidationResultProtocol:
    """
    Wrap a ValidationResult to ensure it conforms to the ValidationResult protocol.

    Args:
        result: The ValidationResult to wrap.

    Returns:
        A ValidationResultWrapper that conforms to the ValidationResult protocol.
    """
    return ValidationResultWrapper(result)
