"""
Length validator for checking text length constraints.

This module provides a validator that checks if text meets length constraints.
"""

from typing import Optional
from ..types import ValidationResult


class LengthValidator:
    """
    Validator that checks if text meets length constraints.

    This validator checks if text is within specified minimum and maximum
    character length constraints.
    """

    def __init__(self, max_chars: Optional[int] = None, min_chars: Optional[int] = None):
        """
        Initialize the length validator.

        Args:
            max_chars: Maximum allowed length in characters (None for no maximum)
            min_chars: Minimum allowed length in characters (None for no minimum)
        """
        self.max_chars = max_chars
        self.min_chars = min_chars

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against length constraints.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult indicating whether the text meets length constraints
        """
        length = len(text)
        passed = True
        issues = []
        suggestions = []

        # Check maximum length
        if self.max_chars is not None and length > self.max_chars:
            passed = False
            issues.append(f"Text exceeds maximum length of {self.max_chars} characters")
            suggestions.append(f"Shorten the text to be at most {self.max_chars} characters")

        # Check minimum length
        if self.min_chars is not None and length < self.min_chars:
            passed = False
            issues.append(f"Text is shorter than minimum length of {self.min_chars} characters")
            suggestions.append(f"Lengthen the text to be at least {self.min_chars} characters")

        # Create message
        if passed:
            message = "Length validation passed"
        else:
            message = "Length validation failed: " + "; ".join(issues)

        # Create score
        if not passed:
            score = 0.0
        elif self.max_chars is not None and self.min_chars is not None:
            # Score based on how close to ideal length (halfway between min and max)
            ideal_length = (self.min_chars + self.max_chars) / 2
            distance = abs(length - ideal_length) / (self.max_chars - self.min_chars)
            score = max(0.1, 1.0 - (distance / 2))  # Never go below 0.1 if passing
        elif self.max_chars is not None:
            # Score based on how close to max (lower is better, but we still passed)
            score = max(
                0.1, min(1.0, 1.0 - (length / self.max_chars / 2))
            )  # Never go below 0.1 if passing
        elif self.min_chars is not None:
            # Score based on how much we exceed min (higher is better)
            ratio = length / self.min_chars
            score = min(
                1.0, max(0.1, min(ratio / 2, 1.0))
            )  # Cap at 1.0, never below 0.1 if passing
        else:
            score = 1.0

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={"length": length},
        )
