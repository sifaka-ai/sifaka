"""
Length validator for Sifaka.

This module provides a validator that checks if text meets length requirements.
"""

from typing import Optional, Dict, Any

from sifaka.interfaces import ValidationResult
from sifaka.registry import register_validator


class ValidationResultImpl:
    """Implementation of the ValidationResult interface."""
    
    def __init__(
        self,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a validation result.
        
        Args:
            passed: Whether the validation passed.
            message: Message describing the result.
            details: Additional details about the validation result.
        """
        self._passed = passed
        self._message = message
        self._details = details or {}
    
    @property
    def passed(self) -> bool:
        """Whether the validation passed."""
        return self._passed
    
    @property
    def message(self) -> str:
        """Message describing the result."""
        return self._message
    
    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the validation result."""
        return self._details


class LengthValidator:
    """Validator that checks if text meets length requirements.
    
    This validator checks if text is within the specified length range,
    measured in characters, words, or tokens.
    
    Attributes:
        min_length: Minimum length (inclusive).
        max_length: Maximum length (inclusive).
        unit: Unit of measurement ("chars", "words", or "tokens").
    """
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "words",
    ):
        """Initialize a length validator.
        
        Args:
            min_length: Minimum length (inclusive).
            max_length: Maximum length (inclusive).
            unit: Unit of measurement ("chars", "words", or "tokens").
            
        Raises:
            ValueError: If both min_length and max_length are None, or if unit is invalid.
        """
        if min_length is None and max_length is None:
            raise ValueError("At least one of min_length or max_length must be specified")
        
        if unit not in ["chars", "words", "tokens"]:
            raise ValueError("Unit must be one of: chars, words, tokens")
        
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
    
    def validate(self, text: str) -> ValidationResult:
        """Validate text length.
        
        Args:
            text: The text to validate.
            
        Returns:
            A validation result.
        """
        # Calculate length based on unit
        if self.unit == "chars":
            length = len(text)
        elif self.unit == "words":
            length = len(text.split())
        else:  # tokens
            # Simple approximation for tokens (words + punctuation)
            length = len([t for t in text.replace(".", " . ").replace(",", " , ").split() if t])
        
        # Check minimum length
        if self.min_length is not None and length < self.min_length:
            return ValidationResultImpl(
                passed=False,
                message=f"Text is too short: {length} {self.unit} (minimum: {self.min_length})",
                details={
                    "validator": "length",
                    "unit": self.unit,
                    "actual": length,
                    "min": self.min_length,
                    "max": self.max_length,
                },
            )
        
        # Check maximum length
        if self.max_length is not None and length > self.max_length:
            return ValidationResultImpl(
                passed=False,
                message=f"Text is too long: {length} {self.unit} (maximum: {self.max_length})",
                details={
                    "validator": "length",
                    "unit": self.unit,
                    "actual": length,
                    "min": self.min_length,
                    "max": self.max_length,
                },
            )
        
        # Validation passed
        return ValidationResultImpl(
            passed=True,
            message=f"Text length is within range: {length} {self.unit}",
            details={
                "validator": "length",
                "unit": self.unit,
                "actual": length,
                "min": self.min_length,
                "max": self.max_length,
            },
        )


@register_validator("length")
def create_length_validator(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    unit: str = "words",
) -> LengthValidator:
    """Create a length validator.
    
    Args:
        min_length: Minimum length (inclusive).
        max_length: Maximum length (inclusive).
        unit: Unit of measurement ("chars", "words", or "tokens").
        
    Returns:
        A length validator.
        
    Raises:
        ValueError: If both min_length and max_length are None, or if unit is invalid.
    """
    return LengthValidator(min_length, max_length, unit)
