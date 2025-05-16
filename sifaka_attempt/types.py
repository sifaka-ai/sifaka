"""
Common types used throughout the Sifaka library.

This module defines standard result types and other common data structures
used throughout the library.
"""

from typing import Dict, List, Any, Optional


class ValidationResult:
    """
    Standard validation result used throughout the library.

    This class represents the result of validating text against a validator.
    It includes information about whether the validation passed, a message,
    a score, issues found, suggestions for improvement, and additional metadata.
    """

    def __init__(
        self,
        passed: bool,
        message: str = "",
        score: float = 0.0,
        issues: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ValidationResult.

        Args:
            passed: Whether the validation passed
            message: A message describing the validation result
            score: A score from 0.0 to 1.0 representing the quality of the text
            issues: A list of issues found in the text
            suggestions: A list of suggestions for improvement
            metadata: Additional metadata about the validation
        """
        self.passed = passed
        self.message = message
        self.score = score
        self.issues = issues or []
        self.suggestions = suggestions or []
        self.metadata = metadata or {}


class ChainResult:
    """
    Result of running a chain on input text.

    This class represents the result of running a chain on input text,
    including the output text, validation results, and whether all
    validations passed.
    """

    def __init__(
        self,
        output: str,
        validation_results: List[ValidationResult],
        passed: bool,
        original_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ChainResult.

        Args:
            output: The final output text
            validation_results: A list of validation results
            passed: Whether all validations passed
            original_output: The original output before any improvements
            metadata: Additional metadata about the chain execution
        """
        self.output = output
        self.validation_results = validation_results
        self.passed = passed
        self.original_output = original_output
        self.metadata = metadata or {}
