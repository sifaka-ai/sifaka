"""
Result types for Sifaka operations.

This module defines the result types returned by various Sifaka components,
including validation results, improvement results, and the overall chain result.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    passed: bool
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self.passed


@dataclass
class ImprovementResult:
    """Result of an improvement operation."""

    original_text: str
    improved_text: str
    changes_made: bool
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self.changes_made


@dataclass
class Result:
    """Result of a chain execution."""

    text: str
    passed: bool
    validation_results: List[ValidationResult]
    improvement_results: List[ImprovementResult]
    metadata: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self.passed
