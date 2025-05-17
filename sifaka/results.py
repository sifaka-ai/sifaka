"""
Result types for Sifaka operations.

This module defines the result types returned by various Sifaka components,
including validation results, improvement results, and the overall chain result.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from sifaka.interfaces import ImprovementResult as ImprovementResultProtocol


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        passed: Whether the validation passed
        message: Human-readable message describing the validation result
        details: Additional details about the validation
        score: Normalized score between 0.0 and 1.0 (optional)
        issues: List of identified issues (optional)
        suggestions: List of suggestions for improvement (optional)
    """

    passed: bool
    message: str = ""
    _details: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    issues: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self.passed

    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the validation result."""
        return self._details


@dataclass
class ImprovementResult(ImprovementResultProtocol):
    """Result of an improvement operation.

    Attributes:
        _original_text: The original text before improvement
        _improved_text: The improved text after improvement
        _changes_made: Whether any changes were made
        message: Human-readable message describing the improvement
        _details: Additional details about the improvement
        processing_time_ms: Processing time in milliseconds (optional)
    """

    _original_text: str
    _improved_text: str
    _changes_made: bool
    message: str = ""
    _details: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self._changes_made

    @property
    def passed(self) -> bool:
        """Whether the operation passed."""
        return True  # Improvement operations always "pass" in the sense that they complete

    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the improvement result."""
        return self._details

    @property
    def original_text(self) -> str:
        """The original text before improvement."""
        return self._original_text

    @property
    def improved_text(self) -> str:
        """The improved text after improvement."""
        return self._improved_text

    @property
    def changes_made(self) -> bool:
        """Whether any changes were made to the text."""
        return self._changes_made


@dataclass
class Result:
    """Result of a chain execution.

    Attributes:
        text: The final text after all validations and improvements
        passed: Whether all validations passed
        validation_results: Results of all validations
        improvement_results: Results of all improvements
        metadata: Additional metadata about the execution
        execution_time_ms: Execution time in milliseconds (optional)
    """

    text: str
    passed: bool
    validation_results: List[ValidationResult]
    improvement_results: List[ImprovementResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context."""
        return self.passed

    @property
    def has_issues(self) -> bool:
        """Check if any validation has issues."""
        for result in self.validation_results:
            if result.issues and len(result.issues) > 0:
                return True
        return False

    @property
    def all_issues(self) -> List[str]:
        """Get all issues from all validations."""
        issues = []
        for result in self.validation_results:
            if result.issues:
                issues.extend(result.issues)
        return issues

    @property
    def all_suggestions(self) -> List[str]:
        """Get all suggestions from all validations."""
        suggestions = []
        for result in self.validation_results:
            if result.suggestions:
                suggestions.extend(result.suggestions)
        return suggestions
