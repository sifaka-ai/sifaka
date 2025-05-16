"""
Grammar validator for checking grammatical correctness of text.

This module provides a validator that checks text for grammatical errors
using the language_tool_python library.
"""

from typing import Dict, Any, List, Optional
from ..types import ValidationResult


class GrammarValidator:
    """
    Validator that checks text for grammatical correctness.

    This validator uses the language_tool_python library to detect grammar
    and spelling errors in text. It can be configured to focus on different
    types of errors and set thresholds for acceptable error counts.

    By default, it requires the 'language_tool_python' package to be installed.
    Install it with: pip install language-tool-python
    """

    # Error categories to check
    DEFAULT_ERROR_CATEGORIES = [
        "GRAMMAR",
        "PUNCTUATION",
        "SPELLING",
        "TYPOGRAPHY",
    ]

    def __init__(
        self,
        max_errors: Optional[int] = None,
        error_categories: Optional[List[str]] = None,
        ignore_categories: Optional[List[str]] = None,
        ignore_rules: Optional[List[str]] = None,
        language: str = "en-US",
        fail_on_any_error: bool = False,
    ):
        """
        Initialize the grammar validator.

        Args:
            max_errors: Maximum number of errors allowed before failing validation
            error_categories: Specific error categories to check (defaults to DEFAULT_ERROR_CATEGORIES)
            ignore_categories: Error categories to ignore
            ignore_rules: Specific error rules to ignore
            language: Language code to use for checking
            fail_on_any_error: If True, fail on any error regardless of max_errors
        """
        self.max_errors = max_errors
        self.error_categories = error_categories or self.DEFAULT_ERROR_CATEGORIES
        self.ignore_categories = ignore_categories or []
        self.ignore_rules = ignore_rules or []
        self.language = language
        self.fail_on_any_error = fail_on_any_error
        self._tool = None  # Lazy-loaded

    @property
    def tool(self):
        """Lazy-load the LanguageTool instance to avoid unnecessary imports."""
        if self._tool is None:
            try:
                import language_tool_python

                self._tool = language_tool_python.LanguageTool(self.language)
            except ImportError:
                raise ImportError(
                    "language_tool_python is required for GrammarValidator. "
                    "Install it with: pip install language-tool-python"
                )
        return self._tool

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text for grammatical correctness.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with grammar check results
        """
        if not text.strip():
            return ValidationResult(
                passed=True,
                message="Empty text passes grammar validation",
                score=1.0,
                issues=[],
                suggestions=[],
                metadata={"error_count": 0},
            )

        # Check grammar using LanguageTool
        try:
            matches = self.tool.check(text)

            # Filter matches by category if specified
            if self.error_categories:
                matches = [m for m in matches if m.ruleIssueType in self.error_categories]

            # Remove ignored categories and rules
            if self.ignore_categories:
                matches = [m for m in matches if m.ruleIssueType not in self.ignore_categories]
            if self.ignore_rules:
                matches = [m for m in matches if m.ruleId not in self.ignore_rules]

            # Extract useful information from matches
            error_count = len(matches)
            errors = []
            suggestions = []

            for match in matches:
                error = f"{match.ruleIssueType}: {match.message} (at position {match.offset}-{match.offset + match.errorLength})"
                errors.append(error)

                if match.replacements:
                    suggestion = f"Replace '{match.context[match.offsetInContext:match.offsetInContext + match.errorLength]}' with '{match.replacements[0]}'"
                    suggestions.append(suggestion)
                else:
                    suggestion = f"Fix issue: {match.message}"
                    suggestions.append(suggestion)

            # Determine if validation passed
            if self.fail_on_any_error:
                passed = error_count == 0
            elif self.max_errors is not None:
                passed = error_count <= self.max_errors
            else:
                # Default behavior: pass if there are no critical errors
                # (only fail on GRAMMAR and SPELLING by default)
                critical_errors = [m for m in matches if m.ruleIssueType in ["GRAMMAR", "SPELLING"]]
                passed = len(critical_errors) == 0

            # Create message
            if passed:
                if error_count == 0:
                    message = "No grammar errors detected"
                else:
                    message = f"Grammar validation passed with {error_count} minor issues"
            else:
                message = f"Grammar validation failed with {error_count} errors"

            # Calculate score (1.0 = perfect grammar)
            if error_count == 0:
                score = 1.0
            else:
                # Base score on error density (errors per character)
                error_density = error_count / len(text)
                # Typical English text has roughly 1 error per 100-200 chars
                # Consider a density of 0.05 (1 per 20 chars) to be score 0.0
                score = max(0.0, min(1.0, 1.0 - (error_density * 20)))

                # Never return less than 0.1 for passing results
                if passed and score < 0.1:
                    score = 0.1

            return ValidationResult(
                passed=passed,
                message=message,
                score=score,
                issues=errors[:10],  # Limit to top 10 issues
                suggestions=suggestions[:10],  # Limit to top 10 suggestions
                metadata={
                    "error_count": error_count,
                    "categories": {m.ruleIssueType for m in matches},
                    "error_density": error_count / len(text) if text else 0,
                },
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Grammar validation error: {str(e)}",
                score=0.0,
                issues=[f"Error during grammar validation: {str(e)}"],
                suggestions=["Check input text format or try again"],
                metadata={"error": str(e)},
            )
