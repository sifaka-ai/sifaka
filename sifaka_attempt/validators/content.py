"""
Content validator for checking text against patterns.

This module provides a validator that checks if text matches or does not match specified patterns.
"""

import re
from typing import Union, List, Optional
from ..types import ValidationResult


class ContentValidator:
    """
    Validator that checks text against patterns.

    This validator checks if text contains or does not contain specific patterns
    using regex matching.
    """

    def __init__(
        self,
        patterns: Union[str, List[str]],
        mode: str = "prohibit",
        case_sensitive: bool = False,
        match_all: bool = False,
        description: Optional[str] = None,
    ):
        """
        Initialize the content validator.

        Args:
            patterns: Pattern or list of patterns to match against
            mode: Mode for matching patterns: 'prohibit' or 'require'
            case_sensitive: Whether to perform case-sensitive matching
            match_all: If True, all patterns must match (require) or none must match (prohibit)
            description: Human-readable description of what the validator checks
        """
        self.patterns = [patterns] if isinstance(patterns, str) else patterns
        self.mode = mode
        self.case_sensitive = case_sensitive
        self.match_all = match_all
        self.description = (
            description or f"Content {'prohibition' if mode == 'prohibit' else 'requirement'}"
        )

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self.compiled_patterns = [re.compile(pattern, flags) for pattern in self.patterns]

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against patterns.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult indicating whether the text meets the criteria
        """
        matches = []
        non_matches = []

        # Check each pattern
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                matches.append(i)
            else:
                non_matches.append(i)

        # Determine if validation passed based on mode and match requirements
        if self.mode == "prohibit":
            # For prohibit mode, we pass if no patterns match (or, with match_all, if not all patterns match)
            if self.match_all:
                passed = len(matches) < len(self.patterns)
            else:
                passed = len(matches) == 0
        else:  # require mode
            # For require mode, we pass if all patterns match (or, without match_all, if any pattern matches)
            if self.match_all:
                passed = len(matches) == len(self.patterns)
            else:
                passed = len(matches) > 0

        # Create issues and suggestions
        issues = []
        suggestions = []

        if self.mode == "prohibit" and not passed:
            for i in matches:
                pattern_desc = self.patterns[i]
                issues.append(f"Text contains prohibited pattern: {pattern_desc}")
                suggestions.append(
                    f"Remove or rephrase content matching the pattern: {pattern_desc}"
                )
        elif self.mode == "require" and not passed:
            for i in non_matches:
                pattern_desc = self.patterns[i]
                issues.append(f"Text does not contain required pattern: {pattern_desc}")
                suggestions.append(f"Add content that matches the pattern: {pattern_desc}")

        # Create message
        if passed:
            message = f"Content validation passed: {self.description}"
        else:
            message = f"Content validation failed: {self.description}"

        # Create score
        if passed:
            score = 1.0
        else:
            # Calculate a score based on how many patterns matched/didn't match
            if self.mode == "prohibit":
                score = max(0.0, 1.0 - (len(matches) / len(self.patterns)))
            else:  # require mode
                score = len(matches) / len(self.patterns)

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "matches": matches,
                "non_matches": non_matches,
                "patterns": self.patterns,
                "mode": self.mode,
            },
        )
