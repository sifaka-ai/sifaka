r"""Pattern-based validator for enforcing text structure and content patterns.

This module provides flexible regex-based validation for ensuring text contains
required patterns, avoids forbidden patterns, and meets structural requirements.
Common use cases include validating code blocks, citations, headings, and
domain-specific formatting requirements.

## Key Features:

- **Required Patterns**: Ensure specific patterns are present in text
- **Forbidden Patterns**: Detect and reject unwanted patterns
- **Pattern Counts**: Validate minimum/maximum occurrences of patterns
- **Prebuilt Validators**: Common validators for code, citations, and structure

## Usage Examples:

    >>> # Validate code blocks are present
    >>> code_validator = create_code_validator()
    >>> result = await code_validator.validate(text, sifaka_result)
    >>>
    >>> # Custom pattern validation
    >>> validator = PatternValidator(
    ...     required_patterns={"email": r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"},
    ...     forbidden_patterns={"phone": r"\\d{3}-\\d{3}-\\d{4}"},
    ...     pattern_counts={"email": (1, 3)}  # 1-3 emails required
    ... )

## Common Patterns:

- **URLs**: `r"https?://[^\s]+"`
- **Email addresses**: `r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"`
- **Phone numbers**: `r"\\d{3}-\\d{3}-\\d{4}"`
- **Code blocks**: `r"```[\\w]*\\n[\\s\\S]+?\\n```"`
- **Citations**: `r"\\[\\d+\\]|\\(\\w+,?\\s*\\d{4}\\)"`
- **Headings**: `r"^#+\\s+.+$|^.+\\n[=-]+$"`

## Design Philosophy:

This validator focuses on structural and formatting requirements rather than
semantic content validation. It's ideal for ensuring documents follow specific
formatting standards, contain required elements, or avoid problematic patterns.
"""

import re
from re import Pattern as PatternType
from typing import Dict, Optional

from ..core.interfaces import Validator
from ..core.models import SifakaResult, ValidationResult


class PatternValidator(Validator):
    r"""Validates text against configurable regex patterns.

    Provides flexible pattern-based validation supporting required patterns,
    forbidden patterns, and pattern count requirements. Useful for enforcing
    structural requirements, formatting standards, and content policies.

    Key capabilities:
    - Required patterns that must be present
    - Forbidden patterns that must not be present
    - Pattern count validation (min/max occurrences)
    - Compiled regex patterns for performance
    - Detailed failure reporting with pattern names

    Example:
        >>> # Validate document structure
        >>> validator = PatternValidator(
        ...     required_patterns={
        ...         "heading": r"^#+\s+.+$",
        ...         "email": r"\b[\w.-]+@[\w.-]+\.[a-z]{2,}\b"
        ...     },
        ...     forbidden_patterns={
        ...         "phone": r"\d{3}-\d{3}-\d{4}"
        ...     },
        ...     pattern_counts={
        ...         "heading": (1, 5),  # 1-5 headings required
        ...         "email": (1, None)  # At least 1 email required
        ...     }
        ... )
        >>>
        >>> result = await validator.validate(text, sifaka_result)
        >>> if not result.passed:
        ...     print(f"Validation failed: {result.details}")

    Pattern naming:
        Use descriptive names for patterns to make validation errors clear.
        Names appear in error messages and help users understand requirements.
    """

    def __init__(
        self,
        required_patterns: Optional[Dict[str, str]] = None,
        forbidden_patterns: Optional[Dict[str, str]] = None,
        pattern_counts: Optional[Dict[str, tuple[int, Optional[int]]]] = None,
    ):
        r"""Initialize pattern validator with configurable pattern rules.

        Creates a validator that checks text against regex patterns with
        flexible requirements for presence, absence, and occurrence counts.

        Args:
            required_patterns: Dictionary mapping pattern names to regex strings
                that must be found in valid text. Patterns are compiled with
                MULTILINE flag for line-based matching.
            forbidden_patterns: Dictionary mapping pattern names to regex strings
                that must NOT be found in valid text. Any match causes validation
                failure with sample text in error message.
            pattern_counts: Dictionary mapping pattern names to (min, max) tuples
                specifying required occurrence counts. Use None for max to allow
                unlimited occurrences. Only applies to required_patterns.

        Example:
            >>> # Document structure validator
            >>> validator = PatternValidator(
            ...     required_patterns={
            ...         "title": r"^#\s+.+$",  # Must have h1 title
            ...         "section": r"^##\s+.+$",  # Must have h2 sections
            ...     },
            ...     forbidden_patterns={
            ...         "todo": r"TODO|FIXME|XXX",  # No TODOs allowed
            ...     },
            ...     pattern_counts={
            ...         "title": (1, 1),  # Exactly one title
            ...         "section": (2, None),  # At least 2 sections
            ...     }
            ... )

        Pattern compilation:
            All patterns are compiled at initialization with re.MULTILINE flag
            for consistent line-based matching behavior.

        Performance:
            Patterns are compiled once at initialization for efficient repeated
            validation across multiple texts.
        """
        self.required_patterns: Dict[str, PatternType[str]] = {}
        self.forbidden_patterns: Dict[str, PatternType[str]] = {}
        self.pattern_counts = pattern_counts or {}

        # Compile required patterns
        if required_patterns:
            for name, pattern in required_patterns.items():
                self.required_patterns[name] = re.compile(pattern, re.MULTILINE)

        # Compile forbidden patterns
        if forbidden_patterns:
            for name, pattern in forbidden_patterns.items():
                self.forbidden_patterns[name] = re.compile(pattern, re.MULTILINE)

    @property
    def name(self) -> str:
        """Return the validator identifier.

        Returns:
            "pattern_validator" - used in validation results and error messages
        """
        return "pattern_validator"

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate text against all configured patterns.

        Performs comprehensive pattern validation checking required patterns,
        forbidden patterns, and pattern count requirements. Returns detailed
        results with specific failure information.

        Args:
            text: Text to validate against patterns
            result: SifakaResult for context (not currently used but available)

        Returns:
            ValidationResult with pass/fail status, score, and detailed feedback.
            Score is 1.0 for pass, 0.0 for fail. Details include specific
            pattern failures and success summaries.

        Validation process:
        1. Check all required patterns are present
        2. Verify pattern counts meet min/max requirements
        3. Ensure no forbidden patterns are found
        4. Return detailed results with first 3 issues for readability

        Example:
            >>> result = await validator.validate("# Title\n\nContent here")
            >>> if result.passed:
            ...     print(f"Validation passed: {result.details}")
            ... else:
            ...     print(f"Validation failed: {result.details}")
        """
        issues = []

        # Check required patterns
        for name, pattern in self.required_patterns.items():
            matches = pattern.findall(text)

            if name in self.pattern_counts:
                min_count, max_count = self.pattern_counts[name]
                match_count = len(matches)

                if match_count < min_count:
                    issues.append(
                        f"Pattern '{name}' must occur at least {min_count} times, found {match_count}"
                    )
                elif max_count is not None and match_count > max_count:
                    issues.append(
                        f"Pattern '{name}' must occur at most {max_count} times, found {match_count}"
                    )
            elif not matches:
                issues.append(f"Required pattern '{name}' not found")

        # Check forbidden patterns
        for name, pattern in self.forbidden_patterns.items():
            matches = pattern.findall(text)
            if matches:
                sample = matches[0] if len(matches[0]) < 50 else matches[0][:50] + "..."
                issues.append(f"Forbidden pattern '{name}' found: '{sample}'")

        # Build result
        if issues:
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues[:3]),  # Limit to first 3 issues
            )

        # Calculate score based on pattern matching quality
        total_patterns = len(self.required_patterns) + len(self.forbidden_patterns)
        if total_patterns == 0:
            score = 1.0
            details = "No patterns configured"
        else:
            score = 1.0
            details = f"All {total_patterns} pattern(s) validated successfully"

        return ValidationResult(
            validator=self.name, passed=True, score=score, details=details
        )


# Convenience factory functions for common validation patterns

# These functions create pre-configured PatternValidator instances for
# common document types and requirements. They serve as both useful
# defaults and examples of how to configure the PatternValidator class.


def create_code_validator() -> PatternValidator:
    """Create a pre-configured validator for documents containing code blocks.

    Creates a validator that ensures text contains at least one properly
    formatted code block using markdown triple-backtick syntax.

    Returns:
        PatternValidator configured to require at least one code block

    Example:
        >>> validator = create_code_validator()
        >>>
        >>> # This would pass validation
        >>> text_with_code = '''
        ... Here's some code:
        ... ```python
        ... print("Hello, world!")
        ... ```
        ... '''
        >>>
        >>> # This would fail validation
        >>> text_without_code = "Just plain text"

    Pattern details:
        Matches markdown code blocks with optional language specification:
        - ```python ... ```
        - ``` ... ```
        - ```javascript ... ```
    """
    return PatternValidator(
        required_patterns={
            "code_block": r"```[\w]*\n[\s\S]+?\n```",
        },
        pattern_counts={
            "code_block": (1, None),  # At least one code block
        },
    )


def create_citation_validator() -> PatternValidator:
    """Create a pre-configured validator for academic citations.

    Creates a validator that ensures text contains at least one citation
    in either numbered format [1] or author-year format (Author, 2023).

    Returns:
        PatternValidator configured to require at least one citation

    Example:
        >>> validator = create_citation_validator()
        >>>
        >>> # These would pass validation
        >>> text_with_numbered = "This is supported by research [1]."
        >>> text_with_author_year = "Studies show (Smith, 2023) that..."
        >>>
        >>> # This would fail validation
        >>> text_without_citations = "This is just my opinion."

    Supported citation formats:
        - Numbered: [1], [23], [456]
        - Author-year: (Smith, 2023), (Johnson, 2022)
        - Simple author: (Smith)
    """
    return PatternValidator(
        required_patterns={
            "citation": r"\[\d+\]|\(\w+,?\s*\d{4}\)",  # [1] or (Author, 2023)
        },
        pattern_counts={
            "citation": (1, None),  # At least one citation
        },
    )


def create_structured_validator() -> PatternValidator:
    """Create a pre-configured validator for structured documents.

    Creates a validator that ensures text has proper document structure
    with headings and list items, typical of well-organized documentation.

    Returns:
        PatternValidator configured to require headings and list items

    Example:
        >>> validator = create_structured_validator()
        >>>
        >>> # This would pass validation
        >>> structured_text = '''
        ... # Main Title
        ...
        ... ## Section 1
        ...
        ... - First point
        ... - Second point
        ... - Third point
        ... '''
        >>>
        >>> # This would fail validation
        >>> unstructured_text = "Just a paragraph of text."

    Required structure:
        - At least 1 heading (markdown # or underlined)
        - At least 2 list items (bullet points or numbered)

    Supported heading formats:
        - Markdown: # Title, ## Section, ### Subsection
        - Underlined: Title\n===== or Section\n-----

    Supported list formats:
        - Bullet points: -, *, +, •
        - Numbered lists: 1., 2., 3.
        - Indented lists supported
    """
    return PatternValidator(
        required_patterns={
            "heading": r"^#+\s+.+$|^.+\n[=-]+$",  # Markdown or underline headings
            "list_item": r"^[\s]*[-*+•]\s+.+$|^[\s]*\d+\.\s+.+$",  # Bullet or numbered lists
        },
        pattern_counts={
            "heading": (1, None),  # At least one heading
            "list_item": (2, None),  # At least two list items
        },
    )
