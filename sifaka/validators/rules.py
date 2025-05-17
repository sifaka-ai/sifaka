"""
Basic validation rules for Sifaka.

This module provides simple, functional validators for common validation tasks.
These validators follow the Validator protocol defined in the chain module.
"""

import re
from typing import Optional, List, Dict, Any, Callable, Union
from re import Pattern

from sifaka.results import ValidationResult


def length(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> Callable[[str], ValidationResult]:
    """Create a validator that checks text length.

    Args:
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.
        min_chars: Minimum number of characters required.
        max_chars: Maximum number of characters allowed.

    Returns:
        A validator function that checks text length.

    Examples:
        >>> validator = length(min_words=10, max_words=100)
        >>> result = validator("This is a short text.")
        >>> result.passed
        False
        >>> result.message
        'Text is too short (5 words, minimum 10)'
    """
    if min_words is None and max_words is None and min_chars is None and max_chars is None:
        raise ValueError("At least one length constraint must be specified")

    def validate(text: str) -> ValidationResult:
        """Validate text length."""
        word_count = len(text.split())
        char_count = len(text)
        details = {"word_count": word_count, "char_count": char_count}

        # Check word count constraints
        if min_words is not None and word_count < min_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too short ({word_count} words, minimum {min_words})",
                details=details,
            )

        if max_words is not None and word_count > max_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too long ({word_count} words, maximum {max_words})",
                details=details,
            )

        # Check character count constraints
        if min_chars is not None and char_count < min_chars:
            return ValidationResult(
                passed=False,
                message=f"Text is too short ({char_count} characters, minimum {min_chars})",
                details=details,
            )

        if max_chars is not None and char_count > max_chars:
            return ValidationResult(
                passed=False,
                message=f"Text is too long ({char_count} characters, maximum {max_chars})",
                details=details,
            )

        # All constraints satisfied
        return ValidationResult(
            passed=True, message="Text length is within constraints", details=details
        )

    return validate


def content(
    required_terms: Optional[List[str]] = None,
    forbidden_terms: Optional[List[str]] = None,
    case_sensitive: bool = False,
) -> Callable[[str], ValidationResult]:
    """Create a validator that checks for required and forbidden terms.

    Args:
        required_terms: Terms that must be present in the text.
        forbidden_terms: Terms that must not be present in the text.
        case_sensitive: Whether to perform case-sensitive matching.

    Returns:
        A validator function that checks text content.

    Examples:
        >>> validator = content(required_terms=["python", "code"], forbidden_terms=["bug"])
        >>> result = validator("This is Python code.")
        >>> result.passed
        True
    """
    if required_terms is None and forbidden_terms is None:
        raise ValueError("At least one of required_terms or forbidden_terms must be specified")

    required_terms = required_terms or []
    forbidden_terms = forbidden_terms or []

    def validate(text: str) -> ValidationResult:
        """Validate text content."""
        if not case_sensitive:
            text = text.lower()
            required_terms_normalized = [term.lower() for term in required_terms]
            forbidden_terms_normalized = [term.lower() for term in forbidden_terms]
        else:
            required_terms_normalized = required_terms
            forbidden_terms_normalized = forbidden_terms

        # Check for required terms
        missing_terms = [term for term in required_terms_normalized if term not in text]

        if missing_terms:
            return ValidationResult(
                passed=False,
                message=f"Text is missing required terms: {', '.join(missing_terms)}",
                details={"missing_terms": missing_terms},
            )

        # Check for forbidden terms
        found_forbidden_terms = [term for term in forbidden_terms_normalized if term in text]

        if found_forbidden_terms:
            return ValidationResult(
                passed=False,
                message=f"Text contains forbidden terms: {', '.join(found_forbidden_terms)}",
                details={"forbidden_terms_found": found_forbidden_terms},
            )

        # All constraints satisfied
        return ValidationResult(
            passed=True,
            message="Text content meets all requirements",
            details={
                "required_terms_found": len(required_terms),
                "forbidden_terms_avoided": len(forbidden_terms),
            },
        )

    return validate


def format(
    pattern: Union[str, Pattern[str]],
    description: Optional[str] = None,
) -> Callable[[str], ValidationResult]:
    """Create a validator that checks if text matches a regex pattern.

    Args:
        pattern: Regular expression pattern to match against the text.
        description: Human-readable description of the expected format.

    Returns:
        A validator function that checks text format.

    Examples:
        >>> validator = format(r"^\d{3}-\d{2}-\d{4}$", "SSN (e.g., 123-45-6789)")
        >>> result = validator("123-45-6789")
        >>> result.passed
        True
    """
    if isinstance(pattern, str):
        compiled_pattern = re.compile(pattern)
    else:
        compiled_pattern = pattern

    format_description = description or f"pattern {compiled_pattern.pattern}"

    def validate(text: str) -> ValidationResult:
        """Validate text format."""
        match = compiled_pattern.search(text)

        if not match:
            return ValidationResult(
                passed=False,
                message=f"Text does not match the required format: {format_description}",
                details={"pattern": compiled_pattern.pattern},
            )

        return ValidationResult(
            passed=True,
            message=f"Text matches the required format: {format_description}",
            details={"pattern": compiled_pattern.pattern, "match": match.group(0)},
        )

    return validate


def json_schema(
    schema: Dict[str, Any],
    extract_json: bool = True,
) -> Callable[[str], ValidationResult]:
    """Create a validator that checks if text contains valid JSON matching a schema.

    Args:
        schema: JSON schema to validate against.
        extract_json: Whether to try to extract JSON from text if it's not valid JSON.

    Returns:
        A validator function that checks JSON content.

    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> validator = json_schema(schema)
        >>> result = validator('{"name": "John"}')
        >>> result.passed
        True
    """
    try:
        import jsonschema
        import json
    except ImportError:
        raise ImportError(
            "JSON schema validation requires jsonschema. "
            "Install it with 'pip install jsonschema'"
        )

    def validate(text: str) -> ValidationResult:
        """Validate text as JSON against schema."""
        # Try to parse the text as JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            if not extract_json:
                return ValidationResult(
                    passed=False,
                    message=f"Text is not valid JSON: {str(e)}",
                    details={"error": str(e)},
                )

            # Try to extract JSON from text
            try:
                # Look for JSON-like patterns
                json_pattern = r"(\{.*\}|\[.*\])"
                match = re.search(json_pattern, text, re.DOTALL)
                if not match:
                    return ValidationResult(
                        passed=False,
                        message="Could not extract JSON from text",
                        details={"error": str(e)},
                    )

                # Try to parse the extracted JSON
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, re.error) as e2:
                return ValidationResult(
                    passed=False,
                    message=f"Could not extract valid JSON from text: {str(e2)}",
                    details={"error": str(e2)},
                )

        # Validate against schema
        try:
            jsonschema.validate(instance=data, schema=schema)
            return ValidationResult(
                passed=True, message="JSON is valid and matches the schema", details={"data": data}
            )
        except jsonschema.exceptions.ValidationError as e:
            return ValidationResult(
                passed=False,
                message=f"JSON does not match the schema: {str(e)}",
                details={"error": str(e), "data": data},
            )

    return validate
