"""
Format validator for Sifaka.

This module provides a validator that checks if text follows a specific format.
"""

import json
import re
from typing import Optional, Dict, Any, Callable, Union

from sifaka.results import ValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator


class FormatValidator:
    """Validator that checks if text follows a specific format.

    This validator checks if text follows a specific format, such as JSON, Markdown, etc.

    Attributes:
        format_type: The format type to check for.
        custom_validator: A custom validation function.
        schema: A JSON schema to validate against (for JSON format).
    """

    # Supported format types
    FORMAT_JSON = "json"
    FORMAT_MARKDOWN = "markdown"
    FORMAT_CUSTOM = "custom"

    def __init__(
        self,
        format_type: str,
        custom_validator: Optional[Callable[[str], Dict[str, Any]]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the format validator.

        Args:
            format_type: The format type to check for.
            custom_validator: A custom validation function.
            schema: A JSON schema to validate against (for JSON format).

        Raises:
            ValidationError: If the format type is not supported or if required parameters are missing.
        """
        self.format_type = format_type.lower()
        self.custom_validator = custom_validator
        self.schema = schema

        # Validate parameters
        if self.format_type not in [self.FORMAT_JSON, self.FORMAT_MARKDOWN, self.FORMAT_CUSTOM]:
            raise ValidationError(f"Unsupported format type: {format_type}")

        if self.format_type == self.FORMAT_CUSTOM and self.custom_validator is None:
            raise ValidationError("Custom validator function must be provided for custom format")

        # Import jsonschema if schema is provided
        if self.schema is not None:
            try:
                import jsonschema

                self.jsonschema = jsonschema
            except ImportError:
                raise ValidationError(
                    "jsonschema package is required for schema validation. "
                    "Install it with 'pip install jsonschema'."
                )

    def validate(self, text: str) -> ValidationResult:
        """Validate text against the specified format.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text follows the specified format.
        """
        if not text:
            return ValidationResult(
                passed=False,
                message="Empty text cannot be validated for format",
                details={"format_type": self.format_type},
            )

        if self.format_type == self.FORMAT_JSON:
            return self._validate_json(text)
        elif self.format_type == self.FORMAT_MARKDOWN:
            return self._validate_markdown(text)
        elif self.format_type == self.FORMAT_CUSTOM:
            return self._validate_custom(text)
        else:
            # This should never happen due to validation in __init__
            return ValidationResult(
                passed=False,
                message=f"Unsupported format type: {self.format_type}",
                details={"format_type": self.format_type},
            )

    def _validate_json(self, text: str) -> ValidationResult:
        """Validate text as JSON.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is valid JSON.
        """
        try:
            # Parse the JSON
            data = json.loads(text)

            # Validate against schema if provided
            if self.schema is not None:
                try:
                    self.jsonschema.validate(instance=data, schema=self.schema)
                except self.jsonschema.exceptions.ValidationError as e:
                    return ValidationResult(
                        passed=False,
                        message=f"JSON does not match schema: {str(e)}",
                        details={
                            "format_type": self.FORMAT_JSON,
                            "error": str(e),
                            "schema": self.schema,
                        },
                    )

            # JSON is valid
            return ValidationResult(
                passed=True,
                message="Text is valid JSON",
                details={
                    "format_type": self.FORMAT_JSON,
                    "has_schema": self.schema is not None,
                },
            )
        except json.JSONDecodeError as e:
            # JSON is invalid
            return ValidationResult(
                passed=False,
                message=f"Invalid JSON: {str(e)}",
                details={
                    "format_type": self.FORMAT_JSON,
                    "error": str(e),
                    "line": e.lineno,
                    "column": e.colno,
                },
            )

    def _validate_markdown(self, text: str) -> ValidationResult:
        """Validate text as Markdown.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is valid Markdown.
        """
        # Markdown is very permissive, so we just check for basic structure
        # like headings, lists, code blocks, etc.

        # Check for headings
        has_headings = bool(re.search(r"^#{1,6}\s+.+$", text, re.MULTILINE))

        # Check for lists
        has_lists = bool(re.search(r"^(\*|\-|\+|\d+\.)\s+.+$", text, re.MULTILINE))

        # Check for code blocks
        has_code_blocks = bool(re.search(r"^```.*$", text, re.MULTILINE))

        # Check for links
        has_links = bool(re.search(r"\[.+\]\(.+\)", text))

        # Check for images
        has_images = bool(re.search(r"!\[.+\]\(.+\)", text))

        # Markdown is valid if it has at least one of these features
        markdown_features = {
            "headings": has_headings,
            "lists": has_lists,
            "code_blocks": has_code_blocks,
            "links": has_links,
            "images": has_images,
        }

        has_markdown_features = any(markdown_features.values())

        if has_markdown_features:
            return ValidationResult(
                passed=True,
                message="Text appears to be valid Markdown",
                details={
                    "format_type": self.FORMAT_MARKDOWN,
                    "features": markdown_features,
                },
            )
        else:
            return ValidationResult(
                passed=False,
                message="Text does not appear to contain Markdown features",
                details={
                    "format_type": self.FORMAT_MARKDOWN,
                    "features": markdown_features,
                },
            )

    def _validate_custom(self, text: str) -> ValidationResult:
        """Validate text using a custom validator.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult based on the custom validator's result.
        """
        try:
            # Call the custom validator
            result = self.custom_validator(text)

            # Extract result fields
            passed = result.get("passed", False)
            message = result.get("message", "")
            details = result.get("details", {})

            # Add format type to details
            details["format_type"] = self.FORMAT_CUSTOM

            return ValidationResult(passed=passed, message=message, details=details)
        except Exception as e:
            # Custom validator raised an exception
            return ValidationResult(
                passed=False,
                message=f"Custom validator failed: {str(e)}",
                details={
                    "format_type": self.FORMAT_CUSTOM,
                    "error": str(e),
                },
            )


@register_validator("json_format")
def create_json_format_validator(
    schema: Optional[Dict[str, Any]] = None, **options: Any
) -> FormatValidator:
    """Create a JSON format validator.

    This factory function creates a FormatValidator for JSON format.
    It is registered with the registry system for dependency injection.

    Args:
        schema: A JSON schema to validate against.
        **options: Additional options (ignored).

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=FormatValidator.FORMAT_JSON,
        schema=schema,
    )


def json_format(schema: Optional[Dict[str, Any]] = None) -> FormatValidator:
    """Create a JSON format validator.

    This is a convenience function for creating a FormatValidator for JSON.

    Args:
        schema: A JSON schema to validate against.

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=FormatValidator.FORMAT_JSON,
        schema=schema,
    )


@register_validator("markdown_format")
def create_markdown_format_validator(**options: Any) -> FormatValidator:
    """Create a Markdown format validator.

    This factory function creates a FormatValidator for Markdown format.
    It is registered with the registry system for dependency injection.

    Args:
        **options: Additional options (ignored).

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=FormatValidator.FORMAT_MARKDOWN,
    )


def markdown_format() -> FormatValidator:
    """Create a Markdown format validator.

    This is a convenience function for creating a FormatValidator for Markdown.

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=FormatValidator.FORMAT_MARKDOWN,
    )


@register_validator("custom_format")
def create_custom_format_validator(
    validator: Callable[[str], Dict[str, Any]], **options: Any
) -> FormatValidator:
    """Create a custom format validator.

    This factory function creates a FormatValidator with a custom validator.
    It is registered with the registry system for dependency injection.

    Args:
        validator: A function that takes a string and returns a dictionary with
            "passed" (bool), "message" (str), and "details" (dict) fields.
        **options: Additional options (ignored).

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=FormatValidator.FORMAT_CUSTOM,
        custom_validator=validator,
    )


def custom_format(validator: Callable[[str], Dict[str, Any]]) -> FormatValidator:
    """Create a custom format validator.

    This is a convenience function for creating a FormatValidator with a custom validator.

    Args:
        validator: A function that takes a string and returns a dictionary with
            "passed" (bool), "message" (str), and "details" (dict) fields.

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=FormatValidator.FORMAT_CUSTOM,
        custom_validator=validator,
    )
