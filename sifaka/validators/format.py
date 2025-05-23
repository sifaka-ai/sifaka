"""Format validator for Sifaka.

This module provides a FormatValidator that checks if text follows specific formats
such as JSON, Markdown, or custom formats. It supports schema validation for JSON
and custom validation functions for other formats.

The FormatValidator is designed to ensure that generated text conforms to expected
structural requirements and formatting standards.
"""

import json
import re
import time
from typing import Any, Callable, Dict, Optional

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ValidationError, validation_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class FormatValidator:
    """Validator that checks if text follows a specific format.

    This validator checks if text follows a specific format, such as JSON, Markdown,
    or custom formats. It supports schema validation for JSON and custom validation
    functions for other formats.

    Attributes:
        format_type: The format type to check for.
        custom_validator: A custom validation function.
        schema: A JSON schema to validate against (for JSON format).
        name: The name of the validator.
    """

    # Supported format types
    FORMAT_JSON = "json"
    FORMAT_MARKDOWN = "markdown"
    FORMAT_CUSTOM = "custom"

    def __init__(
        self,
        format_type: str,
        custom_validator: Optional[Callable[[str], bool]] = None,
        schema: Optional[Dict[str, Any]] = None,
        name: str = "FormatValidator",
    ):
        """Initialize the validator.

        Args:
            format_type: The format type to check for ('json', 'markdown', 'custom').
            custom_validator: A custom validation function for custom formats.
            schema: A JSON schema to validate against (for JSON format).
            name: The name of the validator.

        Raises:
            ValidationError: If the format type is unsupported or configuration is invalid.
        """
        if format_type not in [self.FORMAT_JSON, self.FORMAT_MARKDOWN, self.FORMAT_CUSTOM]:
            raise ValidationError(
                message=f"Unsupported format type: {format_type}",
                component="FormatValidator",
                operation="initialization",
                suggestions=[
                    f"Use one of the supported formats: {self.FORMAT_JSON}, {self.FORMAT_MARKDOWN}, {self.FORMAT_CUSTOM}"
                ],
            )

        if format_type == self.FORMAT_CUSTOM and custom_validator is None:
            raise ValidationError(
                message="Custom validator function is required for custom format type",
                component="FormatValidator",
                operation="initialization",
                suggestions=[
                    "Provide a custom_validator function that takes a string and returns bool"
                ],
            )

        self.format_type = format_type
        self.custom_validator = custom_validator
        self.schema = schema
        self.name = name

    def validate(self, thought: Thought) -> ValidationResult:
        """Validate text against format requirements.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.

        Raises:
            ValidationError: If the validation fails due to an error.
        """
        start_time = time.time()

        with validation_context(
            validator_name=self.name,
            operation="format validation",
            message_prefix="Failed to validate text format",
        ):
            # Check if text is available
            if not thought.text:
                return ValidationResult(
                    passed=False,
                    message="No text available for validation",
                    issues=["Text is empty or None"],
                    suggestions=["Provide text to validate"],
                )

            # Dispatch to the appropriate validation method
            if self.format_type == self.FORMAT_JSON:
                result = self._validate_json(thought.text)
            elif self.format_type == self.FORMAT_MARKDOWN:
                result = self._validate_markdown(thought.text)
            elif self.format_type == self.FORMAT_CUSTOM:
                result = self._validate_custom(thought.text)
            else:
                # This should never happen due to validation in __init__
                logger.error(f"{self.name}: Unsupported format type: {self.format_type}")
                result = ValidationResult(
                    passed=False,
                    message=f"Unsupported format type: {self.format_type}",
                    issues=[f"Format type '{self.format_type}' is not supported"],
                    suggestions=["Use a supported format type"],
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"{self.name}: Format validation completed in {processing_time:.2f}ms")

            return result

    def _validate_json(self, text: str) -> ValidationResult:
        """Validate text as JSON format.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is valid JSON.
        """
        try:
            # Parse JSON
            parsed_data = json.loads(text)
            logger.debug(f"{self.name}: JSON parsing successful")

            # Validate against schema if provided
            if self.schema:
                try:
                    import jsonschema

                    jsonschema.validate(parsed_data, self.schema)
                    logger.debug(f"{self.name}: JSON schema validation successful")
                except ImportError:
                    logger.warning("jsonschema library not available for schema validation")
                    return ValidationResult(
                        passed=False,
                        message="JSON schema validation requires jsonschema library",
                        issues=["jsonschema library is not installed"],
                        suggestions=["Install jsonschema: pip install jsonschema"],
                    )
                except jsonschema.ValidationError as e:
                    logger.debug(f"{self.name}: JSON schema validation failed: {e}")
                    return ValidationResult(
                        passed=False,
                        message="JSON does not match the required schema",
                        issues=[f"Schema validation error: {str(e)}"],
                        suggestions=[
                            "Check that the JSON structure matches the required schema",
                            "Verify all required fields are present",
                            "Ensure data types match schema requirements",
                        ],
                    )

            # JSON is valid
            return ValidationResult(
                passed=True,
                message="Text is valid JSON",
                score=1.0,
            )

        except json.JSONDecodeError as e:
            logger.debug(f"{self.name}: JSON parsing failed: {e}")
            return ValidationResult(
                passed=False,
                message=f"Invalid JSON: {str(e)}",
                issues=[f"JSON parsing error: {str(e)}"],
                suggestions=[
                    f"Check line {e.lineno}, column {e.colno} for syntax errors",
                    "Verify that all quotes, brackets, and braces are properly matched",
                    "Ensure that all keys and string values are enclosed in double quotes",
                ],
            )

    def _validate_markdown(self, text: str) -> ValidationResult:
        """Validate text as Markdown format.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is valid Markdown.
        """
        # Basic Markdown validation - check for common Markdown patterns
        issues = []
        suggestions = []

        # Check for unmatched code blocks
        code_block_pattern = r"```"
        code_blocks = re.findall(code_block_pattern, text)
        if len(code_blocks) % 2 != 0:
            issues.append("Unmatched code blocks (``` markers)")
            suggestions.append("Ensure all code blocks have opening and closing ``` markers")

        # Check for unmatched inline code
        inline_code_pattern = r"`"
        inline_codes = re.findall(inline_code_pattern, text)
        if len(inline_codes) % 2 != 0:
            issues.append("Unmatched inline code (` markers)")
            suggestions.append("Ensure all inline code has opening and closing ` markers")

        # Check for valid heading structure
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        lines = text.split("\n")
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                if not re.match(heading_pattern, line.strip()):
                    issues.append(f"Invalid heading format on line {i}")
                    suggestions.append("Headings should have space after # markers")

        if issues:
            return ValidationResult(
                passed=False,
                message="Text contains Markdown formatting issues",
                issues=issues,
                suggestions=suggestions,
            )

        return ValidationResult(
            passed=True,
            message="Text is valid Markdown",
            score=1.0,
        )

    def _validate_custom(self, text: str) -> ValidationResult:
        """Validate text using custom validation function.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text passes custom validation.
        """
        if self.custom_validator is None:
            return ValidationResult(
                passed=False,
                message="No custom validator function provided",
                issues=["Custom validator function is None"],
                suggestions=["Provide a custom validator function"],
            )

        try:
            is_valid = self.custom_validator(text)

            if is_valid:
                return ValidationResult(
                    passed=True,
                    message="Text passes custom format validation",
                    score=1.0,
                )
            else:
                return ValidationResult(
                    passed=False,
                    message="Text fails custom format validation",
                    issues=["Text does not meet custom format requirements"],
                    suggestions=["Modify text to meet the custom format requirements"],
                )

        except Exception as e:
            logger.error(f"{self.name}: Custom validation function failed: {e}")
            return ValidationResult(
                passed=False,
                message=f"Custom validation function error: {str(e)}",
                issues=[f"Validation function raised an exception: {str(e)}"],
                suggestions=["Check the custom validation function implementation"],
            )


def create_format_validator(
    format_type: str,
    custom_validator: Optional[Callable[[str], bool]] = None,
    schema: Optional[Dict[str, Any]] = None,
    name: str = "FormatValidator",
) -> FormatValidator:
    """Create a format validator.

    Args:
        format_type: The format type to check for ('json', 'markdown', 'custom').
        custom_validator: A custom validation function for custom formats.
        schema: A JSON schema to validate against (for JSON format).
        name: The name of the validator.

    Returns:
        A FormatValidator instance.
    """
    return FormatValidator(
        format_type=format_type,
        custom_validator=custom_validator,
        schema=schema,
        name=name,
    )


def json_format(schema: Optional[Dict[str, Any]] = None) -> FormatValidator:
    """Create a JSON format validator.

    Args:
        schema: Optional JSON schema to validate against.

    Returns:
        A FormatValidator configured for JSON validation.
    """
    return create_format_validator(
        format_type=FormatValidator.FORMAT_JSON,
        schema=schema,
        name="JSONFormatValidator",
    )


def markdown_format() -> FormatValidator:
    """Create a Markdown format validator.

    Returns:
        A FormatValidator configured for Markdown validation.
    """
    return create_format_validator(
        format_type=FormatValidator.FORMAT_MARKDOWN,
        name="MarkdownFormatValidator",
    )


def custom_format(validator_func: Callable[[str], bool]) -> FormatValidator:
    """Create a custom format validator.

    Args:
        validator_func: A function that takes a string and returns bool.

    Returns:
        A FormatValidator configured for custom validation.
    """
    return create_format_validator(
        format_type=FormatValidator.FORMAT_CUSTOM,
        custom_validator=validator_func,
        name="CustomFormatValidator",
    )
