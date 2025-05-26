"""Format validator for Sifaka.

This module provides a FormatValidator that checks if text follows specific formats
such as JSON, Markdown, or custom formats. It supports schema validation for JSON
and custom validation functions for other formats.

The FormatValidator is designed to ensure that generated text conforms to expected
structural requirements and formatting standards.
"""

import json
import re
from typing import Any, Callable, Dict, Optional

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.shared import BaseValidator

# Configure logger
logger = get_logger(__name__)


class FormatValidator(BaseValidator):
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
    FORMAT_EMAIL = "email"
    FORMAT_URL = "url"
    FORMAT_CONTAINS_JSON = "contains_json"

    def __init__(
        self,
        format_type: Optional[str] = None,
        expected_format: Optional[str] = None,
        custom_validator: Optional[Callable[[str], bool]] = None,
        schema: Optional[Dict[str, Any]] = None,
        name: str = "FormatValidator",
    ):
        """Initialize the validator.

        Args:
            format_type: The format type to check for ('json', 'markdown', 'custom').
            expected_format: Alias for format_type.
            custom_validator: A custom validation function for custom formats.
            schema: A JSON schema to validate against (for JSON format).
            name: The name of the validator.

        Raises:
            ValidationError: If the format type is unsupported or configuration is invalid.
        """
        # Handle expected_format as alias for format_type
        if expected_format is not None:
            if format_type is not None:
                raise ValidationError(
                    message="Cannot specify both format_type and expected_format",
                    component="FormatValidator",
                    operation="initialization",
                    suggestions=["Use either format_type or expected_format, not both"],
                )
            format_type = expected_format

        if format_type is None:
            raise ValidationError(
                message="Either format_type or expected_format must be specified",
                component="FormatValidator",
                operation="initialization",
                suggestions=["Provide format_type or expected_format parameter"],
            )

        supported_formats = [
            self.FORMAT_JSON,
            self.FORMAT_MARKDOWN,
            self.FORMAT_CUSTOM,
            self.FORMAT_EMAIL,
            self.FORMAT_URL,
            self.FORMAT_CONTAINS_JSON,
        ]

        if format_type not in supported_formats:
            raise ValidationError(
                message=f"Unsupported format type: {format_type}",
                component="FormatValidator",
                operation="initialization",
                suggestions=[f"Use one of the supported formats: {', '.join(supported_formats)}"],
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

        super().__init__(name)
        self.format_type = format_type
        self.custom_validator = custom_validator
        self.schema = schema

    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text against format requirements.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Check for None text
        if thought.text is None:
            return self.create_empty_text_result(self.name)

        # Dispatch to the appropriate validation method
        if self.format_type == self.FORMAT_JSON:
            return self._validate_json(thought.text)
        elif self.format_type == self.FORMAT_MARKDOWN:
            return self._validate_markdown(thought.text)
        elif self.format_type == self.FORMAT_CUSTOM:
            return self._validate_custom(thought.text)
        elif self.format_type == self.FORMAT_EMAIL:
            return self._validate_email(thought.text)
        elif self.format_type == self.FORMAT_URL:
            return self._validate_url(thought.text)
        elif self.format_type == self.FORMAT_CONTAINS_JSON:
            return self._validate_contains_json(thought.text)
        else:
            # This should never happen due to validation in __init__
            logger.error(f"{self.name}: Unsupported format type: {self.format_type}")
            return self.create_validation_result(
                passed=False,
                message=f"Unsupported format type: {self.format_type}",
                score=0.0,
                issues=[f"Format type '{self.format_type}' is not supported"],
                suggestions=["Use a supported format type"],
            )

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Validate text format asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync validate method but can be called concurrently with other validators.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Format validation is CPU-bound and fast, so we can just call the sync version
        # In a real implementation, you might want to run this in a thread pool for consistency
        return self.validate(thought)

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
                    return self.create_validation_result(
                        passed=False,
                        message="JSON schema validation requires jsonschema library",
                        score=0.0,
                        issues=["jsonschema library is not installed"],
                        suggestions=["Install jsonschema: pip install jsonschema"],
                    )
                except jsonschema.ValidationError as e:
                    logger.debug(f"{self.name}: JSON schema validation failed: {e}")
                    return self.create_validation_result(
                        passed=False,
                        message="JSON does not match the required schema",
                        score=0.0,
                        issues=[f"Schema validation error: {str(e)}"],
                        suggestions=[
                            "Check that the JSON structure matches the required schema",
                            "Verify all required fields are present",
                            "Ensure data types match schema requirements",
                        ],
                    )

            # JSON is valid
            return self.create_validation_result(
                passed=True,
                message="Text is valid JSON",
                score=1.0,
            )

        except json.JSONDecodeError as e:
            logger.debug(f"{self.name}: JSON parsing failed: {e}")
            return self.create_validation_result(
                passed=False,
                message=f"Invalid JSON: {str(e)}",
                score=0.0,
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

    def _validate_email(self, text: str) -> ValidationResult:
        """Validate text as email format.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is a valid email.
        """
        import re

        # Basic email regex pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if re.match(email_pattern, text.strip()):
            return self.create_validation_result(
                passed=True,
                message="Text is a valid email address",
                score=1.0,
            )
        else:
            return self.create_validation_result(
                passed=False,
                message="Text is not a valid email address",
                score=0.0,
                issues=["Invalid email format"],
                suggestions=[
                    "Ensure the email has a valid format: user@domain.com",
                    "Check for missing @ symbol or domain extension",
                ],
            )

    def _validate_url(self, text: str) -> ValidationResult:
        """Validate text as URL format.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is a valid URL.
        """
        import re

        # Basic URL regex pattern
        url_pattern = r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$"

        if re.match(url_pattern, text.strip()):
            return self.create_validation_result(
                passed=True,
                message="Text is a valid URL",
                score=1.0,
            )
        else:
            return self.create_validation_result(
                passed=False,
                message="Text is not a valid URL",
                score=0.0,
                issues=["Invalid URL format"],
                suggestions=[
                    "Ensure the URL starts with http:// or https://",
                    "Check for valid domain format",
                ],
            )

    def _validate_contains_json(self, text: str) -> ValidationResult:
        """Validate that text contains JSON.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text contains valid JSON.
        """
        import re

        # Look for JSON-like patterns in the text
        json_pattern = r"\{[^{}]*\}"
        json_matches = re.findall(json_pattern, text)

        if not json_matches:
            return self.create_validation_result(
                passed=False,
                message="Text does not contain JSON",
                score=0.0,
                issues=["No JSON objects found in text"],
                suggestions=["Include JSON objects in the text"],
            )

        # Try to parse each JSON-like match
        valid_json_found = False
        for match in json_matches:
            try:
                json.loads(match)
                valid_json_found = True
                break
            except json.JSONDecodeError:
                continue

        if valid_json_found:
            return self.create_validation_result(
                passed=True,
                message="Text contains valid JSON",
                score=1.0,
            )
        else:
            return self.create_validation_result(
                passed=False,
                message="Text contains JSON-like patterns but no valid JSON",
                score=0.0,
                issues=["JSON-like patterns found but none are valid JSON"],
                suggestions=["Ensure JSON objects have proper syntax"],
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
