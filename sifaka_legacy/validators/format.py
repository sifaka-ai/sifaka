"""
Format validator for Sifaka.

This module provides a validator that checks if text follows a specific format.
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, Optional

from sifaka.errors import ValidationError
from sifaka.registry import register_validator
from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.utils.error_handling import log_error, validation_context
from sifaka.validators.base import BaseValidator

# Configure logger
logger = logging.getLogger(__name__)


class FormatValidator(BaseValidator):
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
        name: Optional[str] = None,
    ):
        """Initialize the format validator.

        Args:
            format_type: The format type to check for.
            custom_validator: A custom validation function.
            schema: A JSON schema to validate against (for JSON format).
            name: Optional name for the validator.

        Raises:
            ValidationError: If the format type is not supported or if required parameters are missing.
        """
        # Initialize the base validator with a name
        super().__init__(name=name or f"FormatValidator_{format_type.lower()}")

        # Store configuration
        self.format_type = format_type.lower()
        self.custom_validator = custom_validator
        self.schema = schema

        # Log initialization
        logger.debug(
            f"Initialized {self.name} with format_type={self.format_type}, "
            f"has_custom_validator={self.custom_validator is not None}, "
            f"has_schema={self.schema is not None}"
        )

        # Validate parameters with improved error handling
        with validation_context(
            validator_name=self.name,
            operation="initialization",
            message_prefix="Failed to initialize format validator",
            suggestions=["Check the format type and parameters"],
            metadata={
                "format_type": self.format_type,
                "has_custom_validator": self.custom_validator is not None,
                "has_schema": self.schema is not None,
            },
        ):
            # Validate format type
            if self.format_type not in [
                self.FORMAT_JSON,
                self.FORMAT_MARKDOWN,
                self.FORMAT_CUSTOM,
            ]:
                logger.error(f"{self.name}: Unsupported format type: {format_type}")
                raise ValidationError(
                    message=f"Unsupported format type: {format_type}",
                    component="FormatValidator",
                    operation="initialization",
                    suggestions=[
                        f"Use one of the supported format types: {self.FORMAT_JSON}, {self.FORMAT_MARKDOWN}, {self.FORMAT_CUSTOM}"
                    ],
                    metadata={
                        "format_type": format_type,
                        "supported_types": [
                            self.FORMAT_JSON,
                            self.FORMAT_MARKDOWN,
                            self.FORMAT_CUSTOM,
                        ],
                    },
                )

            # Validate custom validator
            if self.format_type == self.FORMAT_CUSTOM and self.custom_validator is None:
                logger.error(
                    f"{self.name}: Custom validator function must be provided for custom format"
                )
                raise ValidationError(
                    message="Custom validator function must be provided for custom format",
                    component="FormatValidator",
                    operation="initialization",
                    suggestions=[
                        "Provide a custom validator function when using the custom format type",
                        "Use a different format type if you don't have a custom validator",
                    ],
                    metadata={"format_type": self.format_type},
                )

            # Import jsonschema if schema is provided
            if self.schema is not None:
                try:
                    import jsonschema

                    self.jsonschema = jsonschema
                    logger.debug(f"{self.name}: Successfully imported jsonschema package")
                except ImportError as e:
                    logger.error(f"{self.name}: Failed to import jsonschema package: {str(e)}")
                    raise ValidationError(
                        message="jsonschema package is required for schema validation. Install it with 'pip install jsonschema'.",
                        component="FormatValidator",
                        operation="initialization",
                        suggestions=[
                            "Install the jsonschema package with 'pip install jsonschema'",
                            "Remove the schema parameter if you don't need schema validation",
                        ],
                        metadata={
                            "format_type": self.format_type,
                            "error_type": "ImportError",
                            "error_message": str(e),
                        },
                    )

    def _validate(self, text: str) -> SifakaValidationResult:
        """Validate text against the specified format.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text follows the specified format.
        """
        start_time = time.time()

        # Log validation attempt
        logger.debug(
            f"{self.name}: Validating text of length {len(text)} against format type {self.format_type}"
        )

        # Validate text against the specified format
        with validation_context(
            validator_name=self.name,
            operation="validation",
            message_prefix=f"Failed to validate text against {self.format_type} format",
            suggestions=[f"Check if the text follows the {self.format_type} format"],
            metadata={
                "format_type": self.format_type,
                "text_length": len(text),
                "has_schema": self.schema is not None,
                "has_custom_validator": self.custom_validator is not None,
            },
        ):
            # Dispatch to the appropriate validation method
            if self.format_type == self.FORMAT_JSON:
                result = self._validate_json(text)
            elif self.format_type == self.FORMAT_MARKDOWN:
                result = self._validate_markdown(text)
            elif self.format_type == self.FORMAT_CUSTOM:
                result = self._validate_custom(text)
            else:
                # This should never happen due to validation in __init__
                logger.error(f"{self.name}: Unsupported format type: {self.format_type}")

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SifakaValidationResult(
                    passed=False,
                    message=f"Unsupported format type: {self.format_type}",
                    _details={
                        "format_type": self.format_type,
                        "validator_name": self.name,
                        "processing_time_ms": processing_time,
                    },
                    score=0.0,
                    issues=[f"Unsupported format type: {self.format_type}"],
                    suggestions=[
                        f"Use one of the supported format types: {self.FORMAT_JSON}, {self.FORMAT_MARKDOWN}, {self.FORMAT_CUSTOM}"
                    ],
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Add processing time to result details
            result._details = result._details or {}
            result._details["processing_time_ms"] = processing_time
            result._details["validator_name"] = self.name

            # Log validation result
            logger.debug(
                f"{self.name}: Validation {'passed' if result.passed else 'failed'} in {processing_time:.2f}ms"
            )

            return result

    def _validate_json(self, text: str) -> SifakaValidationResult:
        """Validate text as JSON.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is valid JSON.
        """
        # Log JSON validation attempt
        logger.debug(f"{self.name}: Validating text as JSON, length={len(text)}")

        try:
            # Parse the JSON
            with validation_context(
                validator_name=self.name,
                operation="json_parsing",
                message_prefix="Failed to parse JSON",
                suggestions=["Check if the text is valid JSON"],
                metadata={
                    "text_length": len(text),
                    "has_schema": self.schema is not None,
                },
            ):
                data = json.loads(text)
                logger.debug(f"{self.name}: Successfully parsed JSON")

            # Validate against schema if provided
            if self.schema is not None:
                try:
                    with validation_context(
                        validator_name=self.name,
                        operation="schema_validation",
                        message_prefix="Failed to validate JSON against schema",
                        suggestions=["Check if the JSON matches the schema"],
                        metadata={"text_length": len(text), "schema": self.schema},
                    ):
                        self.jsonschema.validate(instance=data, schema=self.schema)
                        logger.debug(f"{self.name}: JSON successfully validated against schema")
                except self.jsonschema.exceptions.ValidationError as e:
                    # Log schema validation error
                    log_error(
                        e,
                        logger,
                        component="FormatValidator",
                        operation="schema_validation",
                    )

                    # Create issues and suggestions
                    issues = [f"JSON does not match schema: {str(e)}"]
                    suggestions = [
                        "Check if the JSON structure matches the required schema",
                        "Verify that all required fields are present",
                        "Ensure field types match the schema requirements",
                    ]

                    # Calculate score based on severity of schema violation
                    score = 0.5  # Partial score for valid JSON but schema mismatch

                    return SifakaValidationResult(
                        passed=False,
                        message=f"JSON does not match schema: {str(e)}",
                        _details={
                            "format_type": self.FORMAT_JSON,
                            "error": str(e),
                            "schema": self.schema,
                            "validator_name": self.name,
                        },
                        score=score,
                        issues=issues,
                        suggestions=suggestions,
                    )

            # JSON is valid
            logger.debug(f"{self.name}: JSON is valid")

            return SifakaValidationResult(
                passed=True,
                message="Text is valid JSON",
                _details={
                    "format_type": self.FORMAT_JSON,
                    "has_schema": self.schema is not None,
                    "validator_name": self.name,
                },
                score=1.0,
                issues=[],
                suggestions=[],
            )

        except json.JSONDecodeError as e:
            # Log JSON parsing error
            log_error(e, logger, component="FormatValidator", operation="json_parsing")

            # Create issues and suggestions
            issues = [f"Invalid JSON: {str(e)}"]
            suggestions = [
                f"Check line {e.lineno}, column {e.colno} for syntax errors",
                "Verify that all quotes, brackets, and braces are properly matched",
                "Ensure that all keys and string values are enclosed in double quotes",
            ]

            # Calculate score based on how much of the text was parsed before error
            score = max(0.0, min(1.0, e.pos / len(text)))

            return SifakaValidationResult(
                passed=False,
                message=f"Invalid JSON: {str(e)}",
                _details={
                    "format_type": self.FORMAT_JSON,
                    "error": str(e),
                    "line": e.lineno,
                    "column": e.colno,
                    "position": e.pos,
                    "validator_name": self.name,
                },
                score=score,
                issues=issues,
                suggestions=suggestions,
            )

    def _validate_markdown(self, text: str) -> SifakaValidationResult:
        """Validate text as Markdown.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text is valid Markdown.
        """
        # Log Markdown validation attempt
        logger.debug(f"{self.name}: Validating text as Markdown, length={len(text)}")

        try:
            # Markdown is very permissive, so we just check for basic structure
            # like headings, lists, code blocks, etc.
            with validation_context(
                validator_name=self.name,
                operation="markdown_validation",
                message_prefix="Failed to validate Markdown",
                suggestions=["Check if the text contains Markdown features"],
                metadata={"text_length": len(text)},
            ):
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

                # Check for emphasis (bold, italic)
                has_emphasis = bool(re.search(r"(\*\*|__).+(\*\*|__)|(\*|_).+(\*|_)", text))

                # Check for blockquotes
                has_blockquotes = bool(re.search(r"^>\s+.+$", text, re.MULTILINE))

                # Check for horizontal rules
                has_hr = bool(re.search(r"^(\*\*\*|---|\*\*\*\*\*|_____)$", text, re.MULTILINE))

                # Markdown is valid if it has at least one of these features
                markdown_features = {
                    "headings": has_headings,
                    "lists": has_lists,
                    "code_blocks": has_code_blocks,
                    "links": has_links,
                    "images": has_images,
                    "emphasis": has_emphasis,
                    "blockquotes": has_blockquotes,
                    "horizontal_rules": has_hr,
                }

                has_markdown_features = any(markdown_features.values())
                feature_count = sum(1 for feature in markdown_features.values() if feature)

                # Calculate score based on number of features
                score = min(1.0, feature_count / 4)  # Perfect score if 4+ features

                if has_markdown_features:
                    logger.debug(
                        f"{self.name}: Text appears to be valid Markdown with {feature_count} features"
                    )

                    return SifakaValidationResult(
                        passed=True,
                        message="Text appears to be valid Markdown",
                        _details={
                            "format_type": self.FORMAT_MARKDOWN,
                            "features": markdown_features,
                            "feature_count": feature_count,
                            "validator_name": self.name,
                        },
                        score=score,
                        issues=[],
                        suggestions=[],
                    )
                else:
                    logger.debug(f"{self.name}: Text does not appear to contain Markdown features")

                    # Create issues and suggestions
                    issues = ["Text does not appear to contain Markdown features"]
                    suggestions = [
                        "Add headings using # syntax",
                        "Include lists using * or - or 1. syntax",
                        "Add code blocks using ``` syntax",
                        "Include links using [text](url) syntax",
                        "Add emphasis using *italic* or **bold** syntax",
                    ]

                    return SifakaValidationResult(
                        passed=False,
                        message="Text does not appear to contain Markdown features",
                        _details={
                            "format_type": self.FORMAT_MARKDOWN,
                            "features": markdown_features,
                            "feature_count": 0,
                            "validator_name": self.name,
                        },
                        score=0.0,
                        issues=issues,
                        suggestions=suggestions,
                    )

        except Exception as e:
            # Log Markdown validation error
            log_error(e, logger, component="FormatValidator", operation="markdown_validation")

            # Create issues and suggestions
            issues = [f"Error validating Markdown: {str(e)}"]
            suggestions = [
                "Check if the text is properly formatted",
                "Verify that the text doesn't contain invalid characters",
            ]

            return SifakaValidationResult(
                passed=False,
                message=f"Error validating Markdown: {str(e)}",
                _details={
                    "format_type": self.FORMAT_MARKDOWN,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_name": self.name,
                },
                score=0.0,
                issues=issues,
                suggestions=suggestions,
            )

    def _validate_custom(self, text: str) -> SifakaValidationResult:
        """Validate text using a custom validator.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult based on the custom validator's result.
        """
        # Log custom validation attempt
        logger.debug(f"{self.name}: Validating text with custom validator, length={len(text)}")

        try:
            # Call the custom validator with error handling
            with validation_context(
                validator_name=self.name,
                operation="custom_validation",
                message_prefix="Failed to validate with custom validator",
                suggestions=["Check if the custom validator is properly implemented"],
                metadata={
                    "text_length": len(text),
                    "validator_function": str(self.custom_validator),
                },
            ):
                # Call the custom validator
                if self.custom_validator is None:
                    raise ValidationError(
                        message="Custom validator is None",
                        component="FormatValidator",
                        operation="custom_validation",
                        suggestions=[
                            "Provide a custom validator function when using the custom format type"
                        ],
                        metadata={"format_type": self.format_type},
                    )
                result = self.custom_validator(text)
                logger.debug(f"{self.name}: Custom validator executed successfully")

            # Extract result fields
            passed = result.get("passed", False)
            message = result.get("message", "")
            details = result.get("details", {})
            issues = result.get("issues", [])
            suggestions = result.get("suggestions", [])
            score = result.get("score", 1.0 if passed else 0.0)

            # Add format type and validator name to details
            details["format_type"] = self.FORMAT_CUSTOM
            details["validator_name"] = self.name

            # Log validation result
            logger.debug(
                f"{self.name}: Custom validation {'passed' if passed else 'failed'}: {message}"
            )

            return SifakaValidationResult(
                passed=passed,
                message=message,
                _details=details,
                score=score,
                issues=issues,
                suggestions=suggestions,
            )

        except Exception as e:
            # Log custom validation error
            log_error(e, logger, component="FormatValidator", operation="custom_validation")

            # Create issues and suggestions
            issues = [f"Custom validator failed: {str(e)}"]
            suggestions = [
                "Check if the custom validator is properly implemented",
                "Verify that the custom validator handles the input correctly",
                "Ensure that the custom validator returns the expected format",
            ]

            return SifakaValidationResult(
                passed=False,
                message=f"Custom validator failed: {str(e)}",
                _details={
                    "format_type": self.FORMAT_CUSTOM,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_name": self.name,
                },
                score=0.0,
                issues=issues,
                suggestions=suggestions,
            )


@register_validator("json_format")
def create_json_format_validator(
    schema: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **options: Any
) -> FormatValidator:
    """Create a JSON format validator.

    This factory function creates a FormatValidator for JSON format.
    It is registered with the registry system for dependency injection.

    Args:
        schema: A JSON schema to validate against.
        name: Optional name for the validator.
        **options: Additional options (ignored).

    Returns:
        A FormatValidator instance.

    Raises:
        ValidationError: If the validator cannot be created.
    """
    try:
        # Log factory function call
        logger.debug(f"Creating JSON format validator with schema={schema is not None}")

        # Create the validator
        validator = FormatValidator(
            format_type=FormatValidator.FORMAT_JSON,
            schema=schema,
            name=name or options.get("name") or "JSONFormatValidator",
        )

        # Log successful creation
        logger.debug(f"Successfully created JSON format validator: {validator.name}")

        return validator

    except Exception as e:
        # Log the error
        log_error(
            e,
            logger,
            component="JSONFormatValidatorFactory",
            operation="create_validator",
        )

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create JSON format validator: {str(e)}",
                component="JSONFormatValidatorFactory",
                operation="create_validator",
                suggestions=[
                    "Check if the schema is valid",
                    "Verify that the jsonschema package is installed if using a schema",
                ],
                metadata={
                    "has_schema": schema is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        raise


def json_format(
    schema: Optional[Dict[str, Any]] = None, name: Optional[str] = None
) -> FormatValidator:
    """Create a JSON format validator.

    This is a convenience function for creating a FormatValidator for JSON.

    Args:
        schema: A JSON schema to validate against.
        name: Optional name for the validator.

    Returns:
        A FormatValidator instance.

    Raises:
        ValidationError: If the validator cannot be created.
    """
    try:
        # Create the validator
        return FormatValidator(
            format_type=FormatValidator.FORMAT_JSON,
            schema=schema,
            name=name or "JSONFormatValidator",
        )

    except Exception as e:
        # Log the error
        log_error(e, logger, component="JSONFormatValidatorFactory", operation="json_format")

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create JSON format validator: {str(e)}",
                component="JSONFormatValidatorFactory",
                operation="json_format",
                suggestions=[
                    "Check if the schema is valid",
                    "Verify that the jsonschema package is installed if using a schema",
                ],
                metadata={
                    "has_schema": schema is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        raise


@register_validator("markdown_format")
def create_markdown_format_validator(name: Optional[str] = None, **options: Any) -> FormatValidator:
    """Create a Markdown format validator.

    This factory function creates a FormatValidator for Markdown format.
    It is registered with the registry system for dependency injection.

    Args:
        name: Optional name for the validator.
        **options: Additional options (ignored).

    Returns:
        A FormatValidator instance.

    Raises:
        ValidationError: If the validator cannot be created.
    """
    try:
        # Log factory function call
        logger.debug("Creating Markdown format validator")

        # Create the validator
        validator = FormatValidator(
            format_type=FormatValidator.FORMAT_MARKDOWN,
            name=name or options.get("name") or "MarkdownFormatValidator",
        )

        # Log successful creation
        logger.debug(f"Successfully created Markdown format validator: {validator.name}")

        return validator

    except Exception as e:
        # Log the error
        log_error(
            e,
            logger,
            component="MarkdownFormatValidatorFactory",
            operation="create_validator",
        )

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create Markdown format validator: {str(e)}",
                component="MarkdownFormatValidatorFactory",
                operation="create_validator",
                suggestions=["Check the error message for details"],
                metadata={"error_type": type(e).__name__, "error_message": str(e)},
            )
        raise


def markdown_format(name: Optional[str] = None) -> FormatValidator:
    """Create a Markdown format validator.

    This is a convenience function for creating a FormatValidator for Markdown.

    Args:
        name: Optional name for the validator.

    Returns:
        A FormatValidator instance.

    Raises:
        ValidationError: If the validator cannot be created.
    """
    try:
        # Create the validator
        return FormatValidator(
            format_type=FormatValidator.FORMAT_MARKDOWN,
            name=name or "MarkdownFormatValidator",
        )

    except Exception as e:
        # Log the error
        log_error(
            e,
            logger,
            component="MarkdownFormatValidatorFactory",
            operation="markdown_format",
        )

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create Markdown format validator: {str(e)}",
                component="MarkdownFormatValidatorFactory",
                operation="markdown_format",
                suggestions=["Check the error message for details"],
                metadata={"error_type": type(e).__name__, "error_message": str(e)},
            )
        raise


@register_validator("custom_format")
def create_custom_format_validator(
    validator: Callable[[str], Dict[str, Any]],
    name: Optional[str] = None,
    **options: Any,
) -> FormatValidator:
    """Create a custom format validator.

    This factory function creates a FormatValidator with a custom validator.
    It is registered with the registry system for dependency injection.

    Args:
        validator: A function that takes a string and returns a dictionary with
            "passed" (bool), "message" (str), and "details" (dict) fields.
        name: Optional name for the validator.
        **options: Additional options (ignored).

    Returns:
        A FormatValidator instance.

    Raises:
        ValidationError: If the validator cannot be created.
    """
    try:
        # Log factory function call
        logger.debug("Creating custom format validator")

        # Create the validator
        validator_instance = FormatValidator(
            format_type=FormatValidator.FORMAT_CUSTOM,
            custom_validator=validator,
            name=name or options.get("name") or "CustomFormatValidator",
        )

        # Log successful creation
        logger.debug(f"Successfully created custom format validator: {validator_instance.name}")

        return validator_instance

    except Exception as e:
        # Log the error
        log_error(
            e,
            logger,
            component="CustomFormatValidatorFactory",
            operation="create_validator",
        )

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create custom format validator: {str(e)}",
                component="CustomFormatValidatorFactory",
                operation="create_validator",
                suggestions=[
                    "Check if the custom validator function is properly implemented",
                    "Verify that the custom validator function has the correct signature",
                ],
                metadata={"error_type": type(e).__name__, "error_message": str(e)},
            )
        raise


def custom_format(
    validator: Callable[[str], Dict[str, Any]], name: Optional[str] = None
) -> FormatValidator:
    """Create a custom format validator.

    This is a convenience function for creating a FormatValidator with a custom validator.

    Args:
        validator: A function that takes a string and returns a dictionary with
            "passed" (bool), "message" (str), and "details" (dict) fields.
        name: Optional name for the validator.

    Returns:
        A FormatValidator instance.

    Raises:
        ValidationError: If the validator cannot be created.
    """
    try:
        # Create the validator
        return FormatValidator(
            format_type=FormatValidator.FORMAT_CUSTOM,
            custom_validator=validator,
            name=name or "CustomFormatValidator",
        )

    except Exception as e:
        # Log the error
        log_error(
            e,
            logger,
            component="CustomFormatValidatorFactory",
            operation="custom_format",
        )

        # Re-raise as ValidationError with more context
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create custom format validator: {str(e)}",
                component="CustomFormatValidatorFactory",
                operation="custom_format",
                suggestions=[
                    "Check if the custom validator function is properly implemented",
                    "Verify that the custom validator function has the correct signature",
                ],
                metadata={"error_type": type(e).__name__, "error_message": str(e)},
            )
        raise
