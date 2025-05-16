"""
JSON format validation for Sifaka.

This module provides classes and functions for validating JSON format:
- JsonConfig: Configuration for JSON validation
- DefaultJsonValidator: Default implementation of JSON validation
- _JsonAnalyzer: Helper class for analyzing JSON
- create_json_rule: Factory function for creating JSON rules

## Usage Example
```python
from sifaka.rules.formatting.format.json import create_json_rule

# Create a JSON rule
json_rule = create_json_rule(
    strict=True,
    allow_empty=False
)

# Validate text
result = json_rule.validate('{"key": "value"}') if json_rule else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple, TypeVar, cast, Union, overload

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.rules.base import BaseValidator, Rule as BaseRule, RuleConfig, RuleResult, RuleValidator
from sifaka.utils.state import create_rule_state
from sifaka.utils.logging import get_logger

from .base import FormatValidator, FormatConfig
from .utils import (
    handle_empty_text,
    create_validation_result,
    update_validation_statistics,
    record_validation_error,
)

logger = get_logger(__name__)


class JsonConfig(BaseModel):
    """
    Configuration for JSON format validation.

    This class defines the configuration parameters for JSON validation,
    including parsing strictness, empty object handling, and performance settings.

    Attributes:
        strict: Whether to use strict JSON parsing
        allow_empty: Whether to allow empty JSON
        cache_size: Size of the validation cache
        priority: Priority of the rule
        cost: Cost of running the rule

    Examples:
        ```python
        from sifaka.rules.formatting.format.json import JsonConfig

        # Create a basic configuration
        config = JsonConfig(
            strict=True,
            allow_empty=False
        )

        # Create a configuration with custom settings
        config = JsonConfig(
            strict=False,
            allow_empty=True,
            cache_size=200,
            priority=2,
            cost=1.5
        )
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strict: bool = Field(
        default=True,
        description="Whether to use strict JSON parsing",
    )
    allow_empty: bool = Field(
        default=False,
        description="Whether to allow empty JSON",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )


class _JsonAnalyzer:
    """
    Helper class for analyzing JSON.

    This internal class analyzes text for valid JSON format and determines
    if it meets the requirements specified in the configuration.
    """

    def __init__(self, strict: bool, allow_empty: bool) -> None:
        """
        Initialize with configuration parameters.

        Args:
            strict: Whether to use strict JSON parsing
            allow_empty: Whether to allow empty JSON
        """
        self.strict = strict
        self.allow_empty = allow_empty

    def analyze(self, text: str) -> RuleResult:
        """
        Analyze text for valid JSON.

        Args:
            text: The text to analyze

        Returns:
            Validation result
        """
        try:
            # Parse JSON
            parsed = json.loads(text)

            # Check if empty
            if not parsed and not self.allow_empty:
                return RuleResult(
                    passed=False,
                    message="Empty JSON is not allowed",
                    metadata={
                        "empty": True,
                        "strict": self.strict,
                    },
                    score=0.0,
                    issues=["Empty JSON is not allowed"],
                    suggestions=["Provide non-empty JSON content"],
                    processing_time_ms=0.0,
                )

            return RuleResult(
                passed=True,
                message="Valid JSON format",
                metadata={
                    "empty": not bool(parsed),
                    "strict": self.strict,
                    "json_type": type(parsed).__name__,
                },
                score=1.0,
                issues=[],
                suggestions=[],
                processing_time_ms=0.0,
            )

        except json.JSONDecodeError as e:
            # Get error details
            error_message = str(e)
            line_no = getattr(e, "lineno", 0)
            col_no = getattr(e, "colno", 0)

            # Create suggestions
            suggestions = [
                "Check JSON syntax for errors",
                f"Error at line {line_no}, column {col_no}",
            ]

            if "Expecting property name" in error_message:
                suggestions.append("Make sure property names are enclosed in double quotes")
            elif "Expecting ',' delimiter" in error_message:
                suggestions.append("Check for missing commas between elements")
            elif "Expecting ':' delimiter" in error_message:
                suggestions.append("Check for missing colons between keys and values")

            return RuleResult(
                passed=False,
                message=f"Invalid JSON format: {error_message}",
                metadata={
                    "error": error_message,
                    "line": line_no,
                    "column": col_no,
                    "strict": self.strict,
                },
                score=0.0,
                issues=[f"JSON parsing error: {error_message}"],
                suggestions=suggestions,
                processing_time_ms=0.0,
            )


# Define a pure BaseValidator class
class JsonValidator(BaseValidator[str]):
    """
    Base implementation of JSON validator for internal use.

    We separate this from the DefaultJsonValidator to avoid type conflicts.
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)
    _json_config: JsonConfig

    def __init__(self, config: JsonConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: JSON validation configuration
        """
        super().__init__(validation_type=str)

        # Store the original JsonConfig
        self._json_config = config

        # Store configuration in state
        self._state_manager.update("config", config)
        self._state_manager.update(
            "analyzer", _JsonAnalyzer(strict=config.strict, allow_empty=config.allow_empty)
        )

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    def validate(self, input: str) -> RuleResult:
        """
        Validate JSON format.

        Args:
            input: The text to validate

        Returns:
            Validation result
        """
        return self._validate_impl(input)

    def _validate_impl(self, text: str) -> RuleResult:
        """
        Implementation of JSON validation.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = handle_empty_text(text)
        if empty_result is not None:
            # Rather than casting, which mypy doesn't like, directly create a
            # proper RuleResult with the same properties
            return RuleResult(
                passed=False,
                message="Empty input not allowed",
                metadata={"error": "Empty input not allowed"},
                score=0.0,
                issues=["Empty input not allowed"],
                suggestions=["Provide non-empty input"],
                processing_time_ms=0.0,
            )

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")
            if analyzer is None:
                raise ValueError("Analyzer not initialized")

            # Update validation count in metadata
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Get the result from the analyzer
            analyzer_result = analyzer.analyze(text)

            # Add additional metadata
            updated_result = analyzer_result.with_metadata(
                validator_type=self.__class__.__name__,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            update_validation_statistics(self._state_manager, updated_result)

            return updated_result  # type: ignore

        except Exception as e:
            record_validation_error(self._state_manager, e)
            if logger:
                logger.error(f"JSON validation failed: {e}")

            error_message = f"JSON validation failed: {str(e)}"
            error_result = create_validation_result(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                start_time=start_time,
            )

            # Instead of using the utility, directly create a RuleResult
            # to ensure proper type inference
            update_validation_statistics(self._state_manager, error_result)
            return RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=(time.time() - start_time) * 1000,
            )


# Now create a class that works with FormatValidator protocol
class DefaultJsonValidator(JsonValidator):
    """
    Default implementation of JSON validation.

    This validator checks if text is valid JSON according to the specified
    configuration (strict mode, allow empty).

    Lifecycle:
        1. Initialization: Set up with JSON validation parameters
        2. Validation: Check text for valid JSON format
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format.json import DefaultJsonValidator, JsonConfig

        # Create config
        config = JsonConfig(
            strict=True,
            allow_empty=False
        )

        # Create validator
        validator = DefaultJsonValidator(config)

        # Validate text
        result = validator.validate('{"key": "value"}') if validator else ""
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    _format_config: Optional[FormatConfig] = None

    @property
    def config(self) -> FormatConfig:
        """
        Get the validator configuration as a FormatConfig.

        This property fulfills the FormatValidator protocol requirement.

        Returns:
            A FormatConfig representation of our configuration
        """
        # Create and cache a FormatConfig if we don't have one yet
        if self._format_config is None:
            self._format_config = FormatConfig(
                required_format="json",
                min_length=1,
                max_length=None,
                cache_size=self._json_config.cache_size,
                priority=self._json_config.priority,
                cost=self._json_config.cost,
            )

        return self._format_config

    # We intentionally don't implement the validate method with the FormatValidator
    # signature to avoid conflicts with the BaseValidator signature.
    # Instead, when using the DefaultJsonValidator with FormatValidator protocol,
    # callers should use the existing validate method and ignore any kwargs.
    # This is a compromise to deal with incompatible method signatures.


# Duck typing makes this work in practice with Python even if mypy complains
FormatValidator.register(DefaultJsonValidator)


class JsonRule(BaseRule[str]):
    """
    Rule that validates JSON format.

    This rule checks if text is valid JSON according to the specified
    configuration (strict mode, allow empty).

    Lifecycle:
        1. Initialization: Set up with JSON validation parameters
        2. Validation: Check text for valid JSON format
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format.json import JsonRule, JsonConfig

        # Create config
        config = JsonConfig(
            strict=True,
            allow_empty=False
        )

        # Create rule
        rule = JsonRule(
            name="json_rule",
            description="Validates JSON format",
            config=config
        )

        # Validate text
        result = rule.validate('{"key": "value"}') if rule else ""
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None,
        validator: Optional[RuleValidator[str]] = None,
        json_config: Optional[JsonConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the JSON rule.

        Args:
            name: Name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: JSON validator (created if not provided)
            json_config: JSON configuration (used to create validator if needed)
            **kwargs: Additional configuration parameters
        """
        self._json_config = json_config

        # Handle validator properly
        json_validator = validator
        if validator is None and json_config is not None:
            # Create the validator if needed - cast needed for mypy
            json_validator = cast(RuleValidator[str], DefaultJsonValidator(json_config))

        super().__init__(name, description, config, json_validator, **kwargs)

    def _create_default_validator(self) -> RuleValidator[str]:
        """
        Create the default validator for this rule.

        Returns:
            Default JSON validator instance
        """
        json_config = self._json_config or JsonConfig()
        validator = DefaultJsonValidator(json_config)
        return cast(RuleValidator[str], validator)


def create_json_rule(
    strict: Optional[bool] = None,
    allow_empty: Optional[bool] = None,
    name: str = "json_rule",
    description: str = "Validates JSON format",
    rule_id: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[JsonConfig] = None,
    **kwargs: Any,
) -> BaseRule[str]:
    """
    Create a rule that validates JSON format.

    Args:
        strict: Whether to use strict JSON parsing
        allow_empty: Whether to allow empty JSON
        name: Name of the rule
        description: Description of the rule
        rule_id: Unique identifier for the rule
        severity: Severity level of the rule
        category: Category of the rule
        tags: Tags for the rule
        config: JSON configuration
        **kwargs: Additional configuration parameters

    Returns:
        Rule that validates JSON format
    """
    # Create config if not provided
    if config is None:
        config_params = {}
        if strict is not None:
            config_params["strict"] = strict
        if allow_empty is not None:
            config_params["allow_empty"] = allow_empty

        config = JsonConfig(**config_params)

    # Create rule config
    rule_config = RuleConfig(
        name=name,
        description=description,
        rule_id=rule_id or name,
        severity=severity or "warning",
        category=category or "formatting",
        tags=tags or ["json", "format", "validation"],
        **kwargs,
    )

    # Create validator - no need to cast here since we're passing it directly
    validator = DefaultJsonValidator(config)

    # Create rule - use cast to help mypy understand the type compatibility
    return JsonRule(
        name=name,
        description=description,
        config=rule_config,
        validator=cast(RuleValidator[str], validator),
        json_config=config,
    )


__all__ = ["JsonConfig", "DefaultJsonValidator", "JsonRule", "create_json_rule"]
