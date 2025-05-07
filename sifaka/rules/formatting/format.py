"""
Format validation rules for Sifaka.

This module provides rules for validating text format including markdown, JSON, and plain text.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The FormatConfig, MarkdownConfig, JsonConfig, and PlainTextConfig classes extend RuleConfig
      and provide type-safe access to parameters
    - Factory functions (create_format_rule, create_markdown_rule, etc.) handle configuration
    - Validator factory functions (create_format_validator, create_markdown_validator, etc.)
      create standalone validators

Architecture:
    The format validation system uses a delegation pattern:
    - FormatRule is the main entry point that delegates to specialized validators
    - Each format type (markdown, JSON, plain text) has its own validator and analyzer
    - Analyzers handle the specific validation logic for each format type
    - Factory functions provide a convenient way to create rules and validators

Lifecycle:
    1. Creation: Rules and validators are created using factory functions
    2. Configuration: Parameters are validated and stored in config objects
    3. Validation: Text is validated against the specified format requirements
    4. Result: A RuleResult object is returned with validation status and metadata

Error Handling:
    - Invalid configuration parameters raise ValueError during initialization
    - Non-string inputs to validate() raise ValueError
    - JSON parsing errors are caught and returned as failed validation results
    - Empty strings are handled according to configuration (allow_empty parameter)

Usage Examples:
    Basic Usage:
        ```python
        from sifaka.rules.formatting.format import create_format_rule

        # Create a format rule for markdown
        rule = create_format_rule(
            required_format="markdown",
            markdown_elements={"#", "*", "`"}
        )

        # Validate text
        text = "# Heading\n\nThis is a *formatted* text with `code`."
        result = rule.validate(text)

        # Check result
        if result.passed:
            print("Validation passed!")
            print(f"Found elements: {result.metadata['found_elements']}")
        else:
            print(f"Validation failed: {result.message}")
        ```

    Format-Specific Rules:
        ```python
        from sifaka.rules.formatting.format import (
            create_markdown_rule,
            create_json_rule,
            create_plain_text_rule
        )

        # Create a markdown rule
        markdown_rule = create_markdown_rule(
            required_elements=["#", "*", "`"],
            min_elements=2,
            name="markdown_validator"
        )

        # Create a JSON rule
        json_rule = create_json_rule(
            strict=True,
            allow_empty=False,
            name="json_validator"
        )

        # Create a plain text rule
        plain_text_rule = create_plain_text_rule(
            min_length=10,
            max_length=1000,
            allow_empty=False,
            name="text_length_validator"
        )

        # Validate different formats
        md_text = "# Heading\n\nThis is *formatted*."
        json_text = '{"name": "John", "age": 30}'
        plain_text = "This is a simple text."

        md_result = markdown_rule.validate(md_text)
        json_result = json_rule.validate(json_text)
        text_result = plain_text_rule.validate(plain_text)

        # Access validation metadata
        print(f"Markdown elements found: {md_result.metadata['found_elements']}")
        print(f"JSON validation: {json_result.message}")
        print(f"Text length: {text_result.metadata['length']}")
        ```

    Using Standalone Validators:
        ```python
        from sifaka.rules.formatting.format import (
            create_markdown_validator,
            create_json_validator,
            create_plain_text_validator
        )

        # Create validators
        md_validator = create_markdown_validator(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        json_validator = create_json_validator(
            strict=True,
            allow_empty=False
        )

        text_validator = create_plain_text_validator(
            min_length=10,
            max_length=1000
        )

        # Validate text directly with validators
        md_result = md_validator.validate("# Heading with *emphasis*")
        json_result = json_validator.validate('{"valid": true}')
        text_result = text_validator.validate("Plain text content")

        # Process results
        for name, result in [
            ("Markdown", md_result),
            ("JSON", json_result),
            ("Plain Text", text_result)
        ]:
            print(f"{name}: {'✓' if result.passed else '✗'} - {result.message}")
        ```

    Advanced Configuration:
        ```python
        from sifaka.rules.formatting.format import create_format_rule
        from sifaka.rules.base import RuleConfig

        # Create a rule with advanced configuration
        rule = create_format_rule(
            required_format="markdown",
            markdown_elements={"#", "*", "`", ">", "-"},
            name="advanced_format_rule",
            description="Validates complex markdown formatting",
            priority=2,
            cost=1.5,
            cache_size=200
        )

        # Or use RuleConfig directly
        config = RuleConfig(
            priority=2,
            cost=1.5,
            params={
                "required_format": "json",
                "strict": True,
                "allow_empty": False
            }
        )

        json_rule = create_format_rule(
            name="json_format_rule",
            config=config
        )

        # Validate with configured rules
        result = rule.validate("# Heading\n\n> Blockquote with *emphasis*")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
"""

import json
from typing import Any, Dict, List, Literal, Optional, Protocol, Set, Tuple, runtime_checkable

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult

FormatType = Literal["markdown", "plain_text", "json"]


@runtime_checkable
class FormatValidator(Protocol):
    """Protocol for format validation components."""

    @property
    def config(self) -> "FormatConfig": ...

    def validate(self, text: str, **kwargs) -> RuleResult: ...


class FormatConfig(BaseModel):
    """Configuration for format validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    required_format: FormatType = Field(
        default="plain_text",
        description="The required format type",
    )
    markdown_elements: Set[str] = Field(
        default_factory=lambda: {"headers", "lists", "code_blocks"},
        description="Set of required markdown elements",
    )
    json_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for validation",
    )
    min_length: int = Field(
        default=1,
        ge=0,
        description="Minimum text length",
    )
    max_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum text length",
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

    @field_validator("required_format")
    @classmethod
    def validate_format_type(cls, v: FormatType) -> FormatType:
        """Validate that format type is valid."""
        if v not in ["markdown", "plain_text", "json"]:
            raise ValueError(f"required_format must be one of: markdown, plain_text, json, got {v}")
        return v

    @field_validator("max_length")
    @classmethod
    def validate_lengths(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Validate that max_length is greater than min_length if specified."""
        if v is not None and "min_length" in values and v < values["min_length"]:
            raise ValueError("max_length must be greater than or equal to min_length")
        return v


class MarkdownConfig(BaseModel):
    """Configuration for markdown format validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    required_elements: List[str] = Field(
        default_factory=lambda: ["#", "*", "_", "`", ">", "-", "1.", "[", "]", "(", ")"],
        description="List of required markdown elements",
    )
    min_elements: int = Field(
        default=1,
        ge=0,
        description="Minimum number of elements required",
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

    @field_validator("required_elements")
    @classmethod
    def validate_elements(cls, v: List[str]) -> List[str]:
        """Validate that at least one element is required."""
        if not v:
            raise ValueError("Must provide at least one required element")
        return v


class JsonConfig(BaseModel):
    """Configuration for JSON format validation."""

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


class PlainTextConfig(BaseModel):
    """Configuration for plain text format validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_length: int = Field(
        default=1,
        ge=0,
        description="Minimum text length",
    )
    max_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum text length",
    )
    allow_empty: bool = Field(
        default=False,
        description="Whether to allow empty text",
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

    @field_validator("max_length")
    @classmethod
    def validate_lengths(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Validate that max_length is greater than min_length if specified."""
        if v is not None and "min_length" in values and v < values["min_length"]:
            raise ValueError("max_length must be greater than or equal to min_length")
        return v


class DefaultMarkdownValidator(BaseValidator[str]):
    """Default implementation of markdown validation."""

    def __init__(self, config: MarkdownConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._analyzer = _MarkdownAnalyzer(
            required_elements=config.required_elements, min_elements=config.min_elements
        )

    @property
    def config(self) -> MarkdownConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate markdown format."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        passed, found_elements = self._analyzer.analyze(text)
        return RuleResult(
            passed=passed,
            message=(
                f"Found {len(found_elements)} markdown elements"
                if passed
                else f"Insufficient markdown elements: found {len(found_elements)}, require {self.config.min_elements}"
            ),
            metadata={
                "found_elements": found_elements,
                "required_min": self.config.min_elements,
            },
        )


class DefaultJsonValidator(BaseValidator[str]):
    """Default implementation of JSON validation."""

    def __init__(self, config: JsonConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._analyzer = _JsonAnalyzer(strict=config.strict, allow_empty=config.allow_empty)

    @property
    def config(self) -> JsonConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate JSON format."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return self._analyzer.analyze(text)


class DefaultPlainTextValidator(BaseValidator[str]):
    """Default implementation of plain text validation."""

    def __init__(self, config: PlainTextConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._analyzer = _PlainTextAnalyzer(
            min_length=config.min_length,
            max_length=config.max_length,
            allow_empty=config.allow_empty,
        )

    @property
    def config(self) -> PlainTextConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate plain text format."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return self._analyzer.analyze(text)


class FormatRule(Rule[str, RuleResult, BaseValidator[str], Any]):
    """Rule that checks text format."""

    def __init__(
        self,
        name: str = "format_rule",
        description: str = "Validates text format",
        format_type: FormatType = "plain_text",
        config: Optional[RuleConfig] = None,
        validator: Optional[BaseValidator[str]] = None,
    ) -> None:
        """
        Initialize the format rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            format_type: Type of format to validate
            config: Rule configuration
            validator: Optional custom validator implementation
        """
        # Store format type and parameters for creating the default validator
        self._format_type = format_type
        self._rule_params = {}

        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    @property
    def format_type(self) -> FormatType:
        """Get the format type."""
        return self._format_type

    def _create_default_validator(self) -> BaseValidator[str]:
        """Create a default validator based on format type."""
        if self._format_type == "markdown":
            format_config = MarkdownConfig(**self._rule_params)
            return DefaultMarkdownValidator(format_config)
        elif self._format_type == "json":
            format_config = JsonConfig(**self._rule_params)
            return DefaultJsonValidator(format_config)
        else:  # plain_text
            format_config = PlainTextConfig(**self._rule_params)
            return DefaultPlainTextValidator(format_config)


def create_markdown_validator(
    required_elements: Optional[List[str]] = None,
    min_elements: Optional[int] = None,
    **kwargs,
) -> DefaultMarkdownValidator:
    """
    Create a markdown validator with the specified configuration.

    This factory function creates a configured markdown validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        required_elements: List of markdown elements required in the text
        min_elements: Minimum number of required elements that must be present
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured markdown validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if required_elements is not None:
        config_params["required_elements"] = required_elements
    if min_elements is not None:
        config_params["min_elements"] = min_elements

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = MarkdownConfig(**config_params)

    # Return configured validator
    return DefaultMarkdownValidator(config)


def create_json_validator(
    strict: Optional[bool] = None,
    allow_empty: Optional[bool] = None,
    **kwargs,
) -> DefaultJsonValidator:
    """
    Create a JSON validator with the specified configuration.

    This factory function creates a configured JSON validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        strict: Whether to enforce strict JSON validation
        allow_empty: Whether to allow empty JSON strings
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured JSON validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if strict is not None:
        config_params["strict"] = strict
    if allow_empty is not None:
        config_params["allow_empty"] = allow_empty

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = JsonConfig(**config_params)

    # Return configured validator
    return DefaultJsonValidator(config)


def create_plain_text_validator(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: Optional[bool] = None,
    **kwargs,
) -> DefaultPlainTextValidator:
    """
    Create a plain text validator with the specified configuration.

    This factory function creates a configured plain text validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        min_length: Minimum text length allowed
        max_length: Maximum text length allowed
        allow_empty: Whether to allow empty text
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured plain text validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if min_length is not None:
        config_params["min_length"] = min_length
    if max_length is not None:
        config_params["max_length"] = max_length
    if allow_empty is not None:
        config_params["allow_empty"] = allow_empty

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = PlainTextConfig(**config_params)

    # Return configured validator
    return DefaultPlainTextValidator(config)


def create_markdown_rule(
    name: str = "markdown_rule",
    description: str = "Validates markdown format",
    required_elements: Optional[List[str]] = None,
    min_elements: Optional[int] = None,
    **kwargs,
) -> FormatRule:
    """
    Create a markdown format rule with configuration.

    This factory function creates a configured FormatRule instance for markdown validation.
    It uses create_markdown_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        required_elements: List of markdown elements required in the text
        min_elements: Minimum number of required elements that must be present
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FormatRule instance
    """
    # Create validator using the validator factory
    validator = create_markdown_validator(
        required_elements=required_elements,
        min_elements=min_elements,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return FormatRule(
        name=name,
        description=description,
        format_type="markdown",
        validator=validator,
        **rule_kwargs,
    )


def create_json_rule(
    name: str = "json_rule",
    description: str = "Validates JSON format",
    strict: Optional[bool] = None,
    allow_empty: Optional[bool] = None,
    **kwargs,
) -> FormatRule:
    """
    Create a JSON format rule with configuration.

    This factory function creates a configured FormatRule instance for JSON validation.
    It uses create_json_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        strict: Whether to enforce strict JSON validation
        allow_empty: Whether to allow empty JSON strings
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FormatRule instance
    """
    # Create validator using the validator factory
    validator = create_json_validator(
        strict=strict,
        allow_empty=allow_empty,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return FormatRule(
        name=name,
        description=description,
        format_type="json",
        validator=validator,
        **rule_kwargs,
    )


def create_plain_text_rule(
    name: str = "plain_text_rule",
    description: str = "Validates plain text format",
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: Optional[bool] = None,
    **kwargs,
) -> FormatRule:
    """
    Create a plain text format rule with configuration.

    This factory function creates a configured FormatRule instance for plain text validation.
    It uses create_plain_text_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        min_length: Minimum text length allowed
        max_length: Maximum text length allowed
        allow_empty: Whether to allow empty text
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FormatRule instance
    """
    # Create validator using the validator factory
    validator = create_plain_text_validator(
        min_length=min_length,
        max_length=max_length,
        allow_empty=allow_empty,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return FormatRule(
        name=name,
        description=description,
        format_type="plain_text",
        validator=validator,
        **rule_kwargs,
    )


# Add DefaultFormatValidator class
class DefaultFormatValidator(BaseValidator[str]):
    """Default implementation of format validation."""

    def __init__(self, config: FormatConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._validators = {
            "markdown": DefaultMarkdownValidator(
                MarkdownConfig(
                    **{
                        "required_elements": list(config.markdown_elements),
                        "min_elements": 1,
                        "cache_size": config.cache_size,
                        "priority": config.priority,
                        "cost": config.cost,
                    }
                )
            ),
            "json": DefaultJsonValidator(
                JsonConfig(
                    **{
                        "strict": True,
                        "allow_empty": False,
                        "cache_size": config.cache_size,
                        "priority": config.priority,
                        "cost": config.cost,
                    }
                )
            ),
            "plain_text": DefaultPlainTextValidator(
                PlainTextConfig(
                    **{
                        "min_length": config.min_length,
                        "max_length": config.max_length,
                        "allow_empty": False,
                        "cache_size": config.cache_size,
                        "priority": config.priority,
                        "cost": config.cost,
                    }
                )
            ),
        }

    @property
    def config(self) -> FormatConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text format based on the required format type."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if not text.strip():
            return RuleResult(
                passed=False,
                message="Empty text",
                metadata={"error": "empty_string"},
            )

        # Delegate to the appropriate validator based on format type
        validator = self._validators[self.config.required_format]
        return validator.validate(text, **kwargs)


def create_format_validator(
    required_format: Optional[FormatType] = None,
    markdown_elements: Optional[Set[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    **kwargs,
) -> DefaultFormatValidator:
    """
    Create a format validator with the specified configuration.

    This factory function creates a configured format validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        required_format: Type of format to validate (markdown, json, plain_text)
        markdown_elements: Set of markdown elements to check for
        json_schema: JSON schema for validation
        min_length: Minimum text length allowed
        max_length: Maximum text length allowed
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured format validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if required_format is not None:
        config_params["required_format"] = required_format
    if markdown_elements is not None:
        config_params["markdown_elements"] = markdown_elements
    if json_schema is not None:
        config_params["json_schema"] = json_schema
    if min_length is not None:
        config_params["min_length"] = min_length
    if max_length is not None:
        config_params["max_length"] = max_length

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = FormatConfig(**config_params)

    # Return configured validator
    return DefaultFormatValidator(config)


# Function to create a format rule with any configuration
def create_format_rule(
    name: str = "format_rule",
    description: str = "Validates text format",
    required_format: Optional[FormatType] = None,
    markdown_elements: Optional[Set[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    **kwargs,
) -> FormatRule:
    """
    Create a format rule with configuration.

    This factory function creates a configured FormatRule instance.
    It uses create_format_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        required_format: Type of format to validate (markdown, json, plain_text)
        markdown_elements: Set of markdown elements to check for
        json_schema: JSON schema for validation
        min_length: Minimum text length allowed
        max_length: Maximum text length allowed
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FormatRule instance
    """
    # Create validator using the validator factory
    validator = create_format_validator(
        required_format=required_format,
        markdown_elements=markdown_elements,
        json_schema=json_schema,
        min_length=min_length,
        max_length=max_length,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Get the format type from the validator config
    format_type = validator.config.required_format

    # Create and return rule
    return FormatRule(
        name=name,
        description=description,
        format_type=format_type,
        validator=validator,
        **rule_kwargs,
    )


# Export public classes and functions
__all__ = [
    # Rule classes
    "FormatRule",
    # Type definitions
    "FormatType",
    # Config classes
    "FormatConfig",
    "MarkdownConfig",
    "JsonConfig",
    "PlainTextConfig",
    # Validator classes
    "FormatValidator",
    "DefaultFormatValidator",
    "DefaultMarkdownValidator",
    "DefaultJsonValidator",
    "DefaultPlainTextValidator",
    # Validator factory functions
    "create_format_validator",
    "create_markdown_validator",
    "create_json_validator",
    "create_plain_text_validator",
    # Rule factory functions
    "create_format_rule",
    "create_markdown_rule",
    "create_json_rule",
    "create_plain_text_rule",
    # Internal helpers
    "_MarkdownAnalyzer",
    "_JsonAnalyzer",
    "_PlainTextAnalyzer",
]


# ---------------------------------------------------------------------------
# Analyzer helpers
# ---------------------------------------------------------------------------


class _MarkdownAnalyzer(BaseModel):
    """Count markdown element occurrences."""

    required_elements: List[str] = Field(default_factory=list)
    min_elements: int = 1

    def analyze(self, text: str) -> Tuple[bool, List[str]]:
        found = [el for el in self.required_elements if el in text]
        return len(found) >= self.min_elements, found


class _JsonAnalyzer(BaseModel):
    """Validate JSON strings, optionally requiring non-empty."""

    strict: bool = True
    allow_empty: bool = False

    def analyze(self, text: str) -> RuleResult:  # type: ignore[override]
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if not text.strip() and not self.allow_empty:
            return RuleResult(
                passed=False,
                message="Empty JSON string not allowed",
                metadata={"error": "empty_string"},
            )

        try:
            json.loads(text)
            return RuleResult(
                passed=True, message="Valid JSON format", metadata={"strict": self.strict}
            )
        except json.JSONDecodeError as e:
            return RuleResult(
                passed=False,
                message=f"Invalid JSON format: {e}",
                metadata={"error": str(e), "position": e.pos},
            )


class _PlainTextAnalyzer(BaseModel):
    """Check plain text length constraints."""

    min_length: int = 1
    max_length: Optional[int] = None
    allow_empty: bool = False

    def analyze(self, text: str) -> RuleResult:  # type: ignore[override]
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if not text.strip() and not self.allow_empty:
            return RuleResult(
                passed=False,
                message="Empty text not allowed",
                metadata={"error": "empty_string"},
            )

        length = len(text)
        if length < self.min_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} below minimum {self.min_length}",
                metadata={
                    "length": length,
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                },
            )

        if self.max_length is not None and length > self.max_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} exceeds maximum {self.max_length}",
                metadata={
                    "length": length,
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                },
            )

        return RuleResult(
            passed=True,
            message="Valid plain text format",
            metadata={
                "length": length,
                "min_length": self.min_length,
                "max_length": self.max_length,
            },
        )
