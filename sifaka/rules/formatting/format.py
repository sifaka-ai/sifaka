"""
Format validation rules for Sifaka.

This module provides rules for validating text format including markdown, JSON, and plain text.

Usage Example:
    ```python
    from sifaka.rules.formatting.format import create_markdown_rule, create_json_rule, create_plain_text_rule

    # Create a markdown rule
    markdown_rule = create_markdown_rule(
        required_elements=["#", "*", "`"],
        min_elements=2
    )

    # Create a JSON rule
    json_rule = create_json_rule(
        strict=True,
        allow_empty=False
    )

    # Create a plain text rule
    plain_text_rule = create_plain_text_rule(
        min_length=10,
        max_length=1000
    )

    # Create a format rule with specific format type
    format_rule = create_format_rule(
        required_format="markdown",
        markdown_elements={"headers", "lists", "code_blocks"}
    )

    # Validate text
    result = markdown_rule.validate("# Heading\n\n* List item")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
"""

import json
import time
from typing import Any, Dict, List, Literal, Optional, Protocol, Set, Tuple, runtime_checkable

from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_rule_state

logger = get_logger(__name__)

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
    """
    Default implementation of markdown validation.

    This validator checks if text contains the required markdown elements
    and meets the minimum number of elements required.

    Lifecycle:
        1. Initialization: Set up with required elements and minimum count
        2. Validation: Check text for markdown elements
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format import DefaultMarkdownValidator, MarkdownConfig

        # Create config
        config = MarkdownConfig(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        # Create validator
        validator = DefaultMarkdownValidator(config)

        # Validate text
        result = validator.validate("# Heading\n\n* List item")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, config: MarkdownConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: Markdown validation configuration
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        self._state_manager.update("config", config)
        self._state_manager.update(
            "analyzer",
            _MarkdownAnalyzer(
                required_elements=config.required_elements, min_elements=config.min_elements
            ),
        )

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def config(self) -> MarkdownConfig:
        """
        Get the validator configuration.

        Returns:
            The markdown configuration
        """
        return self._state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate markdown format.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")

            # Update validation count in metadata
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            passed, found_elements = analyzer.analyze(text)

            suggestions = []
            if not passed:
                missing_elements = set(self.config.required_elements) - set(found_elements)
                if missing_elements:
                    suggestions.append(
                        f"Consider adding these markdown elements: {', '.join(missing_elements)}"
                    )
                else:
                    suggestions.append(
                        f"Add more markdown elements to meet the minimum requirement of {self.config.min_elements}"
                    )

            result = RuleResult(
                passed=passed,
                message=(
                    f"Found {len(found_elements)} markdown elements"
                    if passed
                    else f"Insufficient markdown elements: found {len(found_elements)}, require {self.config.min_elements}"
                ),
                metadata={
                    "found_elements": found_elements,
                    "required_min": self.config.min_elements,
                    "validator_type": self.__class__.__name__,
                },
                score=(
                    len(found_elements) / self.config.min_elements
                    if self.config.min_elements > 0
                    else 1.0
                ),
                issues=(
                    []
                    if passed
                    else [
                        f"Insufficient markdown elements: found {len(found_elements)}, require {self.config.min_elements}"
                    ]
                ),
                suggestions=suggestions,
                processing_time_ms=time.time() - start_time,
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            logger.error(f"Markdown validation failed: {e}")

            error_message = f"Markdown validation failed: {str(e)}"
            result = RuleResult(
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
                processing_time_ms=time.time() - start_time,
            )

            self.update_statistics(result)
            return result


class DefaultJsonValidator(BaseValidator[str]):
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
        from sifaka.rules.formatting.format import DefaultJsonValidator, JsonConfig

        # Create config
        config = JsonConfig(
            strict=True,
            allow_empty=False
        )

        # Create validator
        validator = DefaultJsonValidator(config)

        # Validate text
        result = validator.validate('{"key": "value"}')
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, config: JsonConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: JSON validation configuration
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        self._state_manager.update("config", config)
        self._state_manager.update(
            "analyzer", _JsonAnalyzer(strict=config.strict, allow_empty=config.allow_empty)
        )

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def config(self) -> JsonConfig:
        """
        Get the validator configuration.

        Returns:
            The JSON configuration
        """
        return self._state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate JSON format.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")

            # Update validation count in metadata
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            result = analyzer.analyze(text)

            # Add additional metadata
            result = result.with_metadata(
                validator_type=self.__class__.__name__, processing_time_ms=time.time() - start_time
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            logger.error(f"JSON validation failed: {e}")

            error_message = f"JSON validation failed: {str(e)}"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                    "strict": self.config.strict,
                    "allow_empty": self.config.allow_empty,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check JSON syntax and try again"],
                processing_time_ms=time.time() - start_time,
            )

            self.update_statistics(result)
            return result


class DefaultPlainTextValidator(BaseValidator[str]):
    """
    Default implementation of plain text validation.

    This validator checks if text meets the plain text requirements
    such as minimum/maximum length and empty text handling.

    Lifecycle:
        1. Initialization: Set up with plain text validation parameters
        2. Validation: Check text against length constraints
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format import DefaultPlainTextValidator, PlainTextConfig

        # Create config
        config = PlainTextConfig(
            min_length=10,
            max_length=1000,
            allow_empty=False
        )

        # Create validator
        validator = DefaultPlainTextValidator(config)

        # Validate text
        result = validator.validate("This is a plain text example.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, config: PlainTextConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: Plain text validation configuration
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        self._state_manager.update("config", config)
        self._state_manager.update(
            "analyzer",
            _PlainTextAnalyzer(
                min_length=config.min_length,
                max_length=config.max_length,
                allow_empty=config.allow_empty,
            ),
        )

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def config(self) -> PlainTextConfig:
        """
        Get the validator configuration.

        Returns:
            The plain text configuration
        """
        return self._state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate plain text format.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        if not self.config.allow_empty:
            empty_result = self.handle_empty_text(text)
            if empty_result:
                return empty_result

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")

            # Update validation count in metadata
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            result = analyzer.analyze(text)

            # Add additional metadata
            result = result.with_metadata(
                validator_type=self.__class__.__name__, processing_time_ms=time.time() - start_time
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            logger.error(f"Plain text validation failed: {e}")

            error_message = f"Plain text validation failed: {str(e)}"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                    "min_length": self.config.min_length,
                    "max_length": self.config.max_length,
                    "allow_empty": self.config.allow_empty,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check text length and try again"],
                processing_time_ms=time.time() - start_time,
            )

            self.update_statistics(result)
            return result


class FormatRule(Rule[str]):
    """
    Rule that checks text format.

    This rule validates that text meets the requirements of a specific format
    type (markdown, JSON, or plain text).

    Lifecycle:
        1. Initialization: Set up with format type and parameters
        2. Validation: Delegate to appropriate validator based on format type
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format import FormatRule

        # Create a markdown format rule
        rule = FormatRule(
            name="markdown_format_rule",
            description="Validates markdown format",
            format_type="markdown",
            config=RuleConfig(params={
                "required_elements": ["#", "*", "`"],
                "min_elements": 2
            })
        )

        # Validate text
        result = rule.validate("# Heading\n\n* List item")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self,
        name: str = "format_rule",
        description: str = "Validates text format",
        format_type: FormatType = "plain_text",
        config: Optional[RuleConfig] = None,
        validator: Optional[BaseValidator[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the format rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            format_type: Type of format to validate
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store format type for creating the default validator
        self._format_type = format_type
        self._rule_params = {}

        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name,
                description=description,
                rule_id=kwargs.pop("rule_id", name),
                params={"format_type": format_type, **kwargs},
            ),
            validator=validator,
        )

        # Store the validator in state
        format_validator = validator or self._create_default_validator()
        self._state_manager.update("format_validator", format_validator)

        # Set additional metadata
        self._state_manager.set_metadata("rule_type", "FormatRule")
        self._state_manager.set_metadata("format_type", format_type)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def format_type(self) -> FormatType:
        """
        Get the format type.

        Returns:
            The format type (markdown, json, or plain_text)
        """
        return self._format_type

    def _create_default_validator(self) -> BaseValidator[str]:
        """
        Create a default validator based on format type.

        Returns:
            A configured validator for the specified format type
        """
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

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_markdown_validator

        # Create a basic validator
        validator = create_markdown_validator(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        # Validate text
        result = validator.validate("# Heading\n\n* List item")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    try:
        # Create config with default or provided values
        config_params = {}
        if required_elements is not None:
            config_params["required_elements"] = required_elements
        if min_elements is not None:
            config_params["min_elements"] = min_elements

        # Add any remaining config parameters
        config_params.update(kwargs)

        # Create the config
        config = MarkdownConfig(**config_params)

        # Return configured validator
        return DefaultMarkdownValidator(config)

    except Exception as e:
        logger.error(f"Error creating markdown validator: {e}")
        raise ValueError(f"Error creating markdown validator: {str(e)}")


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

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_json_validator

        # Create a basic validator
        validator = create_json_validator(
            strict=True,
            allow_empty=False
        )

        # Validate text
        result = validator.validate('{"key": "value"}')
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    try:
        # Create config with default or provided values
        config_params = {}
        if strict is not None:
            config_params["strict"] = strict
        if allow_empty is not None:
            config_params["allow_empty"] = allow_empty

        # Add any remaining config parameters
        config_params.update(kwargs)

        # Create the config
        config = JsonConfig(**config_params)

        # Return configured validator
        return DefaultJsonValidator(config)

    except Exception as e:
        logger.error(f"Error creating JSON validator: {e}")
        raise ValueError(f"Error creating JSON validator: {str(e)}")


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

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_plain_text_validator

        # Create a basic validator
        validator = create_plain_text_validator(
            min_length=10,
            max_length=1000,
            allow_empty=False
        )

        # Validate text
        result = validator.validate("This is a plain text example.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    try:
        # Create config with default or provided values
        config_params = {}
        if min_length is not None:
            config_params["min_length"] = min_length
        if max_length is not None:
            config_params["max_length"] = max_length
        if allow_empty is not None:
            config_params["allow_empty"] = allow_empty

        # Add any remaining config parameters
        config_params.update(kwargs)

        # Create the config
        config = PlainTextConfig(**config_params)

        # Return configured validator
        return DefaultPlainTextValidator(config)

    except Exception as e:
        logger.error(f"Error creating plain text validator: {e}")
        raise ValueError(f"Error creating plain text validator: {str(e)}")


def create_markdown_rule(
    name: str = "markdown_rule",
    description: str = "Validates markdown format",
    required_elements: Optional[List[str]] = None,
    min_elements: Optional[int] = None,
    rule_id: Optional[str] = None,
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
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured FormatRule instance

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_markdown_rule

        # Create a basic rule
        rule = create_markdown_rule(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        # Create a rule with metadata
        rule = create_markdown_rule(
            required_elements=["#", "*", "`"],
            min_elements=2,
            name="custom_markdown_rule",
            description="Validates markdown has required elements",
            rule_id="markdown_validator",
            severity="warning",
            category="formatting",
            tags=["markdown", "formatting", "validation"]
        )
        ```
    """
    try:
        # Create validator using the validator factory
        validator = create_markdown_validator(
            required_elements=required_elements,
            min_elements=min_elements,
        )

        # Create params dictionary for RuleConfig
        params = {}
        if required_elements is not None:
            params["required_elements"] = required_elements
        if min_elements is not None:
            params["min_elements"] = min_elements

        # Determine rule name
        rule_name = name or rule_id or "markdown_rule"

        # Create RuleConfig
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["priority", "cache_size", "cost", "severity", "category", "tags"]
            },
        )

        # Create and return the rule
        return FormatRule(
            name=rule_name,
            description=description,
            format_type="markdown",
            config=config,
            validator=validator,
        )

    except Exception as e:
        logger.error(f"Error creating markdown rule: {e}")
        raise ValueError(f"Error creating markdown rule: {str(e)}")


def create_json_rule(
    name: str = "json_rule",
    description: str = "Validates JSON format",
    strict: Optional[bool] = None,
    allow_empty: Optional[bool] = None,
    rule_id: Optional[str] = None,
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
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured FormatRule instance

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_json_rule

        # Create a basic rule
        rule = create_json_rule(
            strict=True,
            allow_empty=False
        )

        # Create a rule with metadata
        rule = create_json_rule(
            strict=True,
            allow_empty=False,
            name="custom_json_rule",
            description="Validates strict JSON format",
            rule_id="json_validator",
            severity="warning",
            category="formatting",
            tags=["json", "formatting", "validation"]
        )
        ```
    """
    try:
        # Create validator using the validator factory
        validator = create_json_validator(
            strict=strict,
            allow_empty=allow_empty,
        )

        # Create params dictionary for RuleConfig
        params = {}
        if strict is not None:
            params["strict"] = strict
        if allow_empty is not None:
            params["allow_empty"] = allow_empty

        # Determine rule name
        rule_name = name or rule_id or "json_rule"

        # Create RuleConfig
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["priority", "cache_size", "cost", "severity", "category", "tags"]
            },
        )

        # Create and return the rule
        return FormatRule(
            name=rule_name,
            description=description,
            format_type="json",
            config=config,
            validator=validator,
        )

    except Exception as e:
        logger.error(f"Error creating JSON rule: {e}")
        raise ValueError(f"Error creating JSON rule: {str(e)}")


def create_plain_text_rule(
    name: str = "plain_text_rule",
    description: str = "Validates plain text format",
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: Optional[bool] = None,
    rule_id: Optional[str] = None,
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
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured FormatRule instance

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_plain_text_rule

        # Create a basic rule
        rule = create_plain_text_rule(
            min_length=10,
            max_length=1000,
            allow_empty=False
        )

        # Create a rule with metadata
        rule = create_plain_text_rule(
            min_length=10,
            max_length=1000,
            allow_empty=False,
            name="custom_text_rule",
            description="Validates text length",
            rule_id="text_validator",
            severity="warning",
            category="formatting",
            tags=["text", "formatting", "validation"]
        )
        ```
    """
    try:
        # Create validator using the validator factory
        validator = create_plain_text_validator(
            min_length=min_length,
            max_length=max_length,
            allow_empty=allow_empty,
        )

        # Create params dictionary for RuleConfig
        params = {}
        if min_length is not None:
            params["min_length"] = min_length
        if max_length is not None:
            params["max_length"] = max_length
        if allow_empty is not None:
            params["allow_empty"] = allow_empty

        # Determine rule name
        rule_name = name or rule_id or "plain_text_rule"

        # Create RuleConfig
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["priority", "cache_size", "cost", "severity", "category", "tags"]
            },
        )

        # Create and return the rule
        return FormatRule(
            name=rule_name,
            description=description,
            format_type="plain_text",
            config=config,
            validator=validator,
        )

    except Exception as e:
        logger.error(f"Error creating plain text rule: {e}")
        raise ValueError(f"Error creating plain text rule: {str(e)}")


class DefaultFormatValidator(BaseValidator[str]):
    """
    Default implementation of format validation.

    This validator delegates to the appropriate format-specific validator
    based on the required format type (markdown, JSON, or plain text).

    Lifecycle:
        1. Initialization: Set up with format type and parameters
        2. Validation: Delegate to appropriate validator based on format type
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format import DefaultFormatValidator, FormatConfig

        # Create config
        config = FormatConfig(
            required_format="markdown",
            markdown_elements={"#", "*", "`"}
        )

        # Create validator
        validator = DefaultFormatValidator(config)

        # Validate text
        result = validator.validate("# Heading\n\n* List item")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, config: FormatConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: Format validation configuration
        """
        super().__init__(validation_type=str)
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
        """
        Get the validator configuration.

        Returns:
            The format configuration
        """
        return self._config

    def validate(self, text: str) -> RuleResult:
        """
        Validate text format based on the required format type.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        try:
            if not isinstance(text, str):
                raise ValueError("Text must be a string")

            # Handle empty text
            empty_result = self.handle_empty_text(text)
            if empty_result:
                return empty_result

            # Delegate to the appropriate validator based on format type
            validator = self._validators[self.config.required_format]
            result = validator.validate(text)

            # Add additional metadata
            result = result.with_metadata(
                validator_type=self.__class__.__name__,
                required_format=self.config.required_format,
                processing_time_ms=time.time() - start_time,
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            logger.error(f"Format validation failed: {e}")

            error_message = f"Format validation failed: {str(e)}"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                    "required_format": self.config.required_format,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=time.time() - start_time,
            )

            self.update_statistics(result)
            return result


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

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_format_validator

        # Create a markdown validator
        validator = create_format_validator(
            required_format="markdown",
            markdown_elements={"#", "*", "`"}
        )

        # Create a JSON validator
        validator = create_format_validator(
            required_format="json"
        )

        # Create a plain text validator
        validator = create_format_validator(
            required_format="plain_text",
            min_length=10,
            max_length=1000
        )
        ```
    """
    try:
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
        config_params.update(kwargs)

        # Create the config
        config = FormatConfig(**config_params)

        # Return configured validator
        return DefaultFormatValidator(config)

    except Exception as e:
        logger.error(f"Error creating format validator: {e}")
        raise ValueError(f"Error creating format validator: {str(e)}")


def create_format_rule(
    name: str = "format_rule",
    description: str = "Validates text format",
    required_format: Optional[FormatType] = None,
    markdown_elements: Optional[Set[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    rule_id: Optional[str] = None,
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
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured FormatRule instance

    Examples:
        ```python
        from sifaka.rules.formatting.format import create_format_rule

        # Create a markdown format rule
        rule = create_format_rule(
            required_format="markdown",
            markdown_elements={"#", "*", "`"}
        )

        # Create a JSON format rule
        rule = create_format_rule(
            required_format="json"
        )

        # Create a plain text format rule with metadata
        rule = create_format_rule(
            required_format="plain_text",
            min_length=10,
            max_length=1000,
            name="custom_format_rule",
            description="Validates text format",
            rule_id="format_validator",
            severity="warning",
            category="formatting",
            tags=["format", "formatting", "validation"]
        )
        ```
    """
    try:
        # Create validator using the validator factory
        validator = create_format_validator(
            required_format=required_format,
            markdown_elements=markdown_elements,
            json_schema=json_schema,
            min_length=min_length,
            max_length=max_length,
        )

        # Create params dictionary for RuleConfig
        params = {}
        if required_format is not None:
            params["required_format"] = required_format
        if markdown_elements is not None:
            params["markdown_elements"] = markdown_elements
        if json_schema is not None:
            params["json_schema"] = json_schema
        if min_length is not None:
            params["min_length"] = min_length
        if max_length is not None:
            params["max_length"] = max_length

        # Get the format type from the validator config
        format_type = validator.config.required_format

        # Determine rule name
        rule_name = name or rule_id or "format_rule"

        # Create RuleConfig
        config = RuleConfig(
            name=rule_name,
            description=description,
            rule_id=rule_id or rule_name,
            params=params,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["priority", "cache_size", "cost", "severity", "category", "tags"]
            },
        )

        # Create and return the rule
        return FormatRule(
            name=rule_name,
            description=description,
            format_type=format_type,
            config=config,
            validator=validator,
        )

    except Exception as e:
        logger.error(f"Error creating format rule: {e}")
        raise ValueError(f"Error creating format rule: {str(e)}")


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

        from sifaka.utils.text import is_empty_text

        if is_empty_text(text) and not self.allow_empty:
            return RuleResult(
                passed=False,
                message="Empty JSON string not allowed",
                metadata={"error": "empty_string", "reason": "empty_input"},
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

        from sifaka.utils.text import is_empty_text

        if is_empty_text(text) and not self.allow_empty:
            return RuleResult(
                passed=False,
                message="Empty text not allowed",
                metadata={"error": "empty_string", "reason": "empty_input"},
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
