"""
Format validation rules for Sifaka.

This module provides rules for validating text format including markdown, JSON, and plain text.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult

FormatType = Literal["markdown", "plain_text", "json"]


@dataclass(frozen=True)
class MarkdownConfig(RuleConfig):
    """Configuration for markdown format validation."""

    required_elements: List[str] = field(
        default_factory=lambda: ["#", "*", "_", "`", ">", "-", "1.", "[", "]", "(", ")"]
    )
    min_elements: int = 1
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.required_elements:
            raise ValueError("Must provide at least one required element")
        if self.min_elements < 0:
            raise ValueError("min_elements must be non-negative")


@dataclass(frozen=True)
class JsonConfig(RuleConfig):
    """Configuration for JSON format validation."""

    strict: bool = True
    allow_empty: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()


@dataclass(frozen=True)
class PlainTextConfig(RuleConfig):
    """Configuration for plain text format validation."""

    min_length: int = 1
    max_length: Optional[int] = None
    allow_empty: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if self.max_length is not None and self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")


class DefaultMarkdownValidator(BaseValidator[str]):
    """Default implementation of markdown validation."""

    def __init__(self, config: MarkdownConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> MarkdownConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate markdown format."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        found_elements = [element for element in self.config.required_elements if element in text]

        passed = len(found_elements) >= self.config.min_elements
        return RuleResult(
            passed=passed,
            message=f"Found {len(found_elements)} markdown elements",
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

    @property
    def config(self) -> JsonConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate JSON format."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if not text.strip() and not self.config.allow_empty:
            return RuleResult(
                passed=False,
                message="Empty JSON string not allowed",
                metadata={"error": "empty_string"},
            )

        try:
            json.loads(text)
            return RuleResult(
                passed=True,
                message="Valid JSON format",
                metadata={"strict": self.config.strict},
            )
        except json.JSONDecodeError as e:
            return RuleResult(
                passed=False,
                message=f"Invalid JSON format: {str(e)}",
                metadata={"error": str(e), "position": e.pos},
            )


class DefaultPlainTextValidator(BaseValidator[str]):
    """Default implementation of plain text validation."""

    def __init__(self, config: PlainTextConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> PlainTextConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate plain text format."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if not text.strip() and not self.config.allow_empty:
            return RuleResult(
                passed=False,
                message="Empty text not allowed",
                metadata={"error": "empty_string"},
            )

        text_length = len(text)
        if text_length < self.config.min_length:
            return RuleResult(
                passed=False,
                message=f"Text length {text_length} is below minimum {self.config.min_length}",
                metadata={
                    "length": text_length,
                    "min_length": self.config.min_length,
                    "max_length": self.config.max_length,
                },
            )

        if self.config.max_length and text_length > self.config.max_length:
            return RuleResult(
                passed=False,
                message=f"Text length {text_length} exceeds maximum {self.config.max_length}",
                metadata={
                    "length": text_length,
                    "min_length": self.config.min_length,
                    "max_length": self.config.max_length,
                },
            )

        return RuleResult(
            passed=True,
            message="Valid plain text format",
            metadata={
                "length": text_length,
                "min_length": self.config.min_length,
                "max_length": self.config.max_length,
            },
        )


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

        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

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


def create_markdown_rule(
    name: str = "markdown_rule",
    description: str = "Validates markdown format",
    config: Optional[Dict[str, Any]] = None,
) -> FormatRule:
    """
    Create a markdown format rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured FormatRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return FormatRule(
        name=name,
        description=description,
        format_type="markdown",
        config=rule_config,
    )


def create_json_rule(
    name: str = "json_rule",
    description: str = "Validates JSON format",
    config: Optional[Dict[str, Any]] = None,
) -> FormatRule:
    """
    Create a JSON format rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured FormatRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return FormatRule(
        name=name,
        description=description,
        format_type="json",
        config=rule_config,
    )


def create_plain_text_rule(
    name: str = "plain_text_rule",
    description: str = "Validates plain text format",
    config: Optional[Dict[str, Any]] = None,
) -> FormatRule:
    """
    Create a plain text format rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured FormatRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return FormatRule(
        name=name,
        description=description,
        format_type="plain_text",
        config=rule_config,
    )


# Export public classes and functions
__all__ = [
    "FormatRule",
    "FormatType",
    "MarkdownConfig",
    "JsonConfig",
    "PlainTextConfig",
    "DefaultMarkdownValidator",
    "DefaultJsonValidator",
    "DefaultPlainTextValidator",
    "create_markdown_rule",
    "create_json_rule",
    "create_plain_text_rule",
]
