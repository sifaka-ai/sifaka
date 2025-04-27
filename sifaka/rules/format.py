"""
Format validation rules for Sifaka.

This module provides rules for validating text format including markdown, JSON, and plain text.
"""

from typing import Dict, Any, List, Protocol, runtime_checkable, Final, Literal, TypeVar
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
import json
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RuleValidator


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
        """Validate configuration parameters."""
        if not isinstance(self.required_elements, list):
            raise ValueError("required_elements must be a List[str]")
        if self.min_elements < 0:
            raise ValueError("min_elements must be non-negative")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class JsonConfig(RuleConfig):
    """Configuration for JSON format validation."""

    strict: bool = True
    allow_empty: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class PlainTextConfig(RuleConfig):
    """Configuration for plain text format validation."""

    min_length: int = 1
    max_length: int | None = None
    allow_empty: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if self.max_length is not None and self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@runtime_checkable
class MarkdownValidator(Protocol):
    """Protocol for markdown validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> MarkdownConfig: ...


@runtime_checkable
class JsonValidator(Protocol):
    """Protocol for JSON validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> JsonConfig: ...


@runtime_checkable
class PlainTextValidator(Protocol):
    """Protocol for plain text validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> PlainTextConfig: ...


class DefaultMarkdownValidator:
    """Default implementation of markdown validation."""

    def __init__(self, config: MarkdownConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> MarkdownConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate markdown format."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

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


class DefaultJsonValidator:
    """Default implementation of JSON validation."""

    def __init__(self, config: JsonConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> JsonConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate JSON format."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

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


class DefaultPlainTextValidator:
    """Default implementation of plain text validation."""

    def __init__(self, config: PlainTextConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> PlainTextConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate plain text format."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text_len = len(text.strip())
        if text_len < self.config.min_length and not self.config.allow_empty:
            return RuleResult(
                passed=False,
                message=f"Text length {text_len} below minimum {self.config.min_length}",
                metadata={"length": text_len, "min_length": self.config.min_length},
            )

        if self.config.max_length and text_len > self.config.max_length:
            return RuleResult(
                passed=False,
                message=f"Text length {text_len} exceeds maximum {self.config.max_length}",
                metadata={"length": text_len, "max_length": self.config.max_length},
            )

        return RuleResult(
            passed=True,
            message="Valid plain text format",
            metadata={"length": text_len},
        )


class FormatRule(Rule):
    """Rule that checks if the output adheres to a specific format."""

    def __init__(
        self,
        name: str,
        description: str,
        format_type: FormatType,
        validator: MarkdownValidator | JsonValidator | PlainTextValidator,
    ) -> None:
        """Initialize the format rule."""
        super().__init__(name=name, description=description)
        self._format_type = format_type
        self._validator = validator

    @property
    def format_type(self) -> FormatType:
        """Get the format type."""
        return self._format_type

    @property
    def validator(self) -> MarkdownValidator | JsonValidator | PlainTextValidator:
        """Get the format validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text matches the required format.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The validation result

        Raises:
            ValueError: If text is not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            result = self._validator.validate(text)
            result.metadata["format_type"] = self._format_type
            return result

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during format validation: {str(e)}",
                metadata={
                    "error": str(e),
                    "format_type": self._format_type,
                },
            )


def create_format_rule(
    name: str,
    description: str,
    format_type: FormatType,
    config: MarkdownConfig | JsonConfig | PlainTextConfig | None = None,
) -> FormatRule:
    """
    Factory function to create a format rule with default configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        format_type: The type of format to validate
        config: Optional configuration for the validator

    Returns:
        FormatRule: Configured format rule instance

    Raises:
        ValueError: If format_type is invalid
    """
    if format_type == "markdown":
        validator = DefaultMarkdownValidator(config or MarkdownConfig())
    elif format_type == "json":
        validator = DefaultJsonValidator(config or JsonConfig())
    else:  # plain_text
        validator = DefaultPlainTextValidator(config or PlainTextConfig())

    return FormatRule(
        name=name,
        description=description,
        format_type=format_type,
        validator=validator,
    )
