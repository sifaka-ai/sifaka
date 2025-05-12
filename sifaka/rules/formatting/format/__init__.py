"""
Format validation rules for Sifaka.

This module provides rules for validating text format including markdown, JSON, and plain text.

## Components
- **FormatValidator**: Protocol for format validation components
- **FormatConfig**: Configuration for format validation
- **MarkdownConfig**: Configuration for markdown validation
- **JsonConfig**: Configuration for JSON validation
- **PlainTextConfig**: Configuration for plain text validation
- **DefaultMarkdownValidator**: Default implementation of markdown validation
- **DefaultJsonValidator**: Default implementation of JSON validation
- **DefaultPlainTextValidator**: Default implementation of plain text validation
- **Factory Functions**: Functions for creating format validation rules

## Usage Example
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

## Note on Critic Files
IMPORTANT: All critic implementation files (e.g., files in `sifaka/critics/implementations/`)
should remain as single, self-contained files. The modular approach used in this package
should NOT be applied to critic implementations.
"""

import time
from typing import Any, Dict, List, Optional, Set, Union, TypeVar

from sifaka.rules.base import Rule as BaseRule, RuleConfig, RuleResult

# Import from base module
from .base import FormatType, FormatValidator, FormatConfig

# Import from markdown module
from .markdown import MarkdownConfig, DefaultMarkdownValidator, MarkdownRule, create_markdown_rule

# Import from json module
from .json import JsonConfig, DefaultJsonValidator, JsonRule, create_json_rule

# Import from plain_text module
from .plain_text import (
    PlainTextConfig,
    DefaultPlainTextValidator,
    PlainTextRule,
    create_plain_text_rule,
)


class FormatRule(BaseRule[str]):
    """
    Rule that validates text format based on the specified format type.

    This rule delegates validation to the appropriate format-specific rule
    based on the required format type (markdown, JSON, or plain text).

    Lifecycle:
        1. Initialization: Set up with format type and format-specific parameters
        2. Validation: Delegate to the appropriate format-specific rule
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format import FormatRule

        # Create a markdown format rule
        rule = FormatRule(
            name="format_rule",
            description="Validates text format",
            required_format="markdown",
            markdown_elements={"#", "*", "`"},
            min_elements=2
        )

        # Validate text
        result = rule.validate("# Heading\n\n* List item")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        required_format: FormatType = "plain_text",
        markdown_elements: Set[str] = None,
        min_elements: int = None,
        json_schema: Dict[str, Any] = None,
        strict: bool = None,
        min_length: int = None,
        max_length: int = None,
        allow_empty: bool = None,
        config: Optional[RuleConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the format rule.

        Args:
            name: Name of the rule
            description: Description of the rule
            required_format: The required format type
            markdown_elements: Set of required markdown elements (for markdown format)
            min_elements: Minimum number of elements required (for markdown format)
            json_schema: JSON schema for validation (for JSON format)
            strict: Whether to use strict JSON parsing (for JSON format)
            min_length: Minimum text length (for plain text format)
            max_length: Maximum text length (for plain text format)
            allow_empty: Whether to allow empty text (for JSON and plain text formats)
            config: Rule configuration
            **kwargs: Additional configuration parameters
        """
        self._required_format = required_format
        self._delegate_rule = self._create_delegate_rule(
            required_format=required_format,
            markdown_elements=markdown_elements,
            min_elements=min_elements,
            json_schema=json_schema,
            strict=strict,
            min_length=min_length,
            max_length=max_length,
            allow_empty=allow_empty,
            name=name,
            description=description,
            **kwargs,
        )
        self._validator = None
        super().__init__(name, description, config, None, **kwargs)

    @property
    def validator(self) -> Any:
        """
        Get the validator for this rule.

        Returns:
            The validator from the delegate rule
        """
        return self._delegate_rule.validator

    def _create_default_validator(self) -> Any:
        """
        Create the default validator for this rule.

        Returns:
            The validator from the delegate rule
        """
        return self._delegate_rule.validator

    def _create_delegate_rule(
        self,
        required_format: FormatType,
        markdown_elements: Set[str] = None,
        min_elements: int = None,
        json_schema: Dict[str, Any] = None,
        strict: bool = None,
        min_length: int = None,
        max_length: int = None,
        allow_empty: bool = None,
        name: str = "format_rule",
        description: str = "Validates text format",
        rule_id: str = None,
        severity: str = None,
        category: str = None,
        tags: List[str] = None,
        **kwargs: Any,
    ) -> BaseRule[str]:
        """
        Create the delegate rule based on the format type.

        Args:
            required_format: The required format type
            markdown_elements: Set of required markdown elements (for markdown format)
            min_elements: Minimum number of elements required (for markdown format)
            json_schema: JSON schema for validation (for JSON format)
            strict: Whether to use strict JSON parsing (for JSON format)
            min_length: Minimum text length (for plain text format)
            max_length: Maximum text length (for plain text format)
            allow_empty: Whether to allow empty text (for JSON and plain text formats)
            name: Name of the rule
            description: Description of the rule
            rule_id: Unique identifier for the rule
            severity: Severity level of the rule
            category: Category of the rule
            tags: Tags for the rule
            **kwargs: Additional configuration parameters

        Returns:
            The delegate rule for the specified format type
        """
        # Create rule based on format type
        if required_format == "markdown":
            return create_markdown_rule(
                required_elements=list(markdown_elements) if markdown_elements else None,
                min_elements=min_elements,
                name=name,
                description=description,
                rule_id=rule_id,
                severity=severity,
                category=category,
                tags=tags,
                **kwargs,
            )
        elif required_format == "json":
            return create_json_rule(
                strict=strict,
                allow_empty=allow_empty,
                name=name,
                description=description,
                rule_id=rule_id,
                severity=severity,
                category=category,
                tags=tags,
                **kwargs,
            )
        elif required_format == "plain_text":
            return create_plain_text_rule(
                min_length=min_length,
                max_length=max_length,
                allow_empty=allow_empty,
                name=name,
                description=description,
                rule_id=rule_id,
                severity=severity,
                category=category,
                tags=tags,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported format type: {required_format}")


def create_format_rule(
    required_format: FormatType = "plain_text",
    markdown_elements: Set[str] = None,
    min_elements: int = None,
    json_schema: Dict[str, Any] = None,
    strict: bool = None,
    min_length: int = None,
    max_length: int = None,
    allow_empty: bool = None,
    name: str = "format_rule",
    description: str = "Validates text format",
    rule_id: str = None,
    severity: str = None,
    category: str = None,
    tags: List[str] = None,
    **kwargs: Any,
) -> BaseRule[str]:
    """
    Create a rule that validates text format.

    This factory function creates a rule that validates text according to the
    specified format type (markdown, JSON, or plain text).

    Args:
        required_format: The required format type
        markdown_elements: Set of required markdown elements (for markdown format)
        min_elements: Minimum number of elements required (for markdown format)
        json_schema: JSON schema for validation (for JSON format)
        strict: Whether to use strict JSON parsing (for JSON format)
        min_length: Minimum text length (for plain text format)
        max_length: Maximum text length (for plain text format)
        allow_empty: Whether to allow empty text (for JSON and plain text formats)
        name: Name of the rule
        description: Description of the rule
        rule_id: Unique identifier for the rule
        severity: Severity level of the rule
        category: Category of the rule
        tags: Tags for the rule
        **kwargs: Additional configuration parameters

    Returns:
        Rule that validates text format
    """
    # Create rule config
    rule_config = RuleConfig(
        name=name,
        description=description,
        rule_id=rule_id or name,
        severity=severity or "warning",
        category=category or "formatting",
        tags=tags or ["format", "validation", required_format],
        **kwargs,
    )

    # Create format rule
    return FormatRule(
        name=name,
        description=description,
        required_format=required_format,
        markdown_elements=markdown_elements,
        min_elements=min_elements,
        json_schema=json_schema,
        strict=strict,
        min_length=min_length,
        max_length=max_length,
        allow_empty=allow_empty,
        config=rule_config,
        **kwargs,
    )


# Define what symbols to export
__all__ = [
    # Base types and protocols
    "FormatType",
    "FormatValidator",
    "FormatConfig",
    # Markdown components
    "MarkdownConfig",
    "DefaultMarkdownValidator",
    "MarkdownRule",
    "create_markdown_rule",
    # JSON components
    "JsonConfig",
    "DefaultJsonValidator",
    "JsonRule",
    "create_json_rule",
    # Plain text components
    "PlainTextConfig",
    "DefaultPlainTextValidator",
    "PlainTextRule",
    "create_plain_text_rule",
    # Format components
    "FormatRule",
    "create_format_rule",
]
