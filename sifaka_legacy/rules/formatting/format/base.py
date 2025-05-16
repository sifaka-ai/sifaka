"""
Base classes and protocols for format validation.

This module provides the base protocol and configuration classes for format validation:
- FormatValidator: Protocol for format validation components
- FormatConfig: Configuration for format validation

These components form the foundation for all format validation rules in Sifaka.

## Usage Example
```python
from sifaka.rules.formatting.format.base import FormatValidator, FormatConfig
from sifaka.rules.base import RuleResult

# Create a custom format validator
class CustomFormatValidator(FormatValidator):
    def __init__(self, config: FormatConfig) -> None:
        self._config = config

    @property
    def config(self) -> FormatConfig:
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        # Implement validation logic
        return RuleResult(
            passed=True,
            message="Validation passed",
            metadata={"validator": "CustomFormatValidator"}
        )

# Create configuration
config = FormatConfig(
    required_format="plain_text",
    min_length=10,
    max_length=100
)

# Create validator
validator = CustomFormatValidator(config)

# Validate text
result = validator.validate("Sample text") if validator else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""
from typing import Any, Dict, Literal, Optional, Protocol, Set, runtime_checkable, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sifaka.rules.base import RuleResult
FormatType = Literal['markdown', 'plain_text', 'json']


@runtime_checkable
class FormatValidator(Protocol):
    """Protocol for format validation components."""

    @property
    def config(self) ->'FormatConfig':
        ...

    def validate(self, text: str, **kwargs) ->RuleResult:
        ...


class FormatConfig(BaseModel):
    """
    Configuration for format validation.

    This class defines the configuration parameters for format validation,
    including format type, element requirements, length constraints, and
    performance settings.

    Attributes:
        required_format: The required format type (markdown, plain_text, json)
        markdown_elements: Set of required markdown elements
        json_schema: JSON schema for validation
        min_length: Minimum text length
        max_length: Maximum text length (optional)
        cache_size: Size of the validation cache
        priority: Priority of the rule
        cost: Cost of running the rule

    Examples:
        ```python
        from sifaka.rules.formatting.format.base import FormatConfig

        # Create a markdown format configuration
        config = FormatConfig(
            required_format="markdown",
            markdown_elements={"headers", "lists", "code_blocks"},
            min_length=10,
            max_length=1000
        )

        # Create a JSON format configuration
        config = FormatConfig(
            required_format="json",
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
        )

        # Create a plain text format configuration
        config = FormatConfig(
            required_format="plain_text",
            min_length=10,
            max_length=1000
        )
        ```
    """
    model_config = ConfigDict(frozen=True, extra='forbid')
    required_format: FormatType = Field(default='plain_text', description=
        'The required format type')
    markdown_elements: Set[str] = Field(default_factory=lambda : {'headers',
        'lists', 'code_blocks'}, description=
        'Set of required markdown elements')
    json_schema: Dict[str, Any] = Field(default_factory=dict, description=
        'JSON schema for validation')
    min_length: int = Field(default=1, ge=0, description='Minimum text length')
    max_length: Optional[int] = Field(default=None, ge=0, description=
        'Maximum text length')
    cache_size: int = Field(default=100, ge=1, description=
        'Size of the validation cache')
    priority: int = Field(default=1, ge=0, description='Priority of the rule')
    cost: float = Field(default=1.0, ge=0.0, description=
        'Cost of running the rule')

    @field_validator('required_format')
    @classmethod
    def validate_format_type(cls, v: FormatType) ->Any:
        """Validate that format type is valid."""
        if v not in ['markdown', 'plain_text', 'json']:
            raise ValueError(
                f'required_format must be one of: markdown, plain_text, json, got {v}'
                )
        return v

    @field_validator('max_length')
    @classmethod
    def validate_lengths(cls, v: Optional[int], info) ->Any:
        """Validate that max_length is greater than min_length if specified."""
        if v is not None and hasattr(info, 'data'
            ) and 'min_length' in info.data and v < info.data['min_length']:
            raise ValueError(
                'max_length must be greater than or equal to min_length')
        return v


__all__: List[Any] = ['FormatType', 'FormatValidator', 'FormatConfig']
