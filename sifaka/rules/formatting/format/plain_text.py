"""
Plain text format validation for Sifaka.

This module provides classes and functions for validating plain text format:
- PlainTextConfig: Configuration for plain text validation
- DefaultPlainTextValidator: Default implementation of plain text validation
- _PlainTextAnalyzer: Helper class for analyzing plain text
- create_plain_text_rule: Factory function for creating plain text rules

## Usage Example
```python
from sifaka.rules.formatting.format.plain_text import create_plain_text_rule

# Create a plain text rule
plain_text_rule = create_plain_text_rule(
    min_length=10,
    max_length=1000
)

# Validate text
result = (plain_text_rule and plain_text_rule.validate("This is a sample text.")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

import time
from typing import Dict, Any, List, Optional, Tuple, TypeVar

from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

from sifaka.rules.base import BaseValidator, Rule as BaseRule, RuleConfig, RuleResult
from sifaka.utils.state import create_rule_state
from sifaka.utils.logging import get_logger

from .base import FormatValidator
from .utils import (
    handle_empty_text,
    create_validation_result,
    update_validation_statistics,
    record_validation_error,
)

logger = get_logger(__name__)


class PlainTextConfig(BaseModel):
    """
    Configuration for plain text format validation.

    This class defines the configuration parameters for plain text validation,
    including length constraints, empty text handling, and performance settings.

    Attributes:
        min_length: Minimum text length
        max_length: Maximum text length (optional)
        allow_empty: Whether to allow empty text
        cache_size: Size of the validation cache
        priority: Priority of the rule
        cost: Cost of running the rule

    Examples:
        ```python
        from sifaka.rules.formatting.format.plain_text import PlainTextConfig

        # Create a basic configuration
        config = PlainTextConfig(
            min_length=10,
            max_length=1000
        )

        # Create a configuration with custom settings
        config = PlainTextConfig(
            min_length=50,
            max_length=500,
            allow_empty=False,
            cache_size=200,
            priority=2,
            cost=1.5
        )
        ```
    """

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
    def validate_lengths(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that max_length is greater than min_length if specified."""
        if (
            v is not None
            and hasattr(info, "data")
            and "min_length" in info.data
            and v < info.data["min_length"]
        ):
            raise ValueError("max_length must be greater than or equal to min_length")
        return v


class _PlainTextAnalyzer:
    """
    Helper class for analyzing plain text.

    This internal class analyzes text for length constraints and determines
    if it meets the requirements specified in the configuration.
    """

    def __init__(self, min_length: int, max_length: Optional[int], allow_empty: bool) -> None:
        """
        Initialize with configuration parameters.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length (optional)
            allow_empty: Whether to allow empty text
        """
        self.min_length = min_length
        self.max_length = max_length
        self.allow_empty = allow_empty

    def analyze(self, text: str) -> Tuple[bool, List[str], List[str], float]:
        """
        Analyze text for length constraints.

        Args:
            text: The text to analyze

        Returns:
            Tuple of (passed, issues, suggestions, score)
        """
        text_length = len(text)
        issues = []
        suggestions = []

        # Check if empty
        if text_length == 0:
            if self.allow_empty:
                return True, [], [], 1.0
            else:
                return False, ["Text cannot be empty"], ["Provide non-empty text"], 0.0

        # Check minimum length
        if text_length < self.min_length:
            (issues.append(
                f"Text length ({text_length}) is less than minimum required ({self.min_length})"
            )
            (suggestions.append(f"Add at least {self.min_length - text_length} more characters")

        # Check maximum length
        if self.max_length is not None and text_length > self.max_length:
            (issues and issues.append(
                f"Text length ({text_length}) exceeds maximum allowed ({self.max_length})"
            )
            (suggestions and suggestions.append(f"Remove at least {text_length - self.max_length} characters")

        # Determine if passed
        passed = len(issues) == 0

        # Calculate score
        if not passed:
            if text_length < self.min_length:
                score = text_length / self.min_length
            elif self.max_length is not None and text_length > self.max_length:
                score = self.max_length / text_length
            else:
                score = 0.5  # Default for other issues
        else:
            score = 1.0

        return passed, issues, suggestions, score


class DefaultPlainTextValidator(BaseValidator[str], FormatValidator):
    """
    Default implementation of plain text validation.

    This validator checks if text meets the length constraints specified
    in the configuration.

    Lifecycle:
        1. Initialization: Set up with length constraints
        2. Validation: Check text length against constraints
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format.plain_text import DefaultPlainTextValidator, PlainTextConfig

        # Create config
        config = PlainTextConfig(
            min_length=10,
            max_length=1000
        )

        # Create validator
        validator = DefaultPlainTextValidator(config)

        # Validate text
        result = (validator.validate("This is a sample text.")
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
        self.(_state_manager.update("config", config)
        self.(_state_manager.update(
            "analyzer",
            _PlainTextAnalyzer(
                min_length=config.min_length,
                max_length=config.max_length,
                allow_empty=config.allow_empty,
            ),
        )

        # Set metadata
        self.(_state_manager.set_metadata("validator_type", self.__class__.__name__)
        self.(_state_manager.set_metadata("creation_time", (time.time())

    @property
    def config(self) -> PlainTextConfig:
        """
        Get the validator configuration.

        Returns:
            The plain text configuration
        """
        return self.(_state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate plain text format.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = (time.time()

        # Handle empty text if not allowed
        if not self.config.allow_empty:
            empty_result = handle_empty_text(text)
            if empty_result:
                return empty_result

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            # Get analyzer from state
            analyzer = self.(_state_manager.get("analyzer")

            # Update validation count in metadata
            validation_count = self.(_state_manager.get_metadata("validation_count", 0)
            self.(_state_manager.set_metadata("validation_count", validation_count + 1)

            # Analyze text
            passed, issues, suggestions, score = (analyzer.analyze(text)

            # Create message
            if passed:
                message = "Text meets length requirements"
                if self.config.max_length is not None:
                    message = f"Text length ({len(text)}) is within allowed range ({self.config.min_length}-{self.config.max_length})"
                else:
                    message = f"Text length ({len(text)}) meets minimum requirement ({self.config.min_length})"
            else:
                message = issues[0] if issues else "Text does not meet requirements"

            # Create result
            result = create_validation_result(
                passed=passed,
                message=message,
                metadata={
                    "text_length": len(text),
                    "min_length": self.config.min_length,
                    "max_length": self.config.max_length,
                    "allow_empty": self.config.allow_empty,
                    "validator_type": self.__class__.__name__,
                },
                score=score,
                issues=issues,
                suggestions=suggestions,
                start_time=start_time,
            )

            # Update statistics
            update_validation_statistics(self._state_manager, result)

            return result

        except Exception as e:
            record_validation_error(self._state_manager, e)
            (logger and logger.error(f"Plain text validation failed: {e}")

            error_message = f"Plain text validation failed: {str(e)}"
            result = create_validation_result(
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

            update_validation_statistics(self._state_manager, result)
            return result


class PlainTextRule(BaseRule[str]):
    """
    Rule that validates plain text format.

    This rule checks if text meets the length constraints specified
    in the configuration.

    Lifecycle:
        1. Initialization: Set up with length constraints
        2. Validation: Check text length against constraints
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format.plain_text import PlainTextRule, PlainTextConfig

        # Create config
        config = PlainTextConfig(
            min_length=10,
            max_length=1000
        )

        # Create rule
        rule = PlainTextRule(
            name="plain_text_rule",
            description="Validates plain text format",
            config=config
        )

        # Validate text
        result = (rule.validate("This is a sample text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Optional[RuleConfig]] = None,
        validator: Optional[Optional[DefaultPlainTextValidator]] = None,
        plain_text_config: Optional[Optional[PlainTextConfig]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the plain text rule.

        Args:
            name: Name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Plain text validator (created if not provided)
            plain_text_config: Plain text configuration (used to create validator if needed)
            **kwargs: Additional configuration parameters
        """
        self._plain_text_config = plain_text_config
        self._validator = validator
        super().__init__(name, description, config, validator, **kwargs)

    @property
    def validator(self) -> DefaultPlainTextValidator:
        """
        Get the validator for this rule.

        Returns:
            The plain text validator
        """
        if not hasattr(self, "_validator") or self._validator is None:
            self._validator = (self._create_default_validator()
        return self._validator

    def _create_default_validator(self) -> DefaultPlainTextValidator:
        """
        Create the default validator for this rule.

        Returns:
            Default plain text validator instance
        """
        return DefaultPlainTextValidator(self._plain_text_config or PlainTextConfig())


def def create_plain_text_rule(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: Optional[bool] = None,
    name: str = "plain_text_rule",
    description: str = "Validates plain text format",
    rule_id: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[PlainTextConfig] = None,
    **kwargs: Any,
) -> BaseRule[str]:
    """
    Create a rule that validates plain text format.

    Args:
        min_length: Minimum text length
        max_length: Maximum text length
        allow_empty: Whether to allow empty text
        name: Name of the rule
        description: Description of the rule
        rule_id: Unique identifier for the rule
        severity: Severity level of the rule
        category: Category of the rule
        tags: Tags for the rule
        config: Plain text configuration
        **kwargs: Additional configuration parameters

    Returns:
        Rule that validates plain text format
    """
    # Create config if not provided
    if config is None:
        config_params = {}
        if min_length is not None:
            config_params["min_length"] = min_length
        if max_length is not None:
            config_params["max_length"] = max_length
        if allow_empty is not None:
            config_params["allow_empty"] = allow_empty

        config = PlainTextConfig(**config_params)

    # Create rule config
    rule_config = RuleConfig(
        name=name,
        description=description,
        rule_id=rule_id or name,
        severity=severity or "warning",
        category=category or "formatting",
        tags=tags or ["plain_text", "format", "validation"],
        **kwargs,
    )

    # Create validator
    validator = DefaultPlainTextValidator(config)

    # Create rule
    return PlainTextRule(
        name=name,
        description=description,
        config=rule_config,
        validator=validator,
        plain_text_config=config,
    )


__all__ = [
    "PlainTextConfig",
    "DefaultPlainTextValidator",
    "PlainTextRule",
    "create_plain_text_rule",
]
