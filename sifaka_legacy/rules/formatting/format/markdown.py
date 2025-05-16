"""
Markdown format validation for Sifaka.

This module provides classes and functions for validating markdown format:
- MarkdownConfig: Configuration for markdown validation
- DefaultMarkdownValidator: Default implementation of markdown validation
- _MarkdownAnalyzer: Helper class for analyzing markdown elements
- create_markdown_rule: Factory function for creating markdown rules

## Usage Example
```python
from sifaka.rules.formatting.format.markdown import create_markdown_rule

# Create a markdown rule
markdown_rule = create_markdown_rule(
    required_elements=["#", "*", "`"],
    min_elements=2
)

# Validate text
result = markdown_rule.validate("# Heading\n\n* List item") if markdown_rule else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

import time
from typing import Dict, Any, List, Tuple, Set, TypeVar, Optional, cast

from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

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


class MarkdownConfig(BaseModel):
    """
    Configuration for markdown format validation.

    This class defines the configuration parameters for markdown validation,
    including required elements, minimum element count, and performance settings.

    Attributes:
        required_elements: List of required markdown elements
        min_elements: Minimum number of elements required
        cache_size: Size of the validation cache
        priority: Priority of the rule
        cost: Cost of running the rule

    Examples:
        ```python
        from sifaka.rules.formatting.format.markdown import MarkdownConfig

        # Create a basic configuration
        config = MarkdownConfig(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        # Create a configuration with custom settings
        config = MarkdownConfig(
            required_elements=["#", "*", "`", "-", ">"],
            min_elements=3,
            cache_size=200,
            priority=2,
            cost=1.5
        )
        ```
    """

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


class _MarkdownAnalyzer:
    """
    Helper class for analyzing markdown elements in text.

    This internal class analyzes text for markdown elements and determines
    if it meets the requirements specified in the configuration.
    """

    def __init__(self, required_elements: List[str], min_elements: int) -> None:
        """
        Initialize with required elements and minimum count.

        Args:
            required_elements: List of required markdown elements
            min_elements: Minimum number of elements required
        """
        self.required_elements = required_elements
        self.min_elements = min_elements

    def analyze(self, text: str) -> Tuple[bool, List[str]]:
        """
        Analyze text for markdown elements.

        Args:
            text: The text to analyze

        Returns:
            Tuple of (passed, found_elements)
        """
        found_elements = []

        for element in self.required_elements:
            if element in text:
                found_elements.append(element)

        passed = len(found_elements) >= self.min_elements
        return passed, found_elements


class MarkdownValidator(BaseValidator[str]):
    """
    Base implementation of markdown validator for internal use.

    We separate this from DefaultMarkdownValidator to avoid type conflicts.
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)
    _markdown_config: MarkdownConfig

    def __init__(self, config: MarkdownConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: Markdown validation configuration
        """
        super().__init__(validation_type=str)

        # Store the original MarkdownConfig
        self._markdown_config = config

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

    def validate(self, input: str) -> RuleResult:
        """
        Validate markdown format.

        Args:
            input: The text to validate

        Returns:
            Validation result
        """
        return self._validate_impl(input)

    def _validate_impl(self, text: str) -> RuleResult:
        """
        Implementation of markdown validation.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = handle_empty_text(text)
        if empty_result is not None:
            # Ensure we return a RuleResult
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

            passed, found_elements = analyzer.analyze(text)

            suggestions = []
            if not passed:
                missing_elements = set(self._markdown_config.required_elements) - set(
                    found_elements
                )
                if missing_elements:
                    suggestions.append(
                        f"Consider adding these markdown elements: {', '.join(missing_elements)}"
                    )
                else:
                    suggestions.append(
                        f"Add more markdown elements to meet the minimum requirement of {self._markdown_config.min_elements}"
                    )

            # Create a proper RuleResult directly
            result = RuleResult(
                passed=passed,
                message=(
                    f"Found {len(found_elements)} markdown elements"
                    if passed
                    else f"Insufficient markdown elements: found {len(found_elements)}, require {self._markdown_config.min_elements}"
                ),
                metadata={
                    "found_elements": found_elements,
                    "required_min": self._markdown_config.min_elements,
                    "validator_type": self.__class__.__name__,
                },
                score=(
                    len(found_elements) / self._markdown_config.min_elements
                    if self._markdown_config.min_elements > 0
                    else 1.0
                ),
                issues=(
                    []
                    if passed
                    else [
                        f"Insufficient markdown elements: found {len(found_elements)}, require {self._markdown_config.min_elements}"
                    ]
                ),
                suggestions=suggestions,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            update_validation_statistics(self._state_manager, result)

            return result

        except Exception as e:
            record_validation_error(self._state_manager, e)
            if logger:
                logger.error(f"Markdown validation failed: {e}")

            error_message = f"Markdown validation failed: {str(e)}"

            # Create a proper RuleResult directly for error case
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
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            update_validation_statistics(self._state_manager, result)
            return result


# Now create a class that satisfies the FormatValidator protocol
class DefaultMarkdownValidator(MarkdownValidator):
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
        from sifaka.rules.formatting.format.markdown import DefaultMarkdownValidator, MarkdownConfig

        # Create config
        config = MarkdownConfig(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        # Create validator
        validator = DefaultMarkdownValidator(config)

        # Validate text
        result = validator.validate("# Heading\n\n* List item") if validator else ""
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
                required_format="markdown",
                markdown_elements=set(self._markdown_config.required_elements),
                min_length=1,  # Default value
                cache_size=self._markdown_config.cache_size,
                priority=self._markdown_config.priority,
                cost=self._markdown_config.cost,
            )

        return self._format_config


# Register FormatValidator as a virtual superclass
FormatValidator.register(DefaultMarkdownValidator)


class MarkdownRule(BaseRule[str]):
    """
    Rule that validates markdown format.

    This rule checks if text contains the required markdown elements
    and meets the minimum number of elements required.

    Lifecycle:
        1. Initialization: Set up with required elements and minimum count
        2. Validation: Check text for markdown elements
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.format.markdown import MarkdownRule, MarkdownConfig

        # Create config
        config = MarkdownConfig(
            required_elements=["#", "*", "`"],
            min_elements=2
        )

        # Create rule
        rule = MarkdownRule(
            name="markdown_rule",
            description="Validates markdown format",
            config=config
        )

        # Validate text
        result = rule.validate("# Heading\n\n* List item")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    _markdown_config: Optional[MarkdownConfig]

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None,
        validator: Optional[RuleValidator[str]] = None,
        markdown_config: Optional[MarkdownConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the markdown rule.

        Args:
            name: Name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Markdown validator (created if not provided)
            markdown_config: Markdown configuration (used to create validator if needed)
            **kwargs: Additional configuration parameters
        """
        self._markdown_config = markdown_config

        # Create the validator if it's not provided
        actual_validator = validator
        if actual_validator is None:
            actual_validator = self._create_default_validator()

        super().__init__(name, description, config, actual_validator, **kwargs)

    def _create_default_validator(self) -> RuleValidator[str]:
        """
        Create the default validator for this rule.

        Returns:
            Default markdown validator instance
        """
        validator = DefaultMarkdownValidator(self._markdown_config or MarkdownConfig())
        # Type cast to ensure mypy understands this is a RuleValidator[str]
        # and not RuleValidator[str] | None
        return cast(RuleValidator[str], validator)


def create_markdown_rule(
    required_elements: Optional[List[str]] = None,
    min_elements: Optional[int] = None,
    name: str = "markdown_rule",
    description: str = "Validates markdown format",
    rule_id: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[MarkdownConfig] = None,
    **kwargs: Any,
) -> BaseRule[str]:
    """
    Create a rule that validates markdown format.

    Args:
        required_elements: List of required markdown elements
        min_elements: Minimum number of elements required
        name: Name of the rule
        description: Description of the rule
        rule_id: Unique identifier for the rule
        severity: Severity level of the rule
        category: Category of the rule
        tags: Tags for the rule
        config: Markdown configuration
        **kwargs: Additional configuration parameters

    Returns:
        Rule that validates markdown format
    """
    # Create config if not provided
    if config is None:
        config_params: Dict[str, Any] = {}
        if required_elements is not None:
            config_params["required_elements"] = required_elements
        if min_elements is not None:
            config_params["min_elements"] = min_elements

        config = MarkdownConfig(**config_params)

    # Create rule config
    rule_config = RuleConfig(
        name=name,
        description=description,
        rule_id=rule_id or name,
        severity=severity or "warning",
        category=category or "formatting",
        tags=tags or ["markdown", "format", "validation"],
        **kwargs,
    )

    # Create validator
    validator = DefaultMarkdownValidator(config)

    # Use explicit type casting to help mypy
    typed_validator = cast(RuleValidator[str], validator)

    # Create rule
    return MarkdownRule(
        name=name,
        description=description,
        config=rule_config,
        validator=typed_validator,
        markdown_config=config,
    )


__all__ = ["MarkdownConfig", "DefaultMarkdownValidator", "MarkdownRule", "create_markdown_rule"]
