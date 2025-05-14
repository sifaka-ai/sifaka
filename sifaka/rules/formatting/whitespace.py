"""
Whitespace validation rules for text.

This module provides validators and rules for checking text whitespace constraints
such as leading/trailing whitespace, spacing between words, and newline formatting.

Usage Example:
    ```python
    from sifaka.rules.formatting.whitespace import create_whitespace_rule

    # Create a whitespace rule
    rule = create_whitespace_rule(
        allow_leading_whitespace=False,
        allow_trailing_whitespace=False,
        allow_multiple_spaces=False,
        normalize_whitespace=True
    )

    # Validate text
    result = rule.validate("This is a test.") if rule else ""
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
"""

import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo
from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult
from sifaka.utils.logging import get_logger
from sifaka.utils.patterns import MULTIPLE_SPACES_PATTERN, replace_pattern, compile_pattern

logger = get_logger(__name__)


class WhitespaceConfig(BaseModel):
    """Configuration for text whitespace validation.

    Attributes:
        allow_leading_whitespace: Whether to allow whitespace at the beginning of text
        allow_trailing_whitespace: Whether to allow whitespace at the end of text
        allow_multiple_spaces: Whether to allow multiple consecutive spaces
        allow_tabs: Whether to allow tab characters
        allow_newlines: Whether to allow newline characters
        max_newlines: Maximum number of consecutive newlines allowed
        normalize_whitespace: Whether to normalize whitespace during validation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    allow_leading_whitespace: bool = Field(
        default=False, description="Whether to allow whitespace at the beginning of text"
    )
    allow_trailing_whitespace: bool = Field(
        default=False, description="Whether to allow whitespace at the end of text"
    )
    allow_multiple_spaces: bool = Field(
        default=False, description="Whether to allow multiple consecutive spaces"
    )
    allow_tabs: bool = Field(default=False, description="Whether to allow tab characters")
    allow_newlines: bool = Field(default=True, description="Whether to allow newline characters")
    max_newlines: Optional[int] = Field(
        default=None, ge=0, description="Maximum number of consecutive newlines allowed"
    )
    normalize_whitespace: bool = Field(
        default=False, description="Whether to normalize whitespace during validation"
    )

    @field_validator("max_newlines")
    @classmethod
    def validate_max_newlines(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validate that max_newlines is only set if allow_newlines is True."""
        allow_newlines = info.data.get("allow_newlines", True) if hasattr(info, "data") else True
        if v is not None and not allow_newlines:
            raise ValueError("max_newlines can only be set if allow_newlines is True")
        return v


class WhitespaceValidator(BaseValidator[str]):
    """
    Base class for text whitespace validators.

    This abstract class defines the interface for whitespace validators and provides
    common functionality. Whitespace validators check text against whitespace constraints
    such as leading/trailing whitespace, spacing between words, and newline formatting.

    Lifecycle:
        1. Initialization: Set up with whitespace constraints
        2. Validation: Check text against whitespace constraints
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.whitespace import WhitespaceValidator, WhitespaceConfig

        class CustomWhitespaceValidator(WhitespaceValidator):
            def validate(self, text: str) -> RuleResult:
                # Handle empty text
                empty_result = (self.handle_empty_text(text)
                if empty_result:
                    return empty_result

                # Perform validation
                errors = []
                if (text.startswith(" "):
                    (errors.append("Text starts with whitespace")

                return RuleResult(
                    passed=not errors,
                    message=errors[0] if errors else "Whitespace validation successful",
                    issues=errors
                )

        # Create and use the validator
        config = WhitespaceConfig(allow_leading_whitespace=False)
        validator = CustomWhitespaceValidator(config)
        result = (validator.validate("This is a test")
        ```
    """

    def __init__(self, config: WhitespaceConfig) -> None:
        """
        Initialize validator with a configuration.

        Args:
            config: Whitespace validation configuration
        """
        super().__init__(validation_type=str)
        self.config = config

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against whitespace constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result
        raise NotImplementedError("Subclasses must implement validate method")


class DefaultWhitespaceValidator(WhitespaceValidator):
    """
    Default implementation of text whitespace validator.

    This validator implements standard whitespace validation logic, checking for:
    - Leading whitespace
    - Trailing whitespace
    - Multiple consecutive spaces
    - Tab characters
    - Newline characters and their count

    It can also normalize whitespace during validation if configured to do so.

    Lifecycle:
        1. Initialization: Set up with whitespace constraints
        2. Validation: Check text against whitespace constraints
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.whitespace import DefaultWhitespaceValidator, WhitespaceConfig

        # Create configuration
        config = WhitespaceConfig(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False,
            allow_multiple_spaces=False,
            normalize_whitespace=True
        )

        # Create validator
        validator = DefaultWhitespaceValidator(config)

        # Validate text
        result = (validator.validate("  This  is  a  test  ")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against whitespace constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result
        original_text = text
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        errors = []
        suggestions = []
        if not self.config.allow_leading_whitespace and text and text[0].isspace():
            errors.append("Text contains leading whitespace")
            suggestions.append("Remove leading whitespace")
        if not self.config.allow_trailing_whitespace and text and text[-1].isspace():
            errors.append("Text contains trailing whitespace")
            suggestions.append("Remove trailing whitespace")
        if not self.config.allow_multiple_spaces and "  " in text:
            errors.append("Text contains multiple consecutive spaces")
            suggestions.append("Replace multiple spaces with single spaces")
        if not self.config.allow_tabs and "\t" in text:
            errors.append("Text contains tab characters")
            suggestions.append("Replace tabs with spaces")
        if not self.config.allow_newlines and "\n" in text:
            errors.append("Text contains newline characters")
            suggestions.append("Remove newline characters")
        elif self.config.max_newlines is not None:
            newline_errors = self._validate_max_newlines(text)
            if newline_errors:
                errors.extend(newline_errors)
                suggestions.append(f"Limit consecutive newlines to {self.config.max_newlines}")
        result = RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Whitespace validation successful",
            metadata={
                "original_text": original_text,
                "normalized_text": text if self.config.normalize_whitespace else original_text,
                "validator_type": self.__class__.__name__,
                "allow_leading_whitespace": self.config.allow_leading_whitespace,
                "allow_trailing_whitespace": self.config.allow_trailing_whitespace,
                "allow_multiple_spaces": self.config.allow_multiple_spaces,
                "allow_tabs": self.config.allow_tabs,
                "allow_newlines": self.config.allow_newlines,
                "max_newlines": self.config.max_newlines,
                "normalize_whitespace": self.config.normalize_whitespace,
            },
            score=1.0 if not errors else 0.0,
            issues=errors,
            suggestions=suggestions,
            processing_time_ms=time.time() - start_time,
        )
        self.update_statistics(result)
        return result

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: The text to normalize

        Returns:
            Text with normalized whitespace
        """
        if not self.config.allow_tabs:
            text = text.replace("\t", " ")
        if not self.config.allow_multiple_spaces:
            text = replace_pattern(text, MULTIPLE_SPACES_PATTERN, " ")
        if not self.config.allow_leading_whitespace:
            text = text.lstrip()
        if not self.config.allow_trailing_whitespace:
            text = text.rstrip()
        return text

    def _validate_max_newlines(self, text: str) -> List[str]:
        """Validate maximum consecutive newlines.

        Args:
            text: The text to validate

        Returns:
            List of error messages if validation failed
        """
        if not text or self.config.max_newlines is None:
            return []
        newline_pattern = compile_pattern("\\n+")
        newline_sequences = newline_pattern.findall(text)
        max_found = max([len(seq) for seq in newline_sequences]) if newline_sequences else 0
        if max_found > self.config.max_newlines:
            return [
                f"Text contains {max_found} consecutive newlines, maximum allowed is {self.config.max_newlines}"
            ]
        return []


class WhitespaceRule(Rule[str]):
    """
    Rule for validating text whitespace constraints.

    This rule validates that text meets whitespace requirements such as
    leading/trailing whitespace, spacing between words, and newline formatting.

    Lifecycle:
        1. Initialization: Set up with whitespace constraints
        2. Validation: Check text against whitespace constraints
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.whitespace import WhitespaceRule, WhitespaceConfig, DefaultWhitespaceValidator

        # Create configuration
        config = WhitespaceConfig(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False,
            allow_multiple_spaces=False,
            normalize_whitespace=True
        )

        # Create validator
        validator = DefaultWhitespaceValidator(config)

        # Create rule
        rule = WhitespaceRule(
            name="whitespace_rule",
            description="Validates text whitespace",
            validator=validator
        )

        # Validate text
        result = (rule.validate("  This  is  a  test  ")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str = "whitespace_rule",
        description: str = "Validates text whitespace",
        config: Optional[RuleConfig] = None,
        validator: Optional[WhitespaceValidator] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize the whitespace rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name, description=description, rule_id=kwargs.pop("rule_id", name), **kwargs
            ),
            validator=validator,
        )
        self._whitespace_validator = validator or self._create_default_validator()

    def _create_default_validator(self) -> WhitespaceValidator:
        """
        Create a default validator from config.

        Returns:
            A configured WhitespaceValidator
        """
        params = self.config.params
        config = WhitespaceConfig(
            allow_leading_whitespace=params.get("allow_leading_whitespace", True),
            allow_trailing_whitespace=params.get("allow_trailing_whitespace", True),
            allow_multiple_spaces=params.get("allow_multiple_spaces", True),
            allow_tabs=params.get("allow_tabs", True),
            allow_newlines=params.get("allow_newlines", True),
            max_newlines=params.get("max_newlines"),
            normalize_whitespace=params.get("normalize_whitespace", False),
        )
        return DefaultWhitespaceValidator(config)


def create_whitespace_validator(
    allow_leading_whitespace: bool = False,
    allow_trailing_whitespace: bool = False,
    allow_multiple_spaces: bool = False,
    allow_tabs: bool = False,
    allow_newlines: bool = True,
    max_newlines: Optional[int] = None,
    normalize_whitespace: bool = False,
    **kwargs: Dict[str, Any],
) -> WhitespaceValidator:
    """
    Create a whitespace validator with the specified constraints.

    This factory function creates a configured WhitespaceValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        allow_leading_whitespace: Whether to allow leading whitespace
        allow_trailing_whitespace: Whether to allow trailing whitespace
        allow_multiple_spaces: Whether to allow multiple consecutive spaces
        allow_tabs: Whether to allow tab characters
        allow_newlines: Whether to allow newline characters
        max_newlines: Maximum number of consecutive newlines allowed
        normalize_whitespace: Whether to normalize whitespace during validation
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured WhitespaceValidator

    Examples:
        ```python
        from sifaka.rules.formatting.whitespace import create_whitespace_validator

        # Create a basic validator
        validator = create_whitespace_validator(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False
        )

        # Create a validator with more constraints
        validator = create_whitespace_validator(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False,
            allow_multiple_spaces=False,
            normalize_whitespace=True
        )
        ```
    """
    config = WhitespaceConfig(
        allow_leading_whitespace=allow_leading_whitespace,
        allow_trailing_whitespace=allow_trailing_whitespace,
        allow_multiple_spaces=allow_multiple_spaces,
        allow_tabs=allow_tabs,
        allow_newlines=allow_newlines,
        max_newlines=max_newlines,
        normalize_whitespace=normalize_whitespace,
        **kwargs,
    )
    return DefaultWhitespaceValidator(config)


def create_whitespace_rule(
    allow_leading_whitespace: bool = False,
    allow_trailing_whitespace: bool = False,
    allow_multiple_spaces: bool = False,
    allow_tabs: bool = False,
    allow_newlines: bool = True,
    max_newlines: Optional[int] = None,
    normalize_whitespace: bool = False,
    name: str = "whitespace_rule",
    description: str = "Validates text whitespace",
    rule_id: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> WhitespaceRule:
    """
    Create a whitespace validation rule with the specified constraints.

    This factory function creates a configured WhitespaceRule instance.
    It uses create_whitespace_validator internally to create the validator.

    Args:
        allow_leading_whitespace: Whether to allow leading whitespace
        allow_trailing_whitespace: Whether to allow trailing whitespace
        allow_multiple_spaces: Whether to allow multiple consecutive spaces
        allow_tabs: Whether to allow tab characters
        allow_newlines: Whether to allow newline characters
        max_newlines: Maximum number of consecutive newlines allowed
        normalize_whitespace: Whether to normalize whitespace during validation
        name: The name of the rule
        description: Description of the rule
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured WhitespaceRule

    Examples:
        ```python
        from sifaka.rules.formatting.whitespace import create_whitespace_rule

        # Create a basic rule
        rule = create_whitespace_rule(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False
        )

        # Create a rule with more constraints and metadata
        rule = create_whitespace_rule(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False,
            allow_multiple_spaces=False,
            normalize_whitespace=True,
            name="strict_whitespace_rule",
            description="Validates text has no extra whitespace",
            rule_id="whitespace_validator",
            severity="warning",
            category="formatting",
            tags=["whitespace", "formatting", "style"]
        )
        ```
    """
    validator = create_whitespace_validator(
        allow_leading_whitespace=allow_leading_whitespace,
        allow_trailing_whitespace=allow_trailing_whitespace,
        allow_multiple_spaces=allow_multiple_spaces,
        allow_tabs=allow_tabs,
        allow_newlines=allow_newlines,
        max_newlines=max_newlines,
        normalize_whitespace=normalize_whitespace,
    )
    params = {
        "allow_leading_whitespace": allow_leading_whitespace,
        "allow_trailing_whitespace": allow_trailing_whitespace,
        "allow_multiple_spaces": allow_multiple_spaces,
        "allow_tabs": allow_tabs,
        "allow_newlines": allow_newlines,
        "max_newlines": max_newlines,
        "normalize_whitespace": normalize_whitespace,
    }
    rule_name = name or rule_id or "whitespace_rule"
    config = RuleConfig(
        name=rule_name,
        description=description,
        rule_id=rule_id or rule_name,
        params=params,
        **kwargs,
    )
    return WhitespaceRule(
        name=rule_name, description=description, config=config, validator=validator
    )
