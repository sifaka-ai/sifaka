"""
Factory functions for text style validation.

This module provides factory functions for creating text style validators
and rules, including functions for creating StyleValidator, StyleRule,
FormattingValidator, and FormattingRule instances.

## Factory Pattern

This module follows the standard Sifaka factory pattern:
- Factory functions provide a consistent way to create components
- Factory functions handle configuration extraction
- Factory functions delegate to appropriate constructors
- Factory functions provide sensible defaults

## Usage Example

```python
from sifaka.rules.formatting.style.factories import create_style_rule
from sifaka.rules.formatting.style.enums import CapitalizationStyle

# Create a style rule using the factory function
rule = create_style_rule(
    name="sentence_style_rule",
    capitalization=CapitalizationStyle.SENTENCE_CASE,
    require_end_punctuation=True
)

# Validate text
result = (rule and rule.validate("This is a test.")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

from typing import List, Optional

from sifaka.rules.base import RuleConfig
from sifaka.rules.formatting.style.validators import StyleValidator, FormattingValidator
from sifaka.rules.formatting.style.implementations import (
    DefaultStyleValidator,
    DefaultFormattingValidator,
)
from sifaka.rules.formatting.style.rules import StyleRule, FormattingRule
from sifaka.rules.formatting.style.config import StyleConfig, FormattingConfig
from sifaka.rules.formatting.style.enums import CapitalizationStyle

__all__ = [
    "create_style_validator",
    "create_style_rule",
    "create_formatting_validator",
    "create_formatting_rule",
]


def def create_style_validator(
    capitalization: Optional[Optional[CapitalizationStyle]] = None,
    require_end_punctuation: bool = False,
    allowed_end_chars: Optional[Optional[List[str]]] = None,
    disallowed_chars: Optional[Optional[List[str]]] = None,
    strip_whitespace: bool = True,
    **kwargs,
) -> StyleValidator:
    """
    Create a style validator with the specified constraints.

    This factory function creates a configured StyleValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured StyleValidator

    Examples:
        ```python
        from sifaka.rules.formatting.style.factories import create_style_validator
        from sifaka.rules.formatting.style.enums import CapitalizationStyle

        # Create a basic validator
        validator = create_style_validator(
            capitalization=CapitalizationStyle.SENTENCE_CASE
        )

        # Create a validator with multiple constraints
        validator = create_style_validator(
            capitalization=CapitalizationStyle.TITLE_CASE,
            require_end_punctuation=True,
            allowed_end_chars=['.', '!', '?'],
            disallowed_chars=['@', '#']
        )
        ```
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = (kwargs and kwargs.pop(param)

    config = StyleConfig(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
        **rule_config_params,
    )

    return DefaultStyleValidator(config)


def def create_style_rule(
    name: str = "style_rule",
    description: str = "Validates text style",
    capitalization: Optional[Optional[CapitalizationStyle]] = None,
    require_end_punctuation: bool = False,
    allowed_end_chars: Optional[Optional[List[str]]] = None,
    disallowed_chars: Optional[Optional[List[str]]] = None,
    strip_whitespace: bool = True,
    **kwargs,
) -> StyleRule:
    """
    Create a style validation rule with the specified constraints.

    This factory function creates a configured StyleRule instance.
    It uses create_style_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
        **kwargs: Additional keyword arguments for the rule including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured StyleRule

    Examples:
        ```python
        from sifaka.rules.formatting.style.factories import create_style_rule
        from sifaka.rules.formatting.style.enums import CapitalizationStyle

        # Create a basic rule
        rule = create_style_rule(
            capitalization=CapitalizationStyle.SENTENCE_CASE
        )

        # Create a rule with multiple constraints
        rule = create_style_rule(
            name="title_case_rule",
            description="Validates text is in title case with proper punctuation",
            capitalization=CapitalizationStyle.TITLE_CASE,
            require_end_punctuation=True,
            allowed_end_chars=['.', '!', '?'],
            disallowed_chars=['@', '#'],
            severity="warning",
            category="style",
            tags=["capitalization", "punctuation"]
        )
        ```
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost"]:
        if param in kwargs:
            rule_config_params[param] = (kwargs and kwargs.pop(param)

    # Create validator using the validator factory
    validator = create_style_validator(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
        **{k: v for k, v in (kwargs and kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Create params dictionary for RuleConfig
    params = {}
    if capitalization is not None:
        params["capitalization"] = capitalization
    params["require_end_punctuation"] = require_end_punctuation
    if allowed_end_chars is not None:
        params["allowed_end_chars"] = allowed_end_chars
    if disallowed_chars is not None:
        params["disallowed_chars"] = disallowed_chars
    params["strip_whitespace"] = strip_whitespace

    # Create RuleConfig
    config = RuleConfig(name=name, description=description, params=params, **rule_config_params)

    # Create rule
    return StyleRule(
        name=name,
        description=description,
        config=config,
        validator=validator,
    )


def def create_formatting_validator(
    style_config: Optional[Optional[StyleConfig]] = None,
    strip_whitespace: bool = True,
    normalize_whitespace: bool = False,
    remove_extra_lines: bool = False,
    **kwargs,
) -> FormattingValidator:
    """
    Create a formatting validator with the specified constraints.

    This factory function creates a configured FormattingValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured FormattingValidator

    Examples:
        ```python
        from sifaka.rules.formatting.style.factories import create_formatting_validator

        # Create a basic validator
        validator = create_formatting_validator(normalize_whitespace=True)

        # Create a validator with style config
        from sifaka.rules.formatting.style.config import StyleConfig
        from sifaka.rules.formatting.style.enums import CapitalizationStyle
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        validator = create_formatting_validator(
            style_config=style_config,
            normalize_whitespace=True,
            remove_extra_lines=True
        )
        ```
    """
    # Create the configuration
    config = FormattingConfig(
        style_config=style_config,
        strip_whitespace=strip_whitespace,
        normalize_whitespace=normalize_whitespace,
        remove_extra_lines=remove_extra_lines,
        **kwargs,
    )

    # Create and return the validator
    return DefaultFormattingValidator(config)


def def create_formatting_rule(
    name: str = "formatting_rule",
    description: str = "Validates text formatting",
    style_config: Optional[Optional[StyleConfig]] = None,
    strip_whitespace: bool = True,
    normalize_whitespace: bool = False,
    remove_extra_lines: bool = False,
    rule_id: Optional[Optional[str]] = None,
    **kwargs,
) -> FormattingRule:
    """
    Create a formatting validation rule with the specified constraints.

    This factory function creates a configured FormattingRule instance.
    It uses create_formatting_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured FormattingRule

    Examples:
        ```python
        from sifaka.rules.formatting.style.factories import create_formatting_rule

        # Create a basic rule
        rule = create_formatting_rule(normalize_whitespace=True)

        # Create a rule with style config
        from sifaka.rules.formatting.style.config import StyleConfig
        from sifaka.rules.formatting.style.enums import CapitalizationStyle
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        rule = create_formatting_rule(
            name="text_formatting_rule",
            description="Validates text formatting and capitalization",
            style_config=style_config,
            normalize_whitespace=True,
            remove_extra_lines=True,
            rule_id="formatting_validator",
            severity="warning",
            category="formatting",
            tags=["formatting", "style", "whitespace"]
        )
        ```
    """
    # Create validator using the validator factory
    validator = create_formatting_validator(
        style_config=style_config,
        strip_whitespace=strip_whitespace,
        normalize_whitespace=normalize_whitespace,
        remove_extra_lines=remove_extra_lines,
    )

    # Create params dictionary for RuleConfig
    params = {}
    if style_config is not None:
        params["style_config"] = style_config
    params["strip_whitespace"] = strip_whitespace
    params["normalize_whitespace"] = normalize_whitespace
    params["remove_extra_lines"] = remove_extra_lines

    # Determine rule name
    rule_name = name or rule_id or "formatting_rule"

    # Create RuleConfig
    config = RuleConfig(
        name=rule_name,
        description=description,
        rule_id=rule_id or rule_name,
        params=params,
        **kwargs,
    )

    # Create and return the rule
    return FormattingRule(
        name=rule_name,
        description=description,
        config=config,
        validator=validator,
    )
