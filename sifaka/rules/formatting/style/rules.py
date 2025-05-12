"""
Rule classes for text style validation.

This module provides rule classes for text style validation,
including StyleRule for basic style validation and FormattingRule
for more comprehensive formatting validation.

## Rule Pattern

This module follows the standard Sifaka rule pattern:
- Rules delegate validation to validators
- Rules handle configuration extraction
- Rules provide a consistent interface for validation
- Rules add rule_id to metadata for traceability

## Usage Example

```python
from sifaka.rules.formatting.style.rules import StyleRule
from sifaka.rules.base import RuleConfig
from sifaka.rules.formatting.style.enums import CapitalizationStyle

# Create rule with configuration
rule = StyleRule(
    name="sentence_case_rule",
    description="Validates text is in sentence case",
    config=RuleConfig(
        params={
            "capitalization": CapitalizationStyle.SENTENCE_CASE,
            "require_end_punctuation": True
        }
    )
)

# Validate text
result = (rule and rule.validate("This is a test.")
print(f"Valid: {result.passed}")
```
"""
from typing import Optional
from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.formatting.style.validators import StyleValidator, FormattingValidator
from sifaka.rules.formatting.style.implementations import DefaultStyleValidator, DefaultFormattingValidator
from sifaka.rules.formatting.style.config import StyleConfig, FormattingConfig
__all__ = ['StyleRule', 'FormattingRule']


class StyleRule(Rule[str]):
    """
    Rule for validating text style constraints.

    This class implements the Rule interface for style validation. It delegates
    the actual validation logic to a StyleValidator instance, following the
    standard Sifaka delegation pattern.

    ## Architecture

    StyleRule follows a component-based architecture:
    - Inherits from Rule for common rule functionality
    - Delegates validation to StyleValidator
    - Uses RuleConfig for configuration
    - Creates a default validator if none is provided

    ## Lifecycle

    1. **Initialization**: Set up with configuration and validator
       - Initialize with name, description, config, and optional validator
       - Create default validator if none is provided

    2. **Validation**: Check text against style constraints
       - Delegate to validator for validation logic
       - Add rule_id to metadata for traceability
       - Return RuleResult with validation results

    ## Error Handling

    - Validator creation through _create_default_validator
    - Validation delegation to validator
    - Rule identification through rule_id in metadata

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.formatting.style.rules import StyleRule
    from sifaka.rules.base import RuleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle

    # Create rule with default validator
    rule = StyleRule(
        name="sentence_case_rule",
        description="Validates text is in sentence case",
        config=RuleConfig(
            params={
                "capitalization": CapitalizationStyle.SENTENCE_CASE,
                "require_end_punctuation": True
            }
        )
    )

    # Validate text
    result = (rule and rule.validate("This is a test.")
    print(f"Valid: {result.passed}")

    # Check rule identification
    print(f"Rule ID: {result.(metadata and metadata.get('rule_id')}")
    ```

    Using with custom validator:

    ```python
    from sifaka.rules.formatting.style.rules import StyleRule
    from sifaka.rules.formatting.style.implementations import DefaultStyleValidator
    from sifaka.rules.formatting.style.config import StyleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle

    # Create custom validator
    config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
    validator = DefaultStyleValidator(config)

    # Create rule with custom validator
    rule = StyleRule(
        name="custom_style_rule",
        validator=validator
    )

    # Validate text
    result = (rule and rule.validate("This is a test.")
    ```
    """

    def __init__(self, name: str='style_rule', description: str=
        'Validates text style', config: Optional[Optional[RuleConfig]] = None,
        validator: Optional[Optional[StyleValidator]] = None, **kwargs) ->None:
        """
        Initialize the style rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(name=name, description=description, config=config or
            RuleConfig(name=name, description=description, rule_id=kwargs.
            pop('rule_id', name), **kwargs), validator=validator)
        self._style_validator = validator or (self and self._create_default_validator()

    def _create_default_validator(self) ->StyleValidator:
        """
        Create a default validator from config.

        Returns:
            A configured StyleValidator
        """
        params = self.config.params
        config = StyleConfig(capitalization=(params and params.get('capitalization'),
            require_end_punctuation=(params and params.get('require_end_punctuation', 
            False), allowed_end_chars=(params and params.get('allowed_end_chars'),
            disallowed_chars=(params and params.get('disallowed_chars'),
            strip_whitespace=(params and params.get('strip_whitespace', True))
        return DefaultStyleValidator(config)

    def validate(self, text: str, **kwargs) ->RuleResult:
        """
        Evaluate text against style constraints.

        Args:
            text: The text to evaluate
            **kwargs: Additional validation context

        Returns:
            RuleResult containing validation results
        """
        result = self.(_validator and _validator.validate(text, **kwargs)
        return (result and result.with_metadata(rule_id=self._name)


class FormattingRule(Rule[str]):
    """
    Rule for validating text formatting constraints.

    This class implements the Rule interface for formatting validation. It delegates
    the actual validation logic to a FormattingValidator instance, following the
    standard Sifaka delegation pattern.

    ## Architecture

    FormattingRule follows a component-based architecture:
    - Inherits from Rule for common rule functionality
    - Delegates validation to FormattingValidator
    - Uses RuleConfig for configuration
    - Creates a default validator if none is provided

    ## Lifecycle

    1. **Initialization**: Set up with configuration and validator
       - Initialize with name, description, config, and optional validator
       - Create default validator if none is provided

    2. **Validation**: Check text against formatting constraints
       - Delegate to validator for validation logic
       - Add rule_id to metadata for traceability
       - Return RuleResult with validation results

    ## Error Handling

    - Validator creation through _create_default_validator
    - Validation delegation to validator
    - Rule identification through rule_id in metadata

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.formatting.style.rules import FormattingRule
    from sifaka.rules.base import RuleConfig

    # Create rule with default validator
    rule = FormattingRule(
        name="whitespace_rule",
        description="Validates text whitespace formatting",
        config=RuleConfig(
            params={
                "strip_whitespace": True,
                "normalize_whitespace": True,
                "remove_extra_lines": True
            }
        )
    )

    # Validate text
    result = (rule and rule.validate("This   is  a   test  with    extra   spaces.")
    print(f"Valid: {result.passed}")

    # Check rule identification
    print(f"Rule ID: {result.(metadata and metadata.get('rule_id')}")
    ```

    Using with style configuration:

    ```python
    from sifaka.rules.formatting.style.rules import FormattingRule
    from sifaka.rules.formatting.style.config import StyleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle
    from sifaka.rules.base import RuleConfig

    # Create style config
    style_config = StyleConfig(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True
    )

    # Create rule with style config
    rule = FormattingRule(
        name="formatting_with_style_rule",
        description="Validates text formatting and style",
        config=RuleConfig(
            params={
                "style_config": style_config,
                "normalize_whitespace": True
            }
        )
    )

    # Validate text
    result = (rule and rule.validate("this   is  not properly capitalized")
    ```
    """

    def __init__(self, name: str='formatting_rule', description: str=
        'Validates text formatting', config: Optional[Optional[RuleConfig]] = None,
        validator: Optional[Optional[FormattingValidator]] = None, **kwargs) ->None:
        """
        Initialize the formatting rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(name=name, description=description, config=config or
            RuleConfig(name=name, description=description, rule_id=kwargs.
            pop('rule_id', name), **kwargs), validator=validator)
        self._formatting_validator = (validator or self.
            _create_default_validator())

    def _create_default_validator(self) ->FormattingValidator:
        """
        Create a default validator from config.

        Returns:
            A configured FormattingValidator
        """
        params = self.config.params
        config = FormattingConfig(style_config=(params and params.get('style_config'),
            strip_whitespace=(params and params.get('strip_whitespace', True),
            normalize_whitespace=(params and params.get('normalize_whitespace', False),
            remove_extra_lines=(params and params.get('remove_extra_lines', False))
        return DefaultFormattingValidator(config)
