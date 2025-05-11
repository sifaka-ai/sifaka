"""
Style validation rules for text.

This module provides validators and rules for checking text styling constraints
such as capitalization, punctuation, and other formatting standards.

## Rule and Validator Relationship

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both
- Empty text is handled consistently using BaseValidator.handle_empty_text

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- Configuration is stored in dedicated config classes
- Factory functions handle configuration extraction
- Validator factory functions create standalone validators
- Rule factory functions use validator factory functions internally

## Usage Example

```python
from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle

# Create a style rule using the factory function
rule = create_style_rule(
    name="sentence_style_rule",
    capitalization=CapitalizationStyle.SENTENCE_CASE,
    require_end_punctuation=True
)

# Validate text
result = rule.validate("This is a test.")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

import time
from enum import Enum, auto
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import (
    Rule,
    RuleResult,
    RuleConfig,
    BaseValidator,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.patterns import WHITESPACE_PATTERN, MULTIPLE_NEWLINES_PATTERN, replace_pattern

logger = get_logger(__name__)


__all__ = [
    # Enums
    "CapitalizationStyle",
    # Config classes
    "StyleConfig",
    "FormattingConfig",
    # Validator classes
    "StyleValidator",
    "DefaultStyleValidator",
    "FormattingValidator",
    "DefaultFormattingValidator",
    # Rule classes
    "StyleRule",
    "FormattingRule",
    # Factory functions
    "create_style_validator",
    "create_style_rule",
    "create_formatting_validator",
    "create_formatting_rule",
    # Internal helpers
    "_CapitalizationAnalyzer",
    "_EndingAnalyzer",
    "_CharAnalyzer",
]


class CapitalizationStyle(Enum):
    """
    Enumeration of text capitalization styles.

    This enum defines the different capitalization styles that can be enforced
    by style validators and rules. Each style represents a specific pattern of
    capitalization that text must follow to be considered valid.

    ## Usage

    ```python
    from sifaka.rules.formatting.style import CapitalizationStyle, create_style_rule

    # Create a rule that enforces sentence case
    rule = create_style_rule(capitalization=CapitalizationStyle.SENTENCE_CASE)

    # Validate text against the rule
    result = rule.validate("This is a test.")  # Passes
    result = rule.validate("this is a test.")  # Fails
    ```

    ## Style Descriptions

    - SENTENCE_CASE: First letter capitalized, rest lowercase (e.g., "This is a test")
    - TITLE_CASE: Major words capitalized (e.g., "This Is a Test")
    - LOWERCASE: All letters lowercase (e.g., "this is a test")
    - UPPERCASE: All letters uppercase (e.g., "THIS IS A TEST")
    - CAPITALIZE_FIRST: Only first letter capitalized (e.g., "This is a test")
    """

    SENTENCE_CASE = auto()  # First letter capitalized, rest lowercase
    TITLE_CASE = auto()  # Major Words Capitalized
    LOWERCASE = auto()  # all lowercase
    UPPERCASE = auto()  # ALL UPPERCASE
    CAPITALIZE_FIRST = auto()  # Only first letter capitalized


class StyleConfig(BaseModel):
    """
    Configuration for text style validation.

    This class defines the configuration options for style validation, including
    capitalization requirements, punctuation rules, and character restrictions.
    It's used by StyleValidator implementations to determine validation behavior.

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create from RuleConfig.params

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Validation of enum values
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by StyleValidator implementations
       - Used by StyleRule._create_default_validator
       - Used by create_style_validator factory function

    ## Error Handling

    - Type validation through Pydantic
    - Immutability prevents accidental modification
    - Extra fields rejected with extra="forbid"

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.formatting.style import StyleConfig, CapitalizationStyle

    # Create with default values
    config = StyleConfig()

    # Create with custom values
    config = StyleConfig(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
        allowed_end_chars=['.', '!', '?'],
        disallowed_chars=['@', '#'],
        strip_whitespace=True
    )

    # Create from dictionary
    config_dict = {
        "capitalization": CapitalizationStyle.TITLE_CASE,
        "require_end_punctuation": True
    }
    config = StyleConfig.model_validate(config_dict)
    ```

    Using with validators:

    ```python
    from sifaka.rules.formatting.style import StyleConfig, DefaultStyleValidator

    # Create config
    config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)

    # Create validator with config
    validator = DefaultStyleValidator(config)

    # Validate text
    result = validator.validate("This is a test.")
    print(f"Valid: {result.passed}")
    ```

    Attributes:
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capitalization: Optional[CapitalizationStyle] = Field(
        default=None,
        description="Required capitalization style",
    )
    require_end_punctuation: bool = Field(
        default=False,
        description="Whether text must end with punctuation",
    )
    allowed_end_chars: Optional[List[str]] = Field(
        default=None,
        description="List of allowed ending characters",
    )
    disallowed_chars: Optional[List[str]] = Field(
        default=None,
        description="List of characters not allowed in the text",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip whitespace before validation",
    )


class FormattingConfig(BaseModel):
    """
    Configuration for text formatting validation.

    This class defines the configuration options for formatting validation,
    including whitespace handling, line normalization, and style requirements.
    It's used by FormattingValidator implementations to determine validation behavior.

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create from RuleConfig.params

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Nested validation of StyleConfig
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by FormattingValidator implementations
       - Used by FormattingRule._create_default_validator
       - Used by create_formatting_validator factory function

    ## Error Handling

    - Type validation through Pydantic
    - Immutability prevents accidental modification
    - Extra fields rejected with extra="forbid"

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.formatting.style import FormattingConfig, StyleConfig, CapitalizationStyle

    # Create with default values
    config = FormattingConfig()

    # Create with custom values
    config = FormattingConfig(
        strip_whitespace=True,
        normalize_whitespace=True,
        remove_extra_lines=True
    )

    # Create with nested style config
    style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
    config = FormattingConfig(
        style_config=style_config,
        normalize_whitespace=True
    )

    # Create from dictionary
    config_dict = {
        "strip_whitespace": True,
        "normalize_whitespace": True,
        "style_config": {
            "capitalization": CapitalizationStyle.TITLE_CASE
        }
    }
    config = FormattingConfig.model_validate(config_dict)
    ```

    Using with validators:

    ```python
    from sifaka.rules.formatting.style import FormattingConfig, DefaultFormattingValidator

    # Create config
    config = FormattingConfig(normalize_whitespace=True)

    # Create validator with config
    validator = DefaultFormattingValidator(config)

    # Validate text
    result = validator.validate("This   is  a   test.")
    print(f"Valid: {result.passed}")
    ```

    Attributes:
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    style_config: Optional[StyleConfig] = Field(
        default=None,
        description="Configuration for style validation",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip whitespace before validation",
    )
    normalize_whitespace: bool = Field(
        default=False,
        description="Whether to normalize consecutive whitespace",
    )
    remove_extra_lines: bool = Field(
        default=False,
        description="Whether to remove extra blank lines",
    )


class StyleValidator(BaseValidator[str]):
    """
    Base class for text style validators.

    This abstract class defines the interface for style validators and provides
    common functionality. Style validators check text against style constraints
    such as capitalization, punctuation, and character restrictions.

    ## Architecture

    StyleValidator follows a component-based architecture:
    - Inherits from BaseValidator for common validation functionality
    - Uses StyleConfig for configuration
    - Provides an abstract validate method for subclasses to implement
    - Handles empty text consistently using BaseValidator.handle_empty_text

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with StyleConfig
       - Store configuration for use during validation

    2. **Validation**: Check text against style constraints
       - Handle empty text using BaseValidator.handle_empty_text
       - Validate text against style constraints
       - Return RuleResult with validation results

    ## Error Handling

    - Empty text handling through BaseValidator.handle_empty_text
    - Abstract validate method requires implementation by subclasses
    - Configuration validation through StyleConfig

    ## Examples

    Creating a custom style validator:

    ```python
    from sifaka.rules.formatting.style import StyleValidator, StyleConfig, CapitalizationStyle
    from sifaka.rules.base import RuleResult

    class CustomStyleValidator(StyleValidator):
        def validate(self, text: str) -> RuleResult:
            # Handle empty text
            empty_result = self.handle_empty_text(text)
            if empty_result:
                return empty_result

            # Apply configuration
            if self.config.strip_whitespace:
                text = text.strip()

            # Validate capitalization
            if self.config.capitalization == CapitalizationStyle.LOWERCASE:
                if text != text.lower():
                    return RuleResult(
                        passed=False,
                        message="Text must be lowercase",
                        metadata={"errors": ["Text must be lowercase"]}
                    )

            return RuleResult(
                passed=True,
                message="Style validation successful",
                metadata={}
            )

    # Create and use the validator
    config = StyleConfig(capitalization=CapitalizationStyle.LOWERCASE)
    validator = CustomStyleValidator(config)
    result = validator.validate("this is a test")
    print(f"Valid: {result.passed}")
    ```
    """

    def __init__(self, config: StyleConfig):
        """
        Initialize validator with a configuration.

        Args:
            config: Style validation configuration
        """
        super().__init__(validation_type=str)
        self.config = config

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against style constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        raise NotImplementedError("Subclasses must implement validate method")


class DefaultStyleValidator(StyleValidator):
    """
    Default style validator delegating logic to analyzers.

    This class implements the StyleValidator interface by delegating validation
    logic to specialized analyzer components. It follows the Single Responsibility
    Principle by using separate analyzers for capitalization, ending characters,
    and disallowed characters.

    ## Architecture

    DefaultStyleValidator follows a component-based architecture:
    - Inherits from StyleValidator for common validation functionality
    - Uses specialized analyzers for different validation aspects:
      - _CapitalizationAnalyzer for capitalization validation
      - _EndingAnalyzer for ending character validation
      - _CharAnalyzer for disallowed character validation
    - Aggregates results from all analyzers

    ## Lifecycle

    1. **Initialization**: Set up with configuration and analyzers
       - Initialize with StyleConfig
       - Create specialized analyzers based on configuration
       - Store analyzers for use during validation

    2. **Validation**: Check text against style constraints
       - Handle empty text using BaseValidator.handle_empty_text
       - Apply whitespace stripping if configured
       - Delegate to analyzers for specific validations
       - Aggregate results from all analyzers
       - Return RuleResult with validation results

    ## Error Handling

    - Empty text handling through BaseValidator.handle_empty_text
    - Aggregates errors from all analyzers
    - Returns first error message as primary message
    - Includes all errors in metadata for detailed reporting

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.formatting.style import DefaultStyleValidator, StyleConfig, CapitalizationStyle

    # Create configuration
    config = StyleConfig(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
        disallowed_chars=['@', '#']
    )

    # Create validator
    validator = DefaultStyleValidator(config)

    # Validate text
    result = validator.validate("This is a test.")
    print(f"Valid: {result.passed}")

    # Check for specific errors
    if not result.passed:
        errors = result.metadata.get("errors", [])
        for error in errors:
            print(f"Error: {error}")
    ```
    """

    def __init__(self, config: StyleConfig):
        super().__init__(config)

        self._cap_analyzer = _CapitalizationAnalyzer(style=config.capitalization)
        self._end_analyzer = _EndingAnalyzer(
            require_end_punctuation=config.require_end_punctuation,
            allowed_end_chars=config.allowed_end_chars or [],
        )
        self._char_analyzer = _CharAnalyzer(disallowed=config.disallowed_chars or [])

    def validate(self, text: str) -> RuleResult:  # noqa: D401
        """
        Validate text style by delegating to analyzers.

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

        # Apply whitespace stripping if configured
        if self.config.strip_whitespace:
            text = text.strip()

        # Collect errors and suggestions
        errors: List[str] = []
        suggestions: List[str] = []

        # Validate capitalization
        if cap_err := self._cap_analyzer.analyze(text):
            errors.append(cap_err)
            if self.config.capitalization:
                suggestions.append(
                    f"Fix capitalization to match {self.config.capitalization.name} style"
                )

        # Validate ending
        if end_err := self._end_analyzer.analyze(text):
            errors.append(end_err)
            if self.config.allowed_end_chars:
                suggestions.append(
                    f"End text with one of: {', '.join(self.config.allowed_end_chars)}"
                )
            else:
                suggestions.append("End text with proper punctuation")

        # Validate disallowed characters
        disallowed_found = self._char_analyzer.analyze(text)
        if disallowed_found:
            errors.extend([f"Disallowed character found: '{ch}'" for ch in disallowed_found])
            if self.config.disallowed_chars:
                suggestions.append(
                    f"Remove disallowed characters: {', '.join(self.config.disallowed_chars)}"
                )

        # Create result
        result = RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Style validation successful",
            metadata={
                "errors": errors,
                "validator_type": self.__class__.__name__,
                "capitalization_style": (
                    self.config.capitalization.name if self.config.capitalization else None
                ),
                "require_end_punctuation": self.config.require_end_punctuation,
                "allowed_end_chars": self.config.allowed_end_chars,
                "disallowed_chars": self.config.disallowed_chars,
            },
            score=1.0 if not errors else 0.0,
            issues=errors,
            suggestions=suggestions,
            processing_time_ms=time.time() - start_time,
        )

        # Update statistics
        self.update_statistics(result)

        return result


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
    from sifaka.rules.formatting.style import StyleRule, StyleConfig, CapitalizationStyle
    from sifaka.rules.base import RuleConfig

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
    result = rule.validate("This is a test.")
    print(f"Valid: {result.passed}")

    # Check rule identification
    print(f"Rule ID: {result.metadata.get('rule_id')}")
    ```

    Using with custom validator:

    ```python
    from sifaka.rules.formatting.style import StyleRule, DefaultStyleValidator, StyleConfig

    # Create custom validator
    config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
    validator = DefaultStyleValidator(config)

    # Create rule with custom validator
    rule = StyleRule(
        name="custom_style_rule",
        validator=validator
    )

    # Validate text
    result = rule.validate("This is a test.")
    ```
    """

    def __init__(
        self,
        name: str = "style_rule",
        description: str = "Validates text style",
        config: Optional[RuleConfig] = None,
        validator: Optional[StyleValidator] = None,
        **kwargs,
    ):
        """
        Initialize the style rule.

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

        # Store the validator for reference
        self._style_validator = validator or self._create_default_validator()

    def _create_default_validator(self) -> StyleValidator:
        """Create a default validator from config."""
        # Extract style specific params
        params = self.config.params
        config = StyleConfig(
            capitalization=params.get("capitalization"),
            require_end_punctuation=params.get("require_end_punctuation", False),
            allowed_end_chars=params.get("allowed_end_chars"),
            disallowed_chars=params.get("disallowed_chars"),
            strip_whitespace=params.get("strip_whitespace", True),
        )
        return DefaultStyleValidator(config)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Evaluate text against style constraints.

        Args:
            text: The text to evaluate
            **kwargs: Additional validation context

        Returns:
            RuleResult containing validation results
        """
        # Delegate to validator
        result = self._validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._name)


def create_style_validator(
    capitalization: Optional[CapitalizationStyle] = None,
    require_end_punctuation: bool = False,
    allowed_end_chars: Optional[List[str]] = None,
    disallowed_chars: Optional[List[str]] = None,
    strip_whitespace: bool = True,
    **kwargs,
) -> StyleValidator:
    """Create a style validator with the specified constraints.

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
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    config = StyleConfig(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
        **rule_config_params,
    )

    return DefaultStyleValidator(config)


def create_style_rule(
    name: str = "style_rule",
    description: str = "Validates text style",
    capitalization: Optional[CapitalizationStyle] = None,
    require_end_punctuation: bool = False,
    allowed_end_chars: Optional[List[str]] = None,
    disallowed_chars: Optional[List[str]] = None,
    strip_whitespace: bool = True,
    **kwargs,
) -> StyleRule:
    """Create a style validation rule with the specified constraints.

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
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured StyleRule
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create validator using the validator factory
    validator = create_style_validator(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
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
    config = RuleConfig(params=params, **rule_config_params)

    # Create rule
    return StyleRule(
        name=name,
        description=description,
        config=config,
        validator=validator,
    )


class FormattingValidator(BaseValidator[str]):
    """
    Base class for text formatting validators.

    This abstract class defines the interface for formatting validators and provides
    common functionality. Formatting validators check text against formatting constraints
    such as whitespace handling, line normalization, and style requirements.

    ## Architecture

    FormattingValidator follows a component-based architecture:
    - Inherits from BaseValidator for common validation functionality
    - Uses FormattingConfig for configuration
    - Provides an abstract validate method for subclasses to implement
    - Handles empty text consistently using BaseValidator.handle_empty_text

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with FormattingConfig
       - Store configuration for use during validation

    2. **Validation**: Check text against formatting constraints
       - Handle empty text using BaseValidator.handle_empty_text
       - Validate text against formatting constraints
       - Return RuleResult with validation results

    ## Error Handling

    - Empty text handling through BaseValidator.handle_empty_text
    - Abstract validate method requires implementation by subclasses
    - Configuration validation through FormattingConfig

    ## Examples

    Creating a custom formatting validator:

    ```python
    from sifaka.rules.formatting.style import FormattingValidator, FormattingConfig
    from sifaka.rules.base import RuleResult
    import re

    class CustomFormattingValidator(FormattingValidator):
        def validate(self, text: str, **kwargs) -> RuleResult:
            # Handle empty text
            empty_result = self.handle_empty_text(text)
            if empty_result:
                return empty_result

            # Apply transformations if configured
            original_text = text

            if self.config.strip_whitespace:
                text = text.strip()

            if self.config.normalize_whitespace:
                text = re.sub(r"\\s+", " ", text)

            # Check if transformations were needed
            if text != original_text:
                return RuleResult(
                    passed=False,
                    message="Text formatting needs improvement",
                    metadata={
                        "original_text": original_text,
                        "formatted_text": text,
                        "changes_needed": True
                    }
                )

            return RuleResult(
                passed=True,
                message="Formatting validation successful",
                metadata={}
            )

    # Create and use the validator
    config = FormattingConfig(normalize_whitespace=True)
    validator = CustomFormattingValidator(config)
    result = validator.validate("This   is   a   test")
    print(f"Valid: {result.passed}")
    ```
    """

    def __init__(self, config: FormattingConfig):
        """
        Initialize validator with a configuration.

        Args:
            config: Formatting validation configuration
        """
        super().__init__(validation_type=str)
        self.config = config

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against formatting constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        raise NotImplementedError("Subclasses must implement validate method")


class DefaultFormattingValidator(FormattingValidator):
    """
    Default implementation of text formatting validator.

    This class implements the FormattingValidator interface with standard
    formatting validation logic. It handles whitespace normalization, line
    normalization, and delegates to StyleValidator for style validation
    if a style_config is provided.

    ## Architecture

    DefaultFormattingValidator follows a component-based architecture:
    - Inherits from FormattingValidator for common validation functionality
    - Uses regular expressions for text normalization
    - Delegates to DefaultStyleValidator for style validation if configured
    - Aggregates validation results

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with FormattingConfig
       - Store configuration for use during validation

    2. **Validation**: Check text against formatting constraints
       - Handle empty text using BaseValidator.handle_empty_text
       - Apply configured transformations (strip, normalize, remove extra lines)
       - Delegate to StyleValidator if style_config is provided
       - Aggregate validation results
       - Return RuleResult with validation results and metadata

    ## Error Handling

    - Empty text handling through BaseValidator.handle_empty_text
    - Aggregates errors from style validation
    - Includes detailed metadata about transformations

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.formatting.style import DefaultFormattingValidator, FormattingConfig

    # Create configuration
    config = FormattingConfig(
        strip_whitespace=True,
        normalize_whitespace=True,
        remove_extra_lines=True
    )

    # Create validator
    validator = DefaultFormattingValidator(config)

    # Validate text
    result = validator.validate("This   is  a   test  with    extra   spaces.")
    print(f"Valid: {result.passed}")

    # Check metadata
    if result.passed:
        print(f"Original length: {result.metadata.get('original_length')}")
        print(f"Formatted length: {result.metadata.get('formatted_length')}")
    ```

    Using with style validation:

    ```python
    from sifaka.rules.formatting.style import (
        DefaultFormattingValidator, FormattingConfig, StyleConfig, CapitalizationStyle
    )

    # Create style config
    style_config = StyleConfig(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True
    )

    # Create formatting config with style config
    config = FormattingConfig(
        style_config=style_config,
        normalize_whitespace=True
    )

    # Create validator
    validator = DefaultFormattingValidator(config)

    # Validate text
    result = validator.validate("this   is  not properly capitalized")

    # Check for errors
    if not result.passed:
        errors = result.metadata.get("errors", [])
        for error in errors:
            print(f"Error: {error}")
    ```
    """

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against formatting constraints.

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

        errors = []
        suggestions = []
        original_text = text

        # Apply transformations if configured
        if self.config.strip_whitespace:
            text = text.strip()

        if self.config.normalize_whitespace:
            text = replace_pattern(text, WHITESPACE_PATTERN, " ")
            if original_text != text:
                suggestions.append("Normalize whitespace (remove extra spaces)")

        if self.config.remove_extra_lines:
            text = replace_pattern(text, MULTIPLE_NEWLINES_PATTERN, "\n\n")
            if original_text != text:
                suggestions.append("Remove extra blank lines")

        # Validate against style config if provided
        if self.config.style_config:
            style_validator = DefaultStyleValidator(self.config.style_config)
            style_result = style_validator.validate(text)
            if not style_result.passed:
                errors.append(style_result.message)
                if style_result.suggestions:
                    suggestions.extend(style_result.suggestions)

        # Create result
        result = RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Formatting validation successful",
            metadata={
                "original_length": len(original_text),
                "formatted_length": len(text),
                "strip_whitespace": self.config.strip_whitespace,
                "normalize_whitespace": self.config.normalize_whitespace,
                "remove_extra_lines": self.config.remove_extra_lines,
                "validator_type": self.__class__.__name__,
                "has_style_validation": self.config.style_config is not None,
            },
            score=1.0 if not errors else 0.0,
            issues=errors,
            suggestions=suggestions,
            processing_time_ms=time.time() - start_time,
        )

        # Update statistics
        self.update_statistics(result)

        return result


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
    from sifaka.rules.formatting.style import FormattingRule
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
    result = rule.validate("This   is  a   test  with    extra   spaces.")
    print(f"Valid: {result.passed}")

    # Check rule identification
    print(f"Rule ID: {result.metadata.get('rule_id')}")
    ```

    Using with style configuration:

    ```python
    from sifaka.rules.formatting.style import FormattingRule, StyleConfig, CapitalizationStyle
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
    result = rule.validate("this   is  not properly capitalized")
    ```
    """

    def __init__(
        self,
        name: str = "formatting_rule",
        description: str = "Validates text formatting",
        config: Optional[RuleConfig] = None,
        validator: Optional[FormattingValidator] = None,
        **kwargs,
    ):
        """
        Initialize the formatting rule.

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

        # Store the validator for reference
        self._formatting_validator = validator or self._create_default_validator()

    def _create_default_validator(self) -> FormattingValidator:
        """
        Create a default validator from config.

        Returns:
            A configured FormattingValidator
        """
        # Extract formatting specific params
        params = self.config.params
        config = FormattingConfig(
            style_config=params.get("style_config"),
            strip_whitespace=params.get("strip_whitespace", True),
            normalize_whitespace=params.get("normalize_whitespace", False),
            remove_extra_lines=params.get("remove_extra_lines", False),
        )
        return DefaultFormattingValidator(config)


def create_formatting_validator(
    style_config: Optional[StyleConfig] = None,
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
        from sifaka.rules.formatting.style import create_formatting_validator

        # Create a basic validator
        validator = create_formatting_validator(normalize_whitespace=True)

        # Create a validator with style config
        from sifaka.rules.formatting.style import StyleConfig, CapitalizationStyle
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


def create_formatting_rule(
    name: str = "formatting_rule",
    description: str = "Validates text formatting",
    style_config: Optional[StyleConfig] = None,
    strip_whitespace: bool = True,
    normalize_whitespace: bool = False,
    remove_extra_lines: bool = False,
    rule_id: Optional[str] = None,
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
        from sifaka.rules.formatting.style import create_formatting_rule

        # Create a basic rule
        rule = create_formatting_rule(normalize_whitespace=True)

        # Create a rule with style config
        from sifaka.rules.formatting.style import StyleConfig, CapitalizationStyle
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


# ---------------------------------------------------------------------------
# Analyzer helpers (Single Responsibility)
# ---------------------------------------------------------------------------


class _CapitalizationAnalyzer(BaseModel):
    """
    Validate capitalization according to the configured style.

    This internal helper class analyzes text capitalization against a specified
    style requirement. It follows the Single Responsibility Principle by focusing
    solely on capitalization validation.

    ## Lifecycle

    1. **Initialization**: Set up with capitalization style
       - Initialize with CapitalizationStyle
       - Style can be None to skip validation

    2. **Analysis**: Check text against capitalization style
       - Return None if style is None or text is empty
       - Check text against the specified style
       - Return error message if validation fails
       - Return None if validation passes

    ## Examples

    ```python
    from sifaka.rules.formatting.style import _CapitalizationAnalyzer, CapitalizationStyle

    # Create analyzer for sentence case
    analyzer = _CapitalizationAnalyzer(style=CapitalizationStyle.SENTENCE_CASE)

    # Analyze text
    error = analyzer.analyze("This is a test.")  # Returns None (passes)
    error = analyzer.analyze("this is a test.")  # Returns error message

    # Handle results
    if error:
        print(f"Capitalization error: {error}")
    else:
        print("Capitalization is valid")
    ```
    """

    style: Optional["CapitalizationStyle"] = None  # noqa: F821 forward ref

    def analyze(self, text: str) -> Optional[str]:
        if self.style is None or not text:
            return None

        # Sentence case
        if self.style == CapitalizationStyle.SENTENCE_CASE:
            if not (text[0].isupper() and text[1:].islower()):
                return "Text should be in sentence case"

        # Title case (simple heuristic)
        elif self.style == CapitalizationStyle.TITLE_CASE:
            if any(word and word[0].islower() for word in text.split()):
                return "Text should be in title case"

        # Lowercase
        elif self.style == CapitalizationStyle.LOWERCASE:
            if text.lower() != text:
                return "Text should be all lowercase"

        # Uppercase
        elif self.style == CapitalizationStyle.UPPERCASE:
            if text.upper() != text:
                return "Text should be all uppercase"

        # Capitalize first
        elif self.style == CapitalizationStyle.CAPITALIZE_FIRST:
            if not (text[0].isupper() and text[1:] == text[1:]):
                # second condition basically always true; we accept any remainder
                return "Only the first character should be capitalized"

        return None


class _EndingAnalyzer(BaseModel):
    """
    Check ending punctuation requirements.

    This internal helper class analyzes text ending characters against specified
    requirements. It follows the Single Responsibility Principle by focusing
    solely on ending character validation.

    ## Lifecycle

    1. **Initialization**: Set up with ending requirements
       - Initialize with require_end_punctuation flag
       - Initialize with allowed_end_chars list

    2. **Analysis**: Check text ending against requirements
       - Return None if text is empty
       - Check if text ends with required punctuation
       - Check if text ends with allowed characters
       - Return error message if validation fails
       - Return None if validation passes

    ## Examples

    ```python
    from sifaka.rules.formatting.style import _EndingAnalyzer

    # Create analyzer requiring punctuation
    analyzer = _EndingAnalyzer(
        require_end_punctuation=True,
        allowed_end_chars=['.', '!', '?', ':']
    )

    # Analyze text
    error = analyzer.analyze("This is a test.")  # Returns None (passes)
    error = analyzer.analyze("This is a test")   # Returns error message

    # Handle results
    if error:
        print(f"Ending error: {error}")
    else:
        print("Ending is valid")
    ```

    Using with specific allowed endings:

    ```python
    # Create analyzer with specific allowed endings
    analyzer = _EndingAnalyzer(
        require_end_punctuation=False,
        allowed_end_chars=['A', 'B', 'C']
    )

    # Analyze text
    error = analyzer.analyze("This ends with A")  # Returns error message
    error = analyzer.analyze("This ends with C")  # Returns None (passes)
    ```
    """

    require_end_punctuation: bool = False
    allowed_end_chars: List[str] = Field(default_factory=list)

    def analyze(self, text: str) -> Optional[str]:
        if not text:
            return None

        end_char = text[-1]

        if (
            self.require_end_punctuation
            and end_char not in ".!?"
            and end_char not in self.allowed_end_chars
        ):
            return "Text must end with punctuation"

        if self.allowed_end_chars and end_char not in self.allowed_end_chars:
            return f"Text must end with one of {self.allowed_end_chars}"

        return None


class _CharAnalyzer(BaseModel):
    """
    Detect presence of disallowed characters.

    This internal helper class analyzes text for the presence of disallowed
    characters. It follows the Single Responsibility Principle by focusing
    solely on character presence validation.

    ## Lifecycle

    1. **Initialization**: Set up with disallowed characters
       - Initialize with list of disallowed characters
       - Empty list means no characters are disallowed

    2. **Analysis**: Check text for disallowed characters
       - Return empty list if no disallowed characters are found
       - Return list of disallowed characters found in the text

    ## Examples

    ```python
    from sifaka.rules.formatting.style import _CharAnalyzer

    # Create analyzer with disallowed characters
    analyzer = _CharAnalyzer(disallowed=['@', '#', '$'])

    # Analyze text
    found = analyzer.analyze("This is a test.")  # Returns [] (passes)
    found = analyzer.analyze("This is a #test.")  # Returns ['#']
    found = analyzer.analyze("This is a @#test.")  # Returns ['@', '#']

    # Handle results
    if found:
        print(f"Disallowed characters found: {', '.join(found)}")
    else:
        print("No disallowed characters found")
    ```

    Using with empty disallowed list:

    ```python
    # Create analyzer with no disallowed characters
    analyzer = _CharAnalyzer()  # or _CharAnalyzer(disallowed=[])

    # Analyze text
    found = analyzer.analyze("This can contain any characters @#$%^&*")
    print(f"Found: {found}")  # Will print "Found: []"
    ```
    """

    disallowed: List[str] = Field(default_factory=list)

    def analyze(self, text: str) -> List[str]:
        return [ch for ch in self.disallowed if ch in text]
