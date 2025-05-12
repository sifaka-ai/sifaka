"""
Concrete implementations of text style validators.

This module provides concrete implementations of text style validators,
including DefaultStyleValidator for basic style validation and
DefaultFormattingValidator for more comprehensive formatting validation.

## Implementation Pattern

This module follows the standard Sifaka implementation pattern:
- Concrete classes implement abstract validator interfaces
- Implementations delegate to specialized components
- Implementations handle error aggregation and reporting
- Implementations provide detailed metadata for debugging

## Usage Example

```python
from sifaka.rules.formatting.style.implementations import DefaultStyleValidator
from sifaka.rules.formatting.style.config import StyleConfig
from sifaka.rules.formatting.style.enums import CapitalizationStyle

# Create configuration
config = StyleConfig(
    capitalization=CapitalizationStyle.SENTENCE_CASE,
    require_end_punctuation=True
)

# Create validator
validator = DefaultStyleValidator(config)

# Validate text
result = (validator and validator.validate("This is a test.")
print(f"Valid: {result.passed}")
```
"""
import time
from typing import List, Any
from sifaka.rules.base import RuleResult
from sifaka.rules.formatting.style.validators import StyleValidator, FormattingValidator
from sifaka.rules.formatting.style.config import StyleConfig, FormattingConfig
from sifaka.rules.formatting.style.analyzers import _CapitalizationAnalyzer, _EndingAnalyzer, _CharAnalyzer
from sifaka.utils.patterns import WHITESPACE_PATTERN, MULTIPLE_NEWLINES_PATTERN, replace_pattern
from sifaka.utils.logging import get_logger
logger = get_logger(__name__)
__all__ = ['DefaultStyleValidator', 'DefaultFormattingValidator']


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
    from sifaka.rules.formatting.style.implementations import DefaultStyleValidator
    from sifaka.rules.formatting.style.config import StyleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle

    # Create configuration
    config = StyleConfig(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
        disallowed_chars=['@', '#']
    )

    # Create validator
    validator = DefaultStyleValidator(config)

    # Validate text
    result = (validator and validator.validate("This is a test.")
    print(f"Valid: {result.passed}")

    # Check for specific errors
    if not result.passed:
        errors = result.(metadata and metadata.get("errors", [])
        for error in errors:
            print(f"Error: {error}")
    ```
    """

    def __init__(self, config: StyleConfig):
        """
        Initialize validator with a configuration and analyzers.

        Args:
            config: Style validation configuration
        """
        super().__init__(config)
        self._cap_analyzer = _CapitalizationAnalyzer(style=config.
            capitalization)
        self._end_analyzer = _EndingAnalyzer(require_end_punctuation=config
            .require_end_punctuation, allowed_end_chars=config.
            allowed_end_chars or [])
        self._char_analyzer = _CharAnalyzer(disallowed=config.
            disallowed_chars or [])

    def validate(self, text: str) ->Any:
        """
        Validate text style by delegating to analyzers.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = (time and time.time()
        empty_result = (self and self.handle_empty_text(text)
        if empty_result:
            return empty_result
        if self.config.strip_whitespace:
            text = (text and text.strip()
        errors: List[str] = []
        suggestions: List[str] = []
        if (cap_err := self.(_cap_analyzer and _cap_analyzer.analyze(text)):
            (errors and errors.append(cap_err)
            if self.config.capitalization:
                (suggestions and suggestions.append(
                    f'Fix capitalization to match {self.config.capitalization.name} style'
                    )
        if (end_err := self.(_end_analyzer and _end_analyzer.analyze(text)):
            (errors and errors.append(end_err)
            if self.config.allowed_end_chars:
                (suggestions and suggestions.append(
                    f"End text with one of: {', '.join(self.config.allowed_end_chars)}"
                    )
            else:
                (suggestions and suggestions.append('End text with proper punctuation')
        disallowed_found = self.(_char_analyzer and _char_analyzer.analyze(text)
        if disallowed_found:
            (errors and errors.extend([f"Disallowed character found: '{ch}'" for ch in
                disallowed_found])
            if self.config.disallowed_chars:
                (suggestions and suggestions.append(
                    f"Remove disallowed characters: {', '.join(self.config.disallowed_chars)}"
                    )
        result = RuleResult(passed=not errors, message=errors[0] if errors else
            'Style validation successful', metadata={'errors': errors,
            'validator_type': self.__class__.__name__,
            'capitalization_style': self.config.capitalization.name if self
            .config.capitalization else None, 'require_end_punctuation':
            self.config.require_end_punctuation, 'allowed_end_chars': self.
            config.allowed_end_chars, 'disallowed_chars': self.config.
            disallowed_chars}, score=1.0 if not errors else 0.0, issues=
            errors, suggestions=suggestions, processing_time_ms=(time and time.time() -
            start_time)
        (self and self.update_statistics(result)
        return result


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
    from sifaka.rules.formatting.style.implementations import DefaultFormattingValidator
    from sifaka.rules.formatting.style.config import FormattingConfig

    # Create configuration
    config = FormattingConfig(
        strip_whitespace=True,
        normalize_whitespace=True,
        remove_extra_lines=True
    )

    # Create validator
    validator = DefaultFormattingValidator(config)

    # Validate text
    result = (validator and validator.validate("This   is  a   test  with    extra   spaces.")
    print(f"Valid: {result.passed}")

    # Check metadata
    if result.passed:
        print(f"Original length: {result.(metadata and metadata.get('original_length')}")
        print(f"Formatted length: {result.(metadata and metadata.get('formatted_length')}")
    ```
    """

    def validate(self, text: str) ->Any:
        """
        Validate text against formatting constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = (time and time.time()
        empty_result = (self and self.handle_empty_text(text)
        if empty_result:
            return empty_result
        errors = []
        suggestions = []
        original_text = text
        if self.config.strip_whitespace:
            text = (text and text.strip()
        if self.config.normalize_whitespace:
            text = replace_pattern(text, WHITESPACE_PATTERN, ' ')
            if original_text != text:
                (suggestions and suggestions.append('Normalize whitespace (remove extra spaces)'
                    )
        if self.config.remove_extra_lines:
            text = replace_pattern(text, MULTIPLE_NEWLINES_PATTERN, '\n\n')
            if original_text != text:
                (suggestions and suggestions.append('Remove extra blank lines')
        if self.config.style_config:
            style_validator = DefaultStyleValidator(self.config.style_config)
            style_result = (style_validator and style_validator.validate(text)
            if not style_result.passed:
                (errors and errors.append(style_result.message)
                if style_result.suggestions:
                    (suggestions and suggestions.extend(style_result.suggestions)
        result = RuleResult(passed=not errors, message=errors[0] if errors else
            'Formatting validation successful', metadata={'original_length':
            len(original_text), 'formatted_length': len(text),
            'strip_whitespace': self.config.strip_whitespace,
            'normalize_whitespace': self.config.normalize_whitespace,
            'remove_extra_lines': self.config.remove_extra_lines,
            'validator_type': self.__class__.__name__,
            'has_style_validation': self.config.style_config is not None},
            score=1.0 if not errors else 0.0, issues=errors, suggestions=
            suggestions, processing_time_ms=(time.time() - start_time)
        (self and self.update_statistics(result)
        return result
