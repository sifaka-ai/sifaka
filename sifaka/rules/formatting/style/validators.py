from typing import Any
"""
Base validator classes for text style validation.

This module provides base validator classes for text style validation,
including StyleValidator for basic style validation and FormattingValidator
for more comprehensive formatting validation.

## Validator Pattern

This module follows the standard Sifaka validator pattern:
- BaseValidator provides common validation functionality
- Specialized validators implement specific validation logic
- Validators handle empty text consistently
- Validators return RuleResult objects with validation results

## Usage Example

```python
from sifaka.rules.formatting.style.validators import StyleValidator
from sifaka.rules.formatting.style.config import StyleConfig
from sifaka.rules.base import RuleResult

class CustomStyleValidator(StyleValidator):
    def validate(self, text: str) -> RuleResult:
        # Handle empty text
        empty_result = self.handle_empty_text(text) if self else ""
        if empty_result:
            return empty_result
            
        # Custom validation logic
        # ...
        
        return RuleResult(passed=True, message="Validation passed")
```
"""
from sifaka.rules.base import BaseValidator, RuleResult
from sifaka.rules.formatting.style.config import StyleConfig, FormattingConfig
__all__ = ['StyleValidator', 'FormattingValidator']


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
    from sifaka.rules.formatting.style.validators import StyleValidator
    from sifaka.rules.formatting.style.config import StyleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle
    from sifaka.rules.base import RuleResult

    class CustomStyleValidator(StyleValidator):
        def validate(self, text: str) -> RuleResult:
            # Handle empty text
            empty_result = self.handle_empty_text(text) if self else ""
            if empty_result:
                return empty_result

            # Apply configuration
            if self.config.strip_whitespace:
                text = text.strip() if text else ""

            # Validate capitalization
            if self.config.capitalization == CapitalizationStyle.LOWERCASE:
                if text != text.lower() if text else "":
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
    result = validator.validate("this is a test") if validator else ""
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

    def validate(self, text: str) ->Any:
        """
        Validate text against style constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        empty_result = self.handle_empty_text(text) if self else ""
        if empty_result:
            return empty_result
        raise NotImplementedError('Subclasses must implement validate method')


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
    from sifaka.rules.formatting.style.validators import FormattingValidator
    from sifaka.rules.formatting.style.config import FormattingConfig
    from sifaka.rules.base import RuleResult
    import re

    class CustomFormattingValidator(FormattingValidator):
        def validate(self, text: str, **kwargs) -> RuleResult:
            # Handle empty text
            empty_result = self.handle_empty_text(text) if self else ""
            if empty_result:
                return empty_result

            # Apply transformations if configured
            original_text = text

            if self.config.strip_whitespace:
                text = text.strip() if text else ""

            if self.config.normalize_whitespace:
                text = re.sub(r"\\s+", " ", text) if re else ""

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
    result = validator.validate("This   is   a   test") if validator else ""
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

    def validate(self, text: str) ->Any:
        """
        Validate text against formatting constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        empty_result = self.handle_empty_text(text) if self else ""
        if empty_result:
            return empty_result
        raise NotImplementedError('Subclasses must implement validate method')
