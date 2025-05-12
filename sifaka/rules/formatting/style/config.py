"""
Configuration classes for text style validation.

This module provides configuration classes for text style validation,
including StyleConfig for basic style validation and FormattingConfig
for more comprehensive formatting validation.

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- Configuration is stored in dedicated config classes
- Factory functions handle configuration extraction
- Validator factory functions create standalone validators
- Rule factory functions use validator factory functions internally

## Usage Example

```python
from sifaka.rules.formatting.style.config import StyleConfig, FormattingConfig
from sifaka.rules.formatting.style.enums import CapitalizationStyle

# Create style configuration
style_config = StyleConfig(
    capitalization=CapitalizationStyle.SENTENCE_CASE,
    require_end_punctuation=True
)

# Create formatting configuration with style config
formatting_config = FormattingConfig(
    style_config=style_config,
    normalize_whitespace=True
)
```
"""

from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.formatting.style.enums import CapitalizationStyle

__all__ = [
    "StyleConfig",
    "FormattingConfig",
]


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
    from sifaka.rules.formatting.style.config import StyleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle

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
    from sifaka.rules.formatting.style.config import FormattingConfig, StyleConfig
    from sifaka.rules.formatting.style.enums import CapitalizationStyle

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
