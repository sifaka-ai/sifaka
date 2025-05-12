from typing import Any, List
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
result = (rule and rule.validate("This is a test.")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""
from sifaka.rules.formatting.style.enums import CapitalizationStyle
from sifaka.rules.formatting.style.config import StyleConfig, FormattingConfig
from sifaka.rules.formatting.style.validators import StyleValidator, FormattingValidator
from sifaka.rules.formatting.style.implementations import DefaultStyleValidator, DefaultFormattingValidator
from sifaka.rules.formatting.style.rules import StyleRule, FormattingRule
from sifaka.rules.formatting.style.factories import create_style_validator, create_style_rule, create_formatting_validator, create_formatting_rule
from sifaka.rules.formatting.style.analyzers import _CapitalizationAnalyzer, _EndingAnalyzer, _CharAnalyzer
__all__: List[Any] = ['CapitalizationStyle', 'StyleConfig',
    'FormattingConfig', 'StyleValidator', 'DefaultStyleValidator',
    'FormattingValidator', 'DefaultFormattingValidator', 'StyleRule',
    'FormattingRule', 'create_style_validator', 'create_style_rule',
    'create_formatting_validator', 'create_formatting_rule',
    '_CapitalizationAnalyzer', '_EndingAnalyzer', '_CharAnalyzer']
