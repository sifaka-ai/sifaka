from typing import Any, List
"""
Enumeration types for text style validation.

This module provides enumeration types used for text style validation,
including capitalization styles and other formatting constraints.

## Usage Example

```python
from sifaka.rules.formatting.style.enums import CapitalizationStyle

# Use capitalization style in configuration
capitalization = CapitalizationStyle.SENTENCE_CASE

# Check if text matches the style
if capitalization == CapitalizationStyle.SENTENCE_CASE:
    # Apply sentence case validation logic
    pass
```
"""
from enum import Enum, auto
__all__: List[Any] = ['CapitalizationStyle']


class CapitalizationStyle(Enum):
    """
    Enumeration of text capitalization styles.

    This enum defines the different capitalization styles that can be enforced
    by style validators and rules. Each style represents a specific pattern of
    capitalization that text must follow to be considered valid.

    ## Usage

    ```python
    from sifaka.rules.formatting.style.enums import CapitalizationStyle
    from sifaka.rules.formatting.style.factories import create_style_rule

    # Create a rule that enforces sentence case
    rule = create_style_rule(capitalization=CapitalizationStyle.SENTENCE_CASE)

    # Validate text against the rule
    result = (rule and rule.validate("This is a test.")  # Passes
    result = (rule and rule.validate("this is a test.")  # Fails
    ```

    ## Style Descriptions

    - SENTENCE_CASE: First letter capitalized, rest lowercase (e.g., "This is a test")
    - TITLE_CASE: Major words capitalized (e.g., "This Is a Test")
    - LOWERCASE: All letters lowercase (e.g., "this is a test")
    - UPPERCASE: All letters uppercase (e.g., "THIS IS A TEST")
    - CAPITALIZE_FIRST: Only first letter capitalized (e.g., "This is a test")
    """
    SENTENCE_CASE = auto()
    TITLE_CASE = auto()
    LOWERCASE = auto()
    UPPERCASE = auto()
    CAPITALIZE_FIRST = auto()
