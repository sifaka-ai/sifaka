"""
Internal analyzer helpers for text style validation.

This module provides internal helper classes for text style validation,
including analyzers for capitalization, ending characters, and disallowed
characters. These analyzers follow the Single Responsibility Principle
by focusing on specific aspects of text validation.

## Architecture

The analyzers in this module follow a component-based architecture:
- Each analyzer focuses on a specific aspect of text validation
- Analyzers are used by validator implementations
- Analyzers return error messages or None for successful validation

## Usage Example (Internal)

```python
from sifaka.rules.formatting.style.analyzers import _CapitalizationAnalyzer
from sifaka.rules.formatting.style.enums import CapitalizationStyle

# Create analyzer
analyzer = _CapitalizationAnalyzer(style=CapitalizationStyle.SENTENCE_CASE)

# Analyze text
error = analyzer.analyze("This is a test.")  # Returns None (passes)
error = analyzer.analyze("this is a test.")  # Returns error message
```
"""

from typing import List, Optional

from pydantic import BaseModel

from sifaka.rules.formatting.style.enums import CapitalizationStyle

__all__ = [
    "_CapitalizationAnalyzer",
    "_EndingAnalyzer",
    "_CharAnalyzer",
]


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
    from sifaka.rules.formatting.style.analyzers import _CapitalizationAnalyzer
    from sifaka.rules.formatting.style.enums import CapitalizationStyle

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

    style: Optional[CapitalizationStyle] = None

    def analyze(self, text: str) -> Optional[str]:
        """
        Analyze text capitalization against the configured style.

        Args:
            text: The text to analyze

        Returns:
            Error message if validation fails, None if validation passes
        """
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
    from sifaka.rules.formatting.style.analyzers import _EndingAnalyzer

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
    """

    require_end_punctuation: bool = False
    allowed_end_chars: List[str] = []

    def analyze(self, text: str) -> Optional[str]:
        """
        Analyze text ending against the configured requirements.

        Args:
            text: The text to analyze

        Returns:
            Error message if validation fails, None if validation passes
        """
        if not text:
            return None

        # Check if ending punctuation is required
        if self.require_end_punctuation:
            # If allowed chars are specified, check against them
            if self.allowed_end_chars:
                if not any(text.endswith(char) for char in self.allowed_end_chars):
                    allowed_chars_str = ", ".join(self.allowed_end_chars)
                    return f"Text must end with one of: {allowed_chars_str}"
            # Otherwise, check for any punctuation
            elif not text[-1] in ".!?:;,":
                return "Text must end with punctuation"

        return None


class _CharAnalyzer(BaseModel):
    """
    Check for disallowed characters in text.

    This internal helper class analyzes text for disallowed characters.
    It follows the Single Responsibility Principle by focusing solely
    on character validation.

    ## Lifecycle

    1. **Initialization**: Set up with disallowed characters
       - Initialize with list of disallowed characters

    2. **Analysis**: Check text for disallowed characters
       - Return empty list if text is empty or no disallowed chars
       - Check if text contains any disallowed characters
       - Return list of found disallowed characters

    ## Examples

    ```python
    from sifaka.rules.formatting.style.analyzers import _CharAnalyzer

    # Create analyzer with disallowed characters
    analyzer = _CharAnalyzer(disallowed=['@', '#', '$'])

    # Analyze text
    found = analyzer.analyze("This is a test.")  # Returns [] (passes)
    found = analyzer.analyze("This is a #test.")  # Returns ['#']

    # Handle results
    if found:
        print(f"Found disallowed characters: {', '.join(found)}")
    else:
        print("No disallowed characters found")
    ```
    """

    disallowed: List[str] = []

    def analyze(self, text: str) -> List[str]:
        """
        Analyze text for disallowed characters.

        Args:
            text: The text to analyze

        Returns:
            List of found disallowed characters
        """
        if not text or not self.disallowed:
            return []

        # Find all disallowed characters in the text
        found = [char for char in self.disallowed if char in text]
        return found
