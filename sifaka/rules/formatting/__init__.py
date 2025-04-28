"""
Formatting rules package for text validation.

This package provides validators and rules for checking various text formatting
constraints including style, whitespace, and other formatting standards.

.. deprecated:: 1.0.0
   The monolithic formatting approach is deprecated and will be removed in version 2.0.0.
   Use the following module-specific imports instead:

   - :mod:`sifaka.rules.formatting.format` for format validation
   - :mod:`sifaka.rules.formatting.style` for style and structure validation
   - :mod:`sifaka.rules.formatting.length` for text length validation
   - :mod:`sifaka.rules.formatting.whitespace` for whitespace validation

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.formatting import LengthRule, StyleRule, etc.
   - New: from sifaka.rules.formatting.length import LengthRule
         from sifaka.rules.formatting.style import StyleRule
         etc.

2. Update configuration:
   - Each formatting aspect now has its own configuration class
   - Each has its own set of parameters and validation logic
   - See the respective module documentation for details

Example:
    Old code:
    >>> from sifaka.rules.formatting import StyleRule
    >>> rule = StyleRule()

    New code:
    >>> from sifaka.rules.formatting.style import StyleRule
    >>> rule = StyleRule()
"""

import warnings

# Re-export classes for backward compatibility
from sifaka.rules.formatting.format import (
    DefaultFormatValidator,
    DefaultJsonValidator,
    DefaultMarkdownValidator,
    DefaultPlainTextValidator,
    FormatConfig,
    FormatRule,
    FormatType,
    FormatValidator,
    JsonConfig,
    JsonValidator,
    MarkdownConfig,
    MarkdownValidator,
    PlainTextConfig,
    PlainTextValidator,
    create_format_rule,
    create_json_rule,
    create_markdown_rule,
    create_plain_text_rule,
)
from sifaka.rules.formatting.length import (
    DefaultLengthValidator,
    LengthConfig,
    LengthRule,
    LengthValidator,
    create_length_rule,
)
from sifaka.rules.formatting.style import (
    CapitalizationStyle,
    DefaultFormattingValidator,
    DefaultStyleValidator,
    FormattingConfig,
    FormattingRule,
    FormattingValidator,
    StyleConfig,
    StyleRule,
    StyleValidator,
    create_formatting_rule,
    create_style_rule,
)
from sifaka.rules.formatting.whitespace import (
    WhitespaceConfig,
    WhitespaceValidator,
    DefaultWhitespaceValidator,
    WhitespaceRule,
    create_whitespace_rule,
)

warnings.warn(
    "Importing directly from sifaka.rules.formatting is deprecated and will be removed in version 2.0.0. "
    "Use specific module imports instead: "
    "sifaka.rules.formatting.format, sifaka.rules.formatting.style, "
    "sifaka.rules.formatting.length, and sifaka.rules.formatting.whitespace.",
    DeprecationWarning,
    stacklevel=2,
)

# Export public classes and functions
__all__ = [
    # Format validators and rules
    "FormatRule",
    "FormatType",
    "FormatConfig",
    "MarkdownConfig",
    "JsonConfig",
    "PlainTextConfig",
    "DefaultFormatValidator",
    "FormatValidator",
    "MarkdownValidator",
    "DefaultMarkdownValidator",
    "JsonValidator",
    "DefaultJsonValidator",
    "PlainTextValidator",
    "DefaultPlainTextValidator",
    "create_format_rule",
    "create_markdown_rule",
    "create_json_rule",
    "create_plain_text_rule",
    # Length validators and rules
    "LengthRule",
    "LengthConfig",
    "LengthValidator",
    "DefaultLengthValidator",
    "create_length_rule",
    # Style validators and rules
    "CapitalizationStyle",
    "StyleConfig",
    "StyleValidator",
    "DefaultStyleValidator",
    "StyleRule",
    "create_style_rule",
    # Whitespace validators and rules
    "WhitespaceConfig",
    "WhitespaceValidator",
    "DefaultWhitespaceValidator",
    "WhitespaceRule",
    "create_whitespace_rule",
    # Formatting validators and rules
    "FormattingRule",
    "FormattingConfig",
    "FormattingValidator",
    "DefaultFormattingValidator",
    "create_formatting_rule",
]
