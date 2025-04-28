"""
Format validation rules for Sifaka.

This module provides rules for validating text format including markdown, JSON, and plain text.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use the sifaka.rules.formatting.format module instead.
"""

import warnings
from typing import Any, Dict, Optional, Union

# Re-export classes and functions from the new location
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

warnings.warn(
    "The format module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.formatting.format instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Export public classes and functions
__all__ = [
    "FormatRule",
    "FormatType",
    "FormatConfig",
    "MarkdownConfig",
    "JsonConfig",
    "PlainTextConfig",
    "FormatValidator",
    "DefaultFormatValidator",
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
]
