"""
Formatting rules for Sifaka.

This package provides rules for validating text formatting:
- Length: Length validation for text
- Style: Style validation for text
"""

from .length import (
    LengthRule,
    LengthConfig,
    DefaultLengthValidator,
    create_length_rule,
)

__all__ = [
    "LengthRule",
    "LengthConfig",
    "DefaultLengthValidator",
    "create_length_rule",
]
