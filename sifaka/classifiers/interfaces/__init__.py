"""
Interfaces for classifiers.

This package provides protocol interfaces for classifiers in the Sifaka framework.
"""

from .classifier import (
    ClassifierProtocol,
    TextProcessor,
)

__all__ = [
    "ClassifierProtocol",
    "TextProcessor",
]
