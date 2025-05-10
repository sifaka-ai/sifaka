"""
Interfaces for classifiers.

This package provides protocol interfaces for classifiers in the Sifaka framework.
These interfaces are imported from the main interfaces directory.
"""

# Import from the main interfaces directory
from sifaka.interfaces.classifier import (
    ClassifierProtocol,
    TextProcessor,
)

__all__ = [
    "ClassifierProtocol",
    "TextProcessor",
]
