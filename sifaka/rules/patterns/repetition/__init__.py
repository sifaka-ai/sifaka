"""
Repetition pattern validation.

This module provides functionality for detecting and validating repetitive patterns in text,
including exact repetitions, alternating patterns, and custom patterns.
"""

from .config import RepetitionConfig
from .validator import RepetitionValidator
from .rule import RepetitionRule

__all__ = [
    "RepetitionConfig",
    "RepetitionValidator",
    "RepetitionRule",
]
