"""
Chain module for Sifaka.

This module provides components for orchestrating the validation and improvement
flow between models, rules, and critics.

The module follows the Single Responsibility Principle by breaking down the chain
functionality into smaller, focused components:

1. Chain - Main user-facing class that provides a simple interface
2. ChainOrchestrator - Coordinates the overall flow between components
3. ChainExecutor - Executes a single generation-validation-improvement cycle
4. FeedbackFormatter - Formats feedback for the model
5. RetryManager - Manages retry logic and attempt tracking
"""

from .base import Chain
from .result import ChainResult
from .executor import ChainExecutor
from .orchestrator import ChainOrchestrator
from .feedback import FeedbackFormatter
from .retry import RetryManager

__all__ = [
    "Chain",
    "ChainResult",
    "ChainExecutor",
    "ChainOrchestrator",
    "FeedbackFormatter",
    "RetryManager",
]
