"""
Manager interfaces for chain components.

This package provides interfaces for managers in the chain system.
These interfaces establish a common contract for manager behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **PromptManager**: Interface for prompt managers
2. **ValidationManager**: Interface for validation managers
3. **RetryStrategy**: Interface for retry strategies
4. **ResultFormatter**: Interface for result formatters
"""

from .formatter import ResultFormatter
from .prompt import PromptManager
from .retry import RetryStrategy
from .validation import ValidationManager

__all__ = [
    "PromptManager",
    "ValidationManager",
    "RetryStrategy",
    "ResultFormatter",
]
