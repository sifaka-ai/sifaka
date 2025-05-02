"""
Chain module for Sifaka.

This module provides components for orchestrating the validation and improvement
flow between models, rules, and critics.

The module follows the Single Responsibility Principle by breaking down the chain
functionality into smaller, focused components:

1. ChainCore - Main interface that delegates to specialized components
2. PromptManager - Manages prompt creation and management
3. ValidationManager - Manages validation logic and rule management
4. RetryStrategy - Handles retry logic with different strategies
5. ResultFormatter - Handles formatting and processing of results

It also provides factory functions for creating different types of chains:
- create_simple_chain - Creates a simple chain with a fixed number of retries
- create_backoff_chain - Creates a chain with exponential backoff retry strategy
"""

# Legacy components for backward compatibility
from .compat import Chain
from .result import ChainResult

# New components
from .core import ChainCore
from .factories import create_simple_chain, create_backoff_chain
from .formatters import ResultFormatter
from .managers import PromptManager, ValidationManager
from .strategies import RetryStrategy, SimpleRetryStrategy, BackoffRetryStrategy

__all__ = [
    # Legacy components
    "Chain",
    "ChainResult",
    # New components
    "ChainCore",
    # Factory functions
    "create_simple_chain",
    "create_backoff_chain",
    # Managers
    "PromptManager",
    "ValidationManager",
    # Strategies
    "RetryStrategy",
    "SimpleRetryStrategy",
    "BackoffRetryStrategy",
    # Formatters
    "ResultFormatter",
]
