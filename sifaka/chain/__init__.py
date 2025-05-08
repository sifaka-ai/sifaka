"""
Chain module for Sifaka.

This module provides components for orchestrating the validation and improvement
flow between models, rules, and critics using the composition over inheritance pattern.

The module follows the Single Responsibility Principle by breaking down the chain
functionality into smaller, focused components:

1. Chain - Main class that delegates to specialized implementations
2. ChainImplementation - Protocol for chain implementations
3. PromptManager - Manages prompt creation and management
4. ValidationManager - Manages validation logic and rule management
5. RetryStrategy - Handles retry logic with different strategies
6. ResultFormatter - Handles formatting and processing of results

It also provides factory functions for creating different types of chains:
- create_simple_chain - Creates a simple chain with a fixed number of retries
- create_backoff_chain - Creates a chain with exponential backoff retry strategy
"""

# Core components
from .implementation import Chain, ChainImplementation
from .implementations import SimpleChainImplementation, BackoffChainImplementation
from .result import ChainResult
from .factories import create_simple_chain, create_backoff_chain
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .strategies.retry import RetryStrategy, SimpleRetryStrategy, BackoffRetryStrategy
from .config import ChainConfig

__all__ = [
    # Core components
    "Chain",
    "ChainImplementation",
    "ChainResult",
    "ChainConfig",
    # Implementations
    "SimpleChainImplementation",
    "BackoffChainImplementation",
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
