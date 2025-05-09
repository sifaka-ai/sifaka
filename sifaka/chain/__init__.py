"""
Chain module for Sifaka.

This module provides components for orchestrating the validation and improvement
flow between models, rules, and critics.

The module follows the Single Responsibility Principle by breaking down the chain
functionality into smaller, focused components:

1. ChainCore - Main interface that delegates to specialized components
2. ChainOrchestrator - Main user-facing class for a standardized implementation
3. PromptManager - Manages prompt creation and management
4. ValidationManager - Manages validation logic and rule management
5. RetryStrategy - Handles retry logic with different strategies
6. ResultFormatter - Handles formatting and processing of results

It also provides factory functions for creating different types of chains:
- create_simple_chain - Creates a simple chain with a fixed number of retries
- create_backoff_chain - Creates a chain with exponential backoff retry strategy

The module also provides interfaces for each component type:
- Chain - Interface for chains
- PromptManagerProtocol - Interface for prompt managers
- ValidationManagerProtocol - Interface for validation managers
- RetryStrategyProtocol - Interface for retry strategies
- ResultFormatterProtocol - Interface for result formatters
- CriticProtocol - Interface for critics
"""

# Interfaces
from .interfaces.chain import Chain, AsyncChain
from .interfaces.critic import CriticProtocol
from .interfaces.formatter import ResultFormatterProtocol
from .interfaces.manager import PromptManagerProtocol, ValidationManagerProtocol
from .interfaces.strategy import RetryStrategyProtocol

# Core components
from .core import ChainCore
from .orchestrator import ChainOrchestrator
from .result import ChainResult
from .factories import create_simple_chain, create_backoff_chain
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .strategies.retry import RetryStrategy, SimpleRetryStrategy, BackoffRetryStrategy

__all__ = [
    # Interfaces
    "Chain",
    "AsyncChain",
    "CriticProtocol",
    "ResultFormatterProtocol",
    "PromptManagerProtocol",
    "ValidationManagerProtocol",
    "RetryStrategyProtocol",
    # Core components
    "ChainCore",
    "ChainOrchestrator",
    "ChainResult",
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
