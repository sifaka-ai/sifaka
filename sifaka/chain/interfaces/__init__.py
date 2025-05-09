"""
Chain interface protocols for Sifaka.

This package defines the interfaces for chain components in the Sifaka framework.
These interfaces establish a common contract for chain component behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Chain**: Base interface for all chains
2. **PromptManager**: Interface for prompt managers
3. **ValidationManager**: Interface for validation managers
4. **RetryStrategy**: Interface for retry strategies
5. **ResultFormatter**: Interface for result formatters
6. **Critic**: Interface for critics

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from .chain import Chain, AsyncChain
from .critic import CriticProtocol
from .formatter import ResultFormatterProtocol
from .manager import PromptManagerProtocol, ValidationManagerProtocol
from .strategy import RetryStrategyProtocol

__all__ = [
    "Chain",
    "AsyncChain",
    "CriticProtocol",
    "ResultFormatterProtocol",
    "PromptManagerProtocol",
    "ValidationManagerProtocol",
    "RetryStrategyProtocol",
]
