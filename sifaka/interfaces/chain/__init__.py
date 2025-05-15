"""
Chain interfaces for Sifaka.

This module defines the interfaces for chains in the Sifaka framework.
These interfaces establish a common contract for chain behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Chain**: Base interface for all chains
   - **PromptManager**: Interface for prompt managers
   - **ValidationManager**: Interface for validation managers
   - **RetryStrategy**: Interface for retry strategies
   - **ResultFormatter**: Interface for result formatters
   - **Model**: Interface for text generation models
   - **Validator**: Interface for output validators
   - **Improver**: Interface for output improvers
   - **ChainFormatter**: Interface for result formatters
   - **ChainComponent**: Base interface for all chain components
   - **ChainPlugin**: Interface for chain plugins

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

## State Management

The interfaces support standardized state management:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state manager methods
- Clear separation of configuration and state

## Error Handling

The interfaces define error handling patterns:
- ValueError for invalid inputs
- RuntimeError for execution failures
- TypeError for type mismatches
- ModelError: Raised when text generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails
- Detailed error tracking and reporting

## Execution Tracking

The interfaces support execution tracking:
- Execution count tracking
- Execution time tracking
- Success/failure tracking
- Performance statistics
"""

from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union, runtime_checkable

# Import interfaces
from .model import Model
from .validator import Validator
from .improver import Improver
from .formatter import ChainFormatter
from .components import ChainComponent
from .results import ValidationResult
from .chain import Chain
from .plugin import ChainPlugin
from .manager import PromptManager, ValidationManager
from .retry_strategy import RetryStrategy
from .result_formatter import ResultFormatter

# Type exports for type checking
ChainType = TypeVar("ChainType")
ResultType = TypeVar("ResultType")
RetryStrategyType = TypeVar("RetryStrategyType")
ValidationManagerType = TypeVar("ValidationManagerType")
ResultFormatterType = TypeVar("ResultFormatterType")
PromptManagerType = TypeVar("PromptManagerType")
ModelType = TypeVar("ModelType")
ValidatorType = TypeVar("ValidatorType")
ImproverType = TypeVar("ImproverType")
FormatterType = TypeVar("FormatterType")
PluginType = TypeVar("PluginType")

# Define exports
__all__ = [
    "Chain",
    "Model",
    "Validator",
    "Improver",
    "ChainFormatter",
    "ChainComponent",
    "ValidationResult",
    "ChainPlugin",
    "PromptManager",
    "ValidationManager",
    "RetryStrategy",
    "ResultFormatter",
]
