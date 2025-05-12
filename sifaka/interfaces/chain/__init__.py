from typing import Any, List
"""
Chain interfaces for Sifaka.

This package provides interfaces for chains in the Sifaka framework.
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
from .async_chain import AsyncChain
from .base import ChainComponent
from .chain import Chain
from .components import ChainFormatter, Improver, Model, Validator
from .managers import PromptManager, ResultFormatter, RetryStrategy, ValidationManager
from .models import ValidationResult
from .plugin import ChainPlugin
__all__: List[Any] = ['ChainComponent', 'ChainPlugin', 'Chain',
    'AsyncChain', 'PromptManager', 'ValidationManager', 'RetryStrategy',
    'ResultFormatter', 'Model', 'Validator', 'Improver', 'ChainFormatter',
    'ValidationResult']
