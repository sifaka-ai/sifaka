"""
Chain Interface Module

Protocol interfaces for Sifaka's chain system.

## Overview
This package defines the interfaces for chain components in the Sifaka framework.
These interfaces establish a common contract for chain component behavior, enabling better
modularity and extensibility.

## Components
1. **Chain**: Base interface for all chains
2. **PromptManager**: Interface for prompt managers
3. **ValidationManager**: Interface for validation managers
4. **RetryStrategy**: Interface for retry strategies
5. **ResultFormatter**: Interface for result formatters
6. **Critic**: Interface for critics

## Usage Examples
```python
from sifaka.chain.interfaces import (
    Chain,
    AsyncChain,
    CriticProtocol,
    ResultFormatterProtocol,
    PromptManagerProtocol,
    ValidationManagerProtocol,
    RetryStrategyProtocol
)

# Create a chain implementation
class SimpleChain(Chain[str, str]):
    def execute(self, input_value: str) -> str:
        return f"Processed: {input_value}"

# Create a critic implementation
class SimpleCritic(CriticProtocol[str, str, dict]):
    def evaluate(self, input_value: str, output_value: str) -> bool:
        return len(output_value) > 0

    def get_feedback(self, input_value: str, output_value: str) -> dict:
        return {"feedback": "Good output"}

    def improve(self, input_value: str, output_value: str, feedback: dict) -> str:
        return output_value
```

## Error Handling
- ValueError: Raised for invalid inputs
- RuntimeError: Raised for execution failures
- TypeError: Raised for type mismatches

## Configuration
- input_type: Type of input accepted by the chain
- output_type: Type of output produced by the chain
- feedback_type: Type of feedback provided by critics
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
