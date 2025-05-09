"""
Chain Module

Core orchestration system for Sifaka's validation and improvement flow.

## Overview
This module provides components for orchestrating the validation and improvement
flow between models, rules, and critics. It serves as the central coordinator
for Sifaka's validation and refinement capabilities.

## Components
1. **ChainCore**: Main interface that delegates to specialized components
2. **ChainOrchestrator**: Main user-facing class for standardized implementation
3. **PromptManager**: Manages prompt creation and management
4. **ValidationManager**: Manages validation logic and rule management
5. **RetryStrategy**: Handles retry logic with different strategies
6. **ResultFormatter**: Handles formatting and processing of results

## Usage Examples
```python
from sifaka.chain import ChainOrchestrator
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
rules = [create_length_rule(min_chars=10, max_chars=1000)]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

## Error Handling
- ChainError: Raised when chain execution fails
- ValidationError: Raised when validation fails
- CriticError: Raised when critic refinement fails
- ModelError: Raised when model generation fails

## Configuration
- max_attempts: Maximum number of retry attempts
- retry_strategy: Strategy for handling retries
- fail_fast: Whether to stop on first validation failure
- validation_timeout: Timeout for validation operations

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

# Utility functions
from .utils import create_chain_result, create_error_result, try_chain_operation

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
    # Utility functions
    "create_chain_result",
    "create_error_result",
    "try_chain_operation",
]
