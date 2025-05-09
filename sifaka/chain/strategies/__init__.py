"""
Chain Strategies Package

Strategy implementations for Sifaka's chain system.

## Overview
This package provides specialized strategy implementations for different aspects
of chain behavior, particularly focusing on retry strategies for handling
validation failures and improving outputs.

## Components
1. **RetryStrategy**: Abstract base class for retry strategies
   - Defines interface for retry behavior
   - Implements common retry functionality
   - Provides extension points for specific strategies

2. **SimpleRetryStrategy**: Basic retry implementation
   - Fixed number of retry attempts
   - Feedback-based improvement
   - Basic error handling

3. **BackoffRetryStrategy**: Advanced retry implementation
   - Exponential backoff between attempts
   - Configurable backoff parameters
   - Rate limit handling

## Usage Examples
```python
from sifaka.chain.strategies import SimpleRetryStrategy, BackoffRetryStrategy

# Create simple retry strategy
simple = SimpleRetryStrategy(max_attempts=3)
result = simple.run(
    prompt="Write a story",
    generator=generator,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    result_formatter=result_formatter
)

# Create backoff retry strategy
backoff = BackoffRetryStrategy(
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0
)
result = backoff.run(
    prompt="Write a story",
    generator=generator,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    result_formatter=result_formatter
)
```

## Error Handling
- ValueError: Raised when validation fails after max attempts
- RuntimeError: Raised when operation fails after all retries
- Exception: Base class for all other errors that may occur

## Configuration
- max_attempts: Maximum number of retry attempts
- initial_backoff: Initial backoff time in seconds (BackoffRetryStrategy only)
- backoff_factor: Factor to multiply backoff by each attempt (BackoffRetryStrategy only)
- max_backoff: Maximum backoff time in seconds (BackoffRetryStrategy only)
"""

from .retry import RetryStrategy, SimpleRetryStrategy, BackoffRetryStrategy

__all__ = [
    "RetryStrategy",
    "SimpleRetryStrategy",
    "BackoffRetryStrategy",
]
