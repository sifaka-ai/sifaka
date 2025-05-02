"""
Strategies for chains.

This package provides specialized strategies for different aspects of chains:
- RetryStrategy: Abstract base class for retry strategies
- SimpleRetryStrategy: Simple retry strategy that retries a fixed number of times
- BackoffRetryStrategy: Retry strategy that increases the wait time between retries
"""

from .retry import RetryStrategy, SimpleRetryStrategy, BackoffRetryStrategy

__all__ = [
    "RetryStrategy",
    "SimpleRetryStrategy",
    "BackoffRetryStrategy",
]
