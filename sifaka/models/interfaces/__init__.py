"""
Protocol interfaces for model providers.

This package provides protocol interfaces for model providers, API clients,
and token counters, enabling better separation of concerns and extensibility.
"""

from .provider import ModelProviderProtocol, AsyncModelProviderProtocol
from .client import APIClientProtocol
from .counter import TokenCounterProtocol

__all__ = [
    "ModelProviderProtocol",
    "AsyncModelProviderProtocol",
    "APIClientProtocol",
    "TokenCounterProtocol",
]
